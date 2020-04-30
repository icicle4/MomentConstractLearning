from collections import deque

import torchvision.models as models
from munkres import Munkres
from scipy.spatial import distance

from lib.tracking_utils.utils import *
from lib.moco import builder
from lib.moco.loader import warp_clip

import torch.nn as nn
import torch.distributed as dist
import torch


def py_max_match(scores):
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp


class Tracklet:
    def __init__(self,tracklet_id, history_len=20):
        self.tracklet_id = tracklet_id
        self.history_len = history_len
        self.history_keys = deque(maxlen=history_len)
        self.frame_bboxes = {}
        self.frame_list = []

    @property
    def weighted_sum_keys(self):
        weights = np.linspace(0.1, 0.999, len(self.history_keys)) / np.sum(np.linspace(0.1, 0.999, len(self.history_keys)))
        keys = np.array(self.history_keys)
        weighted_sum_keys = np.einsum("i, ij->j", weights, keys)
        return weighted_sum_keys

    def update(self, key, bbox, frame):
        self.history_keys.append(key)
        self.frame_bboxes[frame] = bbox
        self.frame_list.append(frame)

    @property
    def tlwh(self):
        return self.frame_bboxes[self.frame_list[-1]]

    @property
    def track_id(self):
        return self.tracklet_id


class MCTracker(object):
    def __init__(self, opt):
        self.opt = opt
        self.frame_id = 0

        if self.opt.gpu is not None:
            print("Use GPU: {} for training".format(self.opt.gpu))

        if self.opt.distributed:
            if self.opt.dist_url == "env://" and self.opt.rank == -1:
                self.opt.rank = int(os.environ["RANK"])
            if self.opt.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                pass

            dist.init_process_group(backend=self.opt.dist_backend, init_method=self.opt.dist_url,
                                    world_size=self.opt.world_size, rank=self.opt.rank)
        
        print('Creating model...')
        self.model = builder.MoCo(
            models.__dict__[opt.arch],
            opt.moco_dim, opt.moco_k, opt.moco_m, opt.moco_t, opt.mlp
        )
        
        if self.opt.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.opt.gpu is not None:
                torch.cuda.set_device(self.opt.gpu)
                self.model.cuda(self.opt.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.opt.gpu])
            else:
                self.model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        elif self.opt.gpu is not None:
            torch.cuda.set_device(self.opt.gpu)
            self.model = self.model.cuda(self.opt.gpu)
            # comment out the following line for debugging
            raise NotImplementedError("Only DistributedDataParallel is supported.")
        else:
            # AllGather implementation (batch shuffle, queue update, etc.) in
            # this code only supports DistributedDataParallel.
            raise NotImplementedError("Only DistributedDataParallel is supported.")

        if self.opt.checkpoint:
            if os.path.isfile(self.opt.checkpoint):
                print("=> loading checkpoint '{}'".format(self.opt.checkpoint))
                if self.opt.gpu is None:
                    checkpoint = torch.load(self.opt.checkpoint)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(self.opt.gpu)
                    checkpoint = torch.load(self.opt.checkpoint, map_location=loc)
                self.model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.opt.checkpoint, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.opt.checkpoint))
        self.model.eval()
        self.tracklet_id = 1
        self.tracklet_pools = [] # type: list[Tracklet]

    def encode_q(self, frame_clip):
        clip = warp_clip(frame_clip)
        q = self.model.module.encoder_q(clip.cuda())
        norm_q = nn.functional.normalize(q, dim=1)
        q = norm_q.detach().cpu().numpy()[0]
        return q

    def encode_k(self, frame_clip):
        clip = warp_clip(frame_clip)
        k =  self.model.module.encoder_k(clip.cuda())
        norm_k = nn.functional.normalize(k, dim=1)
        k = norm_k.detach().cpu().numpy()[0]
        return k

    def update(self, frame_clips_and_bboxes):
        self.frame_id += 1

        tracked_targets = list()
        if not self.tracklet_pools:
            for frame_clip_and_bbox in frame_clips_and_bboxes:
                frame_clip, bbox = frame_clip_and_bbox
                new_tracklet = Tracklet(self.tracklet_id)
                self.tracklet_id += 1
                feature = self.encode_k(frame_clip)
                new_tracklet.update(feature, bbox, self.frame_id)
                self.tracklet_pools.append(
                    new_tracklet
                )
                tracked_targets.append(new_tracklet)

        else:
            num_grouped = len(self.tracklet_pools)
            num_added = len(frame_clips_and_bboxes)

            new_added_keys = np.asarray([self.encode_q(frame_clip) for frame_clip, bbox in frame_clips_and_bboxes],
                                        dtype=np.float32)
            grouped_keys = np.asarray([tracklet.weighted_sum_keys for tracklet in self.tracklet_pools],
                                      dtype=np.float32)

            if len(new_added_keys) == 0:
                return tracked_targets

            print('new_added', new_added_keys.shape)
            print('grouped_keys', grouped_keys.shape)

            sim_matrix = np.zeros((len(new_added_keys), len(grouped_keys)))

            for i in range(len(new_added_keys)):
                for j in range(len(grouped_keys)):
                    new_key = new_added_keys[i]
                    old_key = grouped_keys[j]

                    nk = torch.from_numpy(new_key).unsqueeze(dim=0)
                    ok = torch.from_numpy(new_key).unsqueeze(dim=0)
                    sim = torch.einsum('nc,nc->n', [nk, ok]).unsqueeze(-1)
                    print('sim', sim, sim.size())
                    print('old_sim', np.einsum('k, k', new_key, old_key))

                    # 相似度越到越靠近1，相似度矩阵的值应该在【0，2】之间，越相似越接近1
                    similarity = 1 - np.einsum('k, k', new_key, old_key)
                    sim_matrix[i, j] = similarity

            sim_saved = np.copy(sim_matrix)

            if num_added > num_grouped:
                sim_matrix = np.concatenate(
                    (
                        sim_matrix,
                        np.zeros((num_added, num_added - num_grouped)) + 2.0
                    ), axis = 1
                )

            print('sim_matrix', sim_matrix)
            pairs = py_max_match(sim_matrix)

            for row, col in pairs:
                if (
                    row < num_added and col < num_grouped and sim_saved[row][col] < self.opt.dis_threshold
                ):
                    matched_tracklet = self.tracklet_pools[col]
                    frame_clip, bbox = frame_clips_and_bboxes[row]
                    key = self.encode_k(frame_clip)
                    matched_tracklet.update(key, bbox, self.frame_id)
                    tracked_targets.append(matched_tracklet)
                else:
                    frame_clip, bbox = frame_clips_and_bboxes[row]
                    new_tracklet = Tracklet(self.tracklet_id)
                    self.tracklet_id += 1
                    feature = self.encode_k(frame_clip)
                    new_tracklet.update(feature, bbox, self.frame_id)
                    self.tracklet_pools.append(
                        new_tracklet
                    )
                    tracked_targets.append(new_tracklet)
        return tracked_targets
