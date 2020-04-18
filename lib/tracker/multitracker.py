from collections import deque

import torchvision.models as models
from munkres import Munkres
from scipy.spatial import distance

from lib.tracking_utils.utils import *
from lib.moco import builder
from lib.moco.loader import warp_clip


def py_max_match(scores):
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp


class Tracklet:
    def __init__(self,tracklet_id, history_len=10):
        self.tracklet_id = tracklet_id
        self.history_len = history_len
        self.history_keys = deque(history_len)
        self.frame_bboxes = {}
        self.frame_list = []

    @property
    def weighted_sum_keys(self):
        weights = np.linspace(0.1, 0.999, self.history_len) / np.sum(np.linspace(0.1, 0.999, self.history_len))
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
        opt.device = torch.device('cuda')
        print('Creating model...')
        self.model = builder.MoCo(
            models.__dict__[opt.arch],
            opt.moco_dim, opt.moco_k, opt.moco_m, opt.moco_t, opt.mlp
        )
        self.model.cuda()
        self.model = torch.nn.parallel.DistributedDataParallel(self.model)
    
        checkpoint = torch.load(opt.checkpoint)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.tracklet_id = 1
        self.tracklet_pools = [] # type: list[Tracklet]

    def encode_q(self, frame_clip):
        clip = warp_clip(frame_clip)
        q = self.encode_q(clip).detach().cpu().numpy()
        return q

    def encode_k(self, frame_clip):
        clip = warp_clip(frame_clip)
        k = self.encode_k(clip).detach().cpu().numpy()
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

            dis_matrix = distance.cdist(new_added_keys, grouped_keys)
            dis_saved = np.copy(dis_matrix)

            if num_added > num_grouped:
                dis_matrix = np.concatenate(
                    (
                        dis_matrix,
                        np.zeros((num_added, num_added - num_grouped)) + 1e10
                    )
                )
            pairs = py_max_match(dis_matrix)

            for row, col in pairs:
                if (
                    row < num_added and col < num_grouped and dis_saved[row][col] < self.opt.dis_threshold
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
