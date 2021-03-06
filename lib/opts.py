from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import torchvision.models as models
import torch
import torch.multiprocessing as mp

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('task', default='mot', help='mot')
        self.parser.add_argument('--exp_id', default='default')

        self.parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                                 choices=model_names,
                                 help='model architecture: ' +
                                      ' | '.join(model_names) +
                                      ' (default: resnet50)')

        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        self.parser.add_argument('--world-size', default=-1, type=int,
                            help='number of nodes for distributed training')
        self.parser.add_argument('--rank', default=-1, type=int,
                            help='node rank for distributed training')
        self.parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                            help='url used to set up distributed training')
        self.parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        self.parser.add_argument('--seed', default=None, type=int,
                            help='seed for initializing training. ')
        self.parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
        self.parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                 'N processes per node, which has N GPUs. This is the '
                                 'fastest way to use PyTorch for either single node or '
                                 'multi node data parallel training')

        self.parser.add_argument('--min_sim_thresh', default=0.4, type=float,
                                 help="feature match threshold distance")

        # system
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')

        # moco specific configs:
        self.parser.add_argument('--moco-dim', default=128, type=int,
                                 help='feature dimension (default: 128)')
        self.parser.add_argument('--moco-k', default=65536, type=int,
                                 help='queue size; number of negative keys (default: 65536)')
        self.parser.add_argument('--moco-m', default=0.999, type=float,
                                 help='moco momentum of updating key encoder (default: 0.999)')
        self.parser.add_argument('--moco-t', default=0.07, type=float,
                                 help='softmax temperature (default: 0.07)')

        # options for moco v2
        self.parser.add_argument('--mlp', action='store_true',
                                 help='use mlp head')
        self.parser.add_argument('--aug-plus', action='store_true',
                                 help='use moco v2 data augmentation')
        self.parser.add_argument('--cos', action='store_true',
                                 help='use cosine lr schedule')
        # mdoel checkpoint
        self.parser.add_argument('--checkpoint', type=str, help='checkpoint of moco model')

        # test
        self.parser.add_argument('--K', type=int, default=128,
                                 help='max number of output objects.')
        self.parser.add_argument('--not_prefetch_test', action='store_true',
                                 help='not use parallal data pre-processing.')
        self.parser.add_argument('--fix_res', action='store_true',
                                 help='fix testing resolution or keep '
                                      'the original resolution')
        self.parser.add_argument('--keep_res', action='store_true',
                                 help='keep the original resolution'
                                      ' during validation.')
        # tracking
        self.parser.add_argument('--test_mot16', default=False, help='test mot16')
        self.parser.add_argument('--val_mot15', default=False, help='val mot15')
        self.parser.add_argument('--test_mot15', default=False, help='test mot15')
        self.parser.add_argument('--val_mot16', default=False, help='val mot16 or mot15')
        self.parser.add_argument('--test_mot17', default=False, help='test mot17')
        self.parser.add_argument('--val_mot17', default=False, help='val mot17')
        self.parser.add_argument('--val_mot20', default=False, help='val mot20')
        self.parser.add_argument('--test_mot20', default=False, help='test mot20')

        self.parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')

        self.parser.add_argument('--input-video', type=str, default='../videos/MOT16-03.mp4',
                                 help='path to the input video')
        self.parser.add_argument('--output-format', type=str, default='video', help='video or text')
        self.parser.add_argument('--output-root', type=str, default='../results', help='expected output root path')

        # mot
        self.parser.add_argument('--data_cfg', type=str,
                                 default='../src/lib/cfg/data.json',
                                 help='load data from cfg')
        self.parser.add_argument('--data_dir', type=str, default='../')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth')
        return opt

    def init(self, args=''):
        opt = self.parse(args)

        if opt.dist_url == "env://" and opt.world_size == -1:
            opt.world_size = int(os.environ["WORLD_SIZE"])

        opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed

        ngpus_per_node = torch.cuda.device_count()
        if opt.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            opt.world_size = ngpus_per_node * opt.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function

        return opt
