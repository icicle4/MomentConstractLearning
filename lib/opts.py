from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # basic experiment setting
    self.parser.add_argument('task', default='mot', help='mot')
    self.parser.add_argument('--exp_id', default='default')
    self.parser.add_argument('--test', action='store_true')
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')

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

    self.parser.add_argument('--dis_threshold', default=0.8, type=float,
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

    self.parser.add_argument('--input-video', type=str, default='../videos/MOT16-03.mp4', help='path to the input video')
    self.parser.add_argument('--output-format', type=str, default='video', help='video or text')
    self.parser.add_argument('--output-root', type=str, default='../results', help='expected output root path')

    # mot
    self.parser.add_argument('--data_cfg', type=str,
                             default='../src/lib/cfg/data.json',
                             help='load data from cfg')
    self.parser.add_argument('--data_dir', type=str, default='/data/yfzhang/MOT/JDE')

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]


    opt.pad = 31
    opt.num_stacks = 1

    if opt.trainval:
      opt.val_intervals = 100000000

    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)
    
    if opt.resume and opt.load_model == '':
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                  else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')
    return opt

  def update_dataset_info_and_set_heads(self, opt, dataset):
    if opt.task == 'mot':
      opt.heads = {'hm': opt.num_classes,
                   'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes,
                   'id': opt.reid_dim}
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
      opt.nID = dataset.nID
      opt.img_size = (1088, 608)
    else:
      assert 0, 'task not defined!'
    print('heads', opt.heads)
    return opt

  def init(self, args=''):
    default_dataset_info = {
      'mot': {'default_resolution': [608, 1088], 'num_classes': 1,
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'jde', 'nID': 14455},
    }
    class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)
    opt = self.parse(args)
    dataset = Struct(default_dataset_info[opt.task])
    opt.dataset = dataset.dataset
    opt = self.update_dataset_info_and_set_heads(opt, dataset)
    return opt