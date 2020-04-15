#!/usr/bin/env bash

export MOT_DATASET_ROOT='/Users/linda/Downloads/2DMOT2015'
IMAGENET_DATASET_ROOT='/Users/linda/Downloads/MOT_IMAGENET'
CONF_THRESH=0.4
TRAIN_RATIO=0.8

python3 convert_mot_gt_to_imagenet.py --mot_root $MOT_DATASET_ROOT \
                                        --imagenet_root $IMAGENET_DATASET_ROOT \
                                        --conf_thresh $CONF_THRESH \
                                        --train_ratio $TRAIN_RATIO
