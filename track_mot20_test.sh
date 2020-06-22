#!/usr/bin/env bash

export Test_MOT20=true
DATA_DIR="/content"
MIN_SIM_THRESH=0.4
CHECKPOINTPATH="/content/drive/My \Drive/checkpoint_0105.pth.tar"

python track.py --val_mot20 $Test_MOT20  \
                --dis_threshold $MIN_SIM_THRESH \
                --checkpoint $CHECKPOINTPATH \
                --data_dir $DATA_DIR



