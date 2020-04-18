#!/usr/bin/env bash

export Test_MOT15=true
DATA_DIR="/content"
DIS_THRESHOLD=1.0
CHECKPOINTPATH='/content/drive/My Drive/checkpoint_0105.pth.tar'

python track.py --test_mot15 $Test_MOT15  \
                --dis_threshold $DIS_THRESHOLD \
                --load_model $CHECKPOINTPATH \
                --data_dir $DATA_DIR


