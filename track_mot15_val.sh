#!/usr/bin/env bash

export VAL_MOT15=true
DATA_DIR="/content"
DIS_THRESHOLD=1.0
CHECKPOINTPATH="/content/drive/My \Drive/checkpoint_0105.pth.tar"

python track.py --val_mot15 $VAL_MOT15  \
                --dis_threshold $DIS_THRESHOLD \
                --checkpoint $CHECKPOINTPATH \
                --data_dir $DATA_DIR


