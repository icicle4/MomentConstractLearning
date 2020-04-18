#!/usr/bin/env bash

export VAL_MOT15=true
DATA_DIR="../"
DIS_THRESHOLD=1.0
CHECKPOINTPATH='/content/drive/My Drive/checkpoint_0105.pth.tar'

python track.py --val_mot15 $VAL_MOT15  \
                --dis_threshold $DIS_THRESHOLD \
                --load_model $CHECKPOINTPATH


