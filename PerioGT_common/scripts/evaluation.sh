#!/bin/bash

BACKBONE="light"
CONFIG="base"
DATASET="egc"

python evaluation.py \
    --config $CONFIG \
    --backbone $BACKBONE \
    --model_path ../checkpoints/$DATASET/best_model.pth \
    --max_prompt 20 \
    --dataset $DATASET \
    --dropout 0 \
    --device cuda:0
