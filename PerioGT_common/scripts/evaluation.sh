#!/bin/bash

BACKBONE="light"
CONFIG="base"
DATASET="egb"

python evaluation.py \
    --config $CONFIG \
    --backbone $BACKBONE \
    --model_path ../checkpoints/$DATASET/best_model_0.pth \
    --max_prompt 20 \
    --dataset $DATASET \
    --dropout 0 \
    --device cuda:0
