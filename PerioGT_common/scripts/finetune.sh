#!/bin/bash

BACKBONE="light"
CONFIG="base"
DATASET="egc"

python finetune.py \
    --config $CONFIG \
    --backbone $BACKBONE \
    --mode finetune \
    --model_path ../checkpoints/pretrained/$BACKBONE/$CONFIG.pth \
    --max_prompt 20 \
    --dataset $DATASET \
    --weight_decay 0 \
    --dropout 0 \
    --lr 5e-4 \
    --device cuda:0
