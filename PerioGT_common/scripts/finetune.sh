#!/bin/bash

BACKBONE="light"
CONFIG="base"

python finetune.py \
    --config $CONFIG \
    --backbone $BACKBONE \
    --mode finetune \
    --model_path ../checkpoints/pretrained/$BACKBONE/$CONFIG.pth \
    --max_prompt 20 \
    --dataset egb \
    --weight_decay 0 \
    --dropout 0 \
    --lr 2e-4 \
    --device cuda:0 \
    --save \
    --save_suffix 0 \
