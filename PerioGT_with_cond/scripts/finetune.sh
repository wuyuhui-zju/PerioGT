#!/bin/bash

CONFIG="base"

python finetune.py \
    --config $CONFIG \
    --mode finetune \
    --model_path ../checkpoints/pretrained/light/$CONFIG.pth \
    --dataset opv \
    --weight_decay 0 \
    --dropout 0 \
    --lr 2e-4 \
    --device cuda:0
