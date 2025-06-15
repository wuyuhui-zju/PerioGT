#!/bin/bash

CONFIG="base"

python finetune.py \
    --config $CONFIG \
    --mode finetune \
    --model_path ../checkpoints/pretrained/light/$CONFIG.pth \
    --dataset ea \
    --weight_decay 0 \
    --dropout 0 \
    --lr 1e-4 \
    --save \
    --save_suffix 0 \
    --device cuda:0

