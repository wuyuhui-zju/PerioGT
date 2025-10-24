#!/bin/bash

CONFIG="base"
DATASET="ea"

python prepare_ft_dataset.py \
    --dataset $DATASET \
    --device cuda \
    --config $CONFIG \
    --model_path ../checkpoints/pretrained/light/$CONFIG.pth \
    --max_prompt 5 \
    --data_path ../datasets
