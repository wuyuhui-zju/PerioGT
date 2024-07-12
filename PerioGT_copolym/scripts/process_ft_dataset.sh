#!/bin/bash

CONFIG="base"
DATASET="demo"

python process_ft_dataset.py \
    --dataset $DATASET \
    --device cuda \
    --config $CONFIG \
    --model_path ../checkpoints/pretrained/light/$CONFIG.pth \
    --data_path ../datasets
