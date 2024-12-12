#!/bin/bash

CONFIG="base"
DATASET="opv"

python prepare_ft_dataset.py \
    --dataset $DATASET \
    --device cuda \
    --config $CONFIG \
    --model_path ../checkpoints/pretrained/light/$CONFIG.pth \
    --data_path ../datasets
