#!/bin/bash

BACKBONE="light"
CONFIG="base"
DATASET="eat"

python prepare_ft_dataset.py \
    --dataset $DATASET \
    --device cuda \
    --config $CONFIG \
    --backbone $BACKBONE \
    --model_path ../checkpoints/pretrained/$BACKBONE/$CONFIG.pth \
    --data_path ../datasets
