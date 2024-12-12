#!/bin/bash

CONFIG="base"
DATASET="mar1"

python evaluation.py \
    --config $CONFIG \
    --model_path ../checkpoints/$DATASET/best_model.pth \
    --dataset $DATASET \
    --device cuda:0