#!/bin/bash

# === Configuration ===
BACKBONE="light"
CONFIG="base"
DATASET="eps"
DEVICE="cuda"

# === Hyperparameters ===
LR=1e-4
DROPOUT=0.05
WEIGHT_DECAY=0

echo "=== Preparing dataset for finetuning ==="
python prepare_ft_dataset.py \
    --dataset $DATASET \
    --device $DEVICE \
    --config $CONFIG \
    --backbone $BACKBONE \
    --model_path ../checkpoints/pretrained/$BACKBONE/$CONFIG.pth \
    --data_path ../datasets \
    --use_prompt

echo "=== Starting finetuning ==="
python finetune.py \
    --config $CONFIG \
    --backbone $BACKBONE \
    --mode finetune \
    --model_path ../checkpoints/pretrained/$BACKBONE/$CONFIG.pth \
    --dataset $DATASET \
    --weight_decay $WEIGHT_DECAY \
    --dropout $DROPOUT \
    --lr $LR \
    --device $DEVICE
