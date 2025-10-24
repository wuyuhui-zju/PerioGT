#!/bin/bash

# === Configuration ===
BACKBONE="light"
CONFIG="base"
DATASET="eps"
DEVICE="cuda"

echo "=== Preparing dataset for extracting features ==="
python prepare_ft_dataset.py \
    --dataset $DATASET \
    --backbone $BACKBONE \
    --data_path ../datasets

echo "=== Starting extracting features ==="
python extract_features.py \
    --config $CONFIG \
    --backbone $BACKBONE \
    --model_path ../checkpoints/pretrained/$BACKBONE/$CONFIG.pth \
    --data_path ../datasets \
    --dataset $DATASET \
    --device $DEVICE
