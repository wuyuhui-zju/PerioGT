#!/bin/bash

BACKBONE="light"
CONFIG="base"

CUDA_VISIBLE_DEVICES=0,1,2 python -u -m torch.distributed.run \
    --nproc_per_node=3 \
    --nnodes=1 \
    pretrain.py \
    --save_path ../models/checkpoints/pretrained/$BACKBONE \
    --n_threads 8 \
    --n_devices 3 \
    --config $CONFIG \
    --backbone $BACKBONE \
    --n_steps 100000 \
    --data_path ../datasets/pretrain/

