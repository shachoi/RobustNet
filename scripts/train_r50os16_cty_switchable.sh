#!/usr/bin/env bash
    # Example on Cityscapes
     python -m torch.distributed.launch --nproc_per_node=2 train.py \
        --dataset cityscapes \
        --covstat_val_dataset cityscapes \
        --val_dataset bdd100k gtav synthia mapillary \
        --arch network.deepv3.DeepR50V3PlusD \
        --city_mode 'train' \
        --lr_schedule poly \
        --lr 0.01 \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --max_iter 40000 \
        --bs_mult 4 \
        --gblur \
        --color_aug 0.5 \
        --wt_layer 0 0 5 5 5 0 0 \
        --date 0101 \
        --exp r50os16_city_switchable \
        --ckpt ./logs/ \
        --tb_path ./logs/
