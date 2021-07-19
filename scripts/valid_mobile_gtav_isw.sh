#!/usr/bin/env bash
echo "Running inference on" ${1}

     python -m torch.distributed.launch --nproc_per_node=2 valid.py \
        --val_dataset bdd100k cityscapes synthia mapillary \
        --arch network.deepv3.DeepMobileNetV3PlusD \
        --wt_layer 0 0 2 2 2 0 0 \
        --date 0101 \
        --exp mobile_gtav_isw \
        --snapshot ${1}
