#! /bin/bash

# stage 1
python3 train.py --cfg configs/resnet50_cfcd_s1_8gpu.yaml

# stage 2
# python3 train.py --cfg configs/resnet50_cfcd_s2_8gpu.yaml