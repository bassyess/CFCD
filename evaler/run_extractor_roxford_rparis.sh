#! /bin/bash
EXPERIMENT=./outputs/R50_CFCD_D512
CONFIG=${EXPERIMENT}/config.yaml
DIR=./datasets/images
CKPT=${EXPERIMENT}/checkpoints/model_s2_0050.pyth

TEST=roxford5k #roxford5k,rparis6k,revisitop1m # revisitop1m 推荐使用多进程脚本来提取特征。
echo CUDA_VISIBLE_DEVICES=0  python3 evaler/infer.py --cfg $CONFIG INFER.CKPT ${CKPT} INFER.DIR ${DIR} INFER.TEST ${TEST}

TEST=rparis6k #roxford5k,rparis6k,revisitop1m # revisitop1m 推荐使用多进程脚本来提取特征。
echo CUDA_VISIBLE_DEVICES=1  python3 evaler/infer.py --cfg $CONFIG INFER.CKPT ${CKPT} INFER.DIR ${DIR} INFER.TEST ${TEST}