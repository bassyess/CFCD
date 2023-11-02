#! /bin/bash
EXPERIMENT=./outputs/R50_CFCD_D512
CONFIG=${EXPERIMENT}/config.yaml
CKPT=${EXPERIMENT}/checkpoints/model_s2_0050.pyth
DIR=./datasets/images
TEST=revisitop1m # revisitop1m对顺序无要求，可以使用多进程抽取。

total_num=8
total_gpu=8
for cutno in $(seq 1 $total_num);
do
    {
    let gpu_id=cutno%total_gpu
    export CUDA_VISIBLE_DEVICES=$gpu_id
    cmd="python3 evaler/infer.py --cfg $CONFIG INFER.CKPT ${CKPT} INFER.DIR ${DIR} INFER.TEST ${TEST} INFER.TOTAL_NUM ${total_num} INFER.CUT_NUM ${cutno}"
    echo [start cmd:] ${cmd}
    echo ${cmd} | sh 
    } &
    sleep 0.5
done
wait
echo "extracting 1M fea finished~"
python3 evaler/gather_mat.py ${EXPERIMENT}
echo "gather 1M fea finished~"
