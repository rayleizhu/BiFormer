#!/usr/bin/env bash

set -x # display command before command's output

PARTITION=mediasuper
# PARTITION=mediaa
# quota=spot
NOW=$(date '+%m-%d-%H:%M:%S')
OUTPUT_DIR=../outputs/det

CONFIG_DIR=configs/coco

CKPT=/mnt/lustre/zhulei1/.cache/torch/hub/checkpoints/biformer_small_best.pth
# MODEL=maskrcnn.1x.biformer_small
MODEL=retinanet.1x.biformer_small
NUM_GPUS=8

JOB_NAME=${MODEL}
CONFIG=${CONFIG_DIR}/${MODEL}.py
WORK_DIR=${OUTPUT_DIR}/${MODEL}/${NOW}
mkdir -p ${WORK_DIR}

# add project root (object_detection/..) to PYTHONPATH, hence ops/ can be imported
PYTHONPATH="$(dirname $0)":"$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
        --job-name=${JOB_NAME} \
        --gres=gpu:8 \
        --ntasks=${NUM_GPUS} \
        --ntasks-per-node=8 \
        --cpus-per-task=4 \
        --kill-on-bad-exit=1 \
        python -u train.py ${CONFIG} \
            --launcher="slurm" \
            --work-dir=${WORK_DIR} \
            --cfg-options model.pretrained=${CKPT} \
        &> ${WORK_DIR}/srun.${JOB_NAME}.log &
