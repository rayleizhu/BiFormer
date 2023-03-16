#!/usr/bin/env bash


PARTITION=mediasuper
NOW=$(date '+%m-%d-%H:%M:%S')
OUTPUT_DIR=../outputs/seg

CONFIG_DIR=configs/ade20k

# CKPT=/mnt/lustre/zhulei1/.cache/torch/hub/checkpoints/biformer_small_best.pth
# MODEL=sfpn.biformer_small
CKPT=/mnt/lustre/zhulei1/GitRepo_2023/BiFormerDev/outputs/cls/batch_size.128-drop_path.0.1-input_size.224-lr.5e-4-model.maxvit_stl-slurm.ngpus.8-slurm.nodes.1/20230312-12:52:51/best.pth
MODEL=sfpn.maxvit_stl

JOB_NAME=${MODEL}
CONFIG=${CONFIG_DIR}/${MODEL}.py
WORK_DIR=${OUTPUT_DIR}/${MODEL}/${NOW}
mkdir -p ${WORK_DIR}

# add project root (semantic_segmentation/..) to PYTHONPATH
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
        --job-name=${JOB_NAME} \
        --gres=gpu:8 \
        --ntasks=8 \
        --ntasks-per-node=8 \
        --cpus-per-task=4 \
        --kill-on-bad-exit=1 \
        python -u train.py ${CONFIG} \
            --launcher="slurm" \
            --work-dir=${WORK_DIR} \
            --options model.pretrained=${CKPT} \
        &> ${WORK_DIR}/srun.${JOB_NAME}.log &
