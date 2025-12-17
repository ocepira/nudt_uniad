#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# 仅保留必要变量（单卡无需GPUS_PER_NODE）            #
CFG=$1  # 配置文件路径                               #
CKPT=$2 # 权重文件路径                               #
# GPUS变量保留但不再使用（兼容你原有的命令格式）      #
GPUS=$3                                              #
# -------------------------------------------------- #

MASTER_PORT=${MASTER_PORT:-28596}  # 保留变量但无实际作用（单卡不需要）
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# 确保日志目录存在
if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

# 设置Python路径，单卡非分布式运行test.py
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py \
    $CFG \
    $CKPT \
    ${@:4} \  # 传递后续可选参数（如果有）
    --eval bbox track \  # 评估检测+跟踪（Stage1任务）
    --gpu-ids 0 \        # 单卡固定指定gpu-ids=0（对应CUDA_VISIBLE_DEVICES的卡）
    --show-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/eval.$T