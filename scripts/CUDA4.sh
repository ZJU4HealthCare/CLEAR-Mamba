#!/bin/bash
#set -euo pipefail


# EMA 模式：对目标值（kl_end）做 EMA，起点为 kl_start
# 这里对“初始值”做消融：扫描不同的 kl_start
STARTS=0.0
KL_ENDS=(1e-2 5e-3 1e-3 5e-4 1e-4)
BETA=0.9   # 平滑强度，可按需再做一轮 beta 扫描

EPOCHS=100
BS=128
LR=1e-4
SAVE_ROOT="./logs/AS_EDL"
mkdir -p "${SAVE_ROOT}"

for KL_END in "${KL_ENDS[@]}"; do
  NAME="edl_ema_start${STARTS}_to${KL_END}_b${BETA}"
  LOG="${SAVE_ROOT}/${NAME}.log"

  echo ">>> Running ${NAME}"
  CUDA_VISIBLE_DEVICES=4 python MD_train.py \
    --EDL 1 \
    --edl_mode ema \
    --kl_start ${STARTS} \
    --kl_end ${KL_END} \
    --kl_ema_beta ${BETA} \
    --epochs ${EPOCHS} \
    --batch_size ${BS} \
    --lr ${LR} \
    --save_name "${NAME}" | tee "${LOG}"
done
