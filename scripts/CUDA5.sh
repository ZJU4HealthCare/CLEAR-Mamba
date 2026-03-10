#!/bin/bash

# 内部自适应会从 kl_coef 作为初值出发，这里对“初始值”做消融
INIT_KLS=(1e-2 5e-3 1e-3 5e-4 1e-4)

C=0.8
ADAPTIVE_EMA=0.9

EPOCHS=100
BS=128
LR=1e-4
SAVE_ROOT="./logs/AS_EDL"
mkdir -p "${SAVE_ROOT}"

for KL0 in "${INIT_KLS[@]}"; do
  NAME="edl_adaptive_init${KL0}_c${C}_min${KL_MIN}_max${KL_MAX}"
  LOG="${SAVE_ROOT}/${NAME}.log"

  echo ">>> Running ${NAME}"
  CUDA_VISIBLE_DEVICES=5 python MD_train.py \
    --EDL 1 \
    --edl_mode adaptive \
    --kl_coef ${KL0} \
    --kl_scale ${C} \
    --kl_ema_beta ${ADAPTIVE_EMA} \
    --epochs ${EPOCHS} \
    --batch_size ${BS} \
    --lr ${LR} \
    --save_name "${NAME}" | tee "${LOG}"
done
