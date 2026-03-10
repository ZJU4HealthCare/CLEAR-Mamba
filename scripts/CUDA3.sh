#!/bin/bash
#set -euo pipefail

# 线性 warm-up: kl 从 kl_start -> kl_end（在 warmup_epochs 内）
# 这里对“初始值”做消融：扫描不同的 kl_start
STARTS=0.0
KL_ENDS=(1e-2 5e-3 1e-3 5e-4 1e-4)
WARMUP_EPOCHS=20

EPOCHS=100
BS=128
LR=1e-4
SAVE_ROOT="./logs/AS_EDL"
mkdir -p "${SAVE_ROOT}"

for KL_END in "${KL_ENDS[@]}"; do
  NAME="edl_linear_start${STARTS}_to${KL_END}_e${WARMUP_EPOCHS}"
  LOG="${SAVE_ROOT}/${NAME}.log"

  echo ">>> Running ${NAME}"
  CUDA_VISIBLE_DEVICES=3 python MD_train.py \
    --EDL 1 \
    --edl_mode linear \
    --kl_start ${STARTS} \
    --kl_end ${KL_END} \
    --kl_warmup_epochs ${WARMUP_EPOCHS} \
    --epochs ${EPOCHS} \
    --batch_size ${BS} \
    --lr ${LR} \
    --save_name "${NAME}" | tee "${LOG}"
done



#CUDA_VISIBLE_DEVICES=3 python MD_train.py --batch-size 100 --epochs 100 \
#--hyper-ad 1 --reduction-ratio 1 --had-feat-dim 256 \
#--EDL 0 --kl-coef 5e-3 \
#--save-name ASHyper1-256 2>&1 | tee ./logs/AS/ASHyper1-256.log
#
#CUDA_VISIBLE_DEVICES=3 python MD_train.py --batch-size 100 --epochs 100 \
#--hyper-ad 1 --reduction-ratio 2 --had-feat-dim 256 \
#--EDL 0 --kl-coef 5e-3 \
#--save-name ASHyper2-256 2>&1 | tee ./logs/AS/ASHyper2-256.log
#
#CUDA_VISIBLE_DEVICES=3 python MD_train.py --batch-size 100 --epochs 100 \
#--hyper-ad 1 --reduction-ratio 4 --had-feat-dim 256 \
#--EDL 0 --kl-coef 5e-3 \
#--save-name ASHyper4-256 2>&1 | tee ./logs/AS/ASHyper4-256.log
#
#CUDA_VISIBLE_DEVICES=3 python MD_train.py --batch-size 100 --epochs 100 \
#--hyper-ad 1 --reduction-ratio 8 --had-feat-dim 256 \
#--EDL 0 --kl-coef 5e-3 \
#--save-name ASHyper8-256 2>&1 | tee ./logs/AS/ASHyper8-256.log