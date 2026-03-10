#!/bin/bash
#set -euo pipefail


# 固定 kl_coef，直接扫一组取值
INIT_KLS=(1e-2 5e-3 1e-3 5e-4 1e-4)

EPOCHS=100
BS=128
LR=1e-4
SAVE_ROOT="./logs/AS_EDL"
mkdir -p "${SAVE_ROOT}"

for KL in "${INIT_KLS[@]}"; do
  NAME="edl_fixed_kl${KL}"
  LOG="${SAVE_ROOT}/${NAME}.log"

  echo ">>> Running ${NAME}"
  CUDA_VISIBLE_DEVICES=2 python MD_train.py \
    --EDL 1 \
    --edl_mode fixed \
    --kl_coef ${KL} \
    --epochs ${EPOCHS} \
    --batch_size ${BS} \
    --lr ${LR} \
    --save_name "${NAME}" | tee "${LOG}"
done


#CUDA_VISIBLE_DEVICES=2 python MD_train.py --batch_size 128 --epochs 100 \
#--hyper_ad 1 --reduction_ratio 1 --had_feat_dim 128 \
#--EDL 0 --kl_coef 5e-3 \
#--save_name ASHyper1_128 2>&1 | tee ./logs/AS/ASHyper1_128.log
#
#CUDA_VISIBLE_DEVICES=2 python MD_train.py --batch_size 128 --epochs 100 \
#--hyper_ad 1 --reduction_ratio 2 --had_feat_dim 128 \
#--EDL 0 --kl_coef 5e-3 \
#--save_name ASHyper2_128 2>&1 | tee ./logs/AS/ASHyper2_128.log
#
#CUDA_VISIBLE_DEVICES=2 python MD_train.py --batch_size 128 --epochs 100 \
#--hyper_ad 1 --reduction_ratio 4 --had_feat_dim 128 \
#--EDL 0 --kl_coef 5e-3 \
#--save_name ASHyper4_128 2>&1 | tee ./logs/AS/ASHyper4_128.log
#
#CUDA_VISIBLE_DEVICES=2 python MD_train.py --batch_size 128 --epochs 100 \
#--hyper_ad 1 --reduction_ratio 8 --had_feat_dim 128 \
#--EDL 0 --kl_coef 5e-3 \
#--save_name ASHyper8_128 2>&1 | tee ./logs/AS/ASHyper8_128.log
