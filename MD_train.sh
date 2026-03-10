CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch-size 128 --epochs 100 \
--save-name mambatest0 2>&1 | tee ./logs/mamba/mambatest0.log
#--hyper-ad 0 --reduction-ratio 1 --had-feat-dim 96 \
#--EDL 0 --kl-coef 5e-3 \


CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch-size 128 --epochs 150 \
--hyper-ad 1 --reduction-ratio 8 --had-feat-dim 128 \
--EDL 1 --kl-coef 5e-3 \
--save-name main1 2>&1 | tee ./logs/main/main1.log

CUDA_VISIBLE_DEVICES=1 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 8 --had-feat-dim 96 \
--EDL 1 --kl-coef 5e-3 \
--save-name main2 2>&1 | tee ./logs/main/main2.log

CUDA_VISIBLE_DEVICES=2 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 4 --had-feat-dim 96 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper4-96 2>&1 | tee ./logs/AS/ASHyper4-96.log

CUDA_VISIBLE_DEVICES=3 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 8 --had-feat-dim 96 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper8-96 2>&1 | tee ./logs/AS/ASHyper8-96.log

