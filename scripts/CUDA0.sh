CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 1 --had-feat-dim 64 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper1-64 2>&1 | tee ./logs/AS/ASHyper1-64.log

CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 2 --had-feat-dim 64 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper2-64 2>&1 | tee ./logs/AS/ASHyper2-64.log

CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 4 --had-feat-dim 64 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper4-64 2>&1 | tee ./logs/AS/ASHyper4-64.log

CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 8 --had-feat-dim 64 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper8-64 2>&1 | tee ./logs/AS/ASHyper8-64.log
