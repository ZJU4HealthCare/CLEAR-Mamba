#CUDA_VISIBLE_DEVICES=2 python train.py --hyper-ad 0 --save-name Classifier 2>&1 | tee ./logs/train.log
#CUDA_VISIBLE_DEVICES=2 python train.py --hyper-ad 1 --save-name Classifier_hyper 2>&1 | tee ./logs/train_hyper.log
CUDA_VISIBLE_DEVICES=4 python train.py --batch-size 128 --epochs 100 \
--hyper-ad 0 --reduction-ratio 4 \
--EDL 1 --kl-coef 5e-3 \
--save-name test1 2>&1 | tee ./logs/test1.log
#
CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 128 --epochs 100 \
--hyper-ad 0 --reduction-ratio 4 \
--EDL 1 --kl-coef 5e-3 \
--save-name testc1 2>&1 | tee ./logs/testc1.log
#
#CUDA_VISIBLE_DEVICES=1 python train.py --batch-size 128 --epochs 100 \
#--hyper-ad 0 --reduction-ratio 4 \
#--EDL 1 --kl-coef 5e-3 \
#--save-name testc06 2>&1 | tee ./logs/testc06.log
#

