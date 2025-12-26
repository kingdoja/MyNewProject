#!/bin/bash

cd /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch

# 带完整参数的微调命令
python tools/train.py \
  -c /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_cancer_detection.yml \
  --tuning /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/rtdetr-l.pt \   # 修改预训练模型路径
  --use-amp \
  --seed 42 \
  --output-dir ./output/fine_tuned_model \
  -u epoches=30 \
  -u optimizer.lr=0.00001 \
  -u optimizer.params.0.lr=0.000001 \
  -u lr_scheduler.milestones=[10,20]

# ./finetune.sh

# python train.py \
#   --config configs/rtdetrv2/rtdetrv2_r18vd_cancer_detection.yml \
#   --tuning rtdetr-l.pt \
#   --use-amp \
#   --seed 42

 # git clone https://github.com/roboflow/rf-detr.git
