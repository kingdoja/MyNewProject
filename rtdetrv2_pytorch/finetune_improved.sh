#!/bin/bash

cd /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch

# 带完整参数的微调命令 - 改进版
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_cancer_detection_improved.yml \
  --tuning rtdetr-r18vd.pt \
  --use-amp \
  --seed 42 \
  --output-dir ./output/fine_tuned_model_improved \
  -u epoches=50 \
  -u optimizer.lr=0.00001 \
  -u optimizer.params.0.lr=0.000001 \
  -u lr_scheduler.milestones=[15,30,45]

# git clone https://github.com/roboflow/rf-detr.git