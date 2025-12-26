# Add your program code here.

# print("Welcome to Project 1!")

import pandas as pd
import matplotlib.pyplot as plt
import json, re, os

# 1. 解析 log.txt
# log_file = '/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection/log.txt'
log_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection_p4_100_usePre/log.txt"
epochs, map50, map5095 = [], [], []
loss_total, loss_vfl, loss_bbox, loss_giou = [], [], [], []

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        epochs.append(data['epoch'])
        coco = data['test_coco_eval_bbox']
        map50.append(coco[0])          # COCO mAP@0.5
        map5095.append(coco[1])        # COCO mAP@0.5:0.95
        loss_total.append(data['train_loss'])
        loss_vfl.append(data['train_loss_vfl'])
        loss_bbox.append(data['train_loss_bbox'])
        loss_giou.append(data['train_loss_giou'])


df = pd.DataFrame({'epoch':epochs,'mAP50':map50,'mAP5095':map5095,
                   'loss':loss_total,'VFL':loss_vfl,'BBox':loss_bbox,'GIoU':loss_giou})

# 2. mAP 曲线
plt.figure(figsize=(6,3.5))
plt.plot(df.epoch, df.mAP50,  label='mAP@0.5',  color='#1f77b4', lw=2)
plt.plot(df.epoch, df.mAP5095,label='mAP@0.5:0.95', color='#ff7f0e', lw=2)
plt.xlabel('Epoch'); plt.ylabel('mAP'); plt.title('DETR Validation mAP')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('rtdetrv2_pytorch/plot/mAP_curve.png', dpi=300); plt.close()

# 3. Loss 曲线
plt.figure(figsize=(6,3.5))
plt.plot(df.epoch, df.loss, label='Total', lw=2)
plt.plot(df.epoch, df.VFL,   label='VFL', lw=1.5)
plt.plot(df.epoch, df.BBox,  label='BBox', lw=1.5)
plt.plot(df.epoch, df.GIoU,  label='GIoU', lw=1.5)
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('DETR Training Loss')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('rtdetrv2_pytorch/plot/Loss_curve.png', dpi=300); plt.close()

print("✅ 已生成 mAP_curve.png 和 Loss_curve.png，可直接插入 PPT")