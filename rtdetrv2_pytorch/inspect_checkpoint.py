import torch
"""
安全清理建议分两步：
先确认数量（可保留想留的 best/last 等）：
cd /home/ubuntu/lsn/project_new/RT-DETR-main && find rtdetrv2_pytorch/output -type f -name 'checkpoint*.pth' | wc -l
确认后一次性删除：
cd /home/ubuntu/lsn/project_new/RT-DETR-main && find rtdetrv2_pytorch/output -type f -name 'checkpoint*.pth' -delete
如只删某个子目录，可加限定：
find rtdetrv2_pytorch/output/rtdetrv2_r50vd_cancer_detection_split_dataset_aug1 -type f -name 'checkpoint*.pth' -delete
"""