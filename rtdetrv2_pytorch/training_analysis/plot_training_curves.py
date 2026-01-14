#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制训练曲线：loss、mAP/AP50、AR等
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path):
    """解析日志文件"""
    data = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if 'epoch' in entry:
                    data.append(entry)
            except json.JSONDecodeError:
                continue
    return data


def plot_loss_curves(data, output_dir):
    """绘制loss曲线"""
    epochs = [d['epoch'] for d in data]
    train_loss = [d.get('train_loss', 0) for d in data]
    
    # 提取主要loss组件
    loss_vfl = [d.get('train_loss_vfl', 0) for d in data]
    loss_bbox = [d.get('train_loss_bbox', 0) for d in data]
    loss_giou = [d.get('train_loss_giou', 0) for d in data]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 总loss
    axes[0].plot(epochs, train_loss, 'b-', linewidth=2, label='Total Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    
    # Loss组件
    axes[1].plot(epochs, loss_vfl, 'r-', linewidth=1.5, label='VFL Loss', alpha=0.8)
    axes[1].plot(epochs, loss_bbox, 'g-', linewidth=1.5, label='BBox Loss', alpha=0.8)
    axes[1].plot(epochs, loss_giou, 'm-', linewidth=1.5, label='GIoU Loss', alpha=0.8)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Loss Components', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    output_path = output_dir / "loss_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss curves to: {output_path}")
    plt.close()


def plot_map_curves(data, output_dir):
    """绘制mAP和AP50曲线"""
    epochs = []
    ap_50_95 = []
    ap_50 = []
    ap_75 = []
    
    for d in data:
        if 'test_coco_eval_bbox' in d and len(d['test_coco_eval_bbox']) >= 3:
            epochs.append(d['epoch'])
            ap_50_95.append(d['test_coco_eval_bbox'][0])
            ap_50.append(d['test_coco_eval_bbox'][1])
            ap_75.append(d['test_coco_eval_bbox'][2])
    
    if not epochs:
        print("No evaluation data found!")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(epochs, ap_50_95, 'b-', linewidth=2, label='AP@0.50:0.95', marker='o', markersize=4)
    ax.plot(epochs, ap_50, 'g-', linewidth=2, label='AP@0.50', marker='s', markersize=4)
    ax.plot(epochs, ap_75, 'r-', linewidth=2, label='AP@0.75', marker='^', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Precision (AP)', fontsize=12)
    ax.set_title('Average Precision (AP) Curves', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    output_path = output_dir / "map_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved mAP curves to: {output_path}")
    plt.close()


def plot_ar_curves(data, output_dir):
    """绘制AR曲线"""
    epochs = []
    ar_1 = []
    ar_10 = []
    ar_100 = []
    
    for d in data:
        if 'test_coco_eval_bbox' in d and len(d['test_coco_eval_bbox']) >= 9:
            epochs.append(d['epoch'])
            ar_1.append(d['test_coco_eval_bbox'][6])
            ar_10.append(d['test_coco_eval_bbox'][7])
            ar_100.append(d['test_coco_eval_bbox'][8])
    
    if not epochs:
        print("No AR data found!")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(epochs, ar_1, 'b-', linewidth=2, label='AR@0.50:0.95 (maxDets=1)', marker='o', markersize=4)
    ax.plot(epochs, ar_10, 'g-', linewidth=2, label='AR@0.50:0.95 (maxDets=10)', marker='s', markersize=4)
    ax.plot(epochs, ar_100, 'r-', linewidth=2, label='AR@0.50:0.95 (maxDets=100)', marker='^', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Recall (AR)', fontsize=12)
    ax.set_title('Average Recall (AR) Curves', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    output_path = output_dir / "ar_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved AR curves to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument(
        "--log",
        type=str,
        # 针对 rtdetrv2_r50vd_cancer_detection_split_dataset_aug 模型的默认日志路径
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r50vd_cancer_detection_split_dataset_0105/log.txt",
        help="Path to training log file (relative to rtdetrv2_pytorch/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for plots (default: output/)"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    if Path(args.log).is_absolute():
        log_path = Path(args.log)
    else:
        log_path = (script_dir.parent / args.log).resolve()
        if not log_path.exists():
            log_path = (script_dir / args.log).resolve()
        if not log_path.exists():
            log_path = Path(args.log).resolve()
    
    if Path(args.output_dir).is_absolute():
        output_dir = Path(args.output_dir)
    else:
        output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return
    
    print(f"Reading log file: {log_path}")
    data = parse_log_file(log_path)
    print(f"Found {len(data)} epochs")
    
    # 绘制各种曲线
    plot_loss_curves(data, output_dir)
    plot_map_curves(data, output_dir)
    plot_ar_curves(data, output_dir)
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()

