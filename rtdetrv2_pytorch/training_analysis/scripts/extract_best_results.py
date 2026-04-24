#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取训练日志中的最佳评估结果并生成表格
"""

import json
import argparse
from pathlib import Path


def parse_log_file(log_path):
    """解析日志文件，提取所有epoch的数据"""
    data = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if 'test_coco_eval_bbox' in entry:
                    data.append(entry)
            except json.JSONDecodeError:
                continue
    return data


def find_best_epoch(data):
    """找到最佳epoch（基于AP@0.50:0.95）"""
    best_epoch = None
    best_ap = -1
    
    for entry in data:
        if 'test_coco_eval_bbox' in entry and len(entry['test_coco_eval_bbox']) > 0:
            ap = entry['test_coco_eval_bbox'][0]  # AP@0.50:0.95
            if ap > best_ap:
                best_ap = ap
                best_epoch = entry
    
    return best_epoch


def format_coco_output(eval_bbox):
    """
    格式化COCO评估指标，输出格式与终端显示一致
    eval_bbox包含12个值：
    0: AP@0.50:0.95 (all)
    1: AP@0.50 (all)
    2: AP@0.75 (all)
    3: AP@0.50:0.95 (small)
    4: AP@0.50:0.95 (medium)
    5: AP@0.50:0.95 (large)
    6: AR@0.50:0.95 (all, maxDets=1)
    7: AR@0.50:0.95 (all, maxDets=10)
    8: AR@0.50:0.95 (all, maxDets=100)
    9: AR@0.50:0.95 (small, maxDets=100)
    10: AR@0.50:0.95 (medium, maxDets=100)
    11: AR@0.50:0.95 (large, maxDets=100)
    
    注意：AR@IoU=0.50 和 AR@IoU=0.75 不在标准12个值中，需要从评估结果中提取
    如果eval_bbox有14个值，则：
    12: AR@0.50 (all, maxDets=100)
    13: AR@0.75 (all, maxDets=100)
    """
    if len(eval_bbox) < 12:
        return []
    
    lines = []
    # AP metrics
    lines.append(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {eval_bbox[0]:.3f}")
    lines.append(f"Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {eval_bbox[1]:.3f}")
    lines.append(f"Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {eval_bbox[2]:.3f}")
    lines.append(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {eval_bbox[3]:.3f}")
    lines.append(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {eval_bbox[4]:.3f}")
    lines.append(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {eval_bbox[5]:.3f}")
    # AR metrics
    lines.append(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {eval_bbox[6]:.3f}")
    lines.append(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {eval_bbox[7]:.3f}")
    lines.append(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {eval_bbox[8]:.3f}")
    lines.append(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {eval_bbox[9]:.3f}")
    lines.append(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {eval_bbox[10]:.3f}")
    lines.append(f"Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {eval_bbox[11]:.3f}")
    
    # Additional AR metrics (IoU=0.50 and 0.75) - 如果数据中有这些值
    if len(eval_bbox) >= 14:
        lines.append(f"Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = {eval_bbox[12]:.3f}")
        lines.append(f"Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 ] = {eval_bbox[13]:.3f}")
    elif len(eval_bbox) == 12:
        # 如果只有12个值，尝试从其他字段获取或使用近似值
        # 注意：这些值可能不在日志中，需要从实际评估结果中获取
        # 这里我们暂时跳过，因为标准COCO评估输出只有12个值
        pass
    
    return lines




def generate_table(best_epoch, output_dir):
    """生成表格"""
    if not best_epoch:
        print("No best epoch found!")
        return
    
    eval_bbox = best_epoch.get('test_coco_eval_bbox', [])
    epoch = best_epoch.get('epoch', 'N/A')
    
    # 输出COCO格式的结果（与终端显示一致）
    print(f"\nBest Epoch: {epoch}")
    print(f"IoU metric: bbox")
    print()
    
    coco_lines = format_coco_output(eval_bbox)
    for line in coco_lines:
        print(line)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存到文件
    output_file = output_dir / "best_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Best Epoch: {epoch}\n")
        f.write(f"IoU metric: bbox\n\n")
        for line in coco_lines:
            f.write(line + "\n")
        f.write("\n")
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract best evaluation results from training log")
    parser.add_argument(
        "--log",
        type=str,
        # 针对 rtdetrv2_r50vd_cancer_detection_split_dataset_aug 模型
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r50vd_cancer_detection1_incremental_ft_oaug_0309/log.txt",
        help="Path to training log file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for results (default: output/)"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent  # training_analysis/scripts/
    
    if Path(args.log).is_absolute():
        log_path = Path(args.log)
    else:
        # 依次尝试 scripts/、training_analysis/、rtdetrv2_pytorch/
        candidates = [
            (script_dir / args.log).resolve(),
            (script_dir.parent / args.log).resolve(),
            (script_dir.parent.parent / args.log).resolve(),
            Path(args.log).resolve(),
        ]
        log_path = next((p for p in candidates if p.exists()), candidates[-1])
    
    if Path(args.output_dir).is_absolute():
        output_dir = Path(args.output_dir)
    else:
        output_dir = (script_dir.parent / args.output_dir).resolve()
    
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return
    
    print(f"Reading log file: {log_path}")
    data = parse_log_file(log_path)
    print(f"Found {len(data)} epochs")
    
    best_epoch = find_best_epoch(data)
    if best_epoch:
        generate_table(best_epoch, output_dir)
    else:
        print("No best epoch found!")


if __name__ == "__main__":
    main()

