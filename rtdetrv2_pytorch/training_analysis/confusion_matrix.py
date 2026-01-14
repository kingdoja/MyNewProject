#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 COCO 格式的预测结果和标注，绘制类别级别的混淆矩阵。

默认使用与 accuracy.py 相同的数据与路径：
- GT:  split_dataset_aug/val/annotations/instances_val.json
- Pred: training_analysis/output/validation_visualization_split_dataset_aug_r50_unUsePre/val_detections.json

运行方式（在 rtdetrv2_pytorch/training_analysis 目录下）:
    python confusion_matrix.py

也可以通过命令行参数自定义路径:
    python confusion_matrix.py \
        --gt /path/to/instances_val.json \
        --pred /path/to/val_detections.json \
        --output_dir output
"""

import json
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from accuracy import load_class_names_from_data2000, calculate_iou


def build_confusion_pairs(predictions_file,
                          ground_truth_file,
                          iou_threshold=0.5,
                          data_root="/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug"):
    """
    构建混淆矩阵所需的 (y_true, y_pred) 序列

    - y_true: 每个 GT 实例所属的真实类别索引
    - y_pred: 对应 GT 的预测类别索引，如果未匹配则为背景类索引
    """
    # 读取真实标签和预测结果
    with open(ground_truth_file, 'r') as f:
        gt_data = json.load(f)

    with open(predictions_file, 'r') as f:
        pred_data = json.load(f)

    # 提取类别信息
    if 'categories' in gt_data and gt_data['categories']:
        class_names = [cat['name'] for cat in gt_data['categories']]
        class_ids = [cat['id'] for cat in gt_data['categories']]
    else:
        class_names = load_class_names_from_data2000(data_root)
        if class_names:
            class_ids = list(range(len(class_names)))
        else:
            # 与 accuracy.py 中的默认类别保持一致
            class_names = ["AD", "BC", "EC", "L", "LC", "M", "NT", "SM", "SQ", "TC1", "TC2", "TC3"]
            class_ids = list(range(len(class_names)))

    num_classes = len(class_names)

    # 额外增加一个“未检出/背景”类别
    background_idx = num_classes
    all_class_names = class_names + ["BG"]  # BG = 未检测到 / 背景

    # 类别 ID -> 索引
    id_to_index = {id_: idx for idx, id_ in enumerate(class_ids)}

    # 构建 image_id -> anns
    gt_annotations = gt_data.get('annotations', [])
    image_id_to_gt = {}
    for ann in gt_annotations:
        image_id = ann['image_id']
        image_id_to_gt.setdefault(image_id, []).append(ann)

    image_id_to_pred = {}
    for ann in pred_data:
        image_id = ann['image_id']
        image_id_to_pred.setdefault(image_id, []).append(ann)

    y_true = []
    y_pred = []

    # 对每张图片进行匹配
    for image_id, gt_anns in image_id_to_gt.items():
        preds = image_id_to_pred.get(image_id, [])
        unmatched_preds = preds.copy()

        for gt_ann in gt_anns:
            gt_class_id = gt_ann['category_id']
            gt_bbox = gt_ann['bbox']  # [x, y, w, h]

            best_iou = 0.0
            best_pred_idx = -1
            best_pred_class = None

            for i, pred_ann in enumerate(unmatched_preds):
                pred_class_id = pred_ann['category_id']
                pred_bbox = pred_ann['bbox']
                iou = calculate_iou(gt_bbox, pred_bbox)

                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pred_idx = i
                    best_pred_class = pred_class_id

            # 真实类别索引
            gt_idx = id_to_index.get(gt_class_id, None)
            if gt_idx is None:
                # 若 GT 类别在映射中不存在，则跳过
                continue

            # 记录 y_true
            y_true.append(gt_idx)

            # 记录 y_pred（匹配不到则记为背景）
            if best_pred_idx == -1 or best_pred_class is None:
                y_pred.append(background_idx)
            else:
                pred_idx = id_to_index.get(best_pred_class, background_idx)
                y_pred.append(pred_idx)
                # 将该预测从未匹配列表中移除
                unmatched_preds.pop(best_pred_idx)

    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64), all_class_names


def plot_confusion_matrix(y_true,
                          y_pred,
                          class_names,
                          normalize=True,
                          title="Confusion Matrix",
                          cmap=plt.cm.Blues,
                          save_path=None):
    """
    绘制并保存混淆矩阵图像
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    if normalize:
        with np.errstate(all='ignore'):
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm_normalized = cm.astype('float') / np.maximum(cm_sum, 1)
        cm_to_show = cm_normalized
    else:
        cm_to_show = cm

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_to_show, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # 设置坐标轴
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在格子中写数字
    fmt = '.2f' if normalize else 'd'
    thresh = (cm_to_show.max() + cm_to_show.min()) / 2.0
    for i in range(cm_to_show.shape[0]):
        for j in range(cm_to_show.shape[1]):
            value = cm_to_show[i, j]
            if (not normalize and value == 0) or (normalize and np.isnan(value)):
                continue
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="绘制目标检测分类效果的混淆矩阵")
    parser.add_argument(
        "--gt",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug/val/annotations/instances_val.json",
        help="COCO GT 标注文件路径",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/training_analysis/output/validation_visualization_split_dataset_aug_r50_0107/val_detections.json",
        help="模型预测结果（COCO 格式）路径",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug",
        help="数据集根目录，用于 classes.txt 或配置中加载类别名称",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.3,
        help="匹配 GT 与预测框时的 IoU 阈值",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="输出目录，相对于 training_analysis/ 目录",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    if Path(args.output_dir).is_absolute():
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir / args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "confusion_matrix.png"

    print(f"GT file:   {args.gt}")
    print(f"Pred file: {args.pred}")
    print(f"IoU thresh: {args.iou_thresh}")

    y_true, y_pred, class_names = build_confusion_pairs(
        predictions_file=args.pred,
        ground_truth_file=args.gt,
        iou_threshold=args.iou_thresh,
        data_root=args.data_root,
    )

    if len(y_true) == 0:
        print("没有有效的 GT 实例，无法绘制混淆矩阵。请检查标注与预测文件。")
        return

    print(f"有效 GT 实例数: {len(y_true)}")
    print(f"类别数（含 BG）: {len(class_names)}")

    title = f"Confusion Matrix (IoU≥{args.iou_thresh:.2f})"
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        normalize=True,
        title=title,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()


