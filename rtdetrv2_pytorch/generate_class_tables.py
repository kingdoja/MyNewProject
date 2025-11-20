#!/usr/bin/env python3
"""
根据多个推理结果目录下的 statistics.txt 生成
 - 12 类别的数量表
 - 12 类别的占比表
 - 对应的可视化热力图

默认会寻找 <root>/visualization/statistics.txt，可通过 --stats_files
直接传入 statistics.txt 路径。

命令行：
python rtdetrv2_pytorch/generate_class_tables.py \
  --roots DataPatchesInference/Patches1 DataPatchesInference/Patches2 \
          DataPatchesInference/Patches3 DataPatchesInference/Patches4 \
          DataPatchesInference/Patches5 
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 固定类别顺序
COCO_CLASSES = ['AD', 'BC', 'EC', 'L', 'LC', 'M', 'NT', 'SM', 'SQ', 'TC1', 'TC2', 'TC3']


def parse_statistics_file(stat_path: Path) -> Tuple[str, int, Dict[str, int]]:
    """
    解析 visualization/statistics.txt，返回 (数据集名称，总检测数，类别计数字典)
    """
    dataset_name = stat_path.parent.parent.name  # 例如 Patches1
    total_detections = 0
    class_counts: Dict[str, int] = {cls: 0 for cls in COCO_CLASSES}

    with stat_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Total detections:"):
                total_detections = int(line.split(":")[1].strip())
            elif line.startswith(tuple(f"{cls}:" for cls in COCO_CLASSES)):
                cls, value = line.split(":")
                cls = cls.strip()
                class_counts[cls] = int(value.strip())

    return dataset_name, total_detections, class_counts


def gather_statistics(stat_files: List[Path]) -> Tuple[List[str], Dict[str, Dict[str, int]], Dict[str, int]]:
    dataset_order: List[str] = []
    counts_table: Dict[str, Dict[str, int]] = {}
    totals: Dict[str, int] = {}

    for stat_path in stat_files:
        dataset, total_det, class_counts = parse_statistics_file(stat_path)
        dataset_order.append(dataset)
        totals[dataset] = total_det
        counts_table[dataset] = class_counts

    return dataset_order, counts_table, totals


def save_table_csv(table: Dict[str, Dict[str, float]], datasets: List[str], output_path: Path, is_percentage: bool):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        header = ["dataset"] + COCO_CLASSES
        f.write(",".join(header) + "\n")
        for dataset in datasets:
            row = [dataset]
            for cls in COCO_CLASSES:
                value = table[dataset][cls]
                if is_percentage:
                    row.append(f"{value:.2f}")
                else:
                    row.append(str(int(value)))
            f.write(",".join(row) + "\n")


def render_heatmap(table: Dict[str, Dict[str, float]], datasets: List[str], title: str,
                   output_path: Path, fmt: str):
    data = np.array([[table[dataset][cls] for cls in COCO_CLASSES] for dataset in datasets])

    plt.figure(figsize=(len(COCO_CLASSES) * 0.8 + 2, len(datasets) * 0.6 + 2))
    im = plt.imshow(data, aspect='auto', cmap='viridis')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(ticks=np.arange(len(COCO_CLASSES)), labels=COCO_CLASSES, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(datasets)), labels=datasets)
    plt.title(title)

    # 在格子内标注数值
    for i in range(len(datasets)):
        for j in range(len(COCO_CLASSES)):
            value = data[i, j]
            if fmt == 'd':
                text = f"{int(value)}"
            else:
                text = f"{value:.2f}"
            plt.text(j, i, text, ha='center', va='center', color='white',
                     fontsize=8, fontweight='bold')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="生成12类别数量/占比表及热力图")
    parser.add_argument("--roots", nargs="+", default=[],
                        help="推理结果根目录，例如 DataPatchesInference/Patches1")
    parser.add_argument("--stats_files", nargs="+", default=[],
                        help="直接指定 statistics.txt 路径")
    parser.add_argument("--output_dir", type=str, default="/home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesInference/aggregate_outputs",
                        help="表格和图像输出目录")
    args = parser.parse_args()

    stat_paths: List[Path] = []
    for root in args.roots:
        stat_paths.append(Path(root) / "visualization" / "statistics.txt")
    for path in args.stats_files:
        stat_paths.append(Path(path))

    stat_paths = [p for p in stat_paths if p.exists()]
    if not stat_paths:
        raise FileNotFoundError("未找到任何 statistics.txt，请检查路径")

    datasets, counts_table, totals = gather_statistics(stat_paths)

    percentages_table: Dict[str, Dict[str, float]] = {}
    for dataset in datasets:
        total = totals[dataset] or 1  # 避免除零
        percentages_table[dataset] = {
            cls: (counts_table[dataset][cls] / total * 100.0)
            for cls in COCO_CLASSES
        }

    output_dir = Path(args.output_dir)
    counts_csv_path = output_dir / "class_counts.csv"
    perc_csv_path = output_dir / "class_percentages.csv"
    save_table_csv(counts_table, datasets, counts_csv_path, is_percentage=False)
    save_table_csv(percentages_table, datasets, perc_csv_path, is_percentage=True)

    render_heatmap(counts_table, datasets,
                   "Class Counts per Dataset",
                   output_dir / "class_counts_heatmap.png", fmt='d')
    render_heatmap(percentages_table, datasets,
                   "Class Percentages per Dataset (%)",
                   output_dir / "class_percentages_heatmap.png", fmt='.2f')

    print(f"✅ 数量表已保存至: {counts_csv_path}")
    print(f"✅ 占比表已保存至: {perc_csv_path}")
    print(f"✅ 数量热力图: {output_dir / 'class_counts_heatmap.png'}")
    print(f"✅ 占比热力图: {output_dir / 'class_percentages_heatmap.png'}")


if __name__ == "__main__":
    main()

