#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制更适合论文展示的训练与评估曲线：
- loss 曲线（raw + EMA）
- train/val loss 对比 + 泛化差距
- AP 曲线（含最佳 epoch 标注）
- AR 曲线
- 汇总面板图（dashboard）
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#1f77b4",
    "green": "#2ca02c",
    "red": "#d62728",
    "orange": "#ff7f0e",
    "purple": "#9467bd",
    "gray": "#7f7f7f",
}


def setup_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.alpha": 0.25,
    })


def parse_log_file(log_path):
    data = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "epoch" in entry:
                    data.append(entry)
            except json.JSONDecodeError:
                continue
    return data


def resolve_input_path(path_str, script_dir):
    input_path = Path(path_str)
    if input_path.is_absolute():
        return input_path
    for base in (script_dir, script_dir.parent, script_dir.parent.parent):
        candidate = (base / input_path).resolve()
        if candidate.exists():
            return candidate
    return input_path.resolve()


def ema_smooth(values, alpha=0.3):
    if not values:
        return []
    out = []
    ema = values[0]
    for v in values:
        if np.isnan(v):
            out.append(np.nan)
            continue
        ema = alpha * v + (1.0 - alpha) * ema
        out.append(ema)
    return out


def extract_series(data, keys):
    for key in keys:
        vals = [d.get(key, None) for d in data]
        if any(v is not None for v in vals):
            return [np.nan if v is None else float(v) for v in vals], key
    return None, None


def extract_eval_series(data):
    epochs, ap5095, ap50, ap75, ar1, ar10, ar100 = [], [], [], [], [], [], []
    for d in data:
        metrics = d.get("test_coco_eval_bbox", [])
        if len(metrics) >= 9:
            epochs.append(int(d["epoch"]))
            ap5095.append(float(metrics[0]))
            ap50.append(float(metrics[1]))
            ap75.append(float(metrics[2]))
            ar1.append(float(metrics[6]))
            ar10.append(float(metrics[7]))
            ar100.append(float(metrics[8]))
    return epochs, ap5095, ap50, ap75, ar1, ar10, ar100


def annotate_best(ax, x, y, label):
    if not x:
        return
    best_idx = int(np.nanargmax(y))
    bx, by = x[best_idx], y[best_idx]
    ax.scatter([bx], [by], s=45, c=COLORS["red"], zorder=5)
    ax.annotate(
        f"{label}: {by:.3f} (ep {bx})",
        xy=(bx, by),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        color=COLORS["red"],
    )


def plot_loss_curves(data, output_dir):
    epochs = [int(d["epoch"]) for d in data]
    train_loss = [float(d.get("train_loss", np.nan)) for d in data]
    vfl = [float(d.get("train_loss_vfl", np.nan)) for d in data]
    bbox = [float(d.get("train_loss_bbox", np.nan)) for d in data]
    giou = [float(d.get("train_loss_giou", np.nan)) for d in data]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(epochs, train_loss, color=COLORS["blue"], alpha=0.35, lw=1.2, label="Train Loss (raw)")
    axes[0].plot(epochs, ema_smooth(train_loss, alpha=0.35), color=COLORS["blue"], lw=2.2, label="Train Loss (EMA)")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss Trend", fontweight="bold")
    axes[0].legend(loc="upper right")

    axes[1].plot(epochs, ema_smooth(vfl, alpha=0.35), color=COLORS["red"], lw=2.0, label="VFL")
    axes[1].plot(epochs, ema_smooth(bbox, alpha=0.35), color=COLORS["green"], lw=2.0, label="BBox")
    axes[1].plot(epochs, ema_smooth(giou, alpha=0.35), color=COLORS["purple"], lw=2.0, label="GIoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Component Loss")
    axes[1].set_title("Loss Components (EMA Smoothed)", fontweight="bold")
    axes[1].legend(loc="upper right", ncol=3)

    plt.tight_layout()
    path = output_dir / "loss_curves.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved loss curves to: {path}")
    plt.close()


def plot_train_val_loss_comparison(data, output_dir):
    epochs = [int(d["epoch"]) for d in data]
    train_loss = [float(d.get("train_loss", np.nan)) for d in data]
    val_loss, val_key = extract_series(data, ["val_loss", "test_loss", "eval_loss", "validation_loss"])

    if np.all(np.isnan(train_loss)) and (val_loss is None or np.all(np.isnan(val_loss))):
        print("No train/val loss data found for comparison!")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(epochs, ema_smooth(train_loss, alpha=0.35), color=COLORS["blue"], lw=2.2, label="Train Loss (EMA)")
    if val_loss is not None:
        axes[0].plot(epochs, ema_smooth(val_loss, alpha=0.35), color=COLORS["orange"], lw=2.2, label=f"Val Loss ({val_key})")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train vs Validation Loss", fontweight="bold")
    axes[0].legend(loc="upper right")

    if val_loss is not None:
        gap = np.array(val_loss) - np.array(train_loss)
        axes[1].plot(epochs, gap, color=COLORS["gray"], lw=2.0, label="Generalization Gap (Val - Train)")
        axes[1].axhline(0.0, color="#444444", lw=1.0, ls="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Gap")
    axes[1].set_title("Generalization Gap", fontweight="bold")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    path = output_dir / "train_val_loss_curves.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved train/val loss comparison to: {path}")
    plt.close()


def plot_map_curves(data, output_dir):
    epochs, ap5095, ap50, ap75, _, _, _ = extract_eval_series(data)
    if not epochs:
        print("No evaluation data found!")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(epochs, ap5095, color=COLORS["blue"], lw=2.4, label="AP@0.50:0.95")
    ax.plot(epochs, ap50, color=COLORS["green"], lw=2.0, label="AP@0.50")
    ax.plot(epochs, ap75, color=COLORS["red"], lw=2.0, label="AP@0.75")
    annotate_best(ax, epochs, ap5095, "Best AP@0.50:0.95")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Precision")
    ax.set_title("COCO AP Curves", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = output_dir / "map_curves.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved mAP curves to: {path}")
    plt.close()


def plot_ar_curves(data, output_dir):
    epochs, _, _, _, ar1, ar10, ar100 = extract_eval_series(data)
    if not epochs:
        print("No AR data found!")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(epochs, ar1, color=COLORS["blue"], lw=2.0, label="AR@1")
    ax.plot(epochs, ar10, color=COLORS["green"], lw=2.0, label="AR@10")
    ax.plot(epochs, ar100, color=COLORS["purple"], lw=2.4, label="AR@100")
    annotate_best(ax, epochs, ar100, "Best AR@100")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Recall")
    ax.set_title("COCO AR Curves", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = output_dir / "ar_curves.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved AR curves to: {path}")
    plt.close()


def plot_dashboard(data, output_dir):
    epochs = [int(d["epoch"]) for d in data]
    train_loss = [float(d.get("train_loss", np.nan)) for d in data]
    val_loss, _ = extract_series(data, ["val_loss", "test_loss", "eval_loss", "validation_loss"])
    eval_epochs, ap5095, ap50, ap75, ar1, ar10, ar100 = extract_eval_series(data)
    if not epochs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, ema_smooth(train_loss, alpha=0.35), color=COLORS["blue"], lw=2.2, label="Train")
    if val_loss is not None:
        axes[0, 0].plot(epochs, ema_smooth(val_loss, alpha=0.35), color=COLORS["orange"], lw=2.2, label="Val")
    axes[0, 0].set_title("Loss", fontweight="bold")
    axes[0, 0].legend()

    if eval_epochs:
        axes[0, 1].plot(eval_epochs, ap5095, color=COLORS["blue"], lw=2.2, label="AP@0.50:0.95")
        axes[0, 1].plot(eval_epochs, ap50, color=COLORS["green"], lw=1.8, label="AP@0.50")
        axes[0, 1].plot(eval_epochs, ap75, color=COLORS["red"], lw=1.8, label="AP@0.75")
        axes[0, 1].legend()
    axes[0, 1].set_title("Precision", fontweight="bold")

    if eval_epochs:
        axes[1, 0].plot(eval_epochs, ar1, color=COLORS["blue"], lw=1.8, label="AR@1")
        axes[1, 0].plot(eval_epochs, ar10, color=COLORS["green"], lw=1.8, label="AR@10")
        axes[1, 0].plot(eval_epochs, ar100, color=COLORS["purple"], lw=2.2, label="AR@100")
        axes[1, 0].legend()
    axes[1, 0].set_title("Recall", fontweight="bold")

    if val_loss is not None:
        gap = np.array(val_loss) - np.array(train_loss)
        axes[1, 1].plot(epochs, gap, color=COLORS["gray"], lw=2.0)
        axes[1, 1].axhline(0.0, color="#444444", lw=1.0, ls="--")
    axes[1, 1].set_title("Generalization Gap", fontweight="bold")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)

    fig.suptitle("Training & Evaluation Dashboard", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.0, 1, 0.97])
    path = output_dir / "paper_dashboard.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved dashboard to: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training/evaluation curves in publication style")
    parser.add_argument(
        "--log",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r50vd_cancer_detection1_incremental_ft_oaug_0309/log.txt",
        help="Path to training log file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    setup_plot_style()

    script_dir = Path(__file__).parent
    log_path = resolve_input_path(args.log, script_dir)

    if Path(args.output_dir).is_absolute():
        output_dir = Path(args.output_dir)
    else:
        output_dir = (script_dir.parent / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return

    print(f"Reading log file: {log_path}")
    data = parse_log_file(log_path)
    print(f"Found {len(data)} epochs")

    plot_loss_curves(data, output_dir)
    plot_train_val_loss_comparison(data, output_dir)
    plot_map_curves(data, output_dir)
    plot_ar_curves(data, output_dir)
    plot_dashboard(data, output_dir)
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()

