#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制检测评估补充图（论文常用诊断图）：
- Precision-Recall 曲线（IoU 阈值固定）
- F1-Confidence 阈值曲线（含最优阈值）
- 预测置信度分布（TP/FP）
- 每类别召回率柱状图
- 每类别 AP 柱状图（per_class_ap_bar.png）
- 错误分解图：分类错/定位错/漏检（error_breakdown.png）
- 尺寸分组 AP/AR（size_wise_ap_ar.png）
- 校准曲线 + ECE（calibration_curve_ece.png）
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from accuracy import calculate_iou


COLORS = {
    "tp": "#2ca02c",
    "fp": "#d62728",
    "main": "#1f77b4",
    "aux": "#9467bd",
    "orange": "#ff7f0e",
    "gray": "#7f7f7f",
}


def resolve_input_path(path_str, script_dir):
    input_path = Path(path_str)
    if input_path.is_absolute():
        return input_path
    for base in (script_dir, script_dir.parent, script_dir.parent.parent):
        candidate = (base / input_path).resolve()
        if candidate.exists():
            return candidate
    return input_path.resolve()


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "savefig.dpi": 300,
        "grid.alpha": 0.25,
    })


def load_data(gt_path, pred_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(pred_path, "r", encoding="utf-8") as f:
        pred = json.load(f)
    return gt, pred


def build_index(gt_data, pred_data):
    image_to_gt = {}
    for ann in gt_data.get("annotations", []):
        image_to_gt.setdefault(ann["image_id"], []).append(ann)
    image_to_pred = {}
    for ann in pred_data:
        image_to_pred.setdefault(ann["image_id"], []).append(ann)
    for image_id in image_to_pred:
        image_to_pred[image_id].sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return image_to_gt, image_to_pred


def compute_ap(recalls, precisions):
    if len(recalls) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], np.asarray(recalls), [1.0]))
    mpre = np.concatenate(([0.0], np.asarray(precisions), [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def match_predictions(gt_data, pred_data, iou_thresh=0.5):
    """按分数降序进行全局匹配，返回每个预测是否 TP。"""
    image_to_gt, image_to_pred = build_index(gt_data, pred_data)
    gt_matched = {}
    for image_id, gt_list in image_to_gt.items():
        gt_matched[image_id] = [False] * len(gt_list)

    all_preds = []
    for image_id, pred_list in image_to_pred.items():
        for pred_ann in pred_list:
            all_preds.append((image_id, pred_ann))
    all_preds.sort(key=lambda x: float(x[1].get("score", 0.0)), reverse=True)

    pred_scores = []
    pred_is_tp = []
    pred_is_fp = []

    for image_id, pred_ann in all_preds:
        pred_cls = pred_ann["category_id"]
        pred_box = pred_ann["bbox"]
        gt_list = image_to_gt.get(image_id, [])
        matched_flags = gt_matched.get(image_id, [])

        best_iou = 0.0
        best_gt_idx = -1
        for i, gt_ann in enumerate(gt_list):
            if matched_flags[i]:
                continue
            if gt_ann["category_id"] != pred_cls:
                continue
            iou = calculate_iou(gt_ann["bbox"], pred_box)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        is_tp = best_gt_idx >= 0
        if is_tp:
            gt_matched[image_id][best_gt_idx] = True

        pred_scores.append(float(pred_ann.get("score", 0.0)))
        pred_is_tp.append(1 if is_tp else 0)
        pred_is_fp.append(0 if is_tp else 1)

    return np.array(pred_scores), np.array(pred_is_tp), np.array(pred_is_fp)


def evaluate_predictions(gt_data, pred_data, iou_thresh=0.5, score_thresh=0.0):
    gt_anns = gt_data.get("annotations", [])
    categories = gt_data.get("categories", [])
    id_to_name = {c["id"]: c["name"] for c in categories}

    image_to_gt = {}
    for ann in gt_anns:
        image_to_gt.setdefault(ann["image_id"], []).append(ann)

    image_to_pred = {}
    for ann in pred_data:
        if float(ann.get("score", 0.0)) >= score_thresh:
            image_to_pred.setdefault(ann["image_id"], []).append(ann)
    for image_id in image_to_pred:
        image_to_pred[image_id].sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    tp, fp, fn = 0, 0, 0
    tp_scores, fp_scores = [], []
    per_class_total = {}
    per_class_tp = {}

    for ann in gt_anns:
        cid = ann["category_id"]
        per_class_total[cid] = per_class_total.get(cid, 0) + 1
        per_class_tp.setdefault(cid, 0)

    for image_id, gt_list in image_to_gt.items():
        pred_list = image_to_pred.get(image_id, [])
        used = [False] * len(pred_list)

        for gt_ann in gt_list:
            gt_cls = gt_ann["category_id"]
            gt_box = gt_ann["bbox"]
            best_iou = 0.0
            best_j = -1

            for j, pred_ann in enumerate(pred_list):
                if used[j]:
                    continue
                if pred_ann["category_id"] != gt_cls:
                    continue
                iou = calculate_iou(gt_box, pred_ann["bbox"])
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j >= 0:
                used[best_j] = True
                tp += 1
                per_class_tp[gt_cls] = per_class_tp.get(gt_cls, 0) + 1
                tp_scores.append(float(pred_list[best_j].get("score", 0.0)))
            else:
                fn += 1

        # 剩余预测记为 FP
        for j, pred_ann in enumerate(pred_list):
            if not used[j]:
                fp += 1
                fp_scores.append(float(pred_ann.get("score", 0.0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    class_recall = {}
    for cid, total in per_class_total.items():
        class_recall[id_to_name.get(cid, str(cid))] = per_class_tp.get(cid, 0) / max(total, 1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_scores": tp_scores,
        "fp_scores": fp_scores,
        "class_recall": class_recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_per_class_ap(gt_data, pred_data, iou_thresh=0.5):
    categories = gt_data.get("categories", [])
    class_ids = [c["id"] for c in categories]
    id_to_name = {c["id"]: c["name"] for c in categories}
    image_to_gt, image_to_pred = build_index(gt_data, pred_data)

    # 统计每类 GT 数
    class_gt_count = {cid: 0 for cid in class_ids}
    for gt_list in image_to_gt.values():
        for ann in gt_list:
            if ann["category_id"] in class_gt_count:
                class_gt_count[ann["category_id"]] += 1

    per_class_ap = {}
    for cid in class_ids:
        preds = []
        for image_id, pred_list in image_to_pred.items():
            for ann in pred_list:
                if ann["category_id"] == cid:
                    preds.append((image_id, ann))
        preds.sort(key=lambda x: float(x[1].get("score", 0.0)), reverse=True)

        if class_gt_count[cid] == 0:
            per_class_ap[id_to_name.get(cid, str(cid))] = 0.0
            continue

        gt_by_img = {}
        matched = {}
        for image_id, gt_list in image_to_gt.items():
            c_gt = [g for g in gt_list if g["category_id"] == cid]
            gt_by_img[image_id] = c_gt
            matched[image_id] = [False] * len(c_gt)

        tps, fps = [], []
        for image_id, pred_ann in preds:
            pred_box = pred_ann["bbox"]
            gt_list = gt_by_img.get(image_id, [])
            matched_flags = matched.get(image_id, [])
            best_iou = 0.0
            best_idx = -1
            for i, gt_ann in enumerate(gt_list):
                if matched_flags[i]:
                    continue
                iou = calculate_iou(gt_ann["bbox"], pred_box)
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0:
                matched[image_id][best_idx] = True
                tps.append(1)
                fps.append(0)
            else:
                tps.append(0)
                fps.append(1)

        tps = np.cumsum(np.array(tps))
        fps = np.cumsum(np.array(fps))
        recalls = tps / max(class_gt_count[cid], 1)
        precisions = tps / np.maximum(tps + fps, 1)
        per_class_ap[id_to_name.get(cid, str(cid))] = compute_ap(recalls, precisions)

    return per_class_ap


def compute_error_breakdown(gt_data, pred_data, iou_thresh=0.5, loc_iou_low=0.1):
    """返回分类错/定位错/漏检计数（以 GT 为基准）。"""
    image_to_gt, image_to_pred = build_index(gt_data, pred_data)
    cls_err = 0
    loc_err = 0
    miss_err = 0

    for image_id, gt_list in image_to_gt.items():
        preds = image_to_pred.get(image_id, [])
        for gt_ann in gt_list:
            gt_cls = gt_ann["category_id"]
            gt_box = gt_ann["bbox"]
            has_correct = False
            has_cls_confusion = False
            has_loc_issue = False
            for pred_ann in preds:
                iou = calculate_iou(gt_box, pred_ann["bbox"])
                if pred_ann["category_id"] == gt_cls and iou >= iou_thresh:
                    has_correct = True
                    break
                if pred_ann["category_id"] != gt_cls and iou >= iou_thresh:
                    has_cls_confusion = True
                if pred_ann["category_id"] == gt_cls and loc_iou_low <= iou < iou_thresh:
                    has_loc_issue = True
            if has_correct:
                continue
            if has_cls_confusion:
                cls_err += 1
            elif has_loc_issue:
                loc_err += 1
            else:
                miss_err += 1
    return cls_err, loc_err, miss_err


def compute_calibration(scores, labels, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accs, confs, counts = [], [], []
    total = max(len(scores), 1)
    ece = 0.0
    for i in range(n_bins):
        l, r = bins[i], bins[i + 1]
        if i == n_bins - 1:
            idx = (scores >= l) & (scores <= r)
        else:
            idx = (scores >= l) & (scores < r)
        cnt = int(np.sum(idx))
        counts.append(cnt)
        if cnt == 0:
            accs.append(np.nan)
            confs.append(np.nan)
            continue
        acc = float(np.mean(labels[idx]))
        conf = float(np.mean(scores[idx]))
        accs.append(acc)
        confs.append(conf)
        ece += abs(acc - conf) * (cnt / total)
    return bin_centers, np.array(accs), np.array(confs), np.array(counts), ece


def plot_pr_curve(gt_data, pred_data, output_dir, iou_thresh):
    thresholds = np.linspace(0.0, 1.0, 51)
    precisions, recalls = [], []
    for t in thresholds:
        m = evaluate_predictions(gt_data, pred_data, iou_thresh=iou_thresh, score_thresh=float(t))
        precisions.append(m["precision"])
        recalls.append(m["recall"])

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.plot(recalls, precisions, color=COLORS["main"], lw=2.4)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (IoU>={iou_thresh:.2f})", fontweight="bold")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    path = output_dir / "pr_curve.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_f1_threshold_curve(gt_data, pred_data, output_dir, iou_thresh):
    thresholds = np.linspace(0.0, 1.0, 101)
    f1s = []
    for t in thresholds:
        m = evaluate_predictions(gt_data, pred_data, iou_thresh=iou_thresh, score_thresh=float(t))
        f1s.append(m["f1"])

    best_idx = int(np.argmax(f1s))
    best_t = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.plot(thresholds, f1s, color=COLORS["aux"], lw=2.4, label="F1")
    ax.scatter([best_t], [best_f1], c=COLORS["fp"], s=40, zorder=5)
    ax.annotate(f"Best: t={best_t:.2f}, F1={best_f1:.3f}", (best_t, best_f1), textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"F1-Threshold Curve (IoU>={iou_thresh:.2f})", fontweight="bold")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.legend()
    path = output_dir / "f1_threshold_curve.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_confidence_hist(gt_data, pred_data, output_dir, iou_thresh, score_thresh):
    m = evaluate_predictions(gt_data, pred_data, iou_thresh=iou_thresh, score_thresh=score_thresh)
    tp_scores = m["tp_scores"]
    fp_scores = m["fp_scores"]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    bins = np.linspace(0, 1, 25)
    if tp_scores:
        ax.hist(tp_scores, bins=bins, alpha=0.65, color=COLORS["tp"], label="TP", density=True)
    if fp_scores:
        ax.hist(fp_scores, bins=bins, alpha=0.65, color=COLORS["fp"], label="FP", density=True)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title(f"Confidence Distribution (IoU>={iou_thresh:.2f}, score>={score_thresh:.2f})", fontweight="bold")
    ax.legend()
    path = output_dir / "confidence_distribution.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_per_class_recall(gt_data, pred_data, output_dir, iou_thresh, score_thresh):
    m = evaluate_predictions(gt_data, pred_data, iou_thresh=iou_thresh, score_thresh=score_thresh)
    class_recall = m["class_recall"]
    if not class_recall:
        print("No class recall data, skip bar plot.")
        return

    items = sorted(class_recall.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(names, values, color=COLORS["main"], alpha=0.9)
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Recall")
    ax.set_title(f"Per-class Recall (IoU>={iou_thresh:.2f}, score>={score_thresh:.2f})", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    path = output_dir / "per_class_recall.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_per_class_ap_bar(gt_data, pred_data, output_dir, iou_thresh):
    per_class_ap = compute_per_class_ap(gt_data, pred_data, iou_thresh=iou_thresh)
    if not per_class_ap:
        print("No class AP data, skip per_class_ap_bar.")
        return
    items = sorted(per_class_ap.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.bar(names, values, color=COLORS["aux"], alpha=0.9)
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("AP")
    ax.set_title(f"Per-class AP (IoU>={iou_thresh:.2f})", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    path = output_dir / "per_class_ap_bar.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_error_breakdown(gt_data, pred_data, output_dir, iou_thresh):
    cls_err, loc_err, miss_err = compute_error_breakdown(
        gt_data, pred_data, iou_thresh=iou_thresh, loc_iou_low=max(0.1, iou_thresh * 0.3)
    )
    counts = np.array([cls_err, loc_err, miss_err], dtype=np.float64)
    total = max(np.sum(counts), 1.0)
    labels = [
        f"Classification Error\n({cls_err}, {cls_err / total:.1%})",
        f"Localization Error\n({loc_err}, {loc_err / total:.1%})",
        f"Missed Detection\n({miss_err}, {miss_err / total:.1%})",
    ]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.pie(
        counts,
        labels=labels,
        colors=[COLORS["fp"], COLORS["orange"], COLORS["gray"]],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
    )
    ax.set_title(f"Error Breakdown (IoU>={iou_thresh:.2f})", fontweight="bold")
    path = output_dir / "error_breakdown.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_size_wise_ap_ar(gt_path, pred_path, output_dir):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:
        print(f"Skip size_wise_ap_ar: pycocotools unavailable ({exc})")
        return

    try:
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(pred_path))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        # 某些版本中 stats 在 summarize 后才会填充
        coco_eval.summarize()
        stats = coco_eval.stats
    except Exception as exc:
        print(f"Skip size_wise_ap_ar: COCOeval failed ({exc})")
        return

    if not hasattr(stats, "__len__") or len(stats) < 12:
        print(f"Skip size_wise_ap_ar: invalid COCO stats length ({len(stats) if hasattr(stats, '__len__') else 'N/A'})")
        return

    ap_vals = [stats[3], stats[4], stats[5]]   # small/medium/large
    ar_vals = [stats[9], stats[10], stats[11]]  # small/medium/large
    labels = ["Small", "Medium", "Large"]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8.8, 7))
    ax.bar(x - w / 2, ap_vals, width=w, label="AP", color=COLORS["main"], alpha=0.9)
    ax.bar(x + w / 2, ar_vals, width=w, label="AR", color=COLORS["aux"], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Size-wise AP/AR", fontweight="bold")
    ax.legend()
    path = output_dir / "size_wise_ap_ar.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_calibration_curve(scores, labels, output_dir, n_bins=10):
    centers, accs, confs, counts, ece = compute_calibration(scores, labels, n_bins=n_bins)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 9.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})

    axes[0].plot([0, 1], [0, 1], "--", color="#555555", label="Perfect Calibration")
    valid = ~np.isnan(accs)
    axes[0].plot(confs[valid], accs[valid], "-o", color=COLORS["main"], lw=2.2, label=f"Model (ECE={ece:.4f})")
    axes[0].set_ylabel("Empirical Accuracy")
    axes[0].set_title("Calibration Curve", fontweight="bold")
    axes[0].set_xlim(0, 1.0)
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc="upper left")

    axes[1].bar(centers, counts, width=1.0 / n_bins * 0.9, color=COLORS["gray"], alpha=0.85)
    axes[1].set_xlabel("Predicted Confidence")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confidence Histogram", fontweight="bold")

    path = output_dir / "calibration_curve_ece.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot detection diagnostics for evaluation")
    parser.add_argument(
        "--gt",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug/val/annotations/instances_val.json",
        help="COCO GT annotation path",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/training_analysis/output/eval_full_pipeline/evaluation_artifacts/val_detections.json",
        help="COCO detection json path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.3,
        help="IoU threshold for TP/FP matching",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.3,
        help="Default confidence threshold for histogram/bar charts",
    )
    args = parser.parse_args()

    setup_style()
    script_dir = Path(__file__).parent
    gt_path = resolve_input_path(args.gt, script_dir)
    pred_path = resolve_input_path(args.pred, script_dir)
    if Path(args.output_dir).is_absolute():
        output_dir = Path(args.output_dir)
    else:
        output_dir = (script_dir.parent / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not gt_path.exists():
        print(f"GT not found: {gt_path}")
        return
    if not pred_path.exists():
        print(f"Prediction file not found: {pred_path}")
        return

    gt_data, pred_data = load_data(gt_path, pred_path)
    plot_pr_curve(gt_data, pred_data, output_dir, args.iou_thresh)
    plot_f1_threshold_curve(gt_data, pred_data, output_dir, args.iou_thresh)
    plot_confidence_hist(gt_data, pred_data, output_dir, args.iou_thresh, args.score_thresh)
    plot_per_class_recall(gt_data, pred_data, output_dir, args.iou_thresh, args.score_thresh)
    plot_per_class_ap_bar(gt_data, pred_data, output_dir, args.iou_thresh)
    plot_error_breakdown(gt_data, pred_data, output_dir, args.iou_thresh)
    # 以下两项依赖更严格，单独容错，避免影响其它图生成
    try:
        plot_size_wise_ap_ar(gt_path, pred_path, output_dir)
    except Exception as exc:
        print(f"Skip size_wise_ap_ar due to unexpected error: {exc}")
    try:
        scores, is_tp, _ = match_predictions(gt_data, pred_data, iou_thresh=args.iou_thresh)
        if len(scores) > 0:
            plot_calibration_curve(scores, is_tp, output_dir, n_bins=10)
    except Exception as exc:
        print(f"Skip calibration_curve_ece due to unexpected error: {exc}")
    print("Diagnostics plots generated.")


if __name__ == "__main__":
    main()
