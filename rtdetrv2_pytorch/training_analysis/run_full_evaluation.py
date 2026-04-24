#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键执行模型评估与可视化流程：
1) scripts/visualize_validation.py
2) scripts/extract_best_results.py
3) scripts/accuracy.py
4) scripts/confusion_matrix.py
5) scripts/plot_training_curves.py
6) scripts/plot_detection_diagnostics.py



加了config配置文件后，会优先使用config配置文件中的配置，如果没有配置，则使用命令行参数中的配置。

python /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/training_analysis/run_full_evaluation.py \
  --config_json /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/training_analysis/eval_config.yaml




python /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/training_analysis/run_full_evaluation.py \
  --config /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_cancer_detection1.yml \
  --model /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r50vd_cancer_detection1_incremental_ft_oaug_0309/best.pth \
  --log /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r50vd_cancer_detection1_incremental_ft_0305/log.txt \
  --data_root /home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug \
  --output_name eval_full_pipeline \
  --conf_thresh 0.5 \
  --iou_thresh 0.3
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def resolve_input_path(path_str, script_dir):
    """解析输入路径，支持绝对路径及多种相对路径基准。"""
    input_path = Path(path_str)
    if input_path.is_absolute():
        return input_path

    candidate = (script_dir / input_path).resolve()
    if candidate.exists():
        return candidate

    candidate = (script_dir.parent / input_path).resolve()
    if candidate.exists():
        return candidate

    return input_path.resolve()


def run_step(cmd, title):
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_config_file(config_path):
    """读取 JSON/YAML 配置文件，返回 dict。"""
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "读取 YAML 需要安装 PyYAML：`pip install pyyaml`"
            ) from exc
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        raise ValueError(f"不支持的配置文件类型: {config_path}")

    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError("配置文件内容必须是键值对对象（dict）")
    return cfg


def append_validation_eval_to_best_results(gt_path, pred_path, best_results_path):
    """基于 val_detections.json 与 GT 计算 COCO 指标并追加到 best_results.txt。"""
    if not gt_path.exists():
        print(f"⚠️ 未找到 GT 文件，跳过验证集指标追加: {gt_path}")
        return
    if not pred_path.exists():
        print(f"⚠️ 未找到预测文件，跳过验证集指标追加: {pred_path}")
        return

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:
        print(f"⚠️ 导入 pycocotools 失败，跳过验证集指标追加: {exc}")
        return

    try:
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(pred_path))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
    except Exception as exc:
        print(f"⚠️ 计算验证集 COCO 指标失败，跳过追加: {exc}")
        return

    lines = [
        "",
        "验证集：",
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[0]:.3f}",
        f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats[1]:.3f}",
        f" Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats[2]:.3f}",
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[3]:.3f}",
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[4]:.3f}",
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[5]:.3f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {stats[6]:.3f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {stats[7]:.3f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[8]:.3f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[9]:.3f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[10]:.3f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[11]:.3f}",
        "",
    ]
    best_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_results_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已追加验证集评估结果到: {best_results_path}")


def main():
    parser = argparse.ArgumentParser(description="自动化执行 RT-DETR 评估与可视化全流程")
    parser.add_argument(
        "--config_json",
        type=str,
        default="",
        help="配置文件路径（支持 .json/.yaml/.yml）；命令行参数优先级更高",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="模型配置文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型权重路径",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="训练日志路径（用于提取最佳结果和绘制曲线）",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="数据集根目录（包含 val/test）",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="输出目录名（最终保存到 training_analysis/output/<output_name>）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推理设备：auto/cpu/cuda/cuda:0 等",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=None,
        help="可视化预测置信度阈值",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=None,
        help="accuracy/confusion_matrix 的 IoU 匹配阈值",
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="仅处理验证集，跳过测试集可视化",
    )
    parser.add_argument(
        "--python_bin",
        type=str,
        default=None,
        help="执行子脚本使用的 Python 解释器",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    default_config = {
        "config": "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_cancer_detection1.yml",
        "model": "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r50vd_cancer_detection1_incremental_ft_oaug_0309/best.pth",
        "log": "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r50vd_cancer_detection1_incremental_ft_0305/log.txt",
        "data_root": "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug",
        "output_name": "eval_full_pipeline",
        "device": "auto",
        "conf_thresh": 0.5,
        "iou_thresh": 0.3,
        "python_bin": sys.executable,
        "skip_test": False,
    }

    merged = default_config.copy()
    if args.config_json:
        cfg_path = resolve_input_path(args.config_json, script_dir)
        if not cfg_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
        merged.update(load_config_file(cfg_path))

    cli_values = {
        "config": args.config,
        "model": args.model,
        "log": args.log,
        "data_root": args.data_root,
        "output_name": args.output_name,
        "device": args.device,
        "conf_thresh": args.conf_thresh,
        "iou_thresh": args.iou_thresh,
        "python_bin": args.python_bin,
    }
    for key, value in cli_values.items():
        if value is not None:
            merged[key] = value
    if args.skip_test:
        merged["skip_test"] = True

    python_bin = merged["python_bin"]
    config_path = resolve_input_path(str(merged["config"]), script_dir)
    model_path = resolve_input_path(str(merged["model"]), script_dir)
    log_path = resolve_input_path(str(merged["log"]), script_dir)
    data_root = resolve_input_path(str(merged["data_root"]), script_dir)

    output_dir = (script_dir / "output" / str(merged["output_name"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 将“可视化+JSON”结果统一收拢到单独子目录，避免与曲线/摘要文件混放
    artifacts_dir = (output_dir / "evaluation_artifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    gt_path = (data_root / "val" / "annotations" / "instances_val.json").resolve()
    pred_path = (artifacts_dir / "val_detections.json").resolve()
    best_results_path = (output_dir / "best_results.txt").resolve()
    cls_metrics_json = (artifacts_dir / "classification_metrics.json").resolve()

    visualize_script = script_dir / "scripts" / "visualize_validation.py"
    scripts_dir = script_dir / "scripts"
    extract_script = scripts_dir / "extract_best_results.py"
    accuracy_script = scripts_dir / "accuracy.py"
    confusion_script = scripts_dir / "confusion_matrix.py"
    curves_script = scripts_dir / "plot_training_curves.py"
    diagnostics_script = scripts_dir / "plot_detection_diagnostics.py"

    cmd_visualize = [
        python_bin,
        str(visualize_script),
        "--config",
        str(config_path),
        "--model",
        str(model_path),
        "--data_root",
        str(data_root),
        "--output_dir",
        str(artifacts_dir),
        "--device",
        str(merged["device"]),
        "--confidence_threshold",
        str(merged["conf_thresh"]),
    ]
    if bool(merged.get("skip_test", False)):
        cmd_visualize.append("--skip_test")

    cmd_extract = [
        python_bin,
        str(extract_script),
        "--log",
        str(log_path),
        "--output_dir",
        str(output_dir),
    ]

    cmd_accuracy = [
        python_bin,
        str(accuracy_script),
        "--gt",
        str(gt_path),
        "--pred",
        str(pred_path),
        "--data_root",
        str(data_root),
        "--config",
        str(config_path),
        "--iou_thresh",
        str(merged["iou_thresh"]),
        "--best_results_file",
        str(best_results_path),
        "--save_json",
        str(cls_metrics_json),
    ]

    cmd_confusion = [
        python_bin,
        str(confusion_script),
        "--gt",
        str(gt_path),
        "--pred",
        str(pred_path),
        "--data_root",
        str(data_root),
        "--iou_thresh",
        str(merged["iou_thresh"]),
        "--output_dir",
        str(output_dir),
    ]

    cmd_curves = [
        python_bin,
        str(curves_script),
        "--log",
        str(log_path),
        "--output_dir",
        str(output_dir),
    ]

    cmd_diagnostics = [
        python_bin,
        str(diagnostics_script),
        "--gt",
        str(gt_path),
        "--pred",
        str(pred_path),
        "--output_dir",
        str(output_dir),
        "--iou_thresh",
        str(merged["iou_thresh"]),
        "--score_thresh",
        str(merged["conf_thresh"]),
    ]

    run_step(cmd_visualize, "Step 1/6 Visualize & Evaluate")
    run_step(cmd_extract, "Step 2/6 Extract Best Results")
    append_validation_eval_to_best_results(gt_path, pred_path, best_results_path)
    run_step(cmd_accuracy, "Step 3/6 Classification Accuracy")
    run_step(cmd_confusion, "Step 4/6 Confusion Matrix")
    run_step(cmd_curves, "Step 5/6 Training Curves")
    run_step(cmd_diagnostics, "Step 6/6 Detection Diagnostics")

    print("\n流程执行完成，结果目录：", output_dir)
    print("可视化+JSON 子目录：", artifacts_dir)
    print("主要输出：")
    print(" -", best_results_path)
    print(" -", pred_path)
    print(" -", cls_metrics_json)
    print(" -", artifacts_dir / "val_ground_truth")
    print(" -", artifacts_dir / "val_prediction")
    print(" -", artifacts_dir / "test_ground_truth")
    print(" -", artifacts_dir / "test_prediction")
    print(" -", artifacts_dir / "test_detections.json")
    print(" -", artifacts_dir / "val_detections.json")
    print(" -", output_dir / "confusion_matrix.png")
    print(" -", output_dir / "loss_curves.png")
    print(" -", output_dir / "train_val_loss_curves.png")
    print(" -", output_dir / "map_curves.png")
    print(" -", output_dir / "ar_curves.png")
    print(" -", output_dir / "paper_dashboard.png")
    print(" -", output_dir / "pr_curve.png")
    print(" -", output_dir / "f1_threshold_curve.png")
    print(" -", output_dir / "confidence_distribution.png")
    print(" -", output_dir / "per_class_recall.png")
    print(" -", output_dir / "per_class_ap_bar.png")
    print(" -", output_dir / "error_breakdown.png")
    print(" -", output_dir / "size_wise_ap_ar.png")
    print(" -", output_dir / "calibration_curve_ece.png")


if __name__ == "__main__":
    main()
