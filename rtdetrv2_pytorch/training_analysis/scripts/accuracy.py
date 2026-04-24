import json
import argparse
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目路径（scripts/ 上一级是 training_analysis/，再上一级才是 rtdetrv2_pytorch/）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from src.core import YAMLConfig

DEFAULT_CLASSES = ["AD", "BC", "EC", "L", "LC", "M", "NT", "SM", "SQ", "TC1", "TC2", "TC3"]


def resolve_input_path(path_str, script_dir):
    """解析输入路径，支持绝对路径及多种相对路径基准。"""
    input_path = Path(path_str)
    if input_path.is_absolute():
        return input_path

    for base in (script_dir, script_dir.parent, script_dir.parent.parent):
        candidate = (base / input_path).resolve()
        if candidate.exists():
            return candidate

    # 最后回退到当前工作目录
    return input_path.resolve()

def load_class_names_from_data2000(data_root, config_path=None):
    """
    从data2000数据集中加载类别名称
    
    Args:
        data_root: data2000数据集根目录
        
    Returns:
        list: 类别名称列表
    """
    # 尝试从classes.txt文件加载
    try:
        classes_txt_path = os.path.join(data_root, 'classes.txt')
        if os.path.exists(classes_txt_path):
            with open(classes_txt_path, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            return classes
    except Exception as e:
        print(f"从classes.txt加载类别时出错: {e}")
    
    # 尝试从配置文件加载类别
    try:
        if config_path and os.path.exists(config_path):
            cfg = YAMLConfig(config_path)
            if hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'categories'):
                categories = cfg.dataset.categories
                if isinstance(categories, list):
                    return categories
                elif isinstance(categories, dict):
                    # 按ID排序
                    sorted_cats = sorted(categories.items(), key=lambda x: x[0])
                    return [name for _, name in sorted_cats]
    except Exception as e:
        print(f"从配置文件加载类别时出错: {e}")
    
    # 默认返回空列表
    return []

def calculate_classification_metrics(
    predictions_file,
    ground_truth_file,
    iou_threshold=0.5,
    data_root="/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug",
    config_path=None,
):
    """
    计算目标检测模型的分类效果指标
    
    Args:
        predictions_file: 模型预测结果文件路径 (COCO格式)
        ground_truth_file: 真实标签文件路径 (COCO格式)
        iou_threshold: IoU匹配阈值
        data_root: data2000数据集根目录
    
    Returns:
        dict: 包含各类别和总体分类指标的字典
    """
    
    # 读取真实标签和预测结果
    with open(ground_truth_file, 'r') as f:
        gt_data = json.load(f)
    
    with open(predictions_file, 'r') as f:
        pred_data = json.load(f)
    
    # 提取类别信息
    if 'categories' in gt_data and gt_data['categories']:
        # 从GT文件中提取类别信息
        class_names = [cat['name'] for cat in gt_data['categories']]
        class_ids = [cat['id'] for cat in gt_data['categories']]
    else:
        # 如果GT文件中没有类别信息，则从data2000数据集中加载
        class_names = load_class_names_from_data2000(data_root, config_path=config_path)
        if class_names:
            class_ids = list(range(len(class_names)))
        else:
            # 如果无法加载类别名称，则使用默认类别
            class_names = DEFAULT_CLASSES
            class_ids = list(range(len(class_names)))
    
    num_classes = len(class_names)
    
    # 创建类别ID到索引的映射
    id_to_index = {id: idx for idx, id in enumerate(class_ids)}
    
    # 提取真实标签
    gt_annotations = gt_data['annotations'] if 'annotations' in gt_data else []
    image_id_to_gt = {}
    for ann in gt_annotations:
        image_id = ann['image_id']
        if image_id not in image_id_to_gt:
            image_id_to_gt[image_id] = []
        image_id_to_gt[image_id].append(ann)
    
    # 提取预测结果并按图片ID分组
    image_id_to_pred = {}
    for ann in pred_data:
        image_id = ann['image_id']
        if image_id not in image_id_to_pred:
            image_id_to_pred[image_id] = []
        image_id_to_pred[image_id].append(ann)
    
    # 初始化统计变量
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    per_class_metrics = {}
    
    # 对每张图片进行处理
    for image_id in image_id_to_gt.keys():
        gt_anns = image_id_to_gt.get(image_id, [])
        pred_anns = image_id_to_pred.get(image_id, [])
        
        # 为每个真实标注找到最佳匹配的预测
        unmatched_preds = pred_anns.copy()
        
        for gt_ann in gt_anns:
            gt_class_id = gt_ann['category_id']
            gt_bbox = gt_ann['bbox']  # [x, y, width, height]
            
            best_iou = 0
            best_pred_idx = -1
            
            # 遍历所有未匹配的预测
            for i, pred_ann in enumerate(unmatched_preds):
                pred_class_id = pred_ann['category_id']
                pred_bbox = pred_ann['bbox']  # [x, y, width, height]
                iou = calculate_iou(gt_bbox, pred_bbox)
                
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pred_idx = i
                    best_pred_class = pred_class_id
            
            # 更新统计信息
            gt_class_idx = id_to_index.get(gt_class_id, -1)
            if gt_class_idx != -1:
                class_total[gt_class_idx] += 1
                
                # 如果找到了匹配且分类正确
                if best_pred_idx != -1:
                    if best_pred_class == gt_class_id:
                        class_correct[gt_class_idx] += 1
                    # 从未匹配列表中移除
                    unmatched_preds.pop(best_pred_idx)
    
    # 计算各类别准确率
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
        else:
            accuracy = 0.0
            
        per_class_metrics[class_names[i]] = {
            'class_id': class_ids[i],
            'correct': class_correct[i],
            'total': class_total[i],
            'accuracy': accuracy
        }
    
    # 计算总体分类准确率
    total_correct = sum(class_correct)
    total_instances = sum(class_total)
    overall_accuracy = total_correct / total_instances if total_instances > 0 else 0.0
    
    # 计算平均分类准确率 (mCA)
    valid_accuracies = [metrics['accuracy'] for metrics in per_class_metrics.values() if metrics['total'] > 0]
    mean_accuracy = np.mean(valid_accuracies) if valid_accuracies else 0.0
    
    return {
        'per_class_metrics': per_class_metrics,
        'overall_accuracy': overall_accuracy,
        'mean_class_accuracy': mean_accuracy,
        'total_correct': total_correct,
        'total_instances': total_instances
    }

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    
    Args:
        box1, box2: 边界框坐标 [x, y, width, height]
    
    Returns:
        float: IoU值
    """
    # 计算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    # 交集面积
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 各自的面积
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    # 并集面积
    union_area = box1_area + box2_area - intersection_area
    
    # IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

def print_classification_report(metrics):
    """
    打印分类效果报告
    
    Args:
        metrics: 由calculate_classification_metrics返回的指标字典
    """
    print("=" * 60)
    print("目标检测模型分类效果评估报告")
    print("=" * 60)
    
    print(f"总体分类准确率: {metrics['overall_accuracy']:.4f} "
          f"({metrics['total_correct']}/{metrics['total_instances']})")
    print(f"平均分类准确率: {metrics['mean_class_accuracy']:.4f}")
    print()
    
    print("各类别分类准确率:")
    print("-" * 60)
    print(f"{'类别名称':<20} {'正确数':<8} {'总数':<8} {'准确率':<10}")
    print("-" * 60)
    
    for class_name, class_metrics in metrics['per_class_metrics'].items():
        print(f"{class_name:<20} {class_metrics['correct']:<8} "
              f"{class_metrics['total']:<8} {class_metrics['accuracy']:<10.4f}")
    
# 示例使用方法前添加写入函数
def append_metrics_to_best_results(metrics, output_file, iou_threshold):
    """
    将分类指标追加写入 best_results.txt
    
    Args:
        metrics: 分类指标字典
        output_file: best_results.txt 路径
        iou_threshold: IoU 匹配阈值
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    header = (
        "\n"
        "============================================================\n"
        f"Classification Metrics (IoU>={iou_threshold:.2f})\n"
        "============================================================\n"
    )
    
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(header)
        f.write(
            f"Overall Accuracy: {metrics['overall_accuracy']:.4f} "
            f"({metrics['total_correct']}/{metrics['total_instances']})\n"
        )
        f.write(f"Mean Class Accuracy: {metrics['mean_class_accuracy']:.4f}\n\n")
        f.write(f"{'Class':<20} {'Correct':<8} {'Total':<8} {'Accuracy':<10}\n")
        f.write("-" * 60 + "\n")
        for class_name, class_metrics in metrics['per_class_metrics'].items():
            f.write(
                f"{class_name:<20} "
                f"{class_metrics['correct']:<8} "
                f"{class_metrics['total']:<8} "
                f"{class_metrics['accuracy']:<10.4f}\n"
            )
        f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="计算目标检测分类准确率，并可追加写入 best_results.txt")
    parser.add_argument(
        "--gt",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug/val/annotations/instances_val.json",
        help="COCO GT 标注文件路径",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/training_analysis/output/validation_visualization_split_dataset_aug_r50_0309/val_detections.json",
        help="模型预测结果（COCO 格式）路径",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug",
        help="数据集根目录，用于加载 classes.txt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="可选：模型配置文件路径（用于兜底加载类别名）",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.3,
        help="匹配 GT 与预测框时的 IoU 阈值",
    )
    parser.add_argument(
        "--best_results_file",
        type=str,
        default="",
        help="可选：将分类指标追加写入该文件（通常为 best_results.txt）",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default="",
        help="可选：保存分类指标 JSON 文件路径",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    gt_path = resolve_input_path(args.gt, script_dir)
    pred_path = resolve_input_path(args.pred, script_dir)
    data_root_path = resolve_input_path(args.data_root, script_dir)
    config_path = resolve_input_path(args.config, script_dir) if args.config else None

    if not gt_path.exists():
        print(f"Error: GT file not found: {gt_path}")
        return
    if not pred_path.exists():
        print(f"Error: prediction file not found: {pred_path}")
        return

    metrics = calculate_classification_metrics(
        str(pred_path),
        str(gt_path),
        iou_threshold=args.iou_thresh,
        data_root=str(data_root_path),
        config_path=str(config_path) if config_path else None,
    )
    print_classification_report(metrics)

    if args.best_results_file:
        best_results_path = resolve_input_path(args.best_results_file, script_dir)
        append_metrics_to_best_results(metrics, best_results_path, args.iou_thresh)
        print(f"\n分类指标已追加到: {best_results_path}")

    if args.save_json:
        save_json_path = resolve_input_path(args.save_json, script_dir)
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"分类指标 JSON 已保存到: {save_json_path}")


if __name__ == "__main__":
    main()