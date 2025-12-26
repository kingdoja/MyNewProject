"""
高级批量推理：支持滑动窗口、NMS去重、边界目标处理
用于处理大图切片的完整解决方案
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import csv
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.insert(0, '.')
from src.core import YAMLConfig


def box_iou(box1, box2):
    """
    计算两个边界框的IoU
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        float: IoU值
    """
    # 计算交集
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    if x2_min < x1_max or y2_min < y1_max:
        return 0.0
    
    intersection = (x2_min - x1_max) * (y2_min - y1_max)
    
    # 计算并集
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def nms_detections(detections, iou_threshold=0.5):
    """
    对检测结果执行NMS (Non-Maximum Suppression)
    
    Args:
        detections: 检测结果列表
        iou_threshold: IoU阈值
    
    Returns:
        list: 去重后的检测结果
    """
    if len(detections) == 0:
        return []
    
    # 按置信度降序排序
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    
    for i, det in enumerate(detections):
        # 检查是否与已保留的检测框重叠
        should_keep = True
        
        for kept_det in keep:
            # 只对同一类别进行NMS
            if det['label'] != kept_det['label']:
                continue
            
            # 计算IoU
            box1 = [det['global_bbox']['x1'], det['global_bbox']['y1'], 
                   det['global_bbox']['x2'], det['global_bbox']['y2']]
            box2 = [kept_det['global_bbox']['x1'], kept_det['global_bbox']['y1'], 
                   kept_det['global_bbox']['x2'], kept_det['global_bbox']['y2']]
            
            iou = box_iou(box1, box2)
            
            if iou > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            keep.append(det)
    
    print(f"NMS: {len(detections)} -> {len(keep)} 检测框")
    return keep


def merge_overlapping_detections(detections, iou_threshold=0.7):
    """
    合并高度重叠的检测框（通常来自相邻切片）
    
    Args:
        detections: 检测结果列表
        iou_threshold: IoU阈值，高于此值的框会被合并
    
    Returns:
        list: 合并后的检测结果
    """
    if len(detections) == 0:
        return []
    
    # 按类别分组
    detections_by_class = {}
    for det in detections:
        label = det['label']
        if label not in detections_by_class:
            detections_by_class[label] = []
        detections_by_class[label].append(det)
    
    merged = []
    
    for label, dets in detections_by_class.items():
        # 按置信度排序
        dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)
        
        used = [False] * len(dets)
        
        for i, det1 in enumerate(dets):
            if used[i]:
                continue
            
            # 找到所有与det1高度重叠的框
            group = [det1]
            
            for j in range(i + 1, len(dets)):
                if used[j]:
                    continue
                
                det2 = dets[j]
                
                box1 = [det1['global_bbox']['x1'], det1['global_bbox']['y1'], 
                       det1['global_bbox']['x2'], det1['global_bbox']['y2']]
                box2 = [det2['global_bbox']['x1'], det2['global_bbox']['y1'], 
                       det2['global_bbox']['x2'], det2['global_bbox']['y2']]
                
                iou = box_iou(box1, box2)
                
                if iou > iou_threshold:
                    group.append(det2)
                    used[j] = True
            
            # 合并group中的框
            if len(group) == 1:
                merged.append(det1)
            else:
                # 使用加权平均合并
                total_conf = sum(d['confidence'] for d in group)
                
                merged_bbox = {
                    'x1': sum(d['global_bbox']['x1'] * d['confidence'] for d in group) / total_conf,
                    'y1': sum(d['global_bbox']['y1'] * d['confidence'] for d in group) / total_conf,
                    'x2': sum(d['global_bbox']['x2'] * d['confidence'] for d in group) / total_conf,
                    'y2': sum(d['global_bbox']['y2'] * d['confidence'] for d in group) / total_conf,
                }
                merged_bbox['width'] = merged_bbox['x2'] - merged_bbox['x1']
                merged_bbox['height'] = merged_bbox['y2'] - merged_bbox['y1']
                
                merged_det = {
                    'label': label,
                    'class_name': det1['class_name'],
                    'confidence': max(d['confidence'] for d in group),
                    'global_bbox': merged_bbox,
                    'merged_from': len(group)
                }
                
                merged.append(merged_det)
    
    print(f"合并: {len(detections)} -> {len(merged)} 检测框")
    return merged


class AdvancedPatchInference:
    """高级小图批量推理器"""
    
    def __init__(self, config_path, model_path, device='cpu'):
        """初始化推理器"""
        print("=== 初始化高级推理器 ===")
        
        # 加载配置
        print(f"加载配置: {config_path}")
        self.cfg = YAMLConfig(config_path)
        
        # 加载模型
        print(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model' in checkpoint:
            self.cfg.model.load_state_dict(checkpoint['model'])
        elif 'ema' in checkpoint and 'module' in checkpoint['ema']:
            self.cfg.model.load_state_dict(checkpoint['ema']['module'])
        else:
            raise ValueError("无法从checkpoint中加载模型权重")
        
        # 创建推理模型
        print("创建推理模型...")
        class InferenceModel(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()
                
            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs
        
        self.model = InferenceModel(self.cfg).to(device)
        self.model.eval()
        self.device = device
        
        # 图像预处理
        self.transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        
        print("✓ 推理器初始化完成")
    
    def predict_single_patch(self, image_path, confidence_threshold=0.5):
        """预测单个小图"""
        # 加载图像
        image_pil = Image.open(image_path).convert('RGB')
        original_size = image_pil.size
        
        # 预处理
        image_tensor = self.transforms(image_pil).unsqueeze(0).to(self.device)
        orig_size_tensor = torch.tensor([[original_size[0], original_size[1]]], 
                                        dtype=torch.int64, device=self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(image_tensor, orig_size_tensor)
        
        # 解析输出
        if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
            labels, boxes, scores = outputs
            if isinstance(labels, torch.Tensor) and labels.dim() > 1:
                labels = labels[0]
            if isinstance(boxes, torch.Tensor) and boxes.dim() > 2:
                boxes = boxes[0]
            if isinstance(scores, torch.Tensor) and scores.dim() > 1:
                scores = scores[0]
        else:
            raise ValueError(f"输出格式异常: {type(outputs)}")
        
        # 过滤低置信度结果
        keep = scores > confidence_threshold
        labels = labels[keep].cpu()
        boxes = boxes[keep].cpu()
        scores = scores[keep].cpu()
        
        return {
            'labels': labels,
            'boxes': boxes,
            'scores': scores
        }
    
    def batch_inference_with_nms(
        self, 
        patches_dir, 
        coordinates_csv, 
        output_json,
        confidence_threshold=0.5,
        nms_iou_threshold=0.5,
        merge_iou_threshold=0.7,
        class_names=None,
        enable_nms=True,
        enable_merge=True
    ):
        """
        批量推理并使用NMS去重
        
        Args:
            patches_dir: 小图目录
            coordinates_csv: 坐标CSV文件
            output_json: 输出JSON文件
            confidence_threshold: 置信度阈值
            nms_iou_threshold: NMS的IoU阈值
            merge_iou_threshold: 合并的IoU阈值
            class_names: 类别名称列表
            enable_nms: 是否启用NMS
            enable_merge: 是否启用合并
        """
        print("=== 高级批量推理开始 ===")
        print(f"置信度阈值: {confidence_threshold}")
        print(f"NMS IoU阈值: {nms_iou_threshold}")
        print(f"合并 IoU阈值: {merge_iou_threshold}")
        
        # 读取坐标信息
        print(f"读取坐标信息: {coordinates_csv}")
        patch_coords = {}
        with open(coordinates_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patch_coords[row['filename']] = {
                    'patch_id': int(row['patch_id']),
                    'x_start': int(row['x_start']),
                    'y_start': int(row['y_start']),
                    'x_end': int(row['x_end']),
                    'y_end': int(row['y_end']),
                    'width': int(row['width']),
                    'height': int(row['height'])
                }
        
        print(f"找到 {len(patch_coords)} 个切片的坐标信息")
        
        # 获取所有图像文件
        patches_dir = Path(patches_dir)
        image_files = sorted(list(patches_dir.glob('*.png')) + 
                           list(patches_dir.glob('*.jpg')))
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 批量推理
        all_detections = []
        
        for image_file in tqdm(image_files, desc="推理进度"):
            filename = image_file.name
            
            if filename not in patch_coords:
                continue
            
            coord = patch_coords[filename]
            
            try:
                # 推理
                result = self.predict_single_patch(str(image_file), confidence_threshold)
                
                # 转换坐标到大图
                for label, box, score in zip(result['labels'], result['boxes'], result['scores']):
                    x1, y1, x2, y2 = box.tolist()
                    
                    # 坐标转换
                    global_x1 = x1 + coord['x_start']
                    global_y1 = y1 + coord['y_start']
                    global_x2 = x2 + coord['x_start']
                    global_y2 = y2 + coord['y_start']
                    
                    detection = {
                        'patch_id': coord['patch_id'],
                        'patch_filename': filename,
                        'label': int(label.item()),
                        'class_name': class_names[int(label.item())] if class_names else f"class_{int(label.item())}",
                        'confidence': float(score.item()),
                        'patch_bbox': {
                            'x1': float(x1),
                            'y1': float(y1),
                            'x2': float(x2),
                            'y2': float(y2)
                        },
                        'global_bbox': {
                            'x1': float(global_x1),
                            'y1': float(global_y1),
                            'x2': float(global_x2),
                            'y2': float(global_y2),
                            'width': float(global_x2 - global_x1),
                            'height': float(global_y2 - global_y1)
                        },
                        'patch_position': {
                            'x_start': coord['x_start'],
                            'y_start': coord['y_start'],
                            'x_end': coord['x_end'],
                            'y_end': coord['y_end']
                        }
                    }
                    
                    all_detections.append(detection)
                    
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")
                continue
        
        print(f"\n原始检测数: {len(all_detections)}")
        
        # NMS去重
        if enable_nms and len(all_detections) > 0:
            print("\n执行NMS去重...")
            all_detections = nms_detections(all_detections, nms_iou_threshold)
        
        # 合并高度重叠的框
        if enable_merge and len(all_detections) > 0:
            print("\n合并重叠检测框...")
            all_detections = merge_overlapping_detections(all_detections, merge_iou_threshold)
        
        # 统计信息
        stats = self._compute_statistics(all_detections, class_names)
        
        # 保存结果
        print(f"\n保存结果到: {output_json}")
        output_data = {
            'total_patches': len(image_files),
            'total_detections': len(all_detections),
            'confidence_threshold': confidence_threshold,
            'nms_enabled': enable_nms,
            'nms_iou_threshold': nms_iou_threshold if enable_nms else None,
            'merge_enabled': enable_merge,
            'merge_iou_threshold': merge_iou_threshold if enable_merge else None,
            'statistics': stats,
            'detections': all_detections
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 完成！")
        print(f"  - 处理了 {len(image_files)} 个切片")
        print(f"  - 最终检测到 {len(all_detections)} 个目标")
        print(f"  - 结果已保存到: {output_json}")
        
        # 打印统计信息
        print("\n=== 检测统计 ===")
        for class_name, count in stats['detections_per_class'].items():
            print(f"  {class_name}: {count}")
        
        return all_detections
    
    def _compute_statistics(self, detections, class_names):
        """计算统计信息"""
        stats = {
            'detections_per_class': {},
            'avg_confidence_per_class': {},
            'total_detections': len(detections)
        }
        
        # 按类别统计
        class_detections = {}
        for det in detections:
            class_name = det['class_name']
            if class_name not in class_detections:
                class_detections[class_name] = []
            class_detections[class_name].append(det['confidence'])
        
        for class_name, confidences in class_detections.items():
            stats['detections_per_class'][class_name] = len(confidences)
            stats['avg_confidence_per_class'][class_name] = sum(confidences) / len(confidences)
        
        return stats


# ==================== 示例使用 ====================

if __name__ == "__main__":
    
    # ===== 配置参数 =====
    
    # 模型配置
    config_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_cancer_detection.yml"
    model_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection/best.pth"
    
    # 数据路径
    patches_directory = "patches1111"
    coordinates_csv_file = "patches1111/patch_coordinates.csv"
    
    # 输出路径
    output_json_file = "detections_with_nms.json"
    
    # 类别名称
    class_names = ['AD', 'BC', 'EC', 'L', 'LC', 'M', 'NT', 'SM', 'SQ', 'TC1', 'TC2', 'TC3']
    
    # 推理参数
    confidence_threshold = 0.5
    nms_iou_threshold = 0.5  # NMS的IoU阈值
    merge_iou_threshold = 0.7  # 合并的IoU阈值
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ===== 执行推理 =====
    
    # 检查文件
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        exit(1)
    
    if not os.path.exists(model_file):
        print(f"❌ 模型文件不存在: {model_file}")
        exit(1)
    
    if not os.path.exists(patches_directory):
        print(f"❌ 小图目录不存在: {patches_directory}")
        exit(1)
    
    if not os.path.exists(coordinates_csv_file):
        print(f"❌ 坐标CSV文件不存在: {coordinates_csv_file}")
        exit(1)
    
    # 创建推理器
    inferencer = AdvancedPatchInference(config_file, model_file, device=device)
    
    # 批量推理（带NMS和合并）
    detections = inferencer.batch_inference_with_nms(
        patches_dir=patches_directory,
        coordinates_csv=coordinates_csv_file,
        output_json=output_json_file,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
        merge_iou_threshold=merge_iou_threshold,
        class_names=class_names,
        enable_nms=True,
        enable_merge=True
    )
    
    print("\n🎉 全部完成！")

