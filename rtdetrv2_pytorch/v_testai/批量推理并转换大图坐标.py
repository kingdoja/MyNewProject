"""
批量推理小图并将坐标转换为大图坐标
用于处理从大图切片的多个小图的推理结果
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
import sys
import os
import csv
import json
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '.')
from src.core import YAMLConfig


class PatchInference:
    """小图批量推理并转换为大图坐标"""
    
    def __init__(self, config_path, model_path, device='cpu'):
        """
        初始化推理器
        
        Args:
            config_path: 配置文件路径
            model_path: 模型权重路径
            device: 设备 ('cpu' 或 'cuda')
        """
        print("=== 初始化推理器 ===")
        
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
        """
        预测单个小图
        
        Args:
            image_path: 图像路径
            confidence_threshold: 置信度阈值
            
        Returns:
            dict: 包含labels, boxes, scores的字典
        """
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
            'boxes': boxes,  # 格式: [x1, y1, x2, y2]，坐标相对于小图
            'scores': scores
        }
    
    def batch_inference_with_coordinates(
        self, 
        patches_dir, 
        coordinates_csv, 
        output_json,
        confidence_threshold=0.5,
        class_names=None
    ):
        """
        批量推理并转换坐标到大图
        
        Args:
            patches_dir: 小图目录
            coordinates_csv: 坐标CSV文件（由切片脚本生成）
            output_json: 输出JSON文件路径
            confidence_threshold: 置信度阈值
            class_names: 类别名称列表
        """
        print("=== 批量推理开始 ===")
        
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
        all_detections = []  # 存储所有检测结果（大图坐标）
        
        for image_file in tqdm(image_files, desc="推理进度"):
            filename = image_file.name
            
            # 检查是否有坐标信息
            if filename not in patch_coords:
                print(f"警告: {filename} 没有坐标信息，跳过")
                continue
            
            coord = patch_coords[filename]
            
            try:
                # 推理
                result = self.predict_single_patch(str(image_file), confidence_threshold)
                
                # 转换坐标到大图
                for label, box, score in zip(result['labels'], result['boxes'], result['scores']):
                    x1, y1, x2, y2 = box.tolist()
                    
                    # 坐标转换: 小图坐标 + 切片起始位置 = 大图坐标
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
                        # 小图坐标
                        'patch_bbox': {
                            'x1': float(x1),
                            'y1': float(y1),
                            'x2': float(x2),
                            'y2': float(y2)
                        },
                        # 大图坐标（最终结果）
                        'global_bbox': {
                            'x1': float(global_x1),
                            'y1': float(global_y1),
                            'x2': float(global_x2),
                            'y2': float(global_y2),
                            'width': float(global_x2 - global_x1),
                            'height': float(global_y2 - global_y1)
                        },
                        # 切片位置信息
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
        
        # 保存结果
        print(f"\n保存结果到: {output_json}")
        output_data = {
            'total_patches': len(image_files),
            'total_detections': len(all_detections),
            'confidence_threshold': confidence_threshold,
            'detections': all_detections
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 完成！")
        print(f"  - 处理了 {len(image_files)} 个切片")
        print(f"  - 检测到 {len(all_detections)} 个目标")
        print(f"  - 结果已保存到: {output_json}")
        
        return all_detections


def visualize_detections_on_large_image(
    large_image_path,
    detections_json,
    output_image_path,
    downsample_factor=10
):
    """
    在大图上可视化检测结果
    
    Args:
        large_image_path: 大图路径（如果文件太大，建议使用缩略图）
        detections_json: 检测结果JSON文件
        output_image_path: 输出图像路径
        downsample_factor: 下采样因子（用于大图可视化）
    """
    print("=== 可视化大图检测结果 ===")
    
    # 读取检测结果
    with open(detections_json, 'r') as f:
        data = json.load(f)
    detections = data['detections']
    
    print(f"加载了 {len(detections)} 个检测结果")
    
    # 如果有大图，加载并可视化
    if large_image_path and os.path.exists(large_image_path):
        try:
            # 对于非常大的图像，使用OpenSlide或PIL的缩略图
            from PIL import Image
            image = Image.open(large_image_path).convert('RGB')
            
            # 下采样
            if downsample_factor > 1:
                new_size = (image.size[0] // downsample_factor, 
                           image.size[1] // downsample_factor)
                image = image.resize(new_size, Image.LANCZOS)
            
            draw = ImageDraw.Draw(image)
            
            # 绘制检测框
            for det in detections:
                bbox = det['global_bbox']
                x1 = bbox['x1'] / downsample_factor
                y1 = bbox['y1'] / downsample_factor
                x2 = bbox['x2'] / downsample_factor
                y2 = bbox['y2'] / downsample_factor
                
                # 绘制边界框
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                
                # 绘制标签
                label_text = f"{det['class_name']} {det['confidence']:.2f}"
                draw.text((x1, y1-10), label_text, fill='red')
            
            # 保存
            image.save(output_image_path)
            print(f"✓ 可视化结果已保存到: {output_image_path}")
            
        except Exception as e:
            print(f"可视化时出错: {e}")
    else:
        print("未提供大图路径或文件不存在，跳过可视化")


def create_coco_format_output(detections_json, output_coco_json, image_width, image_height):
    """
    将检测结果转换为COCO格式
    
    Args:
        detections_json: 检测结果JSON文件
        output_coco_json: 输出COCO格式JSON文件
        image_width: 大图宽度
        image_height: 大图高度
    """
    print("=== 转换为COCO格式 ===")
    
    with open(detections_json, 'r') as f:
        data = json.load(f)
    detections = data['detections']
    
    # 创建COCO格式
    coco_output = {
        'images': [
            {
                'id': 1,
                'file_name': 'large_image',
                'width': image_width,
                'height': image_height
            }
        ],
        'annotations': [],
        'categories': []
    }
    
    # 收集类别
    class_ids = set()
    for det in detections:
        class_ids.add(det['label'])
    
    for class_id in sorted(class_ids):
        # 从第一个检测中获取类名
        class_name = next((d['class_name'] for d in detections if d['label'] == class_id), f'class_{class_id}')
        coco_output['categories'].append({
            'id': class_id,
            'name': class_name
        })
    
    # 添加annotations
    for idx, det in enumerate(detections):
        bbox = det['global_bbox']
        coco_output['annotations'].append({
            'id': idx + 1,
            'image_id': 1,
            'category_id': det['label'],
            'bbox': [bbox['x1'], bbox['y1'], bbox['width'], bbox['height']],  # COCO格式: [x, y, width, height]
            'area': bbox['width'] * bbox['height'],
            'iscrowd': 0,
            'score': det['confidence']
        })
    
    # 保存
    with open(output_coco_json, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    print(f"✓ COCO格式已保存到: {output_coco_json}")


# ==================== 示例使用 ====================

if __name__ == "__main__":
    
    # ===== 配置参数 =====
    
    # 模型配置
    config_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_cancer_detection.yml"
    model_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection/best.pth"
    
    # 数据路径
    patches_directory = "patches1111"  # 小图目录
    coordinates_csv_file = "patches1111/patch_coordinates.csv"  # 坐标CSV文件
    
    # 输出路径
    output_json_file = "detections_global_coords.json"  # 检测结果（大图坐标）
    output_coco_json_file = "detections_coco_format.json"  # COCO格式
    
    # 类别名称（根据您的数据集修改）
    class_names = ['AD', 'BC', 'EC', 'L', 'LC', 'M', 'NT', 'SM', 'SQ', 'TC1', 'TC2', 'TC3']
    
    # 推理参数
    confidence_threshold = 0.5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 大图信息（用于COCO格式转换）
    large_image_width = 100000  # 替换为实际大图宽度
    large_image_height = 80000  # 替换为实际大图高度
    
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
    inferencer = PatchInference(config_file, model_file, device=device)
    
    # 批量推理
    detections = inferencer.batch_inference_with_coordinates(
        patches_dir=patches_directory,
        coordinates_csv=coordinates_csv_file,
        output_json=output_json_file,
        confidence_threshold=confidence_threshold,
        class_names=class_names
    )
    
    # 转换为COCO格式（可选）
    create_coco_format_output(
        detections_json=output_json_file,
        output_coco_json=output_coco_json_file,
        image_width=large_image_width,
        image_height=large_image_height
    )
    
    print("\n🎉 全部完成！")
    print("\n生成的文件:")
    print(f"  1. {output_json_file} - 详细检测结果（大图坐标）")
    print(f"  2. {output_coco_json_file} - COCO格式检测结果")
    
    print("\n下一步:")
    print("  - 可以使用检测结果进行后续分析")
    print("  - 可以在大图上可视化检测框")
    print("  - 可以进行NMS去除重复检测")

