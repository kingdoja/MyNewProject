from calendar import c
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import glob
from pathlib import Path
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse

# 添加项目路径 - 脚本位于 training_analysis/scripts/，需要回退两级到 rtdetrv2_pytorch/
script_dir = Path(__file__).parent  # training_analysis/scripts/
rtdetrv2_pytorch_dir = script_dir.parent.parent  # rtdetrv2_pytorch/
sys.path.insert(0, str(rtdetrv2_pytorch_dir))

from src.core import YAMLConfig

class BatchVisualizer:
    def __init__(self, config_path, model_path, device='cpu', annotation_file=None):
        """
        初始化批量可视化器
        
        Args:
            config_path: 配置文件路径
            model_path: 模型权重文件路径
            device: 设备选择 ('cpu' 或 'cuda:0')
            annotation_file: 标注文件路径（可选，用于加载类别）
        """
        self.config_path = config_path
        self.model_path = model_path
        self.device = device
        self.model = None
        # 根据您的数据集修改类别名称
        # 修复: 从配置文件或数据集中动态加载类别，而不是硬编码
        self.coco_classes = self._load_classes(annotation_file)
        self._load_model()
    
    def _load_classes(self, annotation_file=None):
        """动态加载类别名称"""
        # 优先从标注文件加载类别
        if annotation_file and os.path.exists(annotation_file):
            try:
                with open(annotation_file, 'r') as f:
                    annotation_data = json.load(f)
                if 'categories' in annotation_data:
                    # 按ID排序
                    sorted_cats = sorted(annotation_data['categories'], key=lambda x: x['id'])
                    classes = [cat['name'] for cat in sorted_cats]
                    print(f"从标注文件加载了 {len(classes)} 个类别: {classes}")
                    return classes
            except Exception as e:
                print(f"从标注文件加载类别时出错: {e}")
        
        # 尝试从配置文件加载类别
        try:
            cfg = YAMLConfig(self.config_path)
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
        
        # 尝试从classes.txt文件加载
        try:
            # 针对 cancer_detection1.yml / split_dataset_aug 数据集
            classes_txt_path = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug/classes.txt"
            if os.path.exists(classes_txt_path):
                with open(classes_txt_path, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
                return classes
        except Exception as e:
            print(f"从classes.txt加载类别时出错: {e}")
        
        # 默认类别列表 - p4_split_aug数据集的类别
        return ["AD","BC","EC","L","LC","M","NT","SM","SQ","TC1","TC2", "TC3"]
    
    def _load_model(self):
        """加载模型"""
        print("=== 加载模型 ===")
        
        # 1. 加载配置
        print(f"1. 加载配置文件: {self.config_path}")
        cfg = YAMLConfig(self.config_path)
        
        # 2. 加载模型权重
        print(f"2. 加载模型权重: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model' in checkpoint:
            cfg.model.load_state_dict(checkpoint['model'])
            print("✓ 成功加载模型权重")
        elif 'ema' in checkpoint and 'module' in checkpoint['ema']:
            cfg.model.load_state_dict(checkpoint['ema']['module'])
            print("✓ 成功加载EMA模型权重")
        else:
            raise ValueError("未找到模型权重")
        
        # 3. 创建推理模型
        print("3. 创建推理模型...")
        class InferenceModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()
                
            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs
        
        self.model = InferenceModel().to(self.device)
        self.model.eval()
        print("✓ 模型准备完成")
    
    def process_single_image(self, image_path, confidence_threshold=0.5):
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            confidence_threshold: 置信度阈值
            
        Returns:
            处理后的图片和检测结果
        """
        # 加载和预处理图片
        image_pil = Image.open(image_path).convert('RGB')
        original_width, original_height = image_pil.size  # PIL: (width, height)
        
        # 图片预处理
        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        
        image_tensor = transforms(image_pil).unsqueeze(0).to(self.device)
        # RT-DETR 的 postprocessor 期望 (height, width)
        orig_size_tensor = torch.tensor(
            [[original_height, original_width]],
            dtype=torch.int64,
            device=self.device,
        )
        
        # 执行推理
        with torch.no_grad():
            outputs = self.model(image_tensor, orig_size_tensor)
        
        # 处理输出结果
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
        
        # 过滤低置信度的检测结果
        valid_detections = scores > confidence_threshold
        filtered_labels = labels[valid_detections]
        filtered_boxes = boxes[valid_detections]
        filtered_scores = scores[valid_detections]
        
        # 准备COCO格式的结果
        coco_results = []
        for label, box, score in zip(filtered_labels, filtered_boxes, filtered_scores):
            x1, y1, x2, y2 = box.tolist()
            coco_results.append({
                'image_id': 0,  # 占位符，实际使用时会替换
                'category_id': int(label.item()),
                'bbox': [x1, y1, x2-x1, y2-y1],
                'score': float(score.item())
            })
        
        return filtered_labels, filtered_boxes, filtered_scores, coco_results
    
    def _draw_detections(self, image, labels, boxes, scores, color='red', offset=0):
        """在图片上绘制检测结果"""
        draw = ImageDraw.Draw(image)
        
        # 使用更大的字体并尝试加粗
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("arialbd.ttf", 16)  # Windows加粗字体
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 16)
                except:
                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()
        
        # 绘制每个检测框
        for i, (label, box, score) in enumerate(zip(labels, boxes, scores)):
            # 获取类别名称
            class_id = int(label.item())
            # 修复：正确映射类别ID到名称
            class_name = self.coco_classes[class_id] if 0 <= class_id < len(self.coco_classes) else f"class_{class_id}"
            
            # 获取边界框坐标
            x1, y1, x2, y2 = box.tolist()
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制标签文本
            label_text = f"{class_name} {score:.2f}"
            
            # 获取文本尺寸
            if hasattr(draw, 'textsize'):
                # Pillow < 8.0.0
                text_width, text_height = draw.textsize(label_text, font=font)
            else:
                # Pillow >= 8.0.0
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            
            # 绘制标签背景（带偏移避免遮挡）
            tag_y = y1 - text_height - 10 + offset
            draw.rectangle([x1, tag_y - 5, x1 + text_width + 20, tag_y + text_height + 5], fill=color)
            
            # 绘制标签文字（带偏移避免遮挡）
            draw.text((x1 + 10, tag_y-5), label_text, fill='white', font=font)
        
        return image
    
    def _draw_ground_truth(self, image, annotations, categories, color='blue'):
        """在图片上绘制真实标签"""
        draw = ImageDraw.Draw(image)
        
        # 使用与_draw_detections相同的字体设置
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("arialbd.ttf", 16)  # Windows加粗字体
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 16)
                except:
                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()
        
        # 创建类别ID到名称的映射
        cat_id_to_name = {}
        if isinstance(categories, list):
            # 如果categories是列表格式
            for cat in categories:
                if isinstance(cat, dict) and 'id' in cat and 'name' in cat:
                    cat_id_to_name[cat['id']] = cat['name']
        elif isinstance(categories, dict):
            # 如果categories是字典格式
            cat_id_to_name = categories
        
        # 绘制每个真实标签框
        for ann in annotations:
            # 处理标准的边界框格式
            if 'bbox' in ann and ann['bbox']:
                # 获取边界框坐标
                x, y, w, h = ann['bbox']
                x1, y1, x2, y2 = x, y, x + w, y + h
            # 处理 segmentation 格式的标注
            elif 'segmentation' in ann and ann['segmentation']:
                if isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                    # 处理多边形格式的 segmentation
                    segmentation = ann['segmentation']
                    if isinstance(segmentation[0], list):
                        # 多个多边形情况
                        all_points = []
                        for poly in segmentation:
                            all_points.extend([(poly[i], poly[i+1]) for i in range(0, len(poly), 2)])
                    else:
                        # 单个多边形情况
                        all_points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
                    
                    if all_points:
                        xs = [p[0] for p in all_points]
                        ys = [p[1] for p in all_points]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        # 使用计算出的边界框
                        x1, y1, x2, y2 = x_min, y_min, x_max, y_max
                    else:
                        continue  # 跳过无效的标注
                elif isinstance(ann['segmentation'], dict) and 'counts' in ann['segmentation']:
                    # 跳过 RLE 格式
                    continue
                else:
                    continue  # 跳过无效格式
            else:
                continue  # 跳过没有 bbox 和有效 segmentation 的标注
            
            # 获取类别名称
            cat_id = ann['category_id']
            # 修复：正确处理类别ID到名称的映射
            if isinstance(cat_id, (list, np.ndarray)) and len(cat_id) > 0:
                cat_id = cat_id[0]  # 如果是数组，取第一个元素
            # 确保cat_id是整数
            cat_id = int(cat_id) if not isinstance(cat_id, int) else cat_id
            class_name = cat_id_to_name.get(cat_id, f"class_{cat_id}")
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制标签文本 - 修改为与预测标签相同的样式
            label_text = f"GT: {class_name}"
            
            # 获取文本尺寸 - 使用与_draw_detections相同的计算方式
            if hasattr(draw, 'textsize'):
                # Pillow < 8.0.0
                text_width, text_height = draw.textsize(label_text, font=font)
            else:
                # Pillow >= 8.0.0
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            
            # 绘制标签背景和文字 - 使用与预测标签相同的样式
            tag_y = y1 - text_height - 10  # 移除额外的偏移量，使其与预测标签对齐
            draw.rectangle([x1, tag_y - 5, x1 + text_width + 20, tag_y + text_height + 5], fill=color)
            
            # 绘制标签文字 - 使用与预测标签相同的偏移
            draw.text((x1 + 10, tag_y-5), label_text, fill='white', font=font)
        
        return image
    
    def batch_process(self, data_dir, output_dir, confidence_threshold=0.5, dataset_type="val", coco_gt=None, filename_to_id=None):
        """
        批量处理数据集中的图片
        
        Args:
            data_dir: 数据集目录 (包含图片的目录)
            output_dir: 输出目录
            confidence_threshold: 置信度阈值
            dataset_type: 数据集类型 ("val" 或 "test")
            coco_gt: COCO格式的真实标签对象（用于验证集）
            filename_to_id: 文件名到图像ID的映射
            
        Returns:
            COCO格式的检测结果
        """
        # 创建输出目录
        pred_output_dir = os.path.join(output_dir, f"{dataset_type}_prediction")
        gt_output_dir = os.path.join(output_dir, f"{dataset_type}_ground_truth")
        os.makedirs(pred_output_dir, exist_ok=True)
        os.makedirs(gt_output_dir, exist_ok=True)
        
        # 获取所有图片文件 - 修复图片搜索路径
        # p4_split_aug数据集：图像在 {split}/images/ 目录下
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        
        # 首先在data_dir本身查找（如果data_dir已经是images目录）
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(data_dir, ext)))
        
        # 如果data_dir是split目录（如val/），则在images子目录中查找
        if not image_files:
            images_subdir = os.path.join(data_dir, 'images')
            if os.path.exists(images_subdir):
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(images_subdir, ext)))
        
        # 如果仍然没有找到，尝试在data_dir的父目录的images子目录查找
        if not image_files:
            parent_images_dir = os.path.join(os.path.dirname(data_dir), 'images')
            if os.path.exists(parent_images_dir):
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(parent_images_dir, ext)))
        
        print(f"找到 {len(image_files)} 张图片")
        
        # COCO格式的结果
        coco_results = []
        
        # 处理每张图片
        for idx, image_file in enumerate(image_files):
            print(f"处理图片 {idx+1}/{len(image_files)}: {os.path.basename(image_file)}")
            
            try:
                # 处理单张图片
                labels, boxes, scores, detections = self.process_single_image(image_file, confidence_threshold)
                
                # 加载原始图片用于绘制
                image_pil = Image.open(image_file).convert('RGB')
                
                # 如果是验证集且有真实标签，绘制真实标签
                if dataset_type == "val" and coco_gt is not None and filename_to_id is not None:
                    image_filename = os.path.basename(image_file)
                    if image_filename in filename_to_id:
                        image_id = filename_to_id[image_filename]
                        # 获取该图像的注释
                        annotations = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
                        
                        # 创建单独的真实标签图像
                        gt_image_pil = image_pil.copy()
                        gt_image_pil = self._draw_ground_truth(gt_image_pil, annotations, coco_gt.dataset['categories'], color='blue')
                        
                        # 保存真实标签图像到专用文件夹
                        gt_output_image_path = os.path.join(gt_output_dir, f"gt_result_{os.path.basename(image_file)}")
                        gt_image_pil.save(gt_output_image_path)
                
                # 创建单独的预测结果图像
                pred_image_pil = image_pil.copy()
                pred_image_pil = self._draw_detections(pred_image_pil, labels, boxes, scores, color='red', offset=0)
                
                # 保存预测结果图片到专用文件夹
                pred_output_image_path = os.path.join(pred_output_dir, f"pred_result_{os.path.basename(image_file)}")
                pred_image_pil.save(pred_output_image_path)
                
                # 更新COCO结果
                image_filename = os.path.basename(image_file)
                image_id = filename_to_id.get(image_filename, idx + 1) if filename_to_id else (idx + 1)
                
                for detection in detections:
                    detection['image_id'] = image_id
                    coco_results.append(detection)
                    
            except Exception as e:
                print(f"处理图片 {image_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 保存COCO格式结果
        coco_results_path = os.path.join(output_dir, f"{dataset_type}_detections.json")
        with open(coco_results_path, 'w') as f:
            json.dump(coco_results, f)
        
        print(f"COCO格式检测结果已保存到: {coco_results_path}")
        print(f"预测结果已保存到: {pred_output_dir}")
        print(f"真实标签已保存到: {gt_output_dir}")
        
        return coco_results

    def evaluate_validation_set(self, data_root, output_dir, confidence_threshold=0.5):
        """
        评估验证集并生成评估报告
        
        Args:
            data_root: 数据集根目录
            output_dir: 输出目录
            confidence_threshold: 置信度阈值
        """
        print("=== 评估验证集 ===")
        
        # 查找验证集的标注文件 - 针对p4_split_aug数据集结构优化
        annotation_files = []
        # 优先查找 p4_split_aug 数据集的标准路径: {data_root}/val/annotations/instances_val.json
        val_annotation_path = os.path.join(data_root, "val", "annotations", "instances_val.json")
        if os.path.exists(val_annotation_path):
            annotation_files.append(val_annotation_path)
        
        # 查找其他可能的标注文件位置
        annotation_files.extend(glob.glob(os.path.join(data_root, "annotations", "instances_val*.json")))
        annotation_files.extend(glob.glob(os.path.join(data_root, "annotations", "val", "*.json")))
        annotation_files.extend(glob.glob(os.path.join(data_root, "val", "annotations", "*.json")))
        annotation_files.extend(glob.glob(os.path.join(data_root, "*val*.json")))
        
        if not annotation_files:
            annotation_files.extend(glob.glob(os.path.join(data_root, '_annotations_*.json')))
            annotation_files.extend(glob.glob(os.path.join(data_root, 'annotations', '*.json')))
            annotation_files.extend(glob.glob(os.path.join(data_root, '*', 'annotations', '*.json')))

        coco_gt = None
        annotation_file = None
        if annotation_files:
            # 优先使用验证集标注文件
            val_files = [f for f in annotation_files if 'val' in f.lower()]
            if val_files:
                annotation_file = val_files[0]
            else:
                annotation_file = annotation_files[0]
                
            print(f"使用标注文件: {annotation_file}")
            try:
                # 读取标注文件并添加必要的info字段
                with open(annotation_file, 'r') as f:
                    annotation_data = json.load(f)
                
                # 如果缺少info字段，则添加一个默认的
                if 'info' not in annotation_data:
                    annotation_data['info'] = {
                        "description": "Dataset for Detection",
                        "url": "",
                        "version": "1.0",
                        "year": 2025,
                        "contributor": "User",
                        "date_created": "2025/01/01"
                    }
                
                # 如果缺少licenses字段，则添加一个默认的
                if 'licenses' not in annotation_data:
                    annotation_data['licenses'] = [{
                        "id": 1,
                        "name": "Default",
                        "url": ""
                    }]
                
                # 处理多边形标注，确保即使没有bbox也能生成bbox
                if 'annotations' in annotation_data:
                    for ann in annotation_data['annotations']:
                        # 如果存在 segmentation 但不存在 bbox，则从 segmentation 计算 bbox
                        if 'segmentation' in ann and 'bbox' not in ann:
                            if isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                                # 处理多边形格式的 segmentation
                                segmentation = ann['segmentation']
                                if isinstance(segmentation[0], list):
                                    # 多个多边形情况
                                    all_points = []
                                    for poly in segmentation:
                                        # 确保多边形点数足够
                                        if len(poly) >= 5:
                                            all_points.extend([(poly[i], poly[i+1]) for i in range(0, len(poly), 2)])
                                else:
                                    # 单个多边形情况
                                    if len(segmentation) >= 5:
                                        all_points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
                                
                                if all_points:
                                    xs = [p[0] for p in all_points]
                                    ys = [p[1] for p in all_points]
                                    x_min, x_max = min(xs), max(xs)
                                    y_min, y_max = min(ys), max(ys)
                                    ann['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
                                    # 如果没有area字段，则计算area
                                    if 'area' not in ann:
                                        ann['area'] = (x_max - x_min) * (y_max - y_min)
                            elif isinstance(ann['segmentation'], dict) and 'counts' in ann['segmentation']:
                                # 处理 RLE 格式 (不进行转换，保留原样)
                                pass
                        # 如果既没有 segmentation 也没有 bbox，则跳过该标注
                        elif 'bbox' not in ann:
                            # 可以选择删除无效标注或保留空bbox
                            pass
                
                # 将修改后的数据写入临时文件
                temp_annotation_file = os.path.join(output_dir, "temp_annotations.json")
                with open(temp_annotation_file, 'w') as f:
                    json.dump(annotation_data, f)
                
                coco_gt = COCO(temp_annotation_file)
                
                # 删除临时文件
                os.remove(temp_annotation_file)
                
            except Exception as e:
                print(f"加载标注文件时出错: {e}")
                import traceback
                traceback.print_exc()
                coco_gt = None
        else:
            print("警告: 未找到验证集的标注文件，无法进行评估")
        
        if coco_gt is None:
            return None
            
        # 创建文件名到图像ID的映射
        filename_to_id = {}
        if 'images' in coco_gt.dataset:
            for img in coco_gt.dataset['images']:
                filename_to_id[img['file_name']] = img['id']
        
        # 批量处理验证集 - 修复路径，图像在 val/images/ 目录下
        val_images_dir = os.path.join(data_root, "val", "images")
        if not os.path.exists(val_images_dir):
            # 如果 images 子目录不存在，尝试使用 val 目录本身
            val_images_dir = os.path.join(data_root, "val")
        
        coco_results = self.batch_process(
            data_dir=val_images_dir,  # 图像目录路径
            output_dir=output_dir,
            confidence_threshold=confidence_threshold,
            dataset_type="val",
            coco_gt=coco_gt,
            filename_to_id=filename_to_id
        )
        
        # 如果有真实标签，进行评估
        if coco_gt is not None and coco_results:
            try:
                # 创建一个临时的JSON文件用于评估
                temp_res_file = os.path.join(output_dir, "temp_detections.json")
                
                # 直接保存检测结果数组（确保是正确的格式）
                with open(temp_res_file, 'w') as f:
                    json.dump(coco_results, f)
                
                # 加载检测结果文件
                coco_dt = coco_gt.loadRes(temp_res_file)
                
                # 执行评估
                cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                
                # 删除临时文件
                os.remove(temp_res_file)
                
                return cocoEval.stats
            except Exception as e:
                print(f"评估过程中出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            if coco_gt is None:
                print("没有真实标签，跳过评估")
            if not coco_results:
                print("没有检测结果，跳过评估")
            
        return None

def resolve_input_path(path_str, script_dir):
    """解析输入路径，支持绝对路径及多种相对路径基准。"""
    input_path = Path(path_str)
    if input_path.is_absolute():
        return input_path

    for base in (script_dir, script_dir.parent, script_dir.parent.parent):
        candidate = (base / input_path).resolve()
        if candidate.exists():
            return candidate

    return input_path.resolve()


def main():
    parser = argparse.ArgumentParser(description="批量可视化并评估 RT-DETR 验证/测试集结果")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_cancer_detection1.yml",
        help="模型配置文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r50vd_cancer_detection1_incremental_ft_oaug_0309/best.pth",
        help="模型权重路径",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug",
        help="数据集根目录（包含 val/test）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="validation_visualization_split_dataset_aug_r50_0309",
        help="输出目录；相对路径默认保存到 training_analysis/output/ 下",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="推理设备：auto/cpu/cuda/cuda:0 等",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="置信度阈值",
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="仅处理验证集，跳过测试集",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    config_file = resolve_input_path(args.config, script_dir)
    model_file = resolve_input_path(args.model, script_dir)
    data_root = resolve_input_path(args.data_root, script_dir)

    if Path(args.output_dir).is_absolute():
        output_dir_path = Path(args.output_dir)
    else:
        output_dir_path = script_dir.parent / "output" / args.output_dir
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir_path.resolve())

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    confidence_threshold = args.confidence_threshold

    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return
    if not model_file.exists():
        print(f"❌ 模型文件不存在: {model_file}")
        return
    if not data_root.exists():
        print(f"❌ 数据集根目录不存在: {data_root}")
        return

    val_annotation_file = data_root / "val" / "annotations" / "instances_val.json"
    annotation_file = str(val_annotation_file) if val_annotation_file.exists() else None

    visualizer = BatchVisualizer(
        config_path=str(config_file),
        model_path=str(model_file),
        device=device,
        annotation_file=annotation_file,
    )

    val_dir = data_root / "val"
    if val_dir.exists():
        print("\n=== 处理验证集 ===")
        eval_results = visualizer.evaluate_validation_set(
            data_root=str(data_root),
            output_dir=output_dir,
            confidence_threshold=confidence_threshold,
        )
        if eval_results is not None:
            print("\n验证集评估结果:")
            print(f"AP@0.50:0.95 = {eval_results[0]:.4f}")
            print(f"AP@0.50 = {eval_results[1]:.4f}")
            print(f"AP@0.75 = {eval_results[2]:.4f}")
    else:
        print(f"❌ 验证集目录不存在: {val_dir}")

    if not args.skip_test:
        test_dir = data_root / "test"
        if test_dir.exists():
            print("\n=== 处理测试集 ===")
            test_images_dir = test_dir / "images"
            if not test_images_dir.exists():
                test_images_dir = test_dir

            visualizer.batch_process(
                data_dir=str(test_images_dir),
                output_dir=output_dir,
                confidence_threshold=confidence_threshold,
                dataset_type="test",
            )
        else:
            print(f"❌ 测试集目录不存在: {test_dir}")

    print(f"\n✅ 所有可视化结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()