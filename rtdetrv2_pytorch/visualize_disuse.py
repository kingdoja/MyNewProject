import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '.')
from src.core import YAMLConfig

def visualize_validation(config_path, model_path, output_dir='validation_results', num_samples=10, device='cpu'):
    """可视化validation数据集的检测结果"""
    
    print("=== RT-DETRv2 Validation 可视化 ===")
    
    # 1. 加载配置
    print(f"1. 加载配置文件: {config_path}")
    cfg = YAMLConfig(config_path)
    
    # 2. 加载模型权重
    print(f"2. 加载模型权重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model' in checkpoint:
        cfg.model.load_state_dict(checkpoint['model'])
        print("✓ 成功加载模型权重")
    elif 'ema' in checkpoint and checkpoint['ema'] is not None and 'module' in checkpoint['ema']:
        cfg.model.load_state_dict(checkpoint['ema']['module'])
        print("✓ 成功加载EMA模型权重")
    else:
        # 直接尝试加载整个checkpoint作为state_dict
        try:
            cfg.model.load_state_dict(checkpoint)
            print("✓ 成功直接加载模型权重")
        except:
            print("❌ 未找到模型权重")
            return
    
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
    
    model = InferenceModel().to(device)
    model.eval()
    print("✓ 模型准备完成")
    
    # 4. 创建validation数据加载器
    print("4. 创建validation数据加载器...")
    val_dataloader = cfg.val_dataloader
    print(f"   Validation数据集大小: {len(val_dataloader.dataset)}")
    
    # 5. 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 6. 处理validation样本
    print(f"5. 处理前{num_samples}个validation样本...")
    
    sample_count = 0
    for batch_idx, (samples, targets) in enumerate(val_dataloader):
        if sample_count >= num_samples:
            break
            
        samples = samples.to(device)
        
        # 获取原始图像尺寸
        orig_sizes = [t["orig_size"] for t in targets]
        orig_size_tensor = torch.stack(orig_sizes, dim=0).to(device)
        
        # 执行推理
        with torch.no_grad():
            outputs = model(samples, orig_size_tensor)
        
        # 处理每个样本
        # outputs 应该是 (labels_batch, boxes_batch, scores_batch)
        if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
            labels_batch, boxes_batch, scores_batch = outputs
            batch_size = labels_batch.shape[0]
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                    
                labels = labels_batch[i]
                boxes = boxes_batch[i]
                scores = scores_batch[i]

                # 只保留分数大于阈值的检测结果
                keep = scores > 0.3
                labels = labels[keep]
                boxes = boxes[keep]
                scores = scores[keep]

                # 获取原始图像
                if 'image_path' in targets[i]:
                    image_path = targets[i]['image_path']
                    if os.path.exists(image_path):
                        image_pil = Image.open(image_path).convert('RGB')
                    else:
                        image_tensor = samples[i].cpu()
                        image_pil = tensor_to_pil(image_tensor)
                else:
                    image_tensor = samples[i].cpu()
                    image_pil = tensor_to_pil(image_tensor)

                # 绘制检测结果
                result_image = draw_detections(image_pil, labels, boxes, scores, targets[i])

                # 保存结果
                output_filename = f"validation_sample_{sample_count:03d}.jpg"
                output_filepath = output_path / output_filename
                result_image.save(output_filepath)

                # 保存检测结果到JSON文件
                json_filename = f"validation_sample_{sample_count:03d}.json"
                json_filepath = output_path / json_filename
                save_detection_results(json_filepath, labels, boxes, scores, targets[i])

                sample_count += 1
        else:
            print(f"   跳过batch - 输出格式异常: {type(outputs)}")
            continue

    print(f"✓ 完成！结果已保存到: {output_path}")
    print(f"   图像文件: {output_path}/*.jpg")
    print(f"   检测结果: {output_path}/*.json")

def tensor_to_pil(tensor):
    """将tensor转换为PIL图像"""
    # 反归一化
    if tensor.max() <= 1.0:
        tensor = tensor * 255
    
    # 转换为uint8
    tensor = tensor.clamp(0, 255).to(torch.uint8)
    
    # 转换为PIL图像
    if tensor.dim() == 3:
        # CHW格式，转换为HWC
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
    
    # 转换为numpy并创建PIL图像
    import numpy as np
    image_array = tensor.cpu().numpy()
    return Image.fromarray(image_array)

def draw_detections(image, labels, boxes, scores, target, threshold=0.3):
    """在图像上绘制检测结果和真实标签"""
    draw = ImageDraw.Draw(image)
    
    # 尝试使用默认字体，如果不可用则使用默认字体
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    # 癌症检测数据集类别名称
    # 根据配置文件和数据集调整这些类别名称
    # 注意：RT-DETR通常类别ID从0开始，不包括背景类
    coco_classes = [
        'False',   # 0
        'No',      # 1
        'True'     # 2     
    ]

    # 绘制真实标签（绿色）
    if 'boxes' in target and target['boxes'] is not None:
        gt_boxes = target['boxes']
        gt_labels = target.get('labels', [])
        
        for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            if isinstance(box, torch.Tensor):
                box = box.tolist()
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            # 绘制真实标签框
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
            
            # 绘制标签文本
            class_name = coco_classes[int(label)] if int(label) < len(coco_classes) else f"class_{int(label)}"
            label_text = f"GT: {class_name}"
            
            # 绘制标签背景和文字
            left, top, right, bottom = draw.textbbox((0, 0), label_text, font=font)
            text_width = right - left
            text_height = bottom - top
            
            draw.rectangle([x1, y1-text_height-5, x1+text_width+10, y1], fill='green')
            draw.text((x1+5, y1-text_height-2), label_text, fill='white', font=font)
    
    # 绘制检测结果（红色）
    if labels is not None and boxes is not None and scores is not None:
        # 过滤低置信度的检测结果
        valid_detections = scores > threshold
        filtered_labels = labels[valid_detections]
        filtered_boxes = boxes[valid_detections]
        filtered_scores = scores[valid_detections]
        
        for label, box, score in zip(filtered_labels, filtered_boxes, filtered_scores):
            # 获取类别名称
            class_id = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
            class_name = coco_classes[class_id] if class_id < len(coco_classes) else f"class_{class_id}"
            
            # 获取边界框坐标
            if isinstance(box, torch.Tensor):
                x1, y1, x2, y2 = box.tolist()
            else:
                x1, y1, x2, y2 = box
            
            # 绘制检测框
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # 绘制标签文本
            score_value = score.item() if isinstance(score, torch.Tensor) else score
            label_text = f"Pred: {class_name} {score_value:.2f}"
            
            # 绘制标签背景和文字
            left, top, right, bottom = draw.textbbox((0, 0), label_text, font=font)
            text_width = right - left
            text_height = bottom - top
            
            draw.rectangle([x1, y1-text_height-5, x1+text_width+10, y1], fill='red')
            draw.text((x1+5, y1-text_height-2), label_text, fill='white', font=font)
    
    return image

def save_detection_results(filepath, labels, boxes, scores, target):
    """保存检测结果到JSON文件"""
    results = {
        'predictions': [],
        'ground_truth': {}
    }
    
    # 保存预测结果
    if labels is not None and boxes is not None and scores is not None:
        for label, box, score in zip(labels, boxes, scores):
            if isinstance(label, torch.Tensor):
                label = label.item()
            if isinstance(box, torch.Tensor):
                box = box.tolist()
            if isinstance(score, torch.Tensor):
                score = score.item()
                
            results['predictions'].append({
                'label': int(label),
                'box': box,
                'score': float(score)
            })
    
    # 保存真实标签
    if 'boxes' in target and target['boxes'] is not None:
        gt_boxes = target['boxes']
        gt_labels = target.get('labels', [])
        
        results['ground_truth']['boxes'] = []
        results['ground_truth']['labels'] = []
        
        for box, label in zip(gt_boxes, gt_labels):
            if isinstance(box, torch.Tensor):
                box = box.tolist()
            if isinstance(label, torch.Tensor):
                label = label.item()
                
            results['ground_truth']['boxes'].append(box)
            results['ground_truth']['labels'].append(int(label))
    
    # 保存到JSON文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # 配置参数 - 请修改这些路径
    config_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_cancer_detection.yml"  # 您的配置文件
    model_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/premodel/best.pth" # 您的模型文件
    #model_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection/best.pth"  # 您的模型文件
    output_directory = "validation_visualization1"  # 输出目录
    num_samples_to_visualize = 20  # 要可视化的样本数量
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测设备
    
    # 检查文件是否存在
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        print("请修改脚本中的 config_file 变量")
        exit(1)
    
    if not os.path.exists(model_file):
        print(f"❌ 模型文件不存在: {model_file}")
        print("请修改脚本中的 model_file 变量")
        exit(1)
    
    # 执行validation可视化
    visualize_validation(
        config_path=config_file,
        model_path=model_file,
        output_dir=output_directory,
        num_samples=num_samples_to_visualize,
        device=device
    )