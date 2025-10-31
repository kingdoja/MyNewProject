import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

from src.core import YAMLConfig

def predict_image(config_path, model_path, image_path, output_path=None, device='cpu', confidence_threshold=0.5):
    """
    使用训练好的模型预测图片
    
    Args:
        config_path: 配置文件路径
        model_path: 模型权重文件路径
        image_path: 输入图片路径
        output_path: 输出图片路径（可选）
        device: 设备选择 ('cpu' 或 'cuda:0')
        confidence_threshold: 置信度阈值
    """
    
    print("=== RT-DETR v2 图片预测 ===")
    
    try:
        # 1. 加载配置
        print(f"1. 加载配置文件: {config_path}")
        cfg = YAMLConfig(config_path)
        
        # 2. 加载模型权重
        print(f"2. 加载模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model' in checkpoint:
            cfg.model.load_state_dict(checkpoint['model'])
            print("✓ 成功加载模型权重")
        elif 'ema' in checkpoint and 'module' in checkpoint['ema']:
            cfg.model.load_state_dict(checkpoint['ema']['module'])
            print("✓ 成功加载EMA模型权重")
        else:
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
        
        # 4. 加载和预处理图片
        print(f"4. 加载图片: {image_path}")
        image_pil = Image.open(image_path).convert('RGB')
        original_size = image_pil.size
        print(f"   原始图片尺寸: {original_size[0]} x {original_size[1]}")
        
        # 图片预处理
        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        
        image_tensor = transforms(image_pil).unsqueeze(0).to(device)
        orig_size_tensor = torch.tensor([[original_size[0], original_size[1]]], dtype=torch.int64, device=device)
        
        print(f"   输入tensor尺寸: {image_tensor.shape}")
        
        # 5. 执行推理
        print("5. 执行推理...")
        start_time = torch.cuda.Event(enable_timing=True) if device != 'cpu' else None
        end_time = torch.cuda.Event(enable_timing=True) if device != 'cpu' else None
        
        if start_time:
            start_time.record()
        
        with torch.no_grad():
            outputs = model(image_tensor, orig_size_tensor)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)  # 毫秒
        else:
            inference_time = 0
        
        print(f"✓ 推理完成")
        
        # 6. 处理输出结果
        print("6. 处理检测结果...")
        if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
            labels, boxes, scores = outputs
            if isinstance(labels, torch.Tensor) and labels.dim() > 1:
                labels = labels[0]
            if isinstance(boxes, torch.Tensor) and boxes.dim() > 2:
                boxes = boxes[0]
            if isinstance(scores, torch.Tensor) and scores.dim() > 1:
                scores = scores[0]
        else:
            print(f"❌ 输出格式异常: {type(outputs)}")
            return
        
        # 7. 绘制检测结果
        print("7. 绘制检测结果...")
        result_image = draw_detections(image_pil, labels, boxes, scores, confidence_threshold)
        
        # 8. 保存结果
        if output_path is None:
            output_path = f"prediction_result_{os.path.basename(image_path)}"
        
        result_image.save(output_path)
        print(f"✓ 结果已保存到: {output_path}")
        
        # 9. 显示统计信息
        print(f"\n=== 检测统计 ===")
        if device != 'cpu':
            print(f"推理时间: {inference_time:.2f} ms")
            print(f"FPS: {1000/inference_time:.2f}")
        print(f"检测目标数: {len(labels)}")
        high_conf_count = len(scores[scores > confidence_threshold])
        print(f"高置信度目标数: {high_conf_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ 预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def draw_detections(image, labels, boxes, scores, threshold=0.5):
    """在图片上绘制检测结果"""
    draw = ImageDraw.Draw(image)
    
    # COCO数据集类别名称
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
    
    # 过滤低置信度的检测结果
    valid_detections = scores > threshold
    filtered_labels = labels[valid_detections]
    filtered_boxes = boxes[valid_detections]
    filtered_scores = scores[valid_detections]
    
    print(f"   检测到 {len(filtered_labels)} 个目标 (置信度 > {threshold})")
    
    # 绘制每个检测框
    for i, (label, box, score) in enumerate(zip(filtered_labels, filtered_boxes, filtered_scores)):
        # 获取类别名称
        class_id = int(label.item())
        class_name = coco_classes[class_id] if class_id < len(coco_classes) else f"class_{class_id}"
        
        # 获取边界框坐标
        x1, y1, x2, y2 = box.tolist()
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # 绘制标签文本
        label_text = f"{class_name} {score:.2f}"
        
        # 计算文本尺寸（简单估算）
        text_width = len(label_text) * 8
        text_height = 16
        
        # 绘制标签背景
        draw.rectangle([x1, y1-text_height-5, x1+text_width+10, y1], fill='red')
        
        # 绘制标签文字
        draw.text((x1+5, y1-text_height-2), label_text, fill='white')
        
        print(f"     {i+1}. {class_name}: 置信度 {score:.3f}, 位置 [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    return image

if __name__ == "__main__":
    # 配置参数
    config_file = "configs/rtdetrv2/rtdetrv2_r18vd_6x_coco.yml"  # 修改为您的配置文件
    model_file = "output/your_model.pth"  # 修改为您的模型文件路径
    image_file = "test_image.jpg"  # 修改为您的测试图片路径
    output_file = "prediction_result.jpg"  # 输出文件名
    device = "cpu"  # 或 "cuda:0" 如果有GPU
    confidence_threshold = 0.5  # 置信度阈值
    
    # 检查文件是否存在
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        print("请修改脚本中的 config_file 变量")
        exit(1)
    
    if not os.path.exists(model_file):
        print(f"❌ 模型文件不存在: {model_file}")
        print("请修改脚本中的 model_file 变量")
        exit(1)
    
    if not os.path.exists(image_file):
        print(f"❌ 图片文件不存在: {image_file}")
        print("请修改脚本中的 image_file 变量")
        exit(1)
    
    # 执行预测
    success = predict_image(
        config_path=config_file,
        model_path=model_file,
        image_path=image_file,
        output_path=output_file,
        device=device,
        confidence_threshold=confidence_threshold
    )
    
    if success:
        print("\n🎉 预测完成！")
        print(f"结果图片已保存为: {output_file}")
    else:
        print("\n❌ 预测失败！")
