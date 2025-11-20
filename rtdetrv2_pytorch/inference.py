import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import glob
import time
import json
from datetime import datetime
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '.')
from src.core import YAMLConfig

def load_model(config_path, model_path, device='cpu'):
    """加载模型（只加载一次，用于批量推理）"""
    
    print("=== 加载 RT-DETR v2 模型 ===")
    
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
        return None
    
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
    print("✓ 模型准备完成\n")
    
    return model

def predict_image(model, image_path, output_path=None, device='cpu', verbose=False, threshold=0.5):
    """使用已加载的模型预测单张图片，返回检测结果"""
    
    # 加载和预处理图片
    try:
        image_pil = Image.open(image_path).convert('RGB')
        original_size = image_pil.size
    except Exception as e:
        if verbose:
            print(f"❌ 无法加载图片 {image_path}: {e}")
        return None
    
    # 图片预处理
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    image_tensor = transforms(image_pil).unsqueeze(0).to(device)
    orig_size_tensor = torch.tensor([[original_size[0], original_size[1]]], dtype=torch.int64, device=device)
    
    # 执行推理
    with torch.no_grad():
        outputs = model(image_tensor, orig_size_tensor)
    
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
        if verbose:
            print(f"❌ 输出格式异常: {type(outputs)}")
        return None
    
    # 过滤低置信度的检测结果
    valid_detections = scores > threshold
    filtered_labels = labels[valid_detections]
    filtered_boxes = boxes[valid_detections]
    filtered_scores = scores[valid_detections]
    
    # 绘制检测结果
    result_image = draw_detections(image_pil.copy(), labels, boxes, scores, threshold=threshold, verbose=verbose)
    
    # 保存结果
    if output_path is None:
        output_path = f"prediction_result_{os.path.basename(image_path)}"
    
    result_image.save(output_path)
    
    # 构建检测结果数据
    detections = []
    coco_classes = ['AD', 'BC', 'EC', 'L', 'LC', 'M', 'NT', 'SM', 'SQ', 'TC1', 'TC2', 'TC3']
    
    for label, box, score in zip(filtered_labels, filtered_boxes, filtered_scores):
        class_id = int(label.item())
        class_name = coco_classes[class_id] if class_id < len(coco_classes) else f"class_{class_id}"
        x1, y1, x2, y2 = box.tolist()
        
        detections.append({
            "class_id": int(class_id),
            "class_name": class_name,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(score.item())
        })
    
    result_data = {
        "image_path": image_path,
        "image_name": os.path.basename(image_path),
        "image_size": {"width": original_size[0], "height": original_size[1]},
        "detections": detections,
        "detection_count": len(detections)
    }
    
    return result_data

def predict_batch(config_path, model_path, input_dir, output_dir=None, device='cpu', pattern='*.png', threshold=0.5):
    """批量推理文件夹中的所有图片"""
    
    # 记录开始时间
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*70)
    print(" " * 20 + "RT-DETR v2 批量推理")
    print("="*70)
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return False
    
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(input_dir, "predictions")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    
    # 加载模型（只加载一次）
    model_load_start = time.time()
    model = load_model(config_path, model_path, device)
    if model is None:
        return False
    model_load_time = time.time() - model_load_start
    print(f"⏱️  模型加载耗时: {model_load_time:.2f} 秒\n")
    
    # 获取所有图片文件
    image_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    
    if len(image_files) == 0:
        print(f"❌ 在 {input_dir} 中未找到匹配 {pattern} 的图片文件")
        return False
    
    print("="*70)
    print(f"📊 找到 {len(image_files)} 张图片，开始批量推理...")
    print(f"🕐 开始时间: {start_datetime}")
    print("="*70 + "\n")
    
    # 批量处理
    success_count = 0
    fail_count = 0
    all_results = []
    total_detections = 0
    
    inference_start_time = time.time()
    
    # 打印统计信息标题（固定位置，只打印一次）
    print("\n" + "─" * 70)
    print("实时统计信息（固定更新）:")
    print("─" * 70)
    
    # 使用 tqdm 创建进度条
    pbar = tqdm(
        enumerate(image_files, 1),
        total=len(image_files),
        desc="推理进度",
        unit="张",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        position=0,
        leave=True
    )
    
    # 用于存储统计信息
    last_image_name = ""
    last_detection_count = 0
    last_class_summary = ""
    
    # 初始化统计信息显示区域（固定4行）
    stats_lines = [
        f"当前处理: {'':<48}",
        f"成功处理: {success_count:>6d} 张  |  失败: {fail_count:>6d} 张  |  检测目标总数: {total_detections:>6d} 个",
        f"当前图片检测: {last_detection_count:>3d} 个目标  |  类别: {last_class_summary:<28}",
        f"已用时间: {0:>6.1f} 秒  |  剩余时间: {0:>6.1f} 秒  |  平均速度: {0:>6.2f} 张/秒"
    ]
    for line in stats_lines:
        print(line)
    print("─" * 70)
    
    for idx, image_path in pbar:
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"pred_{image_name}")
        
        # 更新进度条描述
        elapsed_time = time.time() - inference_start_time
        avg_time_per_image = elapsed_time / idx if idx > 0 else 0
        remaining_images = len(image_files) - idx
        estimated_remaining_time = avg_time_per_image * remaining_images
        speed = idx / elapsed_time if elapsed_time > 0 else 0
        
        # 更新进度条后缀信息
        pbar.set_postfix({
            '当前': image_name[:10] + '...' if len(image_name) > 10 else image_name,
            '成功': success_count,
            '失败': fail_count,
            '检测': total_detections
        })
        
        try:
            result_data = predict_image(model, image_path, output_path, device, verbose=False, threshold=threshold)
            if result_data:
                all_results.append(result_data)
                total_detections += result_data["detection_count"]
                success_count += 1
                
                # 更新当前图片的检测信息
                last_image_name = image_name[:48] if len(image_name) <= 48 else image_name[:45] + "..."
                last_detection_count = result_data["detection_count"]
                
                if result_data["detection_count"] > 0:
                    classes = [d["class_name"] for d in result_data["detections"]]
                    class_counts = {}
                    for cls in classes:
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                    last_class_summary = ", ".join([f"{k}({v})" for k, v in sorted(class_counts.items())])
                    if len(last_class_summary) > 28:
                        last_class_summary = last_class_summary[:25] + "..."
                else:
                    last_class_summary = "无"
            else:
                fail_count += 1
                last_image_name = image_name[:48] if len(image_name) <= 48 else image_name[:45] + "..."
                last_detection_count = 0
                last_class_summary = "处理失败"
        except Exception as e:
            fail_count += 1
            last_image_name = image_name[:48] if len(image_name) <= 48 else image_name[:45] + "..."
            last_detection_count = 0
            last_class_summary = f"错误: {str(e)[:18]}"
        
        # 使用 ANSI 转义码在固定位置更新统计信息（上移5行到统计信息区域）
        # 注意：进度条占1行，统计信息占5行（标题+分隔线+4行数据+分隔线）
        print(f"\033[6A\033[K当前处理: {last_image_name:<48}", end="\n", flush=False)
        print(f"\033[K成功处理: {success_count:>6d} 张  |  失败: {fail_count:>6d} 张  |  检测目标总数: {total_detections:>6d} 个", end="\n", flush=False)
        print(f"\033[K当前图片检测: {last_detection_count:>3d} 个目标  |  类别: {last_class_summary:<28}", end="\n", flush=False)
        print(f"\033[K已用时间: {elapsed_time:>6.1f} 秒  |  剩余时间: {estimated_remaining_time:>6.1f} 秒  |  平均速度: {speed:>6.2f} 张/秒", end="\n", flush=False)
        print(f"\033[K" + "─" * 70, end="", flush=True)
        print(f"\033[5B", end="", flush=True)  # 下移回进度条位置
    
    # 关闭进度条
    pbar.close()
    # 清除统计信息区域（上移6行，清除5行，下移回原位置）
    print("\033[6A", end="")
    for _ in range(5):
        print("\033[K")
    print("\033[6B", end="")
    
    # 计算总耗时
    total_time = time.time() - start_time
    inference_time = time.time() - inference_start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg_time = inference_time / len(image_files) if len(image_files) > 0 else 0
    
    # 保存JSON标注文件
    json_output_path = os.path.join(output_dir, "annotations.json")
    json_data = {
        "info": {
            "description": "RT-DETR v2 批量推理结果",
            "version": "1.0",
            "created": start_datetime,
            "finished": end_datetime,
            "total_time_seconds": round(total_time, 2),
            "inference_time_seconds": round(inference_time, 2),
            "model_load_time_seconds": round(model_load_time, 2),
            "average_time_per_image_seconds": round(avg_time, 3)
        },
        "statistics": {
            "total_images": len(image_files),
            "successful": success_count,
            "failed": fail_count,
            "total_detections": total_detections,
            "average_detections_per_image": round(total_detections / success_count, 2) if success_count > 0 else 0
        },
        "images": all_results
    }
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    # 输出统计信息
    print("\n" + "="*70)
    print(" " * 25 + "批量推理完成")
    print("="*70)
    print(f"📊 统计信息:")
    print(f"   总图片数:     {len(image_files):6d} 张")
    print(f"   成功处理:     {success_count:6d} 张")
    print(f"   处理失败:     {fail_count:6d} 张")
    print(f"   检测目标总数: {total_detections:6d} 个")
    if success_count > 0:
        print(f"   平均每张图片: {total_detections/success_count:.2f} 个目标")
    print()
    print(f"⏱️  时间统计:")
    print(f"   开始时间:     {start_datetime}")
    print(f"   结束时间:     {end_datetime}")
    print(f"   模型加载:     {model_load_time:8.2f} 秒")
    print(f"   推理总耗时:   {inference_time:8.2f} 秒")
    print(f"   总耗时:       {total_time:8.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"   平均每张:     {avg_time:8.3f} 秒")
    if len(image_files) > 0:
        print(f"   处理速度:     {len(image_files)/total_time:8.2f} 张/秒")
    print()
    print(f"📁 输出文件:")
    print(f"   标注图片目录: {output_dir}")
    print(f"   JSON标注文件: {json_output_path}")
    print("="*70 + "\n")
    
    return True

def draw_detections(image, labels, boxes, scores, threshold=0.5, verbose=False, font_size=20):
    """在图片上绘制检测结果"""
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体，如果失败则使用默认字体
    font = None
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:/Windows/Fonts/arial.ttf",  # Windows
        "C:/Windows/Fonts/arialbd.ttf",  # Windows Bold
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        except:
            continue
    
    # 如果所有字体路径都失败，尝试使用默认字体
    if font is None:
        try:
            # 尝试使用PIL的默认字体（如果支持）
            font = ImageFont.load_default()
            # 默认字体通常较小，调整字体大小估算
            font_size = 12
        except:
            font = None
    
    # COCO数据集类别名称
    coco_classes = ['AD',
                    'BC',
                    'EC',
                    'L',
                    'LC',
                    'M',
                    'NT',
                    'SM',
                    'SQ',
                    'TC1',
                    'TC2',
                    'TC3']  # 根据您的数据集修改类别名称
    
    # 过滤低置信度的检测结果
    valid_detections = scores > threshold
    filtered_labels = labels[valid_detections]
    filtered_boxes = boxes[valid_detections]
    filtered_scores = scores[valid_detections]
    
    if verbose:
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
        
        # 获取文本的实际尺寸
        try:
            if font is not None:
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                # 如果没有字体，使用估算值
                text_width = len(label_text) * (font_size // 2)
                text_height = font_size
        except:
            # 如果textbbox不可用（旧版本PIL），使用估算值
            text_width = len(label_text) * (font_size // 2)
            text_height = font_size
        
        # 绘制标签背景和文字
        padding = 5
        draw.rectangle([x1, y1-text_height-padding*2, x1+text_width+padding*2, y1], fill='red')
        if font is not None:
            draw.text((x1+padding, y1-text_height-padding), label_text, fill='white', font=font)
        else:
            draw.text((x1+padding, y1-text_height-padding), label_text, fill='white')
        
        if verbose:
            print(f"     {i+1}. {class_name}: 置信度 {score:.3f}")
    
    return image

if __name__ == "__main__":
    # 配置参数 - 请修改这些路径
    config_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_cancer_detection.yml"  # 您的配置文件
    model_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection/best.pth"  # 您的模型文件
    input_dir = "/home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesKeep/Patches5"  # 输入图片文件夹
    output_dir = "/home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesInference/Patches5"  # 输出结果文件夹（可选，默认会在输入文件夹下创建predictions文件夹）
    device = "cuda:0"  # 或 "cuda:0" 如果有GPU
    
    # 检查文件是否存在
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        print("请修改脚本中的 config_file 变量")
        exit(1)
    
    if not os.path.exists(model_file):
        print(f"❌ 模型文件不存在: {model_file}")
        print("请修改脚本中的 model_file 变量")
        exit(1)
    
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        print("请修改脚本中的 input_dir 变量")
        exit(1)
    
    # 执行批量预测
    success = predict_batch(
        config_path=config_file,
        model_path=model_file,
        input_dir=input_dir,
        output_dir=output_dir,
        device=device,
        pattern='*.png'  # 匹配所有PNG文件
    )
    
    if success:
        print("\n🎉 批量预测完成！")
    else:
        print("\n❌ 批量预测失败！")
