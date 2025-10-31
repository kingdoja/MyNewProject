import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 添加项目路径
sys.path.insert(0, '.')
from src.core import YAMLConfig

def visualize_test(config_path, model_path, data_root, output_dir='test_results', num_samples=10, device='cpu'):
    """可视化test数据集的检测结果，并显示真实标签"""
    
    print("=== RT-DETRv2 Test 可视化 ===")
    
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
    
    # 4. 查找测试集标注文件
    test_annotation_files = []
    test_annotation_files.extend(list(Path(data_root).glob('test/_annotations_*.json')))  # 修改为符合实际命名
    test_annotation_files.extend(list(Path(data_root).glob('test/annotations/*.json')))
    test_annotation_files.extend(list(Path(data_root).glob('test/annotations/instances_test*.json')))
    test_annotation_files.extend(list(Path(data_root).glob('annotations/instances_test*.json')))
    
    if not test_annotation_files:
        print("❌ 未找到测试集标注文件")
        return
    
    annotation_file = test_annotation_files[0]
    print(f"使用测试集标注文件: {annotation_file}")
    
    # 5. 加载测试集标注
    with open(annotation_file, 'r') as f:
        test_annotations = json.load(f)
    
    # 创建文件名到图像ID的映射
    filename_to_id = {}
    id_to_filename = {}
    for img in test_annotations['images']:
        filename_to_id[img['file_name']] = img['id']
        id_to_filename[img['id']] = img['file_name']
    
    # 创建图像ID到注释的映射
    image_id_to_annotations = {}
    for ann in test_annotations['annotations']:
        image_id = ann['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(ann)
    
    # 6. 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 7. 处理测试集样本
    test_image_dir = Path(data_root) / 'test'
    image_files = []
    
    # 改进图像文件搜索逻辑
    if (test_image_dir / 'images').exists():
        # 如果存在images子目录
        image_files.extend(list((test_image_dir / 'images').glob('*.jpg')))
        image_files.extend(list((test_image_dir / 'images').glob('*.jpeg')))
        image_files.extend(list((test_image_dir / 'images').glob('*.png')))
    else:
        # 在test根目录查找
        image_files.extend(list(test_image_dir.glob('*.jpg')))
        image_files.extend(list(test_image_dir.glob('*.jpeg')))
        image_files.extend(list(test_image_dir.glob('*.png')))
    
    print(f"   找到 {len(image_files)} 张测试图像")
    
    # 只处理指定数量的样本
    image_files = image_files[:num_samples]
    
    print(f"   处理前{len(image_files)}个测试样本...")
    
    # 用于评估的检测结果
    coco_detections = []
    detection_id = 1
    
    for idx, image_file in enumerate(image_files):
        print(f"   处理图像 {idx+1}/{len(image_files)}: {image_file.name}")
        
        try:
            # 加载图像
            image_pil = Image.open(image_file).convert('RGB')
            original_size = image_pil.size
            
            # 图像预处理
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
                print(f"   输出格式异常: {type(outputs)}")
                continue
            
            # 只保留分数大于阈值的检测结果
            keep = scores > 0.3
            filtered_labels = labels[keep]
            filtered_boxes = boxes[keep]
            filtered_scores = scores[keep]
            
            # 获取该图像的真实标签
            gt_annotations = []
            image_id = None
            if image_file.name in filename_to_id:
                image_id = filename_to_id[image_file.name]
                if image_id in image_id_to_annotations:
                    gt_annotations = image_id_to_annotations[image_id]
            
            # 绘制检测结果和真实标签到不同图像上
            pred_image, gt_image = draw_test_detections_separate(image_pil, filtered_labels, filtered_boxes, filtered_scores, gt_annotations, test_annotations.get('categories', []), data_root)
            
            # 保存结果
            pred_output_filename = f"test_sample_{idx:03d}_pred.jpg"
            pred_output_filepath = output_path / pred_output_filename
            pred_image.save(pred_output_filepath)
            
            gt_output_filename = f"test_sample_{idx:03d}_gt.jpg"
            gt_output_filepath = output_path / gt_output_filename
            gt_image.save(gt_output_filepath)
            
            # 保存检测结果到JSON文件
            json_filename = f"test_sample_{idx:03d}.json"
            json_filepath = output_path / json_filename
            save_test_detection_results(json_filepath, filtered_labels, filtered_boxes, filtered_scores, gt_annotations)
            
            # 为评估准备检测结果
            if image_id is not None:
                for label, box, score in zip(filtered_labels, filtered_boxes, filtered_scores):
                    if isinstance(box, torch.Tensor):
                        x1, y1, x2, y2 = box.tolist()
                    else:
                        x1, y1, x2, y2 = box
                    
                    coco_detections.append({
                        "id": detection_id,
                        "image_id": image_id,
                        "category_id": int(label.item()) if isinstance(label, torch.Tensor) else int(label),
                        "bbox": [x1, y1, x2-x1, y2-y1],
                        "score": float(score.item()) if isinstance(score, torch.Tensor) else float(score)
                    })
                    detection_id += 1
                    
        except Exception as e:
            print(f"   处理图像 {image_file.name} 时出错: {e}")
            continue
    
    print(f"✓ 完成！结果已保存到: {output_path}")
    print(f"   图像文件: {output_path}/*.jpg")
    print(f"   检测结果: {output_path}/*.json")
    
    # 8. 评估测试集性能
    if coco_detections:
        evaluate_test_performance(annotation_file, coco_detections, output_path)

def draw_test_detections_separate(image, labels, boxes, scores, gt_annotations, categories, data_root, threshold=0.3):
    """在测试图像上分别绘制检测结果和真实标签到不同图像上"""
    
    # 使用更小的字体（从20号字体改为14号字体）并尝试加粗
    try:
        # 尝试加载加粗字体
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
    for cat in categories:
        cat_id_to_name[cat['id']] = cat['name']
    
    # 痰液细胞数据集类别名称 - 从配置或数据集中动态获取
    try:
        # 优先从classes.txt文件加载类别名称
        classes_txt_path = os.path.join(data_root, 'classes.txt')
        if os.path.exists(classes_txt_path):
            with open(classes_txt_path, 'r') as f:
                coco_classes = [line.strip() for line in f.readlines()]
        elif categories:
            # 尝试从类别信息中获取类别名称
            coco_classes = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
        else:
            # 默认类别列表
            coco_classes = ["AD","BC","EC","L","LC","M","NT","SM","SQ","TC1","TC2", "TC3"]
    except:
        coco_classes = ["AD","BC","EC","L","LC","M","NT","SM","SQ","TC1","TC2", "TC3"]
    
    # 创建预测图像副本
    pred_image = image.copy()
    pred_draw = ImageDraw.Draw(pred_image)
    
    # 绘制预测结果（红色）
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
            pred_draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # 绘制标签文本
            score_value = score.item() if isinstance(score, torch.Tensor) else score
            label_text = f"Pred: {class_name} {score_value:.2f}"
            
            # 绘制标签背景和文字（调整背景框大小以适应更小的字体）
            left, top, right, bottom = pred_draw.textbbox((0, 0), label_text, font=font)
            text_width = right - left
            text_height = bottom - top
            
            pred_draw.rectangle([x1, y1-text_height-4, x1+text_width+12, y1], fill='red')
            pred_draw.text((x1+6, y1-text_height-2), label_text, fill='white', font=font)
    
    # 创建真实标签图像副本
    gt_image = image.copy()
    gt_draw = ImageDraw.Draw(gt_image)
    
    # 绘制真实标签（绿色）
    for ann in gt_annotations:
        # 获取边界框坐标
        x, y, w, h = ann['bbox']
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        # 获取类别名称
        cat_id = ann['category_id']
        class_name = cat_id_to_name.get(cat_id, f"class_{cat_id}")
        
        # 绘制真实标签框
        gt_draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
        
        # 绘制标签文本 - 修改为与预测标签相同的样式
        label_text = f"GT: {class_name}"
        
        # 绘制标签背景和文字 - 使用与预测标签相同的样式（调整背景框大小以适应更小的字体）
        left, top, right, bottom = gt_draw.textbbox((0, 0), label_text, font=font)
        text_width = right - left
        text_height = bottom - top
        
        gt_draw.rectangle([x1, y1-text_height-4, x1+text_width+12, y1], fill='green')
        gt_draw.text((x1+6, y1-text_height-2), label_text, fill='white', font=font)
    
    return pred_image, gt_image

def evaluate_test_performance(annotations_file, detections, output_path):
    """评估测试集性能"""
    print("=== 测试集性能评估 ===")
    
    try:
        # 使用pycocotools进行评估
        coco_gt = COCO(str(annotations_file))
        coco_dt = coco_gt.loadRes(detections)
        
        # 创建COCO评估器
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # 保存评估结果
        eval_results = {
            'AP@0.5:0.95': coco_eval.stats[0],
            'AP@0.5': coco_eval.stats[1],
            'AP@0.75': coco_eval.stats[2],
            'AP@small': coco_eval.stats[3],
            'AP@medium': coco_eval.stats[4],
            'AP@large': coco_eval.stats[5],
            'AR@1': coco_eval.stats[6],
            'AR@10': coco_eval.stats[7],
            'AR@100': coco_eval.stats[8],
            'AR@small': coco_eval.stats[9],
            'AR@medium': coco_eval.stats[10],
            'AR@large': coco_eval.stats[11]
        }
        
        eval_report_path = output_path / "test_evaluation.json"
        with open(eval_report_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"测试集评估报告已保存到: {eval_report_path}")
        
        # # 打印评估结果摘要
        # print("\n=== 测试集评估结果 ===")
        # print(f"AP@0.5:0.95: {eval_results['AP@0.5:0.95']:.4f}")
        # print(f"AP@0.5: {eval_results['AP@0.5']:.4f}")
        # print(f"AP@0.75: {eval_results['AP@0.75']:.4f}")
        # print(f"AP@small: {eval_results['AP@small']:.4f}")
        # print(f"AP@medium: {eval_results['AP@medium']:.4f}")
        # print(f"AP@large: {eval_results['AP@large']:.4f}")
        # print(f"AR@1: {eval_results['AR@1']:.4f}")
        # print(f"AR@10: {eval_results['AR@10']:.4f}")
        # print(f"AR@100: {eval_results['AR@100']:.4f}")
        
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

def save_test_detection_results(filepath, labels, boxes, scores, gt_annotations):
    """保存测试集检测结果到JSON文件"""
    results = {
        'predictions': [],
        'ground_truth': []
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
    for ann in gt_annotations:
        results['ground_truth'].append({
            'category_id': ann['category_id'],
            'bbox': ann['bbox']
        })
    
    # 保存到JSON文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # 配置参数 - 请修改这些路径
    config_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_cancer_detection.yml"  # 您的配置文件
    # model_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/premodel/best.pth" # 您的模型文件
    model_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection/best.pth"
    data_root = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split212/"  # 数据集根目录
    
    test_output_directory = "test_visualization"  # 测试集输出目录
    num_samples_to_visualize = 50  # 要可视化的样本数量
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
    
    # 执行test可视化
    visualize_test(
        config_path=config_file,
        model_path=model_file,
        data_root=data_root,
        output_dir=test_output_directory,
        num_samples=num_samples_to_visualize,
        device=device
    )