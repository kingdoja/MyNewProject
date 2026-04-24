#!/usr/bin/env python3
"""
Label Studio ML 后端集成示例
使用 best_full.pt 模型进行预测和训练

使用方法：
1. 安装依赖：pip install label-studio-ml
2. 创建 ML 后端：label-studio-ml create my_rtdetr_backend
3. 将此文件内容复制到 my_rtdetr_backend/model.py
4. 修改 MODEL_PATH 为你的模型路径
5. 启动服务：label-studio-ml start my_rtdetr_backend
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path


class RTDETRModel(LabelStudioMLBase):
    """RT-DETR 模型用于 Label Studio ML 后端"""
    
    def __init__(self, model_path=None, **kwargs):
        """
        初始化模型
        
        Args:
            model_path: 模型文件路径，默认使用 best_full.pt
        """
        super(RTDETRModel, self).__init__(**kwargs)
        
        # 模型路径配置
        # 优先使用传入的路径，否则使用默认路径
        if model_path is None:
            # 默认模型路径（相对于此文件）
            default_path = os.path.join(
                os.path.dirname(__file__),
                '..', 'rtdetrv2_pytorch', 'exported_models', 'best_full.pt'
            )
            model_path = os.path.abspath(default_path)
        
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 图像预处理
        self.input_size = 640
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        print(f"正在加载模型: {self.model_path}")
        
        # 加载 full_model 格式的模型
        self.model = torch.load(self.model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)
        
        print(f"模型加载成功，设备: {self.device}")
    
    def predict(self, tasks, **kwargs):
        """
        预测函数 - 对输入图像进行目标检测
        
        Args:
            tasks: Label Studio 任务列表，每个任务包含图像URL或路径
            
        Returns:
            预测结果列表，格式符合 Label Studio 要求
        """
        results = []
        
        for task in tasks:
            # 获取图像路径
            image_path = get_image_local_path(task['data']['image'])
            
            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在: {image_path}")
                results.append({
                    'result': [],
                    'score': 0.0
                })
                continue
            
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            original_size = image.size  # (width, height)
            
            # 预处理
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            input_size_tensor = torch.tensor([[self.input_size, self.input_size]]).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor, input_size_tensor)
            
            # 解析输出结果
            # 注意：根据你的模型实际输出格式调整这部分
            # RT-DETR 通常输出 boxes, scores, labels
            predictions = []
            
            if isinstance(outputs, dict):
                boxes = outputs.get('boxes', [])
                scores = outputs.get('scores', [])
                labels = outputs.get('labels', [])
            elif isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                boxes, scores, labels = outputs[0], outputs[1], outputs[2]
            else:
                # 如果输出格式不同，需要根据实际情况调整
                boxes = outputs[0] if len(outputs) > 0 else []
                scores = outputs[1] if len(outputs) > 1 else []
                labels = outputs[2] if len(outputs) > 2 else []
            
            # 转换为 Label Studio 格式
            # 计算缩放比例（从模型输入尺寸到原始图像尺寸）
            scale_x = original_size[0] / self.input_size
            scale_y = original_size[1] / self.input_size
            
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                if score < 0.5:  # 置信度阈值
                    continue
                
                # 转换坐标格式
                # RT-DETR 通常使用 [x1, y1, x2, y2] 格式
                if len(box) >= 4:
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    
                    # 缩放到原始图像尺寸
                    x1 = float(x1 * scale_x)
                    y1 = float(y1 * scale_y)
                    x2 = float(x2 * scale_x)
                    y2 = float(y2 * scale_y)
                    
                    # 转换为百分比格式（Label Studio 常用格式）
                    width_pct = (x2 - x1) / original_size[0] * 100
                    height_pct = (y2 - y1) / original_size[1] * 100
                    x_pct = x1 / original_size[0] * 100
                    y_pct = y1 / original_size[1] * 100
                    
                    predictions.append({
                        'id': f'result_{i}',
                        'type': 'rectanglelabels',
                        'value': {
                            'x': x_pct,
                            'y': y_pct,
                            'width': width_pct,
                            'height': height_pct,
                            'rectanglelabels': [f'class_{int(label)}']
                        },
                        'score': float(score),
                        'from_name': 'label',
                        'to_name': 'image'
                    })
            
            results.append({
                'result': predictions,
                'score': float(np.mean([p['score'] for p in predictions])) if predictions else 0.0
            })
        
        return results
    
    def fit(self, annotations, **kwargs):
        """
        训练函数 - 使用新的标注数据更新模型（可选）
        
        Args:
            annotations: Label Studio 标注数据
            
        Returns:
            训练状态信息
        """
        # TODO: 实现模型微调逻辑
        # 这里可以添加使用新标注数据对模型进行微调的代码
        
        print(f"收到 {len(annotations)} 个标注，开始训练...")
        
        # 示例：保存标注数据用于后续训练
        # 实际实现需要根据你的训练流程调整
        
        return {
            'status': 'success',
            'message': f'已处理 {len(annotations)} 个标注'
        }


# 如果直接运行此文件，可以测试模型加载
if __name__ == '__main__':
    # 测试模型加载
    model_path = '/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/exported_models/best_full.pt'
    
    try:
        model = RTDETRModel(model_path=model_path)
        print("✅ 模型加载测试成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")




