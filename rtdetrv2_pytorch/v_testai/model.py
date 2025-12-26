import os
from typing import List, Dict, Optional
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from PIL import Image
import torchvision.transforms as T
from torchvision.ops import nms
import numpy as np

class NewModel(LabelStudioMLBase):
    """RT-DETR集成到Label Studio ML后端"""

    def setup(self):
        """
        初始化RT-DETR模型
        支持从环境变量或参数中读取配置
        """
        # 0. 自动热更新相关配置
        self.auto_reload = os.environ.get('AUTO_RELOAD_MODEL', 'true').lower() in ('1', 'true', 'yes')
        self.model_output_dir = os.environ.get('MODEL_OUTPUT_DIR', os.path.join(os.path.dirname(__file__), 'models'))
        self.model_path = None
        self.model_mtime = None
        
        # 推理设备控制：可通过环境变量强制使用CPU
        self.force_cpu = os.environ.get('FORCE_CPU_INFERENCE', 'false').lower() in ('1', 'true', 'yes')
        self._requires_orig_target_sizes = False
        self._prefer_orig_target_sizes_arg = True  # 默认尝试向模型传入 orig_target_sizes
        
        # 1. 获取模型路径（优先级：环境变量 > 缓存 > 默认值）
        model_path = os.environ.get('MODEL_PATH')
        
        # 尝试从缓存中获取（如果存在）
        if not model_path:
            try:
                cached_path = self.get('model_path')
                if cached_path:
                    model_path = cached_path
            except:
                pass
        
        # 如果传入的是目录，则选取该目录下最新的模型
        if model_path and os.path.isdir(model_path):
            latest = self._find_latest_model_in_dir(model_path)
            if latest:
                model_path = latest
        
        # 使用默认路径
        if not model_path:
            # 首先尝试加载您的自定义模型
            custom_model_path = os.path.join(os.path.dirname(__file__), 'models', 'rtdetr_torchscript_cuda.pt')
            if os.path.exists(custom_model_path):
                model_path = custom_model_path
                print(f"✓ 找到自定义模型: {model_path}")
            else:
                # 如果自定义模型不存在，使用默认模型
                model_path = os.path.join(os.path.dirname(__file__), 'models', 'rtdetr_torchscript.pt')
                print(f"使用默认模型: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件未找到: {model_path}\n"
                f"请将您训练的RT-DETR模型放置在正确位置，或设置 MODEL_PATH 环境变量"
            )
        
        # 2. 加载模型（含CUDA检查与状态记录）
        self._load_model(model_path)
        
        # 4. 获取类别标签（优先级：环境变量 > 参数 > 默认值）
        labels_str = os.environ.get('MODEL_LABELS')
        if labels_str:
            # 从环境变量读取，支持逗号分隔
            self.labels = [label.strip() for label in labels_str.split(',')]
            print(f"从环境变量读取类别标签: {self.labels}")
        else:
            # 默认类别标签（医疗痰液细胞）
            self.labels = ["AD", "BC", "EC", "L", "LC", "M", "NT", "SM", "SQ", "TC1", "TC2", "TC3"]
            print(f"使用默认类别标签: {self.labels}")
        
        # 5. 设置置信度阈值
        self.conf_threshold = float(os.environ.get('CONF_THRESHOLD', '0.25'))
        print(f"置信度阈值: {self.conf_threshold}")
        
        # 6. 设置NMS IoU阈值（用于过滤重叠框）
        self.iou_threshold = float(os.environ.get('IOU_THRESHOLD', '0.45'))
        print(f"NMS IoU阈值: {self.iou_threshold}")
        
        # 6.1 设置跨类别NMS阈值（仅针对不同类别的高度重叠框）
        # 设为更严格的阈值以只移除几乎完全重叠的跨类别框
        self.cross_class_iou_threshold = float(os.environ.get('CROSS_CLASS_IOU_THRESHOLD', '0.90'))
        print(f"跨类别NMS IoU阈值: {self.cross_class_iou_threshold}")
        
        # 6.2 过滤极小框的面积阈值（像素与比例）
        self.min_box_area = float(os.environ.get('MIN_BOX_AREA', '0'))
        self.min_box_area_ratio = float(os.environ.get('MIN_BOX_AREA_RATIO', '0'))
        if self.min_box_area > 0:
            print(f"最小边界框面积(像素): {self.min_box_area}")
        if self.min_box_area_ratio > 0:
            print(f"最小边界框面积(占比): {self.min_box_area_ratio}")
        
        # 7. 训练开关（默认关闭自动训练）
        self.enable_training = os.environ.get('ENABLE_RTDETR_TRAINING', 'false').lower() in ('1', 'true', 'yes')
        if not self.enable_training:
            print("训练功能已禁用（ENABLE_RTDETR_TRAINING=false）")
        else:
            print("训练功能已启用（ENABLE_RTDETR_TRAINING=true）")
        
        # 8. 设置模型版本
        model_version = os.environ.get('MODEL_VERSION', 'rtdetr-custom-1.0')
        self.set("model_version", model_version)
        print(f"模型版本: {model_version}")
        
        # 定义预处理变换
        # 为了与训练项目中的批量推理脚本 `predict_batch_torchscript.py` 保持一致，
        # 这里仅做 Resize + ToTensor，不再做 ImageNet 标准化。
        # 训练/导出时如果未使用 ImageNet 标准化，而这里额外做标准化，会导致输入分布不一致，从而预测效果变差。
        self.transform = T.Compose([
            T.Resize((640, 640)),  # RT-DETR 通常使用固定尺寸输入
            T.ToTensor(),
        ])
    
    def _load_model(self, model_path: str, *, force_cpu: Optional[bool] = None):
        """加载模型并记录路径与修改时间"""
        print(f"正在加载模型: {model_path}")
        
        if force_cpu is None:
            force_cpu = getattr(self, 'force_cpu', False)
        
        # 检查CUDA支持
        cuda_available = torch.cuda.is_available()
        if cuda_available and not force_cpu:
            self.device = torch.device('cuda')
            print(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            if cuda_available and force_cpu:
                print("根据配置/兼容性要求，使用 CPU")
            else:
                print("CUDA 不可用，使用 CPU")
        
        # 加载模型
        try:
            # 对于TorchScript模型，使用map_location加载到指定设备
            # 注意：如果模型是在CPU上trace的，即使map_location='cuda'也可能无法完全在CUDA上运行
            # 因此必须在导出时就在CUDA上trace（见export_pt.py修复）
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            
            # 对于TorchScript模型，尝试将模型移动到目标设备
            # 注意：这不会改变模型内部被trace时的设备，但可以确保某些操作在正确设备上
            if self.device.type == 'cuda':
                try:
                    # 测试模型是否能在CUDA上运行
                    # RT-DETR的DeployModel需要orig_target_sizes参数
                    test_input = torch.rand(1, 3, 640, 640, device=self.device)
                    test_size = torch.tensor([[640, 640]], device=self.device)
                    with torch.no_grad():
                        _ = self.model(test_input, test_size)
                    print("✓ TorchScript模型加载成功（CUDA测试通过）")
                except RuntimeError as cuda_test_error:
                    error_str = str(cuda_test_error)
                    cuda_errors = ["Found no NVIDIA driver", "CUDA", "cuda", "Expected all tensors to be on the same device"]
                    if any(err in error_str for err in cuda_errors):
                        print("⚠️  警告: TorchScript模型无法在CUDA上运行")
                        print(f"   错误: {error_str[:200]}")
                        print("   尝试切换到CPU模式...")
                        self.device = torch.device('cpu')
                        self.force_cpu = True
                        # 重新加载到CPU
                        try:
                            self.model = torch.jit.load(model_path, map_location='cpu')
                            self.model.eval()
                            print("✓ TorchScript模型已重新加载到CPU")
                        except Exception as cpu_load_error:
                            print(f"❌ CPU模式加载也失败: {cpu_load_error}")
                            raise RuntimeError(
                                f"模型无法加载。模型可能是在CUDA上导出的TorchScript，但当前系统没有GPU。\n"
                                f"建议：\n"
                                f"1. 设置环境变量 FORCE_CPU_INFERENCE=true 强制使用CPU\n"
                                f"2. 使用在CPU上导出的模型版本\n"
                                f"3. 或安装NVIDIA驱动和CUDA以使用GPU版本\n"
                                f"原始错误: {error_str[:500]}"
                            )
                    else:
                        # 其他错误，可能是参数问题，但不一定是设备问题
                        print(f"⚠️  CUDA测试时出现错误（可能是参数问题）: {cuda_test_error}")
                        print("   继续使用CUDA，实际推理时会自动处理")
                        print("✓ TorchScript模型加载成功（CUDA，但测试时出现警告）")
            else:
                print("✓ TorchScript模型加载成功（CPU模式）")
        except Exception as e:
            print(f"尝试使用torch.jit.load失败: {e}")
            print("尝试使用torch.load加载...")
            try:
                self.model = torch.load(model_path, map_location=self.device)
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                # 对于普通PyTorch模型，可以移动到设备
                if hasattr(self.model, 'to'):
                    self.model = self.model.to(self.device)
                print("✓ 模型加载成功")
            except Exception as e2:
                raise RuntimeError(f"模型加载失败: {e2}")
        
        # 记录当前模型路径与修改时间，并缓存
        self.model_path = model_path
        try:
            self.model_mtime = os.path.getmtime(model_path)
        except Exception:
            self.model_mtime = None
        try:
            self.set('model_path', model_path)
        except Exception:
            pass
        
        # 根据文件名与mtime更新模型版本标识，便于在预测结果中追踪
        try:
            base = os.path.basename(model_path)
            ts_part = str(int(self.model_mtime)) if self.model_mtime else "na"
            auto_version = f"rtdetr-auto:{base}:{ts_part}"
            self.set("model_version", auto_version)
            print(f"当前模型版本标识: {auto_version}")
        except Exception:
            pass
        
        # 标记当前模型设备
        self._is_cpu_model = self.device.type == 'cpu'
    
    def _find_latest_model_in_dir(self, directory: str) -> Optional[str]:
        """在目录中查找最新的模型文件"""
        import glob
        patterns = ['*.pt', '*.pth', '*torchscript*.pt']
        candidates = []
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(directory, pat)))
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)
    
    def _maybe_reload_model(self):
        """在预测前检测是否有更新的已训练模型并热更新"""
        if not self.auto_reload:
            return
        
        try:
            # 优先使用训练流程写入的路径
            trained_model_path = self.get('trained_model_path')
        except Exception:
            trained_model_path = None
        
        # 如果没有缓存路径，尝试从输出目录中找到最新的
        if not trained_model_path and os.path.isdir(self.model_output_dir):
            trained_model_path = self._find_latest_model_in_dir(self.model_output_dir)
        
        if not trained_model_path or not os.path.exists(trained_model_path):
            return
        
        # 若路径不同或文件更新了，则重新加载
        try:
            new_mtime = os.path.getmtime(trained_model_path)
        except Exception:
            new_mtime = None
        
        should_reload = False
        if self.model_path != trained_model_path:
            should_reload = True
        elif self.model_mtime is not None and new_mtime is not None and new_mtime > self.model_mtime:
            should_reload = True
        
        if should_reload:
            print(f"检测到新模型，将热更新: {trained_model_path}")
            self._load_model(trained_model_path)

    def preprocess(self, image_path):
        # 加载并预处理图片（根据RT-DETR官方代码调整）
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        # 应用预处理变换
        transformed_image = self.transform(image)
        return transformed_image, original_size

    def apply_nms_per_class(self, boxes, scores, labels):
        """
        对每个类别分别应用NMS（非极大值抑制）
        去除同一类别中重叠度高的冗余检测框
        """
        if len(boxes) == 0:
            return boxes, scores, labels
        
        # 将numpy数组转换为torch tensor
        boxes_tensor = torch.from_numpy(boxes).float()
        scores_tensor = torch.from_numpy(scores).float()
        labels_tensor = torch.from_numpy(labels).long()
        
        # 存储保留的索引
        keep_indices = []
        
        # 对每个类别分别应用NMS
        unique_labels = torch.unique(labels_tensor)
        print(f"  应用类别内NMS: 共 {len(unique_labels)} 个类别")
        
        for label_id in unique_labels:
            # 获取当前类别的所有检测框
            label_mask = labels_tensor == label_id
            label_boxes = boxes_tensor[label_mask]
            label_scores = scores_tensor[label_mask]
            label_indices = torch.where(label_mask)[0]
            
            # 对当前类别应用NMS
            keep = nms(label_boxes, label_scores, self.iou_threshold)
            
            # 记录保留的索引
            keep_indices.extend(label_indices[keep].tolist())
            
            print(f"    类别 {int(label_id)} ({self.labels[int(label_id)]}): {len(label_boxes)} -> {len(keep)} 个框")
        
        # 按原始顺序排序保留的索引
        keep_indices = sorted(keep_indices)
        
        # 返回过滤后的结果
        return boxes[keep_indices], scores[keep_indices], labels[keep_indices]
    
    def apply_cross_class_nms(self, boxes, scores, labels):
        """
        跨类别NMS：如果不同类别的框完全重叠（IoU很高），只保留置信度最高的一个
        """
        if len(boxes) == 0:
            return boxes, scores, labels
        
        # 将numpy数组转换为torch tensor
        boxes_tensor = torch.from_numpy(boxes).float()
        scores_tensor = torch.from_numpy(scores).float()
        labels_tensor = torch.from_numpy(labels).long()
        
        # 按置信度降序排序
        sorted_indices = torch.argsort(scores_tensor, descending=True)
        boxes_sorted = boxes_tensor[sorted_indices]
        scores_sorted = scores_tensor[sorted_indices]
        labels_sorted = labels_tensor[sorted_indices]
        
        # 存储保留的索引（在原始数组中的索引）
        keep_mask = torch.ones(len(boxes), dtype=torch.bool)
        sorted_indices_list = sorted_indices.tolist()
        
        print(f"  应用跨类别NMS: 检查 {len(boxes)} 个框")
        
        # 对每个框，检查是否与已保留的框重叠
        for i in range(len(boxes_sorted)):
            orig_idx_i = sorted_indices_list[i]
            if not keep_mask[orig_idx_i]:
                continue
            
            current_box = boxes_sorted[i:i+1]  # 保持2D形状
            
            # 检查与后续所有框的IoU
            for j in range(i + 1, len(boxes_sorted)):
                orig_idx_j = sorted_indices_list[j]
                if not keep_mask[orig_idx_j]:
                    continue
                
                other_box = boxes_sorted[j:j+1]
                
                # 计算IoU
                iou = self._calculate_iou(current_box, other_box)
                
                # 如果IoU很高（几乎完全重叠），移除置信度较低的框（跨类别）
                if iou >= self.cross_class_iou_threshold:
                    keep_mask[orig_idx_j] = False
                    label_i = int(labels_sorted[i])
                    label_j = int(labels_sorted[j])
                    score_i = float(scores_sorted[i])
                    score_j = float(scores_sorted[j])
                    label_name_i = self.labels[label_i] if label_i < len(self.labels) else str(label_i)
                    label_name_j = self.labels[label_j] if label_j < len(self.labels) else str(label_j)
                    print(f"    移除重叠框: {label_name_i} (置信度 {score_i:.3f}) vs "
                          f"{label_name_j} (置信度 {score_j:.3f}), IoU={iou:.3f}")
        
        # 获取保留的索引
        keep_indices = torch.where(keep_mask)[0].tolist()
        
        print(f"  跨类别NMS后: {len(keep_indices)} 个框")
        
        # 返回过滤后的结果
        return boxes[keep_indices], scores[keep_indices], labels[keep_indices]
    
    def _calculate_iou(self, box1, box2):
        """
        计算两个边界框的IoU（交并比）
        box1, box2: [1, 4] 形状的tensor，格式为 [x1, y1, x2, y2]
        """
        # 计算交集区域
        x1_inter = torch.max(box1[0, 0], box2[0, 0])
        y1_inter = torch.max(box1[0, 1], box2[0, 1])
        x2_inter = torch.min(box1[0, 2], box2[0, 2])
        y2_inter = torch.min(box1[0, 3], box2[0, 3])
        
        # 计算交集面积
        inter_width = torch.clamp(x2_inter - x1_inter, min=0)
        inter_height = torch.clamp(y2_inter - y1_inter, min=0)
        inter_area = inter_width * inter_height
        
        # 计算并集面积
        box1_area = (box1[0, 2] - box1[0, 0]) * (box1[0, 3] - box1[0, 1])
        box2_area = (box2[0, 2] - box2[0, 0]) * (box2[0, 3] - box2[0, 1])
        union_area = box1_area + box2_area - inter_area
        
        # 计算IoU
        if union_area > 0:
            iou = inter_area / union_area
        else:
            iou = torch.tensor(0.0)
        
        return iou.item() if torch.is_tensor(iou) else float(iou)

    def postprocess(self, outputs, original_size, image_tensor_size):
        """
        将RT-DETR模型输出转换为Label Studio格式
        支持多种输出格式（dict或tuple）
        """
        predictions = []
        
        # 解析模型输出
        if isinstance(outputs, dict):
            # 字典格式：{'boxes': tensor, 'scores': tensor, 'labels': tensor}
            boxes = outputs['boxes']
            scores = outputs['scores']
            labels = outputs['labels']
        elif isinstance(outputs, (tuple, list)):
            # 元组/列表格式
            if len(outputs) == 3:
                # 检测输出格式
                # 可能是 (boxes, scores, labels) 或 (labels, boxes, scores)
                output0, output1, output2 = outputs
                
                # 根据shape判断格式
                # boxes通常是 [N, 4] 或 [B, N, 4]
                # scores通常是 [N] 或 [B, N]
                # labels通常是 [N] 或 [B, N]
                
                if output0.shape[-1] == 4:
                    # 格式: (boxes, scores, labels)
                    boxes, scores, labels = output0, output1, output2
                elif output1.shape[-1] == 4:
                    # 格式: (labels, boxes, scores)
                    labels, boxes, scores = output0, output1, output2
                else:
                    print(f"警告: 无法识别输出格式")
                    print(f"  output0.shape: {output0.shape}")
                    print(f"  output1.shape: {output1.shape}")
                    print(f"  output2.shape: {output2.shape}")
                    return predictions
            else:
                print(f"警告: 不支持的输出格式，元素数量: {len(outputs)}")
                return predictions
        else:
            print(f"警告: 不支持的输出类型: {type(outputs)}")
            return predictions
        
        # 处理批次维度：移除批次维度（假设batch_size=1）
        if boxes.dim() > 2:
            boxes = boxes[0]
        if scores.dim() > 1:
            scores = scores[0]
        if labels.dim() > 1:
            labels = labels[0]
        
        # 检查是否有检测结果
        if len(boxes) == 0 or len(scores) == 0 or len(labels) == 0:
            print(f"  ⚠️  警告: 模型输出为空 (boxes: {len(boxes)}, scores: {len(scores)}, labels: {len(labels)})")
            return predictions
        
        # 转换为CPU numpy数组
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().detach().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().detach().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().detach().numpy()
        
        # 验证数组长度一致性
        if not (len(boxes) == len(scores) == len(labels)):
            print(f"  ⚠️  警告: 输出数组长度不一致 (boxes: {len(boxes)}, scores: {len(scores)}, labels: {len(labels)})")
            min_len = min(len(boxes), len(scores), len(labels))
            boxes = boxes[:min_len]
            scores = scores[:min_len]
            labels = labels[:min_len]
            print(f"  已截断到最小长度: {min_len}")
        
        # 图像尺寸信息
        orig_w, orig_h = original_size
        if image_tensor_size is not None and len(image_tensor_size) >= 4:
            input_h = image_tensor_size[-2]
            input_w = image_tensor_size[-1]
        else:
            input_h = 640
            input_w = 640
        input_w = max(float(input_w), 1.0)
        input_h = max(float(input_h), 1.0)
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        print(f"坐标缩放: input_size=({input_w:.1f}, {input_h:.1f}), "
              f"orig_size=({orig_w:.1f}, {orig_h:.1f}), "
              f"scale=({scale_x:.3f}, {scale_y:.3f})")
        image_area = max(orig_w * orig_h, 1)
        filtered_by_area = 0
        
        # 统计信息
        total_detections = len(scores)
        print(f"后处理: 共 {total_detections} 个原始检测")
        if total_detections > 0:
            print(f"  置信度范围: {scores.min():.4f} - {scores.max():.4f}")
            print(f"  置信度阈值: {self.conf_threshold}")
            print(f"  面积过滤阈值: min_area={self.min_box_area}, min_area_ratio={self.min_box_area_ratio}")
        else:
            print(f"  ⚠️  警告: 模型没有输出任何检测结果")
            print(f"  可能原因:")
            print(f"    1. 图像中没有目标")
            print(f"    2. 模型输出格式解析错误")
            print(f"    3. 模型未正确加载")
            return predictions
        
        # 1. 首先按置信度过滤
        conf_mask = scores >= self.conf_threshold
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]
        labels = labels[conf_mask]
        
        num_after_conf = len(scores)
        print(f"  置信度过滤后: {num_after_conf} 个检测")
        if num_after_conf == 0 and total_detections > 0:
            print(f"  ⚠️  警告: 所有检测都被置信度阈值过滤掉了")
            print(f"  建议: 降低 CONF_THRESHOLD (当前: {self.conf_threshold})")
            print(f"  原始检测中最高置信度: {scores.max():.4f if len(scores) > 0 else 'N/A'}")
        
        if num_after_conf == 0:
            return predictions
        
        # 2. 转换坐标格式并收集有效框
        valid_boxes = []
        valid_scores = []
        valid_labels = []
        
        for box, score, label in zip(boxes, scores, labels):
            if len(box) != 4:
                print(f"警告: 不支持的边界框格式，长度: {len(box)}")
                continue
            
            x1, y1, x2, y2 = box
            # 模型输出的是相对于模型输入张量（默认 640x640）的像素坐标，
            # 因此按照输入尺寸 -> 原图尺寸的比例进行缩放。
            x_min = x1 * scale_x
            y_min = y1 * scale_y
            x_max = x2 * scale_x
            y_max = y2 * scale_y
            
            # 计算宽高
            width = x_max - x_min
            height = y_max - y_min
            
            # 确保坐标在有效范围内
            x_min = max(0, min(x_min, orig_w))
            y_min = max(0, min(y_min, orig_h))
            x_max = max(0, min(x_max, orig_w))
            y_max = max(0, min(y_max, orig_h))
            
            # 过滤无效框
            if width <= 0 or height <= 0:
                continue
            
            area = width * height
            area_ratio = area / image_area
            if ((self.min_box_area > 0 and area < self.min_box_area) or
                    (self.min_box_area_ratio > 0 and area_ratio < self.min_box_area_ratio)):
                filtered_by_area += 1
                continue
            
            # 获取标签索引（确保是整数）
            label_idx = int(label) if np.isscalar(label) else int(label.item() if hasattr(label, 'item') else label)
            
            # 检查标签索引是否有效
            if label_idx < 0 or label_idx >= len(self.labels):
                print(f"警告: 标签索引 {label_idx} 超出范围 [0, {len(self.labels)-1}]，跳过")
                continue
            
            # 添加到有效列表（保存绝对坐标用于NMS）
            valid_boxes.append([x_min, y_min, x_max, y_max])
            valid_scores.append(float(score) if np.isscalar(score) else float(score.item() if hasattr(score, 'item') else score))
            valid_labels.append(label_idx)
        
        if len(valid_boxes) == 0:
            return predictions
        
        if filtered_by_area:
            print(f"  面积过滤: {filtered_by_area} 个检测被移除")
        
        if len(valid_boxes) == 0:
            print(f"  ⚠️  警告: 所有检测框都被面积过滤或坐标转换过滤掉了")
            print(f"  面积过滤阈值: min_area={self.min_box_area}, min_area_ratio={self.min_box_area_ratio}")
            if num_after_conf > 0:
                print(f"  建议: 检查 MIN_BOX_AREA 和 MIN_BOX_AREA_RATIO 设置是否过于严格")
            return predictions
        
        # 转换为numpy数组
        valid_boxes = np.array(valid_boxes)
        valid_scores = np.array(valid_scores)
        valid_labels = np.array(valid_labels)
        
        # 3. 先应用类别内NMS去除同类别重叠框
        valid_boxes, valid_scores, valid_labels = self.apply_nms_per_class(
            valid_boxes, valid_scores, valid_labels
        )
        
        print(f"  类别内NMS后: {len(valid_boxes)} 个检测")
        
        # 4. 再应用跨类别NMS去除不同类别重叠框
        valid_boxes, valid_scores, valid_labels = self.apply_cross_class_nms(
            valid_boxes, valid_scores, valid_labels
        )
        
        print(f"  最终NMS过滤后: {len(valid_boxes)} 个检测")
        
        if len(valid_boxes) == 0:
            print(f"  ⚠️  警告: NMS过滤后没有剩余检测框")
        
        # 5. 生成Label Studio格式的预测结果
        for box, score, label_idx in zip(valid_boxes, valid_scores, valid_labels):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            # RT-DETR输出的是边界框（4个角），使用矩形标注
            # 如果将来有分割模型输出多边形（>4个角），可以使用多边形标注
            num_corners = 4  # RT-DETR边界框固定为4个角
            
            if num_corners == 4:
                # 矩形框（4个角）：使用RectangleLabels
                rectangle_result = {
                    "from_name": "label",   # RectangleLabels的from_name
                    "to_name": "image",     # 数据名称
                    "type": "rectanglelabels", # 标注类型
                    "value": {
                        "x": x_min / orig_w * 100,
                        "y": y_min / orig_h * 100,
                        "width": width / orig_w * 100,
                        "height": height / orig_h * 100,
                        "rectanglelabels": [self.labels[int(label_idx)]]
                    },
                    "score": float(score)
                }
                predictions.append(rectangle_result)
            else:
                # 多边形（>4个角）：使用PolygonLabels
                # 注意：RT-DETR不会产生这种情况，这是为将来扩展预留的
                # 如果将来集成分割模型，可以在这里处理多边形
                polygon_points = [
                    [x_min / orig_w * 100, y_min / orig_h * 100],  # 左上
                    [x_max / orig_w * 100, y_min / orig_h * 100],  # 右上
                    [x_max / orig_w * 100, y_max / orig_h * 100],  # 右下
                    [x_min / orig_w * 100, y_max / orig_h * 100],  # 左下
                ]
                
                polygon_result = {
                    "from_name": "label2",   # PolygonLabels的from_name
                    "to_name": "image",      # 数据名称
                    "type": "polygonlabels", # 标注类型
                    "value": {
                        "points": polygon_points,
                        "polygonlabels": [self.labels[int(label_idx)]]
                    },
                    "score": float(score)
                }
                predictions.append(polygon_result)
        
        return predictions

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """
        对输入的图像进行目标检测预测
        
        :param tasks: Label Studio任务列表
        :param context: 预测上下文信息（可选）
        :param kwargs: 其他参数
        :return: ModelResponse包含预测结果
        """
        import time
        
        # 预测前尝试热更新模型
        self._maybe_reload_model()
        
        if not tasks:
            print("警告: 收到空任务列表")
            return ModelResponse(predictions=[])
        
        start_time = time.time()
        results = []
        
        print(f"\n{'='*60}")
        print(f"开始处理 {len(tasks)} 个任务")
        if context:
            print(f"上下文信息: {context}")
        
        for task_idx, task in enumerate(tasks):
            # 获取图片URL
            image_url = task['data'].get('image')
            if not image_url:
                print(f"任务 {task.get('id', 'unknown')} 没有图像数据")
                results.append({"result": [], "score": 0})
                continue

            task_start_time = time.time()
            task_id = task.get('id', f'task_{task_idx}')

            try:
                print(f"\n[任务 {task_idx + 1}/{len(tasks)}] 处理任务 ID: {task_id}")
                print(f"图像URL: {image_url}")
                
                # 下载或获取本地图片路径
                try:
                    image_path = self.get_local_path(image_url, task_id=task_id)
                except Exception as path_error:
                    print(f"❌ 无法获取图像路径: {path_error}")
                    try:
                        model_version = self.get("model_version")
                    except:
                        model_version = "unknown"
                    results.append({
                        "model_version": model_version,
                        "model_path": self.model_path,
                        "result": [],
                        "score": 0,
                        "error": f"无法获取图像路径: {str(path_error)}"
                    })
                    continue
                
                if not os.path.exists(image_path):
                    print(f"❌ 图像文件不存在: {image_path}")
                    try:
                        model_version = self.get("model_version")
                    except:
                        model_version = "unknown"
                    results.append({
                        "model_version": model_version,
                        "model_path": self.model_path,
                        "result": [],
                        "score": 0,
                        "error": f"图像文件不存在: {image_path}"
                    })
                    continue
                
                print(f"本地图像路径: {image_path}")

                # 预处理
                image_tensor, original_size = self.preprocess(image_path)
                image_tensor = image_tensor.unsqueeze(0).to(self.device)  # 添加批次维度
                print(f"图像原始尺寸: {original_size}")
                print(f"预处理后张量形状: {image_tensor.shape}")
                if not getattr(self, "_prefer_orig_target_sizes_arg", True):
                    print("⚠️ 当前模型以单输入方式推理（未传入 orig_target_sizes），"
                          "建议重新导出/加载支持双输入的 TorchScript 以确保坐标精度。")
                
                # 推理
                with torch.no_grad():
                    # 与 predict_batch_torchscript.py 保持一致：始终构造 (width, height) 顺序的 orig_target_sizes
                    orig_target_sizes_tensor = torch.tensor(
                        [[original_size[0], original_size[1]]],
                        dtype=torch.int64,
                        device=self.device
                    )

                    def _forward_with_optional_sizes(img_tensor, size_tensor):
                        """
                        部署脚本导出的 TorchScript 模型通常需要同时传入 (images, orig_target_sizes)。
                        个别旧模型只接受单输入，此时会抛出 TypeError，我们自动回退。
                        """
                        if getattr(self, "_prefer_orig_target_sizes_arg", True):
                            try:
                                outputs_local = self.model(img_tensor, size_tensor)
                                self._prefer_orig_target_sizes_arg = True
                                self._requires_orig_target_sizes = True
                                return outputs_local
                            except TypeError:
                                print("模型不支持 orig_target_sizes 参数，改为使用单输入推理")
                                self._prefer_orig_target_sizes_arg = False
                        return self.model(img_tensor)

                    # RT-DETR模型可能需要orig_target_sizes参数
                    try:
                        outputs = _forward_with_optional_sizes(image_tensor, orig_target_sizes_tensor)
                    except RuntimeError as e:
                        error_str = str(e)
                        # 检测 CUDA 相关错误（无 GPU、无驱动、设备不匹配等）
                        cuda_errors = [
                            "Found no NVIDIA driver",
                            "CUDA",
                            "cuda",
                            "Expected all tensors to be on the same device",
                            "device type"
                        ]
                        is_cuda_error = any(err in error_str for err in cuda_errors)
                        
                        if is_cuda_error and not self.force_cpu:
                            # 检测到 CUDA 错误，切换到 CPU
                            print(f"⚠️  检测到 CUDA 相关错误: {error_str[:200]}")
                            print("   正在切换到 CPU 模式并重新加载模型...")
                            self.force_cpu = True
                            self.device = torch.device('cpu')
                            # 重新加载模型到 CPU
                            self._load_model(self.model_path, force_cpu=True)
                            # 将输入数据移到 CPU
                            image_tensor = image_tensor.cpu()
                            orig_target_sizes_tensor = orig_target_sizes_tensor.cpu()
                            # 重试推理
                            try:
                                outputs = _forward_with_optional_sizes(image_tensor, orig_target_sizes_tensor)
                            except RuntimeError as cpu_e:
                                # CPU 模式下仍然失败，可能是模型本身的问题
                                print(f"❌ CPU 模式下推理仍然失败: {cpu_e}")
                                raise RuntimeError(
                                    f"模型推理失败。模型可能是在 CUDA 上导出的 TorchScript，但当前系统没有 GPU。\n"
                                    f"建议：\n"
                                    f"1. 设置环境变量 FORCE_CPU_INFERENCE=true 强制使用 CPU\n"
                                    f"2. 使用在 CPU 上导出的模型版本\n"
                                    f"原始错误: {error_str[:500]}"
                                )
                        elif "orig_target_sizes" in error_str:
                            # 模型需要orig_target_sizes参数（通常不会走到这里，因为我们默认携带该参数）
                            print("模型明确要求 orig_target_sizes 参数，重新尝试传入")
                            self._prefer_orig_target_sizes_arg = True
                            self._requires_orig_target_sizes = True
                            try:
                                outputs = _forward_with_optional_sizes(image_tensor, orig_target_sizes_tensor)
                            except RuntimeError as dev_e:
                                # 处理TorchScript内部常量强制在CPU导致的设备不一致问题
                                if any(err in str(dev_e) for err in cuda_errors) and not self.force_cpu:
                                    print("检测到设备不一致，切换到 CPU 推理并重新加载模型")
                                    self.force_cpu = True
                                    self.device = torch.device('cpu')
                                    self._load_model(self.model_path, force_cpu=True)
                                    image_tensor = image_tensor.cpu()
                                    orig_target_sizes_tensor = orig_target_sizes_tensor.cpu()
                                    outputs = _forward_with_optional_sizes(image_tensor, orig_target_sizes_tensor)
                                else:
                                    raise
                        else:
                            # 其他错误，直接抛出
                            raise
                    
                    print(f"模型输出类型: {type(outputs)}")
                    
                    # 调试：打印输出结构
                    if isinstance(outputs, dict):
                        print(f"输出字典键: {outputs.keys()}")
                        for k, v in outputs.items():
                            if torch.is_tensor(v):
                                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                    elif isinstance(outputs, (tuple, list)):
                        print(f"输出元组/列表长度: {len(outputs)}")
                        for i, item in enumerate(outputs):
                            if torch.is_tensor(item):
                                print(f"  [{i}]: shape={item.shape}, dtype={item.dtype}")
                        
                        # 识别输出格式
                        if len(outputs) == 3:
                            if outputs[0].shape[-1] == 4:
                                print("识别为格式: (boxes, scores, labels)")
                            elif outputs[1].shape[-1] == 4:
                                print("识别为格式: (labels, boxes, scores)")
                            else:
                                print("警告: 无法识别输出格式")
                    
                    # 后处理
                    prediction_result = self.postprocess(outputs, original_size, image_tensor_size=image_tensor.shape)
                    
                    # 统计检测结果
                    num_detections = len(prediction_result)
                    task_time = time.time() - task_start_time
                    print(f"最终输出: {num_detections} 个目标 (耗时: {task_time:.2f}秒)")
                    
                    # 计算整体得分
                    score = 0
                    if prediction_result:
                        scores = [r.get("score", 0) for r in prediction_result]
                        score = max(scores)
                        avg_score = sum(scores) / len(scores) if scores else 0
                        print(f"置信度统计: 最高={score:.3f}, 平均={avg_score:.3f}")
                        
                    try:
                        model_version = self.get("model_version")
                    except:
                        model_version = "unknown"
                        
                    results.append({
                        "model_version": model_version,
                        "result": prediction_result,
                        "score": score,
                        "model_path": self.model_path
                    })
                    
            except Exception as e:
                import traceback
                print(f"❌ 处理任务 {task.get('id', 'unknown')} 时出错:")
                print(traceback.format_exc())
                
                # 获取模型版本（修复self.get()调用）
                try:
                    model_version = self.get("model_version")
                except:
                    model_version = "unknown"
                
                results.append({
                    "model_version": model_version,
                    "model_path": self.model_path,
                    "result": [], 
                    "score": 0,
                    "error": str(e)
                })

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"所有任务处理完成: {len(results)} 个结果")
        print(f"总耗时: {total_time:.2f}秒, 平均: {total_time/len(tasks):.2f}秒/任务")
        print(f"{'='*60}\n")

        return ModelResponse(predictions=results)

    def fit(self, event, data, **kwargs):
        """
        当标注被创建或更新时调用此方法，这个逻辑有问题，后续要修改，要在标注点击提交后才会有可能调用
        可以在这里实现模型训练逻辑
        
        注意：长时间运行的训练操作应该放在单独的进程或线程中执行
        :param event: 事件类型 ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: 从事件接收的数据负载
        """
        import json
        from datetime import datetime
        
        if not getattr(self, "enable_training", False):
            print(f"\n{'='*60}")
            print(f"训练功能已禁用（ENABLE_RTDETR_TRAINING=false），忽略事件: {event}")
            print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")
            return

        print(f"\n{'='*60}")
        print(f"收到训练事件: {event}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 只在特定事件时执行训练
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            print(f"不支持的事件类型: {event}，跳过处理")
            return
        
        # 获取当前模型版本
        try:
            current_model_version = self.get('model_version') or "rtdetr-custom-1.0"
        except:
            current_model_version = "rtdetr-custom-1.0"
        
        print(f"当前模型版本: {current_model_version}")
        
        # 准备训练数据存储目录
        train_data_dir = os.path.join(os.path.dirname(__file__), 'train_data')
        if not os.path.exists(train_data_dir):
            os.makedirs(train_data_dir)
        
        # 准备训练数据
        train_data = []
        task_id = data.get('id', 'unknown')
        
        # 从数据中提取标注信息（兼容不同负载格式）
        annotations = []
        
        # START_TRAINING 事件特殊处理：从缓存或项目中获取所有标注
        if event == 'START_TRAINING':
            print("处理 START_TRAINING 事件，从缓存中获取训练数据...")
            try:
                # 尝试从缓存中获取之前收集的训练数据
                import json as _json_for_cache
                cached_data = self.get('train_data_list')
                if cached_data:
                    if isinstance(cached_data, str):
                        try:
                            cached_data = _json_for_cache.loads(cached_data)
                        except Exception:
                            cached_data = []
                    if isinstance(cached_data, list) and len(cached_data) > 0:
                        print(f"从缓存中获取到 {len(cached_data)} 个训练样本")
                        # 直接使用缓存的数据，跳过标注提取步骤
                        train_data = cached_data.copy()
                        # 继续后续处理（统计、保存等）
                    else:
                        print("缓存中没有训练数据")
                else:
                    print("缓存中没有训练数据")
                    
                # 如果缓存中没有数据，尝试从项目数据中获取
                if not train_data and 'project' in data:
                    print("尝试从项目数据中获取标注...")
                    project_data = data.get('project', {})
                    # 项目数据可能包含任务列表
                    if 'tasks' in project_data:
                        tasks = project_data.get('tasks', [])
                        print(f"项目中有 {len(tasks)} 个任务")
                        # 从任务中提取标注
                        for task in tasks:
                            task_annotations = task.get('annotations', [])
                            if task_annotations:
                                annotations.extend(task_annotations)
                    elif 'id' in project_data:
                        # 如果有项目ID，可以尝试使用Label Studio SDK获取数据
                        print(f"项目ID: {project_data.get('id')}")
                        print("提示: 可以使用 Label Studio SDK 从项目中获取所有标注")
            except Exception as e:
                print(f"从缓存获取训练数据时出错: {e}")
                import traceback
                print(traceback.format_exc())
        else:
            # 对于 ANNOTATION_CREATED 和 ANNOTATION_UPDATED 事件，从数据中提取标注
            try:
                if isinstance(data.get('annotations'), list):
                    annotations = data.get('annotations') or []
                elif isinstance(data.get('annotation'), dict):
                    annotations = [data.get('annotation')]
                elif isinstance(data.get('result'), list):
                    # 顶层直接携带result数组
                    annotations = [{'result': data.get('result')}]
                elif 'task' in data:
                    task_block = data['task'] or {}
                    if isinstance(task_block.get('annotations'), list):
                        annotations = task_block.get('annotations') or []
                    elif isinstance(task_block.get('annotation'), dict):
                        annotations = [task_block.get('annotation')]
                    elif isinstance(task_block.get('result'), list):
                        annotations = [{'result': task_block.get('result')}]
            except Exception as e:
                print(f"提取标注信息时出错: {e}")
        
        # 如果还没有训练数据，尝试从annotations中提取
        if not train_data and annotations:
            print(f"从 {len(annotations)} 个标注中提取训练数据...")
        elif not train_data and not annotations:
            # 打印调试信息以帮助定位
            try:
                print("调试: 未找到 annotations，payload 关键字段：")
                print(f"  顶层键: {list(data.keys())}")
                if 'task' in data and isinstance(data['task'], dict):
                    print(f"  task 键: {list(data['task'].keys())}")
                    if 'data' in data['task'] and isinstance(data['task']['data'], dict):
                        print(f"  task.data 键: {list(data['task']['data'].keys())}")
                if 'data' in data and isinstance(data['data'], dict):
                    print(f"  data 键: {list(data['data'].keys())}")
                if 'project' in data and isinstance(data['project'], dict):
                    print(f"  project 键: {list(data['project'].keys())}")
            except Exception:
                pass
        
        print(f"收到 {len(annotations)} 个标注，已有 {len(train_data)} 个训练样本")
        
        # 处理标注数据（如果还没有从缓存获取）
        if not train_data:
            # 处理标注数据
            for annotation in annotations:
                # 获取标注结果
                results = annotation.get('result', [])
                if not results and isinstance(annotation, dict):
                    # 某些版本里结果在 annotation['results']
                    results = annotation.get('results', [])
                image_url = None
                image_path = None
                
                # 查找图像路径
                if 'data' in data:
                    image_url = data['data'].get('image') or data['data'].get('image_url')
                elif 'task' in data and 'data' in data['task']:
                    image_url = data['task']['data'].get('image') or data['task']['data'].get('image_url')
                # 兜底：若 annotation 层带有 task/data
                if not image_url and 'task' in annotation and isinstance(annotation['task'], dict):
                    task_data = annotation['task'].get('data', {})
                    if isinstance(task_data, dict):
                        image_url = task_data.get('image') or task_data.get('image_url')
                
                if image_url:
                    try:
                        image_path = self.get_local_path(image_url, task_id=task_id)
                    except Exception as e:
                        print(f"警告: 无法获取图像路径: {e}")
                        continue
                
                if not image_path or not os.path.exists(image_path):
                    print(f"警告: 图像文件不存在: {image_path}")
                    continue
                
                # 提取标注信息
                for result in results:
                    # 提取标注类型和值
                    if result.get('type') in ['rectanglelabels', 'polygonlabels']:
                        value = result.get('value', {})
                        labels = value.get('rectanglelabels') or value.get('polygonlabels', [])
                        
                        if not labels:
                            continue
                        
                        try:
                            # 提取边界框坐标
                            if result.get('type') == 'rectanglelabels':
                                x = value.get('x', 0)  # Percentage
                                y = value.get('y', 0)
                                width = value.get('width', 0)
                                height = value.get('height', 0)
                                
                                # 转换为绝对坐标
                                with Image.open(image_path) as img:
                                    img_width, img_height = img.size
                                
                                abs_x = x * img_width / 100
                                abs_y = y * img_height / 100
                                abs_width = width * img_width / 100
                                abs_height = height * img_height / 100
                                
                                # 转换为xyxy格式
                                x1, y1, x2, y2 = abs_x, abs_y, abs_x + abs_width, abs_y + abs_height
                                
                            elif result.get('type') == 'polygonlabels':
                                points = value.get('points', [])
                                if len(points) < 5:  # 至少需要3个点构成多边形
                                    continue
                                
                                # 计算包围盒
                                x_coords = [p[0] for p in points]
                                y_coords = [p[1] for p in points]
                                
                                with Image.open(image_path) as img:
                                    img_width, img_height = img.size
                                
                                # 转换为绝对坐标
                                x1 = min(x_coords) * img_width / 100
                                y1 = min(y_coords) * img_height / 100
                                x2 = max(x_coords) * img_width / 100
                                y2 = max(y_coords) * img_height / 100
                            
                            # 验证边界框有效性
                            if x2 <= x1 or y2 <= y1:
                                print(f"警告: 无效的边界框坐标: ({x1}, {y1}, {x2}, {y2})")
                                continue
                            
                            # 获取标签索引
                            label_name = labels[0]
                            if label_name in self.labels:
                                label_idx = self.labels.index(label_name)
                                
                                train_data.append({
                                    'image_path': image_path,
                                    'image_url': image_url,
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'label': int(label_idx),
                                    'label_name': label_name,
                                    'task_id': task_id,
                                    'annotation_id': annotation.get('id', 'unknown')
                                })
                            else:
                                print(f"警告: 未知标签 '{label_name}'，跳过")
                        except Exception as e:
                            print(f"警告: 处理标注时出错: {e}")
                            continue
        
        # 统计训练数据
        if len(train_data) > 0:
            # 按类别统计
            label_counts = {}
            for item in train_data:
                label_name = item['label_name']
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
            
            print(f"\n训练数据统计:")
            print(f"  总样本数: {len(train_data)}")
            print(f"  类别分布:")
            for label_name, count in sorted(label_counts.items()):
                print(f"    {label_name}: {count} 个")
            
            # 保存训练数据到JSON文件
            train_data_file = os.path.join(train_data_dir, f'train_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            try:
                # 保存训练数据，优先保存本地路径
                save_data = []
                for item in train_data:
                    save_item = item.copy()
                    # 优先保存本地路径（image_path），如果不存在则保存URL
                    # 这样训练时可以直接使用本地路径，避免路径解析问题
                    if 'image_path' in save_item and save_item['image_path'] and os.path.exists(save_item['image_path']):
                        # 保存本地路径
                        save_item['image_path'] = save_item['image_path']
                    elif 'image_url' in save_item and save_item['image_url']:
                        # 如果没有本地路径，保存URL（训练脚本会尝试解析）
                        save_item['image_path'] = save_item['image_url']
                    save_data.append(save_item)
                
                with open(train_data_file, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
                print(f"  训练数据已保存到: {train_data_file}")
            except Exception as e:
                print(f"警告: 保存训练数据失败: {e}")
            
            # 更新训练数据缓存
            try:
                import json as _json_for_cache
                existing_data = self.get('train_data_list')
                if isinstance(existing_data, str):
                    try:
                        existing_data = _json_for_cache.loads(existing_data)
                    except Exception:
                        existing_data = []
                elif not isinstance(existing_data, list):
                    existing_data = []
                existing_data.extend(train_data)
                # 将列表序列化为字符串缓存
                self.set('train_data_list', _json_for_cache.dumps(existing_data, ensure_ascii=False))
                # 计数转字符串
                self.set('train_data_count', str(len(existing_data)))
                print(f"  缓存中的总训练样本数: {len(existing_data)}")
            except Exception as e:
                print(f"警告: 更新训练数据缓存失败: {e}")
            
            # 检查是否需要启动训练
            min_samples_for_training = int(os.environ.get('MIN_SAMPLES_FOR_TRAINING', '100000000000000000000000000000000000000000'))
            
            # 是否对 START_TRAINING 事件也应用最小样本数限制
            # 默认：START_TRAINING 不受限制（手动触发，用户明确要训练）
            # 设置为 true：START_TRAINING 也需要达到最小样本数
            enforce_min_samples_for_manual = os.environ.get('ENFORCE_MIN_SAMPLES_FOR_MANUAL_TRAINING', 'false').lower() in ('1', 'true', 'yes')
            
            # 判断是否启动训练
            should_train = False
            if event == 'START_TRAINING':
                if enforce_min_samples_for_manual:
                    # 手动训练也需要达到最小样本数
                    if len(train_data) >= min_samples_for_training:
                        should_train = True
                        print(f"\n手动训练：样本数已达到最小要求")
                    else:
                        print(f"\n手动训练：样本数不足 ({len(train_data)} < {min_samples_for_training})，无法启动训练")
                        print("  提示: 设置 ENFORCE_MIN_SAMPLES_FOR_MANUAL_TRAINING=false 可允许手动训练不受限制")
                else:
                    # 手动训练不受限制（默认行为）
                    should_train = True
                    print(f"\n手动训练：不受最小样本数限制")
            elif len(train_data) >= min_samples_for_training:
                # 自动训练：需要达到最小样本数
                should_train = True
                print(f"\n自动训练：样本数已达到最小要求")
            
            if should_train:
                print(f"\n准备开始训练...")
                print(f"  最小训练样本数: {min_samples_for_training}")
                print(f"  当前样本数: {len(train_data)}")
                
                # 生成训练配置
                try:
                    train_config = {
                        'model_version': current_model_version,
                        'train_data_file': train_data_file,
                        'sample_count': len(train_data),
                        'label_counts': label_counts,
                        'timestamp': datetime.now().isoformat(),
                        'labels': self.labels
                    }
                    
                    config_file = os.path.join(train_data_dir, f'train_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(train_config, f, indent=2, ensure_ascii=False)
                    print(f"  训练配置已保存到: {config_file}")
                    
                    # 启动训练进程
                    self._start_training_process(
                        train_data_dir=train_data_dir,
                        config_file=config_file,
                        current_model_version=current_model_version
                    )
                    
                except Exception as e:
                    print(f"❌ 训练准备过程中出现错误: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
            else:
                # 不满足训练条件
                if event == 'START_TRAINING' and enforce_min_samples_for_manual:
                    # 手动训练但样本数不足（已在上面打印过提示）
                    pass
                else:
                    print(f"\n训练样本数不足 ({len(train_data)} < {min_samples_for_training})，跳过训练")
                    print("  继续收集训练数据...")
        else:
            print("没有提取到有效的训练数据，跳过处理")
        
        print(f"{'='*60}\n")
    
    def _start_training_process(self, train_data_dir: str, config_file: str, current_model_version: str):
        """
        启动训练进程（异步执行，避免阻塞主线程）
        
        :param train_data_dir: 训练数据目录
        :param config_file: 训练配置文件路径
        :param current_model_version: 当前模型版本
        """
        import subprocess
        import threading
        import sys
        from datetime import datetime
        
        def run_training():
            """在后台线程中运行训练"""
            try:
                # 获取训练脚本路径
                script_dir = os.path.dirname(__file__)
                train_script = os.path.join(script_dir, 'train_model.py')
                
                # 检查训练脚本是否存在
                if not os.path.exists(train_script):
                    print(f"❌ 训练脚本不存在: {train_script}")
                    print("  请确保 train_model.py 文件存在")
                    return
                
                # 获取模型输出目录
                model_output_dir = os.environ.get('MODEL_OUTPUT_DIR', 
                                                 os.path.join(script_dir, 'models'))
                os.makedirs(model_output_dir, exist_ok=True)
                
                # 构建训练命令
                labels_str = ','.join(self.labels)
                cmd = [
                    sys.executable,  # Python解释器
                    train_script,
                    '--data_dir', train_data_dir,
                    '--output_dir', model_output_dir,
                    '--labels', labels_str,
                    '--batch_size', os.environ.get('TRAIN_BATCH_SIZE', '8'),
                    '--num_epochs', os.environ.get('TRAIN_NUM_EPOCHS', '10'),
                    '--learning_rate', os.environ.get('TRAIN_LEARNING_RATE', '0.0001'),
                    '--config_file', config_file
                ]
                
                print(f"\n🚀 启动训练进程...")
                print(f"  训练脚本: {train_script}")
                print(f"  数据目录: {train_data_dir}")
                print(f"  输出目录: {model_output_dir}")
                print(f"  命令: {' '.join(cmd)}")
                
                # 创建日志文件
                log_dir = os.path.join(script_dir, 'train_logs')
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
                
                # 执行训练（在后台运行）
                with open(log_file, 'w', encoding='utf-8') as log:
                    process = subprocess.Popen(
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=script_dir,
                        env=os.environ.copy()
                    )
                
                # 保存训练进程ID
                self.set('training_process_id', str(process.pid))
                self.set('training_log_file', log_file)
                self.set('training_start_time', datetime.now().isoformat())
                
                print(f"  ✓ 训练进程已启动 (PID: {process.pid})")
                print(f"  📝 训练日志: {log_file}")
                print(f"  💡 提示: 训练在后台运行，可通过日志文件查看进度")
                
                # 等待训练完成（可选：如果需要在fit中等待）
                wait_for_completion = os.environ.get('WAIT_FOR_TRAINING', 'false').lower() == 'true'
                
                if wait_for_completion:
                    print(f"  等待训练完成...")
                    process.wait()
                    
                    if process.returncode == 0:
                        print(f"  ✓ 训练成功完成")
                        # 更新模型版本
                        self._update_model_after_training(current_model_version, model_output_dir)
                    else:
                        print(f"  ❌ 训练失败，返回码: {process.returncode}")
                        print(f"  请查看日志文件: {log_file}")
                else:
                    print(f"  ⚠️  训练在后台运行，不会等待完成")
                    print(f"  训练完成后，请手动更新模型或使用训练完成回调")
                
            except Exception as e:
                print(f"❌ 启动训练进程时出错: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        # 在后台线程中启动训练
        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()
        print(f"  ✓ 训练线程已启动")
    
    def _update_model_after_training(self, current_model_version: str, model_output_dir: str):
        """
        训练完成后更新模型
        
        :param current_model_version: 当前模型版本
        :param model_output_dir: 模型输出目录
        """
        try:
            from datetime import datetime
            import glob
            
            # 查找最新训练的模型文件
            model_files = glob.glob(os.path.join(model_output_dir, 'rtdetr_trained_*.pt'))
            if not model_files:
                # 也查找checkpoint文件
                model_files = glob.glob(os.path.join(model_output_dir, 'checkpoint_epoch_*.pt'))
            
            if model_files:
                # 按修改时间排序，获取最新的
                latest_model = max(model_files, key=os.path.getmtime)
                print(f"  找到训练后的模型: {latest_model}")
                
                # 更新模型版本
                model_version_parts = current_model_version.split('-')
                if len(model_version_parts) >= 2:
                    try:
                        version_num = float(model_version_parts[-1]) + 0.1
                        new_model_version = f"{'-'.join(model_version_parts[:-1])}-{version_num:.1f}"
                    except:
                        new_model_version = f"{current_model_version}-retrained"
                else:
                    new_model_version = f"{current_model_version}-retrained"
                
                # 保存新模型路径
                self.set('model_version', new_model_version)
                self.set('trained_model_path', latest_model)
                self.set('last_training_time', datetime.now().isoformat())
                
                print(f"  ✓ 模型版本已更新为: {new_model_version}")
                print(f"  📦 新模型路径: {latest_model}")
                print(f"  💡 提示: 请更新 docker-compose.yml 中的 MODEL_PATH 并重启服务")
            else:
                print(f"  ⚠️  未找到训练后的模型文件")
                
        except Exception as e:
            print(f"  ❌ 更新模型时出错: {str(e)}")
