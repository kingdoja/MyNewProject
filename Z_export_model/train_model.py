#!/usr/bin/env python3
"""
RT-DETR模型训练脚本
用于从Label Studio收集的训练数据中训练模型

使用方法:
    python train_model.py --data_dir train_data --output_dir models
"""

import os
import json
import argparse
import glob
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from PIL import Image
from datetime import datetime
import yaml

# 尝试导入 ultralytics
try:
    from ultralytics import RTDETR
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("警告: ultralytics 库未安装，请运行: pip install ultralytics>=8.0.0")


def resolve_image_path(raw_path: Optional[str],
                      path_map_from: Optional[str],
                      path_map_to: Optional[str],
                      download_images: bool,
                      cache_dir: str) -> str:
    """
    将 Label Studio 的图像路径解析为本地可访问路径：
    1) 解析 Label Studio 特殊路径格式（/data/local-files/?d=...）
    2) 若本地存在该路径，直接返回
    3) 若设置了路径映射前缀（TRAIN_PATH_MAP_FROM/TO），进行前缀替换后再判定
    4) 尝试相对于当前工作目录查找
    5) 尝试使用 Label Studio SDK 的 get_local_path 解析路径
    6) 若为URL且允许下载（TRAIN_DOWNLOAD_IMAGES=true），下载到缓存目录
    7) 否则抛出 FileNotFoundError
    """
    if not raw_path:
        raise FileNotFoundError("图像路径为空")
    
    # 处理 Label Studio 本地存储路径格式: /data/local-files/?d=实际路径
    # 例如: /data/local-files/?d=data/PatchesK265/patch_1023.png
    # 提取实际路径: /data/PatchesK265/patch_1023.png
    if '/data/local-files/?d=' in raw_path or '/data/local-files?d=' in raw_path:
        try:
            from urllib.parse import unquote, urlparse, parse_qs
            # 解析路径，提取 ?d= 后面的实际路径
            if '?d=' in raw_path:
                # 分离路径和查询参数
                path_part, query_part = raw_path.split('?d=', 1)
                # 解码 URL 编码的路径
                actual_path = unquote(query_part)
                # 如果路径不是绝对路径，添加 /data/ 前缀
                if not actual_path.startswith('/'):
                    actual_path = '/' + actual_path
                # 如果路径以 /data/ 开头，直接使用
                if actual_path.startswith('/data/'):
                    raw_path = actual_path
                else:
                    # 否则添加 /data/ 前缀
                    raw_path = '/data/' + actual_path.lstrip('/')
                print(f"解析 Label Studio 路径: {raw_path.split('?d=')[0] if '?d=' in raw_path else raw_path} -> {raw_path}")
        except Exception as e:
            print(f"警告: 解析 Label Studio 路径格式时出错: {e}")
    
    # 直接存在
    if os.path.exists(raw_path):
        return os.path.abspath(raw_path)
    
    # 路径前缀映射（例如 /data/upload -> /root/label-studio/data/upload）
    if path_map_from and path_map_to and raw_path.startswith(path_map_from):
        mapped = raw_path.replace(path_map_from, path_map_to, 1)
        if os.path.exists(mapped):
            return os.path.abspath(mapped)
    
    # 尝试相对于当前工作目录查找（非 Docker 环境）
    # 如果路径以 /data/upload 开头，可能需要在当前工作目录或父目录中查找
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_bases = [
        os.getcwd(),  # 当前工作目录
        script_dir,  # 脚本所在目录
        os.path.dirname(script_dir),  # 父目录
        os.path.expanduser('~'),  # 用户主目录
    ]
    
    # 如果是绝对路径，尝试在不同的基础路径下查找
    if raw_path.startswith('/'):
        # 提取路径的相对部分（例如 /data/upload/26/file.png -> upload/26/file.png）
        path_parts = raw_path.strip('/').split('/')
        if len(path_parts) >= 2:
            # 尝试在可能的基础路径下查找
            for base in possible_bases:
                # 尝试不同的路径组合
                possible_paths = [
                    os.path.join(base, *path_parts),  # 完整路径
                    os.path.join(base, path_parts[-2], path_parts[-1]),  # 最后两级
                    os.path.join(base, 'data', *path_parts[1:]),  # data/upload/...
                    os.path.join(base, 'label-studio', 'data', *path_parts[1:]),  # label-studio/data/upload/...
                ]
                for possible_path in possible_paths:
                    if os.path.exists(possible_path):
                        return os.path.abspath(possible_path)
    
    # 尝试使用 Label Studio SDK 的 get_local_path 解析路径
    # 这对于 Label Studio 的路径格式（如 /data/upload/...）很有用
    try:
        from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
        from label_studio_sdk._extensions.label_studio_tools.core.utils.params import get_env
        
        # 尝试解析 Label Studio 路径
        ls_host = get_env('HOSTNAME') or get_env('LABEL_STUDIO_URL') or os.environ.get('LABEL_STUDIO_URL')
        ls_token = get_env('LABEL_STUDIO_API_KEY') or get_env('API_KEY') or os.environ.get('LABEL_STUDIO_API_KEY')
        
        # 确保 hostname 是完整的 URL（包含协议）
        if ls_host and not ls_host.startswith(('http://', 'https://')):
            # 如果没有协议，尝试添加 http://
            # 但这里不自动添加，因为可能是主机名而不是 URL
            # 如果只是主机名，Label Studio SDK 可能会失败，我们跳过 SDK 解析
            pass
        
        if ls_host and ls_host.startswith(('http://', 'https://')):
            try:
                resolved_path = get_local_path(
                    url=raw_path,
                    hostname=ls_host,
                    access_token=ls_token,
                    task_id=None
                )
                
                if resolved_path and os.path.exists(resolved_path):
                    return os.path.abspath(resolved_path)
            except Exception as sdk_error:
                # SDK 解析失败，继续尝试其他方法
                print(f"警告: 使用 Label Studio SDK 解析路径失败: {sdk_error}")
        elif ls_host:
            # hostname 不是完整 URL，跳过 SDK 解析
            print(f"提示: LABEL_STUDIO_URL 需要完整 URL（包含 http:// 或 https://），当前值: {ls_host}")
    except ImportError:
        # Label Studio SDK 不可用，跳过
        pass
    except Exception as e:
        # 解析失败，继续尝试其他方法
        print(f"警告: 使用 Label Studio SDK 解析路径失败: {e}")
    
    # URL 下载
    if download_images and (raw_path.startswith('http://') or raw_path.startswith('https://')):
        try:
            import hashlib
            import urllib.request
            os.makedirs(cache_dir, exist_ok=True)
            fname = hashlib.md5(raw_path.encode('utf-8')).hexdigest() + '.img'
            dst = os.path.join(cache_dir, fname)
            if not os.path.exists(dst):
                print(f"下载图像: {raw_path} -> {dst}")
                urllib.request.urlretrieve(raw_path, dst)
            return os.path.abspath(dst)
        except Exception as e:
            raise FileNotFoundError(f"无法下载图像: {raw_path}, 错误: {e}")
    
    # 提示可能的修复方式
    error_msg = (
        f"图像文件不存在: {raw_path}\n"
        f"已尝试的路径:\n"
        f"  - 原始路径: {raw_path}\n"
    )
    
    if path_map_from and path_map_to:
        error_msg += f"  - 映射路径: {raw_path.replace(path_map_from, path_map_to, 1)}\n"
    
    error_msg += (
        f"\n请确认:\n"
        f"- 若Label Studio在不同环境/挂载路径，请设置 TRAIN_PATH_MAP_FROM 与 TRAIN_PATH_MAP_TO 进行路径映射\n"
        f"  例如: TRAIN_PATH_MAP_FROM=/data/upload TRAIN_PATH_MAP_TO=/root/label-studio/data/upload\n"
        f"- 或设置 TRAIN_DOWNLOAD_IMAGES=true 并确保提供可访问的URL\n"
        f"- 或确保 Label Studio SDK 已安装且环境变量 HOSTNAME 和 LABEL_STUDIO_API_KEY 已正确设置\n"
        f"- 或确保图像文件在可访问的路径下\n"
    )
    
    raise FileNotFoundError(error_msg)


def convert_to_yolo_format(train_data: List[Dict], labels: List[str], 
                          dataset_dir: str, split_ratio: float = 0.8) -> Tuple[str, str]:
    """
    将训练数据转换为 YOLO 格式
    
    :param train_data: 训练数据列表
    :param labels: 类别标签列表
    :param dataset_dir: 数据集目录
    :param split_ratio: 训练集比例
    :return: (train_dir, val_dir) 训练集和验证集目录
    """
    # 创建目录结构
    train_images_dir = os.path.join(dataset_dir, 'train', 'images')
    train_labels_dir = os.path.join(dataset_dir, 'train', 'labels')
    val_images_dir = os.path.join(dataset_dir, 'val', 'images')
    val_labels_dir = os.path.join(dataset_dir, 'val', 'labels')
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 路径解析配置
    path_map_from = os.environ.get('TRAIN_PATH_MAP_FROM')
    path_map_to = os.environ.get('TRAIN_PATH_MAP_TO')
    download_images = os.environ.get('TRAIN_DOWNLOAD_IMAGES', 'false').lower() in ('1', 'true', 'yes')
    cache_dir = os.environ.get('TRAIN_IMAGE_CACHE_DIR', os.path.join(os.path.dirname(__file__), 'train_cache'))
    
    # 按图像路径分组（一个图像可能有多个标注）
    from collections import defaultdict
    image_groups = defaultdict(list)
    for sample in train_data:
        image_path = sample.get('image_path') or sample.get('image_url')
        if image_path:
            image_groups[image_path].append(sample)
    
    print(f"总共 {len(image_groups)} 张图像, {len(train_data)} 个标注")
    
    # 打乱图像顺序
    import random
    image_list = list(image_groups.items())
    random.shuffle(image_list)
    
    # 分割训练集和验证集
    split_idx = int(len(image_list) * split_ratio)
    train_images = image_list[:split_idx]
    val_images = image_list[split_idx:]
    
    print(f"数据集划分: 训练集 {len(train_images)} 张图像, 验证集 {len(val_images)} 张图像")
    
    # 处理训练集
    train_count = 0
    for image_path, samples in train_images:
        try:
            # 解析图像路径
            resolved_path = resolve_image_path(
                image_path,
                path_map_from,
                path_map_to,
                download_images,
                cache_dir
            )
            
            if not os.path.exists(resolved_path):
                print(f"警告: 图像文件不存在，跳过: {resolved_path}")
                continue
            
            # 读取图像尺寸
            with Image.open(resolved_path) as img:
                img_width, img_height = img.size
            
            # 复制图像到训练集目录
            image_filename = f"train_{train_count:06d}{os.path.splitext(resolved_path)[1]}"
            dst_image_path = os.path.join(train_images_dir, image_filename)
            shutil.copy2(resolved_path, dst_image_path)
            
            # 创建标注文件（YOLO 格式）- 一个图像可能有多个标注
            label_filename = f"train_{train_count:06d}.txt"
            label_path = os.path.join(train_labels_dir, label_filename)
            
            # 写入所有标注
            with open(label_path, 'w') as f:
                for sample in samples:
                    # 获取边界框和标签
                    bbox = sample['bbox']  # [x1, y1, x2, y2]
                    label_idx = sample['label']
                    
                    # 转换为 YOLO 格式 (归一化的中心点坐标和宽高)
                    x1, y1, x2, y2 = bbox
                    x_center = (x1 + x2) / 2.0 / img_width
                    y_center = (y1 + y2) / 2.0 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # 确保坐标在 [0, 1] 范围内
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    # 验证边界框有效性
                    if width > 0 and height > 0:
                        f.write(f"{label_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            train_count += 1
            
        except Exception as e:
            print(f"警告: 处理训练图像时出错: {e}")
            import traceback
            print(traceback.format_exc())
            continue
    
    # 处理验证集
    val_count = 0
    for image_path, samples in val_images:
        try:
            # 解析图像路径
            resolved_path = resolve_image_path(
            image_path,
                path_map_from,
                path_map_to,
                download_images,
                cache_dir
            )
            
            if not os.path.exists(resolved_path):
                print(f"警告: 图像文件不存在，跳过: {resolved_path}")
                continue
            
            # 读取图像尺寸
            with Image.open(resolved_path) as img:
                img_width, img_height = img.size
            
            # 复制图像到验证集目录
            image_filename = f"val_{val_count:06d}{os.path.splitext(resolved_path)[1]}"
            dst_image_path = os.path.join(val_images_dir, image_filename)
            shutil.copy2(resolved_path, dst_image_path)
            
            # 创建标注文件（YOLO 格式）- 一个图像可能有多个标注
            label_filename = f"val_{val_count:06d}.txt"
            label_path = os.path.join(val_labels_dir, label_filename)
            
            # 写入所有标注
            with open(label_path, 'w') as f:
                for sample in samples:
                    # 获取边界框和标签
                    bbox = sample['bbox']  # [x1, y1, x2, y2]
                    label_idx = sample['label']
                    
                    # 转换为 YOLO 格式
                    x1, y1, x2, y2 = bbox
                    x_center = (x1 + x2) / 2.0 / img_width
                    y_center = (y1 + y2) / 2.0 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # 确保坐标在 [0, 1] 范围内
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    # 验证边界框有效性
                    if width > 0 and height > 0:
                        f.write(f"{label_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            val_count += 1
            
        except Exception as e:
            print(f"警告: 处理验证图像时出错: {e}")
            import traceback
            print(traceback.format_exc())
            continue
    
    print(f"成功转换: 训练集 {train_count} 个样本, 验证集 {val_count} 个样本")
    
    # 创建数据集配置文件
    dataset_config = {
        'path': os.path.abspath(dataset_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(labels),
        'names': labels
    }
    
    config_path = os.path.join(dataset_dir, 'dataset.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"数据集配置已保存到: {config_path}")
    
    return os.path.join(dataset_dir, 'train'), os.path.join(dataset_dir, 'val')


def train_rtdetr_model(dataset_config: str, output_dir: str, 
                      epochs: int = 10, batch_size: int = 8, 
                      imgsz: int = 640, device: str = 'cuda',
                      model_size: str = 'l', custom_model_path: str = None) -> str:
    """
    训练 RT-DETR 模型
    
    :param dataset_config: 数据集配置文件路径
    :param output_dir: 输出目录
    :param epochs: 训练轮数
    :param batch_size: 批次大小
    :param imgsz: 图像尺寸
    :param device: 设备 ('cuda' 或 'cpu')
    :param model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
    :param custom_model_path: 自定义模型路径（可选）
    :return: 训练后的模型路径
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("ultralytics 库未安装，请运行: pip install ultralytics>=8.0.0")
    
    # 检查设备
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU 训练")
        device = 'cpu'
    
    print("=" * 60)
    print("开始训练 RT-DETR 模型")
    print("=" * 60)
    print(f"数据集配置: {dataset_config}")
    print(f"输出目录: {output_dir}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"图像尺寸: {imgsz}")
    print(f"设备: {device}")
    print(f"模型大小: {model_size}")
    print(f"自定义模型路径: {custom_model_path}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化 RT-DETR 模型
    # 优先使用自定义模型，如果未提供则使用预训练模型
    if custom_model_path and os.path.exists(custom_model_path):
        print(f"加载自定义模型: {custom_model_path}")
        model = RTDETR(custom_model_path)
    else:
        # 使用预训练模型，模型大小可选: n, s, m, l, x
        model_name = f'rtdetr{model_size}.pt'
        print(f"加载预训练模型: {model_name}")
        
        try:
            model = RTDETR(model_name)
        except Exception as e:
            print(f"警告: 无法加载预训练模型 {model_name}，尝试使用默认模型: {e}")
            # 如果无法加载，尝试使用基础模型
            model = RTDETR('rtdetr-l.pt')
    
    # 训练模型
    print("\n开始训练...")
    results = model.train(
        data=dataset_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=output_dir,
        name='train',
        save=True,
        save_period=5,  # 每5个epoch保存一次检查点
        val=True,  # 启用验证
        plots=True,  # 生成训练图表
        verbose=True,  # 显示详细输出
    )
    
    # 获取训练后的模型路径
    trained_model_path = os.path.join(output_dir, 'train', 'weights', 'best.pt')
    if not os.path.exists(trained_model_path):
        # 如果没有 best.pt，尝试使用 last.pt
        trained_model_path = os.path.join(output_dir, 'train', 'weights', 'last.pt')
    
    if not os.path.exists(trained_model_path):
        raise FileNotFoundError(f"训练后的模型未找到: {trained_model_path}")
    
    print(f"\n训练完成！模型保存在: {trained_model_path}")
    
    # 导出为 TorchScript 格式（用于推理）
    print("\n导出为 TorchScript 格式...")
    export_path = os.path.join(output_dir, f'rtdetr_trained_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
    
    try:
        # 加载训练后的模型
        model = RTDETR(trained_model_path)
        # 导出为 TorchScript
        # ultralytics 导出后会在模型目录生成 .torchscript 文件
        export_result = model.export(format='torchscript', imgsz=imgsz, simplify=True)
        
        # ultralytics 的 export 方法返回导出文件的路径
        if export_result and os.path.exists(export_result):
            # 复制到输出目录
            shutil.copy2(export_result, export_path)
            print(f"TorchScript 模型已导出到: {export_path}")
        else:
            # 尝试查找导出的文件
            export_dir = os.path.dirname(trained_model_path)
            # ultralytics 可能导出为 .torchscript 或 .pt 文件
            exported_files = (
                glob.glob(os.path.join(export_dir, '*.torchscript')) +
                glob.glob(os.path.join(export_dir, '*_torchscript.pt'))
            )
            if exported_files:
                # 使用最新的导出文件
                latest_export = max(exported_files, key=os.path.getmtime)
                shutil.copy2(latest_export, export_path)
                print(f"TorchScript 模型已导出到: {export_path}")
            else:
                # 如果导出失败，直接使用原始模型
                print("警告: 无法找到导出的 TorchScript 文件，使用原始模型")
                shutil.copy2(trained_model_path, export_path)
    except Exception as e:
        print(f"警告: 导出 TorchScript 时出错: {e}")
        print("使用原始模型文件")
        import traceback
        print(traceback.format_exc())
        # 直接使用训练后的模型
        shutil.copy2(trained_model_path, export_path)
    
    print(f"\n最终模型文件: {export_path}")
    return export_path


def load_training_data(data_dir: str) -> List[Dict]:
    """
    从目录中加载所有训练数据
    
    :param data_dir: 训练数据目录
    :return: 训练数据列表
    """
    data_files = glob.glob(os.path.join(data_dir, 'train_data_*.json'))
    data_files.sort()  # 按时间排序
    
    if not data_files:
        raise ValueError(f"在 {data_dir} 中未找到训练数据文件")
    
    print(f"找到 {len(data_files)} 个训练数据文件")
    
    # 加载所有训练数据
    all_data = []
    for data_file in data_files:
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    print(f"警告: 文件 {data_file} 格式不正确，跳过")
        except Exception as e:
            print(f"警告: 无法加载文件 {data_file}: {e}")
    
    print(f"总共加载了 {len(all_data)} 个训练样本")
    return all_data


def main():
    parser = argparse.ArgumentParser(description='RT-DETR模型训练脚本')
    parser.add_argument('--data_dir', type=str, default='train_data',
                       help='训练数据目录')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='模型输出目录')
    parser.add_argument('--labels', type=str,
                       default='AD,BC,EC,L,LC,M,NT,SM,SQ,TC1,TC2,TC3',
                       help='类别标签（逗号分隔）')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--image_size', type=int, default=640,
                       help='图像尺寸')
    parser.add_argument('--model_size', type=str, default='l',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='模型大小 (n=nanos, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='训练配置文件路径（可选）')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='训练设备')
    parser.add_argument('--dataset_dir', type=str, default=None,
                       help='YOLO格式数据集目录（如果提供，跳过数据转换）')
    parser.add_argument('--custom_model', type=str, default=None,
                       help='自定义模型路径（可选，如果提供则使用该模型进行训练）')
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，从中读取参数
    if args.config_file and os.path.exists(args.config_file):
        print(f"从配置文件加载参数: {args.config_file}")
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            if 'labels' in config:
                args.labels = ','.join(config['labels'])
            print(f"  模型版本: {config.get('model_version', 'unknown')}")
            print(f"  样本数量: {config.get('sample_count', 0)}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析标签
    labels = [label.strip() for label in args.labels.split(',')]
    print(f"类别标签: {labels}")
    print(f"类别数量: {len(labels)}")
    
    # 加载训练数据
    print("=" * 60)
    print("加载训练数据...")
    train_data = load_training_data(args.data_dir)
    
    if len(train_data) == 0:
        raise ValueError("没有找到训练数据，请先收集训练数据")
    
    # 检查数据格式
    print(f"\n训练数据统计:")
    label_counts = {}
    for item in train_data:
        label_name = item.get('label_name', 'unknown')
        label_counts[label_name] = label_counts.get(label_name, 0) + 1
    
    for label_name, count in sorted(label_counts.items()):
        print(f"  {label_name}: {count} 个样本")
    
    # 转换为 YOLO 格式
    if args.dataset_dir and os.path.exists(args.dataset_dir):
        print(f"\n使用现有的数据集目录: {args.dataset_dir}")
        dataset_config = os.path.join(args.dataset_dir, 'dataset.yaml')
    else:
        print("\n转换训练数据为 YOLO 格式...")
        dataset_dir = os.path.join(args.output_dir, 'dataset')
        train_dir, val_dir = convert_to_yolo_format(train_data, labels, dataset_dir)
        dataset_config = os.path.join(dataset_dir, 'dataset.yaml')
    
    # 训练模型
    print("\n" + "=" * 60)
    try:
        trained_model_path = train_rtdetr_model(
            dataset_config=dataset_config,
            output_dir=args.output_dir,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            imgsz=args.image_size,
            device=args.device,
            model_size=args.model_size,
            custom_model_path=args.custom_model
        )
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        print(f"训练后的模型: {trained_model_path}")
        print(f"\n下一步:")
        print(f"1. 更新 MODEL_PATH 环境变量指向新模型")
        print(f"2. 重启 ML 后端服务")
        print(f"3. 或设置 AUTO_RELOAD_MODEL=true 自动加载新模型")
        print("=" * 60)
        
        # 保存训练信息
        train_info = {
            'model_path': trained_model_path,
            'dataset_config': dataset_config,
            'labels': labels,
            'num_samples': len(train_data),
            'epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'image_size': args.image_size,
            'model_size': args.model_size,
            'train_time': datetime.now().isoformat()
        }
        
        info_path = os.path.join(args.output_dir, 'train_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(train_info, f, indent=2, ensure_ascii=False)
        print(f"\n训练信息已保存到: {info_path}")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
