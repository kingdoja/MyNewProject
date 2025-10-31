#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
痰液细胞检测专用训练脚本
针对小目标检测进行了优化
"""

import sys
import os
import torch
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 添加项目根目录到Python路径，解决相对导入问题
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 设置环境变量以更好地管理GPU内存
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 用于调试

def setup_training_environment():
    """设置训练环境"""
    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # 设置CUDA设备
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告: 未检测到GPU，将使用CPU训练")

def create_argument_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='痰液细胞检测训练脚本')
    
    # 必需参数
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('-d', '--device', type=str, default='cuda:0',
                       help='训练设备 (default: cuda:0)')
    
    # 可选参数
    parser.add_argument('-r', '--resume', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('-t', '--tuning', type=str, default=None,
                       help='从预训练模型微调')
    parser.add_argument('--no-tuning', action='store_true', default=False,
                       help='不使用任何预训练权重进行微调')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (default: 42)')
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='使用自动混合精度训练')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--summary-dir', type=str, default=None,
                       help='TensorBoard摘要目录')
    parser.add_argument('--test-only', action='store_true', default=False,
                       help='仅进行测试')
    # 训练工具所需参数（避免缺失导致报错）
    parser.add_argument('--print-method', type=str, default='builtin',
                       help='打印方法')
    parser.add_argument('--print-rank', type=int, default=0,
                       help='允许打印信息的rank id')
    parser.add_argument('--local-rank', type=int, default=0,
                       help='分布式训练的本地rank id')
    parser.add_argument('-u', '--update', nargs='+', default=None,
                       help='更新YAML配置')
    
    # 痰液细胞检测特定参数
    parser.add_argument('--small-object-mode', action='store_true', default=True,
                       help='启用小目标检测模式')
    parser.add_argument('--focal-loss-alpha', type=float, default=0.75,
                       help='Focal Loss alpha参数')
    parser.add_argument('--focal-loss-gamma', type=float, default=2.0,
                       help='Focal Loss gamma参数')
    parser.add_argument('--num-queries', type=int, default=500,
                       help='查询数量')
    parser.add_argument('--batch-size', type=int, default=6,
                       help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=0.0002,
                       help='学习率')
    parser.add_argument('--epochs', type=int, default=150,
                       help='训练轮数')
    
    return parser

def is_supported_checkpoint(path: str) -> bool:
    """检查checkpoint是否为受支持的结构。
    受支持: state['ema']['module'] 或 state['model'] 或 纯state_dict(全是Tensor)。
    """
    try:
        state = torch.load(path, map_location='cpu')
    except Exception:
        return False

    if isinstance(state, dict):
        if 'ema' in state and isinstance(state['ema'], dict) and 'module' in state['ema'] and isinstance(state['ema']['module'], dict):
            return True
        if 'model' in state and isinstance(state['model'], dict):
            return True
        if state and all(isinstance(v, torch.Tensor) for v in state.values()):
            return True
    return False

def update_config_for_sputum_cell(args, config_updates):
    """为痰液细胞检测更新配置"""
    if args.small_object_mode:
        config_updates.extend([
            f"RTDETRTransformerv2.num_queries={args.num_queries}",
            f"train_dataloader.total_batch_size={args.batch_size}",
            f"optimizer.lr={args.learning_rate}",
            f"epoches={args.epochs}",
            f"RTDETRCriterionv2.alpha={args.focal_loss_alpha}",
            f"RTDETRCriterionv2.gamma={args.focal_loss_gamma}",
        ])
    
    return config_updates

# 直接运行训练工具
if __name__ == "__main__":
    # 设置命令行参数
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 设置训练环境
    setup_training_environment()
    
    # 检查CUDA设备
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告: 未检测到GPU，将使用CPU训练")
    
    # 加载配置文件
    try:
        # 导入YAMLConfig和train_main
        from src.core import YAMLConfig
        from tools.train import main as train_main
        cfg = YAMLConfig(args.config, use_amp=args.use_amp)
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 验证配置
    # 修复：通过配置文件获取num_classes而不是直接访问属性
    num_classes = getattr(cfg, 'num_classes', None)
    if num_classes is None:
        # 尝试从配置的其他部分获取num_classes
        if hasattr(cfg, 'model') and cfg.model is not None:
            if hasattr(cfg.model, 'num_classes'):
                num_classes = cfg.model.num_classes
        # 如果还是没有找到，尝试从配置文件中查找
        if num_classes is None:
            print("警告: 未找到num_classes配置，将使用默认值12")
            num_classes = 12
    
    # 确保配置中的num_classes正确传递给模型
    if hasattr(cfg, 'model') and cfg.model is not None:
        cfg.model.num_classes = num_classes
    
    print(f"配置文件: {args.config}")
    print(f"设备: {args.device}")
    # 修复：安全地获取批次大小
    batch_size = getattr(cfg, 'batch_size', '未知')
    print(f"批次大小: {batch_size}")
    # 修复：安全地获取学习率
    lr = '未知'
    if hasattr(cfg, 'optimizer') and cfg.optimizer is not None:
        if isinstance(cfg.optimizer, dict) and 'lr' in cfg.optimizer:
            lr = cfg.optimizer['lr']
        elif hasattr(cfg.optimizer, 'defaults'):
            lr = cfg.optimizer.defaults.get('lr', '未知')
        elif hasattr(cfg.optimizer, 'param_groups'):
            lr = cfg.optimizer.param_groups[0].get('lr', '未知') if cfg.optimizer.param_groups else '未知'
    print(f"学习率: {lr}")
    # 修复：安全地获取训练轮数
    epoches = getattr(cfg, 'epoches', '未知')
    print(f"训练轮数: {epoches}")
    # 修复：安全地获取查询数量
    num_queries = '未知'
    if hasattr(cfg, 'model') and cfg.model is not None:
        if hasattr(cfg.model, 'decoder') and cfg.model.decoder is not None:
            if hasattr(cfg.model.decoder, 'num_queries'):
                num_queries = cfg.model.decoder.num_queries
    print(f"查询数量: {num_queries}")
    # 修复：安全地获取输出目录
    output_dir = getattr(cfg, 'output_dir', '未知')
    print(f"输出目录: {output_dir}")
    print(f"小目标检测模式: {getattr(args, 'small_object_mode', False)}")
    print(f"类别数: {num_classes}")
    
    # 启用详细CUDA错误信息
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    # 执行训练
    try:
        # 确保类别数正确传递给训练参数
        if args.update is None:
            args.update = []
        args.update.append(f"num_classes={num_classes}")
        train_main(args)
    except RuntimeError as e:
        if "CUDA error: device-side assert triggered" in str(e):
            print("\n" + "="*60)
            print("CUDA设备端断言错误")
            print("可能的原因:")
            print("1. 数据集中的类别标签ID超出模型配置的类别数范围")
            print("2. 数据集标注文件中存在无效的类别ID")
            print("3. 配置文件中的num_classes参数设置不正确")
            print("\n解决方法:")
            print("1. 检查训练数据集标注文件中的类别ID范围")
            print("2. 确保配置文件中的num_classes参数正确")
            print("3. 确保类别ID从0开始连续编号")
            print("="*60)
        raise e
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    sys.exit(0)