"""
将训练好的RT-DETR模型导出为.pt格式
支持多种导出模式：
1. state_dict: 只导出模型权重（推荐，体积小）
2. full_model: 导出完整模型
3. torchscript: 导出TorchScript格式（用于推理部署）
"""

import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch
import torch.nn as nn 

from src.core import YAMLConfig


# 全局DeployModel类（用于deploy和torchscript模式）
class DeployModel(nn.Module):
    """部署模型类，包含模型和后处理器"""
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


def export_state_dict(cfg, checkpoint, output_file):
    """
    导出模型权重（state_dict）
    这是最常用的方式，体积小，便于加载
    """
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
        print('使用EMA模型权重')
    else:
        state = checkpoint['model']
        print('使用标准模型权重')
    
    # 保存state_dict
    torch.save(state, output_file)
    print(f'✅ 模型权重已保存到: {output_file}')
    print(f'文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB')


def export_full_model(cfg, checkpoint, output_file):
    """
    导出完整模型（包括模型结构和权重）
    """
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
        print('使用EMA模型权重')
    else:
        state = checkpoint['model']
        print('使用标准模型权重')
    
    # 加载权重到模型
    cfg.model.load_state_dict(state)
    cfg.model.eval()
    
    # 保存完整模型
    torch.save(cfg.model, output_file)
    print(f'✅ 完整模型已保存到: {output_file}')
    print(f'文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB')


def export_torchscript(cfg, checkpoint, output_file, input_size=640):
    """
    导出TorchScript格式模型
    这种格式可以在C++等环境中使用，适合生产部署
    """
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
        print('使用EMA模型权重')
    else:
        state = checkpoint['model']
        print('使用标准模型权重')
    
    # 加载权重到模型并转换为deploy模式
    cfg.model.load_state_dict(state)
    
    deploy_model = DeployModel(cfg.model, cfg.postprocessor)
    deploy_model.eval()
    
    # 创建示例输入
    example_images = torch.rand(1, 3, input_size, input_size)
    example_sizes = torch.tensor([[input_size, input_size]])
    
    # 测试模型
    print('测试模型推理...')
    with torch.no_grad():
        _ = deploy_model(example_images, example_sizes)
    print('模型测试通过')
    
    # 导出为TorchScript
    print('正在转换为TorchScript格式...')
    traced_model = torch.jit.trace(deploy_model, (example_images, example_sizes))
    traced_model.save(output_file)
    
    print(f'✅ TorchScript模型已保存到: {output_file}')
    print(f'文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB')


def export_deploy_model(cfg, checkpoint, output_file):
    """
    导出部署模式的完整模型（包含后处理）
    注意：这个模式现在使用state_dict保存，使用时需要重新构建模型
    """
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
        print('使用EMA模型权重')
    else:
        state = checkpoint['model']
        print('使用标准模型权重')
    
    # 加载权重到模型
    cfg.model.load_state_dict(state)
    
    deploy_model = DeployModel(cfg.model, cfg.postprocessor)
    deploy_model.eval()
    
    # 测试模型
    print('测试模型推理...')
    example_images = torch.rand(1, 3, 640, 640)
    example_sizes = torch.tensor([[640, 640]])
    with torch.no_grad():
        _ = deploy_model(example_images, example_sizes)
    print('模型测试通过')
    
    # 保存模型的state_dict和配置
    save_dict = {
        'model_state_dict': deploy_model.state_dict(),
        'model_class': 'DeployModel',
        'info': '这是一个部署就绪的模型，包含模型和后处理器'
    }
    torch.save(save_dict, output_file)
    print(f'✅ 部署模型已保存到: {output_file}')
    print(f'文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB')
    print('\n⚠️  注意: deploy模式保存的是state_dict，推荐使用torchscript模式用于生产部署')


def main(args):
    """
    主函数
    """
    print('='*60)
    print('RT-DETR模型导出工具')
    print('='*60)
    
    # 加载配置
    print(f'\n📝 加载配置文件: {args.config}')
    print(f'📦 加载检查点: {args.resume}')
    
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    # 加载checkpoint
    if not args.resume:
        raise AttributeError('必须提供--resume参数来指定训练好的模型checkpoint路径')
    
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f'找不到checkpoint文件: {args.resume}')
    
    checkpoint = torch.load(args.resume, map_location='cpu')
    print(f'✅ Checkpoint加载成功')
    
    # 显示checkpoint信息
    if 'epoch' in checkpoint:
        print(f'📊 训练轮数: {checkpoint["epoch"]}')
    if 'ema' in checkpoint:
        print('✨ 包含EMA权重')
    if 'model' in checkpoint:
        print('✨ 包含模型权重')
    
    # 确定输出文件路径
    if args.output:
        output_file = args.output
    else:
        # 自动生成输出文件名
        base_name = os.path.splitext(os.path.basename(args.resume))[0]
        if args.mode == 'state_dict':
            output_file = f'{base_name}_weights.pt'
        elif args.mode == 'full_model':
            output_file = f'{base_name}_full.pt'
        elif args.mode == 'torchscript':
            output_file = f'{base_name}_torchscript.pt'
        elif args.mode == 'deploy':
            output_file = f'{base_name}_deploy.pt'
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f'\n🎯 导出模式: {args.mode}')
    print(f'💾 输出文件: {output_file}\n')
    
    # 根据模式导出
    if args.mode == 'state_dict':
        export_state_dict(cfg, checkpoint, output_file)
        print('\n💡 使用方法:')
        print('   model = build_model(config)')
        print('   state_dict = torch.load("model_weights.pt")')
        print('   model.load_state_dict(state_dict)')
        
    elif args.mode == 'full_model':
        export_full_model(cfg, checkpoint, output_file)
        print('\n💡 使用方法:')
        print('   model = torch.load("model_full.pt")')
        print('   model.eval()')
        
    elif args.mode == 'torchscript':
        export_torchscript(cfg, checkpoint, output_file, args.input_size)
        print('\n💡 使用方法:')
        print('   model = torch.jit.load("model_torchscript.pt")')
        print('   output = model(images, sizes)')
        
    elif args.mode == 'deploy':
        export_deploy_model(cfg, checkpoint, output_file)
        print('\n💡 使用方法:')
        print('   model = torch.load("model_deploy.pt")')
        print('   output = model(images, sizes)')
    
    print('\n✨ 导出完成!')
    print('='*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将RT-DETR模型导出为.pt格式')
    
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='配置文件路径 (例如: configs/rtdetr/rtdetr_r18vd_6x_coco.yml)')
    
    parser.add_argument('--resume', '-r', type=str, required=True,
                        help='训练好的checkpoint路径 (例如: output/best.pth)')
    
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出文件路径 (默认: 自动生成)')
    
    parser.add_argument('--mode', '-m', type=str, default='state_dict',
                        choices=['state_dict', 'full_model', 'torchscript', 'deploy'],
                        help='''导出模式:
                        - state_dict: 只导出权重 (推荐，体积小)
                        - full_model: 导出完整模型
                        - torchscript: 导出TorchScript格式 (用于C++部署)
                        - deploy: 导出部署模型 (包含后处理)
                        ''')
    
    parser.add_argument('--input-size', type=int, default=640,
                        help='输入图像尺寸 (仅用于torchscript模式, 默认: 640)')
    
    args = parser.parse_args()
    
    main(args)

