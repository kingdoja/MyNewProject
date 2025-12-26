#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练监控脚本 - 实时分析训练过程中的loss和指标变化
帮助诊断训练问题
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def parse_training_log(log_file):
    """解析训练日志文件"""
    epochs = []
    train_losses = []
    val_metrics = defaultdict(list)
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析训练loss
    # 匹配格式: Epoch 1/100: train_loss=2.345
    train_loss_pattern = r'Epoch\s+(\d+).*?train_loss[=:]\s*([\d.]+)'
    for match in re.finditer(train_loss_pattern, content):
        epoch = int(match.group(1))
        loss = float(match.group(2))
        epochs.append(epoch)
        train_losses.append(loss)
    
    # 解析验证指标
    # 匹配格式: AP=0.345, AP50=0.456, AP75=0.321
    val_pattern = r'(?:Test|Val|Eval).*?AP[=:]\s*([\d.]+).*?AP50[=:]\s*([\d.]+).*?AP75[=:]\s*([\d.]+)'
    for match in re.finditer(val_pattern, content):
        ap = float(match.group(1))
        ap50 = float(match.group(2))
        ap75 = float(match.group(3))
        val_metrics['AP'].append(ap)
        val_metrics['AP50'].append(ap50)
        val_metrics['AP75'].append(ap75)
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_metrics': val_metrics
    }

def analyze_training_health(data):
    """分析训练健康状况"""
    print("\n" + "="*60)
    print("训练健康状况分析")
    print("="*60)
    
    epochs = data['epochs']
    train_losses = data['train_losses']
    val_metrics = data['val_metrics']
    
    if not epochs:
        print("⚠️  未找到训练数据")
        return
    
    print(f"\n训练轮数: {len(epochs)}")
    print(f"当前轮次: {max(epochs) if epochs else 0}")
    
    # 分析loss趋势
    if len(train_losses) > 10:
        recent_losses = train_losses[-10:]
        early_losses = train_losses[:10]
        
        recent_avg = np.mean(recent_losses)
        early_avg = np.mean(early_losses)
        loss_reduction = (early_avg - recent_avg) / early_avg * 100
        
        print(f"\nLoss分析:")
        print(f"  初始平均loss: {early_avg:.4f}")
        print(f"  最近平均loss: {recent_avg:.4f}")
        print(f"  Loss下降: {loss_reduction:.2f}%")
        
        if loss_reduction < 10:
            print("  ⚠️  警告: Loss下降不明显，可能的原因:")
            print("    - 学习率过小")
            print("    - 数据增强过度")
            print("    - 模型容量不足")
        elif loss_reduction > 80:
            print("  ⚠️  警告: Loss下降过快，可能过拟合")
    
    # 检查loss是否还在下降
    if len(train_losses) > 5:
        recent_trend = np.polyfit(range(len(train_losses[-5:])), train_losses[-5:], 1)[0]
        if recent_trend > 0:
            print("  ⚠️  警告: 最近loss有上升趋势，可能过拟合")
        else:
            print("  ✅ Loss仍在下降")
    
    # 分析验证指标
    if val_metrics.get('AP'):
        ap_values = val_metrics['AP']
        ap50_values = val_metrics['AP50']
        
        print(f"\n验证指标分析:")
        print(f"  当前AP: {ap_values[-1]:.4f}" if ap_values else "  无数据")
        print(f"  当前AP50: {ap50_values[-1]:.4f}" if ap50_values else "  无数据")
        print(f"  最佳AP: {max(ap_values):.4f}" if ap_values else "  无数据")
        print(f"  最佳AP50: {max(ap50_values):.4f}" if ap50_values else "  无数据")
        
        if ap_values:
            if max(ap_values) < 0.3:
                print("  ⚠️  警告: AP较低，建议:")
                print("    - 检查数据标注质量")
                print("    - 增加num_queries")
                print("    - 调整损失函数权重")
                print("    - 检查类别分布是否平衡")
            elif max(ap_values) < 0.5:
                print("  ⚠️  AP中等，仍有提升空间")
            else:
                print("  ✅ AP表现良好")
    
    # 检查过拟合
    if train_losses and val_metrics.get('AP'):
        if len(train_losses) == len(val_metrics['AP']):
            # 计算训练loss和验证AP的相关性
            # 理想情况下，训练loss下降，验证AP上升
            train_loss_trend = np.polyfit(range(len(train_losses)), train_losses, 1)[0]
            val_ap_trend = np.polyfit(range(len(val_metrics['AP'])), val_metrics['AP'], 1)[0]
            
            if train_loss_trend < 0 and val_ap_trend < 0:
                print("\n  ⚠️  警告: 可能出现过拟合（训练loss下降但验证AP下降）")
            elif train_loss_trend < 0 and val_ap_trend > 0:
                print("\n  ✅ 训练正常（训练loss下降，验证AP上升）")

def plot_training_curves(data, output_dir):
    """绘制训练曲线"""
    epochs = data['epochs']
    train_losses = data['train_losses']
    val_metrics = data['val_metrics']
    
    if not epochs:
        print("⚠️  没有数据可绘制")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 训练Loss曲线
    axes[0, 0].plot(epochs[:len(train_losses)], train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # 2. 验证AP曲线
    if val_metrics.get('AP'):
        ap_epochs = epochs[:len(val_metrics['AP'])]
        axes[0, 1].plot(ap_epochs, val_metrics['AP'], 'g-', linewidth=2, label='AP', marker='o')
        if val_metrics.get('AP50'):
            axes[0, 1].plot(ap_epochs, val_metrics['AP50'], 'r-', linewidth=2, label='AP50', marker='s')
        if val_metrics.get('AP75'):
            axes[0, 1].plot(ap_epochs, val_metrics['AP75'], 'm-', linewidth=2, label='AP75', marker='^')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AP')
        axes[0, 1].set_title('Validation Metrics')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
    
    # 3. Loss趋势（最近20个epoch）
    if len(train_losses) > 20:
        recent_epochs = epochs[-20:]
        recent_losses = train_losses[-20:]
        axes[1, 0].plot(recent_epochs, recent_losses, 'b-', linewidth=2, marker='o')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Recent Training Loss (Last 20 Epochs)')
        axes[1, 0].grid(True)
    
    # 4. Loss和AP对比
    if train_losses and val_metrics.get('AP'):
        min_len = min(len(train_losses), len(val_metrics['AP']))
        if min_len > 0:
            ax2 = axes[1, 1]
            ax1 = ax2.twinx()
            
            epochs_common = epochs[:min_len]
            losses_common = train_losses[:min_len]
            ap_common = val_metrics['AP'][:min_len]
            
            line1 = ax1.plot(epochs_common, losses_common, 'b-', linewidth=2, label='Train Loss')
            line2 = ax2.plot(epochs_common, ap_common, 'g-', linewidth=2, label='AP')
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss', color='b')
            ax2.set_ylabel('AP', color='g')
            ax1.set_title('Loss vs AP')
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='g')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'training_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n训练曲线已保存到: {output_path}")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='监控训练过程')
    parser.add_argument('--log-file', type=str, required=True,
                       help='训练日志文件路径')
    parser.add_argument('--output-dir', type=str, default='./training_monitor',
                       help='输出目录')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("训练监控工具")
    print("="*60)
    
    if not os.path.exists(args.log_file):
        print(f"❌ 日志文件不存在: {args.log_file}")
        return
    
    # 解析日志
    data = parse_training_log(args.log_file)
    
    # 分析
    analyze_training_health(data)
    
    # 绘图
    plot_training_curves(data, output_dir)
    
    print("\n" + "="*60)
    print("监控完成！")
    print("="*60)

if __name__ == '__main__':
    main()

