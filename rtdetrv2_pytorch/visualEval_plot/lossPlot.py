import re
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd
from pathlib import Path
from scipy.interpolate import make_interp_spline

def smooth_curve(x, y, smoothing_factor=100):
    """使用样条插值平滑曲线"""
    if len(x) < 2:
        return x, y
    
    # 当数据点少于3个时，不进行平滑处理
    if len(x) < 3:
        return x, y
    
    try:
        # 创建插值函数
        x_new = np.linspace(min(x), max(x), smoothing_factor)
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_new)
        return x_new, y_smooth
    except Exception:
        # 如果插值失败，返回原始数据
        return x, y

def parse_log_file(file_path):
    """解析日志文件，提取训练指标"""
    epochs = []
    train_losses = []
    train_losses_vfl = []
    train_losses_bbox = []
    train_losses_giou = []
    train_lrs = []
    
    # COCO评估指标
    coco_bbox = []
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                epochs.append(data['epoch'])
                train_losses.append(data['train_loss'])
                train_losses_vfl.append(data['train_loss_vfl'])
                train_losses_bbox.append(data['train_loss_bbox'])
                train_losses_giou.append(data['train_loss_giou'])
                train_lrs.append(data['train_lr'])
                
                # 解析COCO评估指标
                if 'test_coco_eval_bbox' in data:
                    # 过滤掉无效值(-1)
                    metrics = [v if v >= 0 else np.nan for v in data['test_coco_eval_bbox']]
                    coco_bbox.append(metrics)
                else:
                    coco_bbox.append([np.nan] * 12)  # 如果没有评估结果，填充NaN
                    
            except json.JSONDecodeError:
                print(f"跳过无法解析的行: {line}")
                continue
    
    # 将COCO评估指标转换为DataFrame
    coco_columns = [
        'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 
        'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl'
    ]
    coco_df = pd.DataFrame(coco_bbox, columns=coco_columns)
    
    # 创建结果字典
    results = {
        'epochs': epochs,
        'train_loss': train_losses,
        'train_loss_vfl': train_losses_vfl,
        'train_loss_bbox': train_losses_bbox,
        'train_loss_giou': train_losses_giou,
        'train_lr': train_lrs,
        'coco_metrics': coco_df
    }
    
    return results

def plot_training_curves(results, save_path=None):
    """绘制训练曲线"""
    epochs = results['epochs']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RT-DETR Training Process Metrics', fontsize=16, fontweight='bold')
    
    # 总损失
    if len(epochs) > 2:
        x_smooth, y_smooth = smooth_curve(epochs, results['train_loss'])
        ax1.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Total Loss')
    else:
        ax1.plot(epochs, results['train_loss'], 'b-', linewidth=2, marker='o', markersize=4, label='Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training Total Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 各组件损失
    if len(epochs) > 2:
        x_smooth_vfl, y_smooth_vfl = smooth_curve(epochs, results['train_loss_vfl'])
        x_smooth_bbox, y_smooth_bbox = smooth_curve(epochs, results['train_loss_bbox'])
        x_smooth_giou, y_smooth_giou = smooth_curve(epochs, results['train_loss_giou'])
        
        ax2.plot(x_smooth_vfl, y_smooth_vfl, 'r-', linewidth=2, label='VFL Loss')
        ax2.plot(x_smooth_bbox, y_smooth_bbox, 'g-', linewidth=2, label='BBox Loss')
        ax2.plot(x_smooth_giou, y_smooth_giou, 'm-', linewidth=2, label='GIoU Loss')
    else:
        ax2.plot(epochs, results['train_loss_vfl'], 'r-', linewidth=2, marker='s', markersize=3, label='VFL Loss')
        ax2.plot(epochs, results['train_loss_bbox'], 'g-', linewidth=2, marker='^', markersize=3, label='BBox Loss')
        ax2.plot(epochs, results['train_loss_giou'], 'm-', linewidth=2, marker='d', markersize=3, label='GIoU Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Value')
    ax2.set_title('Component Training Losses', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 学习率
    if len(epochs) > 2:
        x_smooth, y_smooth = smooth_curve(epochs, results['train_lr'])
        ax3.plot(x_smooth, y_smooth, 'c-', linewidth=2, label='Learning Rate')
    else:
        ax3.plot(epochs, results['train_lr'], 'c-', linewidth=2, marker='.', markersize=4, label='Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Changes', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # 使用对数刻度更好地显示学习率变化
    
    # 损失对比（归一化）
    max_loss = max(max(results['train_loss_vfl']), 
                  max(results['train_loss_bbox']), 
                  max(results['train_loss_giou']))
    
    if max_loss > 0 and len(epochs) > 0:
        norm_vfl = [x/max_loss for x in results['train_loss_vfl']]
        norm_bbox = [x/max_loss for x in results['train_loss_bbox']]
        norm_giou = [x/max_loss for x in results['train_loss_giou']]
        
        if len(epochs) > 2:
            x_smooth_vfl, y_smooth_vfl = smooth_curve(epochs, norm_vfl)
            x_smooth_bbox, y_smooth_bbox = smooth_curve(epochs, norm_bbox)
            x_smooth_giou, y_smooth_giou = smooth_curve(epochs, norm_giou)
            
            ax4.plot(x_smooth_vfl, y_smooth_vfl, 'r-', linewidth=2, label='VFL Loss (Normalized)')
            ax4.plot(x_smooth_bbox, y_smooth_bbox, 'g-', linewidth=2, label='BBox Loss (Normalized)')
            ax4.plot(x_smooth_giou, y_smooth_giou, 'm-', linewidth=2, label='GIoU Loss (Normalized)')
        else:
            ax4.plot(epochs, norm_vfl, 'r-', linewidth=2, marker='s', markersize=3, label='VFL Loss (Normalized)')
            ax4.plot(epochs, norm_bbox, 'g-', linewidth=2, marker='^', markersize=3, label='BBox Loss (Normalized)')
            ax4.plot(epochs, norm_giou, 'm-', linewidth=2, marker='d', markersize=3, label='GIoU Loss (Normalized)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Normalized Loss Value')
        ax4.set_title('Normalized Loss Comparison', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Normalized Loss Comparison', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_coco_metrics(results, save_path=None):
    """绘制COCO评估指标"""
    coco_df = results['coco_metrics']
    epochs = results['epochs']
    
    # 过滤掉无效值(-1)
    valid_indices = ~coco_df['AP'].isna()
    valid_coco_df = coco_df[valid_indices]
    valid_epochs = [epochs[i] for i in range(len(epochs)) if valid_indices.iloc[i]]
    
    if len(valid_epochs) == 0:
        print("No valid COCO evaluation data")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('COCO Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # AP指标
    if len(valid_epochs) > 2:
        x_smooth, y_smooth = smooth_curve(valid_epochs, valid_coco_df['AP'])
        ax1.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='AP')
        
        x_smooth, y_smooth = smooth_curve(valid_epochs, valid_coco_df['AP50'])
        ax1.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='AP50')
        
        x_smooth, y_smooth = smooth_curve(valid_epochs, valid_coco_df['AP75'])
        ax1.plot(x_smooth, y_smooth, 'g-', linewidth=2, label='AP75')
    else:
        ax1.plot(valid_epochs, valid_coco_df['AP'], 'b-', linewidth=2, marker='o', markersize=4, label='AP')
        ax1.plot(valid_epochs, valid_coco_df['AP50'], 'r-', linewidth=2, marker='s', markersize=4, label='AP50')
        ax1.plot(valid_epochs, valid_coco_df['AP75'], 'g-', linewidth=2, marker='^', markersize=4, label='AP75')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('AP Value')
    ax1.set_title('Average Precision (AP) Metrics', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 不同尺度上的AP
    if len(valid_epochs) > 2:
        x_smooth, y_smooth = smooth_curve(valid_epochs, valid_coco_df['APs'])
        ax2.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='APs (Small Objects)')
        
        x_smooth, y_smooth = smooth_curve(valid_epochs, valid_coco_df['APm'])
        ax2.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='APm (Medium Objects)')
        
        x_smooth, y_smooth = smooth_curve(valid_epochs, valid_coco_df['APl'])
        ax2.plot(x_smooth, y_smooth, 'g-', linewidth=2, label='APl (Large Objects)')
    else:
        ax2.plot(valid_epochs, valid_coco_df['APs'], 'b-', linewidth=2, marker='o', markersize=4, label='APs (Small Objects)')
        ax2.plot(valid_epochs, valid_coco_df['APm'], 'r-', linewidth=2, marker='s', markersize=4, label='APm (Medium Objects)')
        ax2.plot(valid_epochs, valid_coco_df['APl'], 'g-', linewidth=2, marker='^', markersize=4, label='APl (Large Objects)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AP Value')
    ax2.set_title('AP by Object Scale', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # AR指标
    if len(valid_epochs) > 2:
        x_smooth, y_smooth = smooth_curve(valid_epochs, valid_coco_df['AR1'])
        ax3.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='AR1')
        
        x_smooth, y_smooth = smooth_curve(valid_epochs, valid_coco_df['AR10'])
        ax3.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='AR10')
        
        x_smooth, y_smooth = smooth_curve(valid_epochs, valid_coco_df['AR100'])
        ax3.plot(x_smooth, y_smooth, 'g-', linewidth=2, label='AR100')
    else:
        ax3.plot(valid_epochs, valid_coco_df['AR1'], 'b-', linewidth=2, marker='o', markersize=4, label='AR1')
        ax3.plot(valid_epochs, valid_coco_df['AR10'], 'r-', linewidth=2, marker='s', markersize=4, label='AR10')
        ax3.plot(valid_epochs, valid_coco_df['AR100'], 'g-', linewidth=2, marker='^', markersize=4, label='AR100')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AR Value')
    ax3.set_title('Average Recall (AR) Metrics', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 不同尺度上的AR
    if len(valid_epochs) > 2:
        x_smooth, y_smooth = smooth_curve(valid_epochs, valid_coco_df['ARs'])
        ax4.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='ARs (Small Objects)')
        
        x_smooth, y_smooth = smooth_curve(valid_epochs, valid_coco_df['ARm'])
        ax4.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='ARm (Medium Objects)')
        
        x_smooth, y_smooth = smooth_curve(valid_epochs, valid_coco_df['ARl'])
        ax4.plot(x_smooth, y_smooth, 'g-', linewidth=2, label='ARl (Large Objects)')
    else:
        ax4.plot(valid_epochs, valid_coco_df['ARs'], 'b-', linewidth=2, marker='o', markersize=4, label='ARs (Small Objects)')
        ax4.plot(valid_epochs, valid_coco_df['ARm'], 'r-', linewidth=2, marker='s', markersize=4, label='ARm (Medium Objects)')
        ax4.plot(valid_epochs, valid_coco_df['ARl'], 'g-', linewidth=2, marker='^', markersize=4, label='ARl (Large Objects)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('AR Value')
    ax4.set_title('AR by Object Scale', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_best_metrics(results, save_path=None):
    """绘制最佳指标及其对应的epoch"""
    coco_df = results['coco_metrics']
    epochs = results['epochs']
    
    # 过滤掉无效值(-1)
    valid_indices = ~coco_df['AP'].isna()
    valid_coco_df = coco_df[valid_indices]
    valid_epochs = [epochs[i] for i in range(len(epochs)) if valid_indices.iloc[i]]
    
    if len(valid_epochs) == 0:
        print("No valid COCO evaluation data")
        return {}
    
    # 找到最佳AP和AR的epoch
    best_ap_idx = np.nanargmax(valid_coco_df['AP'])
    best_ap50_idx = np.nanargmax(valid_coco_df['AP50'])
    best_ar100_idx = np.nanargmax(valid_coco_df['AR100'])
    
    best_ap_epoch = valid_epochs[best_ap_idx]
    best_ap50_epoch = valid_epochs[best_ap50_idx]
    best_ar100_epoch = valid_epochs[best_ar100_idx]
    
    best_ap = valid_coco_df['AP'].iloc[best_ap_idx]
    best_ap50 = valid_coco_df['AP50'].iloc[best_ap50_idx]
    best_ar100 = valid_coco_df['AR100'].iloc[best_ar100_idx]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制AP和AR曲线
    ax.plot(valid_epochs, valid_coco_df['AP'], 'b-', linewidth=2, marker='o', markersize=4, label='AP')
    ax.plot(valid_epochs, valid_coco_df['AP50'], 'r-', linewidth=2, marker='s', markersize=4, label='AP50')
    ax.plot(valid_epochs, valid_coco_df['AR100'], 'g-', linewidth=2, marker='^', markersize=4, label='AR100')
    
    # 标记最佳点
    ax.plot(best_ap_epoch, best_ap, 'bo', markersize=10, label=f'Best AP: {best_ap:.4f}')
    ax.plot(best_ap50_epoch, best_ap50, 'ro', markersize=10, label=f'Best AP50: {best_ap50:.4f}')
    ax.plot(best_ar100_epoch, best_ar100, 'go', markersize=10, label=f'Best AR100: {best_ar100:.4f}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric Value')
    ax.set_title('Best Evaluation Metrics', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加文本注释
    textstr = f'Best AP: {best_ap:.4f} (Epoch {best_ap_epoch})\nBest AP50: {best_ap50:.4f} (Epoch {best_ap50_epoch})\nBest AR100: {best_ar100:.4f} (Epoch {best_ar100_epoch})'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return {
        'best_ap': best_ap,
        'best_ap_epoch': best_ap_epoch,
        'best_ap50': best_ap50,
        'best_ap50_epoch': best_ap50_epoch,
        'best_ar100': best_ar100,
        'best_ar100_epoch': best_ar100_epoch
    }

def plot_loss_vs_metrics(results, save_path=None):
    """绘制损失与评估指标的关系"""
    epochs = results['epochs']
    train_loss = results['train_loss']
    coco_df = results['coco_metrics']
    
    # 过滤掉无效值(-1)
    valid_indices = ~coco_df['AP'].isna()
    valid_coco_df = coco_df[valid_indices]
    valid_epochs = [epochs[i] for i in range(len(epochs)) if valid_indices.iloc[i]]
    valid_train_loss = [train_loss[i] for i in range(len(train_loss)) if valid_indices.iloc[i]]
    
    if len(valid_epochs) == 0:
        print("No valid COCO evaluation data")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Relationship between Training Loss and Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # 损失与AP的关系
    scatter1 = ax1.scatter(valid_train_loss, valid_coco_df['AP'], c=valid_epochs, cmap='viridis', alpha=0.7, s=50)
    ax1.set_xlabel('Training Loss')
    ax1.set_ylabel('AP')
    ax1.set_title('Training Loss vs AP', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Epoch')
    
    # 损失与AP50的关系
    scatter2 = ax2.scatter(valid_train_loss, valid_coco_df['AP50'], c=valid_epochs, cmap='viridis', alpha=0.7, s=50)
    ax2.set_xlabel('Training Loss')
    ax2.set_ylabel('AP50')
    ax2.set_title('Training Loss vs AP50', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Epoch')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def print_summary(results):
    """打印训练摘要"""
    epochs = results['epochs']
    train_loss = results['train_loss']
    coco_df = results['coco_metrics']
    
    print("="*50)
    print("Training Summary")
    print("="*50)
    
    print(f"Total Training Epochs: {len(epochs)}")
    if epochs:
        print(f"Final Training Loss: {train_loss[-1]:.4f}")
        print(f"Minimum Training Loss: {min(train_loss):.4f} (Epoch {epochs[train_loss.index(min(train_loss))]})")
    
    # 过滤掉无效值(-1)
    valid_indices = ~coco_df['AP'].isna()
    valid_coco_df = coco_df[valid_indices]
    valid_epochs = [epochs[i] for i in range(len(epochs)) if valid_indices.iloc[i]]
    
    if len(valid_epochs) > 0:
        best_ap_idx = np.nanargmax(valid_coco_df['AP'])
        best_ap = valid_coco_df['AP'].iloc[best_ap_idx]
        best_ap_epoch = valid_epochs[best_ap_idx]
        
        best_ap50_idx = np.nanargmax(valid_coco_df['AP50'])
        best_ap50 = valid_coco_df['AP50'].iloc[best_ap50_idx]
        best_ap50_epoch = valid_epochs[best_ap50_idx]
        
        print(f"Best AP: {best_ap:.4f} (Epoch {best_ap_epoch})")
        print(f"Best AP50: {best_ap50:.4f} (Epoch {best_ap50_epoch})")
    else:
        print("No valid evaluation data")
    
    print("="*50)

def main():
    # 解析日志文件
    log_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection/log.txt"
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    results = parse_log_file(log_file)
    
    # 创建输出目录
    output_dir = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/plot"
    os.makedirs(output_dir, exist_ok=True)
    
    # 打印摘要
    print_summary(results)
    
    # 绘制训练曲线
    plot_training_curves(results, save_path=os.path.join(output_dir, "training_curves.png"))
    
    # 绘制COCO评估指标
    plot_coco_metrics(results, save_path=os.path.join(output_dir, "coco_metrics.png"))
    
    # 绘制最佳指标
    best_metrics = plot_best_metrics(results, save_path=os.path.join(output_dir, "best_metrics.png"))
    
    # 绘制损失与评估指标的关系
    plot_loss_vs_metrics(results, save_path=os.path.join(output_dir, "loss_vs_metrics.png"))
    
    # 打印最佳指标
    if best_metrics:
        print("\nBest Evaluation Metrics:")
        print(f"Best AP: {best_metrics['best_ap']:.4f} (Epoch {best_metrics['best_ap_epoch']})")
        print(f"Best AP50: {best_metrics['best_ap50']:.4f} (Epoch {best_metrics['best_ap50_epoch']})")
        print(f"Best AR100: {best_metrics['best_ar100']:.4f} (Epoch {best_metrics['best_ar100_epoch']})")

if __name__ == "__main__":
    main()