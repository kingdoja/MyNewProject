import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import torch
from collections import defaultdict

def plot_precision_recall(files, save_path=None):
    """
    绘制精确率-召回率曲线
    """
    if not files:
        print("No evaluation files found")
        return
    
    # 按照文件名排序，通常是按epoch排序
    files.sort()
    
    # 存储每个epoch的评估结果
    epochs = []
    precisions = []
    recalls = []
    
    for file in files:
        try:
            # 从文件名中提取epoch数
            epoch = int(file.stem)
            epochs.append(epoch)
            
            # 加载评估结果
            eval_result = torch.load(file)
            
            # 提取精确率和召回率（这里假设使用COCO评估格式）
            # 注意：根据实际的评估结果格式进行调整
            if 'precision' in eval_result and 'recall' in eval_result:
                # COCO评估格式
                precision = eval_result['precision']
                recall = eval_result['recall']
                
                # 计算平均精确率和召回率
                # precision是一个5维数组 [TxRxKxAxM]，其中T是IoU阈值，R是召回率点，K是类别，A是面积，M是最大检测数
                # 我们取所有IoU阈值下的平均精确率
                avg_precision = np.mean(precision[precision > -1]) if np.sum(precision > -1) > 0 else 0
                avg_recall = np.mean(recall[recall > -1]) if np.sum(recall > -1) > 0 else 0
                
                precisions.append(avg_precision * 100)  # 转换为百分比
                recalls.append(avg_recall * 100)  # 转换为百分比
            else:
                print(f"Warning: File {file} does not contain expected precision/recall data")
                precisions.append(0)
                recalls.append(0)
                
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
    
    # 绘制精确率-召回率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Recall (%)')
    plt.ylabel('Precision (%)')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    # 添加数值标签
    for i, (recall, precision) in enumerate(zip(recalls, precisions)):
        if i % max(1, len(recalls)//10) == 0:  # 每隔几个点添加标签
            plt.annotate(f'{epochs[i]}', (recall, precision), 
                        textcoords="offset points", xytext=(0,10), ha='center')
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    
    plt.show()

def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt', save_path=None):
    """
    绘制训练日志曲线
    """
    try:
        # 读取日志文件
        log_path = logs / log_name
        if not log_path.exists():
            print(f"Log file not found: {log_path}")
            return
        
        # 解析日志文件
        data = defaultdict(list)
        epochs = []
        
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    epoch = log_entry.get('epoch', 0)
                    epochs.append(epoch)
                    
                    # 提取指定字段
                    for field in fields:
                        if field in log_entry:
                            data[field].append(log_entry[field])
                        elif field.startswith('test_') and field[5:] in log_entry:
                            # 如果字段以test_开头，尝试查找不带前缀的字段
                            data[field].append(log_entry[field[5:]])
                        elif 'test_coco_eval_bbox' in log_entry and field in ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']:
                            # 如果是COCO评估字段
                            coco_fields = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
                            if field in coco_fields:
                                idx = coco_fields.index(field)
                                if idx < len(log_entry['test_coco_eval_bbox']):
                                    data[field].append(log_entry['test_coco_eval_bbox'][idx])
                                else:
                                    data[field].append(0)
                            else:
                                data[field].append(0)
                        else:
                            data[field].append(0)  # 默认值
                            
                except json.JSONDecodeError:
                    continue  # 跳过无法解析的行
        
        # 绘制曲线
        n_fields = len(fields)
        fig, axes = plt.subplots(n_fields, 1, figsize=(12, 4*n_fields))
        if n_fields == 1:
            axes = [axes]
        
        for i, field in enumerate(fields):
            if data[field]:
                axes[i].plot(epochs, data[field], 'b-', linewidth=2, marker='o', markersize=4)
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(field)
                axes[i].set_title(f'{field} over Training')
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, f'No data for {field}', ha='center', va='center', 
                            transform=axes[i].transAxes)
                axes[i].set_title(f'{field} over Training')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training logs plot saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error plotting logs: {e}")

if __name__ == '__main__':
    # 设置保存路径
    plot_dir = Path('/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/plot')
    plot_dir.mkdir(exist_ok=True)  # 创建目录（如果不存在）
    
    # 修正路径中的反斜杠为正斜杠
    eval_files = list(Path('/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection/eval').glob('*.pth'))
    
    # 绘制并保存精确率-召回率曲线
    plot_precision_recall(eval_files, save_path=plot_dir / 'precision_recall_curve.png')
    
    # 绘制并保存训练日志曲线
    plot_logs(
        logs=Path('/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection'),
        fields=('train_loss', 'test_coco_eval_bbox'), 
        log_name='log.txt',
        save_path=plot_dir / 'training_logs.png'
    )