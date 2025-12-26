#评估结果可视化
from re import L
from syslog import LOG_ALERT
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# 直接在代码中指定固定路径
# LOG_FILE_PATH = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection/log.txt"
LOG_FILE_PATH = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection_p4_100_usePre/log.txt"
OUTPUT_DIR = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/plot"

# 读取日志内容
if os.path.exists(LOG_FILE_PATH):
    with open(LOG_FILE_PATH, 'r') as f:
        log_content = f.read()
else:
    print(f"Error: Log file not found at {LOG_FILE_PATH}")
    exit(1)

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 提取日志数据
epochs = []
ap = []      # AP@0.5:0.95
ap50 = []    # AP@0.5
ap75 = []    # AP@0.75
aps = []     # AP small
apm = []     # AP medium
apl = []     # AP large
ar = []      # AR@0.5:0.95
ar50 = []    # AR@0.5
ar75 = []    # AR@0.75

# 解析日志内容
log_entries = []
for line in log_content.strip().split('\n'):
    if line.strip():
        try:
            data = json.loads(line)
            log_entries.append(data)
        except json.JSONDecodeError:
            continue

# 提取评估指标
for entry in log_entries:
    if 'test_coco_eval_bbox' in entry:
        eval_data = entry['test_coco_eval_bbox']
        if len(eval_data) >= 12:  # 确保有足够的数据点
            epochs.append(entry.get('epoch', 0))
            ap.append(eval_data[0] * 100)    # 转换为百分比
            ap50.append(eval_data[1] * 100)
            ap75.append(eval_data[2] * 100)
            aps.append(eval_data[3] * 100)
            apm.append(eval_data[4] * 100)
            apl.append(eval_data[5] * 100)
            ar.append(eval_data[6] * 100)
            ar50.append(eval_data[7] * 100)
            ar75.append(eval_data[8] * 100)

# 检查是否有数据
if len(epochs) == 0:
    print("Error: No evaluation data found in log file.")
    exit(1)

# 创建图表
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('RT-DETR Training Evaluation Metrics', fontsize=16, fontweight='bold')

# AP指标
axes[0, 0].plot(epochs, ap, 'b-', marker='o', label='AP@0.5:0.95', linewidth=2, markersize=6)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('AP (%)')
axes[0, 0].set_title('AP@0.5:0.95')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# AP@0.5
axes[0, 1].plot(epochs, ap50, 'r-', marker='s', label='AP@0.5', linewidth=2, markersize=6)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('AP50 (%)')
axes[0, 1].set_title('AP@0.5')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# AP@0.75
axes[0, 2].plot(epochs, ap75, 'g-', marker='^', label='AP@0.75', linewidth=2, markersize=6)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('AP75 (%)')
axes[0, 2].set_title('AP@0.75')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend()

# 不同尺寸的AP
axes[1, 0].plot(epochs, aps, 'c-', marker='d', label='AP-small', linewidth=2, markersize=6)
axes[1, 0].plot(epochs, apm, 'm-', marker='v', label='AP-medium', linewidth=2, markersize=6)
axes[1, 0].plot(epochs, apl, 'y-', marker='*', label='AP-large', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('AP (%)')
axes[1, 0].set_title('AP by Scale')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# 所有AP指标组合图
axes[1, 1].plot(epochs, ap, 'b-', marker='o', label='AP@0.5:0.95', linewidth=2, markersize=6)
axes[1, 1].plot(epochs, ap50, 'r-', marker='s', label='AP@0.5', linewidth=2, markersize=6)
axes[1, 1].plot(epochs, ap75, 'g-', marker='^', label='AP@0.75', linewidth=2, markersize=6)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('AP (%)')
axes[1, 1].set_title('All AP Metrics')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

# AR指标
axes[1, 2].plot(epochs, ar, 'b-', marker='o', label='AR@0.5:0.95', linewidth=2, markersize=6)
axes[1, 2].plot(epochs, ar50, 'r-', marker='s', label='AR@0.5', linewidth=2, markersize=6)
axes[1, 2].plot(epochs, ar75, 'g-', marker='^', label='AR@0.75', linewidth=2, markersize=6)
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('AR (%)')
axes[1, 2].set_title('AR Metrics')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].legend()

plt.tight_layout()

# 保存图表
output_path = os.path.join(OUTPUT_DIR, 'evaluation_metrics.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to: {output_path}")

# 显示图表
plt.show()

# 打印最佳结果
best_epoch_idx = np.argmax(ap)
print("\n=== Best Evaluation Results ===")
print(f"Best AP@0.5:0.95: {ap[best_epoch_idx]:.2f}% (Epoch {epochs[best_epoch_idx]})")
print(f"  AP@0.5: {ap50[best_epoch_idx]:.2f}%")
print(f"  AP@0.75: {ap75[best_epoch_idx]:.2f}%")
print(f"  AP-small: {aps[best_epoch_idx]:.2f}%")
print(f"  AP-medium: {apm[best_epoch_idx]:.2f}%")
print(f"  AP-large: {apl[best_epoch_idx]:.2f}%")
print(f"  AR@0.5:0.95: {ar[best_epoch_idx]:.2f}%")
print(f"  AR@0.5: {ar50[best_epoch_idx]:.2f}%")
print(f"  AR@0.75: {ar75[best_epoch_idx]:.2f}%")

# 打印最终结果
if len(ap) > 1:
    print("\n=== Final Evaluation Results ===")
    print(f"Final AP@0.5:0.95: {ap[-1]:.2f}% (Epoch {epochs[-1]})")
    print(f"  AP@0.5: {ap50[-1]:.2f}%")
    print(f"  AP@0.75: {ap75[-1]:.2f}%")
    print(f"  AP-small: {aps[-1]:.2f}%")
    print(f"  AP-medium: {apm[-1]:.2f}%")
    print(f"  AP-large: {apl[-1]:.2f}%")
    print(f"  AR@0.5:0.95: {ar[-1]:.2f}%")
    print(f"  AR@0.5: {ar50[-1]:.2f}%")
    print(f"  AR@0.75: {ar75[-1]:.2f}%")

# 计算和显示精确率与召回率的详细信息
def calculate_precision_recall():
    """
    Calculate precision and recall of the model
    In object detection:
    - Precision is typically measured by Average Precision (AP)
    - Recall is typically measured by Average Recall (AR)
    """
    print("\n=== Precision and Recall Analysis ===")
    
    # Latest evaluation results for precision and recall
    latest_precision = ap[-1]  # AP as comprehensive precision metric
    latest_recall = ar[-1]     # AR as comprehensive recall metric
    
    print(f"Latest Precision (AP@0.5:0.95): {latest_precision:.2f}%")
    print(f"Latest Recall (AR@0.5:0.95): {latest_recall:.2f}%")
    
    # Best evaluation results for precision and recall
    best_precision = max(ap)
    best_recall = max(ar)
    best_precision_epoch = epochs[np.argmax(ap)]
    best_recall_epoch = epochs[np.argmax(ar)]
    
    print(f"Best Precision: {best_precision:.2f}% (Epoch {best_precision_epoch})")
    print(f"Best Recall: {best_recall:.2f}% (Epoch {best_recall_epoch})")
    
    # Precision and recall at IoU=0.5
    print(f"\nPrecision at IoU=0.5 (AP@0.5): {ap50[-1]:.2f}%")
    print(f"Recall at IoU=0.5 (AR@0.5): {ar50[-1]:.2f}%")
    
    # Precision and recall at IoU=0.75
    print(f"\nPrecision at IoU=0.75 (AP@0.75): {ap75[-1]:.2f}%")
    print(f"Recall at IoU=0.75 (AR@0.75): {ar75[-1]:.2f}%")
    
    # Precision and recall for different object scales
    print(f"\nPerformance by Object Scale:")
    print(f"  Small objects - Precision: {aps[-1]:.2f}%, Recall: {ar[np.argmax(aps) if aps else -1]:.2f}%")
    print(f"  Medium objects - Precision: {apm[-1]:.2f}%, Recall: {ar[np.argmax(apm) if apm else -1]:.2f}%")
    print(f"  Large objects - Precision: {apl[-1]:.2f}%, Recall: {ar[np.argmax(apl) if apl else -1]:.2f}%")
    
    return {
        'latest_precision': latest_precision,
        'latest_recall': latest_recall,
        'best_precision': best_precision,
        'best_recall': best_recall
    }

# 调用函数计算精确率和召回率
precision_recall_metrics = calculate_precision_recall()

# 可视化精确率与召回率的关系
def plot_precision_recall_curve():
    """Plot precision-recall curve"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot AP and AR curves
    ax.plot(epochs, ap, 'b-', marker='o', label='Precision (AP@0.5:0.95)', linewidth=2)
    ax.plot(epochs, ar, 'r-', marker='s', label='Recall (AR@0.5:0.95)', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Precision and Recall Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save chart
    pr_curve_path = os.path.join(OUTPUT_DIR, 'precision_recall_curve.png')
    plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
    print(f"Precision-recall curve saved to: {pr_curve_path}")
    
    plt.show()

# 绘制精确率与召回率曲线
plot_precision_recall_curve()