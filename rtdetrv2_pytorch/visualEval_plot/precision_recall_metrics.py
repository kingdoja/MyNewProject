import json
import os
import numpy as np
import matplotlib.pyplot as plt

# 直接在代码中指定固定路径
LOG_FILE_PATH = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection/log.txt"
OUTPUT_DIR = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/plotEval"

def parse_log_file(log_file_path):
    """解析日志文件，提取评估指标"""
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at {log_file_path}")
        return None

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

    # 读取日志内容
    with open(log_file_path, 'r') as f:
        log_content = f.read()

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
    
    return {
        'epochs': epochs,
        'ap': ap,      # AP@0.5:0.95 (相当于mAP)
        'ap50': ap50,  # AP@0.5
        'ap75': ap75,
        'aps': aps,
        'apm': apm,
        'apl': apl,
        'ar': ar,      # AR@0.5:0.95 (相当于召回率)
        'ar50': ar50,
        'ar75': ar75
    }

def calculate_precision_recall(metrics):
    """
    计算模型的精确率和召回率
    在目标检测中:
    - 精确率(Precision)通常通过Average Precision (AP)来衡量
    - 召回率(Recall)通常通过Average Recall (AR)来衡量
    """
    if not metrics or len(metrics['ap']) == 0:
        print("No metrics data available")
        return None

    epochs = metrics['epochs']
    ap = metrics['ap']      # AP@0.5:0.95 (mAP)
    ap50 = metrics['ap50']  # AP@0.5
    ar = metrics['ar']      # AR@0.5:0.95 (召回率)
    ar50 = metrics['ar50']
    aps = metrics['aps']
    apm = metrics['apm']
    apl = metrics['apl']

    print("\n=== 精确率与召回率分析 ===")
    
    # 最新评估结果的精确率和召回率
    latest_precision = ap[-1]  # AP即为精确率的综合指标
    latest_recall = ar[-1]     # AR即为召回率的综合指标
    
    print(f"最新精确率 (mAP@0.5:0.95): {latest_precision:.2f}%")
    print(f"最新召回率 (AR@0.5:0.95): {latest_recall:.2f}%")
    
    # 最佳评估结果的精确率和召回率
    best_precision = max(ap)
    best_recall = max(ar)
    best_precision_epoch = epochs[np.argmax(ap)]
    best_recall_epoch = epochs[np.argmax(ar)]
    
    print(f"最佳精确率 (mAP): {best_precision:.2f}% (Epoch {best_precision_epoch})")
    print(f"最佳召回率: {best_recall:.2f}% (Epoch {best_recall_epoch})")
    
    # IoU阈值为0.5时的精确率和召回率
    print(f"\nIoU=0.5时的精确率 (AP@0.5): {ap50[-1]:.2f}%")
    print(f"IoU=0.5时的召回率 (AR@0.5): {ar50[-1]:.2f}%")
    
    # IoU阈值为0.75时的精确率和召回率
    print(f"\nIoU=0.75时的精确率 (AP@0.75): {metrics['ap75'][-1]:.2f}%")
    print(f"IoU=0.75时的召回率 (AR@0.75): {metrics['ar75'][-1]:.2f}%")
    
    # 不同目标尺寸的精确率和召回率
    print(f"\n不同尺寸目标的性能:")
    print(f"  小目标 - 精确率: {aps[-1]:.2f}%, 召回率: {ar[np.argmax(aps) if aps else -1]:.2f}%")
    print(f"  中目标 - 精确率: {apm[-1]:.2f}%, 召回率: {ar[np.argmax(apm) if apm else -1]:.2f}%")
    print(f"  大目标 - 精确率: {apl[-1]:.2f}%, 召回率: {ar[np.argmax(apl) if apl else -1]:.2f}%")
    
    return {
        'latest_precision': latest_precision,  # mAP@0.5:0.95
        'latest_recall': latest_recall,        # AR@0.5:0.95
        'latest_map': latest_precision,        # mAP@0.5:0.95
        'latest_ap50': ap50[-1],               # AP@0.5
        'best_precision': best_precision,
        'best_recall': best_recall,
        'best_map': best_precision,
        'best_ap50': max(ap50) if ap50 else 0
    }

def plot_precision_recall_curve(metrics, output_dir):
    """绘制精确率-召回率曲线"""
    epochs = metrics['epochs']
    ap = metrics['ap']  # mAP
    ar = metrics['ar']  # 召回率
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制AP和AR的关系曲线
    ax.plot(epochs, ap, 'b-', marker='o', label='Precision (mAP@0.5:0.95)', linewidth=2)
    ax.plot(epochs, ar, 'r-', marker='s', label='Recall (AR@0.5:0.95)', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Precision and recall change with the training process')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    pr_curve_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
    print(f"精确率-召回率曲线已保存到: {pr_curve_path}")
    
    plt.show()

def print_final_results(metrics):
    """打印最终结果"""
    if not metrics or len(metrics['ap']) <= 1:
        return

    epochs = metrics['epochs']
    ap = metrics['ap']
    ap50 = metrics['ap50']
    ap75 = metrics['ap75']
    aps = metrics['aps']
    apm = metrics['apm']
    apl = metrics['apl']
    ar = metrics['ar']
    ar50 = metrics['ar50']
    ar75 = metrics['ar75']

    print("\n=== 最终评估结果 ===")
    print(f"最终mAP@0.5:0.95: {ap[-1]:.2f}% (Epoch {epochs[-1]})")
    print(f"  AP@0.5: {ap50[-1]:.2f}%")
    print(f"  AP@0.75: {ap75[-1]:.2f}%")
    print(f"  AP-small: {aps[-1]:.2f}%")
    print(f"  AP-medium: {apm[-1]:.2f}%")
    print(f"  AP-large: {apl[-1]:.2f}%")
    print(f"  AR@0.5:0.95: {ar[-1]:.2f}%")
    print(f"  AR@0.5: {ar50[-1]:.2f}%")
    print(f"  AR@0.75: {ar75[-1]:.2f}%")

def print_key_metrics_summary(precision_recall_metrics):
    """打印关键指标摘要"""
    if not precision_recall_metrics:
        return
        
    print("\n=== 关键指标摘要 ===")
    print(f"mAP@0.5:0.95: {precision_recall_metrics['latest_map']:.2f}%")
    print(f"AP@0.5: {precision_recall_metrics['latest_ap50']:.2f}%")
    print(f"Precision: {precision_recall_metrics['latest_precision']:.2f}%")
    print(f"Recall: {precision_recall_metrics['latest_recall']:.2f}%")

def main():
    # 解析日志文件
    metrics = parse_log_file(LOG_FILE_PATH)
    
    if metrics is None:
        return
    
    if len(metrics['ap']) == 0:
        print("Error: No evaluation data found in log file.")
        return

    # 打印最终结果
    print_final_results(metrics)
    
    # 计算和显示精确率与召回率的详细信息
    precision_recall_metrics = calculate_precision_recall(metrics)
    
    # 打印关键指标摘要
    print_key_metrics_summary(precision_recall_metrics)
    
    # 绘制精确率与召回率曲线
    plot_precision_recall_curve(metrics, OUTPUT_DIR)

if __name__ == "__main__":
    main()