#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验可视化脚本
生成各种评估效果图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class AblationVisualizer:
    """消融实验可视化器"""
    
    def __init__(self, data_dir: Path):
        """
        初始化可视化器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        self.output_dir = Path(__file__).parent / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        csv_path = data_dir / "ablation_data.csv"
        if csv_path.exists():
            self.df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"数据文件不存在: {csv_path}")
        
        # 实验顺序（用于排序）
        self.exp_order = [
            "Baseline", "Mamba", "HIFM", "DSCA",
            "Mamba+HIFM", "Mamba+DSCA", "HIFM+DSCA", "Full"
        ]
        self.df['Order'] = self.df['Experiment_ID'].apply(
            lambda x: self.exp_order.index(x) if x in self.exp_order else 999
        )
        self.df = self.df.sort_values('Order')
    
    def plot_ap_comparison(self):
        """绘制AP指标对比柱状图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('消融实验 - AP指标对比', fontsize=16, fontweight='bold')
        
        # AP@0.50:0.95
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(self.df)), self.df['AP'], 
                       color=sns.color_palette("husl", len(self.df)))
        ax1.set_xticks(range(len(self.df)))
        ax1.set_xticklabels(self.df['Name'], rotation=45, ha='right')
        ax1.set_ylabel('AP@0.50:0.95', fontsize=12)
        ax1.set_title('平均精度 (AP@0.50:0.95)', fontsize=14)
        ax1.grid(axis='y', alpha=0.3)
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars1, self.df['AP'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # AP@0.50
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(self.df)), self.df['AP50'],
                       color=sns.color_palette("husl", len(self.df)))
        ax2.set_xticks(range(len(self.df)))
        ax2.set_xticklabels(self.df['Name'], rotation=45, ha='right')
        ax2.set_ylabel('AP@0.50', fontsize=12)
        ax2.set_title('平均精度 (AP@0.50)', fontsize=14)
        ax2.grid(axis='y', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars2, self.df['AP50'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # AP@0.75
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(self.df)), self.df['AP75'],
                       color=sns.color_palette("husl", len(self.df)))
        ax3.set_xticks(range(len(self.df)))
        ax3.set_xticklabels(self.df['Name'], rotation=45, ha='right')
        ax3.set_ylabel('AP@0.75', fontsize=12)
        ax3.set_title('平均精度 (AP@0.75)', fontsize=14)
        ax3.grid(axis='y', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars3, self.df['AP75'])):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # AR@100
        ax4 = axes[1, 1]
        bars4 = ax4.bar(range(len(self.df)), self.df['AR100'],
                       color=sns.color_palette("husl", len(self.df)))
        ax4.set_xticks(range(len(self.df)))
        ax4.set_xticklabels(self.df['Name'], rotation=45, ha='right')
        ax4.set_ylabel('AR@100', fontsize=12)
        ax4.set_title('平均召回率 (AR@100)', fontsize=14)
        ax4.grid(axis='y', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars4, self.df['AR100'])):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ap_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {self.output_dir / 'ap_comparison.png'}")
    
    def plot_ap_by_size(self):
        """绘制不同尺寸目标的AP对比"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.df))
        width = 0.25
        
        bars1 = ax.bar(x - width, self.df['APs'], width, label='AP$_{s}$ (Small)', alpha=0.8)
        bars2 = ax.bar(x, self.df['APm'], width, label='AP$_{m}$ (Medium)', alpha=0.8)
        bars3 = ax.bar(x + width, self.df['APl'], width, label='AP$_{l}$ (Large)', alpha=0.8)
        
        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel('平均精度 (AP)', fontsize=12)
        ax.set_title('不同目标尺寸的AP对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.df['Name'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ap_by_size.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {self.output_dir / 'ap_by_size.png'}")
    
    def plot_component_contribution(self):
        """绘制组件贡献分析图"""
        # 计算相对于Baseline的提升
        baseline_df = self.df[self.df['Experiment_ID'] == 'Baseline']
        if len(baseline_df) > 0:
            baseline_ap = baseline_df['AP'].values[0]
            self.df['AP_Improvement'] = self.df['AP'] - baseline_ap
        else:
            self.df['AP_Improvement'] = 0.0
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['red' if x < 0 else 'green' for x in self.df['AP_Improvement']]
        bars = ax.barh(range(len(self.df)), self.df['AP_Improvement'], color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(self.df)))
        ax.set_yticklabels(self.df['Name'])
        ax.set_xlabel('AP提升 (相对于Baseline)', fontsize=12)
        ax.set_title('组件贡献分析 - AP提升', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, self.df['AP_Improvement'])):
            ax.text(val + (0.001 if val >= 0 else -0.001), bar.get_y() + bar.get_height()/2,
                   f'{val:+.3f}', ha='left' if val >= 0 else 'right', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'component_contribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {self.output_dir / 'component_contribution.png'}")
    
    def plot_classification_accuracy(self):
        """绘制分类准确率对比"""
        if 'Overall_Accuracy' not in self.df.columns:
            print("警告: 分类准确率数据不存在，跳过此图")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 总体准确率
        ax1 = axes[0]
        bars1 = ax1.bar(range(len(self.df)), self.df['Overall_Accuracy'],
                       color=sns.color_palette("husl", len(self.df)))
        ax1.set_xticks(range(len(self.df)))
        ax1.set_xticklabels(self.df['Name'], rotation=45, ha='right')
        ax1.set_ylabel('准确率', fontsize=12)
        ax1.set_title('总体分类准确率', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars1, self.df['Overall_Accuracy'])):
            if not pd.isna(val):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 平均类别准确率
        ax2 = axes[1]
        bars2 = ax2.bar(range(len(self.df)), self.df['Mean_Class_Accuracy'],
                       color=sns.color_palette("husl", len(self.df)))
        ax2.set_xticks(range(len(self.df)))
        ax2.set_xticklabels(self.df['Name'], rotation=45, ha='right')
        ax2.set_ylabel('准确率', fontsize=12)
        ax2.set_title('平均类别准确率', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars2, self.df['Mean_Class_Accuracy'])):
            if not pd.isna(val):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'classification_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {self.output_dir / 'classification_accuracy.png'}")
    
    def plot_radar_chart(self):
        """绘制雷达图对比"""
        # 选择关键指标
        metrics = ['AP', 'AP50', 'AP75', 'AR100']
        
        # 归一化到0-1范围
        df_normalized = self.df[metrics].copy()
        for col in metrics:
            max_val = df_normalized[col].max()
            min_val = df_normalized[col].min()
            if max_val > min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        # 选择几个关键实验进行对比
        key_experiments = ['Baseline', 'Mamba', 'HIFM', 'DSCA', 'Full']
        available_experiments = [exp for exp in key_experiments if exp in self.df['Experiment_ID'].values]
        if len(available_experiments) == 0:
            print("警告: 没有可用的实验数据用于雷达图，跳过此图")
            return
        
        df_key = self.df[self.df['Experiment_ID'].isin(available_experiments)].copy()
        df_normalized_key = df_normalized[self.df['Experiment_ID'].isin(available_experiments)].copy()
        
        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for idx, row in df_key.iterrows():
            values = df_normalized_key.loc[idx, metrics].tolist()
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Name'])
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('消融实验雷达图对比', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {self.output_dir / 'radar_chart.png'}")
    
    def generate_all_plots(self):
        """生成所有图表"""
        print("开始生成消融实验可视化图表...")
        self.plot_ap_comparison()
        self.plot_ap_by_size()
        self.plot_component_contribution()
        self.plot_classification_accuracy()
        self.plot_radar_chart()
        print(f"\n所有图表已保存到: {self.output_dir}")


def main():
    """主函数"""
    data_dir = Path(__file__).parent / "data"
    
    visualizer = AblationVisualizer(data_dir)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()

