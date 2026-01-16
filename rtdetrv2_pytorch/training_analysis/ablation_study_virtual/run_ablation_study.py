#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验主运行脚本
依次执行数据提取、表格生成、可视化和报告生成
"""

import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from extract_ablation_data import AblationDataExtractor
from generate_ablation_table import AblationTableGenerator
from visualize_ablation import AblationVisualizer
from generate_ablation_report import AblationReportGenerator


def main():
    """主函数"""
    print("=" * 60)
    print("痰液肺癌细胞检测消融实验分析")
    print("=" * 60)
    print()
    
    base_dir = Path(__file__).parent
    output_base_dir = Path("/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output")
    
    # 步骤1: 提取数据
    print("步骤 1/4: 提取消融实验数据...")
    print("-" * 60)
    extractor = AblationDataExtractor(output_base_dir)
    df = extractor.extract_ablation_data()
    extractor.save_data(df)
    print()
    
    # 步骤2: 生成表格
    print("步骤 2/4: 生成消融实验表格...")
    print("-" * 60)
    data_dir = base_dir / "data"
    table_generator = AblationTableGenerator(data_dir)
    table_generator.save_tables()
    print()
    
    # 步骤3: 生成可视化图表
    print("步骤 3/4: 生成可视化图表...")
    print("-" * 60)
    visualizer = AblationVisualizer(data_dir)
    visualizer.generate_all_plots()
    print()
    
    # 步骤4: 生成报告
    print("步骤 4/4: 生成消融实验报告...")
    print("-" * 60)
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"
    report_generator = AblationReportGenerator(data_dir, tables_dir, figures_dir)
    report_path = report_generator.save_report()
    print()
    
    # 总结
    print("=" * 60)
    print("消融实验分析完成！")
    print("=" * 60)
    print()
    print("生成的文件:")
    print(f"  数据文件: {data_dir}")
    print(f"  表格文件: {tables_dir}")
    print(f"  图表文件: {figures_dir}")
    print(f"  报告文件: {report_path}")
    print()
    print("查看完整报告:")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()

