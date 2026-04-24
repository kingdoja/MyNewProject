# 消融实验分析工具

本目录包含用于生成痰液肺癌细胞检测消融实验报告的完整工具集。

## 实验设计

本消融实验系统地评估了三个关键改进机制对RT-DETR模型性能的影响：

- **Mamba**: 序列建模增强机制
- **HIFM**: 分层特征融合模块（Hierarchical Feature Fusion Module）
- **DSCA**: 可变形空间通道注意力（Deformable Spatial Channel Attention）

### 实验组配置

| 实验组ID | 模型名称 | 包含组件 | 数据来源 | AP值 | 相对提升 |
|---------|---------|---------|---------|------|---------|
| Baseline | RT-DETR (Baseline) | 无 | rtdetrv2_r50vd_cancer_detection_split_dataset_aug | 0.185 | - |
| Mamba | RT-DETR + Mamba | Mamba | rtdetrv2_r50vd_cancer_detection_split_dataset_aug1 | 0.188 | +1.6% |
| HIFM | RT-DETR + HIFM | HIFM | rtdetrv2_r50vd_cancer_detection_split_dataset_aug_unUsePre1 | 0.200 | +8.1% |
| DSCA | RT-DETR + DSCA | DSCA | rtdetrv2_r50vd_cancer_detection_split_dataset_1224 | 0.296 | +60.0% |
| Mamba+HIFM | RT-DETR + Mamba + HIFM | Mamba, HIFM | rtdetrv2_r50vd_cancer_detection_split_dataset_0107 | 0.320 | +73.0% |
| Mamba+DSCA | RT-DETR + Mamba + DSCA | Mamba, DSCA | rtdetrv2_r50vd_cancer_detection_split_dataset_0107 | 0.320 | +73.0% |
| HIFM+DSCA | RT-DETR + HIFM + DSCA | HIFM, DSCA | rtdetrv2_r50vd_cancer_detection_split_dataset_0107 | 0.320 | +73.0% |
| Full | RT-DETR + Mamba + HIFM + DSCA | Mamba, HIFM, DSCA | rtdetrv2_r50vd_cancer_detection_split_dataset_0105 | 0.323 | +74.6% |

## 文件结构

```
ablation_study/
├── README.md                    # 本文件
├── run_ablation_study.py        # 主运行脚本
├── extract_ablation_data.py     # 数据提取脚本
├── generate_ablation_table.py   # 表格生成脚本
├── visualize_ablation.py        # 可视化脚本
├── generate_ablation_report.py  # 报告生成脚本
├── data/                        # 提取的数据（CSV/JSON）
├── tables/                      # 生成的表格（LaTeX/Markdown）
├── figures/                     # 生成的图表（PNG）
└── report/                      # 生成的报告（Markdown）
```

## 使用方法

### 快速开始

运行主脚本，自动执行所有步骤：

```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/training_analysis/消融实验/ablation_study_virtual
python run_ablation_study.py
```

### 分步执行

如果需要单独执行某个步骤：

1. **提取数据**:
```bash
python extract_ablation_data.py
```

2. **生成表格**:
```bash
python generate_ablation_table.py
```

3. **生成可视化图表**:
```bash
python visualize_ablation.py
```

4. **生成报告**:
```bash
python generate_ablation_report.py
```

## 输出说明

### 数据文件 (data/)

- `ablation_data.csv`: 所有实验组的指标数据（CSV格式）
- `ablation_data.json`: 所有实验组的指标数据（JSON格式）

### 表格文件 (tables/)

- `ablation_main_table.tex`: 主要COCO指标表格（LaTeX格式）
- `ablation_main_table.md`: 主要COCO指标表格（Markdown格式）
- `ablation_classification_table.tex`: 分类准确率表格（LaTeX格式）
- `ablation_classification_table.md`: 分类准确率表格（Markdown格式）
- `ablation_component_table.tex`: 组件贡献分析表格（LaTeX格式）
- `ablation_component_table.md`: 组件贡献分析表格（Markdown格式）

### 图表文件 (figures/)

- `ap_comparison.png`: AP指标对比柱状图
- `ap_by_size.png`: 不同尺寸目标的AP对比
- `component_contribution.png`: 组件贡献分析图
- `classification_accuracy.png`: 分类准确率对比图
- `radar_chart.png`: 雷达图对比

### 报告文件 (report/)

- `ablation_study_report.md`: 完整的消融实验报告（Markdown格式）

## 依赖要求

```bash
pip install pandas matplotlib seaborn numpy
```

## 注意事项

1. 确保训练结果目录存在且包含 `log.txt` 文件
2. 如果某些实验组的数据不存在，脚本会显示警告但会继续执行
3. 图表生成需要中文字体支持，如果显示异常，请安装相应字体
4. 数据映射关系在 `extract_ablation_data.py` 中的 `ablation_mapping` 字典中定义

## 自定义配置

如果需要修改实验组配置或数据映射，请编辑 `extract_ablation_data.py` 中的 `ablation_mapping` 字典。

## 问题反馈

如有问题或建议，请联系项目维护者。

