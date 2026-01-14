#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验表格生成脚本
生成LaTeX格式和Markdown格式的表格
"""

import pandas as pd
from pathlib import Path
import json


class AblationTableGenerator:
    """消融实验表格生成器"""
    
    def __init__(self, data_dir: Path):
        """
        初始化表格生成器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        self.output_dir = Path(__file__).parent / "tables"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        csv_path = data_dir / "ablation_data.csv"
        if csv_path.exists():
            self.df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"数据文件不存在: {csv_path}")
    
    def format_value(self, value, decimals=3):
        """格式化数值"""
        if pd.isna(value):
            return "-"
        return f"{value:.{decimals}f}"
    
    def generate_main_table(self) -> str:
        """生成主要消融实验表格（COCO指标）"""
        # 选择主要指标
        columns = [
            "Name",
            "AP", "AP50", "AP75",
            "APs", "APm", "APl",
            "AR100"
        ]
        
        # 创建表格数据
        table_data = []
        for _, row in self.df.iterrows():
            table_row = {
                "Model": row["Name"],
                "AP": self.format_value(row.get("AP")),
                "AP$_{50}$": self.format_value(row.get("AP50")),
                "AP$_{75}$": self.format_value(row.get("AP75")),
                "AP$_{s}$": self.format_value(row.get("APs")),
                "AP$_{m}$": self.format_value(row.get("APm")),
                "AP$_{l}$": self.format_value(row.get("APl")),
                "AR$_{100}$": self.format_value(row.get("AR100")),
            }
            table_data.append(table_row)
        
        # 生成LaTeX表格
        latex = self._generate_latex_table(table_data, "主要消融实验结果 (COCO指标)")
        
        # 生成Markdown表格
        markdown = self._generate_markdown_table(table_data, "主要消融实验结果 (COCO指标)")
        
        return latex, markdown
    
    def generate_classification_table(self) -> str:
        """生成分类准确率表格"""
        table_data = []
        for _, row in self.df.iterrows():
            table_row = {
                "Model": row["Name"],
                "Overall Accuracy": self.format_value(row.get("Overall_Accuracy", 0), decimals=4),
                "Mean Class Accuracy": self.format_value(row.get("Mean_Class_Accuracy", 0), decimals=4),
            }
            table_data.append(table_row)
        
        latex = self._generate_latex_table(table_data, "分类准确率结果")
        markdown = self._generate_markdown_table(table_data, "分类准确率结果")
        
        return latex, markdown
    
    def generate_component_table(self) -> str:
        """生成组件贡献表格"""
        table_data = []
        for _, row in self.df.iterrows():
            components = row["Components"]
            # 处理NaN值
            if pd.isna(components):
                components = ""
            components_str = str(components) if components else ""
            has_mamba = "Mamba" in components_str
            has_hifm = "HIFM" in components_str
            has_dsca = "DSCA" in components_str
            
            table_row = {
                "Model": row["Name"],
                "Mamba": "✓" if has_mamba else "✗",
                "HIFM": "✓" if has_hifm else "✗",
                "DSCA": "✓" if has_dsca else "✗",
                "AP": self.format_value(row.get("AP")),
                "AP$_{50}$": self.format_value(row.get("AP50")),
                "ΔAP": self._calculate_delta_ap(row)
            }
            table_data.append(table_row)
        
        latex = self._generate_latex_table(table_data, "组件贡献分析")
        markdown = self._generate_markdown_table(table_data, "组件贡献分析")
        
        return latex, markdown
    
    def _calculate_delta_ap(self, row) -> str:
        """计算相对于Baseline的AP提升"""
        baseline_df = self.df[self.df["Experiment_ID"] == "Baseline"]
        if len(baseline_df) == 0:
            return "-"
        baseline_ap = baseline_df["AP"].values[0]
        current_ap = row.get("AP")
        if pd.isna(baseline_ap) or pd.isna(current_ap):
            return "-"
        delta = current_ap - baseline_ap
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.3f}"
    
    def _generate_latex_table(self, data: list, caption: str) -> str:
        """生成LaTeX表格"""
        if not data:
            return ""
        
        # 获取表头
        headers = list(data[0].keys())
        
        # 生成LaTeX代码
        latex = f"\\begin{{table}}[h]\n"
        latex += f"\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{tab:ablation}}\n"
        latex += f"\\begin{{tabular}}{{{'|'.join(['c'] * len(headers))}}}\n"
        latex += f"\\hline\n"
        
        # 表头
        header_row = " & ".join(headers) + " \\\\\n"
        latex += header_row
        latex += "\\hline\n"
        
        # 数据行
        for row in data:
            values = [str(row.get(h, "")) for h in headers]
            data_row = " & ".join(values) + " \\\\\n"
            latex += data_row
            latex += "\\hline\n"
        
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def _generate_markdown_table(self, data: list, caption: str) -> str:
        """生成Markdown表格"""
        if not data:
            return ""
        
        # 获取表头
        headers = list(data[0].keys())
        
        # 生成Markdown代码
        markdown = f"## {caption}\n\n"
        
        # 表头
        header_row = "| " + " | ".join(headers) + " |\n"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |\n"
        markdown += header_row
        markdown += separator
        
        # 数据行
        for row in data:
            values = [str(row.get(h, "")) for h in headers]
            data_row = "| " + " | ".join(values) + " |\n"
            markdown += data_row
        
        markdown += "\n"
        
        return markdown
    
    def save_tables(self):
        """保存所有表格"""
        # 主要表格
        latex_main, md_main = self.generate_main_table()
        self._save_file("ablation_main_table.tex", latex_main)
        self._save_file("ablation_main_table.md", md_main)
        
        # 分类表格
        latex_class, md_class = self.generate_classification_table()
        self._save_file("ablation_classification_table.tex", latex_class)
        self._save_file("ablation_classification_table.md", md_class)
        
        # 组件表格
        latex_comp, md_comp = self.generate_component_table()
        self._save_file("ablation_component_table.tex", latex_comp)
        self._save_file("ablation_component_table.md", md_comp)
        
        print(f"表格已保存到: {self.output_dir}")
    
    def _save_file(self, filename: str, content: str):
        """保存文件"""
        file_path = self.output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)


def main():
    """主函数"""
    data_dir = Path(__file__).parent / "data"
    
    generator = AblationTableGenerator(data_dir)
    generator.save_tables()
    
    print("\n生成的表格预览:")
    print("\n=== 主要表格 (Markdown) ===")
    _, md = generator.generate_main_table()
    print(md)


if __name__ == "__main__":
    main()

