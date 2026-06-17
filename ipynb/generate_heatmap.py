"""
ipynb/generate_heatmap.py - 生成热力图 / Generate heatmap

生成热力图
Generate heatmap

功能模块 / Modules:
- 基于矩阵数据生成 PNG/PDF 热力图
- (详见源代码 / see source code)

输入 / Inputs:
- 命令行参数(超参、数据路径等)/ CLI args (hyperparams, data paths, etc.)
- 配置文件(json/yaml) / config files (json/yaml)
- 上一阶段产物(.npy/.pt/.json)/ prior-stage outputs

输出 / Outputs:
- 产物文件(.pt/.json/.npy/.png/.pdf)/ output files
- 标准输出 / 日志 / stdout / logs

数据流 / Data Flow:
1. 加载数据/配置 / Load data/config
2. 主循环(训练/推理/分析)/ Main loop
3. 保存/导出 / Save/export

相关文件 / Related Files:
- 调用 / Calls: ipynb/cls_compare*.ipynb、utils/rna_visualization.py
- 被调用 / Called by: ipynb/cls_compare*.ipynb、utils/rna_visualization.py 相关的训练/推理/分析脚本

使用示例 / Usage Example:
    python ipynb/generate_heatmap.py

作者 / Author: 项目组 / Project Team
版本 / Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 1. 读取数据
df = pd.read_csv('data/res.csv')

# 2. 定义 RNA 修改类型和评估指标顺序
class_order = ['m6A', 'm1A', 'm5C', 'm6Am', 'm7G', 'ac4C', 'Am', 'Cm', 'Gm', 'Tm', 'Y', 'Atol']
metric_order = ['Acc', 'Precision', 'Recall', 'AUC', 'F1', 'MCC', 'AUPRC', 'Sn', 'Sp']

# 3. 确保 Class 列有正确的顺序
df['Mod Name'] = pd.Categorical(df['Mod Name'], categories=class_order, ordered=True)
df = df.sort_values('Mod Name')

# 4. 创建数据矩阵：12行（RNA修改类型）x 9列（评估指标）
heatmap_data = df.set_index('Mod Name')[metric_order].values * 100  # 转换为百分比

# 5. 设置字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

# 6. 创建热图
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# 使用 seaborn.heatmap
sns.set_style("white")
cmap = sns.color_palette("Blues", as_cmap=True)

im = sns.heatmap(
    heatmap_data,
    ax=ax,
    cmap=cmap,
    annot=True,
    fmt='.2f',
    annot_kws={"size": 8, "color": "black"},
    xticklabels=metric_order,
    yticklabels=class_order,
    vmin=0,
    vmax=100,
    linewidths=0.5,
    linecolor='lightgray',
    cbar=True,
    cbar_kws={
        "label": "Performance Value (%)",
        "shrink": 0.8,
        "ticks": [0, 20, 40, 60, 80, 100]
    }
)

# 7. 调整文本颜色（深色背景用白色，浅色背景用黑色）
for i in range(len(class_order)):
    for j in range(len(metric_order)):
        val = heatmap_data[i, j]
        text = ax.texts[i * len(metric_order) + j]
        if val > 75:  # 高值用白色文字
            text.set_color('white')
        else:  # 低值用黑色文字
            text.set_color('black')

# 8. 调整坐标轴和图表样式
ax.set_xlabel('')
ax.set_ylabel('')
ax.xaxis.tick_top()  # X轴标签在顶部
ax.xaxis.set_label_position('top')

# 设置标题和刻度样式
plt.xticks(rotation=45, ha='left')
plt.yticks(rotation=0)

# 去掉多余边框
sns.despine(left=True, bottom=True)

# 9. 保存图表
plt.tight_layout()
output_path = 'png2/rgcnformer_cls_heatmap.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"热图已保存至: {output_path}")

# 也保存为 PNG 格式
output_png = 'png2/rgcnformer_cls_heatmap.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"热图已保存至: {output_png}")

plt.close()

# 验证数据一致性
print("\n=== 数据验证 ===")
y_idx = class_order.index('Y')
auc_idx = metric_order.index('AUC')
print(f"Y 的 AUC 值: {heatmap_data[y_idx, auc_idx]:.2f}%")

# 检查 TreeX 在大多数修改上的 AUC
print("\n各修改类型的 AUC 值:")
for i, cls in enumerate(class_order):
    auc_val = heatmap_data[i, auc_idx]
    print(f"  {cls}: {auc_val:.2f}%")