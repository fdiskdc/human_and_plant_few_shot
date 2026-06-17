"""
ipynb/para_comp/generate_bubble_plot.py - 生成气泡图(参数对比) / Generate bubble plot (parameter comparison)

生成气泡图(参数对比)
Generate bubble plot (parameter comparison)

功能模块 / Modules:
- 三参数 (x/y/size) 气泡图,展示超参扫描
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
- 调用 / Calls: ipynb/cls_compare*.ipynb、plot_zero_fewshot_*.R
- 被调用 / Called by: ipynb/cls_compare*.ipynb、plot_zero_fewshot_*.R 相关的训练/推理/分析脚本

使用示例 / Usage Example:
    python ipynb/para_comp/generate_bubble_plot.py

作者 / Author: 项目组 / Project Team
版本 / Version: 1.0
"""

#!/usr/bin/env python3
"""
Generate bubble chart for model performance vs parameter count across different sequence lengths.
Morandi color palette styling.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for academic publication
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3

# Morandi color palette
MORANDI_COLORS = {
    51: '#94a7ae',   # 灰蓝 (gray blue)
    101: '#b99d8e',  # 灰褐 (gray brown)
    1001: '#a6bba1', # 豆绿 (bean green)
}

# Read the three Excel files
df51 = pd.read_excel('51.xlsx', sheet_name='Sheet1')
df101 = pd.read_excel('101.xlsx', sheet_name='Sheet1')
df1001 = pd.read_excel('1001.xlsx', sheet_name='Sheet1')

# Rename the first column to 'Model' and add Sequence_Length
df51 = df51.rename(columns={'Unnamed: 0': 'Model'})
df101 = df101.rename(columns={'Unnamed: 0': 'Model'})
df1001 = df1001.rename(columns={'Unnamed: 0': 'Model'})

df51['Sequence_Length'] = 51
df101['Sequence_Length'] = 101
df1001['Sequence_Length'] = 1001

# Merge all data
df = pd.concat([df51, df101, df1001], ignore_index=True)

# Ensure PARA and AUC are numeric
df['PARA'] = pd.to_numeric(df['PARA'])
df['AUC'] = pd.to_numeric(df['AUC'])

print("Merged data:")
print(df)

# Create the bubble chart
fig, ax = plt.subplots(figsize=(10, 7))

# Plot each sequence length separately for better control
for seq_len in [51, 101, 1001]:
    data = df[df['Sequence_Length'] == seq_len]
    ax.scatter(
        data['PARA'],
        data['AUC'],
        s=400,  # fixed bubble size
        alpha=0.75,
        color=MORANDI_COLORS[seq_len],
        label=f'Sequence Length {seq_len}',
        edgecolors='white',
        linewidth=1.5
    )

# Set log scale for x-axis if PARA spans orders of magnitude
ax.set_xscale('log')

# Axis labels
ax.set_xlabel('Parameters (log scale)', fontsize=13, fontweight='bold')
ax.set_ylabel('AUC (%)', fontsize=13, fontweight='bold')

# Title
ax.set_title('Model Performance vs Parameter Count\nAcross Different Sequence Lengths',
             fontsize=14, fontweight='bold', pad=15)

# Legend
legend = ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=False)
legend.get_frame().set_edgecolor('#cccccc')
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.95)

# Add model labels with adjusted positions to avoid overlap
for idx, row in df.iterrows():
    model = row['Model']
    x = row['PARA']
    y = row['AUC']
    seq_len = row['Sequence_Length']

    # Calculate offset based on sequence length to avoid overlap
    if seq_len == 51:
        offset_y = -1.5
        offset_x = 1.03
        ha = 'left'
    elif seq_len == 101:
        offset_y = 1.5
        offset_x = 1.03
        ha = 'left'
    else:  # 1001
        offset_y = 0
        offset_x = 1.08
        ha = 'left'

    ax.annotate(model, (x, y),
                xytext=(offset_x, offset_y),
                textcoords='axes fraction',
                fontsize=9,
                ha=ha,
                va='center',
                color=MORANDI_COLORS[seq_len],
                fontweight='500')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Set background color
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('white')

# Grid styling
ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)

# Set y-axis limits with some padding
y_min = df['AUC'].min() - 5
y_max = df['AUC'].max() + 5
ax.set_ylim(y_min, y_max)

# Tight layout
plt.tight_layout()

# Save as high-resolution PNG and PDF
plt.savefig('bubble_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('bubble_chart.pdf', bbox_inches='tight', facecolor='white')

print("\nBubble chart saved as 'bubble_chart.png' and 'bubble_chart.pdf'")

# Also save merged data as CSV for reference
df.to_csv('merged_data.csv', index=False)
print("Merged data saved as 'merged_data.csv'")
