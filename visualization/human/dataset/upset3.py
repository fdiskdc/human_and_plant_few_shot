#!/usr/bin/env python3
"""
upset3.py - Python UpSetPlot version of upset2 / Python UpSetPlot 版 UpSet 图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from upsetplot import UpSet, from_indicators

mods = ['Am', 'Atol', 'Cm', 'Gm', 'Tm', 'Y', 'ac4C', 'm1A', 'm5C', 'm6A', 'm6Am', 'm7G']
y = np.load('/home/dc/vscode/vscode20260406/rgcnformer_sum/npy/human3/12loc.npy')
df = pd.DataFrame(y.astype(bool), columns=mods)

morandi_intersect = '#91A3B0'
morandi_set = '#957E6E'

upset_data = from_indicators(df)

plt.rcParams.update({'font.size': 16})

upset = UpSet(
    upset_data,
    subset_size='count',
    sort_by='cardinality',
    sort_categories_by='cardinality',
    min_subset_size=20,
    facecolor=morandi_intersect,
    show_counts=True,
    element_size=28,
)

fig = plt.figure(figsize=(10, 10))
upset.plot(fig=fig)

totals_ax = None
for ax in fig.get_axes():
    ax.title.set_size(16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(14)
    for text in ax.texts:
        if text.get_fontsize() > 12:
            text.set_fontsize(10)
    widths = [p.get_width() for p in ax.patches if hasattr(p, 'get_width')]
    if len(widths) >= 10 and len(set(round(w, -2) for w in widths[:3])) > 1:
        totals_ax = ax

if totals_ax is not None:
    totals_ax.set_xscale('log')
    totals_ax.set_xlim(500, 200000)
    totals_ax.set_xticks([1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5])
    totals_ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1000:.0f}K' if v >= 1000 else str(int(v))))
    totals_ax.set_xlabel('Set Size')
    totals_ax.autoscale_view(scalex=False)

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

dir_out = '/home/dc/vscode/vscode20260603/human_and_plant_few_shot/visualization/human/dataset/fig'
fig.savefig(f'{dir_out}/upset3.png', dpi=120, bbox_inches='tight', facecolor='white')
fig.savefig(f'{dir_out}/upset3.pdf', bbox_inches='tight', facecolor='white')

print('Done — upset3.png and upset3.pdf saved.')
