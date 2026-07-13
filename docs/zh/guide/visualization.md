# 可视化 / Visualization

## 1. 目录结构 / Directory Layout

```
visualization/
├── human/                  # Human 数据集可视化
│   ├── mrmodn/             # mRModN 模型可视化
│   │   ├── attention_comparison.py
│   │   ├── spatial_motif.py
│   │   ├── spatial_motif_nobg.py
│   │   ├── select_representatives.py
│   │   ├── view_v3.py / view_total.py / view_data.py
│   │   ├── run_attention_comparison.py
│   │   ├── run_attention_comparison_v2.py
│   │   └── visualize_attention_comparison_v2.py
│   ├── modx/               # ModX 模型可视化
│   │   └── inference_full.py
│   └── (其他数据集目录)
├── tools/                  # 通用工具
│   ├── prepare_umap_data.py
│   ├── prepare_umap_from_npz.py
│   └── view_npz.py
├── notebooks/              # Jupyter 交互式分析
├── fig/                    # 静态图（gitignored）
├── att_fig/                # 注意力图
└── motif_logo/             # motif logo
```

## 2. 注意力对比可视化 / Attention Comparison

**文件 / File**: `visualization/human/mrmodn/attention_comparison.py`

**功能 / Purpose**: 比较多模型（mRModN、ModX、MultiRM）在同一序列上的注意力分布

Compares attention distributions of multiple models (mRModN, ModX, MultiRM) on the same sequence.

```bash
uv run python visualization/human/mrmodn/run_attention_comparison_v2.py \
    --models mrmodn modx multirm \
    --sequence examples/seq1.fa \
    --output fig/attention_comparison.png
```

**输出 / Output**: 12 类（每类一个子图）注意力热图叠加 motif logo

12-class (one subplot per class) attention heatmap overlaid with motif logo.

## 3. 空间 Motif 可视化 / Spatial Motif

**文件 / File**: `visualization/human/mrmodn/spatial_motif.py`

**功能 / Purpose**: 提取每类修饰的空间 motif 模式

Extracts spatial motif patterns for each modification class.

```python
from visualization.human.mrmodn.spatial_motif import (
    extract_spatial_motifs,
    plot_spatial_motifs
)

motifs = extract_spatial_motifs(
    sequences=test_seqs,
    attentions=test_attns,  # (N, 12, L)
    top_k=20
)
plot_spatial_motifs(motifs, save_path="fig/spatial_motif/")
```

## 4. UMAP 降维可视化 / UMAP Visualization

**文件 / File**: `visualization/tools/prepare_umap_data.py`

**功能 / Purpose**: 将高维特征（128 维 GCN 输出）降维到 2D 用于可视化

Reduces high-dim features (128-d GCN output) to 2D for visualization.

```python
from visualization.tools.prepare_umap_data import (
    extract_features, run_umap, plot_umap
)

# 1. 提取特征
features = extract_features(model, dataloader, device="cuda")
# features: (N, 128)

# 2. UMAP 降维
coords_2d = run_umap(features, n_neighbors=15, min_dist=0.1)
# coords_2d: (N, 2)

# 3. 绘制
plot_umap(coords_2d, labels=dataset.y, save_path="fig/umap.png")
```

**输出**: 散点图，12 类用不同颜色

**Output**: scatter plot with 12 colors.

## 5. R 脚本可视化 / R Visualization

**文件 / File**: `analysis/plot_zero_fewshot_analysis.R`

**功能**: 生成零样本/少样本分析的发表级图表

**Purpose**: Generate publication-quality figures for zero-shot/few-shot analysis.

```bash
Rscript analysis/plot_zero_fewshot_analysis.R \
    --csv output/zeroshot_results.csv \
    --out fig/zero_fewshot.pdf
```

**生成图表 / Generated plots**:
- 各类别 F1 柱状图 / Per-class F1 bar plot
- 少样本规模 vs 性能曲线 / Few-shot size vs performance
- 跨物种迁移热图 / Cross-species transfer heatmap

## 6. Jupyter Notebook

启动 JupyterLab：

```bash
jupyter lab visualization/notebooks/
```

可用 notebook：
- `view_gen3.ipynb` — Gen3 数据集交互式分析
- `view_total.ipynb` — 全部数据概览
- `rna_visualization_new.ipynb` — 新的 RNA 可视化流程

## 7. `view_npz.py` 工具 / `view_npz.py` Tool

**文件 / File**: `visualization/tools/view_npz.py`

**功能**: 快速查看 `.npz` 推理结果文件的内容

Quickly inspect the contents of `.npz` inference result files.

```bash
uv run python visualization/tools/view_npz.py output/inference/human_mrmodn_full.npz
```

输出各数组的形状、统计量（min/max/mean）和示例数据。

Prints shape, statistics (min/max/mean), and sample data of each array.

## 8. 注意力权重提取 / Attention Weight Extraction

使用 `collect_atten_*.py` 系列脚本：

Use the `collect_atten_*.py` series:

```bash
uv run python collect_atten_human_mrmodn.py \
    --checkpoint output/human_mrmodn/best.pt \
    --output output/atten/human_mrmodn.npz
```

输出：
- `attn` (N, 12, L) — 12 类注意力
- `logits` (N, 12) — 12 类 logits
- `labels` (N, 12) — 真实标签
- `sequences` (N, L) — 原始 one-hot 序列

## 9. 配色与样式 / Colors & Style

可视化使用统一的配色方案：

The visualization uses a unified color scheme:

- **正样本 / Positive**: `#0a7d4d` (深绿 / dark green)
- **负样本 / Negative**: `#cccccc` (浅灰 / light gray)
- **12 类修饰颜色 / 12-class colors**: `tab20` colormap

## 10. 下一步 / Next Steps

- 消融实验：[消融实验](/guide/ablation)
- 少样本分析：[少/零样本分析](/guide/fewshot)
