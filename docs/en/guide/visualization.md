# Visualization

## 1. Directory Layout

```
visualization/
├── human/                  # Human dataset viz
│   ├── mrmodn/             # mRModN model viz
│   │   ├── attention_comparison.py
│   │   ├── spatial_motif.py
│   │   ├── spatial_motif_nobg.py
│   │   ├── select_representatives.py
│   │   ├── view_v3.py / view_total.py / view_data.py
│   │   ├── run_attention_comparison.py
│   │   ├── run_attention_comparison_v2.py
│   │   └── visualize_attention_comparison_v2.py
│   ├── modx/               # ModX model viz
│   │   └── inference_full.py
│   └── (other dataset dirs)
├── tools/                  # Common tools
│   ├── prepare_umap_data.py
│   ├── prepare_umap_from_npz.py
│   └── view_npz.py
├── notebooks/              # Jupyter interactive analysis
├── fig/                    # Static figures (gitignored)
├── att_fig/                # Attention figures
└── motif_logo/             # Motif logos
```

## 2. Attention Comparison

**File**: `visualization/human/mrmodn/attention_comparison.py`

**Purpose**: Compares attention distributions of multiple models (mRModN, ModX, MultiRM) on the same sequence.

```bash
uv run python visualization/human/mrmodn/run_attention_comparison_v2.py \
    --models mrmodn modx multirm \
    --sequence examples/seq1.fa \
    --output fig/attention_comparison.png
```

**Output**: 12-class (one subplot per class) attention heatmap overlaid with motif logo.

## 3. Spatial Motif

**File**: `visualization/human/mrmodn/spatial_motif.py`

**Purpose**: Extracts spatial motif patterns for each modification class.

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

## 4. UMAP Visualization

**File**: `visualization/tools/prepare_umap_data.py`

**Purpose**: Reduces high-dim features (128-d GCN output) to 2D for visualization.

```python
from visualization.tools.prepare_umap_data import (
    extract_features, run_umap, plot_umap
)

# 1. Extract features
features = extract_features(model, dataloader, device="cuda")
# features: (N, 128)

# 2. UMAP reduction
coords_2d = run_umap(features, n_neighbors=15, min_dist=0.1)
# coords_2d: (N, 2)

# 3. Plot
plot_umap(coords_2d, labels=dataset.y, save_path="fig/umap.png")
```

**Output**: scatter plot with 12 colors.

## 5. R Visualization

**File**: `analysis/plot_zero_fewshot_analysis.R`

**Purpose**: Generate publication-quality figures for zero-shot/few-shot analysis.

```bash
Rscript analysis/plot_zero_fewshot_analysis.R \
    --csv output/zeroshot_results.csv \
    --out fig/zero_fewshot.pdf
```

**Generated plots**:
- Per-class F1 bar plot
- Few-shot size vs performance
- Cross-species transfer heatmap

## 6. Jupyter Notebooks

```bash
jupyter lab visualization/notebooks/
```

Available notebooks:
- `view_gen3.ipynb` — Interactive Gen3 analysis
- `view_total.ipynb` — Full data overview
- `rna_visualization_new.ipynb` — New RNA visualization pipeline

## 7. `view_npz.py` Tool

**File**: `visualization/tools/view_npz.py`

**Purpose**: Quickly inspect the contents of `.npz` inference result files.

```bash
uv run python visualization/tools/view_npz.py output/inference/human_mrmodn_full.npz
```

Prints shape, statistics (min/max/mean), and sample data of each array.

## 8. Attention Weight Extraction

Use the `collect_atten_*.py` series:

```bash
uv run python collect_atten_human_mrmodn.py \
    --checkpoint output/human_mrmodn/best.pt \
    --output output/atten/human_mrmodn.npz
```

Output:
- `attn` (N, 12, L) — 12-class attention
- `logits` (N, 12) — 12-class logits
- `labels` (N, 12) — true labels
- `sequences` (N, L) — raw one-hot sequences

## 9. Colors & Style

The visualization uses a unified color scheme:

- **Positive**: `#0a7d4d` (dark green)
- **Negative**: `#cccccc` (light gray)
- **12-class colors**: `tab20` colormap

## 10. Next Steps

- Ablation: [Ablation Studies](/en/guide/ablation)
- Few-shot: [Few/Zero-shot Analysis](/en/guide/fewshot)
