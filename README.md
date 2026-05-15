# RGCNFormer

RNA Modification Multi-label Classification using CNN + GCN + Class-Query Attention

## Overview

RGCNFormer is a deep learning framework for predicting 12 types of RNA post-transcriptional modifications from 1001-nucleotide sequences. The core model (`RNA_ClassQuery_Model`) combines:

- **Parallel CNN**: Multi-scale convolutional kernels (1, 3, 5, 7) for local motif extraction
- **GCN**: Graph Convolutional Network propagating information along RNA secondary structure edges
- **Class-Query Attention**: Per-class cross-attention heads with hierarchical 4-group (A/C/G/U) -> 12-class derivation

### Supported Modification Types

| Index | Type  | Nucleotide | Index | Type  | Nucleotide |
|-------|-------|------------|-------|-------|------------|
| 0     | Am    | A          | 6     | ac4C  | C          |
| 1     | Atol  | A          | 7     | m1A   | A          |
| 2     | Cm    | C          | 8     | m5C   | C          |
| 3     | Gm    | G          | 9     | m6A   | A          |
| 4     | Tm    | U          | 10    | m6Am  | A          |
| 5     | Y     | U          | 11    | m7G   | G          |

### Baseline Models

- **MultiRM** (`model/multirm.py`): BiLSTM + Bahdanau Attention
- **ModX** (`model/modx.py`): BiLSTM + Bahdanau Attention with Word2Vec-style input
- **EvoRMD** (`model/evormd_human.py`): Conv1d Embedder + Trainable Attention + MLP
- **mRModN**: GCN + per-class MHA (full-length inference)

---

## Project Structure

```
rgcnformer_sum/
├── model/                          # Model definitions
│   ├── main_model.py               # Core RGCNFormer model (CNN + GCN + Class-Query)
│   ├── main_model_multirm.py       # Variant for MultIRM dataset
│   ├── main_model_collect_atten.py # Variant returning intermediate attention
│   ├── multirm.py                  # MultiRM baseline (BiLSTM + Bahdanau)
│   ├── multirm_collect_atten.py    # MultiRM with attention collection
│   ├── modx.py                     # ModX baseline (BiLSTM + Word2Vec)
│   ├── modx_collect_atten.py       # ModX with attention collection
│   ├── evormd_human.py             # EvoRMD baseline (Conv1d + Attention)
│   ├── abla_model.py               # Ablation study variants
│   └── EvoRMD/Script/              # Original EvoRMD reference implementation
│
├── dataset/                        # Dataset classes
│   ├── human.py                    # Human RNA modification dataset (primary)
│   ├── human_with_seq.py           # Enhanced human dataset with sequence strings
│   ├── human_motif.py              # Human dataset for spatial motif analysis
│   ├── plant.py                    # Plant RNA modification dataset
│   ├── plant_single.py             # Combined plant + zero dataset
│   ├── multirm.py                  # MultIRM 12-class dataset (HDF5)
│   ├── ac4c.py                     # AC4C modification dataset
│   ├── gen3.py                     # 3-generation dataset
│   └── gen3_zero.py                # Gen3 with background samples
│
├── utils/                          # Utility functions
│   ├── common.py                   # Core: constants, config, samplers, splits, training
│   ├── metrics.py                  # Evaluation metrics and threshold optimization
│   ├── logging.py                  # Logging and formatted table output
│   ├── few_shot.py                 # Few-shot learning benchmarks
│   ├── fewshot_analysis_*.py       # Few-shot analysis pipeline modules
│   ├── fewshot_export_helpers.py   # CSV/JSON export for R visualization
│   ├── rna_visualization.py        # RNA structure and attention visualization
│   ├── train_gen3.py               # Gen3 training script
│   └── Zero_structures.py          # Precompute structures for zero dataset
│
├── atten_comp/                     # Attention comparison scripts (v2)
│   ├── run_attention_comparison_v2.py
│   ├── visualize_attention_comparison_v2.py
│   └── inference_modx_full.py
│
├── json/                           # Configuration files
├── npy/                            # Dataset files (.npy)
├── cache/                          # Structure cache files
├── logs/                           # Training logs
│
├── train_human.py                  # Main human training script
├── train_plant.py                  # Plant training script
├── train_multirm_dataset.py        # MultIRM training script
├── train_human_multirm.py          # Human training with MultiRM model
├── train_human_modx.py             # Human training with ModX model
├── train_human_evormd.py           # Human training with EvoRMD model
│
├── inference_modx_segmented.py     # ModX sliding window inference
├── inference_mrmodn_full.py        # mRModN full-length inference
├── inference_multirm_segmented.py  # MultiRM sliding window inference
├── inference_evormd_segmented.py   # EvoRMD sliding window inference
│
├── fewshot_plant_3way_independent.py   # Plant 3-way few-shot
├── fewshot_ac4c_balance.py             # AC4C balanced few-shot
├── fewshot_ac4c_unbalan.py             # AC4C unbalanced few-shot
├── zero_shot_fewshot_analysis.py       # Zero/few-shot analysis pipeline
├── zero_shot_fewshot_extract_only.py   # Data export for R visualization
│
├── run_attention_comparison.py     # Attention comparison orchestrator
├── visualize_attention_comparison.py
├── select_representative_sequences.py
├── sliding_window_utils.py         # Gaussian-weighted stitching utilities
├── SpatialMotif.py                 # Spatial motif visualization
├── SpatialMotif_nobackground.py    # Spatial motif (clean version)
├── prepare_umap_data.py            # UMAP pipeline
├── prepare_umap_from_npz.py        # UMAP from pre-collected data
├── abla_mohe.py                    # Ablation study runner
├── cal_flops_mohe.py               # FLOPs calculator
├── collect_human_atten.py          # Collect human attention outputs
├── collect_human.py                # Collect human outputs to Excel
├── collect_modx_atten.py           # Collect ModX attention
├── collect_multirm_atten.py        # Collect MultiRM attention
├── view_human.py                   # Violin plots (per-class)
├── view_human_total.py             # Violin plot (aggregated)
├── view_npz.py                     # NPZ file inspector
├── view3.py                        # Multi-dataset violin plots
├── 3x3.py                          # Human-to-Plant zero-shot transfer
├── 3x3_2.py                        # Transfer analysis v2
├── test_gen3.py                    # Gen3 evaluation script
├── test_multirm_4class.py          # MultIRM 4-class test
├── test_multirm_oversampling.py    # MultIRM oversampling test
├── cal_mean_median_mode.py         # Per-class statistics
│
├── plot_zero_fewshot_analysis.R    # R: Publication figures from CSV
└── plot_zero_fewshot_export.R      # R: Export + figures from CSV/JSON
```

---

## Python Scripts

### Training Scripts

| Script | Description |
|--------|-------------|
| `train_human.py` | Main training script for human 12-class RNA modification classification. Multi-label disjoint split (7:3), smoothed class weighting, dynamic balanced batch sampling, dual evaluation (Unbalance & BalanceB), TensorBoard logging. |
| `train_plant.py` | Plant dataset training with few-shot benchmark support. Uses `RNA_ClassQuery_Model` with support set sampling. |
| `train_multirm_dataset.py` | Training on the MultIRM 12-class dataset (51nt sequences). Positive/negative pairing, per-class losses, HDF5 data. |
| `train_human_multirm.py` | Human training using the MultiRM baseline model (`model_v3`: BiLSTM + Bahdanau Attention). |
| `train_human_modx.py` | Human training using the ModX baseline model (`RNAClassifierWithWord2Vec`: BiLSTM + Bahdanau). |
| `train_human_evormd.py` | Human training using the EvoRMD baseline model (`EvoRMDForHuman`: Conv1d + Trainable Attention). |

### Inference Scripts

| Script | Description |
|--------|-------------|
| `inference_modx_segmented.py` | ModX inference with 101nt sliding window over 1001nt sequences. Collects per-class attention `[N, 12, 1001]`, Gaussian-weighted stitching. |
| `inference_mrmodn_full.py` | mRModN full-length 1001nt inference for attention collection. GCN + per-class MHA on full sequences. |
| `inference_multirm_segmented.py` | MultiRM inference with 51nt sliding window (stride=1). BiLSTM + Bahdanau with Gaussian-weighted stitching. |
| `inference_evormd_segmented.py` | EvoRMD inference with 41nt sliding window (stride=20). Conv1d embedder with distance-weighted stitching. |

### Few-Shot Scripts

| Script | Description |
|--------|-------------|
| `fewshot_plant_3way_independent.py` | Plant 3-way classification using independent binary one-vs-rest models (Y/m5C/m6A). Trains 3 separate models with binary focal loss. |
| `fewshot_ac4c_balance.py` | AC4C balanced dataset training. Prunes model to compute only AC4C queries (class index 6), label smoothing loss. |
| `fewshot_ac4c_unbalan.py` | AC4C unbalanced dataset training. Same architecture as balanced variant. |
| `zero_shot_fewshot_analysis.py` | Main entry point for zero-shot feature alignment and few-shot trajectory analysis. Outputs PNG+PDF figures. |
| `zero_shot_fewshot_extract_only.py` | Data-only export pipeline. Extracts features, computes metrics, generates UMAP coordinates, exports CSV/JSON for R visualization. |

### Analysis and Visualization

| Script | Description |
|--------|-------------|
| `run_attention_comparison.py` | Orchestrator: runs the full attention comparison pipeline (select sequences -> 4 models inference -> visualize). |
| `visualize_attention_comparison.py` | Generates per-sequence m6A attention comparison figures for 4 models. Journal-quality PDF+PNG output. |
| `sliding_window_utils.py` | Gaussian-weighted stitching utilities for sliding window inference. |
| `select_representative_sequences.py` | Selects representative sequences by m6A modification density (high/low groups). |
| `SpatialMotif.py` | Visualizes "Spatial Motifs" using Top-K Sequence Logos with latent space clustering (PCA + K-Means). |
| `SpatialMotif_nobackground.py` | Spatial Motif visualization with hard zeroing (no background noise). |
| `abla_mohe.py` | Ablation study for `HierarchicalClassQueryHeadPooling`. Runs 8 configs, outputs CSV + 3 figures. |
| `cal_flops_mohe.py` | Manual FLOPs calculator for 16 ablation configurations. |
| `prepare_umap_data.py` | Full UMAP pipeline: model inference -> feature collection -> UMAP -> JSON export. |
| `prepare_umap_from_npz.py` | Prepares UMAP data from pre-collected `human_atten.npz`. |
| `collect_human_atten.py` | Collects attention outputs for human dataset. Saves to `human_atten.npz`. |
| `collect_human.py` | Collects model outputs (logits + attention) for human dataset to Excel. |
| `collect_modx_atten.py` | Collects ModX model attention outputs. Saves to `modx_atten.npy`. |
| `collect_multirm_atten.py` | Collects MultiRM model attention outputs. Saves to `multirm_atten.npy`. |
| `view_human.py` | Violin plot analysis of per-class modification site distributions (human). |
| `view_human_total.py` | Violin plot of total modification sites (sum of all 12 types). |
| `view_npz.py` | NPZ file inspector / integrity checker. |
| `view3.py` | Multi-dataset violin plot visualization with Morandi color scheme. |
| `3x3.py` | 3x3 Human-to-Plant zero-shot transfer analysis (one-vs-rest). |
| `3x3_2.py` | Transfer analysis v2 (plant positives + plant other-modification negatives). |
| `test_gen3.py` | Evaluation script for 3-generation (gen3) dataset with top-k recall. |
| `test_multirm_4class.py` | Test script for MultIRM 4-class modification mappings. |
| `test_multirm_oversampling.py` | Tests MultIRM oversampling with max-length alignment strategy. |
| `cal_mean_median_mode.py` | Computes mean/median/mode of per-class modification site counts. |

### Model Definitions (`model/`)

| Script | Description |
|--------|-------------|
| `main_model.py` | **Core RGCNFormer model**: `ParallelCNNBlock` (multi-scale kernels), `GCNBlock` (residual GCNConv), `HierarchicalClassQueryHeadPooling` (4-group hierarchical MHA), `RNA_ClassQuery_Model`. |
| `main_model_multirm.py` | Variant adapted for MultIRM dataset (51nt sequences, 4-group A/C/G/U). |
| `main_model_collect_atten.py` | Modified main_model returning intermediate attention outputs for analysis. |
| `multirm.py` | MultiRM baseline: `NaiveNet` (CNN-only), `model_v3` (BiLSTM + Bahdanau shared attention). |
| `multirm_collect_atten.py` | MultiRM variant returning attention weights and context vectors. |
| `modx.py` | ModX baseline: `BahdanauAttention`, `RNAClassifierWithWord2Vec` (input projection + BiLSTM). |
| `modx_collect_atten.py` | ModX variant returning attention weights and context vectors. |
| `evormd_human.py` | EvoRMD baseline: `Conv1dEmbedder`, `TrainableAttention` (MIL pooling), `MulticlassClassifier`. |
| `abla_model.py` | Ablation variants: `1Query`, `4Query`, `12Query`, `FullAttn`. |

### Dataset Classes (`dataset/`)

| Script | Description |
|--------|-------------|
| `human.py` | Primary human RNA modification dataset (`Mer100Dataset`). Loads `seq.npy`/`12loc.npy`/`1001loc.npy`/`4loc.npy`, LinearFold secondary structure with caching, PyG Data objects. |
| `human_with_seq.py` | Enhanced dataset with raw sequence string in Data objects. |
| `human_motif.py` | Standalone dataset for spatial motif analysis. |
| `plant.py` | Plant RNA modification dataset (`PlantDataset`). Same format as human. |
| `plant_single.py` | Combined plant + zero dataset for binary classification. |
| `multirm.py` | MultIRM 12-class dataset. HDF5-based, positive/negative pairing, oversampling. |
| `ac4c.py` | AC4C modification dataset (balanced and unbalanced variants). |
| `gen3.py` | 3-generation RNA modification dataset. |
| `gen3_zero.py` | Gen3 dataset with background/negative samples. |

### Utilities (`utils/`)

| Script | Description |
|--------|-------------|
| `common.py` | Core utilities: constants (`MOD_NAMES`, `GROUP_TO_CLASS_INDICES`), config loading, checkpointing, batch samplers (`MultilabelBalancedBatchSampler`, `DynamicBalancedBatchSampler`), data splits, training/testing loops. |
| `metrics.py` | Evaluation metrics: `evaluate_unbalance()`, `evaluate_balanceb()`, `evaluate_with_optimal_threshold()`, `evaluate_4class_with_optimal_threshold()`, top-k recall, comprehensive localization metrics. |
| `logging.py` | Logging setup (file + console), TensorBoard integration, formatted table output. |
| `few_shot.py` | Few-shot learning benchmarks: `run_few_shot_benchmark()`, `run_few_shot_benchmark_ac4c()`, support set construction. |
| `fewshot_analysis_constants.py` | Constants for the few-shot analysis pipeline (target classes, shot counts, colors). |
| `fewshot_analysis_utils.py` | Shared utilities: output dir management, figure saving, plot styling. |
| `fewshot_analysis_features.py` | Feature extraction from intermediate model layers (hook-based). |
| `fewshot_analysis_zeroshot.py` | Zero-shot feature alignment analysis with UMAP. |
| `fewshot_analysis_fewshot.py` | Few-shot trajectory analysis (fine-tuning, distance metrics). |
| `fewshot_analysis_gen3.py` | 3-generation data analysis with human comparison. |
| `fewshot_analysis_spatial_motif.py` | Spatial motif analysis using Integrated Gradients (captum). |
| `fewshot_export_helpers.py` | CSV/JSON export helpers for R visualization. |
| `rna_visualization.py` | RNA structure and attention visualization (forgi/matplotlib). |
| `train_gen3.py` | Gen3 training script (same protocol as train_human.py). |
| `test_gen3_analyse.py` | Gen3 test with probability density distribution plots. |
| `check_m6a_data_integrity.py` | m6A data integrity checker. |
| `audit_12loc_structure.py` | Deep audit of 12loc.npy label matrix. |
| `Zero_structures.py` | Precomputes RNA secondary structures for zero dataset. |

### Attention Comparison (`atten_comp/`)

| Script | Description |
|--------|-------------|
| `run_attention_comparison_v2.py` | Orchestrator v2: full attention comparison pipeline for 4 models. |
| `visualize_attention_comparison_v2.py` | Attention comparison v2 with background color blocks instead of dashed lines. |
| `inference_modx_full.py` | ModX full-length 1001nt inference (single pass, no sliding window). |

---

## R Scripts

| Script | Description |
|--------|-------------|
| `plot_zero_fewshot_analysis.R` | Generates publication-quality figures from Python-exported CSV data. Handles plant and gen3 data visualizations. Uses ggplot2, dplyr, tidyr, viridis. |
| `plot_zero_fewshot_export.R` | Generates publication-quality figures from Python-exported CSV/JSON data. Export + visualization pipeline. |

---

## Quick Start

### Training

```bash
# Train RGCNFormer on human dataset
python train_human.py --config json/human.json

# Train with MultiRM baseline
python train_human_multirm.py --config json/human.json

# Train with ModX baseline
python train_human_modx.py --config json/human.json

# Train with EvoRMD baseline
python train_human_evormd.py --config json/human.json
```

### Inference

```bash
# Collect attention outputs
python collect_human_atten.py

# Run attention comparison across 4 models
python run_attention_comparison.py
```

### Few-Shot Analysis

```bash
# Run zero-shot and few-shot analysis
python zero_shot_fewshot_analysis.py

# Export data for R visualization
python zero_shot_fewshot_extract_only.py

# Generate publication figures (R)
Rscript plot_zero_fewshot_analysis.R --input_dir output --output_dir fig
```

---

## Dependencies

### Python

```
torch >= 1.12
torch-geometric >= 2.1
numpy
scikit-learn
pandas
tqdm
prettytable
tensorboard
captum (for SpatialMotif)
logomaker (for SpatialMotif)
matplotlib
```

### R

```
ggplot2
dplyr
tidyr
readr
scales
forcats
viridis
ggrastr (optional, for rasterized PDF points)
```

---

## License

This project is for research purposes.
