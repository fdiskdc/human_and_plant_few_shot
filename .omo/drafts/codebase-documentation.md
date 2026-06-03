# Draft: Codebase Documentation Project

## Requirements (confirmed)
- Add header comments to every code file (R, py, ipynb) in the project
- Documentation should include:
  1. File purpose/function description
  2. Input/output specification
  3. Modules and data flow
  4. Related files (calls and called by)

## Technical Decisions
- **Language**: Bilingual comments (Chinese + English) for better accessibility
- **Format**: Standard Python docstring format for .py files, R comment format for .R files
- **Placement**: At the very beginning of each file, before imports
- **Scope**: All 93 code files identified in the project

## Codebase Structure Analysis

### Project Overview
This is an RNA modification classification project using RGCNFormer (Relational Graph Convolutional Network Transformer). The project implements multi-label classification for 12 types of RNA modifications.

### Directory Structure
```
rgcnformer_sum/
├── model/           # Model definitions (8 files)
├── dataset/         # Dataset classes (9 files)
├── utils/           # Utility functions (15 files)
├── train_*.py       # Training scripts (6 files)
├── inference_*.py   # Inference scripts (4 files)
├── collect_*.py     # Data collection scripts (4 files)
├── fewshot_*.py     # Few-shot analysis (6 files)
├── view_*.py        # Visualization scripts (8 files)
├── *.R              # R analysis scripts (2 files)
├── *.ipynb          # Jupyter notebooks (3 files)
└── other utils      # Various utility scripts (15+ files)
```

### Key Components

#### Model Layer (model/)
- `main_model.py` - Core RGCNFormer model with ParallelCNN, GCN, ClassQuery attention
- `multirm.py` - Multi-RM model variant
- `modx.py` - MODX model variant
- `evormd_human.py` - EvoRMD model for human data
- `abla_model.py` - Ablation study models
- `*_collect_atten.py` - Attention collection variants

#### Dataset Layer (dataset/)
- `human.py` - Human RNA modification dataset (12 classes)
- `plant.py` - Plant RNA dataset
- `ac4c.py` - AC4C modification dataset
- `multirm.py` - Multi-RM dataset
- `gen3.py` - Generation 3 dataset
- `human_motif.py` - Human motif dataset
- `human_with_seq.py` - Human dataset with sequence info
- `gen3_zero.py` - Zero-shot gen3 dataset
- `plant_single.py` - Single plant dataset

#### Utility Layer (utils/)
- `common.py` - Common utilities and constants
- `metrics.py` - Evaluation metrics
- `logging.py` - Logging utilities
- `train_gen3.py` - Gen3 training utilities
- `test_gen3_analyse.py` - Gen3 test analysis
- `fewshot_*.py` - Few-shot learning utilities
- `rna_visualization*.py/ipynb` - RNA visualization tools

#### Training Scripts
- `train_human.py` - Human RNA training (main)
- `train_plant.py` - Plant RNA training
- `train_human_multirm.py` - Multi-RM human training
- `train_human_modx.py` - MODX human training
- `train_human_evormd.py` - EvoRMD human training
- `train_multirm_dataset.py` - Multi-RM dataset training

#### Inference Scripts
- `inference_modx_segmented.py` - MODX segmented inference
- `inference_mrmodn_full.py` - MRModN full inference
- `inference_multirm_segmented.py` - Multi-RM segmented inference
- `inference_evormd_segmented.py` - EvoRMD segmented inference

## Documentation Template

### Python Files (.py)
```python
"""
[文件名] - [简短描述]

[详细描述文件的功能和用途]

功能模块:
- [模块1]: [描述]
- [模块2]: [描述]

输入:
- [输入1]: [类型] - [描述]
- [输入2]: [类型] - [描述]

输出:
- [输出1]: [类型] - [描述]
- [输出2]: [类型] - [描述]

数据流:
1. [步骤1]
2. [步骤2]
3. [步骤3]

相关文件:
- 调用: [文件列表]
- 被调用: [文件列表]

使用示例:
    [示例代码]

作者: [作者]
日期: [日期]
版本: [版本]
"""
```

### R Files (.R)
```r
# ==============================================================================
# [文件名] - [简短描述]
# ==============================================================================
#
# [详细描述文件的功能和用途]
#
# 功能模块:
# - [模块1]: [描述]
# - [模块2]: [描述]
#
# 输入:
# - [输入1]: [类型] - [描述]
#
# 输出:
# - [输出1]: [类型] - [描述]
#
# 数据流:
# 1. [步骤1]
# 2. [步骤2]
#
# 相关文件:
# - 调用: [文件列表]
# - 被调用: [文件列表]
#
# ==============================================================================
```

### Jupyter Notebooks (.ipynb)
```json
{
  "cell_type": "markdown",
  "metadata": {},
  "source": [
    "# [Notebook标题]\n",
    "\n",
    "[详细描述notebook的功能和用途]\n",
    "\n",
    "## 功能模块\n",
    "- [模块1]: [描述]\n",
    "\n",
    "## 输入\n",
    "- [输入1]: [类型] - [描述]\n",
    "\n",
    "## 输出\n",
    "- [输出1]: [类型] - [描述]\n",
    "\n",
    "## 数据流\n",
    "1. [步骤1]\n",
    "2. [步骤2]"
  ]
}
```

## User Preferences (Confirmed)
- **Language**: Bilingual (Chinese + English)
- **Depth**: Comprehensive (purpose, I/O, data flow, related files, usage examples)
- **Scope**: File-level only (headers at top of each file)

## Open Questions
- None - requirements are clear

## Scope Boundaries
- INCLUDE: All .py, .R, .ipynb files in the project
- EXCLUDE: Non-code files (json, npy, cache, etc.)
- EXCLUDE: __pycache__, .git, and other system directories

## File Inventory (93 files)

### Model Files (8)
1. model/main_model.py
2. model/multirm.py
3. model/modx.py
4. model/evormd_human.py
5. model/abla_model.py
6. model/main_model_multirm.py
7. model/main_model_collect_atten.py
8. model/modx_collect_atten.py
9. model/multirm_collect_atten.py

### Dataset Files (9)
1. dataset/human.py
2. dataset/plant.py
3. dataset/ac4c.py
4. dataset/multirm.py
5. dataset/gen3.py
6. dataset/human_motif.py
7. dataset/human_with_seq.py
8. dataset/gen3_zero.py
9. dataset/plant_single.py

### Utility Files (15)
1. utils/common.py
2. utils/metrics.py
3. utils/logging.py
4. utils/train_gen3.py
5. utils/test_gen3_analyse.py
6. utils/fewshot_analysis_*.py (8 files)
7. utils/rna_visualization.py
8. utils/rna_visualization.ipynb (2)
9. utils/audit_12loc_structure.py
10. utils/check_m6a_data_integrity.py
11. utils/Zero_structures.py

### Training Scripts (6)
1. train_human.py
2. train_plant.py
3. train_human_multirm.py
4. train_human_modx.py
5. train_human_evormd.py
6. train_multirm_dataset.py

### Inference Scripts (4)
1. inference_modx_segmented.py
2. inference_mrmodn_full.py
3. inference_multirm_segmented.py
4. inference_evormd_segmented.py

### Collection Scripts (4)
1. collect_human.py
2. collect_human_atten.py
3. collect_modx_atten.py
4. collect_multirm_atten.py

### Few-shot Analysis (6)
1. fewshot_ac4c_balance.py
2. fewshot_ac4c_unbalan.py
3. fewshot_plant_3way_independent.py
4. zero_shot_fewshot_analysis.py
5. zero_shot_fewshot_extract_only.py
6. utils/few_shot.py

### Visualization/View (8)
1. view_human.py
2. view_human_total.py
3. view_npz.py
4. view3.py
5. view_gen3.ipynb
6. visualize_attention_comparison.py
7. run_attention_comparison.py
8. select_representative_sequences.py

### R Scripts (2)
1. plot_zero_fewshot_analysis.R
2. plot_zero_fewshot_export.R

### Other Utilities (15+)
1. 3x3.py
2. 3x3_2.py
3. abla_mohe.py
4. cal_flops_mohe.py
5. cal_mean_median_mode.py
6. prepare_umap_data.py
7. prepare_umap_from_npz.py
8. sliding_window_utils.py
9. SpatialMotif.py
10. SpatialMotif_nobackground.py
11. test_gen3.py
12. test_multirm_4class.py
13. test_multirm_oversampling.py
14. atten_comp/*.py (3 files)
15. model/EvoRMD/Script/*.py (7 files)
