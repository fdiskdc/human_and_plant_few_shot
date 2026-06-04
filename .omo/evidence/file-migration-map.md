# 文件迁移映射表 / File Migration Map

> RGCNFormer 项目重构 — 根目录 .py 文件迁移计划
> RGCNFormer project refactor — root .py file migration plan
>
> 生成时间 / Generated: 2026-06-04
> 命名规则 / Naming rule: `任务_数据集_模型.py` / `task_dataset_model.py`

## 根目录文件 (38 个) / Root Directory Files (38)

### 训练脚本 / Training Scripts

| 旧路径 / Old Path | 新路径 / New Path | 备注 / Notes |
|---|---|---|
| `train_human.py` | `train_human_mrmodn.py` | mRModN 模型, human 数据集 |
| `train_plant.py` | `train_plant_mrmodn.py` | mRModN 模型, plant 数据集 |
| `train_multirm_dataset.py` | `train_multirm_mrmodn.py` | mRModN 模型, multirm 数据集 |
| `train_human_evormd.py` | (保持) `train_human_evormd.py` | 已符合命名规范 |
| `train_human_modx.py` | (保持) `train_human_modx.py` | 已符合命名规范 |
| `train_human_multirm.py` | (保持) `train_human_multirm.py` | 已符合命名规范 |

### 推理脚本 / Inference Scripts (保留 segmented/full 模式)

| 旧路径 / Old Path | 新路径 / New Path | 备注 / Notes |
|---|---|---|
| `inference_evormd_segmented.py` | `inference_human_evormd_segmented.py` | human 滑窗推理 |
| `inference_modx_segmented.py` | `inference_human_modx_segmented.py` | human 滑窗推理 |
| `inference_mrmodn_full.py` | `inference_human_mrmodn_full.py` | human 全长推理 |
| `inference_multirm_segmented.py` | `inference_multirm_multirm_segmented.py` | multirm 滑窗推理 |

### 数据收集 / Data Collection

| 旧路径 / Old Path | 新路径 / New Path | 备注 / Notes |
|---|---|---|
| `collect_human.py` | `collect_human_mrmodn.py` | 收集 logits/特征 |
| `collect_human_atten.py` | `collect_atten_human_mrmodn.py` | 收集注意力 |
| `collect_modx_atten.py` | `collect_atten_human_modx.py` | modx 注意力收集 |
| `collect_multirm_atten.py` | `collect_atten_multirm_multirm.py` | multirm 注意力收集 |

### 少样本/零样本分析 / Few-shot / Zero-shot Analysis

| 旧路径 / Old Path | 新路径 / New Path | 备注 / Notes |
|---|---|---|
| `fewshot_ac4c_balance.py` | `fewshot_ac4c_mrmodn_balance.py` | ac4c 平衡集少样本 |
| `fewshot_ac4c_unbalan.py` | `fewshot_ac4c_mrmodn_unbalan.py` | ac4c 非平衡少样本 |
| `fewshot_plant_3way_independent.py` | `fewshot_plant_mrmodn_3way.py` | plant 三向独立少样本 |
| `zero_shot_fewshot_analysis.py` | `zeroshot_human_mrmodn_analysis.py` | human 零样本分析 |
| `zero_shot_fewshot_extract_only.py` | `zeroshot_human_mrmodn_extract.py` | human 零样本特征抽取 |

### 测试脚本 → tests/ / Test Scripts → tests/

| 旧路径 / Old Path | 新路径 / New Path | 备注 / Notes |
|---|---|---|
| `test_gen3.py` | `tests/test_gen3.py` | gen3 数据集回归测试 |
| `test_multirm_4class.py` | `tests/test_multirm_4class.py` | multirm 4 类测试 |
| `test_multirm_oversampling.py` | `tests/test_multirm_oversampling.py` | multirm 过采样测试 |

### 消融实验 → ablation/ / Ablation Experiments → ablation/

| 旧路径 / Old Path | 新路径 / New Path | 备注 / Notes |
|---|---|---|
| `3x3.py` | `ablation/ablation_3x3_human_mrmodn.py` | 3×3 矩阵实验 |
| `3x3_2.py` | `ablation/ablation_3x3_v2_human_mrmodn.py` | 3×3 矩阵实验 v2 |
| `abla_mohe.py` | `ablation/ablation_mohe_human_mrmodn.py` | MoHE 消融 |
| `cal_flops_mohe.py` | `ablation/cal_flops_human_mrmodn.py` | FLOPs 计算 |
| `cal_mean_median_mode.py` | `ablation/cal_stats_human_mrmodn.py` | 统计指标计算 |

### 可视化脚本 → visualization/ / Visualization Scripts → visualization/

| 旧路径 / Old Path | 新路径 / New Path | 备注 / Notes |
|---|---|---|
| `view_human.py` | `visualization/human/mrmodn/view_data.py` | human 数据查看 |
| `view_human_total.py` | `visualization/human/mrmodn/view_total.py` | human 整体查看 |
| `view3.py` | `visualization/human/mrmodn/view_v3.py` | 注意力视图 v3 |
| `view_npz.py` | `visualization/tools/view_npz.py` | npz 文件查看工具 |
| `visualize_attention_comparison.py` | `visualization/human/mrmodn/attention_comparison.py` | 注意力对比 |
| `run_attention_comparison.py` | `visualization/human/mrmodn/run_attention_comparison.py` | 注意力对比运行器 |
| `select_representative_sequences.py` | `visualization/human/mrmodn/select_representatives.py` | 代表序列选择 |
| `prepare_umap_data.py` | `visualization/tools/prepare_umap_data.py` | UMAP 数据准备 |
| `prepare_umap_from_npz.py` | `visualization/tools/prepare_umap_from_npz.py` | UMAP from npz |
| `SpatialMotif.py` | `visualization/human/mrmodn/spatial_motif.py` | 空间 motif 可视化 |
| `SpatialMotif_nobackground.py` | `visualization/human/mrmodn/spatial_motif_nobg.py` | 无背景空间 motif |

### 工具 / Utilities

| 旧路径 / Old Path | 新路径 / New Path | 备注 / Notes |
|---|---|---|
| `sliding_window_utils.py` | `utils/sliding_window_utils.py` | 滑窗辅助函数 |

### R 脚本 → analysis/ / R Scripts → analysis/

| 旧路径 / Old Path | 新路径 / New Path | 备注 / Notes |
|---|---|---|
| `plot_zero_fewshot_analysis.R` | `analysis/plot_zero_fewshot_analysis.R` | R 绘图脚本 |
| `plot_zero_fewshot_export.R` | `analysis/plot_zero_fewshot_export.R` | R 导出脚本 |
| `Rplots.pdf` | `analysis/Rplots.pdf` | R 默认输出 |

### 子目录文件移动 / Subdirectory file moves

| 旧路径 / Old Path | 新路径 / New Path | 备注 / Notes |
|---|---|---|
| `atten_comp/inference_modx_full.py` | `visualization/human/modx/inference_full.py` | modx 全长推理 |
| `atten_comp/run_attention_comparison_v2.py` | `visualization/human/mrmodn/run_attention_comparison_v2.py` | 注意力对比 v2 运行器 |
| `atten_comp/visualize_attention_comparison_v2.py` | `visualization/human/mrmodn/visualize_attention_comparison_v2.py` | 注意力对比 v2 可视化 |

### 图片目录 / Figure directories

| 旧路径 / Old Path | 新路径 / New Path | 备注 / Notes |
|---|---|---|
| `motif_logo/` | `visualization/motif_logo/` | motif logo PDF |
| `fig/` | `visualization/fig/` | 主图集合 |
| `att_fig/` | `visualization/att_fig/` | 注意力图集合 |
| `view_gen3.ipynb` (root) | `visualization/notebooks/view_gen3.ipynb` | gen3 数据查看 notebook |

### model/ 重命名 / model/ renames (Wave 1)

| 旧路径 / Old Path | 新路径 / New Path | 备注 / Notes |
|---|---|---|
| `model/main_model.py` | `model/mrmodn.py` | mRModN 主模型 |
| `model/main_model_collect_atten.py` | `model/mrmodn_collect_atten.py` | 注意力收集变体 |
| `model/main_model_multirm.py` | `model/mrmodn_multirm.py` | MultIRM 数据集变体 |

## 不动文件 / Untouched Files

- `npy/` — 数据目录 (软链接到 /home/dc/vscode/npyForTrain)
- `dataset/`, `utils/`, `json/`, `logs_abla/`, `output/` — 维持原结构
- `model/abla_model.py`, `model/evormd_human.py`, `model/modx.py`, `model/multirm.py`, `model/modx_collect_atten.py`, `model/multirm_collect_atten.py` — 命名已规范
- `model/__init__.py` — 仅更新内部 import
