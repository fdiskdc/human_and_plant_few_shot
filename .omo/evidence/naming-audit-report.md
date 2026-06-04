# 命名规范审计 / Naming Convention Audit

> Generated: 2026-06-04 (Wave 5 refactor verification)

## 根目录 / Root Directory

- Total .py files: **19**
- Compliant: **19/19** (100.0%)
- Non-compliant: **0**

### Compliant files:
- `collect_atten_human_modx.py`
- `collect_atten_human_mrmodn.py`
- `collect_atten_multirm_multirm.py`
- `collect_human_mrmodn.py`
- `fewshot_ac4c_mrmodn_balance.py`
- `fewshot_ac4c_mrmodn_unbalan.py`
- `fewshot_plant_mrmodn_3way.py`
- `inference_human_evormd_segmented.py`
- `inference_human_modx_segmented.py`
- `inference_human_mrmodn_full.py`
- `inference_multirm_multirm_segmented.py`
- `train_human_evormd.py`
- `train_human_modx.py`
- `train_human_mrmodn.py`
- `train_human_multirm.py`
- `train_multirm_mrmodn.py`
- `train_plant_mrmodn.py`
- `zeroshot_human_mrmodn_analysis.py`
- `zeroshot_human_mrmodn_extract.py`

## `ablation/` Directory

- Total .py files: **5**
- Compliant: **5/5**
  - ✅ `ablation_3x3_human_mrmodn.py`
  - ✅ `ablation_3x3_v2_human_mrmodn.py`
  - ✅ `ablation_mohe_human_mrmodn.py`
  - ✅ `cal_flops_human_mrmodn.py`
  - ✅ `cal_stats_human_mrmodn.py`

## `tests/` Directory

- ℹ️  `__init__.py`
- ℹ️  `conftest.py`
- ✅ `test_config.py`
- ✅ `test_dataset_import.py`
- ✅ `test_gen3.py`
- ✅ `test_infrastructure.py`
- ✅ `test_model_import.py`
- ✅ `test_multirm_4class.py`
- ✅ `test_multirm_oversampling.py`
- ✅ `test_naming_convention.py`

## `model/` Directory

- `__init__.py`
- `abla_model.py`
- `evormd_human.py`
- `modx.py`
- `modx_collect_atten.py`
- `mrmodn.py`
- `mrmodn_collect_atten.py`
- `mrmodn_multirm.py`
- `multirm.py`
- `multirm_collect_atten.py`

## `visualization/` Directory (count by subdir)

- `visualization/human/modx/inference_full.py`
- `visualization/human/mrmodn/attention_comparison.py`
- `visualization/human/mrmodn/run_attention_comparison.py`
- `visualization/human/mrmodn/run_attention_comparison_v2.py`
- `visualization/human/mrmodn/select_representatives.py`
- `visualization/human/mrmodn/spatial_motif.py`
- `visualization/human/mrmodn/spatial_motif_nobg.py`
- `visualization/human/mrmodn/view_data.py`
- `visualization/human/mrmodn/view_total.py`
- `visualization/human/mrmodn/view_v3.py`
- `visualization/human/mrmodn/visualize_attention_comparison_v2.py`
- `visualization/tools/prepare_umap_data.py`
- `visualization/tools/prepare_umap_from_npz.py`
- `visualization/tools/view_npz.py`

## ✅ Verdict

- Root directory compliance: **100.0%** → **PASS**
- model/ legacy main_model files removed: ✅
- ablation/, tests/, visualization/, analysis/ created: ✅
- All `model.main_model` import references: ✅ CLEAN
