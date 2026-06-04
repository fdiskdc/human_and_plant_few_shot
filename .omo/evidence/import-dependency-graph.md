# Import 依赖关系图 / Import Dependency Graph

> 生成时间 / Generated: 2026-06-04
> 用途 / Purpose: Wave 4 import 更新前的引用清单

## `model.main_model` 引用 (40 处) / References to `model.main_model` (40)

### 训练脚本 / Training scripts
- `train_human.py:65` — `from model.main_model import RNA_ClassQuery_Model`
- `train_plant.py:66` — `from model.main_model import RNA_ClassQuery_Model`
- `train_multirm_dataset.py:64` — `from model.main_model import RNA_ClassQuery_Model`

### 推理脚本 / Inference scripts
- `inference_mrmodn_full.py:54` — `from model.main_model_collect_atten import RNA_ClassQuery_Model_Collect_Atten`

### 数据收集 / Data collection
- `collect_human.py:61` — `from model.main_model import RNA_ClassQuery_Model`
- `collect_human_atten.py:58` — `from model.main_model_collect_atten import RNA_ClassQuery_Model_Collect_Atten`

### 少样本 / Few-shot
- `fewshot_ac4c_balance.py:63` — `from model.main_model import RNA_ClassQuery_Model`
- `fewshot_ac4c_unbalan.py:63` — `from model.main_model import RNA_ClassQuery_Model`
- `fewshot_plant_3way_independent.py:72` — `from model.main_model import RNA_ClassQuery_Model`
- `zero_shot_fewshot_analysis.py` — 字符串注释引用
- `zero_shot_fewshot_extract_only.py` — 字符串注释引用

### 消融 / Ablation
- `3x3.py:65` — `from model.main_model import RNA_ClassQuery_Model`
- `3x3_2.py:63` — `from model.main_model import RNA_ClassQuery_Model`

### 可视化 / Visualization
- `prepare_umap_data.py:62` — `from model.main_model_collect_atten import RNA_ClassQuery_Model_Collect_Atten`
- `SpatialMotif.py:77` — `from model.main_model import RNA_ClassQuery_Model`
- `SpatialMotif_nobackground.py:73` — `from model.main_model import RNA_ClassQuery_Model`

### 测试 / Tests
- `test_gen3.py:56` — `from model.main_model import RNA_ClassQuery_Model`
- `test_multirm_4class.py:42` — `from model.main_model_multirm import RNA_ClassQuery_Model`
- `test_multirm_oversampling.py` — 字符串注释引用
- `utils/test_gen3_analyse.py:60` — `from model.main_model import RNA_ClassQuery_Model`

### 包入口 / Package entry
- `model/__init__.py:75` — `from .main_model import (...)` ✅ 已更新为 `from .mrmodn import (...)`

## `model.main_model_collect_atten` 引用 (6 处) / References (6)

- `prepare_umap_data.py:62`
- `inference_mrmodn_full.py:54`
- `collect_human_atten.py:58`
- `model/main_model_collect_atten.py:3` (自引用注释)
- `model/main_model_collect_atten.py:37` (自引用注释)
- 共 ~3 个实际 `from ... import` 语句需要替换

## `model.main_model_multirm` 引用 (2 处) / References (2)

- `test_multirm_4class.py:42` — `from model.main_model_multirm import RNA_ClassQuery_Model`
- `model/main_model_multirm.py:38` (注释字符串)

## 更新策略 / Update Strategy

1. **批量替换** (Wave 4):
   - `from model.main_model import` → `from model.mrmodn import`
   - `from model.main_model_collect_atten import` → `from model.mrmodn_collect_atten import`
   - `from model.main_model_multirm import` → `from model.mrmodn_multirm import`
   - 同时更新文件头部 docstring 中的 `model.main_model` 文本

2. **保留符号名**: `RNA_ClassQuery_Model`, `RNA_ClassQuery_Model_Collect_Atten` 等类名不变

3. **验证**: 每次替换后用 `python -c "import ast; ast.parse(...)"` 检查语法
