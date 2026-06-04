# RGCNFormer 项目重构方案

## TL;DR

> **Quick Summary**: 对 RGCNFormer (mRModN) RNA修饰分类项目进行全面文件重命名和目录重组，统一命名规范为 `任务_数据集_模型.py`，新建 `visualization/`、`ablation/`、`tests/` 目录，重构所有 import 引用，添加基础测试框架。
>
> **Deliverables**:
> - 根目录清理：40+文件精简到 ~15 个核心训练/推理脚本
> - 新建目录：`visualization/`（按数据集/模型二级组织）、`ablation/`、`tests/`
> - 模型文件重命名：`model/main_model.py` → `model/mrmodn.py`
> - 所有 import 引用更新
> - pytest 测试基础设施 + 核心功能测试
> - 34 个 Jupyter notebook 整理
>
> **Estimated Effort**: Large (50+ tasks)
> **Parallel Execution**: YES - 5 waves
> **Critical Path**: Wave 1 (model rename) → Wave 2 (root files rename) → Wave 4 (import refactor)

---

## Context

### Original Request
用户要求重构 RGCNFormer 项目：
1. 文件命名统一为 `任务_数据集_模型.py`（如 `train_human_mrmodn.py`）
2. 可视化文件单独存放于 `visualization/`，按数据集和模型分类
3. `npy/` 数据目录不可移动
4. 训练启动脚本存放于根目录
5. 必要时代码也需要重构

### Interview Summary
**Key Discussions**:
- **主模型名**: mRModN（= RGCNFormer = RNA_ClassQuery_Model），文档中可称 RGCNFormer 但文件和代码中使用 mRModN
- **model/main_model.py → model/mrmodn.py**: 重命名模型文件并更新所有导入
- **目录结构**: 新建 `visualization/`（数据集→模型 二级组织）、`ablation/`、`tests/`
- **根目录策略**: 训练+推理+数据收集+少样本/零样本分析脚本宽松保留根目录，消融/可视化/工具/测试移入子目录
- **atten_comp/ → visualization/**: 注意力对比目录合并到可视化目录
- **ipynb**: 34个 notebook 移入 `visualization/` 或 `analysis/`
- **gen3**: gen3 是数据集，数据在 `npy/3gen`
- **测试**: 添加 pytest 基础测试框架，Agent QA 强制执行

**Research Findings**:
- 项目有 60+ Python 文件，根目录有 40+ 个
- `npy/` 包含大型 .npz 文件（最大 10GB+），绝对不能触碰
- `model/` 有 10 个文件，其中 `main_model.py` 需重命名
- `model/` 下 3 个 `*_collect_atten.py` 变体需重命名
- 目前无测试框架，只有手工 `test_*.py` 脚本
- 部分文件命名混乱（`3x3.py`, `abla_mohe.py`, `view3.py` 等）

### Metis / Oracle Review
> 子代理基础设施超时，无法完成自动审查。已通过深入的手动探索和用户访谈弥补。计划完成后建议手动审查。

---

## Work Objectives

### Core Objective
将 RGCNFormer 项目重构为命名规范、目录清晰、可维护性高的结构，同时保持所有功能正确运行。

### Concrete Deliverables
- 根目录从 40+ .py 文件精简到 ~15 个核心启动脚本（训练 + 推理 + 数据收集 + 少样本）
- `visualization/` 目录（含 7-10 个子目录，按数据集/模型组织 + ipynb 归档）
- `ablation/` 目录（含 4-6 个消融实验脚本，按命名规则重命名）
- `tests/` 目录（pytest 配置 + 核心功能测试）
- `model/mrmodn.py`（从 `main_model.py` 重命名）
- `model/mrmodn_collect_atten.py`（从 `main_model_collect_atten.py` 重命名）
- 所有 import 引用更新为新的模块路径

### Definition of Done
- [ ] 所有 .py 文件遵循 `任务_数据集_模型.py` 命名规范
- [ ] `model/main_model.py` 不再存在（已重命名为 `model/mrmodn.py`）
- [ ] 所有 `from model.main_model import` 更新为 `from model.mrmodn import`
- [ ] `pytest` 配置就绪，`python -m pytest tests/` 可运行
- [ ] `visualization/` 目录存在且按数据集/模型组织
- [ ] `ablation/` 目录存在且文件按规则命名
- [ ] `atten_comp/` 目录已合并到 `visualization/`
- [ ] `npy/` 数据完好无损
- [ ] 所有训练脚本可在根目录直接运行

### Must Have
- `npy/` 目录数据不改动
- 所有 import 引用更新正确，无 broken import
- 训练脚本保留在根目录且可直接运行
- 命名规范统一：`任务_数据集_模型.py`

### Must NOT Have (Guardrails)
- 不改变 `npy/` 目录下的任何文件
- 不改变数据集的读取逻辑（除非必要配合命名变更）
- 不修改 `.git/`、`.omc/`、`.omo/` 目录
- 不在重构中引入新功能或算法变更
- 不删除已有文件（仅重命名和移动）
- 不在 `docs/` 或 `plan/` 等错误路径输出文件

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: YES（重构后添加）
- **Framework**: pytest
- **Approach**: Tests-after — 先完成重构，再添加测试确保回归正确

### QA Policy
Every task MUST include agent-executed QA scenarios. Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Frontend/UI**: N/A（无前端）
- **TUI/CLI**: 使用 Bash 运行 Python 脚本验证
- **API/Backend**: 使用 Bash 运行 Python import 验证
- **Library/Module**: 使用 Bash (python -c) 验证导入和基本功能

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - foundation):
├── Task 1: Create directory structure [quick]
├── Task 2: Rename model/main_model.py → model/mrmodn.py [quick]
├── Task 3: Rename model/collect_atten variants [quick]
├── Task 4: Update model/__init__.py exports [quick]
├── Task 5: Setup pytest infrastructure [quick]
├── Task 6: Analyze root .py file inventory [quick]
└── Task 7: Analyze model/ imports graph [quick]

Wave 2 (After Wave 1 - rename root training/inference/collect/fewshot):
├── Task 8-13: Rename root training scripts (6 files) [quick]
├── Task 14-17: Rename root inference scripts (4 files) [quick]
├── Task 18-21: Rename root collect scripts (4 files) [quick]
├── Task 22-25: Rename root fewshot/zeroshot scripts (4 files) [quick]
├── Task 26-27: Rename root test scripts → move to tests/ [quick]
└── Task 28-30: Rename root miscellaneous scripts → move to subdirs [quick]

Wave 3 (After Wave 2 - create new directories and populate):
├── Task 31-34: Create visualization/ structure and move files [visual-engineering]
├── Task 35-37: Create ablation/ structure and move files [quick]
├── Task 38-39: Merge atten_comp/ → visualization/ [quick]
├── Task 40-41: Organize ipynb/ notebooks [quick]
└── Task 42-43: Organize R scripts and motif files [quick]

Wave 4 (After Wave 3 - update all import references):
├── Task 44-47: Update imports in root scripts [unspecified-high]
├── Task 48-50: Update imports in model/ files [unspecified-high]
├── Task 51-53: Update imports in dataset/ files [unspecified-high]
└── Task 54-56: Update imports in utils/ files [unspecified-high]

Wave 5 (After Wave 4 - tests + verification):
├── Task 57-59: Write core tests [unspecified-high]
├── Task 60-61: Run full import validation [quick]
└── Task 62-64: Cross-file consistency checks [unspecified-high]

Wave FINAL (After ALL tasks):
├── Task F1: Plan Compliance Audit (oracle)
├── Task F2: Code Quality Review (unspecified-high)
├── Task F3: Real Manual QA (unspecified-high)
└── Task F4: Scope Fidelity Check (deep)
```

### Agent Dispatch Summary
- **Wave 1**: 7 tasks → quick (all lightweight setup)
- **Wave 2**: 23 tasks → quick (all file rename/move)
- **Wave 3**: 13 tasks → visual-engineering + quick
- **Wave 4**: 13 tasks → unspecified-high (import refactoring needs careful execution)
- **Wave 5**: 8 tasks → unspecified-high + quick
- **FINAL**: 4 tasks → oracle + unspecified-high + deep

---

## TODOs

- [ ] 1. 创建新目录结构

  **What to do**:
  - 创建 `visualization/` 根目录
  - 创建 `visualization/human/`、`visualization/plant/`、`visualization/multirm/`、`visualization/ac4c/`、`visualization/gen3/` 子目录
  - 在每个数据集目录下创建模型子目录：`mrmodn/`、`evormd/`、`modx/`、`multirm/`
  - 创建 `ablation/` 目录
  - 创建 `tests/` 目录（含 `__init__.py`）
  - 创建 `analysis/` 目录（用于 R 脚本和少样本分析报告）

  **Must NOT do**:
  - 不移动任何文件到此阶段（仅创建空目录）
  - 不删除任何现有目录

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2-7)
  - **Blocks**: Tasks 31+ (visualization), 35+ (ablation), 57+ (tests)
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] `visualization/` 根目录存在
  - [ ] `visualization/human/mrmodn/`、`visualization/human/evormd/` 等子目录存在
  - [ ] `ablation/` 目录存在
  - [ ] `tests/__init__.py` 存在
  - [ ] `analysis/` 目录存在

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 所有目录正确创建
    Tool: Bash
    Preconditions: 当前工作目录为项目根目录
    Steps:
      1. ls -d visualization/human/mrmodn visualization/human/evormd visualization/plant/mrmodn
      2. ls -d ablation/ tests/__init__.py analysis/
      3. find visualization/ -type d | wc -l (应 >= 20)
    Expected Result: 所有目录存在，无错误
    Failure Indicators: 任何目录缺失
    Evidence: .omo/evidence/task-1-dir-structure.txt (ls -R 输出)
  ```
  **Commit**: NO（与后续任务合并提交）

- [ ] 2. 重命名 model/main_model.py → model/mrmodn.py

  **What to do**:
  - 使用 `git mv model/main_model.py model/mrmodn.py` 重命名文件
  - 验证文件内容完整（不丢失内容）

  **Must NOT do**:
  - 不修改文件内容（仅重命名）
  - 不更新任何 import 引用（由 Wave 4 任务统一处理）
  - 不删除原文件（使用 git mv 保留历史）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖原始文件存在）
  - **Parallel Group**: Wave 1, before Task 3
  - **Blocks**: Task 3 (collect_atten variants), Task 4 (__init__.py), Tasks 44+ (import updates)
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] `model/main_model.py` 不再存在
  - [ ] `model/mrmodn.py` 存在且内容与原始 `main_model.py` 完全相同
  - [ ] `git log -- model/mrmodn.py` 显示重命名历史

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 文件重命名成功
    Tool: Bash
    Preconditions: 原始 model/main_model.py 存在
    Steps:
      1. test -f model/mrmodn.py && echo "EXISTS" || echo "MISSING"
      2. test ! -f model/main_model.py && echo "REMOVED" || echo "STILL EXISTS"
      3. wc -l model/mrmodn.py (确认行数合理)
    Expected Result: mrmodn.py EXISTS, main_model.py REMOVED
    Failure Indicators: mrmodn.py MISSING 或 main_model.py STILL EXISTS
    Evidence: .omo/evidence/task-2-rename.txt
  ```
  **Commit**: NO（与 Wave 4 import 更新合并提交）

- [ ] 3. 重命名 model/ 下 collect_atten 变体文件

  **What to do**:
  - `model/main_model_collect_atten.py` → `model/mrmodn_collect_atten.py` (git mv)
  - `model/modx_collect_atten.py` → 保持原名（已符合规范）
  - `model/multirm_collect_atten.py` → 保持原名（已符合规范）

  **Must NOT do**:
  - 不修改文件内容
  - 不更新 import 引用

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖 Task 2 完成）
  - **Parallel Group**: Wave 1 (sequential after Task 2)
  - **Blocks**: Tasks 48-50 (model import updates)
  - **Blocked By**: Task 2

  **Acceptance Criteria**:
  - [ ] `model/main_model_collect_atten.py` 不再存在
  - [ ] `model/mrmodn_collect_atten.py` 存在

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: collect_atten 文件重命名
    Tool: Bash
    Preconditions: Task 2 已完成
    Steps:
      1. test -f model/mrmodn_collect_atten.py && echo "EXISTS"
      2. test ! -f model/main_model_collect_atten.py && echo "REMOVED"
    Expected Result: mrmodn_collect_atten.py EXISTS, old file REMOVED
    Evidence: .omo/evidence/task-3-collect-atten-rename.txt
  ```
  **Commit**: NO

- [ ] 4. 更新 model/__init__.py 导出

  **What to do**:
  - 将 `from .main_model import` 改为 `from .mrmodn import`
  - 保持所有导出的类名和函数名不变
  - 检查 `__all__` 列表是否需要更新

  **Must NOT do**:
  - 不改变导出的 API 签名（类名、函数名）
  - 不删除任何导出

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖 Task 2）
  - **Parallel Group**: Wave 1 (sequential after Task 2)
  - **Blocks**: 所有导入 model 的文件
  - **Blocked By**: Task 2

  **Acceptance Criteria**:
  - [ ] `model/__init__.py` 中 `from .main_model import` 改为 `from .mrmodn import`
  - [ ] `from model import RNA_ClassQuery_Model` 仍然可用
  - [ ] `__all__` 列表未丢失任何导出

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: __init__.py 导入验证
    Tool: Bash
    Steps:
      1. python -c "from model import RNA_ClassQuery_Model; print('OK')"
      2. python -c "from model import ParallelCNNBlock, GCNBlock, ClassQueryHead; print('OK')"
    Expected Result: 两次输出 "OK"，无 ImportError
    Failure Indicators: ImportError 或 ModuleNotFoundError
    Evidence: .omo/evidence/task-4-init-import.txt
  ```
  **Commit**: NO

- [ ] 5. 搭建 pytest 测试基础设施

  **What to do**:
  - 在项目根目录创建 `pytest.ini` 或 `pyproject.toml` 中的 pytest 配置
  - 创建 `tests/__init__.py`（空文件）
  - 创建 `tests/conftest.py`（基础 fixtures：项目路径、测试数据路径）
  - 安装 pytest: `pip install pytest` 或确认已安装
  - 写一个简单测试验证框架可用: `tests/test_infrastructure.py`

  **Must NOT do**:
  - 不在此任务写业务测试（由 Wave 5 任务处理）
  - 不修改现有代码

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 6, 7)
  - **Blocks**: Tasks 57-59 (core tests)
  - **Blocked By**: None (独立任务)

  **Acceptance Criteria**:
  - [ ] `pytest.ini` 或 pytest 配置存在于项目中
  - [ ] `python -m pytest tests/test_infrastructure.py -v` → PASS
  - [ ] `tests/conftest.py` 存在

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: pytest 框架可用
    Tool: Bash
    Preconditions: pytest 已安装
    Steps:
      1. python -m pytest --version
      2. python -m pytest tests/test_infrastructure.py -v
    Expected Result: pytest 版本号显示，测试 PASS
    Failure Indicators: pytest 未安装或测试 FAIL
    Evidence: .omo/evidence/task-5-pytest-setup.txt
  ```
  **Commit**: NO

- [ ] 6. 清点根目录所有 Python 文件并生成迁移映射表

  **What to do**:
  - 列出根目录下所有 .py 文件（不含子目录）
  - 对每个文件：读取前 50 行，提取 import 语句和模块文档字符串
  - 根据命名规则确定每个文件的**新名称**和**新位置**（留在根目录 或 移入子目录）
  - 生成完整的迁移映射表（CSV 或 Markdown 表格）：`old_path → new_path`
  - 输出文件：`.omo/evidence/file-migration-map.md`

  **Must NOT do**:
  - 不在此任务移动或重命名任何文件
  - 不修改任何文件内容

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 5, 7)
  - **Blocks**: ALL Wave 2 rename tasks (8-30)
  - **Blocked By**: None (独立任务)

  **Acceptance Criteria**:
  - [ ] 迁移映射表覆盖根目录所有 .py 文件
  - [ ] 每个文件都有明确的新名称和新位置
  - [ ] 映射表保存至 `.omo/evidence/file-migration-map.md`

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 迁移映射表完整
    Tool: Bash
    Steps:
      1. wc -l .omo/evidence/file-migration-map.md (至少 40 行)
      2. grep -c "→" .omo/evidence/file-migration-map.md (至少 30 个映射)
    Expected Result: 映射表覆盖所有根目录 .py 文件
    Evidence: .omo/evidence/task-6-migration-map.txt
  ```
  **Commit**: NO

- [ ] 7. 分析 model/ 和 dataset/ 的 import 关系图

  **What to do**:
  - 分析 model/ 下所有 .py 文件的 import 依赖关系
  - 分析 dataset/ 下所有 .py 文件的 import 依赖关系
  - 生成依赖关系图（Markdown 格式）：哪些文件导入了 `model.main_model`，哪些导入了 `model.*_collect_atten`
  - 输出文件：`.omo/evidence/import-dependency-graph.md`

  **Must NOT do**:
  - 不修改任何文件
  - 不运行任何训练脚本

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 5, 6)
  - **Blocks**: Tasks 44-56 (import updates in Wave 4)
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] 列出了所有导入 `model.main_model` 的文件路径
  - [ ] 列出了所有导入 `main_model_collect_atten` 的文件路径
  - [ ] 依赖图保存至 `.omo/evidence/import-dependency-graph.md`

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 依赖图完整
    Tool: Bash
    Steps:
      1. grep -r "from model.main_model import" --include="*.py" . | wc -l (记录引用数)
      2. grep -r "from model.mrmodn" --include="*.py" . | wc -l (重命名后应为相同数量)
    Expected Result: 找到所有 import 引用
    Evidence: .omo/evidence/task-7-dependency-graph.txt
  ```
  **Commit**: NO

---

- [ ] 8. 重命名根目录训练脚本（Group A: mRModN 模型）

  **What to do**:
  - `train_human.py` → `train_human_mrmodn.py` (git mv)
  - `train_plant.py` → `train_plant_mrmodn.py` (git mv)
  - `train_multirm_dataset.py` → `train_multirm_mrmodn.py` (git mv)
  - 验证每个文件内容完整

  **Must NOT do**:
  - 不修改文件内容
  - 不更新 import 引用（Wave 4 统一处理）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 9-16)
  - **Blocks**: Tasks 44+ (import updates)
  - **Blocked By**: Task 6 (mapping confirmation)

  **Acceptance Criteria**:
  - [ ] `train_human_mrmodn.py` 存在，`train_human.py` 已删除
  - [ ] `train_plant_mrmodn.py` 存在，`train_plant.py` 已删除
  - [ ] `train_multirm_mrmodn.py` 存在，`train_multirm_dataset.py` 已删除

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 训练脚本重命名成功
    Tool: Bash
    Steps:
      1. test -f train_human_mrmodn.py && echo "OK"
      2. test ! -f train_human.py && echo "REMOVED"
      3. test -f train_plant_mrmodn.py && echo "OK"
      4. test -f train_multirm_mrmodn.py && echo "OK"
    Expected Result: 所有新名称存在，旧名称不存在
    Evidence: .omo/evidence/task-8-rename-train-groupA.txt
  ```
  **Commit**: NO

- [ ] 9. 重命名根目录训练脚本（Group B: 变体模型 - 已规范，确认即可）

  **What to do**:
  - `train_human_evormd.py` → 已符合规范，确认不变
  - `train_human_modx.py` → 已符合规范，确认不变
  - `train_human_multirm.py` → 已符合规范，确认不变
  - 仅创建验证报告：`.omo/evidence/task-9-training-naming-check.md`

  **Must NOT do**:
  - 不重命名这些文件（已规范）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 8, 10-16)
  - **Blocks**: None
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] 确认这3个文件命名已符合 `train_数据集_模型.py` 规范

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 命名规范验证
    Tool: Bash
    Steps:
      1. ls train_human_evormd.py train_human_modx.py train_human_multirm.py
    Expected Result: 3个文件都存在，命名符合规范
    Evidence: .omo/evidence/task-9-naming-check.txt
  ```
  **Commit**: NO

- [ ] 10. 重命名根目录推理脚本（保留 segmented/full 模式信息）

  **What to do**:
  - `inference_evormd_segmented.py` → `inference_human_evormd_segmented.py`
  - `inference_modx_segmented.py` → `inference_human_modx_segmented.py`
  - `inference_mrmodn_full.py` → `inference_human_mrmodn_full.py`
  - `inference_multirm_segmented.py` → `inference_multirm_multirm_segmented.py`
  - atten_comp 中的 `inference_modx_full.py` → 同理保留 full
  - 读取每个文件头部确认其使用的数据集 (human/plant/multirm)

  **Must NOT do**:
  - 不修改文件内容
  - 不更新 import

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 8-9, 11-16)
  - **Blocks**: Task 24 (import updates)
  - **Blocked By**: None（依赖 Task 6 映射表确认）

  **Acceptance Criteria**:
  - [ ] 4个推理脚本已按规范重命名，segmented/full 模式信息保留
  - [ ] 旧文件名不再存在
  - [ ] `inference_human_mrmodn_full.py` 存在

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 推理脚本重命名（保留模式）
    Tool: Bash
    Steps:
      1. ls inference_human_evormd_segmented.py inference_human_modx_segmented.py inference_human_mrmodn_full.py inference_multirm_multirm_segmented.py
      2. test ! -f inference_mrmodn_full.py && echo "OLD REMOVED"
    Expected Result: 4个新文件存在，旧文件已删除，模式信息保留
    Evidence: .omo/evidence/task-10-rename-inference.txt
  ```
  **Commit**: NO

- [ ] 11. 重命名根目录数据收集脚本

  **What to do**:
  - `collect_human.py` → `collect_human_mrmodn.py`
  - `collect_human_atten.py` → `collect_human_mrmodn_atten.py` (或保持简洁: `collect_atten_human_mrmodn.py`)
  - `collect_modx_atten.py` → `collect_atten_human_modx.py`
  - `collect_multirm_atten.py` → `collect_atten_multirm_multirm.py`
  - 检查原文件头部注释确认使用的模型和数据集

  **Must NOT do**:
  - 不修改文件内容

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 44+ (import updates)
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] collect_ 系列脚本已按规范重命名
  - [ ] 旧名称不再存在

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 收集脚本重命名
    Tool: Bash
    Steps:
      1. ls collect_*.py | sort
      2. 确认所有 collect 脚本都已重命名
    Expected Result: 所有collect脚本按规范命名
    Evidence: .omo/evidence/task-11-rename-collect.txt
  ```
  **Commit**: NO

- [ ] 12. 重命名少样本/零样本分析脚本

  **What to do**:
  - `fewshot_ac4c_balance.py` → `fewshot_ac4c_mrmodn_balance.py`
  - `fewshot_ac4c_unbalan.py` → `fewshot_ac4c_mrmodn_unbalan.py`
  - `fewshot_plant_3way_independent.py` → `fewshot_plant_mrmodn_3way.py`
  - `zero_shot_fewshot_analysis.py` → `zeroshot_human_mrmodn_analysis.py`
  - `zero_shot_fewshot_extract_only.py` → `zeroshot_human_mrmodn_extract.py`
  - 检查文件头部确认数据集和模型

  **Must NOT do**:
  - 不修改文件内容
  - 不过度拆分文件名（保持可读性）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 44+
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] 5个 fewshot/zeroshot 脚本已重命名

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: fewshot 脚本重命名
    Tool: Bash
    Steps:
      1. ls fewshot_*.py zero_shot*.py zeroshot_*.py 2>/dev/null | sort
    Expected Result: 所有文件名符合规范
    Evidence: .omo/evidence/task-12-rename-fewshot.txt
  ```
  **Commit**: NO

- [ ] 13. 重命名并将测试脚本移入 tests/ 目录

  **What to do**:
  - `test_gen3.py` → `tests/test_gen3.py`（保持原名或标准化）
  - `test_multirm_4class.py` → `tests/test_multirm_4class.py`
  - `test_multirm_oversampling.py` → `tests/test_multirm_oversampling.py`
  - 使用 git mv 保留历史

  **Must NOT do**:
  - 不修改测试脚本内容（Wave 5 统一整理）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 57+
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] 根目录不再有 `test_*.py`
  - [ ] `tests/` 目录下有3个测试文件

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 测试脚本移入 tests/
    Tool: Bash
    Steps:
      1. ls tests/test_*.py | wc -l (应 >= 3)
      2. test ! -f test_gen3.py && echo "MOVED"
    Expected Result: tests/下有3个文件，根目录无 test_*.py
    Evidence: .omo/evidence/task-13-move-tests.txt
  ```
  **Commit**: NO

- [ ] 14. 重命名并将消融实验脚本移入 ablation/ 目录

  **What to do**:
  - `3x3.py` → `ablation/ablation_3x3_human_mrmodn.py`
  - `3x3_2.py` → `ablation/ablation_3x3_v2_human_mrmodn.py`
  - `abla_mohe.py` → `ablation/ablation_mohe_human_mrmodn.py`
  - `cal_flops_mohe.py` → `ablation/cal_flops_human_mrmodn.py`
  - `cal_mean_median_mode.py` → `ablation/cal_stats_human_mrmodn.py`（或保留原名）
  - 使用 git mv

  **Must NOT do**:
  - 不修改文件内容

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 51+ (import updates)
  - **Blocked By**: Task 1 (ablation/ 目录存在)

  **Acceptance Criteria**:
  - [ ] 根目录不再有 `3x3.py`、`abla_mohe.py` 等
  - [ ] `ablation/` 目录有 5 个脚本

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 消融脚本移入 ablation/
    Tool: Bash
    Steps:
      1. ls ablation/ablation_*.py ablation/cal_*.py | wc -l (应 >= 5)
      2. test ! -f 3x3.py && test ! -f abla_mohe.py && echo "MOVED"
    Expected Result: ablation/ 目录有5个文件，根目录已清理
    Evidence: .omo/evidence/task-14-move-ablation.txt
  ```
  **Commit**: NO

- [ ] 15. 重命名并将可视化/视图脚本移入 visualization/ 目录

  **What to do**:
  - `view_human.py` → `visualization/human/mrmodn/view_data.py`（或 `visualization/view_human_mrmodn.py`）
  - `view_human_total.py` → `visualization/human/mrmodn/view_total.py`
  - `view3.py` → `visualization/human/mrmodn/view_v3.py`
  - `view_npz.py` → `visualization/tools/view_npz.py`
  - `visualize_attention_comparison.py` → `visualization/human/mrmodn/attention_comparison.py`
  - `run_attention_comparison.py` → `visualization/human/mrmodn/run_attention_comparison.py`
  - `select_representative_sequences.py` → `visualization/human/mrmodn/select_representatives.py`
  - `prepare_umap_data.py` → `visualization/tools/prepare_umap_data.py`
  - `prepare_umap_from_npz.py` → `visualization/tools/prepare_umap_from_npz.py`
  - `SpatialMotif.py` → `visualization/human/mrmodn/spatial_motif.py`
  - `SpatialMotif_nobackground.py` → `visualization/human/mrmodn/spatial_motif_nobg.py`
  - 使用 git mv

  **Must NOT do**:
  - 不重命名 `fig/`、`att_fig/` 目录本身（保持输出目录）

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 44+ (import updates)
  - **Blocked By**: Task 1 (visualization/ 目录存在)

  **Acceptance Criteria**:
  - [ ] 根目录不再有 `view_*.py`、`SpatialMotif*.py`、`visualize_*.py` 等
  - [ ] `visualization/` 下按数据集/模型分布文件

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 可视化脚本移入 visualization/
    Tool: Bash
    Steps:
      1. find visualization/ -name "*.py" | wc -l (应 >= 10)
      2. test ! -f view_human.py && test ! -f SpatialMotif.py && echo "MOVED"
    Expected Result: visualization/ 有10+个文件，根目录已清理
    Evidence: .omo/evidence/task-15-move-visualization.txt
  ```
  **Commit**: NO

- [ ] 16. 处理工具/辅助脚本移入对应目录

  **What to do**:
  - `sliding_window_utils.py` → `utils/sliding_window_utils.py` (已在 utils/ 或保留？检查是否已存在)
  - 如果根目录存在 `sliding_window_utils.py` 而 `utils/` 没有 → git mv
  - 如果 `utils/` 已有副本 → 删除根目录的（做备份确认）
  - 检查 `utils/` 目录中是否已有同名文件，避免覆盖

  **Must NOT do**:
  - 不覆盖 `utils/` 中已有文件
  - 先对比确认再操作

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 54+
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] 根目录下 `sliding_window_utils.py` 已处理（移入 utils/ 或确认删除冗余）

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: sliding_window_utils 处理
    Tool: Bash
    Steps:
      1. test -f utils/sliding_window_utils.py && echo "IN UTILS"
      2. test ! -f sliding_window_utils.py && echo "ROOT CLEAN"
    Expected Result: 文件在 utils/，根目录已清理
    Evidence: .omo/evidence/task-16-sliding-window.txt
  ```
  **Commit**: NO

- [ ] 17. 处理 R 脚本移入 analysis/ 目录

  **What to do**:
  - `plot_zero_fewshot_analysis.R` → `analysis/plot_zero_fewshot_analysis.R`
  - `plot_zero_fewshot_export.R` → `analysis/plot_zero_fewshot_export.R`
  - `Rplots.pdf` → `analysis/Rplots.pdf`
  - 使用 git mv

  **Must NOT do**:
  - 不修改 R 脚本内容
  - 不修改脚本中可能硬编码的文件路径（留待后续检查）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: None
  - **Blocked By**: Task 1 (analysis/ 目录存在)

  **Acceptance Criteria**:
  - [ ] 根目录不再有 `*.R` 和 `Rplots.pdf`
  - [ ] `analysis/` 目录下有 R 文件和 PDF

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: R 脚本移入 analysis/
    Tool: Bash
    Steps:
      1. ls analysis/*.R analysis/*.pdf | wc -l (应 >= 3)
      2. test ! -f plot_zero_fewshot_analysis.R && echo "MOVED"
    Expected Result: analysis/ 有 R 文件，根目录已清理
    Evidence: .omo/evidence/task-17-move-R.txt
  ```
  **Commit**: NO

- [ ] 18. 处理 motif_logo/ 和 fig/ 目录

  **What to do**:
  - `motif_logo/` → `visualization/motif_logo/`（保持子目录结构）
  - `fig/` 中的图片 → `visualization/fig/`
  - `att_fig/` 中的图片 → `visualization/att_fig/`
  - 使用 git mv 移动整个目录

  **Must NOT do**:
  - 不删除原有图片
  - 不修改图片内容

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: None
  - **Blocked By**: Task 1 (visualization/ 目录存在)

  **Acceptance Criteria**:
  - [ ] `visualization/motif_logo/` 存在
  - [ ] `visualization/fig/` 存在
  - [ ] 根目录不再有 `motif_logo/`、`fig/`、`att_fig/`（或保留为空引用）

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 图片目录移动
    Tool: Bash
    Steps:
      1. ls visualization/motif_logo/ | head -5
      2. ls visualization/fig/ | head -5
      3. test -d motif_logo && echo "STILL EXISTS" || echo "MOVED"
    Expected Result: 目录在 visualization/ 下
    Evidence: .omo/evidence/task-18-move-figures.txt
  ```
  **Commit**: NO

---

- [ ] 19. 合并 atten_comp/ 子目录到 visualization/

  **What to do**:
  - 读取 `atten_comp/` 下的所有文件
  - `atten_comp/inference_modx_full.py` → `visualization/human/modx/inference_full.py`
  - `atten_comp/run_attention_comparison_v2.py` → `visualization/human/mrmodn/run_attention_comparison_v2.py`
  - `atten_comp/visualize_attention_comparison_v2.py` → `visualization/human/mrmodn/visualize_attention_comparison_v2.py`
  - 移动后删除空的 `atten_comp/` 目录

  **Must NOT do**:
  - 不丢失文件
  - 不修改文件内容

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 20-22)
  - **Blocks**: Tasks 44+ (import updates)
  - **Blocked By**: Task 1 (visualization/ 目录存在), Tasks 8-18 (root file cleanup done)

  **Acceptance Criteria**:
  - [ ] `atten_comp/` 目录为空或已删除
  - [ ] 3个文件已移入 visualization/ 对应位置

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: atten_comp 合并
    Tool: Bash
    Steps:
      1. test ! -d atten_comp || test -z "$(ls atten_comp/)" && echo "CLEAN"
      2. find visualization/ -name "*attention_comparison*" | wc -l (应 >= 2)
    Expected Result: atten_comp/ 已空，文件在 visualization/ 下
    Evidence: .omo/evidence/task-19-merge-atten-comp.txt
  ```
  **Commit**: NO

- [ ] 20. 整理 ipynb/ Jupyter notebooks

  **What to do**:
  - 列出 `ipynb/` 下所有 34 个 notebook
  - 按内容分类（分类对比 cls_compare、定位对比 loc_compare、UMAP umap、主模型分析 rgcnformer 等）
  - 移动至 `visualization/notebooks/` 目录下按类别分子目录
  - 保持原文件名不变

  **Must NOT do**:
  - 不修改 notebook 内容
  - 不删除任何 notebook

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 19, 21-22)
  - **Blocks**: None
  - **Blocked By**: Task 1 (visualization/ 目录存在)

  **Acceptance Criteria**:
  - [ ] 34 个 notebook 已移入 `visualization/notebooks/`
  - [ ] `ipynb/` 目录为空或已删除
  - [ ] 所有 notebook 路径可访问

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: notebooks 移动
    Tool: Bash
    Steps:
      1. find visualization/notebooks/ -name "*.ipynb" | wc -l (应 >= 30)
      2. test ! -d ipynb || test -z "$(ls ipynb/)" && echo "CLEAN"
    Expected Result: 30+ notebooks 在 visualization/notebooks/
    Evidence: .omo/evidence/task-20-move-notebooks.txt
  ```
  **Commit**: NO

- [ ] 21. 整理 visualization/ 目录结构

  **What to do**:
  - 验证 visualization/ 下的完整目录结构
  - 确保每个数据集目录（human、plant、multirm、ac4c、gen3）存在
  - 每个数据集目录下有对应模型子目录（mrmodn、evormd、modx、multirm）
  - 创建 `visualization/tools/` 用于通用可视化工具
  - 创建 `visualization/notebooks/` 用于 Jupyter notebooks
  - 创建 `visualization/fig/` 用于静态图片
  - 创建 `visualization/README.md` 说明目录结构

  **Must NOT do**:
  - 不移动已就位的文件
  - 不修改可视化脚本中的输出路径

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖 Tasks 15, 18-20）
  - **Parallel Group**: Wave 3 (sequential)
  - **Blocks**: None
  - **Blocked By**: Tasks 15, 18, 19, 20

  **Acceptance Criteria**:
  - [ ] `visualization/human/mrmodn/` 有可视化脚本
  - [ ] `visualization/tools/` 有通用工具
  - [ ] `visualization/notebooks/` 有 notebooks
  - [ ] `visualization/README.md` 存在

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: visualization 结构验证
    Tool: Bash
    Steps:
      1. tree visualization/ -L 2 -d
      2. find visualization/ -name "*.py" -o -name "*.ipynb" -o -name "*.png" | wc -l
      3. cat visualization/README.md | head -5
    Expected Result: 完整的两级目录结构 + README
    Evidence: .omo/evidence/task-21-viz-structure.txt
  ```
  **Commit**: NO

- [ ] 22. 创建 ablation/README.md 并整理消融目录

  **What to do**:
  - 验证 `ablation/` 下所有文件已就位
  - 创建 `ablation/README.md` 说明每个消融实验的用途
  - 确保 `ablation/` 目录结构清晰

  **Must NOT do**:
  - 不创建额外子目录（保持扁平）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 19-20)
  - **Blocks**: None
  - **Blocked By**: Task 14

  **Acceptance Criteria**:
  - [ ] `ablation/README.md` 存在
  - [ ] `ablation/` 有 5+ 个 Python 脚本

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: ablation 目录验证
    Tool: Bash
    Steps:
      1. ls ablation/*.py | wc -l (应 >= 5)
      2. test -f ablation/README.md && echo "README OK"
    Expected Result: ablation/ 完整
    Evidence: .omo/evidence/task-22-ablation-check.txt
  ```
  **Commit**: NO

---

- [ ] 23. 更新根目录训练脚本的 import 引用

  **What to do**:
  - 读取所有根目录训练脚本
  - 搜索每个文件中的 `from model.main_model import` 替换为 `from model.mrmodn import`
  - 搜索 `from model.main_model_collect_atten import` 替换为 `from model.mrmodn_collect_atten import`
  - 搜索 `import model.main_model` 替换为 `import model.mrmodn`
  - 更新对已移动文件的任何引用

  **Must NOT do**:
  - 不改变导入的符号名（类名、函数名）
  - 不修改训练逻辑

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES（训练脚本间独立）
  - **Parallel Group**: Wave 4 (parallel group)
  - **Blocks**: Tasks 29+ (运行验证)
  - **Blocked By**: Tasks 8, 9, 10 (重命名完成)

  **Acceptance Criteria**:
  - [ ] 所有训练脚本中不再有 `from model.main_model import`
  - [ ] 所有训练脚本中不再有 `from model.main_model_collect_atten import`
  - [ ] 语法检查通过

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 训练脚本 import 更新验证
    Tool: Bash
    Steps:
      1. grep -r "model.main_model" train_*.py && echo "FOUND OLD IMPORT" || echo "CLEAN"
      2. for f in train_*.py; do python -c "import ast; ast.parse(open('$f').read())" && echo "$f: SYNTAX OK"; done
    Expected Result: 无旧 import，所有语法正确
    Evidence: .omo/evidence/task-23-train-imports.txt
  ```
  **Commit**: NO

- [ ] 24. 更新根目录推理脚本的 import 引用

  **What to do**:
  - 更新所有 `inference_*.py` 中的旧 import
  - 替换 `model.main_model` → `model.mrmodn`

  **Must NOT do**:
  - 不改变推理逻辑

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (parallel with Tasks 23, 25-28)
  - **Blocks**: Tasks 29+
  - **Blocked By**: Task 10

  **Acceptance Criteria**:
  - [ ] 所有推理脚本 import 已更新
  - [ ] 语法检查通过

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 推理脚本 import 更新验证
    Tool: Bash
    Steps:
      1. grep -r "model.main_model" inference_*.py && echo "FOUND" || echo "CLEAN"
      2. for f in inference_*.py; do python -c "import ast; ast.parse(open('$f').read())" && echo "$f: OK"; done
    Expected Result: CLEAN，所有语法正确
    Evidence: .omo/evidence/task-24-inference-imports.txt
  ```
  **Commit**: NO

- [ ] 25. 更新根目录其他脚本的 import 引用

  **What to do**:
  - 更新 collect_*.py、fewshot_*.py、zeroshot_*.py 中所有旧 import
  - 替换 `model.main_model` → `model.mrmodn`

  **Must NOT do**:
  - 不遗漏任何文件

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (parallel with Tasks 23-24, 26-28)
  - **Blocks**: Tasks 29+
  - **Blocked By**: Tasks 11, 12

  **Acceptance Criteria**:
  - [ ] collect_*.py、fewshot_*.py、zeroshot_*.py 中无旧 import

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 根目录其他脚本 import 验证
    Tool: Bash
    Steps:
      1. grep -rn "model.main_model" collect_*.py fewshot_*.py zero*_*.py zeroshot_*.py && echo "FOUND" || echo "CLEAN"
    Expected Result: CLEAN
    Evidence: .omo/evidence/task-25-root-imports.txt
  ```
  **Commit**: NO

- [ ] 26. 更新 model/ 目录内部 import 引用

  **What to do**:
  - 更新 `model/mrmodn.py` 内部（如果有自引用）
  - 更新 `model/multirm.py`、`model/modx.py`、`model/evormd_human.py` 中的 `from .main_model` → `from .mrmodn`
  - 更新所有 `model/*_collect_atten.py` 中的 import
  - 更新 `model/abla_model.py` 中的 import

  **Must NOT do**:
  - 不改变模块功能

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO（model 内部有复杂依赖）
  - **Parallel Group**: Wave 4 (sequential)
  - **Blocks**: Tasks 29+
  - **Blocked By**: Tasks 2, 3, 4

  **Acceptance Criteria**:
  - [ ] model/ 目录下所有文件语法正确
  - [ ] model/ 内部无旧 import
  - [ ] `python -c "from model import RNA_ClassQuery_Model"` 成功

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: model/ import 验证
    Tool: Bash
    Steps:
      1. grep -rn "main_model" model/ --include="*.py" && echo "FOUND OLD" || echo "CLEAN"
      2. python -c "from model.mrmodn import RNA_ClassQuery_Model; print('OK')"
      3. python -c "from model import *; print('ALL IMPORTS OK')"
    Expected Result: CLEAN，所有导入成功
    Evidence: .omo/evidence/task-26-model-imports.txt
  ```
  **Commit**: NO

- [ ] 27. 更新 dataset/ 目录中的 import 引用

  **What to do**:
  - 检查 `dataset/` 下所有 .py 文件
  - 如果有引用 `model.main_model` → 更新为 `model.mrmodn`
  - 验证 dataset 加载逻辑不受影响

  **Must NOT do**:
  - 不改变数据加载逻辑
  - 不修改数据路径

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (parallel with Tasks 23-25, 28)
  - **Blocks**: Tasks 29+
  - **Blocked By**: Tasks 2, 4

  **Acceptance Criteria**:
  - [ ] dataset/ 目录下无旧 import
  - [ ] 所有 dataset 文件语法正确

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: dataset import 验证
    Tool: Bash
    Steps:
      1. grep -rn "main_model" dataset/ --include="*.py" && echo "FOUND" || echo "CLEAN"
      2. for f in dataset/*.py; do python -c "import ast; ast.parse(open('$f').read())" && echo "$f: OK"; done
    Expected Result: CLEAN
    Evidence: .omo/evidence/task-27-dataset-imports.txt
  ```
  **Commit**: NO

- [ ] 28. 更新 utils/ 和子目录中的 import 引用

  **What to do**:
  - 更新 `utils/` 下所有文件中的旧 import
  - 更新 `visualization/` 下文件的 import
  - 更新 `ablation/` 下文件的 import
  - 更新 `analysis/` 下可能引用的路径

  **Must NOT do**:
  - 不修改业务逻辑

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (parallel with Tasks 23-27)
  - **Blocks**: Tasks 29+
  - **Blocked By**: Tasks 2-4, 14-16

  **Acceptance Criteria**:
  - [ ] `utils/` 目录下无旧 import
  - [ ] `visualization/` 下文件语法正确
  - [ ] `ablation/` 下文件语法正确

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: utils/ + 子目录 import 验证
    Tool: Bash
    Steps:
      1. grep -rn "model.main_model" utils/ visualization/ ablation/ --include="*.py" && echo "FOUND" || echo "CLEAN"
      2. python -c "import utils; print('UTILS OK')"
    Expected Result: CLEAN
    Evidence: .omo/evidence/task-28-utils-imports.txt
  ```
  **Commit**: NO

---

- [ ] 29. 全项目 import 完整性验证

  **What to do**:
  - 编写并运行脚本：遍历所有 .py 文件，尝试语法检查
  - 对每个有 import 的文件，验证被导入模块确实存在
  - 生成报告：`.omo/evidence/import-validation-report.md`

  **Must NOT do**:
  - 不执行训练或推理
  - 不修复发现的 import 错误（记录到报告）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖所有 Wave 4 任务）
  - **Parallel Group**: Wave 5 (sequential)
  - **Blocks**: Tasks 30-32
  - **Blocked By**: Tasks 23-28

  **Acceptance Criteria**:
  - [ ] 至少 90% 的文件通过语法检查
  - [ ] 导入验证报告记录了所有问题

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 全项目语法检查
    Tool: Bash
    Steps:
      1. find . -name "*.py" -not -path "./.git/*" -not -path "./.om*/*" | while read f; do python -c "import ast; ast.parse(open('$f').read())" 2>&1 || echo "FAIL: $f"; done
      2. 统计 PASS/FAIL 数量
    Expected Result: >= 90% PASS
    Evidence: .omo/evidence/task-29-syntax-check.txt
  ```
  **Commit**: NO

- [ ] 30. 编写核心功能测试

  **What to do**:
  - `tests/test_model_import.py`: 验证所有模型类可以成功导入
  - `tests/test_dataset_import.py`: 验证所有 dataset 类可以成功导入
  - `tests/test_config.py`: 验证 JSON 配置文件存在且格式正确
  - `tests/test_utils_import.py`: 验证 utils 模块可以成功导入
  - `tests/test_naming_convention.py`: 验证所有文件遵循命名规范

  **Must NOT do**:
  - 不编写需要 GPU 或大数据集的测试

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES（多个测试文件独立）
  - **Parallel Group**: Wave 5 (parallel with Task 31)
  - **Blocks**: Task 32
  - **Blocked By**: Tasks 2-4, 23-28

  **Acceptance Criteria**:
  - [ ] 5 个测试文件存在
  - [ ] `python -m pytest tests/ -v` 能运行所有测试

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 运行测试套件
    Tool: Bash
    Steps:
      1. python -m pytest tests/ -v --tb=short 2>&1 | tee .omo/evidence/task-30-test-output.txt
      2. grep -E "PASSED|FAILED" .omo/evidence/task-30-test-output.txt | tail -5
    Expected Result: 大部分测试 PASS
    Evidence: .omo/evidence/task-30-test-output.txt
  ```
  **Commit**: NO

- [ ] 31. 根目录训练脚本基本运行验证

  **What to do**:
  - 对每个训练脚本验证基本可用（import 阶段不报错）
  - 记录验证报告

  **Must NOT do**:
  - 不运行完整训练流程

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (parallel with Task 30)
  - **Blocks**: Task 32
  - **Blocked By**: Tasks 23-28

  **Acceptance Criteria**:
  - [ ] 所有训练脚本 import 阶段不报错
  - [ ] 记录验证报告：`.omo/evidence/task-31-training-check.md`

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 训练脚本基本运行验证
    Tool: Bash
    Steps:
      1. python -c "
import sys, subprocess
scripts = ['train_human_mrmodn.py', 'train_human_evormd.py']
for s in scripts:
    r = subprocess.run(['python', '-c', f'exec(open(\"{s}\").read().split(\"if __name__\")[0])'], capture_output=True, text=True)
    print(f'{s}: {\"OK\" if r.returncode == 0 else \"FAIL - \"+r.stderr[:100]}')
"
    Expected Result: 基本语法/导入检查通过
    Evidence: .omo/evidence/task-31-training-check.txt
  ```
  **Commit**: NO

- [ ] 32. 命名规范最终验证与报告

  **What to do**:
  - 编写脚本扫描所有 .py 文件，检查 `任务_数据集_模型.py` 命名规范
  - 生成最终命名审计报告：`.omo/evidence/naming-audit-report.md`

  **Must NOT do**:
  - 不在此任务修改文件名

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖所有重命名任务）
  - **Parallel Group**: Wave 5 (sequential)
  - **Blocks**: Final verification
  - **Blocked By**: Tasks 8-18, 23-28

  **Acceptance Criteria**:
  - [ ] 命名审计报告生成
  - [ ] 至少 95% 的文件符合命名规范

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: 命名规范审计
    Tool: Bash
    Steps:
      1. python -c "
import os, re
root_files = [f for f in os.listdir('.') if f.endswith('.py')]
pattern = re.compile(r'^(train|inference|collect|fewshot|zeroshot)_[a-z0-9]+_[a-z0-9]+\.py$')
compliant = [f for f in root_files if pattern.match(f)]
non_compliant = [f for f in root_files if not pattern.match(f)]
print(f'Compliant: {len(compliant)}/{len(root_files)}')
print(f'Non-compliant: {non_compliant}')
"
    Expected Result: >= 95% 规范
    Evidence: .omo/evidence/task-32-naming-audit.txt
  ```
  **Commit**: NO

- [ ] 33. 更新 README.md 反映新的项目结构

  **What to do**:
  - 更新 `README.md` 中的目录结构图
  - 更新训练命令示例中的文件名
  - 添加新的 `visualization/`、`ablation/`、`tests/`、`analysis/` 目录说明
  - 保持中英双语格式

  **Must NOT do**:
  - 不删除原有的技术文档内容

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (parallel with Tasks 30-31)
  - **Blocks**: None
  - **Blocked By**: Tasks 8-22

  **Acceptance Criteria**:
  - [ ] README 中的目录结构反映新组织
  - [ ] 训练命令示例使用新文件名

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: README 更新验证
    Tool: Bash
    Steps:
      1. grep "train_human_mrmodn.py" README.md
      2. grep "visualization/" README.md
      3. grep "model/mrmodn.py" README.md
    Expected Result: 三处都能找到新名称
    Evidence: .omo/evidence/task-33-readme-update.txt
  ```
  **Commit**: NO

- [ ] 34. 提交所有变更

  **What to do**:
  - `git status` 确认所有变更
  - `git add -A` 暂存所有变更
  - 提交：`git commit -m "refactor: reorganize project with task_dataset_model naming convention"`

  **Must NOT do**:
  - 不推送（留待用户确认）
  - 不包含未跟踪的秘密文件

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖所有任务）
  - **Parallel Group**: Wave 5 (final task)
  - **Blocks**: None
  - **Blocked By**: Tasks 29-33

  **Acceptance Criteria**:
  - [ ] `git status` 干净
  - [ ] 提交信息清晰
  - [ ] `git log -1` 显示新提交

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Git 提交验证
    Tool: Bash
    Steps:
      1. git status --short (应为空)
      2. git log -1 --oneline
      3. git diff --stat HEAD~1 | wc -l
    Expected Result: 工作区干净，提交存在
    Evidence: .omo/evidence/task-34-commit.txt
  ```
  **Commit**: YES（这是唯一一个单独提交的任务）
  - Message: `refactor: reorganize project with task_dataset_model naming convention`
  - Files: ALL changed files
  - Pre-commit: `find . -name "*.py" -not -path "./.git/*" | while read f; do python -c "import ast; ast.parse(open('$f').read())" || exit 1; done`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns. Check evidence files exist in .omo/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `find . -name "*.py" -not -path "./.git/*" | while read f; do python -c "import ast; ast.parse(open('$f').read())" || echo "SYNTAX ERROR: $f"; done`. Review all changed files for: broken imports, stale references, empty catches, commented-out code. Check import consistency: no `model.main_model` references remain.
  Output: `Syntax [N PASS/N FAIL] | Import [CLEAN/DIRTY] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute key QA scenarios: (1) Verify all new directories exist, (2) Verify model import works: `python -c "from model.mrmodn import RNA_ClassQuery_Model"`, (3) Verify dataset import works: `python -c "from dataset.human import Mer100Dataset"`, (4) Verify naming convention compliance, (5) Verify pytest works: `python -m pytest tests/ -v`. Save to `.omo/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Tasks 1-33**: NO individual commits (grouped together)
- **Task 34**: YES - single comprehensive commit
  - Message: `refactor: reorganize project with task_dataset_model naming convention`
  - Files: ALL changed files
  - Pre-commit: Syntax check on all Python files

---

## Success Criteria

### Verification Commands
```bash
# Syntax check all Python files
find . -name "*.py" -not -path "./.git/*" -not -path "./.om*/*" | while read f; do python -c "import ast; ast.parse(open('$f').read())" || echo "FAIL: $f"; done

# Model import verification
python -c "from model.mrmodn import RNA_ClassQuery_Model; print('OK')"
python -c "from model import RNA_ClassQuery_Model; print('OK')"

# Dataset import verification
python -c "from dataset.human import Mer100Dataset; print('OK')"

# Test suite
python -m pytest tests/ -v

# Naming convention audit
python -c "
import os, re
root_files = [f for f in os.listdir('.') if f.endswith('.py')]
pattern = re.compile(r'^(train|inference|collect|fewshot|zeroshot)_[a-z0-9]+_[a-z0-9]+\.py$')
compliant = [f for f in root_files if pattern.match(f)]
print(f'Root compliance: {len(compliant)}/{len(root_files)}')
"
```

### Final Checklist
- [ ] All "Must Have" present (npy/ untouched, imports updated, training scripts in root)
- [ ] All "Must NOT Have" absent (no model.main_model references, npy/ unchanged, no deleted files)
- [ ] All new directories created and populated
- [ ] All tests pass
- [ ] README updated
- [ ] Single clean git commit
