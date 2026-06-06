# VitePress GitBook — RGCNFormer/mRModN 项目教程

## TL;DR

> **Quick Summary**: 为 RGCNFormer RNA 修饰分类项目构建 VitePress 静态文档站（GitBook），包含 11 个中英双语章节；同时为所有 87 个 Python 文件的每个函数添加双语 docstring。
>
> **Deliverables**:
> - VitePress 文档项目（`docs/` 目录，含 zh/ 和 en/ 独立语言页面 + 语言切换器）
> - 11 个章节 × 2 种语言 = 22 个 Markdown 教程页面
> - 87 个 Python 文件的函数级双语 docstring
> - Python 语法编译验证 + VitePress 构建验证
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 5 waves
> **Critical Path**: T1 (VitePress setup) → T2 (template) → T3-T7 (code comments) → T8-T14 (chapters) → T15-T16 (verification)

---

## Context

### Original Request
用户期望为 RGCNFormer/mRModN 项目编写 GitBook 教程，包含：
1. 项目背景（基于论文 mRModN: Mixture of Hierarchical Experts with self-adaptive balanced sampling）
2. 项目算法结构（M2D、MoHE、ABS、A2P 四大模块）
3. 对每一个文件的说明
4. 可中英双语切换
5. 主要启动流程

同时，用户要求在代码文件中为每个函数也加上双语注释（用途、调用、输入、输出）。

### Interview Summary
**Key Discussions**:
- **平台**: VitePress（静态文档生成器，支持 i18n，Markdown 编写）
- **双语方案**: 独立 zh/ 和 en/ 页面 + 顶部语言切换器
- **代码注释**: 函数级双语 docstring，覆盖全部 87 个 Python 文件的每个函数
- **读者定位**: 综合用途（兼顾论文评审 + 新人上手）
- **章节结构**: 11 章（概述→算法架构→快速开始→模型详解→数据集→训练→推理→可视化→消融→少零样本→附录）
- **验证策略**: py_compile + VitePress build + 链接检查 + 双语审计 + pytest 回归
- **部署**: 暂不部署，仅本地构建

**Research Findings**:
- 项目已有完整双语 README.md（289 行）和 93+ 文件的双语文件级头注释
- 论文内容已提取：详细描述了 M2D、MoHE、ABS、A2P 四大创新模块
- 核心架构：ParallelCNNBlock + GCNBlock + ClassQueryHead（三种模式）
- 数据流：one-hot → 多尺度 CNN → GCN 图传播 → 类查询注意力分类

---

## Work Objectives

### Core Objective
构建 VitePress 文档站 + 为全部 Python 文件添加函数级双语 docstring，形成完整的项目教程。

### Concrete Deliverables
- `docs/` 目录：VitePress 项目，含 config、zh/、en/、public/
- 11 个章节 × 2 语言 = 22 个 .md 教程文件
- 87 个 Python 文件的函数级双语 docstring
- 本地可构建的静态文档站

### Definition of Done
- [ ] 所有 87 个 Python 文件的每个函数/类方法都有双语 docstring
- [ ] VitePress 项目可成功构建 (`npm run docs:build`)
- [ ] 11 个章节在 zh/ 和 en/ 中各有一份完整内容
- [ ] 语言切换器正常工作
- [ ] `python -m py_compile` 对所有修改过的文件通过
- [ ] `pytest tests/ -v` 回归测试通过

### Must Have
- 函数级双语 docstring（用途、调用、输入、输出）
- VitePress i18n 配置（zh/en 独立页面 + 切换按钮）
- 11 个章节的完整双语内容
- 算法架构图（M2D、MoHE、ABS、A2P 流程）

### Must NOT Have (Guardrails)
- 不修改任何功能性代码（只添加注释/docstring）
- 不修改已有的文件级头注释（保持不动）
- 不部署到任何服务器
- 不添加 GitBook 之外的外部依赖
- 不创建新的训练/推理脚本
- 不修改 .omo/ 目录下的已有计划/证据文件
- 不在 VitePress 中嵌入交互式代码执行器（纯文档即可）

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES（项目已有 pytest 测试套件）
- **Automated tests**: Tests-after（注释添加后运行回归测试）
- **Framework**: pytest

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Python 注释**: `python -m py_compile` 语法检查
- **VitePress 构建**: `npm run docs:build` 验证
- **双语审计**: 检查 zh/ 和 en/ 文件数量一致、章节标题对应

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — start immediately):
├── T1: VitePress 项目脚手架 + i18n 配置 [quick]
└── T2: 双语函数 docstring 模板 + 编写指南 [quick]

Wave 2 (Code Comments — 5 parallel tasks):
├── T3: model/ 函数 docstrings (10 files) [unspecified-high]
├── T4: dataset/ 函数 docstrings (9 files) [unspecified-high]
├── T5: utils/ 函数 docstrings (20 files) [unspecified-high]
├── T6: 根目录 train/inference/collect 脚本函数 docstrings (16 files) [unspecified-high]
└── T7: 根目录 fewshot/zeroshot 脚本函数 docstrings (5 files) [unspecified-high]

Wave 3 (Remaining Code Comments + GitBook Core Chapters):
├── T8: ablation/ + visualization/ + tests/ 函数 docstrings (29 files) [unspecified-high]
├── T9: 章 Ch1 概述 + Ch2 算法架构 [zh + en] [writing]
├── T10: 章 Ch3 快速开始 + Ch4 模型详解 [zh + en] [writing]
└── T11: 章 Ch5 数据集 + Ch6 训练流程 [zh + en] [writing]

Wave 4 (GitBook Remaining Chapters):
├── T12: 章 Ch7 推理流程 + Ch8 可视化 [zh + en] [writing]
├── T13: 章 Ch9 消融实验 + Ch10 少零样本分析 [zh + en] [writing]
└── T14: 章 Ch11 附录 + 首页美化 [zh + en] [writing]

Wave 5 (Verification):
├── T15: Python 语法编译检查 (all 87 files) [quick]
└── T16: VitePress 构建验证 + 链接检查 + 双语审计 [quick]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── T-F1: Plan compliance audit (oracle)
├── T-F2: Code quality review (unspecified-high)
├── T-F3: VitePress build + 功能验证 (unspecified-high)
└── T-F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: T1 → T2 → T3-T7 → T9-T14 → T15-T16 → F1-F4 → user okay
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 5 (Wave 2)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| T1 | — | T3-T14 | 1 |
| T2 | — | T3-T8 | 1 |
| T3 | T1, T2 | T9-T14 | 2 |
| T4 | T1, T2 | T9-T14 | 2 |
| T5 | T1, T2 | T9-T14 | 2 |
| T6 | T1, T2 | T9-T14 | 2 |
| T7 | T1, T2 | T9-T14 | 2 |
| T8 | T1, T2 | T9-T14 | 3 |
| T9 | T1 | T15-T16 | 3 |
| T10 | T1, T3, T4 | T15-T16 | 3 |
| T11 | T1, T5, T6 | T15-T16 | 3 |
| T12 | T1, T6, T8 | T15-T16 | 4 |
| T13 | T1, T7, T8 | T15-T16 | 4 |
| T14 | T1 | T15-T16 | 4 |
| T15 | T3-T8 | F1-F4 | 5 |
| T16 | T9-T14 | F1-F4 | 5 |

### Agent Dispatch Summary

- **Wave 1**: 2 tasks → T1 `quick`, T2 `quick`
- **Wave 2**: 5 tasks → T3-T7 `unspecified-high`
- **Wave 3**: 4 tasks → T8 `unspecified-high`, T9-T11 `writing`
- **Wave 4**: 3 tasks → T12-T14 `writing`
- **Wave 5**: 2 tasks → T15-T16 `quick`
- **FINAL**: 4 tasks → F1 `oracle`, F2 `unspecified-high`, F3 `unspecified-high`, F4 `deep`

---

## TODOs

- [ ] 1. VitePress 项目脚手架 + i18n 配置

  **What to do**:
  - 在项目根目录创建 `docs/` 目录作为 VitePress 项目
  - 初始化 `docs/package.json`（含 `vuepress`、`vitepress` 依赖和 `docs:dev`、`docs:build`、`docs:preview` 脚本）
  - 创建 `docs/.vitepress/config.ts` 配置文件，包含：
    - `locales` 配置：`root`（中文）、`en`（英文），各自独立的 `label`、`lang`、`title`、`description`、`themeConfig`
    - 中文侧边栏：11 个章节（概述、算法架构、快速开始、模型详解、数据集、训练流程、推理流程、可视化、消融实验、少零样本分析、附录）
    - 英文侧边栏：对应 11 个章节
    - 中文导航栏 + 英文导航栏
    - 语言切换按钮
  - 创建目录结构：
    ```
    docs/
    ├── .vitepress/
    │   └── config.ts
    ├── zh/
    │   ├── index.md          (中文首页)
    │   ├── guide/
    │   │   ├── overview.md
    │   │   ├── architecture.md
    │   │   ├── quickstart.md
    │   │   ├── model.md
    │   │   ├── dataset.md
    │   │   ├── training.md
    │   │   ├── inference.md
    │   │   ├── visualization.md
    │   │   ├── ablation.md
    │   │   ├── fewshot.md
    │   │   └── appendix.md
    │   └── index.md
    ├── en/
    │   ├── index.md          (English homepage)
    │   ├── guide/
    │   │   ├── overview.md
    │   │   ├── architecture.md
    │   │   ├── quickstart.md
    │   │   ├── model.md
    │   │   ├── dataset.md
    │   │   ├── training.md
    │   │   ├── inference.md
    │   │   ├── visualization.md
    │   │   ├── ablation.md
    │   │   ├── fewshot.md
    │   │   └── appendix.md
    │   └── index.md
    └── public/
        └── logo.svg          (项目 logo 占位)
    ```
  - 创建 `docs/zh/index.md` 和 `docs/en/index.md` 首页模板（含 hero section + feature cards）
  - 运行 `cd docs && npm install && npm run docs:build` 验证配置正确

  **Must NOT do**:
  - 不要修改项目根目录的任何文件
  - 不要添加 VitePress 之外的框架依赖
  - 不要创建博客或 changelog 功能

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 脚手架配置任务，主要是文件创建和配置编写
  - **Skills**: []
    - 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T2)
  - **Blocks**: T3-T14（所有后续任务都依赖 docs/ 目录结构）
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `README.md:1-289` - 现有双语内容，用于首页和侧边栏文案参考
  - `.omo/drafts/gitbook.md` - 确认的章节结构和双语方案

  **External References**:
  - VitePress 官方文档: https://vitepress.dev/guide/i18n
  - VitePress 侧边栏配置: https://vitepress.dev/reference/default-theme-sidebar

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: VitePress 构建成功
    Tool: Bash
    Preconditions: docs/ 目录已创建，npm install 已完成
    Steps:
      1. cd docs && npm run docs:build
      2. 检查退出码是否为 0
    Expected Result: 构建成功，无错误，退出码 0
    Failure Indicators: 任何构建错误或非零退出码
    Evidence: .omo/evidence/task-1-vitepress-build.txt

  Scenario: 语言切换器配置存在
    Tool: Bash
    Preconditions: config.ts 已创建
    Steps:
      1. grep "locales" docs/.vitepress/config.ts
      2. grep "zh" docs/.vitepress/config.ts
      3. grep "en" docs/.vitepress/config.ts
    Expected Result: config.ts 中包含 locales 配置，含 zh 和 en 两个语言
    Failure Indicators: 缺少 locales 或任一语言配置
    Evidence: .omo/evidence/task-1-i18n-config.txt

  Scenario: 目录结构完整
    Tool: Bash
    Preconditions: 脚手架已创建
    Steps:
      1. ls docs/zh/guide/ | wc -l  (应为 11 个章节 .md)
      2. ls docs/en/guide/ | wc -l  (应为 11 个章节 .md)
      3. ls docs/zh/index.md && ls docs/en/index.md
    Expected Result: zh/ 和 en/ 各有 11 个章节 + 1 个首页 = 12 个 .md 文件
    Failure Indicators: 文件数量不匹配或缺少文件
    Evidence: .omo/evidence/task-1-directory-structure.txt
  ```

  **Commit**: YES
  - Message: `docs(gitbook): scaffold VitePress project with i18n config`
  - Files: `docs/`
  - Pre-commit: `cd docs && npm run docs:build`

- [ ] 2. 双语函数 docstring 模板 + 编写指南

  **What to do**:
  - 创建 `.omo/drafts/docstring-guide.md` 编写指南，定义：
    - Python 函数 docstring 模板（Google 风格，中英双语）
    - 类方法 docstring 模板
    - 类级 docstring 模板
    - 模块级 docstring 说明（已有头注释保持不动，函数 docstring 是新增）
  - 模板内容：
    ```python
    def function_name(param1: type, param2: type) -> return_type:
        """
        [中文简述] / [English brief description]

        [中文详细描述函数功能] / [Detailed description of function purpose]

        Args / 参数:
            param1 (type): [中文描述] / [English description]
            param2 (type): [中文描述] / [English description]

        Returns / 返回:
            type: [中文描述] / [English description]

        Raises / 异常:
            ValueError: [中文描述] / [English description]

        Calls / 调用:
            - other_function(): [中文描述] / [English description]

        Example / 示例:
            >>> result = function_name(1, "test")
        """
    ```
  - 简化版（对于短函数）：
    ```python
    def simple_func(x: int) -> bool:
        """[中文简述] / [English brief]. Args: x (int): 输入值 / input. Returns: bool: 结果 / result."""
    ```
  - 在指南中说明：不修改已有文件级头注释，只在函数定义处添加 docstring
  - 说明中英文在同一 docstring 内并列（不使用独立页面）

  **Must NOT do**:
  - 不要修改任何 Python 文件（此任务仅创建指南）
  - 不要使用 NumPy 风格 docstring（使用 Google 风格）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 编写指南文档，无代码修改
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1)
  - **Blocks**: T3-T8（所有代码注释任务参考此模板）
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `model/mrmodn.py:2-47` - 现有文件级头注释格式（参考风格，但不要修改）
  - `model/mrmodn.py:65-67` - 现有类级 docstring（简短英文，需要扩展为双语）
  - `model/mrmodn.py:107` - 现有 forward 方法（需要添加 docstring）

  **External References**:
  - Google Python Style Guide docstring: https://google.github.io/styleguide/pyguide.html#383-functions-and-methods

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 指南文件存在且内容完整
    Tool: Bash
    Preconditions: 指南已创建
    Steps:
      1. ls .omo/drafts/docstring-guide.md
      2. grep "Args" .omo/drafts/docstring-guide.md
      3. grep "Returns" .omo/drafts/docstring-guide.md
      4. grep "Calls" .omo/drafts/docstring-guide.md
    Expected Result: 指南文件存在，包含 Args、Returns、Calls 模板说明
    Failure Indicators: 文件不存在或缺少关键模板字段
    Evidence: .omo/evidence/task-2-guide-exists.txt
  ```

  **Commit**: NO（仅创建草稿指南，不提交到主分支）

- [ ] 3. model/ 函数 docstrings (10 files)

  **What to do**:
  - 阅读 `.omo/drafts/docstring-guide.md` 获取 docstring 模板
  - 为 `model/` 目录下所有 10 个 Python 文件的每个函数/类/方法添加双语 docstring
  - 文件清单：
    - `model/mrmodn.py` - 主模型（ParallelCNNBlock, GCNBlock, ClassQueryHead, ClassQueryHeadPooling, HierarchicalClassQueryHeadPooling, RNA_ClassQuery_Model）
    - `model/mrmodn_collect_atten.py` - 注意力收集变体
    - `model/mrmodn_multirm.py` - MultIRM 51nt 变体
    - `model/multirm.py` - 多任务多修饰变体
    - `model/modx.py` - 修饰类型消融变体
    - `model/modx_collect_atten.py` - ModX 注意力收集
    - `model/multirm_collect_atten.py` - MultiRM 注意力收集
    - `model/evormd_human.py` - EvoRMD 集成
    - `model/abla_model.py` - 消融实验模型
    - `model/__init__.py` - 模块初始化
  - 每个函数/方法的 docstring 包含：用途、参数、返回值、调用关系
  - 保留已有的文件级头注释（不修改）
  - 对已有简短英文 docstring 的类/方法，扩展为双语格式

  **Must NOT do**:
  - 不修改任何功能性代码
  - 不修改已有文件级头注释
  - 不删除已有注释（只添加或扩展）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要理解深度学习模型架构，分析每个函数的输入输出和调用关系
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T4, T5, T6, T7)
  - **Blocks**: T9-T14（GitBook 章节引用代码时需要 docstring）
  - **Blocked By**: T2（需要 docstring 模板）

  **References**:

  **Pattern References**:
  - `.omo/drafts/docstring-guide.md` - docstring 模板和编写指南
  - `model/mrmodn.py:2-47` - 现有文件级头注释（不修改）
  - `model/mrmodn.py:64-67` - 现有 ParallelCNNBlock 类 docstring（需扩展为双语）
  - `model/mrmodn.py:107-136` - forward 方法（需添加 docstring）

  **API/Type References**:
  - `utils/common.py:GROUP_TO_CLASS_INDICES` - 被 HierarchicalClassQueryHeadPooling 使用的常量
  - `torch_geometric.nn.GCNConv` - GCN 卷积层

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Python 语法检查通过
    Tool: Bash
    Preconditions: 所有 10 个 model/ 文件已添加 docstring
    Steps:
      1. python -m py_compile model/mrmodn.py
      2. python -m py_compile model/mrmodn_collect_atten.py
      3. python -m py_compile model/mrmodn_multirm.py
      4. python -m py_compile model/multirm.py
      5. python -m py_compile model/modx.py
      6. python -m py_compile model/modx_collect_atten.py
      7. python -m py_compile model/multirm_collect_atten.py
      8. python -m py_compile model/evormd_human.py
      9. python -m py_compile model/abla_model.py
      10. python -m py_compile model/__init__.py
    Expected Result: 全部通过，退出码 0
    Failure Indicators: 任何 SyntaxError
    Evidence: .omo/evidence/task-3-model-syntax.txt

  Scenario: Docstring 覆盖率
    Tool: Bash
    Preconditions: docstring 已添加
    Steps:
      1. python3 -c "
         import ast, sys
         for f in ['model/mrmodn.py','model/multirm.py','model/modx.py','model/evormd_human.py','model/abla_model.py']:
             tree = ast.parse(open(f).read())
             funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
             with_doc = [n for n in funcs if ast.get_docstring(n)]
             print(f'{f}: {len(with_doc)}/{len(funcs)} functions have docstrings')
         "
    Expected Result: 所有核心文件的函数 docstring 覆盖率 ≥ 90%
    Failure Indicators: 覆盖率 < 90%
    Evidence: .omo/evidence/task-3-model-coverage.txt
  ```

  **Commit**: YES
  - Message: `docs(model): add bilingual function-level docstrings`
  - Files: `model/*.py`
  - Pre-commit: `python -m py_compile model/*.py`

- [ ] 4. dataset/ 函数 docstrings (9 files)

  **What to do**:
  - 为 `dataset/` 目录下所有 9 个 Python 文件的每个函数/类/方法添加双语 docstring
  - 文件清单：
    - `dataset/human.py` - Human 标准 12 修饰数据集（Mer100Dataset）
    - `dataset/plant.py` - Plant 数据集（PlantDataset）
    - `dataset/ac4c.py` - ac4C 数据集（AC4CDataset）
    - `dataset/multirm.py` - 多 RM 数据集
    - `dataset/gen3.py` - 第 3 代数据集
    - `dataset/gen3_zero.py` - 零样本 gen3 数据集
    - `dataset/human_motif.py` - Human motif 数据集
    - `dataset/human_with_seq.py` - Human 序列数据集
    - `dataset/plant_single.py` - 单 plant 数据集
  - 每个数据集类的 docstring 需说明：数据来源、标签映射、__getitem__ 返回的 Data 对象结构
  - 保留已有的文件级头注释和标签映射注释

  **Must NOT do**:
  - 不修改数据加载逻辑
  - 不修改标签映射
  - 不修改文件级头注释

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要理解 PyTorch Geometric 数据集结构和 RNA 数据格式
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T3, T5, T6, T7)
  - **Blocks**: T10-T11（GitBook 数据集和训练章节引用）
  - **Blocked By**: T2

  **References**:

  **Pattern References**:
  - `.omo/drafts/docstring-guide.md` - docstring 模板
  - `dataset/human.py:1-48` - 现有文件级头注释和标签映射（不修改）

  **API/Type References**:
  - `torch_geometric.data.Data` - PyG Data 对象格式
  - `npy/` 数据目录结构

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Python 语法检查 + docstring 覆盖率
    Tool: Bash
    Preconditions: 所有 9 个 dataset/ 文件已修改
    Steps:
      1. for f in dataset/*.py; do python -m py_compile "$f"; done
      2. python3 -c "
         import ast
         for f in ['dataset/human.py','dataset/plant.py','dataset/ac4c.py','dataset/multirm.py','dataset/gen3.py']:
             tree = ast.parse(open(f).read())
             funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
             with_doc = [n for n in funcs if ast.get_docstring(n)]
             print(f'{f}: {len(with_doc)}/{len(funcs)}')
         "
    Expected Result: 语法通过，覆盖率 ≥ 90%
    Failure Indicators: SyntaxError 或覆盖率 < 90%
    Evidence: .omo/evidence/task-4-dataset-syntax.txt
  ```

  **Commit**: YES
  - Message: `docs(dataset): add bilingual function-level docstrings`
  - Files: `dataset/*.py`

- [ ] 5. utils/ 函数 docstrings (20 files)

  **What to do**:
  - 为 `utils/` 目录下所有 20 个 Python 文件的每个函数/类/方法添加双语 docstring
  - 文件清单：
    - `utils/common.py` - 公共常量（NUCLEOTIDE_MAP, k-mer, GROUP_TO_CLASS_INDICES）和评估函数
    - `utils/metrics.py` - ACC / F1 / MCC 等指标
    - `utils/logging.py` - 日志工具
    - `utils/few_shot.py` - 少样本核心算法
    - `utils/fewshot_analysis_*.py` (8 files) - 少样本分析工具集
    - `utils/fewshot_export_helpers.py` - 少样本导出工具
    - `utils/sliding_window_utils.py` - 滑窗工具
    - `utils/train_gen3.py` - Gen3 训练工具
    - `utils/test_gen3_analyse.py` - Gen3 测试分析
    - `utils/rna_visualization.py` - RNA 可视化工具
    - `utils/audit_12loc_structure.py` - 12 位点结构审计
    - `utils/check_m6a_data_integrity.py` - m6A 数据完整性检查
    - `utils/Zero_structures.py` - 零样本结构
    - `utils/__init__.py` - 模块初始化

  **Must NOT do**:
  - 不修改任何工具函数逻辑
  - 不修改常量定义
  - 不修改文件级头注释

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 20 个文件，工作量大，需要理解各工具函数的用途
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T3, T4, T6, T7)
  - **Blocks**: T11-T13（GitBook 章节引用 utils）
  - **Blocked By**: T2

  **References**:

  **Pattern References**:
  - `.omo/drafts/docstring-guide.md` - docstring 模板
  - `utils/common.py` - 核心常量文件

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Python 语法检查 + docstring 覆盖率
    Tool: Bash
    Preconditions: 所有 20 个 utils/ 文件已修改
    Steps:
      1. for f in utils/*.py; do python -m py_compile "$f"; done
      2. python3 -c "
         import ast
         for f in ['utils/common.py','utils/metrics.py','utils/few_shot.py','utils/sliding_window_utils.py']:
             tree = ast.parse(open(f).read())
             funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
             with_doc = [n for n in funcs if ast.get_docstring(n)]
             print(f'{f}: {len(with_doc)}/{len(funcs)}')
         "
    Expected Result: 语法通过，覆盖率 ≥ 90%
    Failure Indicators: SyntaxError 或覆盖率 < 90%
    Evidence: .omo/evidence/task-5-utils-syntax.txt
  ```

  **Commit**: YES
  - Message: `docs(utils): add bilingual function-level docstrings`
  - Files: `utils/*.py`

- [ ] 6. 根目录 train/inference/collect 脚本函数 docstrings (16 files)

  **What to do**:
  - 为根目录下所有训练、推理、数据收集脚本的每个函数添加双语 docstring
  - 文件清单（16 个）：
    - `train_human_mrmodn.py` - Human 主训练
    - `train_plant_mrmodn.py` - Plant 训练
    - `train_multirm_mrmodn.py` - 多 RM 训练
    - `train_human_multirm.py` - Human MultiRM 训练
    - `train_human_modx.py` - Human ModX 训练
    - `train_human_evormd.py` - Human EvoRMD 训练
    - `inference_human_mrmodn_full.py` - Human 全长推理
    - `inference_human_modx_segmented.py` - ModX 分段推理
    - `inference_human_evormd_segmented.py` - EvoRMD 分段推理
    - `inference_multirm_multirm_segmented.py` - 多 RM 分段推理
    - `collect_human_mrmodn.py` - Human 特征收集
    - `collect_atten_human_mrmodn.py` - Human 注意力收集
    - `collect_atten_human_modx.py` - ModX 注意力收集
    - `collect_atten_multirm_multirm.py` - MultiRM 注意力收集

  **Must NOT do**:
  - 不修改训练/推理逻辑
  - 不修改超参数
  - 不修改文件级头注释

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 训练脚本结构复杂（main 函数、数据加载、训练循环、评估），需要深入理解
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T3, T4, T5, T7)
  - **Blocks**: T11-T12（GitBook 训练/推理章节）
  - **Blocked By**: T2

  **References**:

  **Pattern References**:
  - `.omo/drafts/docstring-guide.md` - docstring 模板
  - `train_human_mrmodn.py:1-45` - 现有文件级头注释（不修改）

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Python 语法检查
    Tool: Bash
    Preconditions: 所有 16 个文件已修改
    Steps:
      1. for f in train_*.py inference_*.py collect_*.py; do python -m py_compile "$f"; done
    Expected Result: 全部通过
    Failure Indicators: 任何 SyntaxError
    Evidence: .omo/evidence/task-6-scripts-syntax.txt
  ```

  **Commit**: YES
  - Message: `docs(scripts): add bilingual function-level docstrings to train/inference/collect scripts`
  - Files: `train_*.py`, `inference_*.py`, `collect_*.py`

- [ ] 7. 根目录 fewshot/zeroshot 脚本函数 docstrings (5 files)

  **What to do**:
  - 为根目录下所有少样本/零样本分析脚本的每个函数添加双语 docstring
  - 文件清单（5 个）：
    - `fewshot_ac4c_mrmodn_balance.py` - ac4C 平衡少样本
    - `fewshot_ac4c_mrmodn_unbalan.py` - ac4C 非平衡少样本
    - `fewshot_plant_mrmodn_3way.py` - Plant 三向独立少样本
    - `zeroshot_human_mrmodn_analysis.py` - 零样本综合分析
    - `zeroshot_human_mrmodn_extract.py` - 零样本特征抽取

  **Must NOT do**:
  - 不修改分析逻辑
  - 不修改文件级头注释

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要理解少样本/零样本学习在 RNA 修饰检测中的应用
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T3, T4, T5, T6)
  - **Blocks**: T13（GitBook 少零样本章节）
  - **Blocked By**: T2

  **References**:

  **Pattern References**:
  - `.omo/drafts/docstring-guide.md` - docstring 模板
  - `utils/few_shot.py` - 少样本核心算法（T5 会为其添加 docstring）

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Python 语法检查
    Tool: Bash
    Preconditions: 所有 5 个文件已修改
    Steps:
      1. for f in fewshot_*.py zeroshot_*.py; do python -m py_compile "$f"; done
    Expected Result: 全部通过
    Failure Indicators: 任何 SyntaxError
    Evidence: .omo/evidence/task-7-fewshot-syntax.txt
  ```

  **Commit**: YES
  - Message: `docs(fewshot): add bilingual function-level docstrings to few-shot/zero-shot scripts`
  - Files: `fewshot_*.py`, `zeroshot_*.py`

- [ ] 8. ablation/ + visualization/ + tests/ 函数 docstrings (29 files)

  **What to do**:
  - 为 `ablation/`、`visualization/`、`tests/` 目录下所有 Python 文件添加函数级双语 docstring
  - 文件清单：
    - `ablation/` (5 files): ablation_3x3_human_mrmodn.py, ablation_3x3_v2_human_mrmodn.py, ablation_mohe_human_mrmodn.py, cal_flops_human_mrmodn.py, cal_stats_human_mrmodn.py
    - `visualization/` (14 files): human/mrmodn/*.py (8 files), tools/*.py (2 files), human/modx/*.py (1 file), human/mrmodn/run_*.py (2 files)
    - `tests/` (10 files): conftest.py, test_infrastructure.py, test_model_import.py, test_dataset_import.py, test_config.py, test_naming_convention.py, test_gen3.py, test_multirm_4class.py, test_multirm_oversampling.py, __init__.py

  **Must NOT do**:
  - 不修改消融/可视化/测试逻辑
  - 不修改文件级头注释

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 29 个文件，工作量最大，需要理解消融实验和可视化流程
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T9, T10, T11)
  - **Blocks**: T12-T14（GitBook 消融/可视化/测试章节）
  - **Blocked By**: T2

  **References**:

  **Pattern References**:
  - `.omo/drafts/docstring-guide.md` - docstring 模板
  - `ablation/README.md` - 消融实验说明
  - `visualization/README.md` - 可视化说明

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Python 语法检查
    Tool: Bash
    Preconditions: 所有文件已修改
    Steps:
      1. for f in ablation/*.py; do python -m py_compile "$f"; done
      2. for f in $(find visualization/ -name "*.py"); do python -m py_compile "$f"; done
      3. for f in tests/*.py; do python -m py_compile "$f"; done
    Expected Result: 全部通过
    Failure Indicators: 任何 SyntaxError
    Evidence: .omo/evidence/task-8-remaining-syntax.txt

  Scenario: pytest 回归测试
    Tool: Bash
    Preconditions: 所有文件已修改
    Steps:
      1. pytest tests/ -v
    Expected Result: 所有测试通过
    Failure Indicators: 任何测试失败
    Evidence: .omo/evidence/task-8-pytest-regression.txt
  ```

  **Commit**: YES
  - Message: `docs(ablation/viz/tests): add bilingual function-level docstrings`
  - Files: `ablation/*.py`, `visualization/**/*.py`, `tests/*.py`

- [ ] 9. 章 Ch1 概述 + Ch2 算法架构 [zh + en]

  **What to do**:
  - 创建 `docs/zh/guide/overview.md`（中文）和 `docs/en/guide/overview.md`（English）
  - 创建 `docs/zh/guide/architecture.md`（中文）和 `docs/en/guide/architecture.md`（English）

  **Ch1 概述** 内容：
  - 项目背景：RNA 修饰在转录后调控中的作用，170+ 种 RNA 修饰
  - 研究动机：传统实验成本高、深度学习可高通量预测
  - mRModN 定位：Mixture of Hierarchical Experts 框架
  - 核心创新：M2D、MoHE、ABS、A2P
  - 支持的 12 种修饰类型列表
  - 与其他项目的关系（Web 后端、微信小程序、Web 前端）

  **Ch2 算法架构** 内容：
  - 整体架构图（文字描述 + ASCII art 或 Mermaid 图）
  - 模块详解：
    1. **Embedding 模块**: one-hot 编码 + RNA 二级结构编码
    2. **M2D (Multi-view Motif Discovery)**: 多尺度 CNN (k=1,3,5,7) + 位置编码 + 结构拓扑
    3. **MoHE (Mixture of Hierarchical Experts)**: HQR (层级查询路由) + Experts Pool + HCA (层级交叉注意力)
    4. **A2P (Multi-Anchor Attention Pooling)**: 多锚点注意力池化 + KL 散度监督
    5. **ABS (Adaptive Balanced Sampler)**: 自适应平衡采样
  - 数据流：one-hot → CNN → GCN → MoHE → A2P → 12 类 logits
  - 关键公式（如有）：注意力计算、损失函数

  **Must NOT do**:
  - 不要在文档中直接嵌入源代码（可引用关键函数名）
  - 不要添加交互式代码执行器

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 需要技术写作能力，将论文内容转化为教程文档
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T8, T10, T11)
  - **Blocks**: T15-T16（验证阶段）
  - **Blocked By**: T1（需要 docs/ 目录结构）

  **References**:

  **Pattern References**:
  - `README.md:1-16` - 项目背景描述（双语）
  - `model/mrmodn.py:2-47` - 模型架构说明（M2D、GCN、ClassQueryHead）

  **External References**:
  - 论文内容（已提取）：mRModN 架构描述，M2D、MoHE、ABS、A2P 四大模块
  - VitePress Markdown 扩展: https://vitepress.dev/guide/using-vue

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Ch1 和 Ch2 文件存在且内容完整
    Tool: Bash
    Preconditions: 章节文件已创建
    Steps:
      1. ls docs/zh/guide/overview.md && ls docs/en/guide/overview.md
      2. ls docs/zh/guide/architecture.md && ls docs/en/guide/architecture.md
      3. wc -l docs/zh/guide/overview.md  (应 > 30 行)
      4. wc -l docs/zh/guide/architecture.md  (应 > 80 行)
      5. grep "M2D" docs/zh/guide/architecture.md
      6. grep "MoHE" docs/zh/guide/architecture.md
      7. grep "A2P" docs/zh/guide/architecture.md
    Expected Result: 4 个文件存在，内容充实，涵盖所有核心模块
    Failure Indicators: 文件不存在或内容过少
    Evidence: .omo/evidence/task-9-ch1-ch2.txt

  Scenario: 中英文内容对应
    Tool: Bash
    Preconditions: 中英文文件都已创建
    Steps:
      1. diff <(grep "^##" docs/zh/guide/overview.md) <(grep "^##" docs/en/guide/overview.md)
      2. diff <(grep "^##" docs/zh/guide/architecture.md) <(grep "^##" docs/en/guide/architecture.md)
    Expected Result: 中英文章节标题结构一致
    Failure Indicators: 章节标题数量或顺序不匹配
    Evidence: .omo/evidence/task-9-bilingual-match.txt
  ```

  **Commit**: YES
  - Message: `docs(ch1-ch2): add overview and algorithm architecture chapters`
  - Files: `docs/zh/guide/overview.md`, `docs/en/guide/overview.md`, `docs/zh/guide/architecture.md`, `docs/en/guide/architecture.md`

- [ ] 10. 章 Ch3 快速开始 + Ch4 模型详解 [zh + en]

  **What to do**:
  - 创建 4 个 Markdown 文件（zh + en × 2 章）

  **Ch3 快速开始** 内容：
  - 环境要求（Python 3.8+, PyTorch 1.10+, PyG, numpy, pandas, scikit-learn, matplotlib, R 4.0+）
  - 安装步骤（pip install）
  - 数据准备（npy/ 软链接）
  - 最小可运行示例（训练 Human mRModN 3 行代码）
  - 常见问题（GPU OOM、数据路径错误）

  **Ch4 模型详解** 内容：
  - `model/mrmodn.py` 详解：
    - ParallelCNNBlock：多尺度并行 1D 卷积（k=1,3,5,7），输入输出
    - GCNBlock：残差图卷积（3 层 GCNConv + LayerNorm），输入输出
    - ClassQueryHead：TransformerDecoder 交叉注意力头
    - ClassQueryHeadPooling：简化注意力池化头
    - HierarchicalClassQueryHeadPooling：4 组到 12 类层级查询
    - RNA_ClassQuery_Model：端到端模型整合
  - 模型变体对比表（mrmodn vs multirm vs modx vs evormd）
  - 关键参数说明（cnn_hidden_dim, gcn_hidden_dim, num_classes 等）

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 技术教程写作，需要将代码转化为可读文档
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T8, T9, T11)
  - **Blocks**: T15-T16
  - **Blocked By**: T1, T3（需要 VitePress 结构和 model/ docstrings）

  **References**:

  **Pattern References**:
  - `README.md:120-167` - 现有 Getting Started 内容
  - `README.md:218-258` - 现有 Key Files 说明
  - `model/mrmodn.py` - 主模型文件（T3 会添加 docstring）

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Ch3 和 Ch4 文件存在且内容完整
    Tool: Bash
    Preconditions: 章节文件已创建
    Steps:
      1. ls docs/zh/guide/quickstart.md && ls docs/en/guide/quickstart.md
      2. ls docs/zh/guide/model.md && ls docs/en/guide/model.md
      3. grep "pip install" docs/zh/guide/quickstart.md
      4. grep "ParallelCNNBlock" docs/zh/guide/model.md
      5. grep "GCNBlock" docs/zh/guide/model.md
      6. grep "ClassQueryHead" docs/zh/guide/model.md
    Expected Result: 文件存在，包含安装步骤和模型组件说明
    Failure Indicators: 文件缺失或关键内容缺失
    Evidence: .omo/evidence/task-10-ch3-ch4.txt
  ```

  **Commit**: YES
  - Message: `docs(ch3-ch4): add quickstart and model details chapters`
  - Files: `docs/zh/guide/quickstart.md`, `docs/en/guide/quickstart.md`, `docs/zh/guide/model.md`, `docs/en/guide/model.md`

- [ ] 11. 章 Ch5 数据集 + Ch6 训练流程 [zh + en]

  **What to do**:
  - 创建 4 个 Markdown 文件（zh + en × 2 章）

  **Ch5 数据集** 内容：
  - HRMD-m 数据集（Human 12 修饰）：来源、预处理、CD-HIT 去冗余
  - PRMD-m 数据集（Plant）
  - 第 3 代直接 RNA 测序数据集
  - ac4C 数据集
  - 多 RM 数据集
  - 数据格式说明（npy 文件结构、标签格式）
  - 各数据集加载器对比表

  **Ch6 训练流程** 内容：
  - 训练脚本一览表（6 个脚本的用途、数据集、模型）
  - `train_human_mrmodn.py` 详解：配置加载、数据划分、模型初始化、训练循环、评估
  - 命令行参数说明
  - 训练产物（checkpoints、logs、TensorBoard）
  - 自定义训练（如何修改脚本适配新数据集）

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 技术教程写作
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T8, T9, T10)
  - **Blocks**: T15-T16
  - **Blocked By**: T1, T4, T5, T6

  **References**:

  **Pattern References**:
  - `README.md:139-165` - 数据准备和训练示例
  - `train_human_mrmodn.py` - 主训练脚本
  - `dataset/human.py` - 主数据集类

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Ch5 和 Ch6 文件存在且内容完整
    Tool: Bash
    Preconditions: 章节文件已创建
    Steps:
      1. ls docs/zh/guide/dataset.md && ls docs/en/guide/dataset.md
      2. ls docs/zh/guide/training.md && ls docs/en/guide/training.md
      3. grep "HRMD" docs/zh/guide/dataset.md
      4. grep "train_human_mrmodn" docs/zh/guide/training.md
    Expected Result: 文件存在，包含数据集说明和训练流程
    Failure Indicators: 文件缺失或关键内容缺失
    Evidence: .omo/evidence/task-11-ch5-ch6.txt
  ```

  **Commit**: YES
  - Message: `docs(ch5-ch6): add dataset and training pipeline chapters`
  - Files: `docs/zh/guide/dataset.md`, `docs/en/guide/dataset.md`, `docs/zh/guide/training.md`, `docs/en/guide/training.md`

- [ ] 12. 章 Ch7 推理流程 + Ch8 可视化 [zh + en]

  **What to do**:
  - 创建 4 个 Markdown 文件（zh + en × 2 章）

  **Ch7 推理流程** 内容：
  - 推理脚本一览表（4 个脚本）
  - 全长推理 vs 分段推理的区别
  - `inference_human_mrmodn_full.py` 详解：模型加载、数据输入、预测输出
  - 滑窗推理原理（sliding_window_utils.py）
  - 推理输出格式（logits、attention weights）
  - 零样本推理流程

  **Ch8 可视化** 内容：
  - 可视化目录结构说明
  - 注意力对比可视化（attention_comparison.py）
  - 空间 motif 可视化（spatial_motif.py）
  - UMAP 降维可视化（prepare_umap_data.py）
  - R 脚本可视化（plot_zero_fewshot_analysis.R）
  - Jupyter notebook 交互式分析

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with T13, T14)
  - **Blocks**: T15-T16
  - **Blocked By**: T1, T6, T8

  **References**:

  **Pattern References**:
  - `README.md:169-179` - 推理示例
  - `README.md:181-196` - 可视化示例
  - `visualization/README.md` - 可视化目录说明

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Ch7 和 Ch8 文件存在且内容完整
    Tool: Bash
    Steps:
      1. ls docs/zh/guide/inference.md && ls docs/en/guide/inference.md
      2. ls docs/zh/guide/visualization.md && ls docs/en/guide/visualization.md
      3. grep "sliding" docs/zh/guide/inference.md
      4. grep "UMAP" docs/zh/guide/visualization.md
    Expected Result: 文件存在，包含推理和可视化说明
    Evidence: .omo/evidence/task-12-ch7-ch8.txt
  ```

  **Commit**: YES
  - Message: `docs(ch7-ch8): add inference and visualization chapters`
  - Files: `docs/zh/guide/inference.md`, `docs/en/guide/inference.md`, `docs/zh/guide/visualization.md`, `docs/en/guide/visualization.md`

- [ ] 13. 章 Ch9 消融实验 + Ch10 少零样本分析 [zh + en]

  **What to do**:
  - 创建 4 个 Markdown 文件（zh + en × 2 章）

  **Ch9 消融实验** 内容：
  - 消融实验目的：验证各模块贡献
  - 3×3 矩阵消融（ablation_3x3_human_mrmodn.py）
  - MoHE 消融（ablation_mohe_human_mrmodn.py）
  - FLOPs 计算（cal_flops_human_mrmodn.py）
  - 消融结果表格

  **Ch10 少零样本分析** 内容：
  - 少样本学习在 RNA 修饰检测中的意义
  - ac4C 平衡/非平衡少样本实验
  - Plant 三向独立少样本实验
  - 零样本迁移能力分析
  - 特征抽取与可视化

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with T12, T14)
  - **Blocks**: T15-T16
  - **Blocked By**: T1, T7, T8

  **References**:

  **Pattern References**:
  - `ablation/README.md` - 消融实验说明
  - `fewshot_ac4c_mrmodn_balance.py` - 少样本脚本

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Ch9 和 Ch10 文件存在
    Tool: Bash
    Steps:
      1. ls docs/zh/guide/ablation.md && ls docs/en/guide/ablation.md
      2. ls docs/zh/guide/fewshot.md && ls docs/en/guide/fewshot.md
      3. grep "MoHE" docs/zh/guide/ablation.md
      4. grep "zero.shot" docs/zh/guide/fewshot.md || grep "零样本" docs/zh/guide/fewshot.md
    Expected Result: 文件存在，内容完整
    Evidence: .omo/evidence/task-13-ch9-ch10.txt
  ```

  **Commit**: YES
  - Message: `docs(ch9-ch10): add ablation and few-shot analysis chapters`
  - Files: `docs/zh/guide/ablation.md`, `docs/en/guide/ablation.md`, `docs/zh/guide/fewshot.md`, `docs/en/guide/fewshot.md`

- [ ] 14. 章 Ch11 附录 + 首页美化 [zh + en]

  **What to do**:
  - 创建 2 个 Markdown 文件（zh + en × 1 章）
  - 美化 `docs/zh/index.md` 和 `docs/en/index.md` 首页

  **Ch11 附录** 内容：
  - 项目目录结构完整树形图
  - 命名规范说明（`<task>_<dataset>_<model>[_<suffix>].py`）
  - 常见问题 FAQ
  - 引用格式（论文 BibTeX）
  - 致谢
  - Web 部署说明（https://cmb.bnu.edu.cn/rgcnformer/）

  **首页美化** 内容：
  - Hero section：项目名称、简短描述、快速开始按钮
  - Feature cards：核心功能（多标签分类、少样本学习、零样本迁移、可视化分析）
  - 统计数据（12 种修饰、3 个数据集、0.86 Top-1 recall）

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with T12, T13)
  - **Blocks**: T15-T16
  - **Blocked By**: T1

  **References**:

  **Pattern References**:
  - `README.md:36-118` - 目录结构
  - `README.md:259-274` - 与其他项目关系
  - VitePress 首页配置: https://vitepress.dev/guide/landing-page

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Ch11 和首页内容完整
    Tool: Bash
    Steps:
      1. ls docs/zh/guide/appendix.md && ls docs/en/guide/appendix.md
      2. grep "BibTeX\|@" docs/zh/guide/appendix.md
      3. grep "hero" docs/zh/index.md
      4. grep "features" docs/zh/index.md
    Expected Result: 文件存在，附录含引用格式，首页含 hero 和 features
    Evidence: .omo/evidence/task-14-ch11-homepage.txt
  ```

  **Commit**: YES
  - Message: `docs(ch11): add appendix and beautify homepage`
  - Files: `docs/zh/guide/appendix.md`, `docs/en/guide/appendix.md`, `docs/zh/index.md`, `docs/en/index.md`

- [ ] 15. Python 语法编译检查 (all 87 files)

  **What to do**:
  - 对项目中所有 87 个 Python 文件运行 `python -m py_compile` 语法检查
  - 覆盖范围：model/ (10), dataset/ (9), utils/ (20), 根目录 train/inference/fewshot/zeroshot/collect (20), ablation/ (5), visualization/ (14), tests/ (10)
  - 排除 .omo/ 目录
  - 记录任何语法错误并修复

  **Must NOT do**:
  - 不修改功能性代码（只修复 docstring 引入的语法错误）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的批量语法检查
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with T16)
  - **Blocks**: F1-F4
  - **Blocked By**: T3-T8（所有代码注释任务）

  **References**:

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 全量语法检查
    Tool: Bash
    Preconditions: 所有代码注释任务已完成
    Steps:
      1. find . -name "*.py" -not -path "./.omo/*" -exec python -m py_compile {} \; 2>&1 | grep -i error
    Expected Result: 无错误输出，退出码 0
    Failure Indicators: 任何 SyntaxError 或 CompilationError
    Evidence: .omo/evidence/task-15-syntax-check.txt
  ```

  **Commit**: NO（如果需要修复，合并到对应 Wave 2/3 的 commit）

- [ ] 16. VitePress 构建验证 + 链接检查 + 双语审计

  **What to do**:
  - 运行 `cd docs && npm run docs:build` 验证构建成功
  - 检查所有内部链接是否有效（相对路径 .md 文件存在）
  - 双语审计：
    - zh/ 和 en/ 文件数量一致（各 12 个 .md）
    - 章节标题结构一致
    - 语言切换器配置正确
  - 修复任何发现的问题

  **Must NOT do**:
  - 不修改项目逻辑

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 构建验证和审计
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with T15)
  - **Blocks**: F1-F4
  - **Blocked By**: T9-T14（所有 GitBook 章节任务）

  **References**:

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: VitePress 构建成功
    Tool: Bash
    Preconditions: 所有章节已完成
    Steps:
      1. cd docs && npm run docs:build
      2. echo $?
    Expected Result: 构建成功，退出码 0
    Failure Indicators: 构建错误
    Evidence: .omo/evidence/task-16-vitepress-build.txt

  Scenario: 双语完整性
    Tool: Bash
    Steps:
      1. echo "zh files:" && ls docs/zh/guide/*.md | wc -l
      2. echo "en files:" && ls docs/en/guide/*.md | wc -l
      3. diff <(ls docs/zh/guide/ | sort) <(ls docs/en/guide/ | sort)
    Expected Result: zh/ 和 en/ 各有 11 个章节文件，文件名完全一致
    Failure Indicators: 文件数量不匹配或文件名不一致
    Evidence: .omo/evidence/task-16-bilingual-audit.txt
  ```

  **Commit**: YES
  - Message: `docs: verify VitePress build and bilingual completeness`
  - Files: 修复文件（如有）

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns. Check evidence files exist in .omo/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `python -m py_compile` on all modified Python files. Check docstring format consistency. Verify no functional code was modified. Run `pytest tests/ -v`.
  Output: `Syntax [PASS/FAIL] | Pytest [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **VitePress Build + 功能验证** — `unspecified-high` (+ `playwright` skill if available)
  Run `npm run docs:build`. Verify all 22 chapter pages exist and render. Test language switcher. Check all internal links resolve. Verify Chinese/English content matches.
  Output: `Build [PASS/FAIL] | Pages [N/N] | Links [N valid/N broken] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Detect cross-task contamination.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1**: `docs(gitbook): scaffold VitePress project with i18n config`
- **Wave 2**: `docs(model/dataset/utils): add bilingual function-level docstrings`
- **Wave 3**: `docs(gitbook): add core chapters (Ch1-Ch6) in zh + en`
- **Wave 4**: `docs(gitbook): add remaining chapters (Ch7-Ch11) in zh + en`
- **Wave 5**: `docs: verify build and bilingual completeness`

---

## Success Criteria

### Verification Commands
```bash
# Python syntax check all modified files
find . -name "*.py" -not -path "./.omo/*" -exec python -m py_compile {} \;

# VitePress build
cd docs && npm run docs:build

# Pytest regression
pytest tests/ -v

# Count docstrings (should be ~87 files with function docstrings)
grep -rl '"""' --include="*.py" model/ dataset/ utils/ train_*.py inference_*.py fewshot_*.py zeroshot_*.py collect_*.py ablation/ visualization/ tests/ | wc -l
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] VitePress 构建成功
- [ ] 语言切换器工作正常
- [ ] 11 章节 × 2 语言 = 22 页面完整
- [ ] pytest 回归测试通过
