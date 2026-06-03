# Codebase Documentation Project

## TL;DR

> **Quick Summary**: Add comprehensive bilingual (Chinese + English) header comments to all 93 code files (Python, R, Jupyter notebooks) in the RGCNFormer RNA modification classification project.
>
> **Deliverables**:
> - File-level documentation for 93 code files
> - Bilingual comments (Chinese + English)
> - Comprehensive documentation including purpose, I/O, data flow, related files, and usage examples
>
> **Estimated Effort**: Large (93 files across 13 categories)
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: Model files → Dataset files → Training scripts → Final verification

---

## Context

### Original Request
用户期望为当前项目下面每一个代码文件（R, py, ipynb）在文件开头加上注释，介绍该文件的作用、文件的输入输出、文件的模块和数据流、与该文件相关的文件（调用和被调用）。

### Interview Summary
**Key Discussions**:
- **Language**: Bilingual (Chinese + English) for maximum accessibility
- **Depth**: Comprehensive documentation including purpose, I/O, data flow, related files, and usage examples
- **Scope**: File-level only (headers at top of each file, not function-level)

**Research Findings**:
- Project is an RNA modification classification system using RGCNFormer
- 93 code files identified across 13 categories
- Existing code already has some Chinese comments (e.g., dataset/human.py)
- Project structure: model/, dataset/, utils/, training scripts, inference scripts, analysis scripts

### Metis Review
**Identified Gaps** (addressed):
- Need to maintain consistency with existing documentation style
- Must handle different file types (.py, .R, .ipynb) with appropriate comment formats
- Should identify cross-file dependencies for "related files" section

---

## Work Objectives

### Core Objective
Add comprehensive bilingual header comments to all 93 code files to improve codebase maintainability and onboarding experience.

### Concrete Deliverables
- 93 documented code files with header comments
- Each file includes: purpose, I/O specification, data flow, related files, usage example

### Definition of Done
- [ ] All 93 files have header comments
- [ ] Comments are bilingual (Chinese + English)
- [ ] Each comment includes all required sections
- [ ] No syntax errors introduced
- [ ] Existing functionality preserved

### Must Have
- Bilingual comments (Chinese + English)
- File purpose description
- Input/output specification
- Data flow description
- Related files (calls and called by)
- Usage example

### Must NOT Have (Guardrails)
- Do NOT modify any functional code
- Do NOT add function-level documentation (file-level only)
- Do NOT change existing comments or docstrings
- Do NOT introduce syntax errors
- Do NOT modify non-code files (json, npy, cache, etc.)

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: None (documentation-only task)
- **Framework**: N/A

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Verification**: Use Python syntax check (`python -m py_compile`) to ensure no syntax errors
- **Visual inspection**: Spot-check documentation quality on representative files

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - Model & Dataset):
├── Task 1: Document model/ directory (9 files) [unspecified-high]
├── Task 2: Document dataset/ directory (9 files) [unspecified-high]
└── Task 3: Document utils/ directory (15 files) [unspecified-high]

Wave 2 (Core Scripts):
├── Task 4: Document training scripts (6 files) [unspecified-high]
├── Task 5: Document inference scripts (4 files) [unspecified-high]
└── Task 6: Document collection scripts (4 files) [unspecified-high]

Wave 3 (Analysis & Visualization):
├── Task 7: Document few-shot analysis scripts (6 files) [unspecified-high]
├── Task 8: Document visualization/view scripts (8 files) [unspecified-high]
├── Task 9: Document R scripts (2 files) [unspecified-high]
└── Task 10: Document Jupyter notebooks (3 files) [unspecified-high]

Wave 4 (Remaining Utilities):
├── Task 11: Document remaining utility scripts (15+ files) [unspecified-high]
└── Task 12: Document EvoRMD scripts (7 files) [unspecified-high]

Wave FINAL (Verification):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Spot-check documentation quality (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay
```

### Dependency Matrix
- **Task 1-3**: Independent, can run in parallel
- **Task 4-6**: Independent, can run in parallel (after Wave 1)
- **Task 7-10**: Independent, can run in parallel (after Wave 2)
- **Task 11-12**: Independent, can run in parallel (after Wave 3)
- **Task F1-F4**: Final verification, run after all documentation tasks

### Agent Dispatch Summary
- **Wave 1**: 3 tasks → `unspecified-high` (documentation-heavy)
- **Wave 2**: 3 tasks → `unspecified-high`
- **Wave 3**: 4 tasks → `unspecified-high`
- **Wave 4**: 2 tasks → `unspecified-high`
- **FINAL**: 4 tasks → `oracle`, `unspecified-high`, `deep`

---

## TODOs

- [ ] 1. Document model/ directory (9 files)

  **What to do**:
  - Add bilingual header comments to all 9 Python files in model/ directory
  - Files: main_model.py, multirm.py, modx.py, evormd_human.py, abla_model.py, main_model_multirm.py, main_model_collect_atten.py, modx_collect_atten.py, multirm_collect_atten.py
  - For each file, analyze:
    - Purpose: What model component does it implement?
    - Input: What tensors/data does it accept?
    - Output: What does it return?
    - Data flow: How does data flow through the model?
    - Related files: What calls this? What does it call?
  - Use the bilingual template (Chinese + English)
  - Place comments at the very beginning of each file, before imports

  **Must NOT do**:
  - Do NOT modify any model code
  - Do NOT add function-level docstrings
  - Do NOT change existing comments

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Documentation task requiring code analysis and bilingual writing
  - **Skills**: []
    - No special skills needed for documentation task

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Task 4 (training scripts depend on models)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `model/main_model.py:1-16` - Existing documentation style (English only, good structure)
  - `dataset/human.py:1-48` - Existing Chinese comment style

  **API/Type References** (contracts to implement against):
  - `utils/common.py` - Common constants and utilities used across models

  **External References** (libraries and frameworks):
  - PyTorch documentation for tensor types
  - PyTorch Geometric for graph data structures

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Verify documentation syntax
    Tool: Bash
    Preconditions: All 9 model files have been modified
    Steps:
      1. Run `python -m py_compile model/main_model.py`
      2. Run `python -m py_compile model/multirm.py`
      3. Run `python -m py_compile model/modx.py`
      4. Run `python -m py_compile model/evormd_human.py`
      5. Run `python -m py_compile model/abla_model.py`
      6. Run `python -m py_compile model/main_model_multirm.py`
      7. Run `python -m py_compile model/main_model_collect_atten.py`
      8. Run `python -m py_compile model/modx_collect_atten.py`
      9. Run `python -m py_compile model/multirm_collect_atten.py`
    Expected Result: All files compile without syntax errors
    Failure Indicators: Any SyntaxError or compilation error
    Evidence: .omo/evidence/task-1-syntax-check.txt

  Scenario: Verify documentation completeness
    Tool: Bash
    Preconditions: All 9 model files have been modified
    Steps:
      1. For each file, check if header comment exists with grep
      2. Verify bilingual content (Chinese + English)
      3. Verify all required sections present (purpose, I/O, data flow, related files)
    Expected Result: All 9 files have complete bilingual documentation
    Failure Indicators: Missing sections, missing bilingual content
    Evidence: .omo/evidence/task-1-completeness-check.txt
  ```

  **Commit**: YES
  - Message: `docs(model): add bilingual header comments to model directory`
  - Files: `model/*.py`
  - Pre-commit: `python -m py_compile model/*.py`

- [ ] 2. Document dataset/ directory (9 files)

  **What to do**:
  - Add bilingual header comments to all 9 Python files in dataset/ directory
  - Files: human.py, plant.py, ac4c.py, multirm.py, gen3.py, human_motif.py, human_with_seq.py, gen3_zero.py, plant_single.py
  - For each file, analyze:
    - Purpose: What dataset does it implement?
    - Input: What data files does it load? (npy, csv, etc.)
    - Output: What PyTorch Data objects does it return?
    - Data flow: How is data loaded, processed, and returned?
    - Related files: What training scripts use this dataset?
  - Use the bilingual template (Chinese + English)

  **Must NOT do**:
  - Do NOT modify dataset code
  - Do NOT change data loading logic
  - Do NOT modify label mappings

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Documentation task requiring data flow analysis
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Task 4 (training scripts depend on datasets)
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `dataset/human.py:1-48` - Existing Chinese comment style with label mappings

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: Verify dataset documentation syntax
    Tool: Bash
    Preconditions: All 9 dataset files have been modified
    Steps:
      1. Run `python -m py_compile dataset/human.py`
      2. Run `python -m py_compile dataset/plant.py`
      3. Run `python -m py_compile dataset/ac4c.py`
      4. Run `python -m py_compile dataset/multirm.py`
      5. Run `python -m py_compile dataset/gen3.py`
      6. Run `python -m py_compile dataset/human_motif.py`
      7. Run `python -m py_compile dataset/human_with_seq.py`
      8. Run `python -m py_compile dataset/gen3_zero.py`
      9. Run `python -m py_compile dataset/plant_single.py`
    Expected Result: All files compile without syntax errors
    Evidence: .omo/evidence/task-2-syntax-check.txt
  ```

  **Commit**: YES
  - Message: `docs(dataset): add bilingual header comments to dataset directory`
  - Files: `dataset/*.py`

- [ ] 3. Document utils/ directory (15 files)

  **What to do**:
  - Add bilingual header comments to all 15 Python files in utils/ directory
  - Files: common.py, metrics.py, logging.py, train_gen3.py, test_gen3_analyse.py, fewshot_analysis_*.py (8 files), rna_visualization.py, audit_12loc_structure.py, check_m6a_data_integrity.py, Zero_structures.py
  - For each file, analyze its utility purpose and usage

  **Must NOT do**:
  - Do NOT modify utility functions
  - Do NOT change existing constants or mappings

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Tasks 4-12 (all scripts depend on utils)
  - **Blocked By**: None

  **References**:
  - `utils/common.py` - Central utility file with constants

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: Verify utils documentation syntax
    Tool: Bash
    Preconditions: All 15 utils files have been modified
    Steps:
      1. Run `python -m py_compile utils/common.py`
      2. Run `python -m py_compile utils/metrics.py`
      3. Run `python -m py_compile utils/logging.py`
      4. Continue for all 15 files...
    Expected Result: All files compile without syntax errors
    Evidence: .omo/evidence/task-3-syntax-check.txt
  ```

  **Commit**: YES
  - Message: `docs(utils): add bilingual header comments to utils directory`
  - Files: `utils/*.py`

- [ ] 4. Document training scripts (6 files)

  **What to do**:
  - Add bilingual header comments to 6 training scripts
  - Files: train_human.py, train_plant.py, train_human_multirm.py, train_human_modx.py, train_human_evormd.py, train_multirm_dataset.py
  - Document training pipeline, configuration, and outputs

  **Must NOT do**:
  - Do NOT modify training logic
  - Do NOT change hyperparameters

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6)
  - **Blocks**: None
  - **Blocked By**: Tasks 1-3 (models, datasets, utils)

  **References**:
  - `train_human.py:1-12` - Existing English documentation style

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: Verify training script documentation
    Tool: Bash
    Preconditions: All 6 training scripts have been modified
    Steps:
      1. Run `python -m py_compile train_human.py`
      2. Run `python -m py_compile train_plant.py`
      3. Continue for all 6 files...
    Expected Result: All files compile without syntax errors
    Evidence: .omo/evidence/task-4-syntax-check.txt
  ```

  **Commit**: YES
  - Message: `docs(training): add bilingual header comments to training scripts`
  - Files: `train_*.py`

- [ ] 5. Document inference scripts (4 files)

  **What to do**:
  - Add bilingual header comments to 4 inference scripts
  - Files: inference_modx_segmented.py, inference_mrmodn_full.py, inference_multirm_segmented.py, inference_evormd_segmented.py
  - Document inference pipeline, model loading, and prediction outputs

  **Must NOT do**:
  - Do NOT modify inference logic

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6)
  - **Blocks**: None
  - **Blocked By**: Tasks 1-3

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: Verify inference script documentation
    Tool: Bash
    Preconditions: All 4 inference scripts have been modified
    Steps:
      1. Run `python -m py_compile inference_modx_segmented.py`
      2. Continue for all 4 files...
    Expected Result: All files compile without syntax errors
    Evidence: .omo/evidence/task-5-syntax-check.txt
  ```

  **Commit**: YES
  - Message: `docs(inference): add bilingual header comments to inference scripts`
  - Files: `inference_*.py`

- [ ] 6. Document collection scripts (4 files)

  **What to do**:
  - Add bilingual header comments to 4 collection scripts
  - Files: collect_human.py, collect_human_atten.py, collect_modx_atten.py, collect_multirm_atten.py
  - Document data collection and attention extraction processes

  **Must NOT do**:
  - Do NOT modify collection logic

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5)
  - **Blocks**: None
  - **Blocked By**: Tasks 1-3

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: Verify collection script documentation
    Tool: Bash
    Preconditions: All 4 collection scripts have been modified
    Steps:
      1. Run `python -m py_compile collect_human.py`
      2. Continue for all 4 files...
    Expected Result: All files compile without syntax errors
    Evidence: .omo/evidence/task-6-syntax-check.txt
  ```

  **Commit**: YES
  - Message: `docs(collection): add bilingual header comments to collection scripts`
  - Files: `collect_*.py`

- [ ] 7. Document few-shot analysis scripts (6 files)

  **What to do**:
  - Add bilingual header comments to 6 few-shot analysis scripts
  - Files: fewshot_ac4c_balance.py, fewshot_ac4c_unbalan.py, fewshot_plant_3way_independent.py, zero_shot_fewshot_analysis.py, zero_shot_fewshot_extract_only.py, utils/few_shot.py

  **Must NOT do**:
  - Do NOT modify analysis logic

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8, 9, 10)
  - **Blocks**: None
  - **Blocked By**: Tasks 1-3

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: Verify few-shot script documentation
    Tool: Bash
    Preconditions: All 6 few-shot scripts have been modified
    Steps:
      1. Run `python -m py_compile fewshot_ac4c_balance.py`
      2. Continue for all 6 files...
    Expected Result: All files compile without syntax errors
    Evidence: .omo/evidence/task-7-syntax-check.txt
  ```

  **Commit**: YES
  - Message: `docs(fewshot): add bilingual header comments to few-shot scripts`
  - Files: `fewshot_*.py`, `zero_shot_*.py`, `utils/few_shot.py`

- [ ] 8. Document visualization/view scripts (8 files)

  **What to do**:
  - Add bilingual header comments to 8 visualization scripts
  - Files: view_human.py, view_human_total.py, view_npz.py, view3.py, visualize_attention_comparison.py, run_attention_comparison.py, select_representative_sequences.py, atten_comp/*.py (3 files)

  **Must NOT do**:
  - Do NOT modify visualization logic

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 9, 10)
  - **Blocks**: None
  - **Blocked By**: Tasks 1-3

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: Verify visualization script documentation
    Tool: Bash
    Preconditions: All 8 visualization scripts have been modified
    Steps:
      1. Run `python -m py_compile view_human.py`
      2. Continue for all 8 files...
    Expected Result: All files compile without syntax errors
    Evidence: .omo/evidence/task-8-syntax-check.txt
  ```

  **Commit**: YES
  - Message: `docs(visualization): add bilingual header comments to visualization scripts`
  - Files: `view_*.py`, `visualize_*.py`, `run_*.py`, `select_*.py`, `atten_comp/*.py`

- [ ] 9. Document R scripts (2 files)

  **What to do**:
  - Add bilingual header comments to 2 R scripts
  - Files: plot_zero_fewshot_analysis.R, plot_zero_fewshot_export.R
  - Use R comment format (# style)

  **Must NOT do**:
  - Do NOT modify R code

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 8, 10)
  - **Blocks**: None
  - **Blocked By**: Tasks 1-3

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: Verify R script documentation
    Tool: Bash
    Preconditions: Both R scripts have been modified
    Steps:
      1. Run `Rscript -e "parse('plot_zero_fewshot_analysis.R')"`
      2. Run `Rscript -e "parse('plot_zero_fewshot_export.R')"`
    Expected Result: Both files parse without syntax errors
    Evidence: .omo/evidence/task-9-syntax-check.txt
  ```

  **Commit**: YES
  - Message: `docs(R): add bilingual header comments to R scripts`
  - Files: `*.R`

- [ ] 10. Document Jupyter notebooks (3 files)

  **What to do**:
  - Add bilingual header comments to 3 Jupyter notebooks
  - Files: view_gen3.ipynb, utils/rna_visualization.ipynb, utils/rna_visualization_new.ipynb
  - Add markdown cell at the beginning of each notebook

  **Must NOT do**:
  - Do NOT modify notebook code cells

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 8, 9)
  - **Blocks**: None
  - **Blocked By**: Tasks 1-3

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: Verify notebook documentation
    Tool: Bash
    Preconditions: All 3 notebooks have been modified
    Steps:
      1. Run `python -c "import json; json.load(open('view_gen3.ipynb'))"`
      2. Continue for all 3 notebooks...
    Expected Result: All notebooks are valid JSON
    Evidence: .omo/evidence/task-10-syntax-check.txt
  ```

  **Commit**: YES
  - Message: `docs(notebooks): add bilingual header comments to Jupyter notebooks`
  - Files: `*.ipynb`

- [ ] 11. Document remaining utility scripts (15+ files)

  **What to do**:
  - Add bilingual header comments to remaining utility scripts
  - Files: 3x3.py, 3x3_2.py, abla_mohe.py, cal_flops_mohe.py, cal_mean_median_mode.py, prepare_umap_data.py, prepare_umap_from_npz.py, sliding_window_utils.py, SpatialMotif.py, SpatialMotif_nobackground.py, test_gen3.py, test_multirm_4class.py, test_multirm_oversampling.py

  **Must NOT do**:
  - Do NOT modify utility code

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 12)
  - **Blocks**: None
  - **Blocked By**: Tasks 1-3

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: Verify utility script documentation
    Tool: Bash
    Preconditions: All utility scripts have been modified
    Steps:
      1. Run `python -m py_compile 3x3.py`
      2. Continue for all files...
    Expected Result: All files compile without syntax errors
    Evidence: .omo/evidence/task-11-syntax-check.txt
  ```

  **Commit**: YES
  - Message: `docs(utils): add bilingual header comments to remaining utility scripts`
  - Files: `*.py` (remaining)

- [ ] 12. Document EvoRMD scripts (7 files)

  **What to do**:
  - Add bilingual header comments to 7 EvoRMD scripts
  - Files: model/EvoRMD/Script/*.py (dataset.py, utils.py, main.py, preprocess_data.py, embedding.py, downsampling.py, train_val_test.py, model.py)

  **Must NOT do**:
  - Do NOT modify EvoRMD code

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 11)
  - **Blocks**: None
  - **Blocked By**: Tasks 1-3

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: Verify EvoRMD script documentation
    Tool: Bash
    Preconditions: All 7 EvoRMD scripts have been modified
    Steps:
      1. Run `python -m py_compile model/EvoRMD/Script/dataset.py`
      2. Continue for all 7 files...
    Expected Result: All files compile without syntax errors
    Evidence: .omo/evidence/task-12-syntax-check.txt
  ```

  **Commit**: YES
  - Message: `docs(evormd): add bilingual header comments to EvoRMD scripts`
  - Files: `model/EvoRMD/Script/*.py`

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, check comments). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .omo/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `python -m py_compile` on all modified files. Review all changed files for: syntax errors, incorrect comment format, missing bilingual content. Check that no functional code was modified.
  Output: `Syntax [PASS/FAIL] | Format [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Spot-Check Documentation Quality** — `unspecified-high`
  Select 5 representative files (1 from each major category). Read the header comments and verify: bilingual content present, all required sections included, accurate description of file purpose, correct related files listed.
  Output: `Quality [N/N pass] | Completeness [N/N] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination: Task N touching Task M's files. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1**: `docs(model/dataset/utils): add bilingual header comments` - model/*.py, dataset/*.py, utils/*.py
- **Wave 2**: `docs(training/inference/collection): add bilingual header comments` - train_*.py, inference_*.py, collect_*.py
- **Wave 3**: `docs(fewshot/visualization/R/notebooks): add bilingual header comments` - fewshot_*.py, view_*.py, *.R, *.ipynb
- **Wave 4**: `docs(remaining): add bilingual header comments` - remaining *.py, model/EvoRMD/Script/*.py

---

## Success Criteria

### Verification Commands
```bash
# Syntax check all Python files
find . -name "*.py" -exec python -m py_compile {} \;

# Syntax check all R files
Rscript -e "for(f in list.files(pattern='*.R')) parse(f)"

# Verify all notebooks are valid JSON
find . -name "*.ipynb" -exec python -c "import json; json.load(open('{}'))" \;

# Count documented files
grep -r "^\"\"\"" --include="*.py" | wc -l  # Should be 93
```

### Final Checklist
- [ ] All 93 code files have header comments
- [ ] All comments are bilingual (Chinese + English)
- [ ] All comments include: purpose, I/O, data flow, related files
- [ ] No syntax errors introduced
- [ ] No functional code modified
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent

---

## Documentation Template

### Python Files (.py)
```python
"""
[文件名] - [简短描述] / [Filename] - [Brief Description]

[详细描述文件的功能和用途] / [Detailed description of file purpose and functionality]

功能模块 / Modules:
- [模块1]: [描述] / [Module1]: [Description]
- [模块2]: [描述] / [Module2]: [Description]

输入 / Inputs:
- [输入1]: [类型] - [描述] / [Input1]: [Type] - [Description]
- [输入2]: [类型] - [描述] / [Input2]: [Type] - [Description]

输出 / Outputs:
- [输出1]: [类型] - [描述] / [Output1]: [Type] - [Description]
- [输出2]: [类型] - [描述] / [Output2]: [Type] - [Description]

数据流 / Data Flow:
1. [步骤1] / [Step1]
2. [步骤2] / [Step2]
3. [步骤3] / [Step3]

相关文件 / Related Files:
- 调用 / Calls: [文件列表 / file list]
- 被调用 / Called by: [文件列表 / file list]

使用示例 / Usage Example:
    [示例代码 / example code]

作者 / Author: [作者 / author]
日期 / Date: [日期 / date]
版本 / Version: [版本 / version]
"""
```

### R Files (.R)
```r
# ==============================================================================
# [文件名] - [简短描述] / [Filename] - [Brief Description]
# ==============================================================================
#
# [详细描述文件的功能和用途] / [Detailed description of file purpose]
#
# 功能模块 / Modules:
# - [模块1]: [描述] / [Module1]: [Description]
#
# 输入 / Inputs:
# - [输入1]: [类型] - [描述] / [Input1]: [Type] - [Description]
#
# 输出 / Outputs:
# - [输出1]: [类型] - [描述] / [Output1]: [Type] - [Description]
#
# 数据流 / Data Flow:
# 1. [步骤1] / [Step1]
# 2. [步骤2] / [Step2]
#
# 相关文件 / Related Files:
# - 调用 / Calls: [文件列表 / file list]
# - 被调用 / Called by: [文件列表 / file list]
#
# ==============================================================================
```

### Jupyter Notebooks (.ipynb)
```json
{
  "cell_type": "markdown",
  "metadata": {},
  "source": [
    "# [Notebook标题] / [Notebook Title]\n",
    "\n",
    "[详细描述notebook的功能和用途] / [Detailed description]\n",
    "\n",
    "## 功能模块 / Modules\n",
    "- [模块1]: [描述] / [Module1]: [Description]\n",
    "\n",
    "## 输入 / Inputs\n",
    "- [输入1]: [类型] - [描述] / [Input1]: [Type] - [Description]\n",
    "\n",
    "## 输出 / Outputs\n",
    "- [输出1]: [类型] - [描述] / [Output1]: [Type] - [Description]\n",
    "\n",
    "## 数据流 / Data Flow\n",
    "1. [步骤1] / [Step1]\n",
    "2. [步骤2] / [Step2]"
  ]
}
```
