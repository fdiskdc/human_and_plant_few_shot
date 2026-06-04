"""
tests/test_naming_convention.py - 文件命名规范审计 / File naming convention audit

强制根目录 .py 文件遵循 `任务_数据集_模型.py` 规范：
Enforces root .py files follow `task_dataset_model.py`:

    task ∈ {train, inference, collect, fewshot, zeroshot}
    + 可选的子词 (例如 collect_atten_*) / optional sub-words (e.g. collect_atten_*)

并验证特定子目录的命名规则：
And validates naming rules for specific subdirectories:
    ablation/*.py     → 以 ablation_ 或 cal_ 开头
    tests/test_*.py   → pytest 标准

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-04
"""

import re
from pathlib import Path

import pytest


# ============================================================================
# 命名模式 / Naming patterns
# ============================================================================

# 任务前缀 / Task prefixes for root scripts
TASKS = ("train", "inference", "collect", "fewshot", "zeroshot")

# 根目录脚本规范: <task>_<...>.py (允许 1+ 段后缀)
# Root script pattern: <task>_<...>.py (1+ underscore segments after task)
ROOT_PATTERN = re.compile(
    r"^(?:" + "|".join(TASKS) + r")_[a-z0-9_]+\.py$"
)

# ablation/ 目录: ablation_* 或 cal_*.py
ABLATION_PATTERN = re.compile(r"^(ablation|cal)_[a-z0-9_]+\.py$")


# ============================================================================
# 根目录扫描 / Root directory scan
# ============================================================================

def test_root_no_legacy_main_model_file(project_root: Path):
    """根目录不应有以 main_model 开头的旧文件 / No legacy main_model file in root."""
    root_py = [p.name for p in project_root.glob("*.py")]
    assert not any("main_model" in n for n in root_py), \
        f"Found legacy main_model files: {[n for n in root_py if 'main_model' in n]}"


def test_root_py_files_follow_convention(project_root: Path):
    """根目录 .py 文件应遵循 task_dataset_model.py 命名 / Root .py files follow convention."""
    root_py = sorted(p.name for p in project_root.glob("*.py"))
    non_compliant = [n for n in root_py if not ROOT_PATTERN.match(n)]
    assert not non_compliant, (
        f"Non-compliant root files: {non_compliant}\n"
        f"Expected pattern: <{'|'.join(TASKS)}>_<dataset>_<model>[_<suffix>].py"
    )


def test_root_files_count_reasonable(project_root: Path):
    """根目录 .py 文件应精简 (重构目标 ~15 个，允许 25 上限)。"""
    root_py = [p.name for p in project_root.glob("*.py")]
    assert len(root_py) <= 25, (
        f"Root has {len(root_py)} .py files (expected ≤25 after refactor)"
    )


# ============================================================================
# 子目录扫描 / Subdirectory scans
# ============================================================================

def test_ablation_dir_naming(project_root: Path):
    """ablation/ 下文件应以 ablation_ 或 cal_ 开头 / ablation/ files start with ablation_ or cal_."""
    ablation_dir = project_root / "ablation"
    if not ablation_dir.is_dir():
        pytest.skip("ablation/ directory not yet created")
    py = sorted(p.name for p in ablation_dir.glob("*.py"))
    non_compliant = [n for n in py if not ABLATION_PATTERN.match(n)]
    assert not non_compliant, f"Non-compliant ablation files: {non_compliant}"


def test_tests_dir_naming(project_root: Path):
    """tests/ 下文件应以 test_ 开头 (除 conftest.py)."""
    tests_dir = project_root / "tests"
    py = [p.name for p in tests_dir.glob("*.py")]
    test_files = [n for n in py if n not in ("conftest.py", "__init__.py")]
    non_compliant = [n for n in test_files if not n.startswith("test_")]
    assert not non_compliant, f"Non-compliant tests/ files: {non_compliant}"


# ============================================================================
# Renamed model files / 重命名的 model 文件
# ============================================================================

def test_model_dir_no_legacy_main_model(model_dir: Path):
    """model/ 不应有 main_model*.py / No main_model*.py in model/."""
    files = [p.name for p in model_dir.glob("main_model*.py")]
    assert files == [], f"Legacy main_model* files in model/: {files}"


def test_model_dir_has_mrmodn_files(model_dir: Path):
    """model/ 应有 mrmodn.py, mrmodn_collect_atten.py, mrmodn_multirm.py."""
    expected = {"mrmodn.py", "mrmodn_collect_atten.py", "mrmodn_multirm.py"}
    actual = {p.name for p in model_dir.glob("mrmodn*.py")}
    missing = expected - actual
    assert not missing, f"Missing renamed model files: {missing}"
