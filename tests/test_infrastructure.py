"""
tests/test_infrastructure.py - 测试基础设施验证 / Test infrastructure validation

验证 pytest 框架本身可用，项目目录结构存在，关键路径可访问。
Validate that the pytest framework works, the project directory structure exists,
and key paths are reachable.

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-04
"""

from pathlib import Path


# ============================================================================
# 基础 sanity 检查 / Basic sanity checks
# ============================================================================

def test_pytest_works():
    """pytest 自身可用 / pytest itself works."""
    assert True


def test_python_version():
    """Python 版本 >= 3.8 / Python version >= 3.8."""
    import sys
    assert sys.version_info >= (3, 8), f"Python {sys.version_info} too old"


# ============================================================================
# 项目目录验证 / Project directory checks
# ============================================================================

def test_project_root_exists(project_root: Path):
    """项目根目录存在 / Project root exists."""
    assert project_root.exists()
    assert project_root.is_dir()


def test_model_dir_exists(model_dir: Path):
    """model/ 目录存在 / model/ directory exists."""
    assert model_dir.is_dir()
    assert (model_dir / "__init__.py").is_file()


def test_dataset_dir_exists(dataset_dir: Path):
    """dataset/ 目录存在 / dataset/ directory exists."""
    assert dataset_dir.is_dir()


def test_utils_dir_exists(utils_dir: Path):
    """utils/ 目录存在 / utils/ directory exists."""
    assert utils_dir.is_dir()


def test_visualization_dir_exists(project_root: Path):
    """visualization/ 重构后目录存在 / visualization/ refactor target directory exists."""
    assert (project_root / "visualization").is_dir()


def test_ablation_dir_exists(project_root: Path):
    """ablation/ 重构后目录存在 / ablation/ refactor target directory exists."""
    assert (project_root / "ablation").is_dir()


def test_tests_dir_exists(project_root: Path):
    """tests/ 目录存在 / tests/ directory exists."""
    tests_dir = project_root / "tests"
    assert tests_dir.is_dir()
    assert (tests_dir / "__init__.py").is_file()


def test_analysis_dir_exists(project_root: Path):
    """analysis/ 重构后目录存在 / analysis/ refactor target directory exists."""
    assert (project_root / "analysis").is_dir()
