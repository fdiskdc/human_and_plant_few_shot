"""
tests/conftest.py - pytest 全局 fixtures / pytest global fixtures

为整个测试套件提供共享的 fixtures：项目根路径、配置文件路径、
数据目录路径、临时输出目录等。
Provides shared fixtures for the entire test suite: project root,
config paths, data directory, temporary output directory, etc.

使用 / Usage:
    在测试函数参数中注入 fixture 即可自动获得对应资源。
    Inject fixtures as test function arguments to receive resources.

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-04
"""

from pathlib import Path

import pytest


# ============================================================================
# 路径相关 fixtures / Path-related fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """项目根目录 / Project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def model_dir(project_root: Path) -> Path:
    """model/ 目录 / model/ directory."""
    return project_root / "model"


@pytest.fixture(scope="session")
def dataset_dir(project_root: Path) -> Path:
    """dataset/ 目录 / dataset/ directory."""
    return project_root / "dataset"


@pytest.fixture(scope="session")
def utils_dir(project_root: Path) -> Path:
    """utils/ 目录 / utils/ directory."""
    return project_root / "utils"


@pytest.fixture(scope="session")
def json_dir(project_root: Path) -> Path:
    """json/ 配置目录 / json/ config directory."""
    return project_root / "json"


@pytest.fixture(scope="session")
def npy_dir(project_root: Path) -> Path:
    """npy/ 数据目录 (软链接) / npy/ data directory (symlink). 测试中不应修改其中数据。"""
    return project_root / "npy"


# ============================================================================
# 文件清单 fixtures / File inventory fixtures
# ============================================================================

@pytest.fixture(scope="session")
def root_py_files(project_root: Path) -> list:
    """根目录所有 .py 文件 / All .py files in project root."""
    return sorted(p.name for p in project_root.glob("*.py"))


@pytest.fixture(scope="session")
def model_py_files(model_dir: Path) -> list:
    """model/ 下所有 .py 文件 / All .py files under model/."""
    return sorted(p.name for p in model_dir.glob("*.py"))
