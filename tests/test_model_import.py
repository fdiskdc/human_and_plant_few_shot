"""
tests/test_model_import.py - 模型层导入回归测试 / Model layer import regression tests

验证重命名后 model.* 子模块仍可正确导入，且关键类符号 (RNA_ClassQuery_Model,
ParallelCNNBlock, GCNBlock, ClassQueryHead 等) 通过 model 包入口可见。
Validates that after renaming, model.* submodules import correctly and that key
class symbols (RNA_ClassQuery_Model, ParallelCNNBlock, GCNBlock, ClassQueryHead,
etc.) remain accessible via the `model` package entry.

注意 / Notes:
- 这些测试仅做 import 层面的回归 / These tests only do import-layer regression.
- 不构造张量、不前向、不依赖 GPU / No tensor construction, no forward, no GPU.

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-04
"""

import importlib

import pytest


# ============================================================================
# model 包入口 / model package entry
# ============================================================================

def test_model_package_imports():
    """model 包顶层导入成功 / model package top-level import succeeds."""
    import model
    assert hasattr(model, "RNA_ClassQuery_Model")
    assert hasattr(model, "ParallelCNNBlock")
    assert hasattr(model, "GCNBlock")
    assert hasattr(model, "ClassQueryHead")
    assert hasattr(model, "ClassQueryHeadPooling")
    assert hasattr(model, "HierarchicalClassQueryHeadPooling")


def test_model_version_string():
    """model.__version__ 存在 / model.__version__ exists."""
    import model
    assert hasattr(model, "__version__")
    assert isinstance(model.__version__, str)


# ============================================================================
# 子模块导入 / Submodule imports
# ============================================================================

@pytest.mark.parametrize("module_name", [
    "model.mrmodn",                 # 重命名自 main_model
    "model.mrmodn_collect_atten",   # 重命名自 main_model_collect_atten
    "model.mrmodn_multirm",         # 重命名自 main_model_multirm
])
def test_renamed_submodules_importable(module_name: str):
    """重命名后的 mrmodn* 子模块可导入 / Renamed mrmodn* submodules import."""
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", [
    "model.modx",
    "model.multirm",
    "model.evormd_human",
    "model.abla_model",
    "model.modx_collect_atten",
    "model.multirm_collect_atten",
])
def test_other_submodules_importable(module_name: str):
    """其他未重命名的 model 子模块仍可导入 / Other un-renamed model submodules still import."""
    mod = importlib.import_module(module_name)
    assert mod is not None


# ============================================================================
# 关键符号 / Key symbols
# ============================================================================

def test_main_model_class_via_renamed_module():
    """RNA_ClassQuery_Model 可从 model.mrmodn 直接导入 / RNA_ClassQuery_Model importable from model.mrmodn."""
    from model.mrmodn import RNA_ClassQuery_Model
    assert RNA_ClassQuery_Model is not None


def test_collect_atten_class_via_renamed_module():
    """RNA_ClassQuery_Model_Collect_Atten 可从 model.mrmodn_collect_atten 导入."""
    from model.mrmodn_collect_atten import RNA_ClassQuery_Model_Collect_Atten
    assert RNA_ClassQuery_Model_Collect_Atten is not None


def test_old_main_model_path_is_gone():
    """旧的 model.main_model 路径不应再可导入 / Old model.main_model path is gone."""
    with pytest.raises(ImportError):
        importlib.import_module("model.main_model")
    with pytest.raises(ImportError):
        importlib.import_module("model.main_model_collect_atten")
    with pytest.raises(ImportError):
        importlib.import_module("model.main_model_multirm")
