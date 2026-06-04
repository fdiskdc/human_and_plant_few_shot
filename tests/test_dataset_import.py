"""
tests/test_dataset_import.py - dataset 包导入回归测试 / dataset package import regression tests

验证所有 dataset/*.py 在 import 层不报错。
Validates that every dataset/*.py imports without error.

排除 / Excluded:
- plant_single.py (已知 pre-existing SyntaxError 自 commit dd6cd0b) /
  (known pre-existing SyntaxError from commit dd6cd0b — outside refactor scope)

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-04
"""

import importlib

import pytest


# dataset 子模块列表 / dataset submodule list (排除 pre-existing 坏文件)
DATASET_MODULES = [
    "dataset.ac4c",
    "dataset.gen3",
    "dataset.gen3_zero",
    "dataset.human",
    "dataset.human_motif",
    "dataset.human_with_seq",
    "dataset.multirm",
    "dataset.plant",
    # "dataset.plant_single",  # pre-existing SyntaxError, not in refactor scope
]


@pytest.mark.parametrize("module_name", DATASET_MODULES)
def test_dataset_submodules_importable(module_name: str):
    """每个 dataset 子模块可导入 / Each dataset submodule imports."""
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_dataset_human_has_expected_class():
    """dataset.human 应有 Mer100Dataset / dataset.human should expose Mer100Dataset."""
    from dataset.human import Mer100Dataset
    assert Mer100Dataset is not None


def test_dataset_multirm_has_expected_class():
    """dataset.multirm 应有 MultirmDataset / dataset.multirm should expose MultirmDataset."""
    from dataset.multirm import MultirmDataset
    assert MultirmDataset is not None
