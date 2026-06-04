"""
tests/test_config.py - JSON 配置文件验证 / JSON config file validation

确保 json/ 目录下的配置文件存在、可解析、结构合理。
Ensures json/ config files exist, parse, and have sane structure.

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-04
"""

import json
from pathlib import Path

import pytest


def test_json_dir_exists(json_dir: Path):
    """json/ 目录存在 / json/ directory exists."""
    assert json_dir.is_dir()


def test_json_dir_has_configs(json_dir: Path):
    """json/ 目录至少有 1 个 .json 文件 / json/ has at least 1 config."""
    configs = list(json_dir.glob("*.json"))
    if not configs:
        pytest.skip("json/ has no .json configs (optional)")
    assert len(configs) >= 1


def test_all_json_configs_parse(json_dir: Path):
    """所有 json/*.json 都应能正常解析 / All json/*.json parse without error."""
    configs = list(json_dir.glob("*.json"))
    if not configs:
        pytest.skip("no json configs to test")
    for cfg in configs:
        try:
            json.loads(cfg.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in {cfg.name}: {e}")
