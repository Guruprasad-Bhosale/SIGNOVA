"""
Unit tests for SIGNOVA configuration management and path resolution.
"""

from pathlib import Path
import pytest
from signova.config.loader import ConfigManager, get_config
from signova.config.schema import BaseConfig, DatasetConfig, ModelConfig, SignovaConfig


def test_config_loader_discovers_configs():
    cm = ConfigManager()
    cfg = cm.load_all()

    assert isinstance(cfg, SignovaConfig)
    assert cfg.base.project.name == "SIGNOVA"
    assert cfg.base.project.phase == 0
    assert cfg.dataset.landmarks.total_landmarks == 543


def test_path_resolution():
    cm = ConfigManager()
    resolved = cm.resolve_path("data/manifests")
    assert resolved.name == "manifests"
    assert resolved.is_absolute()


def test_model_config_defaults():
    cm = ConfigManager()
    model_cfg = cm.load_model_config()
    assert model_cfg.model.get("name") in ["baseline_rnn", "baseline_tcn", "baseline_pooled", "pending_dataset_audit"] or "architecture" in model_cfg.model
