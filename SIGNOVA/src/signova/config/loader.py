"""
Configuration loader for SIGNOVA.

Loads YAML configuration files, expands environment variables, resolves relative paths
against the project root, and provides validated config objects.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from signova.config.schema import (
    BaseConfig,
    DatasetConfig,
    InferenceConfig,
    ModelConfig,
    PathsConfig,
    SignovaConfig,
    TrainingConfig,
)


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand environment variables in configuration strings (${VAR:-default})."""
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(v) for v in data]
    elif isinstance(data, str):
        # Match ${VAR} or ${VAR:-default}
        pattern = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")

        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default_val)

        return pattern.sub(replace, data)
    return data


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and parse a YAML file safely.

    Args:
        file_path: Absolute or relative path to the YAML file.

    Returns:
        Dictionary parsed from the YAML content.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return _expand_env_vars(data)


class ConfigManager:
    """
    Manages loading and resolving all SIGNOVA configuration components.
    """

    def __init__(self, config_dir: Optional[Union[str, Path]] = None, project_root: Optional[Union[str, Path]] = None):
        """
        Initialize ConfigManager.

        Args:
            config_dir: Path to directory containing YAML configs. If None, auto-detected.
            project_root: Root directory of the SIGNOVA project.
        """
        if config_dir is not None:
            self.config_dir = Path(config_dir).resolve()
        else:
            # Default to ../../../configs relative to this file
            self.config_dir = (Path(__file__).resolve().parent.parent.parent.parent / "configs").resolve()

        if project_root is not None:
            self.project_root = Path(project_root).resolve()
        else:
            self.project_root = self.config_dir.parent.resolve()

    def load_base_config(self) -> BaseConfig:
        data = load_yaml(self.config_dir / "base.yaml")
        return BaseConfig(**data)

    def load_dataset_config(self) -> DatasetConfig:
        data = load_yaml(self.config_dir / "dataset.yaml")
        return DatasetConfig(**data)

    def load_model_config(self) -> ModelConfig:
        data = load_yaml(self.config_dir / "model.yaml")
        return ModelConfig(**data)

    def load_training_config(self) -> TrainingConfig:
        data = load_yaml(self.config_dir / "training.yaml")
        return TrainingConfig(**data)

    def load_inference_config(self) -> InferenceConfig:
        data = load_yaml(self.config_dir / "inference.yaml")
        return InferenceConfig(**data)

    def load_paths_config(self) -> PathsConfig:
        data = load_yaml(self.config_dir / "paths.yaml")
        return PathsConfig(**data)

    def load_all(self) -> SignovaConfig:
        """Load all configuration modules into a unified SignovaConfig object."""
        return SignovaConfig(
            base=self.load_base_config(),
            dataset=self.load_dataset_config(),
            model=self.load_model_config(),
            training=self.load_training_config(),
            inference=self.load_inference_config(),
            paths=self.load_paths_config(),
        )

    def resolve_path(self, relative_path: Union[str, Path]) -> Path:
        """Resolve a path relative to the project root."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()


def get_config(config_dir: Optional[Union[str, Path]] = None) -> SignovaConfig:
    """Convenience function to get fully loaded configuration."""
    manager = ConfigManager(config_dir=config_dir)
    return manager.load_all()
