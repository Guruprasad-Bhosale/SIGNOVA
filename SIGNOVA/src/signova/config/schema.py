"""
Configuration schema definitions for SIGNOVA using Pydantic.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "SIGNOVA"
    version: str = "0.1.0"
    task: str = "continuous_isl_translation"
    source_language: str = "Indian Sign Language (ISL)"
    target_language: str = "English"
    phase: int = 0
    description: str = "Continuous Indian Sign Language to English Translation Pipeline"


class RuntimeConfig(BaseModel):
    seed: int = 42
    device: str = "auto"
    num_threads: int = 4
    log_level: str = "INFO"


class BaseConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


class DatasetSplits(BaseModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


class DatasetEntry(BaseModel):
    enabled: bool = True
    description: Optional[str] = None
    root: Optional[str] = None
    annotations_csv: Optional[str] = None
    validation_csv: Optional[str] = None
    splits: Optional[DatasetSplits] = None


class PreprocessingConfig(BaseModel):
    target_fps: int = 30
    max_frames: int = 128
    min_frames: int = 8
    normalize_landmarks: bool = True
    center_on: str = "mid_hip"
    scale_by: str = "shoulder_distance"


class LandmarksConfig(BaseModel):
    extractor: str = "mediapipe_holistic"
    num_pose_landmarks: int = 33
    num_face_landmarks: int = 468
    num_hand_landmarks: int = 42
    total_landmarks: int = 543
    coordinates_per_landmark: int = 3
    interpolate_missing: bool = True


class DatasetConfig(BaseModel):
    datasets: Dict[str, DatasetEntry] = Field(default_factory=dict)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    landmarks: LandmarksConfig = Field(default_factory=LandmarksConfig)


class ModelSubConfig(BaseModel):
    type: str = "pending_dataset_audit"
    in_channels: Optional[int] = None
    hidden_channels: Optional[int] = None
    num_layers: Optional[int] = None
    dropout: Optional[float] = None
    d_model: Optional[int] = None
    nhead: Optional[int] = None
    num_encoder_layers: Optional[int] = None
    dim_feedforward: Optional[int] = None
    vocab_size: Optional[Any] = None
    blank_token: Optional[int] = 0
    source_vocab_size: Optional[Any] = None
    target_vocab_size: Optional[Any] = None


class ModelConfig(BaseModel):
    model: Dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    training: Dict[str, Any] = Field(default_factory=dict)
    augmentation: Dict[str, Any] = Field(default_factory=dict)


class WebcamConfig(BaseModel):
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    visualize_landmarks: bool = True


class InferenceConfig(BaseModel):
    inference: Dict[str, Any] = Field(default_factory=dict)


class PathsConfig(BaseModel):
    paths: Dict[str, str] = Field(default_factory=dict)


class SignovaConfig(BaseModel):
    base: BaseConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    paths: PathsConfig
