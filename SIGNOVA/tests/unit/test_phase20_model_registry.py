"""
Unit tests for Phase 20 Model Registry, ModelInputSpec, and Phase 19 Gate Validation.
"""

from pathlib import Path
import pytest

from signova.live.errors import ModelUnauthorizedError
from signova.live.model_registry import LiveModelRegistry, ModelInputSpec, ModelMetadata
from signova.live.status import ModelStatus


def test_model_input_spec_serialization():
    spec = ModelInputSpec(
        feature_group="HANDS_POSE",
        feature_schema_version="1.0.0",
        temporal_window=64,
        temporal_stride=16,
        landmark_topology=543,
        normalization_version="1.0.0",
        vocabulary_version="1.0.0",
    )
    spec_dict = spec.to_dict()
    assert spec_dict["feature_group"] == "HANDS_POSE"
    assert spec_dict["temporal_window"] == 64
    assert spec_dict["landmark_topology"] == 543

    reconstructed = ModelInputSpec.from_dict(spec_dict)
    assert reconstructed.feature_group == spec.feature_group
    assert reconstructed.temporal_window == spec.temporal_window


def test_registry_inspect_model_availability_under_state_b():
    registry = LiveModelRegistry()
    status, meta, reason = registry.inspect_model_availability()
    # In current repository state (STATE_B, 0 annotations), real model must be UNAVAILABLE
    assert status == ModelStatus.UNAVAILABLE
    assert meta is None
    assert "Supervision gate STATE_B" in reason or "Real CTC model training is blocked" in reason


def test_registry_rejection_of_load_under_state_b():
    registry = LiveModelRegistry()
    with pytest.raises(ModelUnauthorizedError):
        registry.load_authorized_model()


def test_registry_feature_compatibility_verification():
    registry = LiveModelRegistry()
    # Matching HANDS_POSE with 543 topology
    assert registry.verify_feature_compatibility("HANDS_POSE", 543, "1.0.0") is True
    # Incompatible feature group
    assert registry.verify_feature_compatibility("FULL", 543, "1.0.0") is False
    # Incompatible topology
    assert registry.verify_feature_compatibility("HANDS_POSE", 100, "1.0.0") is False
