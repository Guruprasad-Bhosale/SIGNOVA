"""
Model Registry and Real-Model Safety Gate for SIGNOVA Live Inference.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from signova.features.feature_groups import LandmarkGroup
from signova.live.errors import ModelIncompatibleError, ModelUnauthorizedError
from signova.live.status import ModelStatus


@dataclass
class ModelInputSpec:
    """Explicit specification of features and temporal geometry expected by the model."""
    feature_group: str = "HANDS_POSE"
    feature_schema_version: str = "1.0.0"
    temporal_window: int = 64
    temporal_stride: int = 16
    landmark_topology: int = 543
    normalization_version: str = "1.0.0"
    vocabulary_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_group": self.feature_group,
            "feature_schema_version": self.feature_schema_version,
            "temporal_window": self.temporal_window,
            "temporal_stride": self.temporal_stride,
            "landmark_topology": self.landmark_topology,
            "normalization_version": self.normalization_version,
            "vocabulary_version": self.vocabulary_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInputSpec":
        return cls(
            feature_group=data.get("feature_group", "HANDS_POSE"),
            feature_schema_version=data.get("feature_schema_version", "1.0.0"),
            temporal_window=int(data.get("temporal_window", 64)),
            temporal_stride=int(data.get("temporal_stride", 16)),
            landmark_topology=int(data.get("landmark_topology", 543)),
            normalization_version=data.get("normalization_version", "1.0.0"),
            vocabulary_version=data.get("vocabulary_version", "1.0.0"),
        )


@dataclass
class ModelMetadata:
    model_id: str
    model_version: str
    architecture: str
    training_state: str
    input_spec: ModelInputSpec
    checkpoint_path: str
    checkpoint_sha256: str
    vocabulary_path: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    is_synthetic_fixture: bool = False
    is_authorized_real_model: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "architecture": self.architecture,
            "training_state": self.training_state,
            "input_spec": self.input_spec.to_dict(),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "vocabulary_path": self.vocabulary_path,
            "provenance": self.provenance,
            "is_synthetic_fixture": self.is_synthetic_fixture,
            "is_authorized_real_model": self.is_authorized_real_model,
        }


class LiveModelRegistry:
    """
    Manages model discovery, strict Phase 19 supervision gate verification,
    device selection, and input specification compatibility.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "models/checkpoints",
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

        self._active_model: Optional[nn.Module] = None
        self._active_metadata: Optional[ModelMetadata] = None
        self._default_spec = ModelInputSpec()

    @property
    def default_spec(self) -> ModelInputSpec:
        return self._default_spec

    def get_hardware_info(self) -> Dict[str, Any]:
        """Returns active compute hardware and CUDA telemetry."""
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        }
        return info

    def inspect_model_availability(self) -> Tuple[ModelStatus, Optional[ModelMetadata], str]:
        """
        Dynamically checks whether a genuine authorized CTC model exists.

        Returns:
            (ModelStatus, Optional[ModelMetadata], reason_str)
        """
        # 1. First inspect the canonical Phase 19 readiness gate
        from signova.operations.phase19_orchestrator import evaluate_phase19_readiness
        readiness = evaluate_phase19_readiness()

        supervision_state = readiness.get("supervision_state", "STATE_B")
        real_ctc_status = readiness.get("real_ctc_status", "BLOCKED")
        auth_reason = readiness.get("training_authorization", {}).get("reason", "no_genuine_human_annotations_present")

        if supervision_state == "STATE_B" or real_ctc_status != "ALLOWED":
            return (
                ModelStatus.UNAVAILABLE,
                None,
                f"Supervision gate {supervision_state} ({auth_reason}). Real CTC model training is blocked.",
            )

        # 2. Check model checkpoint files if authorized
        # First check for Phase 21 live model pointer
        pointer_file = Path("models/experiments/phase21_real_ctc/live_model_pointer.json")
        target_dir = self.checkpoint_dir
        if pointer_file.exists():
            try:
                pdata = json.loads(pointer_file.read_text(encoding="utf-8"))
                active_dir = Path(pdata.get("active_run_dir", ""))
                if active_dir.exists():
                    target_dir = active_dir
            except Exception:
                pass

        meta_file = target_dir / "model_metadata.json"
        if not meta_file.exists():
            meta_file = target_dir / "best_model_meta.json"
        ckpt_file = target_dir / "checkpoint.pt"
        if not ckpt_file.exists():
            ckpt_file = target_dir / "best_model.pt"

        if not meta_file.exists() or not ckpt_file.exists():
            return (
                ModelStatus.UNAVAILABLE,
                None,
                f"Authorized state detected, but no trained checkpoint file exists at {target_dir}.",
            )

        try:
            meta_dict = json.loads(meta_file.read_text(encoding="utf-8"))
            input_spec = ModelInputSpec.from_dict(meta_dict.get("input_spec", {}))
            vocab_path = str(target_dir / "vocabulary.json") if (target_dir / "vocabulary.json").exists() else meta_dict.get("vocabulary_path", "data/annotations/phase11/vocabulary.json")
            metadata = ModelMetadata(
                model_id=meta_dict.get("model_id", "signova_ctc_v1"),
                model_version=meta_dict.get("model_version", "1.0.0"),
                architecture=meta_dict.get("architecture", "CTCContinuousRecognizer"),
                training_state=meta_dict.get("training_state", supervision_state),
                input_spec=input_spec,
                checkpoint_path=str(ckpt_file),
                checkpoint_sha256=meta_dict.get("checkpoint_sha256", ""),
                vocabulary_path=vocab_path,
                provenance=meta_dict.get("provenance", {}),
                is_synthetic_fixture=meta_dict.get("is_synthetic_fixture", False),
                is_authorized_real_model=True,
            )

            # Reject synthetic fixtures claiming to be real
            if metadata.is_synthetic_fixture:
                return (
                    ModelStatus.BLOCKED,
                    None,
                    "Rejected synthetic test fixture checkpoint attempting to masquerade as a real model.",
                )

            return ModelStatus.AVAILABLE, metadata, "Genuine authorized model available."
        except Exception as e:
            return (
                ModelStatus.INCOMPATIBLE,
                None,
                f"Failed to parse model metadata: {e}",
            )

    def load_authorized_model(self) -> Tuple[Optional[nn.Module], Optional[ModelMetadata]]:
        """
        Loads the authorized model into memory and places it on the selected device.
        Throws ModelUnauthorizedError if called while supervision state is unauthorized.
        """
        status, metadata, reason = self.inspect_model_availability()
        if status != ModelStatus.AVAILABLE or metadata is None:
            raise ModelUnauthorizedError(f"Cannot load real model: {reason}")

        # Verify checkpoint SHA-256
        ckpt_path = Path(metadata.checkpoint_path)
        hasher = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        calc_sha = hasher.hexdigest().upper()

        if metadata.checkpoint_sha256 and calc_sha != metadata.checkpoint_sha256.upper():
            raise ModelIncompatibleError(
                f"Checkpoint SHA-256 mismatch! Expected {metadata.checkpoint_sha256}, got {calc_sha}"
            )

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        if metadata.architecture == "Phase21BiGRUCTCModel":
            from signova.operations.phase21_orchestrator import Phase21BiGRUCTCModel
            model = Phase21BiGRUCTCModel(
                input_dim=checkpoint.get("input_dim", 150),
                hidden_dim=checkpoint.get("hidden_dim", 128),
                num_layers=checkpoint.get("num_layers", 2),
                num_classes=checkpoint.get("num_classes", 10),
            )
        else:
            from signova.models.ctc_recognizer import CTCContinuousRecognizer
            model = CTCContinuousRecognizer(
                num_classes=checkpoint.get("num_classes", 100),
                landmark_group=metadata.input_spec.feature_group,
            )
            
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        model.to(self.device)
        model.eval()

        self._active_model = model
        self._active_metadata = metadata
        return model, metadata

    def verify_feature_compatibility(
        self,
        live_feature_group: Union[str, LandmarkGroup],
        live_topology: int = 543,
        live_normalization_version: str = "1.0.0",
    ) -> bool:
        """
        Verifies that live stream features conform exactly to the model input spec.
        """
        spec = self._active_metadata.input_spec if self._active_metadata else self._default_spec
        grp_str = live_feature_group.value if isinstance(live_feature_group, LandmarkGroup) else str(live_feature_group)

        if grp_str.upper() != spec.feature_group.upper():
            return False
        if live_topology != spec.landmark_topology:
            return False
        if live_normalization_version != spec.normalization_version:
            return False
        return True
