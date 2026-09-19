"""
Canonical data model definitions for SIGNOVA sign language samples.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class SampleAvailability(str, Enum):
    REMOTE_ONLY = "REMOTE_ONLY"
    LOCAL_AVAILABLE = "LOCAL_AVAILABLE"
    BOTH = "BOTH"
    MISSING = "MISSING"


@dataclass
class SignSample:
    """
    Canonical representation of a sign language data sample in SIGNOVA.
    Dataset-agnostic contract for continuous sentences, isolated signs, and multi-modal streams.
    """
    sample_id: str
    dataset: str
    video_reference: str
    split: str = "train"  # "train", "val", "test", "unassigned"
    local_video_path: Optional[str] = None
    features_path: Optional[str] = None
    source_language: str = "Indian Sign Language (ISL)"
    target_language: str = "English"
    source_annotation: Optional[str] = None
    target_translation: Optional[str] = None
    gloss_sequence: Optional[List[str]] = None
    signer_id: Optional[str] = None  # None if unverified
    session_id: Optional[str] = None
    duration_sec: Optional[float] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    availability: SampleAvailability = SampleAvailability.REMOTE_ONLY
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["availability"] = self.availability.value
        if self.gloss_sequence is not None:
            d["gloss_sequence"] = json.dumps(self.gloss_sequence)
        if isinstance(d.get("metadata"), dict) and len(d["metadata"]) > 0:
            d["metadata"] = json.dumps(d["metadata"])
        else:
            d["metadata"] = ""
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SignSample":
        d_copy = dict(d)
        if "availability" in d_copy:
            try:
                d_copy["availability"] = SampleAvailability(d_copy["availability"])
            except ValueError:
                d_copy["availability"] = SampleAvailability.REMOTE_ONLY

        if "gloss_sequence" in d_copy and isinstance(d_copy["gloss_sequence"], str) and d_copy["gloss_sequence"].strip():
            try:
                d_copy["gloss_sequence"] = json.loads(d_copy["gloss_sequence"])
            except json.JSONDecodeError:
                d_copy["gloss_sequence"] = None

        if "metadata" in d_copy and isinstance(d_copy["metadata"], str) and d_copy["metadata"].strip():
            try:
                d_copy["metadata"] = json.loads(d_copy["metadata"])
            except json.JSONDecodeError:
                d_copy["metadata"] = {}
        elif "metadata" not in d_copy or not isinstance(d_copy["metadata"], dict):
            d_copy["metadata"] = {}

        return cls(**d_copy)
