"""
Pydantic schemas for data sample validation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from signova.data.models import SampleAvailability


class SignSampleSchema(BaseModel):
    sample_id: str
    dataset: str
    video_reference: str
    split: str = "train"
    local_video_path: Optional[str] = None
    features_path: Optional[str] = None
    source_language: str = "Indian Sign Language (ISL)"
    target_language: str = "English"
    source_annotation: Optional[str] = None
    target_translation: Optional[str] = None
    gloss_sequence: Optional[List[str]] = None
    signer_id: Optional[str] = None
    session_id: Optional[str] = None
    duration_sec: Optional[float] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    availability: SampleAvailability = SampleAvailability.REMOTE_ONLY
    metadata: Dict[str, Any] = Field(default_factory=dict)
