"""
Canonical Human Annotation Data Model for SIGNOVA.

Distinguishes:
1. SEQUENCE-ONLY ANNOTATION: Ordered list of sign glosses without frame boundaries.
2. TEMPORALLY-ALIGNED ANNOTATION: Segment-level timestamps or frame boundaries aligned to individual glosses.
"""

from dataclasses import dataclass, field
import datetime
from typing import Any, Dict, List, Optional
from signova.annotation.constants import (
    QUALITY_UNVERIFIED,
    REVIEW_STATE_UNANNOTATED,
)


@dataclass
class GlossToken:
    token: str
    confidence: str = "HIGH"  # "HIGH", "MEDIUM", "LOW", "UNCERTAIN"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class TemporalSegment:
    gloss: str
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    start_time_ms: Optional[float] = None
    end_time_ms: Optional[float] = None
    confidence: str = "HIGH"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gloss": self.gloss,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class VideoAnnotation:
    annotation_id: str
    sample_id: str
    annotator_id: str
    is_temporally_aligned: bool
    glosses: List[str]
    segments: List[TemporalSegment] = field(default_factory=list)
    english_translation: str = ""
    review_status: str = REVIEW_STATE_UNANNOTATED
    quality_grade: str = QUALITY_UNVERIFIED
    reviewer_id: Optional[str] = None
    reviewer_notes: str = ""
    dataset_split: str = "unassigned"  # "train", "val", "test", "unassigned"
    training_eligible: bool = False
    provenance_id: str = "prov_unknown"
    version: str = "0.1.0"
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "sample_id": self.sample_id,
            "annotator_id": self.annotator_id,
            "is_temporally_aligned": self.is_temporally_aligned,
            "glosses": self.glosses,
            "segments": [s.to_dict() for s in self.segments],
            "english_translation": self.english_translation,
            "review_status": self.review_status,
            "quality_grade": self.quality_grade,
            "reviewer_id": self.reviewer_id,
            "reviewer_notes": self.reviewer_notes,
            "dataset_split": self.dataset_split,
            "training_eligible": self.training_eligible,
            "provenance_id": self.provenance_id,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass
class AnnotationSet:
    sample_id: str
    video_id: str
    annotations: List[VideoAnnotation] = field(default_factory=list)
    consensus_annotation_id: Optional[str] = None
    disagreement_status: str = "NONE"  # "NONE", "DISAGREEMENT_OPEN", "RESOLVED"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_annotation(self, annotation: VideoAnnotation) -> None:
        self.annotations.append(annotation)

    def get_annotator_annotation(self, annotator_id: str) -> Optional[VideoAnnotation]:
        for a in self.annotations:
            if a.annotator_id == annotator_id:
                return a
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "video_id": self.video_id,
            "annotation_count": len(self.annotations),
            "consensus_annotation_id": self.consensus_annotation_id,
            "disagreement_status": self.disagreement_status,
            "annotations": [a.to_dict() for a in self.annotations],
            "metadata": self.metadata,
        }
