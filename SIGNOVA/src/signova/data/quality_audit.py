"""
Phase 9 Annotation Quality Audit Engine for SIGNOVA.

Measures:
- Empty gloss rate
- Malformed token rate
- Duplicate sample rate
- Missing video / annotation rate
- Unknown token rate
- Sequence length distribution
- Temporal alignment validity
- CTC feasibility ($T \\ge L$ and pooling margin)
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd


VALID_GLOSS_PATTERN = re.compile(r"^[A-Z0-9]+(?:-[A-Z0-9]+)*(?:::[A-Z0-9]+)?$")


@dataclass
class AnnotationQualityReport:
    total_samples: int
    empty_gloss_samples: int
    empty_gloss_rate: float
    malformed_tokens_count: int
    malformed_token_rate: float
    duplicate_samples_count: int
    duplicate_sample_rate: float
    missing_video_count: int
    missing_video_rate: float
    missing_annotation_count: int
    missing_annotation_rate: float
    unknown_tokens_count: int
    unknown_token_rate: float
    sequence_length_stats: Dict[str, float]
    temporal_alignment_validity_rate: float
    ctc_feasible_samples_count: int
    ctc_feasibility_rate: float
    overall_quality_grade: str  # "UNVERIFIED", "WEAK", "PARTIAL", "VERIFIED", "LINGUIST_REVIEWED"
    training_eligible: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "empty_gloss_samples": self.empty_gloss_samples,
            "empty_gloss_rate": round(self.empty_gloss_rate, 4),
            "malformed_tokens_count": self.malformed_tokens_count,
            "malformed_token_rate": round(self.malformed_token_rate, 4),
            "duplicate_samples_count": self.duplicate_samples_count,
            "duplicate_sample_rate": round(self.duplicate_sample_rate, 4),
            "missing_video_count": self.missing_video_count,
            "missing_video_rate": round(self.missing_video_rate, 4),
            "missing_annotation_count": self.missing_annotation_count,
            "missing_annotation_rate": round(self.missing_annotation_rate, 4),
            "unknown_tokens_count": self.unknown_tokens_count,
            "unknown_token_rate": round(self.unknown_token_rate, 4),
            "sequence_length_stats": self.sequence_length_stats,
            "temporal_alignment_validity_rate": round(self.temporal_alignment_validity_rate, 4),
            "ctc_feasible_samples_count": self.ctc_feasible_samples_count,
            "ctc_feasibility_rate": round(self.ctc_feasibility_rate, 4),
            "overall_quality_grade": self.overall_quality_grade,
            "training_eligible": self.training_eligible,
            "details": self.details,
        }


def validate_gloss_token(token: str) -> bool:
    """Check if token follows uppercase standardized gloss conventions."""
    if not token or not isinstance(token, str):
        return False
    return bool(VALID_GLOSS_PATTERN.match(token.strip()))


def audit_annotation_dataset(
    samples: List[Dict[str, Any]],
    known_vocabulary: Optional[Set[str]] = None,
    default_frame_rate: float = 30.0,
    downsample_factor: int = 1,
) -> AnnotationQualityReport:
    """
    Audits a list of annotation records for quality, validity, and CTC feasibility.
    """
    total_samples = len(samples)
    if total_samples == 0:
        return AnnotationQualityReport(
            total_samples=0,
            empty_gloss_samples=0,
            empty_gloss_rate=0.0,
            malformed_tokens_count=0,
            malformed_token_rate=0.0,
            duplicate_samples_count=0,
            duplicate_sample_rate=0.0,
            missing_video_count=0,
            missing_video_rate=0.0,
            missing_annotation_count=0,
            missing_annotation_rate=0.0,
            unknown_tokens_count=0,
            unknown_token_rate=0.0,
            sequence_length_stats={"min": 0, "max": 0, "mean": 0.0, "median": 0.0},
            temporal_alignment_validity_rate=1.0,
            ctc_feasible_samples_count=0,
            ctc_feasibility_rate=0.0,
            overall_quality_grade="UNVERIFIED",
            training_eligible=False,
            details={"error": "Empty dataset"},
        )

    empty_count = 0
    total_tokens = 0
    malformed_tokens = 0
    unknown_tokens = 0
    missing_video = 0
    missing_annot = 0
    valid_temporal_count = 0
    ctc_feasible_count = 0
    seq_lengths = []
    seen_ids = set()
    duplicate_count = 0

    for s in samples:
        sample_id = s.get("sample_id") or s.get("video_id") or ""
        if sample_id in seen_ids:
            duplicate_count += 1
        elif sample_id:
            seen_ids.add(sample_id)

        if not s.get("video_path") and not s.get("video_exists", True):
            missing_video += 1

        glosses = s.get("glosses") or s.get("ordered_glosses") or []
        if not glosses or (isinstance(glosses, list) and len(glosses) == 0):
            empty_count += 1
            seq_lengths.append(0)
            missing_annot += 1
        else:
            seq_lengths.append(len(glosses))
            for g in glosses:
                total_tokens += 1
                if not validate_gloss_token(str(g)):
                    malformed_tokens += 1
                if known_vocabulary and str(g).upper() not in known_vocabulary:
                    unknown_tokens += 1

        # Temporal validity
        start_t = s.get("start_time", 0.0)
        end_t = s.get("end_time", 0.0)
        if end_t > start_t or (start_t == 0.0 and end_t == 0.0):
            valid_temporal_count += 1

        # CTC feasibility check: T_effective >= L
        num_frames = s.get("num_frames") or s.get("duration_frames") or 0
        effective_frames = num_frames // downsample_factor if downsample_factor > 0 else num_frames
        num_glosses = len(glosses)
        if num_glosses > 0 and (effective_frames >= num_glosses or num_frames == 0):
            ctc_feasible_count += 1
        elif num_glosses == 0:
            pass

    empty_rate = empty_count / total_samples
    malformed_rate = malformed_tokens / max(1, total_tokens)
    unknown_rate = unknown_tokens / max(1, total_tokens)
    duplicate_rate = duplicate_count / total_samples
    missing_vid_rate = missing_video / total_samples
    missing_ann_rate = missing_annot / total_samples
    temporal_valid_rate = valid_temporal_count / total_samples
    ctc_feasibility_rate = ctc_feasible_count / max(1, (total_samples - empty_count))

    seq_stats = {
        "min": float(np.min(seq_lengths)) if seq_lengths else 0.0,
        "max": float(np.max(seq_lengths)) if seq_lengths else 0.0,
        "mean": float(np.mean(seq_lengths)) if seq_lengths else 0.0,
        "median": float(np.median(seq_lengths)) if seq_lengths else 0.0,
    }

    # Grade determination
    if empty_rate > 0.5 or malformed_rate > 0.3:
        grade = "WEAK"
    elif malformed_rate > 0.05 or empty_rate > 0.1:
        grade = "PARTIAL"
    elif malformed_rate == 0.0 and empty_rate == 0.0 and temporal_valid_rate == 1.0:
        grade = "VERIFIED"
    else:
        grade = "PARTIALLY_VERIFIED"

    training_eligible = (
        grade in {"VERIFIED", "LINGUIST_REVIEWED"}
        and ctc_feasibility_rate >= 0.95
        and duplicate_rate == 0.0
    )

    return AnnotationQualityReport(
        total_samples=total_samples,
        empty_gloss_samples=empty_count,
        empty_gloss_rate=empty_rate,
        malformed_tokens_count=malformed_tokens,
        malformed_token_rate=malformed_rate,
        duplicate_samples_count=duplicate_count,
        duplicate_sample_rate=duplicate_rate,
        missing_video_count=missing_video,
        missing_video_rate=missing_vid_rate,
        missing_annotation_count=missing_annot,
        missing_annotation_rate=missing_ann_rate,
        unknown_tokens_count=unknown_tokens,
        unknown_token_rate=unknown_rate,
        sequence_length_stats=seq_stats,
        temporal_alignment_validity_rate=temporal_valid_rate,
        ctc_feasible_samples_count=ctc_feasible_count,
        ctc_feasibility_rate=ctc_feasibility_rate,
        overall_quality_grade=grade,
        training_eligible=training_eligible,
        details={
            "total_tokens_audited": total_tokens,
            "unique_samples": len(seen_ids),
        },
    )
