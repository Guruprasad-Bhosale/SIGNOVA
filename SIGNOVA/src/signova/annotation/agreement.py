"""
Phase 11 Inter-Annotator Agreement and Disagreement Taxonomy Module for SIGNOVA.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from signova.annotation.constants import (
    ALL_DISAGREEMENT_TYPES,
    DISAGREEMENT_EXTRA_SIGN,
    DISAGREEMENT_OMITTED_SIGN,
    DISAGREEMENT_OTHER,
    DISAGREEMENT_SEGMENT_BOUNDARY,
    DISAGREEMENT_TOKEN_IDENTITY,
    DISAGREEMENT_TOKEN_ORDER,
    DISAGREEMENT_UNCERTAIN_SIGN,
    STATUS_BLOCKED_HUMAN_RESOURCE,
)
from signova.annotation.schema import VideoAnnotation


@dataclass
class AnnotationDisagreementRecord:
    sample_id: str
    annotator_a_id: str
    annotator_b_id: str
    annotation_a_glosses: List[str]
    annotation_b_glosses: List[str]
    disagreement_type: str
    details: str
    resolution_status: str = "OPEN"
    adjudicated_glosses: Optional[List[str]] = None
    adjudicated_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "annotator_a_id": self.annotator_a_id,
            "annotator_b_id": self.annotator_b_id,
            "annotation_a_glosses": self.annotation_a_glosses,
            "annotation_b_glosses": self.annotation_b_glosses,
            "disagreement_type": self.disagreement_type,
            "details": self.details,
            "resolution_status": self.resolution_status,
            "adjudicated_glosses": self.adjudicated_glosses,
            "adjudicated_by": self.adjudicated_by,
        }


def classify_phase11_disagreement(
    sample_id: str,
    annot_a: VideoAnnotation,
    annot_b: VideoAnnotation,
) -> AnnotationDisagreementRecord:
    """Classifies linguistic disagreement between two independent annotators."""
    seq_a = annot_a.glosses
    seq_b = annot_b.glosses

    # Check omitted or extra sign
    if len(seq_a) < len(seq_b) and set(seq_a).issubset(set(seq_b)):
        dtype = DISAGREEMENT_OMITTED_SIGN
        details = f"Annotator A omitted signs present in Annotator B: {set(seq_b) - set(seq_a)}"
    elif len(seq_a) > len(seq_b) and set(seq_b).issubset(set(seq_a)):
        dtype = DISAGREEMENT_EXTRA_SIGN
        details = f"Annotator A included extra signs absent in Annotator B: {set(seq_a) - set(seq_b)}"
    # Check token order
    elif sorted(seq_a) == sorted(seq_b) and seq_a != seq_b:
        dtype = DISAGREEMENT_TOKEN_ORDER
        details = f"Same vocabulary tokens produced in different sequence orders."
    # Check boundary if temporally aligned
    elif annot_a.is_temporally_aligned and annot_b.is_temporally_aligned:
        dtype = DISAGREEMENT_SEGMENT_BOUNDARY
        details = "Differences in segment frame/timestamp boundaries."
    else:
        dtype = DISAGREEMENT_TOKEN_IDENTITY
        details = f"Lexical identity divergence: {seq_a} vs {seq_b}"

    return AnnotationDisagreementRecord(
        sample_id=sample_id,
        annotator_a_id=annot_a.annotator_id,
        annotator_b_id=annot_b.annotator_id,
        annotation_a_glosses=seq_a,
        annotation_b_glosses=seq_b,
        disagreement_type=dtype,
        details=details,
    )


def compute_sequence_token_f1(seq1: List[str], seq2: List[str]) -> Tuple[float, float, float]:
    """Computes Token Precision, Recall, and F1 between two sequences."""
    set1, set2 = set(seq1), set(seq2)
    if not set1 and not set2:
        return 1.0, 1.0, 1.0
    if not set1 or not set2:
        return 0.0, 0.0, 0.0

    common = len(set1.intersection(set2))
    precision = common / len(set1)
    recall = common / len(set2)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    return precision, recall, f1
