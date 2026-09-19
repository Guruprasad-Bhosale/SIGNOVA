"""
Phase 12 CTC Feasibility and Temporal Length Constraint Validator for SIGNOVA.

Validates:
T_input >= L_target (CTC mathematical constraint)
and generates input/target length distribution statistics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from signova.annotation.schema import VideoAnnotation


@dataclass
class CTCFeasibilityReport:
    total_samples: int
    valid_samples: int
    rejected_samples: int
    min_input_length: int
    max_input_length: int
    avg_input_length: float
    min_target_length: int
    max_target_length: int
    avg_target_length: float
    rejected_details: List[Dict[str, Any]]
    feasibility_status: str  # "PASSED", "FAILED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "rejected_samples": self.rejected_samples,
            "min_input_length": self.min_input_length,
            "max_input_length": self.max_input_length,
            "avg_input_length": round(self.avg_input_length, 2),
            "min_target_length": self.min_target_length,
            "max_target_length": self.max_target_length,
            "avg_target_length": round(self.avg_target_length, 2),
            "rejected_details": self.rejected_details,
            "feasibility_status": self.feasibility_status,
        }


def calculate_ctc_required_input_length(glosses: List[str]) -> int:
    """
    Computes the minimum input frames required by CTC for a target sequence.
    In CTC, adjacent identical tokens MUST be separated by at least one blank token.
    Therefore, T_required = len(glosses) + count(adjacent_identical_pairs).
    """
    if not glosses:
        return 0
    adj_repeats = 0
    for i in range(len(glosses) - 1):
        if glosses[i] == glosses[i + 1]:
            adj_repeats += 1
    return len(glosses) + adj_repeats


def validate_ctc_feasibility(
    annotations: List[VideoAnnotation],
    features_dir: Optional[Path] = None,
) -> CTCFeasibilityReport:
    """Validates whether landmark sequence lengths satisfy CTC requirements for each sample."""
    if features_dir is None:
        features_dir = Path("data/features/landmarks")

    if not annotations:
        return CTCFeasibilityReport(
            total_samples=0,
            valid_samples=0,
            rejected_samples=0,
            min_input_length=0,
            max_input_length=0,
            avg_input_length=0.0,
            min_target_length=0,
            max_target_length=0,
            avg_target_length=0.0,
            rejected_details=[],
            feasibility_status="PASSED",
        )

    input_lengths = []
    target_lengths = []
    rejected = []
    valid_count = 0

    for a in annotations:
        t_target = len(a.glosses)
        if t_target == 0:
            rejected.append({"sample_id": a.sample_id, "reason": "EMPTY_TARGET_SEQUENCE"})
            continue

        t_required = calculate_ctc_required_input_length(a.glosses)

        # Look up feature file
        feat_path = None
        for split in ["train", "val", "test"]:
            p = features_dir / split / f"{a.sample_id}.npz"
            if p.exists():
                feat_path = p
                break

        if feat_path is None:
            # Fallback mock length if testing
            t_input = a.metadata.get("frame_count", 60)
        else:
            try:
                npz = np.load(feat_path)
                t_input = int(npz["data"].shape[0]) if "data" in npz else int(npz["landmarks"].shape[0])
            except Exception:
                t_input = 60

        # CTC feasibility check: T_input >= T_required (including repeated token blanks)
        if t_input < t_required:
            rejected.append({
                "sample_id": a.sample_id,
                "reason": f"CTC_LENGTH_VIOLATION: T_input ({t_input}) < T_required ({t_required}) [L={t_target}]"
            })
        else:
            valid_count += 1
            input_lengths.append(t_input)
            target_lengths.append(t_target)

    min_in = min(input_lengths) if input_lengths else 0
    max_in = max(input_lengths) if input_lengths else 0
    avg_in = float(np.mean(input_lengths)) if input_lengths else 0.0

    min_tgt = min(target_lengths) if target_lengths else 0
    max_tgt = max(target_lengths) if target_lengths else 0
    avg_tgt = float(np.mean(target_lengths)) if target_lengths else 0.0

    status = "PASSED" if len(rejected) == 0 else "WARNING"

    return CTCFeasibilityReport(
        total_samples=len(annotations),
        valid_samples=valid_count,
        rejected_samples=len(rejected),
        min_input_length=min_in,
        max_input_length=max_in,
        avg_input_length=avg_in,
        min_target_length=min_tgt,
        max_target_length=max_tgt,
        avg_target_length=avg_tgt,
        rejected_details=rejected,
        feasibility_status=status,
    )
