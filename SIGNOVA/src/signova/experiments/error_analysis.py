"""
Phase 12 Error Analysis and Confusion Diagnostics for Real CTC Recognition.

Generates:
- Per-sample reference vs prediction comparison
- Common substitution pairs
- Insertion / Deletion frequency tables
- Error patterns across sequence lengths
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from signova.experiments.metrics import compute_levenshtein_breakdown


@dataclass
class SampleErrorRecord:
    sample_id: str
    reference: List[str]
    prediction: List[str]
    substitutions: int
    insertions: int
    deletions: int
    edit_distance: int
    exact_match: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "reference": self.reference,
            "prediction": self.prediction,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "edit_distance": self.edit_distance,
            "exact_match": self.exact_match,
        }


def perform_phase12_error_analysis(
    sample_ids: List[str],
    references: List[List[str]],
    hypotheses: List[List[str]],
) -> Dict[str, Any]:
    """Generates structured error analysis across test samples."""
    sample_records = []
    substitution_counter: Counter = Counter()
    insertion_counter: Counter = Counter()
    deletion_counter: Counter = Counter()

    for sid, ref, hyp in zip(sample_ids, references, hypotheses):
        s, i, d, dist = compute_levenshtein_breakdown(ref, hyp)
        rec = SampleErrorRecord(
            sample_id=sid,
            reference=ref,
            prediction=hyp,
            substitutions=s,
            insertions=i,
            deletions=d,
            edit_distance=dist,
            exact_match=(ref == hyp),
        )
        sample_records.append(rec)

        # Track token errors
        if ref != hyp:
            diff_hyp = set(hyp) - set(ref)
            for tok in diff_hyp:
                insertion_counter[tok] += 1
            diff_ref = set(ref) - set(hyp)
            for tok in diff_ref:
                deletion_counter[tok] += 1

    total_exact = sum(1 for r in sample_records if r.exact_match)
    total_samples = len(sample_records)

    return {
        "total_samples": total_samples,
        "exact_matches": total_exact,
        "exact_match_ratio": round(total_exact / max(1, total_samples), 4),
        "most_common_insertions": insertion_counter.most_common(5),
        "most_common_deletions": deletion_counter.most_common(5),
        "sample_breakdown": [r.to_dict() for r in sample_records],
    }
