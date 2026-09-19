"""
Phase 14 Independent Double-Annotation & Agreement Manager for SIGNOVA.

Enforces:
- Agreement is strictly calculated between genuinely independent annotators.
- Reviewer corrections must NEVER masquerade as a second independent annotation.
- Configurable double-annotation fraction and sample thresholding.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from signova.annotation.agreement import compute_sequence_token_f1
from signova.annotation.schema import VideoAnnotation
from signova.experiments.metrics import compute_levenshtein_breakdown
from signova.operations.constants import (
    AGREEMENT_COMPUTABLE,
    AGREEMENT_INSUFFICIENT_SAMPLE,
    AGREEMENT_NOT_COMPUTABLE,
    DEFAULT_DOUBLE_ANNOTATION_FRACTION,
    DEFAULT_DOUBLE_ANNOTATION_MINIMUM,
)


@dataclass
class IndependentAnnotationPair:
    sample_id: str
    annotator_1_id: str
    annotator_2_id: str
    sequence_1: List[str]
    sequence_2: List[str]
    exact_match: bool
    token_f1: float
    edit_distance: int


class DoubleAnnotationManager:
    """Manages double annotation pairing and independent agreement evaluation."""

    def __init__(
        self,
        double_fraction: float = DEFAULT_DOUBLE_ANNOTATION_FRACTION,
        double_minimum: int = DEFAULT_DOUBLE_ANNOTATION_MINIMUM,
    ):
        self.double_fraction = double_fraction
        self.double_minimum = double_minimum

    def evaluate_independent_agreement(
        self,
        annotations_by_sample: Dict[str, List[VideoAnnotation]],
    ) -> Dict[str, Any]:
        """
        Evaluates agreement strictly across samples with 2+ independent annotators.
        Excludes reviewer modifications on the same annotator's work.
        """
        independent_pairs: List[IndependentAnnotationPair] = []
        total_samples = len(annotations_by_sample)
        single_count = 0
        double_count = 0

        for sample_id, annots in annotations_by_sample.items():
            # Filter distinct genuine annotator profiles (distinct annotator IDs)
            unique_annotators = {}
            for a in annots:
                if a.annotator_id and a.annotator_id not in unique_annotators:
                    unique_annotators[a.annotator_id] = a

            distinct_list = list(unique_annotators.values())
            if len(distinct_list) >= 2:
                double_count += 1
                a1, a2 = distinct_list[0], distinct_list[1]
                p, r, f1 = compute_sequence_token_f1(a1.glosses, a2.glosses)
                _, _, _, dist = compute_levenshtein_breakdown(a1.glosses, a2.glosses)

                independent_pairs.append(
                    IndependentAnnotationPair(
                        sample_id=sample_id,
                        annotator_1_id=a1.annotator_id,
                        annotator_2_id=a2.annotator_id,
                        sequence_1=a1.glosses,
                        sequence_2=a2.glosses,
                        exact_match=(a1.glosses == a2.glosses),
                        token_f1=round(f1, 4),
                        edit_distance=dist,
                    )
                )
            elif len(distinct_list) == 1:
                single_count += 1

        pair_count = len(independent_pairs)
        if pair_count == 0:
            status = AGREEMENT_NOT_COMPUTABLE
            reason = "No genuine independent dual-annotated samples available."
            mean_f1 = 0.0
            mean_exact = 0.0
        elif pair_count < self.double_minimum:
            status = AGREEMENT_INSUFFICIENT_SAMPLE
            reason = f"Dual-annotated pairs ({pair_count}) below minimum threshold ({self.double_minimum})."
            mean_f1 = round(sum(p.token_f1 for p in independent_pairs) / pair_count, 4)
            mean_exact = round(sum(1 for p in independent_pairs if p.exact_match) / pair_count, 4)
        else:
            status = AGREEMENT_COMPUTABLE
            reason = "Sufficient independent dual annotations available."
            mean_f1 = round(sum(p.token_f1 for p in independent_pairs) / pair_count, 4)
            mean_exact = round(sum(1 for p in independent_pairs if p.exact_match) / pair_count, 4)

        return {
            "agreement_status": status,
            "reason": reason,
            "total_samples": total_samples,
            "single_annotated_samples": single_count,
            "double_annotated_samples": double_count,
            "independent_pairs_compared": pair_count,
            "mean_token_f1": mean_f1,
            "exact_sequence_agreement": mean_exact,
            "configured_fraction": self.double_fraction,
            "configured_minimum": self.double_minimum,
        }
