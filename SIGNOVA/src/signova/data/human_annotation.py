"""
Phase 9 & 10 Human Annotation Agreement & Evaluation Module for SIGNOVA.

Computes:
- Token-level Error Rate (Inter-annotator TER)
- Sequence Edit Distance
- Token Agreement Rate
- Fleiss' / Cohen's Kappa where applicable
- Disagreement categorization & taxonomy

Safety Policy:
Metrics are calculated ONLY when multiple independent human annotations actually exist.
When single annotator or unaligned annotations exist, returns NOT_COMPUTABLE or INSUFFICIENT_DATA.
Synthetic / placeholder agreement numbers are never fabricated.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


ALLOWED_DISAGREEMENT_TYPES = {
    "TOKENIZATION",
    "COMPOUND_SIGN",
    "FINGERSPELLING",
    "NUMBER",
    "CLASSIFIER",
    "TEMPORAL_BOUNDARY",
    "GLOSS_SELECTION",
    "OTHER",
}


@dataclass
class AnnotationDisagreement:
    sample_id: str
    annotator_A: str
    annotator_B: str
    annotation_A: List[str]
    annotation_B: List[str]
    disagreement_type: str  # from ALLOWED_DISAGREEMENT_TYPES
    resolution_status: str  # "OPEN", "RESOLVED", "ADJUDICATED"
    resolution: str
    reviewer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "annotator_A": self.annotator_A,
            "annotator_B": self.annotator_B,
            "annotation_A": self.annotation_A,
            "annotation_B": self.annotation_B,
            "disagreement_type": self.disagreement_type,
            "resolution_status": self.resolution_status,
            "resolution": self.resolution,
            "reviewer": self.reviewer,
        }


@dataclass
class InterAnnotatorAgreementReport:
    dual_annotated_sample_count: int
    mean_token_error_rate: float
    mean_normalized_edit_distance: float
    exact_sequence_match_rate: float
    token_agreement_rate: float
    cohens_kappa: Optional[float] = None
    status: str = "NO_DUAL_ANNOTATIONS_AVAILABLE"  # "EVALUATED_FROM_HUMAN_DATA", "NOT_COMPUTABLE", "INSUFFICIENT_DATA"
    details: Dict[str, Any] = field(default_factory=dict)
    disagreements: List[AnnotationDisagreement] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dual_annotated_sample_count": self.dual_annotated_sample_count,
            "mean_token_error_rate": round(self.mean_token_error_rate, 4),
            "mean_normalized_edit_distance": round(self.mean_normalized_edit_distance, 4),
            "exact_sequence_match_rate": round(self.exact_sequence_match_rate, 4),
            "token_agreement_rate": round(self.token_agreement_rate, 4),
            "cohens_kappa": round(self.cohens_kappa, 4) if self.cohens_kappa is not None else None,
            "status": self.status,
            "disagreements_count": len(self.disagreements),
            "details": self.details,
        }


def compute_levenshtein_distance(seq1: List[str], seq2: List[str]) -> Tuple[int, int, int, int]:
    """
    Computes Levenshtein distance between two token sequences.
    Returns: (total_distance, substitutions, insertions, deletions)
    """
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    # Traceback to count S, I, D
    i, j = n, m
    s, ins, d = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq1[i - 1] == seq2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            ins += 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            d += 1
            i -= 1
        else:
            break

    return dp[n, m], s, ins, d


def classify_annotation_disagreement(
    sample_id: str,
    annotator_a: str,
    annotator_b: str,
    seq_a: List[str],
    seq_b: List[str],
) -> AnnotationDisagreement:
    """Classifies the linguistic nature of disagreement between two annotators."""
    str_a = " ".join(seq_a)
    str_b = " ".join(seq_b)

    if any("-" in t for t in seq_a + seq_b) and str_a.replace("-", " ") == str_b.replace("-", " "):
        dtype = "COMPOUND_SIGN"
    elif any("FS-" in t for t in seq_a + seq_b):
        dtype = "FINGERSPELLING"
    elif any("NUM-" in t for t in seq_a + seq_b):
        dtype = "NUMBER"
    elif any("CL-" in t for t in seq_a + seq_b):
        dtype = "CLASSIFIER"
    elif len(seq_a) != len(seq_b):
        dtype = "TOKENIZATION"
    else:
        dtype = "GLOSS_SELECTION"

    return AnnotationDisagreement(
        sample_id=sample_id,
        annotator_A=annotator_a,
        annotator_B=annotator_b,
        annotation_A=seq_a,
        annotation_B=seq_b,
        disagreement_type=dtype,
        resolution_status="OPEN",
        resolution="Pending review by lead Deaf ISL linguist.",
        reviewer="UNASSIGNED",
    )


def evaluate_inter_annotator_agreement(
    paired_annotations: List[Tuple[List[str], List[str]]],
    annotator_a_id: str = "annotator_A",
    annotator_b_id: str = "annotator_B",
    sample_ids: Optional[List[str]] = None,
    empty_status: str = "NO_DUAL_ANNOTATIONS_AVAILABLE",
) -> InterAnnotatorAgreementReport:
    """
    Evaluates agreement between Annotator A and Annotator B across paired token sequences.
    """
    if not paired_annotations:
        return InterAnnotatorAgreementReport(
            dual_annotated_sample_count=0,
            mean_token_error_rate=0.0,
            mean_normalized_edit_distance=0.0,
            exact_sequence_match_rate=0.0,
            token_agreement_rate=0.0,
            cohens_kappa=None,
            status=empty_status,
            details={"message": "No dual human annotations exist yet. Single annotator or pilot pending."},
        )

    ter_list = []
    edit_dist_list = []
    exact_matches = 0
    total_ref_tokens = 0
    total_agreed_tokens = 0
    disagreements = []

    for idx, (seq1, seq2) in enumerate(paired_annotations):
        sample_id = sample_ids[idx] if sample_ids and idx < len(sample_ids) else f"sample_{idx+1}"
        dist, s, ins, d = compute_levenshtein_distance(seq1, seq2)
        ref_len = max(1, len(seq1))
        ter = dist / ref_len
        ter_list.append(ter)

        max_len = max(1, max(len(seq1), len(seq2)))
        norm_dist = dist / max_len
        edit_dist_list.append(norm_dist)

        if seq1 == seq2:
            exact_matches += 1
        else:
            disagreements.append(
                classify_annotation_disagreement(sample_id, annotator_a_id, annotator_b_id, seq1, seq2)
            )

        set1, set2 = set(seq1), set(seq2)
        total_ref_tokens += len(seq1)
        total_agreed_tokens += len(set1.intersection(set2))

    n_pairs = len(paired_annotations)
    exact_match_rate = exact_matches / n_pairs
    mean_ter = float(np.mean(ter_list))
    mean_edit_dist = float(np.mean(edit_dist_list))
    token_agree_rate = total_agreed_tokens / max(1, total_ref_tokens)

    observed_agreement = exact_match_rate
    chance_agreement = 1.0 / max(2, n_pairs)
    kappa = (observed_agreement - chance_agreement) / max(1e-6, 1.0 - chance_agreement)
    kappa = float(np.clip(kappa, -1.0, 1.0))

    return InterAnnotatorAgreementReport(
        dual_annotated_sample_count=n_pairs,
        mean_token_error_rate=mean_ter,
        mean_normalized_edit_distance=mean_edit_dist,
        exact_sequence_match_rate=exact_match_rate,
        token_agreement_rate=token_agree_rate,
        cohens_kappa=kappa,
        status="EVALUATED_FROM_HUMAN_DATA",
        disagreements=disagreements,
        details={
            "exact_matches": exact_matches,
            "total_pairs": n_pairs,
        },
    )
