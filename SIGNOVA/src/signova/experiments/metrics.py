"""
Phase 12 Comprehensive Sequence Evaluation Metrics for SIGNOVA.

Calculates:
- Token Error Rate (TER) = (S + I + D) / max(1, N_ref)
- Normalized Edit Distance
- Substitution / Insertion / Deletion counts
- Exact Sequence Match
- Token Precision, Recall, Macro F1
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SequenceMetricsResult:
    ter: float
    normalized_edit_distance: float
    substitutions: int
    insertions: int
    deletions: int
    total_ref_tokens: int
    exact_sequence_match: float
    token_precision: float
    token_recall: float
    macro_f1: float
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ter": round(self.ter, 4),
            "normalized_edit_distance": round(self.normalized_edit_distance, 4),
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "total_ref_tokens": self.total_ref_tokens,
            "exact_sequence_match": round(self.exact_sequence_match, 4),
            "token_precision": round(self.token_precision, 4),
            "token_recall": round(self.token_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "sample_count": self.sample_count,
        }


def compute_levenshtein_breakdown(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """
    Computes Levenshtein distance with exact (substitutions, insertions, deletions, total_dist).
    """
    m, n = len(ref), len(hyp)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    # Backtrack to obtain S, I, D counts
    i, j = m, n
    subs, ins, dels = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            ins += 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            dels += 1
            i -= 1
        else:
            if i > 0:
                dels += 1
                i -= 1
            elif j > 0:
                ins += 1
                j -= 1

    total_dist = dp[m, n]
    return subs, ins, dels, total_dist


def evaluate_sequence_predictions(
    references: List[List[str]],
    hypotheses: List[List[str]],
) -> SequenceMetricsResult:
    """Computes aggregate sequence metrics across reference and hypothesis lists."""
    if not references or len(references) == 0:
        return SequenceMetricsResult(
            ter=0.0,
            normalized_edit_distance=0.0,
            substitutions=0,
            insertions=0,
            deletions=0,
            total_ref_tokens=0,
            exact_sequence_match=1.0,
            token_precision=1.0,
            token_recall=1.0,
            macro_f1=1.0,
            sample_count=0,
        )

    total_s, total_i, total_d = 0, 0, 0
    total_ref_tokens = 0
    exact_matches = 0
    norm_edit_distances = []

    precisions = []
    recalls = []
    f1s = []

    for ref, hyp in zip(references, hypotheses):
        s, i, d, dist = compute_levenshtein_breakdown(ref, hyp)
        total_s += s
        total_i += i
        total_d += d
        total_ref_tokens += len(ref)

        if ref == hyp:
            exact_matches += 1

        norm_dist = dist / max(1, max(len(ref), len(hyp)))
        norm_edit_distances.append(norm_dist)

        # Token set metrics
        ref_set = set(ref)
        hyp_set = set(hyp)
        if not ref_set and not hyp_set:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
        elif not ref_set or not hyp_set:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
        else:
            common = len(ref_set.intersection(hyp_set))
            p = common / len(hyp_set)
            r = common / len(ref_set)
            f = 2 * p * r / max(1e-6, p + r)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

    ter = (total_s + total_i + total_d) / max(1, total_ref_tokens)
    mean_norm_edit = float(np.mean(norm_edit_distances))
    exact_acc = exact_matches / len(references)
    mean_p = float(np.mean(precisions))
    mean_r = float(np.mean(recalls))
    mean_f1 = float(np.mean(f1s))

    return SequenceMetricsResult(
        ter=ter,
        normalized_edit_distance=mean_norm_edit,
        substitutions=total_s,
        insertions=total_i,
        deletions=total_d,
        total_ref_tokens=total_ref_tokens,
        exact_sequence_match=exact_acc,
        token_precision=mean_p,
        token_recall=mean_r,
        macro_f1=mean_f1,
        sample_count=len(references),
    )
