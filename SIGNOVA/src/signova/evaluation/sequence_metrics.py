"""
Continuous Sequence Evaluation Metrics for SIGNOVA.

Calculates comprehensive sign/gloss sequence metrics:
- Token Error Rate (TER / CER-like edit rate)
- Substitutions (S), Deletions (D), Insertions (I) breakdown
- Sequence Exact Match accuracy
- Token-level precision, recall, and Macro/Weighted F1 score
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def compute_levenshtein_alignment(
    ref: List[Union[str, int]],
    hyp: List[Union[str, int]],
) -> Tuple[int, int, int, int]:
    """
    Computes Levenshtein edit distance and counts of Substitutions, Deletions, Insertions.

    Returns:
        (total_edit_distance, substitutions, deletions, insertions)
    """
    r_len, h_len = len(ref), len(hyp)
    if r_len == 0:
        return (h_len, 0, 0, h_len)
    if h_len == 0:
        return (r_len, 0, r_len, 0)

    dp = np.zeros((r_len + 1, h_len + 1), dtype=int)
    for i in range(r_len + 1):
        dp[i, 0] = i
    for j in range(h_len + 1):
        dp[0, j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(
                    dp[i - 1, j],     # Deletion
                    dp[i, j - 1],     # Insertion
                    dp[i - 1, j - 1]  # Substitution
                )

    # Backtrace
    i, j = r_len, h_len
    subs, dels, ins = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            ins += 1
            j -= 1
        else:
            break

    return (dp[r_len, h_len], subs, dels, ins)


def compute_sequence_metrics(
    references: List[List[Union[str, int]]],
    hypotheses: List[List[Union[str, int]]],
) -> Dict[str, Any]:
    """
    Evaluates batched sequence predictions against ground-truth references.

    Args:
        references: List of reference token lists.
        hypotheses: List of predicted token lists.

    Returns:
        Dictionary of sequence metrics.
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match.")

    total_samples = len(references)
    if total_samples == 0:
        return {"error": "Empty evaluation lists provided."}

    total_ref_tokens = sum(len(r) for r in references)
    total_hyp_tokens = sum(len(h) for h in hypotheses)
    total_edit_distance = 0
    total_subs = 0
    total_dels = 0
    total_ins = 0
    exact_matches = 0

    all_ref_tokens = []
    all_hyp_tokens = []

    for ref, hyp in zip(references, hypotheses):
        dist, s, d, i = compute_levenshtein_alignment(ref, hyp)
        total_edit_distance += dist
        total_subs += s
        total_dels += d
        total_ins += i
        if ref == hyp:
            exact_matches += 1

        all_ref_tokens.extend([str(t) for t in ref])
        all_hyp_tokens.extend([str(t) for t in hyp])

    ter = float(total_edit_distance) / max(1, total_ref_tokens)
    exact_match_rate = float(exact_matches) / total_samples

    # Token-level Precision, Recall, F1
    ref_counts = Counter(all_ref_tokens)
    hyp_counts = Counter(all_hyp_tokens)
    all_vocab = set(ref_counts.keys()).union(set(hyp_counts.keys()))

    precisions = []
    recalls = []
    f1s = []

    for token in all_vocab:
        tp = min(ref_counts.get(token, 0), hyp_counts.get(token, 0))
        fp = max(0, hyp_counts.get(token, 0) - tp)
        fn = max(0, ref_counts.get(token, 0) - tp)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "total_sequences": total_samples,
        "total_reference_tokens": total_ref_tokens,
        "total_hypothesis_tokens": total_hyp_tokens,
        "token_error_rate": round(ter, 4),
        "exact_match_rate": round(exact_match_rate, 4),
        "exact_matches_count": exact_matches,
        "substitutions": total_subs,
        "deletions": total_dels,
        "insertions": total_ins,
        "total_edit_distance": total_edit_distance,
        "token_macro_precision": round(macro_prec, 4),
        "token_macro_recall": round(macro_rec, 4),
        "token_macro_f1": round(macro_f1, 4),
    }
