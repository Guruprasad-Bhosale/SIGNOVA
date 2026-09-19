"""
Evaluation metrics and qualitative report generation for sequence-to-sequence translation.

Guiding Principles:
1. Multi-dimensional evaluation: BLEU (1-4), chrF, Token Precision/Recall/F1, Exact Match, Normalized Edit Distance.
2. Synthetic fixture diagnostic labeling vs Real dataset performance reporting.
3. Detailed length statistics and qualitative inspection breakdown.
4. Robust to empty inputs and zero division.
"""

from collections import Counter
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def get_ngrams(tokens: Sequence[str], n: int) -> Counter:
    """Extract n-grams from a token sequence."""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_sentence_bleu(
    hypothesis: Sequence[str],
    reference: Sequence[str],
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """
    Compute sentence-level BLEU score with optional smoothing.
    """
    if len(hypothesis) == 0 or len(reference) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = get_ngrams(hypothesis, n)
        ref_ngrams = get_ngrams(reference, n)
        
        total_hyp = sum(hyp_ngrams.values())
        if total_hyp == 0:
            p_n = 0.0 if not smooth else 1.0 / (len(hypothesis) + 1)
        else:
            clipped_hits = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
            if clipped_hits == 0 and smooth:
                p_n = 1.0 / (2 * total_hyp)
            else:
                p_n = clipped_hits / total_hyp
        precisions.append(p_n)

    if min(precisions) <= 0.0 and not smooth:
        return 0.0

    # Geometric mean
    log_sum = sum(math.log(max(p, 1e-12)) for p in precisions) / max_n
    geo_mean = math.exp(log_sum)

    # Brevity penalty
    c = len(hypothesis)
    r = len(reference)
    bp = 1.0 if c > r else math.exp(1.0 - (r / c)) if c > 0 else 0.0

    return bp * geo_mean * 100.0


def compute_corpus_bleu(
    hypotheses: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
    max_n: int = 4,
) -> Dict[str, float]:
    """
    Compute standard corpus-level BLEU-1, BLEU-2, BLEU-3, and BLEU-4.
    """
    if len(hypotheses) == 0 or len(references) == 0:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    total_hyp_lens = sum(len(h) for h in hypotheses)
    total_ref_lens = sum(len(r) for r in references)

    bp = 1.0 if total_hyp_lens > total_ref_lens else math.exp(1.0 - (total_ref_lens / max(1, total_hyp_lens)))

    results = {}
    p_values = []
    for n in range(1, max_n + 1):
        total_hits = 0
        total_ngrams = 0
        for hyp, ref in zip(hypotheses, references):
            hyp_ngrams = get_ngrams(hyp, n)
            ref_ngrams = get_ngrams(ref, n)
            total_ngrams += sum(hyp_ngrams.values())
            total_hits += sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())

        p_n = (total_hits / total_ngrams) if total_ngrams > 0 else 0.0
        p_values.append(p_n)
        
        # Intermediate cumulative BLEU
        if min(p_values) > 0:
            log_mean = sum(math.log(p) for p in p_values) / len(p_values)
            cum_bleu = bp * math.exp(log_mean) * 100.0
        else:
            cum_bleu = 0.0
        results[f"bleu_{n}"] = round(cum_bleu, 2)

    return results


def compute_chrf(
    hypothesis: str,
    reference: str,
    n: int = 6,
    beta: float = 2.0,
) -> float:
    """
    Compute sentence-level chrF (character n-gram F-score).
    """
    hyp_chars = hypothesis.replace(" ", "")
    ref_chars = reference.replace(" ", "")

    if len(hyp_chars) == 0 or len(ref_chars) == 0:
        return 0.0

    total_prec = 0.0
    total_rec = 0.0

    for order in range(1, n + 1):
        hyp_ngrams = Counter(hyp_chars[i : i + order] for i in range(len(hyp_chars) - order + 1))
        ref_ngrams = Counter(ref_chars[i : i + order] for i in range(len(ref_chars) - order + 1))

        hyp_tot = sum(hyp_ngrams.values())
        ref_tot = sum(ref_ngrams.values())

        hits = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())

        p_order = hits / hyp_tot if hyp_tot > 0 else 0.0
        r_order = hits / ref_tot if ref_tot > 0 else 0.0

        total_prec += p_order
        total_rec += r_order

    avg_p = total_prec / n
    avg_r = total_rec / n

    if avg_p + avg_r == 0:
        return 0.0

    beta_sq = beta ** 2
    f_score = (1 + beta_sq) * (avg_p * avg_r) / ((beta_sq * avg_p) + avg_r)
    return f_score * 100.0


def compute_token_f1(
    hypothesis_tokens: Sequence[str],
    reference_tokens: Sequence[str],
) -> Dict[str, float]:
    """
    Compute token-level precision, recall, and F1.
    """
    hyp_counts = Counter(hypothesis_tokens)
    ref_counts = Counter(reference_tokens)

    hits = sum(min(count, ref_counts[t]) for t, count in hyp_counts.items())
    total_hyp = len(hypothesis_tokens)
    total_ref = len(reference_tokens)

    prec = (hits / total_hyp) if total_hyp > 0 else 0.0
    rec = (hits / total_ref) if total_ref > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "precision": round(prec * 100.0, 2),
        "recall": round(rec * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
    }


def compute_levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> Tuple[int, int, int, int]:
    """
    Compute Levenshtein distance with S/I/D breakdown.
    Returns: (total_distance, substitutions, insertions, deletions)
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    # Backtrace
    i, j = m, n
    subs, ins, dels = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq1[i - 1] == seq2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return dp[m][n], subs, ins, dels


def compute_translation_metrics(
    hypotheses: Sequence[str],
    references: Sequence[str],
    source_gloss_sequences: Optional[Sequence[Sequence[str]]] = None,
    is_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Comprehensive multi-dimensional translation evaluation across a test corpus.
    """
    if len(hypotheses) != len(references):
        raise ValueError(f"Length mismatch: {len(hypotheses)} predictions vs {len(references)} references.")

    total_samples = len(hypotheses)
    if total_samples == 0:
        return {"error": "Empty evaluation batch"}

    hyp_token_lists = [h.strip().split() for h in hypotheses]
    ref_token_lists = [r.strip().split() for r in references]

    # Corpus BLEU
    corpus_bleu = compute_corpus_bleu(hyp_token_lists, ref_token_lists)

    # Sentence-level aggregations
    bleu4_scores = []
    chrf_scores = []
    exact_matches = 0
    total_edit_dist = 0
    total_subs, total_ins, total_dels = 0, 0, 0
    total_ref_len = 0
    total_hyp_len = 0
    total_src_len = 0

    token_f1_scores = []

    for i in range(total_samples):
        h_str = hypotheses[i].strip()
        r_str = references[i].strip()
        h_toks = hyp_token_lists[i]
        r_toks = ref_token_lists[i]

        s_bleu = compute_sentence_bleu(h_toks, r_toks)
        bleu4_scores.append(s_bleu)

        s_chrf = compute_chrf(h_str, r_str)
        chrf_scores.append(s_chrf)

        if h_str.lower() == r_str.lower():
            exact_matches += 1

        dist, s, ins, d = compute_levenshtein_distance(h_toks, r_toks)
        total_edit_dist += dist
        total_subs += s
        total_ins += ins
        total_dels += d

        total_ref_len += len(r_toks)
        total_hyp_len += len(h_toks)
        if source_gloss_sequences is not None:
            total_src_len += len(source_gloss_sequences[i])

        f1_dict = compute_token_f1(h_toks, r_toks)
        token_f1_scores.append(f1_dict["f1"])

    avg_bleu4 = sum(bleu4_scores) / total_samples
    avg_chrf = sum(chrf_scores) / total_samples
    exact_match_pct = (exact_matches / total_samples) * 100.0
    avg_token_f1 = sum(token_f1_scores) / total_samples
    norm_edit_dist = (total_edit_dist / max(1, total_ref_len)) * 100.0

    report: Dict[str, Any] = {
        "is_synthetic_fixture": is_synthetic,
        "evaluation_disclaimer": (
            "SYNTHETIC FIXTURE — NOT REAL ISL TRANSLATION PERFORMANCE"
            if is_synthetic
            else "REAL TRANSLATION DATASET EVALUATION"
        ),
        "total_samples": total_samples,
        "exact_match_count": exact_matches,
        "exact_match_percentage": round(exact_match_pct, 2),
        "corpus_bleu_1": corpus_bleu["bleu_1"],
        "corpus_bleu_2": corpus_bleu["bleu_2"],
        "corpus_bleu_3": corpus_bleu["bleu_3"],
        "corpus_bleu_4": corpus_bleu["bleu_4"],
        "sentence_bleu_4_mean": round(avg_bleu4, 2),
        "chrf_mean": round(avg_chrf, 2),
        "token_f1_mean": round(avg_token_f1, 2),
        "normalized_edit_distance": round(norm_edit_dist, 2),
        "length_statistics": {
            "avg_reference_length": round(total_ref_len / total_samples, 2),
            "avg_predicted_length": round(total_hyp_len / total_samples, 2),
            "length_ratio": round(total_hyp_len / max(1, total_ref_len), 3),
            "total_substitutions": total_subs,
            "total_insertions": total_ins,
            "total_deletions": total_dels,
        },
    }

    if source_gloss_sequences is not None:
        report["length_statistics"]["avg_source_gloss_length"] = round(total_src_len / total_samples, 2)

    return report


def format_qualitative_table(
    source_glosses: Sequence[Sequence[str]],
    references: Sequence[str],
    hypotheses: Sequence[str],
    max_rows: int = 15,
) -> str:
    """Format qualitative table with Markdown columns."""
    lines = [
        "| # | Source ISL Gloss Sequence | Reference English | Predicted English | Sentence BLEU | Exact Match |",
        "| :--- | :--- | :--- | :--- | :---: | :---: |",
    ]

    limit = min(len(source_glosses), max_rows)
    for i in range(limit):
        src_str = "[" + ", ".join(f'"{g}"' for g in source_glosses[i]) + "]"
        ref_str = references[i].strip()
        hyp_str = hypotheses[i].strip()
        s_bleu = round(compute_sentence_bleu(hyp_str.split(), ref_str.split()), 1)
        match_icon = "MATCH" if hyp_str.lower() == ref_str.lower() else "DIFF"
        lines.append(f"| {i+1} | `{src_str}` | {ref_str} | {hyp_str} | {s_bleu}% | `{match_icon}` |")

    return "\n".join(lines)
