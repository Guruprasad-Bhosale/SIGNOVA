"""
Unit tests for translation metrics (BLEU, chrF, Token F1, Levenshtein distance, tables).
"""

from signova.translation.metrics import (
    compute_chrf,
    compute_corpus_bleu,
    compute_levenshtein_distance,
    compute_sentence_bleu,
    compute_token_f1,
    compute_translation_metrics,
    format_qualitative_table,
)


def test_sentence_and_corpus_bleu():
    hyp = ["i", "am", "going", "to", "school"]
    ref = ["i", "am", "going", "to", "school"]

    # Perfect match
    bleu_perfect = compute_sentence_bleu(hyp, ref)
    assert round(bleu_perfect, 2) == 100.0

    # Partial match
    hyp_partial = ["i", "am", "walking", "to", "school"]
    bleu_partial = compute_sentence_bleu(hyp_partial, ref)
    assert 0.0 < bleu_partial < 100.0

    # Corpus BLEU
    corpus_res = compute_corpus_bleu([hyp, hyp_partial], [ref, ref])
    assert "bleu_1" in corpus_res
    assert "bleu_4" in corpus_res


def test_chrf_and_token_f1():
    hyp_text = "I am going to school."
    ref_text = "I am going to school."

    chrf_score = compute_chrf(hyp_text, ref_text)
    assert round(chrf_score, 1) == 100.0

    f1_res = compute_token_f1(["i", "go"], ["i", "go"])
    assert f1_res["f1"] == 100.0
    assert f1_res["precision"] == 100.0
    assert f1_res["recall"] == 100.0


def test_levenshtein_alignment():
    seq1 = ["i", "eat", "food"]
    seq2 = ["i", "ate", "good", "food"]
    dist, subs, ins, dels = compute_levenshtein_distance(seq1, seq2)
    assert dist == 2  # 'eat' -> 'ate' (sub), insert 'good'


def test_compute_translation_metrics_and_qualitative_table():
    hypotheses = ["I go to college today.", "What is your name?"]
    references = ["I go to college today.", "What is your name?"]
    source_glosses = [["I", "GO", "COLLEGE", "TODAY"], ["YOU", "NAME", "WHAT"]]

    metrics = compute_translation_metrics(
        hypotheses=hypotheses,
        references=references,
        source_gloss_sequences=source_glosses,
        is_synthetic=True,
    )

    assert metrics["is_synthetic_fixture"] is True
    assert "SYNTHETIC FIXTURE" in metrics["evaluation_disclaimer"]
    assert metrics["exact_match_count"] == 2
    assert metrics["exact_match_percentage"] == 100.0
    assert metrics["corpus_bleu_4"] > 90.0

    table_str = format_qualitative_table(source_glosses, references, hypotheses)
    assert "| # | Source ISL Gloss Sequence |" in table_str
    assert "COLLEGE" in table_str
