"""
Text analytics script for ISLTranslate.
"""

import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

def compute_translation_stats():
    csv_path = Path("g:/SingLang/ISLTranslate-main/data/ISLTranslate.csv")
    df = pd.read_csv(csv_path, dtype=str)
    valid_texts = df["text"].dropna().astype(str).tolist()
    valid_texts = [t.strip() for t in valid_texts if t.strip()]

    words_all = []
    char_lengths = []
    word_counts = []

    for text in valid_texts:
        char_lengths.append(len(text))
        tokens = re.findall(r"\b[a-zA-Z0-9'-]+\b", text.lower())
        word_counts.append(len(tokens))
        words_all.extend(tokens)

    word_freq = Counter(words_all)
    vocab_size = len(word_freq)
    total_words = len(words_all)

    unique_sentences = len(set(valid_texts))
    sentence_freq = Counter(valid_texts)
    dup_sentences = {k: v for k, v in sentence_freq.items() if v > 1}

    stats = {
        "total_samples": len(df),
        "valid_sentence_count": len(valid_texts),
        "unique_sentence_count": unique_sentences,
        "total_token_count": total_words,
        "vocabulary_size": vocab_size,
        "tokens_per_sentence": {
            "mean": round(float(np.mean(word_counts)), 2),
            "std": round(float(np.std(word_counts)), 2),
            "median": int(np.median(word_counts)),
            "min": int(np.min(word_counts)),
            "max": int(np.max(word_counts)),
            "p25": int(np.percentile(word_counts, 25)),
            "p75": int(np.percentile(word_counts, 75)),
            "p90": int(np.percentile(word_counts, 90)),
            "p95": int(np.percentile(word_counts, 95)),
            "p99": int(np.percentile(word_counts, 99)),
        },
        "character_lengths": {
            "mean": round(float(np.mean(char_lengths)), 2),
            "median": int(np.median(char_lengths)),
            "min": int(np.min(char_lengths)),
            "max": int(np.max(char_lengths)),
            "p95": int(np.percentile(char_lengths, 95)),
        },
        "top_30_frequent_words": word_freq.most_common(30),
        "single_occurrence_words_count": sum(1 for _, c in word_freq.items() if c == 1),
        "words_count_le_5": sum(1 for _, c in word_freq.items() if c <= 5),
        "duplicate_sentence_strings_count": len(dup_sentences),
        "total_duplicate_sentence_instances": sum(dup_sentences.values()),
    }

    out_file = Path("g:/SingLang/SIGNOVA/outputs/reports/translation_statistics.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Computed stats for {len(valid_texts)} sentences. Vocabulary size: {vocab_size}")
    return stats

if __name__ == "__main__":
    compute_translation_stats()
