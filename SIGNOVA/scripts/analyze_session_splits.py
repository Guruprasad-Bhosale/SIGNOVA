"""
Signer and Session Overlap Analysis Script for ISLTranslate.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
import pandas as pd


def analyze_session_splits():
    manifest_path = Path("g:/SingLang/SIGNOVA/data/manifests/isltranslate_manifest.csv")
    df = pd.read_csv(manifest_path, dtype=str)

    # Extract source session/video prefix (part before first hyphen or double hyphen)
    def extract_prefix(uid: str) -> str:
        s = str(uid).split("_dup")[0]
        if "--" in s:
            return s.split("--")[0]
        elif "-" in s:
            return s.rsplit("-", 1)[0]
        return s

    df["session_prefix"] = df["sample_id"].apply(extract_prefix)
    unique_prefixes = set(df["session_prefix"])

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_sessions: Set[str] = set(train_df["session_prefix"])
    val_sessions: Set[str] = set(val_df["session_prefix"])
    test_sessions: Set[str] = set(test_df["session_prefix"])

    overlap_train_val = train_sessions.intersection(val_sessions)
    overlap_train_test = train_sessions.intersection(test_sessions)
    overlap_val_test = val_sessions.intersection(test_sessions)

    # 1. Signer / Session Split Analysis JSON
    analysis = {
        "signer_metadata_status": "UNKNOWN — NOT EXPLICITLY ANNOTATED IN RAW CSV",
        "verified_signer_id_available": False,
        "source_session_prefix_available": True,
        "total_samples": len(df),
        "total_unique_source_sessions": len(unique_prefixes),
        "official_split_distribution": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "official_split_session_overlap": {
            "train_sessions_count": len(train_sessions),
            "val_sessions_count": len(val_sessions),
            "test_sessions_count": len(test_sessions),
            "train_val_overlap_count": len(overlap_train_val),
            "train_test_overlap_count": len(overlap_train_test),
            "val_test_overlap_count": len(overlap_val_test),
            "session_leakage_present_in_random_split": len(overlap_train_test) > 0,
        },
        "research_recommendation": (
            "Because ISLTranslate.csv contains segmented sentences from continuous YouTube/educational video sessions, "
            "random sample hashing distributes segments of the same session across train and test. "
            "For rigorous research, a Session-Independent Split is proposed as a benchmark."
        ),
    }

    report_out = Path("g:/SingLang/SIGNOVA/outputs/reports/signer_split_analysis.json")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    # 2. Build Proposed Session/Signer-Independent Split (Analysis artifact)
    session_list = sorted(list(unique_prefixes))
    train_sessions_prop = set()
    val_sessions_prop = set()
    test_sessions_prop = set()

    for sess in session_list:
        h = hashlib.sha256(f"signova_session_independent_{sess}".encode("utf-8")).hexdigest()
        val = int(h[:8], 16) / 0xFFFFFFFF
        if val < 0.8:
            train_sessions_prop.add(sess)
        elif val < 0.9:
            val_sessions_prop.add(sess)
        else:
            test_sessions_prop.add(sess)

    proposed_records = []
    for _, row in df.iterrows():
        sess = row["session_prefix"]
        if sess in train_sessions_prop:
            p_split = "train"
        elif sess in val_sessions_prop:
            p_split = "val"
        else:
            p_split = "test"

        proposed_records.append({
            "sample_id": row["sample_id"],
            "session_id": sess,
            "official_split": row["split"],
            "proposed_independent_split": p_split,
            "translation": row["translation"],
        })

    proposed_df = pd.DataFrame(proposed_records)
    prop_out = Path("g:/SingLang/SIGNOVA/data/manifests/proposed_signer_independent_split.csv")
    proposed_df.to_csv(prop_out, index=False, encoding="utf-8")

    print(f"Generated {report_out} and {prop_out}")
    print("Proposed split distribution:")
    print(proposed_df["proposed_independent_split"].value_counts())
    return analysis


if __name__ == "__main__":
    analyze_session_splits()
