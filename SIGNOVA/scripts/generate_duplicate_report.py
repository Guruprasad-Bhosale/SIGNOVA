"""
Duplicate detection and repetition analysis script.
"""

from collections import Counter
from pathlib import Path
import pandas as pd


def generate_duplicate_report():
    manifest_path = Path("g:/SingLang/SIGNOVA/data/manifests/isltranslate_manifest.csv")
    df = pd.read_csv(manifest_path, dtype=str)

    records = []

    # 1. Duplicate sample IDs (should be 0)
    id_counts = Counter(df["sample_id"])
    for sid, count in id_counts.items():
        if count > 1:
            records.append({
                "duplicate_type": "DUPLICATE_SAMPLE_ID",
                "key": sid,
                "occurrence_count": count,
                "classification": "LEAKAGE_RISK",
                "description": f"Sample ID appears {count} times in manifest",
            })

    # 2. Raw UID duplicates disambiguated
    raw_uids = [sid.split("_dup")[0] for sid in df["sample_id"]]
    raw_counts = Counter(raw_uids)
    for r_uid, count in raw_counts.items():
        if count > 1:
            records.append({
                "duplicate_type": "RAW_UID_DUPLICATE",
                "key": r_uid,
                "occurrence_count": count,
                "classification": "SAFE",
                "description": f"Raw UID occurred {count} times in raw CSV; disambiguated in manifest",
            })

    # 3. Cross-split identical sentence text
    sentence_splits = {}
    for _, row in df.iterrows():
        text = str(row["translation"]).strip()
        split = str(row["split"]).strip()
        if not text or text == "nan":
            continue
        if text not in sentence_splits:
            sentence_splits[text] = set()
        sentence_splits[text].add(split)

    cross_split = {t: s for t, s in sentence_splits.items() if len(s) > 1}
    for text, splits in cross_split.items():
        records.append({
            "duplicate_type": "CROSS_SPLIT_SENTENCE_OVERLAP",
            "key": text[:100],
            "occurrence_count": len(splits),
            "classification": "REVIEW",
            "description": f"Identical English target appears in {', '.join(sorted(splits))}",
        })

    out_df = pd.DataFrame(records)
    out_path = Path("g:/SingLang/SIGNOVA/data/manifests/duplicate_report.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Generated duplicate report with {len(out_df)} rows in {out_path}")
    print(f"Total cross-split identical sentence texts: {len(cross_split)}")
    return out_df


if __name__ == "__main__":
    generate_duplicate_report()
