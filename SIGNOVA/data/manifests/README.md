# SIGNOVA Canonical Manifest System

This directory stores canonical manifests, integrity audits, and split definitions for SIGNOVA.

---

## 1. Manifest Files

| File | Purpose | Sample Count | Format |
|---|---|---|---|
| `isltranslate_manifest.csv` | Primary production manifest with official 80/10/10 split | 31,222 | CSV |
| `isltranslate_manifest.json` | JSON serialization of primary production manifest | 31,222 | JSON |
| `isltranslate_integrity.csv` | Sample-by-sample validity and availability audit | 31,222 | CSV |
| `duplicate_report.csv` | Identified duplicate raw keys and cross-split text overlap | 535 | CSV |
| `proposed_signer_independent_split.csv` | Proposed session/signer-independent benchmark split | 31,222 | CSV |
| `manifest_metadata.json` | Manifest versioning and generation provenance | N/A | JSON |
| `reference_integrity_baseline.json` | Cryptographic SHA-256 baseline of external reference files | N/A | JSON |

---

## 2. Canonical Sample Contract (`SignSample`)

```python
@dataclass
class SignSample:
    sample_id: str                          # Globally unique ID in SIGNOVA
    dataset: str                            # e.g., "ISLTranslate"
    video_reference: str                    # Remote UID key (e.g. "1782bea75c7d-1")
    split: str                              # "train", "val", "test"
    local_video_path: Optional[str]         # Path to local video if mounted, else None
    features_path: Optional[str]            # Path to pre-extracted .npy landmark array
    source_language: str                    # "Indian Sign Language (ISL)"
    target_language: str                    # "English"
    target_translation: Optional[str]       # English sentence transcript
    gloss_sequence: Optional[List[str]]     # None for ISLTranslate (unannotated)
    signer_id: Optional[str]                # None (unverified in raw CSV)
    session_id: Optional[str]               # Extracted prefix (e.g. "1782bea75c7d")
    availability: SampleAvailability        # REMOTE_ONLY, LOCAL_AVAILABLE, BOTH
    metadata: Dict[str, Any]
```

---

## 3. Reproducibility & Generation Command

```powershell
python scripts/create_manifest.py --dataset isltranslate --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```
Deterministic partitioning uses SHA-256 hashing (`hashlib.sha256(f"signova_v1_{uid}")`) to ensure identical split assignments across all platforms without pseudo-random seed drift.
