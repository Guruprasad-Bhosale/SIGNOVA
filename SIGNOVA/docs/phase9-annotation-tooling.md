# SIGNOVA Zero-Cost Human Annotation Tooling Guide

**Tooling Baseline**: Free, Open-Source, Offline Software  
**Primary Tool**: **ELAN (EUDICO Linguistic Annotator)** by Max Planck Institute for Psycholinguistics  
**Secondary Tools**: VGG Image Annotator (VIA), Anvil Video Annotation  

---

## 1. Tool Selection Rationale

ELAN is the international gold standard in sign language and gesture linguistics:
- **Cost**: 100% Free & Open Source (GPL License).
- **Offline**: Runs entirely locally on Windows/macOS/Linux without cloud dependencies or subscriptions.
- **Multi-Tier Architecture**: Supports hierarchical, time-aligned tiers for glosses, phonetics, translations, and signer IDs.
- **SIGNOVA Integration**: Fully compatible with SIGNOVA's Phase 8 `ElanAnnotationAdapter` (`.eaf` parser).

---

## 2. Recommended ELAN Tier Structure Template

Create an ELAN template (`.etf`) with the following standardized tiers:

| Tier Name | Linguistic Type | Parent Tier | Content Description |
| :--- | :--- | :--- | :--- |
| **`Sign-Gloss`** | Time-Aligned | None | Ordered ISL gloss tokens (`SCHOOL`, `EAT`, `FS-DELHI`) |
| **`Translation-EN`** | Symbolic Association | `Sign-Gloss` | Natural fluent English sentence |
| **`Confidence`** | Controlled Vocabulary | `Sign-Gloss` | `HIGH`, `MEDIUM`, `LOW`, `UNCERTAIN` |
| **`Signer-ID`** | Symbolic Association | None | Unique anonymized signer identifier (`signer_01`) |
| **`Non-Manual`** | Time-Aligned (Optional) | None | Head tilt, eyebrow raise, mouthing |

---

## 3. Step-by-Step Offline Annotation Workflow

1. **Load Video**: Open `.mp4` file in ELAN (`File` $\to$ `New` $\to$ Select Media).
2. **Segment Sign Boundaries**: Double-click time-line or drag cursor to create boundary when hands move to form a sign.
3. **Enter Standard Gloss**: Type uppercase gloss in `Sign-Gloss` tier following `docs/phase9-human-annotation-protocol.md`.
4. **Enter English Sentence**: On `Translation-EN` tier, type the natural English equivalent sentence.
5. **Set Confidence**: Select confidence rating (`HIGH`/`MEDIUM`/`LOW`/`UNCERTAIN`).
6. **Save EAF**: Save as `sample_001.eaf` directly adjacent to video or in `data/annotations/`.

---

## 4. Ingestion into SIGNOVA

Run SIGNOVA's Phase 8 Annotation Ingestion Engine:

```python
from signova.data.adapters.elan_adapter import ElanAnnotationAdapter

adapter = ElanAnnotationAdapter()
sample = adapter.parse_file("data/annotations/sample_001.eaf")

print(sample.glosses)       # ['NAMASTE', 'MY', 'NAME', 'FS-RAHUL']
print(sample.english_text)   # 'Hello, my name is Rahul.'
print(sample.quality_grade)  # 'VERIFIED' or 'LINGUIST_REVIEWED'
```
