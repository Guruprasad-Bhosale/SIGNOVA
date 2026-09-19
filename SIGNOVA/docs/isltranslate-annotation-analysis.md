# ISLTranslate Annotation Analysis

Comprehensive examination of textual supervision fields available in the ISLTranslate dataset.

---

## 1. Field Inventory & Missing Value Rates

### 1.1 Primary Translation Dataset (`data/ISLTranslate.csv`)

| Field Name | Data Type | Meaning / Purpose | Example | Total Records | Missing Count | Missing Rate (%) |
|---|---|---|---|---|---|---|
| `uid` | String | Unique sample identifier (contains signer hash prefix & sequence index) | `1782bea75c7d-10` | 31,222 | 0 | 0.00% |
| `text` | String | Natural English translation transcript | `"Birbal started smiling..."` | 31,222 | 5 | 0.016% |

**Note on Missing Values**:
Exactly 5 samples have null or blank text strings:
- `b005df5e411b-60` (NaN)
- `aTB_lu2Im8Y--70` (blank)
- `nSQUhb44s5M--144` (blank)
- `nSQUhb44s5M--146` (blank)
- `iMlR4QLSWok--259` (NaN)

These 5 samples are classified as `INVALID_ANNOTATION` and excluded from model vocabulary construction.

---

### 1.2 Signer Validation Dataset (`data/ISL-signer_validation.csv`)

| Field Name | Data Type | Meaning | Missing Rate |
|---|---|---|---|
| `uid` | String | Sample identifier matching ISLTranslate UID | 0% |
| `Transcribed Text` | String | English transcript from automated / primary dataset | 0% |
| `Gold Translation` | String | Verified translation produced independently by a certified ISL signer | 0% |

- **Sample Count**: 291 paired comparisons.
- **Role in SIGNOVA**: Benchmark evaluation subset for testing translation fidelity against human expert standards.

---

## 2. Text Representation Characteristics

- **Language**: English (Spoken / Written translation).
- **Structure**: Continuous natural language sentences and conversational phrases (not isolated word glosses).
- **Punctuation**: Contains standard English punctuation (`.`, `,`, `!`, `?`, `"`, `'`).
- **Casing**: Mixed sentence case and proper nouns (e.g. Akbar, Birbal).
