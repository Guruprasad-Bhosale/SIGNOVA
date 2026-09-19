# Phase 8 Starting-State Audit: Real Sequential ISL Supervision & Pipeline Bridge

## 1. Executive Summary & Hard Data Gate Status

**Date**: 2026-09-18  
**Gate Status**: **STATE C — REAL SEQUENTIAL ISL GLOSS SUPERVISION BLOCKED**  

> [!CAUTION]
> **CRITICAL SCIENTIFIC GUARDRAIL**:
> 1. **Phase 8 executes pipeline-contract and engineering diagnostics across the 72-video real continuous feature corpus, and functional end-to-end validation using synthetic fixtures.**
> 2. The 72 real continuous videos lack verified Deaf ISL linguist gloss annotations. We make zero claims of real CTC recognition accuracy, real gloss prediction, or real ISL translation on the 72 videos.
> 3. Real CTC training is permitted ONLY if: (STATE A) AND (verified sequential gloss annotations exist) AND (video-annotation pairing verified) AND (split leakage checks pass). Otherwise, STATE C blocks real training.

---

## 2. Inventory of Continuous Video Features & Supervision Assets

| Asset Category | Specification | Quantity / Format | Real Sequential ISL Gloss Annotation? | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Real Continuous Video Corpus** | `ISLTranslate` educational videos | 72 MP4 video files | **NO** (English text sentences only) | Unsupervised Video Corpus |
| **Real Extracted Landmark Features** | Normalized 543 MediaPipe landmarks (Full / Hands+Pose / Hands) | 72 `.npz` files (11,980 frames total, $\sim 5.55\text{s}$ avg) | **NO** (No paired glosses) | Feature Corpus for Pipeline Diagnostics |
| **Isolated Sign Datasets** | `INCLUDE` 263 classes | 4,287 clips | **NO** (Isolated lexical signs only) | Isolated Classification Baseline Only |
| **Translation Data** | `ISLTranslate.csv` | 31,225 rows | **NO** (English sentence text only) | Blocked for Gloss Translation |
| **Synthetic Fixtures** | Continuous CTC & Seq2Seq fixtures | JSON / `.npz` test fixtures | Marked `SYNTHETIC FIXTURE` | Engineering & Functional Verification |

---

## 3. Real CTC Training Gate Specification

Real CTC training will be unblocked if and only if all gate requirements are verified:
1. **Gate Condition 1 — State Classification**: Formally verified as **STATE A**.
2. **Gate Condition 2 — Sequential Annotation**: Ordered lexical ISL gloss sequence provided per video.
3. **Gate Condition 3 — Video-Annotation Pairing**: Explicit file-level matching between video IDs and annotation records.
4. **Gate Condition 4 — Annotation Quality Grade**: Rated `VERIFIED` or `LINGUIST_REVIEWED`.
5. **Gate Condition 5 — Split Leakage Audit**: Signer-independent and session-independent splits verified without cross-split overlap.

---

## 4. Multi-Tier Annotation Standard (ELAN `.eaf`)

Phase 8 defines the official ELAN tier hierarchy for future continuous ISL corpus annotation:

```
TIER_ID: GLOSS_RH (Right hand lexical sign gloss)
TIER_ID: GLOSS_LH (Left hand lexical sign gloss, if asymmetric)
TIER_ID: GLOSS_MAIN (Primary canonical ISL gloss sequence, ordered)
TIER_ID: TRANSLATION_EN (Natural English translation sentence)
TIER_ID: TRANSLATION_HI (Optional Hindi translation sentence)
TIER_ID: NON_MANUAL (Head nod, eyebrow, mouth gesture markers)
TIER_ID: SIGNER_ID (Verified signer metadata)
```
