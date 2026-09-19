# SIGNOVA Phase 22: Controlled Dataset Formation & CTC Unlock

## 1. Pipeline Architecture

The dataset construction and freeze pipeline enforces a strict 7-stage gate sequence:

```
ANNOTATIONS (HUMAN_DIRECT)
       ↓
PROVENANCE & VALIDATION (SHA-256 Match)
       ↓
REVIEW LIFECYCLE (VERIFIED)
       ↓
TRAINING ELIGIBILITY (Qualified Annotator + Verified State)
       ↓
SAMPLE CTC FEASIBILITY (T_required <= T_features)
       ↓
DATASET FREEZE (DATASET_FROZEN + Fingerprint SHA-256)
       ↓
CANONICAL PHASE 19 GATE
       ↓
PHASE 21 TRAINING UNLOCK
```

---

## 2. Sample-Level Repeated-Token CTC Feasibility

CTC decoding requires intermediate blank tokens between identical adjacent labels. The required timesteps $T_{\text{required}}$ are defined as:

$$T_{\text{required}} = L + \sum_{i=1}^{L-1} \mathbb{I}(y_i = y_{i+1})$$

A sample is **CTC Feasible** if and only if:
$$T_{\text{required}} \le T_{\text{features}}$$

### Decoupled Policy
`TRAINING_ELIGIBLE` and `CTC_FEASIBLE` are evaluated independently:
- A verified human annotation is legitimate (`HUMAN_VERIFIED = TRUE, TRAINING_ELIGIBLE = TRUE`).
- If $T_{\text{required}} > T_{\text{features}}$, it is marked `CTC_FEASIBLE = FALSE` without invalidating the human annotation.

---

## 3. Strict Split Hierarchy

Datasets are partitioned strictly according to the hierarchical independence policy:
1. `SIGNER_INDEPENDENT` (Highest priority, prevents identity memorization)
2. `SESSION_INDEPENDENT` (Second priority, prevents background/lighting memorization)
3. `SOURCE_GROUP_INDEPENDENT` (Third priority)
4. `RANDOM` (Prohibited by default; requires explicit `--allow-random-split` flag with mandatory leakage warnings)

---

## 4. Dataset Freeze & Invalidation Semantics

Datasets operate under strict freeze semantics:
- `DATASET_DRAFT`: Mutable dataset under construction.
- `DATASET_FROZEN`: Immutable release with deterministic SHA-256 fingerprint computed across sample annotations, vocabulary, and splits.
- Any subsequent change to annotations, video hashes, or vocabulary invalidates the previous fingerprint and requires an explicit unfreeze or new release.
