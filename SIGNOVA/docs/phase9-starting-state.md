# SIGNOVA Phase 9 — Starting State & Data Gate Definition

**System Date**: 2026-09-18  
**Current Phase**: Phase 9 — Sequential ISL Dataset Acquisition, Verification & Human Annotation Readiness  
**Initial Gate**: **STATE C (REAL SEQUENTIAL SUPERVISION BLOCKED)**  
**Authorized Project Spend**: **₹0** (Zero-Cost Mode: `DEFAULT`)  

---

## 1. Executive Context & Objective

SIGNOVA has constructed a complete end-to-end continuous sign language recognition and translation architecture:
- **Perception**: MediaPipe holistic landmark extraction with kinematic normalization (Phase 2)
- **Isolated Modeling**: BiGRU and TCN temporal classifiers (Phase 3)
- **Continuous Representation**: Frame sliding windows and connectionist temporal classification heads (Phase 4)
- **Continuous CTC Engine**: Connectionist Temporal Classification loss & greedy/beam decoding (Phase 5 & 6)
- **Neural Machine Translation**: Attentive BiGRU Seq2Seq Gloss $\to$ English NMT engine (Phase 7)
- **Software Pipeline Bridge**: Unified `EndToEndSignTranslationPipeline` with deterministic collapse (Phase 8)

### The Scientific Bottleneck
While the continuous CTC architecture and neural translation pipeline function with 100% test coverage and sub-30ms execution, **real-world continuous ISL recognition and translation cannot be trained or validated without genuine, ordered sequential ISL sign gloss supervision**.

Phase 9 resolves this bottleneck by:
1. Auditing all known public and institutional continuous ISL datasets.
2. Formally evaluating candidate datasets against standardized rejection and eligibility criteria.
3. Enforcing cryptographic provenance and zero-cost constraints.
4. Implementing audit engines for annotation quality, label malformations, and signer leakage.
5. Formulating a standardized Deaf-ISL human annotation protocol and open-source tooling specification.
6. Establishing a decisive fork for Phase 10 based on real data availability.

---

## 2. Hard Data Gate Definitions

SIGNOVA strictly forbids synthetic pseudo-labeling, LLM gloss hallucination, or unverified label assignment. The Phase 9 Data Gate operates under three mutually exclusive states:

### `STATE A` — Verified Public Sequential ISL Dataset Exists
- **Condition**: A public, open-license dataset is identified that contains continuous ISL video paired with ordered, frame- or sequence-level sign glosses. Provenance, video-annotation synchronization, and licensing are fully verified.
- **Action**: Acquire a small pilot ($20 \text{--} 100$ videos), extract features, verify CTC feasibility, and proceed to **Phase 10: Real Sequential ISL CTC Training & Signer-Independent Evaluation**.

### `STATE B` — Candidate Dataset Exists but Requires Manual Verification / Access Approval
- **Condition**: A candidate dataset exists with potential sequential glosses, but access is restricted behind institutional forms/email requests, licensing requires clarification, or annotation semantics require deep manual inspection.
- **Action**: Document access requirements and block real CTC training until access is granted and verified.

### `STATE C` — No Suitable Public Sequential Dataset Found
- **Condition**: *"No publicly accessible candidate has yet been verified to provide reliable, ordered ISL gloss sequences with sufficient annotation provenance and licensing for SIGNOVA training."*
- **Action**: Maintain `REAL CTC TRAINING = BLOCKED`. Formalize human annotation protocols, configure open tooling (ELAN), and transition to **Phase 10: SIGNOVA Human Annotation Dataset Creation**.

---

## 3. Zero-Cost Policy (`ZERO_COST_MODE = DEFAULT`)

All Phase 9 operations run strictly under a zero-cost policy:
- **Budget**: ₹0.
- **System Behavior**: Any acquisition action requiring payment is automatically blocked and flagged `PAID_NOT_USED`.
- **Compute**: Local NVIDIA GeForce RTX 3050 GPU (6GB VRAM) and local CPU.
- **Tooling**: Free, open-source, offline software (e.g., ELAN, Python 3.12, PyTorch).
- **Authorized Spend**: `₹0`.

---

## 4. Scientific Honesty & Label Integrity Contract

1. **No Pseudo-Glosses**: Spoken English sentences from datasets like `ISLTranslate` are NOT converted into pseudo-glosses via LLMs, heuristic tokenizers, or rule systems for training.
2. **Rejection Taxonomy**: Every rejected or unverified dataset is cataloged with a formal standardized reason code.
3. **Conditional Metrics**: Inter-annotator agreement metrics are computed only from genuine dual-annotator human data, never synthetic placeholders.
