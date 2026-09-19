# SIGNOVA Phase 10 — Controlled ISL Dataset Acquisition, Ingestion, Human Annotation Pilot & Evidence Closure Completion Report

**Date**: 2026-09-18  
**Final Supervision Gate**: **STATE C — REAL SEQUENTIAL ISL SUPERVISION BLOCKED**  
**Authorized Project Spend**: **₹0** (`ZERO_COST_MODE = DEFAULT`)  
**Acquisition Outcome**: **`NO_ELIGIBLE_DATASET`**  
**Test Suite Status**: **129 / 129 Unit Tests Passing (100%)**  
**Protected External Integrity**: **44 / 44 Reference Files Cryptographically Unchanged**  

---

## 1. Executive Summary & Required Questionnaire Responses

### 1. What dataset/source was actually acquired?
No external public dataset met all zero-cost, open-license, and ordered ISL sign gloss sequence requirements. Candidate sources (`ISL-CSLTR`, `ISLTranslate`, `ISL-Fingerspelling`, `ISLVT`, `INCLUDE`) were audited and recorded in the acquisition manifest as `NO_ELIGIBLE_DATASET` due to absence of verified lexical sign glosses. Local diagnostic records were ingested for pipeline validation under `data/interim/phase10/`.

### 2. Why was it eligible under ₹0 zero-cost mode?
All candidate evaluation and local processing operated under `ZERO_COST_MODE = DEFAULT`. Paid sources were automatically intercepted and rejected as `PAID_NOT_USED`. Total project spend is strictly **₹0**.

### 3. What license/usage evidence was found?
Recorded in `outputs/reports/phase10_license_audit.json`:
- `ISL-CSLTR`: CC BY 4.0 (Open research, but lacks lexical sign glosses).
- `ISLTranslate`: Research Only / Open metadata (English translations only).
- `ISL-Fingerspelling`: CC BY-NC 4.0 (Restricted to fingerspelled alphabet).
- `INCLUDE`: CC BY-NC-SA 4.0 (Isolated words only).
- `IIIT-H ISL Lab`: Restricted Institutional (No open license; `ACCESS_RESTRICTED`).

### 4. What artifacts were downloaded?
Zero untrusted external archives were downloaded into production storage because no candidate met the ordered-gloss prerequisite. Download operations were simulated in offline unit tests using non-executing local mock files.

### 5. What SHA-256 hashes were recorded?
All manifest records and interim canonical files were hashed with SHA-256 and cataloged in `outputs/reports/phase9_checksums.csv` and `outputs/reports/phase10_dataset_inventory.csv`.

### 6. How many samples were successfully ingested?
10 local diagnostic video-landmark samples were canonically ingested into `data/interim/phase10/` to validate the ingestion and normalization engine.

### 7. How many samples were rejected and why?
The 10 diagnostic samples were classified as `INVALID` (`quality_status = WEAK`) for model training because they lack ordered sign gloss sequences (`EMPTY_GLOSS_SEQUENCE`), reinforcing `CANONICAL_INGESTED != TRAINING_READY`.

### 8. Were duplicate samples found?
The duplicate audit (`outputs/reports/phase10_duplicate_audit.json`) confirmed 0 exact SHA-256 duplicates and 0 metadata collisions. Near-duplicate perceptual video hashing was reported as `NEAR_DUPLICATE_AUDIT_NOT_IMPLEMENTED` to prevent unverified heuristic complexity.

### 9. Was signer-independent splitting possible?
Yes, for signer-tagged subsets (`signer_01` vs distinct test signers). When signer metadata is missing in external sources, the leakage engine safely reports `SIGNER_INDEPENDENT_SPLIT_NOT_POSSIBLE` without fabricating pseudo-identities.

### 10. What annotation pilot was completed?
The deterministic pilot sample specification was produced in `outputs/reports/phase10_annotation_sample.json` (`seed = 42`, 10 stratified samples across compounds, fingerspelling, numbers, classifiers, and complex boundaries) following the Phase 9 Deaf-ISL annotation protocol (`docs/phase10-human-annotation-pilot.md`).

### 11. How many genuine human annotators participated?
0 native Deaf annotators were actively recruited during this automated engineering phase.

### 12. Was inter-annotator agreement computable?
No. In strict compliance with scientific honesty safeguards, `outputs/reports/phase10_interannotator_agreement.json` reports `status = "NOT_COMPUTABLE"` (`reason = "BLOCKED_HUMAN_RESOURCE"`). Agreement metrics were not fabricated.

### 13. What disagreements occurred?
0 real disagreements occurred due to single-annotator/pending-pilot state. The 8-category disagreement taxonomy (`TOKENIZATION`, `COMPOUND_SIGN`, `FINGERSPELLING`, `NUMBER`, `CLASSIFIER`, `TEMPORAL_BOUNDARY`, `GLOSS_SELECTION`, `OTHER`) is registered in `outputs/reports/phase10_annotation_disagreements.json`.

### 14. Were all provenance links preserved?
Yes. The extended provenance chain (`SOURCE` $\to$ `ACQUISITION` $\to$ `DOWNLOAD` $\to$ `RAW_ARTIFACT` $\to$ `INGESTION` $\to$ `ANNOTATION` $\to$ `VIDEO` $\to$ `FEATURE` $\to$ `MODEL_SAMPLE`) was verified with parent-child linkage in `src/signova/data/provenance.py`.

### 15. Were all 44 protected reference files unchanged?
Yes. Cryptographic verification in `outputs/reports/phase10_integrity.json` confirms 44 / 44 files in `ISLTranslate-main` and `isl-translator-main` match the original baseline hashes.

### 16. Did all existing tests remain passing?
Yes. All 108 existing unit tests from Phases 1–9 plus 21 new Phase 10 unit tests pass (129 / 129 total tests passing, 0 failures).

### 17. Was real CTC training attempted?
No. Training remained strictly forbidden and uninvoked.

### 18. Did the CTC gate remain enforced?
Yes. `RealCTCTrainingGate` in `src/signova/recognition/phase9_gate.py` raised `RealCTCTrainingBlockedError` on any unverified execution attempt.

### 19. What is the final supervision state?
**`STATE C`**.

### 20. What exact evidence is required before Phase 11?
Phase 11 entry requires:
1. Ingestion of double-annotated Deaf-ISL human annotations for the pilot sample set.
2. Verified inter-annotator agreement (Token Error Rate $< 15\%$, exact sequence match $> 70\%$).
3. Complete Deaf linguist adjudication of any annotation disagreements.
4. Formal transition gate evaluation from `STATE C` to `STATE A` before real CTC training is unlocked.
