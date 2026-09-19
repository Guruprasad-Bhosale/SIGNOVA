# SIGNOVA Phase 10 — Starting State Baseline

**System Date**: 2026-09-18  
**Current Phase**: Phase 10 — Controlled ISL Dataset Acquisition, Ingestion, Human Annotation Pilot & Evidence Closure  
**Starting Supervision Gate**: **STATE C (REAL SEQUENTIAL SUPERVISION BLOCKED)**  
**Authorized Project Spend**: **₹0** (Zero-Cost Mode: `DEFAULT`)  

---

## 1. System & Environment Baseline

- **Operating System**: Windows 11 (AMD64)
- **Python Version**: 3.12.10
- **PyTest Version**: 9.1.1
- **PyTorch Version**: 2.14.0+cpu (GPU available: NVIDIA GeForce RTX 3050 6GB)
- **MediaPipe Version**: 0.10.35
- **Unit Test Baseline**: 108 / 108 tests passing across Phases 1–9
- **Protected External Files**: 44 / 44 reference files in `ISLTranslate-main` and `isl-translator-main` verified against cryptographic SHA-256 baseline

---

## 2. Phase 10 Mission & Scope

Phase 10 implements a controlled, reproducible, non-destructive data acquisition and ingestion pipeline paired with a rigorous Deaf-ISL human annotation pilot:

1. **Deterministic Acquisition Pipeline**: Candidates are evaluated for eligibility, zero-cost compliance, license evidence, and source authenticity before acquisition.
2. **Untrusted Data Policy**: Downloaded files are never executed. Allowed operations: hashing, parsing, decoding supported media, validation, copying.
3. **Immutable Raw Data**: Raw downloaded artifacts are preserved unchanged under `data/raw/phase10/`. Derived canonical entities are generated in `data/interim/phase10/` and `data/annotations/phase10/`.
4. **Canonical Ingestion $\neq$ Training Ready**: Ingesting samples into SIGNOVA canonical format does NOT grant training readiness. `STATE C` remains strictly active.
5. **Dynamic Acquisition Outcome**: If zero-cost, license, and provenance audits find no eligible continuous sequential ISL dataset, the status is recorded honestly as `NO_ELIGIBLE_DATASET` rather than forcing synthetic downloads.
6. **Human Annotation Pilot**: Deterministic pilot sampling (fixed seed) with strict adherence to Phase 9 token conventions (`ICE-CREAM`, `FS-DELHI`, `NUM-2024`, `CL-V-WALK`). If human annotators are unavailable, status is flagged `BLOCKED_HUMAN_RESOURCE` without fabricating agreement statistics.
7. **Real CTC Gate**: `src/signova/recognition/phase9_gate.py` remains authoritative; real CTC training raises `RealCTCTrainingBlockedError`.
