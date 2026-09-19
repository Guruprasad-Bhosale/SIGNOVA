# Phase 6 Continuous ISL Dataset Acquisition Specification

## 1. Specification Objective

This document defines the strict formal requirements for acquiring or ingesting an external continuous Indian Sign Language (ISL) dataset for Continuous Sign Language Recognition (CSLR) and CTC training in SIGNOVA.

Candidate datasets (such as CI-ISL, ISL-CSLR, or custom institutional recordings) are treated as **unverified leads** until their physical files, annotations, semantics, provenance, licenses, and sequence formats are inspected and validated.

## 2. Mandatory Dataset Properties

To satisfy Phase 6 CSLR training requirements, any candidate dataset must fulfill the following criteria:

| Requirement Category | Specification Detail | Validation Method |
|---|---|---|
| **Language & Modality** | Natural, continuous Indian Sign Language (ISL) | Linguistic review / visual inspection |
| **Ground Truth Targets** | Ordered sequence of lexical sign glosses (e.g. `['NAMASTE', 'MEETING', 'START']`) | Annotation semantics audit |
| **Temporal Alignment** | Video frame count $T_{\text{video}}$ must satisfy CTC constraint: $T_{\text{feature}} \ge T_{\text{gloss}}$ | Automated length verification |
| **Signer Diversity** | At least 5+ distinct biological signers with explicit signer IDs | Metadata integrity check |
| **Split Protocol** | Pre-defined train, validation, and test splits with zero video or session overlap | Automated leakage detection |
| **Video Quality** | Minimum 720p resolution, $\ge 25$ FPS, unobstructed view of face, upper torso, and both hands | MediaPipe landmark extraction quality check |
| **Research License** | Permissible non-commercial / academic research license without redistribution violations | Legal / provenance check |

## 3. Storage and Preprocessing Estimation

- **Estimated Video Volume**: 500–2,000 continuous video sentences (~10–50 GB raw MP4).
- **Extracted Landmark Features**: Normalized `.npz` files (~1–3 GB total for full 543-landmark topology).
- **Processing Time on RTX 3050**: ~15–30 minutes for MediaPipe landmark extraction across 1,000 clips using multi-threaded batching.

## 4. Ingestion Protocol

Once a verified dataset is obtained:
1. Wrap the raw format in `signova.data.adapters.generic_sequential.GenericSequentialAdapter`.
2. Generate `data/manifests/phase6_sequential_manifest.csv` with explicit `NULL` handling.
3. Build the vocabulary strictly from training targets in `data/manifests/phase6_vocabulary.csv`.
4. Validate zero split leakage across video and signer dimensions.
5. Proceed to real CTC training using `SequentialCTCTrainer`.
