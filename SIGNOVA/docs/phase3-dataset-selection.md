# Phase 3 Dataset Suitability & Selection Analysis

## Executive Summary

Phase 3 establishes the **isolated dynamic sign recognition baseline** for SIGNOVA. This document details the comparative suitability analysis between candidate sign language datasets and establishes why **INCLUDE** is selected as the primary dataset for Phase 3.

---

## 1. Candidate Dataset Comparison

| Evaluation Criterion | ISLTranslate (ACL 2023) | INCLUDE (Zenodo 4010759 / ACM MM 2020) |
| :--- | :--- | :--- |
| **Dataset Paradigm** | Continuous sentence translation | Isolated word-level dynamic signs |
| **Total Samples** | 31,222 continuous sentences | 4,287 isolated video clips |
| **Target Classes** | N/A (Continuous Translation) | 263 distinct sign categories |
| **Annotation Granularity** | Full English translation sentences | Discrete categorical sign glosses |
| **Gloss Alignment** | **None** (unaligned sentence pairs) | **Native** (one sign class per video) |
| **Signer Metadata** | Session-level hashes (`signer_id` in metadata) | Multi-signer recordings across 7 sessions |
| **Video Availability** | Unmounted (57.89 GB on Hugging Face) | Unmounted (Zenodo Record `4010759`) |
| **SIGNOVA 543 Compatibility** | Compatible (via Phase 2 pipeline) | Compatible (via Phase 2 pipeline) |
| **Suitability for Phase 3** | **UNSUITABLE** | **SELECTED AS PRIMARY** |

---

## 2. Engineering Decision Rationale

### Why ISLTranslate Cannot Support Phase 3:
1. **Absence of Word Boundaries**: ISLTranslate consists of natural continuous sentences without frame-level or word-level segmentations.
2. **Grammar Mismatch**: ISL follows Subject-Object-Verb (SOV) / topic-comment structure, whereas English follows SVO. Slicing sentences into naive word chunks would fabricate false ground truth.
3. **Scientific Integrity**: Training an isolated classifier on sentence-level translations is invalid. ISLTranslate is appropriately reserved for Phase 5 (Continuous Sign Recognition with CTC Loss) and Phase 7 (NMT).

### Why INCLUDE is Selected for Phase 3:
1. **Native Isolated Sign Vocabulary**: INCLUDE provides isolated clips for 263 distinct signs across diverse semantic categories (Greetings, Colours, Animals, Numbers, Daily Actions).
2. **Direct Validation of Temporal Encoders**: Enables direct evaluation of whether skeletal temporal backbones (BiGRU, TCN) effectively distinguish dynamic sign motions.
3. **Manageable Acquisition Footprint**: Category zip bundles (e.g. Greetings, Colours) allow controlled pilot acquisition without multi-gigabyte disk saturation.
