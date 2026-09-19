# SIGNOVA Data Provenance & Lineage Specification

Documentation of dataset origins, local storage boundaries, and transformation lineage.

---

## 1. Primary Dataset Lineage

### 1.1 ISLTranslate
- **Source Paper**: *ISLTranslate: Dataset for Translating Indian Sign Language*, Findings of ACL 2023.
- **Authors**: Abhinav Joshi, Susmit Agrawal, Ashutosh Modi.
- **Publisher / Repository**: Exploration Lab, IIT Kanpur ([Hugging Face Exploration-Lab/iSign](https://huggingface.co/datasets/Exploration-Lab/iSign)).
- **License**: **CC BY-NC 4.0** (Attribution-NonCommercial).
- **Physical Ingestion in SIGNOVA**:
  - `../ISLTranslate-main/data/ISLTranslate.csv` and `ISL-signer_validation.csv` are read-only input sources.
  - Zero modifications were performed to the original CSV files.
  - Generated canonical manifests (`data/manifests/isltranslate_manifest.csv`) reside solely within SIGNOVA.

### 1.2 INCLUDE Dataset
- **Source Paper**: *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition*, ACM Multimedia 2020.
- **Host**: Zenodo (Record ID `4010759`).
- **License**: Academic Non-Commercial.

---

## 2. Immutability & Processing Pipeline

```
[Raw External Datasets] (READ-ONLY)
         │
         ▼
[SIGNOVA Dataset Adapters] (Pydantic / Dataclass parsing)
         │
         ▼
[Canonical Manifests & Integrity Audits] (CSV/JSON in data/manifests/)
         │
         ▼
[Future Phase 2: Feature Extraction & Feature Arrays] (data/features/)
```
