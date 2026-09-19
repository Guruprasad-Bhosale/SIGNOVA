# ISLTranslate Reference Analysis: `ISLTranslate-main`

Detailed technical review of the ISLTranslate dataset and repository structure.

---

## 1. Overview and Paper Context

- **Paper Title**: *ISLTranslate: Dataset for Translating Indian Sign Language* (ACL 2023 Findings).
- **Authors**: Abhinav Joshi, Susmit Agrawal, Ashutosh Modi (Exploration Lab, IIT Kanpur).
- **Domain**: Continuous Indian Sign Language (ISL) → Spoken English translation.
- **Dataset Size**: 31,222 sentence/phrase pairs.
- **License**: **CC BY-NC 4.0** (Attribution-NonCommercial).

---

## 2. Dataset Schema & Structure

### Files in Repository:
1. `data/ISLTranslate.csv` (1.64 MB):
   - `uid`: Unique identifier for each video/sentence pair (e.g., `1782bea75c7d-10`).
   - `text`: English transcript/translation of the corresponding continuous ISL sign video.
2. `data/ISL-signer_validation.csv` (25 KB):
   - 291 gold standard samples verified by a certified ISL signer.
   - Columns: `uid`, `Transcribed Text`, `Gold Translation`.
   - Reported human validation metrics: BLEU-4: 0.489, METEOR: 0.573, WER: 0.619.

---

## 3. Storage and Extraction Specifications

- **Remote Video Archives**: `ISL-videos.tar.gz` hosted on Hugging Face (`Exploration-Lab/iSign`).
- **Remote Pre-extracted Features**: `mediapipe_holistic_poses1.tar.gz` through `poses13.tar.gz`.
- **Ingestion Strategy for SIGNOVA**:
  - In Phase 0, SIGNOVA ingests the CSV annotations and builds deterministic train/val/test splits without downloading massive tar archives.
  - In Phase 1/2, a controlled download script will fetch required video subsets for feature extraction and validation.
