# Phase 25: Human Sequential ISL Annotation Contributor Guide

## Overview

This guide explains how a real human Indian Sign Language (ISL) signer or linguist contributes genuine sequential gloss annotations into SIGNOVA.

---

## 1. Ethical & Scientific Invariants

- **HUMAN_DIRECT Only**: Every annotation must be entered by a human watching a real ISL video.
- **Forbidden**: LLM generation, English sentence translation heuristic mapping, pseudo-labeling, and automated gloss generation are strictly forbidden and rejected at the gate.
- **Ordered Sequential Glosses**: Annotators enter glosses in the exact chronological order they are signed in the video.

---

## 2. Concrete Annotation Workflow

```
Real ISL Video (e.g. video_001.mp4)
       │
       ▼
Annotation Form / CLI / Web Platform
       │
       ▼
Human Annotator watches continuous sign video
       │
       ▼
Annotator enters ordered gloss sequence:
["HELLO", "NAME", "ISL", "LEARN"]
       │
       ▼
Save Draft (review_state = DRAFT)
       │
       ▼
Submit Annotation (review_state = SUBMITTED)
       │
       ▼
Lead Reviewer Review & Verification (review_state = VERIFIED)
       │
       ▼
Sample becomes Training Eligible
```

---

## 3. Contributing via CLI

To submit a new human annotation:
```bash
python scripts/run_phase25_acquisition.py \
  --annotation-id ann_human_001 \
  --video-id vid_001 \
  --video-path data/raw_videos/vid_001.mp4 \
  --annotator-id annotator_rajesh_01 \
  --glosses HELLO HOW ARE YOU \
  --submit
```

To review and verify submitted annotations:
```bash
python scripts/run_phase25_acquisition.py \
  --annotation-id ann_human_001 \
  --review VERIFY \
  --reviewer-id lead_linguist_priya \
  --data-root data
```

---

## 4. Contributing via Web Platform

Run the local annotation platform:
```bash
python run_signova.py --server
```
Navigate to `http://127.0.0.1:8001` to view video clips frame-by-frame, mark sign boundaries, assign gloss tokens, and submit annotations.
