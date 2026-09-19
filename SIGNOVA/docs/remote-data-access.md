# SIGNOVA Remote Video Acquisition Strategy & Architecture

Technical blueprint for acquiring, streaming, or caching remote sign language video assets in Phase 2.

---

## 1. Remote Host Specifications

- **Repository**: [`Exploration-Lab/iSign`](https://huggingface.co/datasets/Exploration-Lab/iSign)
- **Access URL**: `https://huggingface.co/datasets/Exploration-Lab/iSign`
- **Authentication**: **Public** (No private Hugging Face token required for read access).
- **License**: `CC BY-NC-SA 4.0` (Attribution-NonCommercial-ShareAlike).

---

## 2. Remote Storage Structure & File Inventory

The dataset is stored in monolithic multi-part split binary archives:

```
Exploration-Lab/iSign/
├── iSign-videos_v1.1_part_aa        (32.21 GB)
├── iSign-videos_v1.1_part_ab        (25.68 GB)
│   └── Total Raw Video Archive:      57.89 GB (Compressed .tar.gz)
│
├── iSign-poses_v1.1_part_aa         (48.32 GB)
├── iSign-poses_v1.1_part_ab         (48.32 GB)
├── iSign-poses_v1.1_part_ac         (48.32 GB)
├── iSign-poses_v1.1_part_ad         (25.28 GB)
│   └── Total Pose Landmark Archive:  170.24 GB
│
└── iSign_v1.1.csv                   (9.81 MB)
```

---

## 3. Workstation Storage Budget & Feasibility Analysis

- **Local Machine Partition**: ~38 GB free disk space.
- **Feasibility Constraint**: Downloading the full 57.89 GB video archive exceeds local disk headroom.
- **Decompression Requirement**: Multi-part files must be concatenated first:
  ```powershell
  # Windows PowerShell concatenation:
  cmd /c "copy /b iSign-videos_v1.1_part_aa + iSign-videos_v1.1_part_ab ISL-videos.tar.gz"
  tar -xzf ISL-videos.tar.gz
  ```

---

## 4. Phase 2 Recommended Acquisition Paths

### Strategy A: Cloud Pre-Extraction & Feature Sync (Recommended)
1. Run a lightweight extraction script on Kaggle (20 GB disk / free T4 GPU) or Google Colab.
2. Download and unpack `iSign-videos_v1.1` in the cloud environment.
3. Extract 543 MediaPipe Holistic landmarks per frame, normalize, and compress to `.npz` arrays.
4. Total compressed `.npz` feature volume will be ~2.5–4.0 GB (fits easily on the local RTX 3050 workstation).
5. Transfer the compact `.npz` feature archives directly into `SIGNOVA/data/features/`.

### Strategy B: Selective Local Subset Extraction
1. Download a single split chunk or sample videos for local unit testing and visual debugging.
2. Maintain `SampleAvailability.REMOTE_ONLY` for the remainder of the dataset until cloud features are synced.
