# ISLTranslate Video & Media Architecture Analysis

Analysis of video storage, packaging, and availability for the ISLTranslate continuous sign language dataset.

---

## 1. Physical Location & Video Availability Summary

- **Local Machine Status**: **0 video files present locally**.
- **Remote Host**: Hugging Face Hub dataset repository [`Exploration-Lab/iSign`](https://huggingface.co/datasets/Exploration-Lab/iSign).
- **Video Storage Format**: Multi-part split binary archives:
  - `iSign-videos_v1.1_part_aa`: 32.21 GB
  - `iSign-videos_v1.1_part_ab`: 25.68 GB
  - **Total Compressed Video Size**: **57.89 GB**
- **Feature Storage Format**: 4 multi-part split binary archives:
  - `iSign-poses_v1.1_part_aa` to `part_ad`: **170.24 GB Total**

---

## 2. Remote Smoke Test & Discovery Findings

1. **Packaging Mechanism**: Raw video clips are packaged inside concatenated `.tar.gz` chunks rather than exposed as 31k individual direct HTTP video URLs.
2. **Access Requirements**: The repository is public on Hugging Face and does not require private token authentication for read access.
3. **Local Workstation Constraint**: With 38 GB free disk space on the primary partition, downloading all 57.89 GB of raw video archives and uncompressing them locally would exceed disk limits.
4. **Phase 2 Ingestion Recommendation**:
   - For lightweight local development on the RTX 3050: Stream/download specific video subsets or download sample chunks onto secondary storage / cloud GPU (Kaggle/Colab).
   - Alternatively, evaluate the pre-extracted MediaPipe pose archives for landmark alignment.

---

## 3. Video Integrity Classification

| Category | Count | Status | Notes |
|---|---|---|---|
| **Total Annotation Records** | 31,222 | Validated | CSV metadata parsed |
| **Local Videos Present** | 0 | Expected | Not downloaded in Phase 0/1 |
| **Remote References** | 31,222 | Mapped | UID keys map to filenames within the remote tar archive |
| **Full Video Integrity** | Unverified | Deferred to Phase 2 | Full verification requires selective archive decompression in Phase 2 |
