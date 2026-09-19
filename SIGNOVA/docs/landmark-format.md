# SIGNOVA Landmark Feature Specification & Topology

## Overview

SIGNOVA utilizes an explicit, fixed-topology skeletal keypoint representation for continuous Indian Sign Language (ISL) processing. Each extracted video sequence is converted into a tensor array of shape $(T \times 543 \times 3)$ accompanied by an explicit multi-joint binary detection mask of shape $(T \times 4)$.

---

## 1. Landmark Topology (543 Keypoints)

The 543 keypoint indices correspond to the standardized MediaPipe Holistic skeletal structure:

| Component | Index Range | Landmark Count | Coordinates per Point | Key Anatomical Anchors |
| :--- | :--- | :--- | :--- | :--- |
| **Pose** | `0 .. 32` | 33 | $(x, y, v)$ | `0`: Nose, `11-12`: Shoulders, `13-14`: Elbows, `15-16`: Wrists, `23-24`: Hips |
| **Face** | `33 .. 500` | 468 | $(x, y, v)$ | Eye contours, eyebrow arcs, lips, jawline |
| **Left Hand** | `501 .. 521` | 21 | $(x, y, v)$ | `501`: Left Wrist, `502-505`: Thumb, `506-509`: Index, `510-513`: Middle, `514-517`: Ring, `518-521`: Pinky |
| **Right Hand** | `522 .. 542` | 21 | $(x, y, v)$ | `522`: Right Wrist, `523-526`: Thumb, `527-530`: Index, `531-534`: Middle, `535-538`: Ring, `539-542`: Pinky |
| **Total** | `0 .. 542` | **543** | $(x, y, v)$ | — |

Where:
- $x \in [-5.0, 5.0]$: Normalized, zero-centered horizontal coordinate.
- $y \in [-5.0, 5.0]$: Normalized, zero-centered vertical coordinate.
- $v \in [0.0, 1.0]$: Detector confidence / visibility score.

---

## 2. Detection Mask Specification $(T \times 4)$

Because sign language involves rapid hand movements and frequent self-occlusions, SIGNOVA maintains an explicit $(T \times 4)$ binary presence mask for each frame $t$:

```python
detection_mask[t] = [
    pose_detected,      # 1.0 if pose present, 0.0 otherwise
    face_detected,      # 1.0 if face present, 0.0 otherwise
    left_hand_detected, # 1.0 if left hand present, 0.0 otherwise
    right_hand_detected # 1.0 if right hand present, 0.0 otherwise
]
```

This prevents downstream models (ST-GCN, Transformer encoders) from confusing missing landmarks (zeros) with the spatial origin.

---

## 3. Storage Serialization Schema (.npz)

Extracted features are persisted in compressed `.npz` containers with version metadata:

```python
np.savez_compressed(
    "data/features/landmarks/<split>/<sample_id>.npz",
    landmarks=landmarks,          # float32, (T, 543, 3)
    detection_masks=masks,        # float32, (T, 4)
    timestamps_ms=timestamps_ms,  # float32, (T,)
    frame_indices=frame_indices,  # int32, (T,)
    metadata_json=metadata_json,  # json string header
)
```

### Metadata Header Schema
- `sample_id`: Canonical sample identifier.
- `split`: Canonical dataset split (`train`, `val`, `test`).
- `total_source_frames`: Raw video container frame count.
- `extracted_frames`: Extracted sequence length $T$.
- `extractor_version`: SemVer schema string (e.g., `"0.1.0"`).
- `pose_rate`: Sequence pose detection rate.
- `any_hand_rate`: Sequence hand presence rate.
- `is_usable`: Boolean quality gate flag.
