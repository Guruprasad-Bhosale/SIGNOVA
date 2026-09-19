# SIGNOVA Landmark Spatial Normalization Specification

## Overview

Sign language videos exhibit significant variability in signer morphology, camera focal distance, aspect ratio, and signer positioning within the video frame. To make skeletal sequence representations invariant to camera distance and signer offset, SIGNOVA applies a multi-tiered spatial normalization routine.

---

## 1. Mathematical Formulation

Given raw coordinates $(x_i, y_i)$ for joint $i \in \{0 \dots 542\}$:

$$
\mathbf{p}_i^{\text{norm}} = \frac{\mathbf{p}_i - \mathbf{c}}{s + \epsilon}
$$

Where:
- $\mathbf{c} \in \mathbb{R}^2$ is the translation anchor center.
- $s \in \mathbb{R}^+$ is the scale normalization factor.
- $\epsilon = 10^{-6}$ prevents numerical division by zero.
- Joint visibility / confidence $v_i$ is preserved unaltered.

---

## 2. Multi-Tiered Fallback Strategy

In sign language footage, occlusions can cause primary anatomical reference points (e.g., hips or shoulders) to be temporarily undetected. SIGNOVA uses robust multi-tiered fallback cascades:

### Center Anchor Cascades ($\mathbf{c}$)
1. **Primary (`mid_hip`)**: Midpoint of Left Hip (idx 23) and Right Hip (idx 24):
   $$\mathbf{c} = \frac{\mathbf{p}_{23} + \mathbf{p}_{24}}{2}$$
2. **Secondary (`nose`)**: Nose joint (idx 0) if hips are occluded.
3. **Tertiary (`bbox_center`)**: Arithmetic mean center of all detected landmarks in the current frame.
4. **Final (`origin`)**: Frame origin $(0, 0)$ if no keypoints are detected.

### Scale Anchor Cascades ($s$)
1. **Primary (`shoulder_dist`)**: Euclidean distance between Left Shoulder (idx 11) and Right Shoulder (idx 12):
   $$s = \|\mathbf{p}_{11} - \mathbf{p}_{12}\|_2$$
2. **Secondary (`bbox_diagonal`)**: Bounding box diagonal length of all valid landmarks in the frame:
   $$s = \|\mathbf{p}_{\max} - \mathbf{p}_{\min}\|_2$$
3. **Tertiary (`unit`)**: Constant unit scale ($s = 1.0$).

---

## 3. Numerical Safety & Sanitization

- **NaN / Inf Clamping**: Any `NaN`, `+Inf`, or `-Inf` values originating from corrupted video decodes or zero division are sanitized using `np.nan_to_num` with zero substitution.
- **Bounding Box Verification**: Normalized coordinates are guaranteed to remain within the numerical envelope $[-5.0, 5.0]$.
- **Zero Invariance**: Undetected joints $(0.0, 0.0, 0.0)$ remain strictly zeroed and are not shifted by center translation.
