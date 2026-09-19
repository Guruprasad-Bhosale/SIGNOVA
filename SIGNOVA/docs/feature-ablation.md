# SIGNOVA Feature & Temporal Ablation Study

## Executive Summary

To systematically understand the contributions of anatomical landmark subsets and temporal dynamics to dynamic sign recognition, SIGNOVA executed two controlled ablation suites:

1. **Feature Group Ablation**: Comparing `HANDS` (42 joints), `HANDS_POSE` (75 joints), and `FULL_BODY` (543 joints).
2. **Temporal Dynamics Ablation**: Comparing `Static Pooled MLP` (zero temporal recurrence) against `BiGRU` and `TCN` temporal encoders.

---

## 1. Feature Group Slicing & Anatomical Dimensions

| Feature Configuration | Active Joints | Total Input Values | Feature Group Description |
| :--- | :--- | :--- | :--- |
| **HANDS** | 42 | 126 ($42 \times 3$) | Left Hand (21) + Right Hand (21). Focuses exclusively on manual finger configurations and hand shapes. |
| **HANDS_POSE** | 75 | 225 ($75 \times 3$) | Pose (33) + Left Hand (21) + Right Hand (21). Captures gross torso posture, shoulder/elbow/wrist kinematics, and manual finger joints. |
| **FULL (Full Body)** | 543 | 1,629 ($543 \times 3$) | Pose (33) + Face Mesh (468) + Left Hand (21) + Right Hand (21). Full holistic skeletal representation. |

---

## 2. Research Findings & Key Observations

### A. Manual vs Full Body Efficiency
- **Hands + Pose (75 joints)** captures the vast majority of kinetic variance required for isolated dynamic sign recognition while reducing input dimensionality by **86.2%** compared to Full Body (543 joints).
- **Face Mesh (468 joints)** introduces high-dimensional micro-features that require substantial model capacity without proportional accuracy gains for isolated manual signs, though facial expressions become critical in later continuous grammar phases (non-manual markers).

### B. Temporal Sequence vs Static Pooling
- Static Mean-Pooled representations discard inter-frame trajectory ordering and speed variations.
- Dynamic sequence models (`BiGRU` and `TCN`) establish clear predictive superiority over static pooling, confirming that directional motion vectors and temporal joint transitions are essential to sign discrimination.
