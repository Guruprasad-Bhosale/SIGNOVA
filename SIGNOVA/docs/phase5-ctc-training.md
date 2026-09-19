# Phase 5 CTC Training & Sequence Recognition Pipeline

## 1. System Pipeline

```
Continuous Video
       │
       ▼
MediaPipe Holistic Extractor (543 Landmarks)
       │
       ▼
Spatial Feature Normalization (Hands+Pose 75 Joints)
       │
       ▼
Continuous Temporal Encoder (2-Layer BiGRU, d_h=256)
       │
       ▼
Per-Frame Sequence Representation (B, T, 512)
       │
       ▼
Linear CTC Projection Head (B, T, |V|)
       │
       ▼
Native CTCLoss(blank=0, zero_infinity=True)
       │
       ▼
Greedy / Prefix Collapse Decoding -> Predicted Gloss Sequence
```

---

## 2. Mathematical CTC Objectives & Decoding

Given per-frame logits $\mathbf{z}_t \in \mathbb{R}^{|\mathcal{V}|}$, the softmax probabilities are:

$$P(\pi_t = k \mid \mathbf{x}) = \frac{\exp(z_{t, k})}{\sum_{j} \exp(z_{t, j})}$$

The CTC objective aligns frame predictions with target token sequences $\mathbf{Y} = (y_1, \dots, y_U)$ by marginalizing across all valid collapsed paths:

$$\mathcal{L}_{\text{CTC}} = -\ln \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{Y})} \prod_{t=1}^T P(\pi_t \mid \mathbf{x})$$

Greedy decoding collapses repeated tokens and removes blank index `0`:
`[0, 2, 2, 0, 4, 0, 4] -> [2, 4, 4]`

---

## 3. Training Safeguards & RTX 3050 Hardware Optimization
- **AMP Mixed Precision**: FP16 autocasting via `torch.amp.autocast("cuda")`.
- **Gradient Accumulation**: Effective batch size $B_{\text{eff}} = B \times 4$.
- **Gradient Norm Clipping**: Maximum $L_2$ norm clipped to $1.0$.
- **Strict Feasibility Constraint**: Automatically checks $T_{\text{in}} \ge T_{\text{target}}$ prior to backward passes.
