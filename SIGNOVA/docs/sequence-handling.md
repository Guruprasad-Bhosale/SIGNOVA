# SIGNOVA Variable-Length Sequence Handling & Padding Masks

## Overview

Sign language gestures inherently vary in temporal duration depending on signer cadence, sign complexity, and recording frame rate. SIGNOVA employs a dynamic collation strategy that avoids aggressive fixed-length distortion while providing explicit validity masks for temporal neural models.

---

## 1. Batch Collation Strategy (`PadCollate`)

Given a batch of $B$ variable-length landmark sequences with temporal lengths $\{T_1, T_2, \dots, T_B\}$:

1. **Dynamic Length Bound**: The batch maximum length is determined on the fly:
   $$T_{\max} = \max_{i=1}^B T_i$$
2. **Zero-Padding**: Sequences are right-padded with zeros to shape $(B, T_{\max}, N_{\text{joints}}, 3)$.
3. **Validity Padding Mask**: A boolean validity mask of shape $(B, T_{\max})$ is constructed:
   $$\mathbf{M}_{i, t} = \begin{cases} \text{True} & \text{if } t < T_i \text{ (valid gesture frame)} \\ \text{False} & \text{if } t \ge T_i \text{ (padding)} \end{cases}$$

---

## 2. Preventing Padding Misinterpretation

> [!IMPORTANT]
> Zero-padding $(0.0, 0.0, 0.0)$ in normalized coordinate space represents a valid anatomical location (the signer's center anchor). If processed naively, neural networks will interpret zero-padded tail frames as an abrupt physical gesture to the body center.

To eliminate this artifact:
- **Masked Temporal Pooling**: Average and max pooling operations strictly sum over valid frames using $\mathbf{M}$ and divide by sequence length $T_i$:
  $$\mathbf{h}_{\text{pooled}} = \frac{\sum_{t=1}^{T_{\max}} \mathbf{h}_t \cdot \mathbf{M}_t}{\sum_{t=1}^{T_{\max}} \mathbf{M}_t}$$
- **PyTorch PackPaddedSequence**: Recurrent backbones (BiGRU / BiLSTM) utilize packed sequences enforcing zero recurrence across padding steps.
- **Causal / Same Masked Convolutions**: TCN residual blocks mask out receptive field outputs that extend into padded regions.
