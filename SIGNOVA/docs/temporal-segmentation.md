# Temporal Segmentation & Action Boundary Specification: SIGNOVA Phase 4

## 1. Scientific Principles & Boundary Truth

In continuous sign language, temporal boundaries delineate the transition between distinct lexical signs, fingerspelled units, and non-signing/rest positions.

> [!CRITICAL]
> **Research Guardrail**:
> If ground-truth temporal boundary annotations are absent in the dataset:
> - Supervised boundary heads are **NOT** trained on pseudo-labels.
> - Kinematic motion changes are strictly treated as **heuristic candidate boundary proposals**.
> - Candidate proposals are used **exclusively for diagnostic visualization and exploratory analysis**, never as training targets for CTC or segmentation models.

---

## 2. Heuristic Candidate Boundary Detector

Implemented in [`HeuristicBoundaryDetector`](file:///g:/SingLang/SIGNOVA/src/signova/models/boundary_head.py):

1. **Inter-Frame Kinematic Velocity**:
   $$v_t = \|\mathbf{x}_t - \mathbf{x}_{t-1}\|_2$$
2. **Temporal Gaussian / Moving Average Smoothing**:
   $$\tilde{v}_t = \frac{1}{W} \sum_{k=-w}^w v_{t+k}$$
3. **Thresholded Candidate State Extraction**:
   - Active gesture interval: $\tilde{v}_t \ge \theta_{\text{energy}}$.
   - Minimum candidate duration: $\Delta t \ge 6\text{ frames}$ ($0.2\text{s}$).

---

## 3. Modular Boundary Head Infrastructure

For future annotated datasets (Phase 5+), [`TemporalBoundaryHead`](file:///g:/SingLang/SIGNOVA/src/signova/models/boundary_head.py) provides a 3-state frame classifier:
- State 0: Non-sign / Rest / Background
- State 1: Sign Active
- State 2: Boundary Transition (Epenthesis)
