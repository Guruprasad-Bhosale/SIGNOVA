# CTC Architecture & Readiness Specification: SIGNOVA Phase 4

## 1. Mathematical Formulation of CTC

Connectionist Temporal Classification (CTC) maps an unsegmented, variable-length frame representation sequence $\mathbf{H} = (\mathbf{h}_1, \dots, \mathbf{h}_T)$ to a shorter target label sequence $\mathbf{Y} = (y_1, \dots, y_U)$ where $U \le T$.

Given alphabet $\mathcal{V}$ and blank token $\epsilon \notin \mathcal{V}$, the augmented vocabulary is $\mathcal{V}' = \mathcal{V} \cup \{\epsilon\}$.

The probability of an alignment path $\pi = (\pi_1, \dots, \pi_T) \in \mathcal{V}'^T$ is:

$$P(\pi \mid \mathbf{X}) = \prod_{t=1}^T P(\pi_t \mid \mathbf{X})$$

The conditional probability of the target sequence $\mathbf{Y}$ marginalizes over all valid paths under the collapse operator $\mathcal{B}$:

$$P(\mathbf{Y} \mid \mathbf{X}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{Y})} P(\pi \mid \mathbf{X})$$

The loss is the negative log-likelihood:

$$\mathcal{L}_{\text{CTC}} = -\ln P(\mathbf{Y} \mid \mathbf{X})$$

---

## 2. SIGNOVA CTC Model Architecture

The continuous CTC model is implemented in [`CTCContinuousRecognizer`](file:///g:/SingLang/SIGNOVA/src/signova/models/ctc_recognizer.py):

$$\text{Landmarks } (B, T, 75, 3) \xrightarrow{\text{Projection}} (B, T, 256) \xrightarrow{\text{BiGRU/TCN}} (B, T, 512) \xrightarrow{\text{Linear Head}} (B, T, |\mathcal{V}'|)$$

- **Blank Index**: Explicitly set to index `0`.
- **Greedy Decoding**: Collapses identical consecutive tokens and removes blank tokens:
  `[0, 1, 1, 0, 2, 2, 0, 3] -> [1, 2, 3]`.
- **Constraint Enforcement**: Strictly requires $T \ge U$; fails loudly via [`check_ctc_data.py`](file:///g:/SingLang/SIGNOVA/scripts/check_ctc_data.py) if $T < U$.

---

## 3. Current Supervision Status & Blocker

| Component | Status | Empirical Note |
|---|---|---|
| **CTC Architecture** | ✅ Implemented & Unit-Tested | Passed synthetic fixture verification. |
| **Greedy Decoder & TER Metrics** | ✅ Implemented & Unit-Tested | Levenshtein token error rate and alignment verified. |
| **Pre-Flight Validator** | ✅ Implemented (`check_ctc_data.py`) | Flags missing supervision and infeasible sequence lengths. |
| **Real-Data Training** | 🛑 **BLOCKED** | ISLTranslate provides English translations, not aligned sign glosses. |
