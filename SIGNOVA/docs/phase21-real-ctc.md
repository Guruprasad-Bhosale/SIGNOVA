# SIGNOVA Phase 21: First Genuine Sequential ISL CTC Training, Evaluation & Live-Model Integration

## 1. Executive Summary

Phase 21 establishes the complete mathematical, operational, and architectural infrastructure required to advance SIGNOVA from diagnostic perception (`STATE_B`, `REAL_CTC = BLOCKED`, `LIVE_CAMERA = OPERATIONAL_DIAGNOSTIC_MODE`) toward continuous recognition (`STATE_A_DATA_LIMITED` or `STATE_A`).

Under SIGNOVA's core scientific principle:
$$\text{GENUINE HUMAN SUPERVISION} > \text{DATA QUALITY} > \text{SCIENTIFIC VALIDITY} > \text{MODEL TRAINING}$$

Real model training is strictly contingent upon genuine human annotations verified by the canonical Phase 19 readiness gate. When training-eligible human data is absent or insufficient, Phase 21 executes an immediate **NO-DATA Fast Exit**, ensuring zero synthetic labels, zero ungrounded checkpoints, and zero hallucinated live translations.

---

## 2. Architecture & Pipeline

### 2.1 The 7-Stage End-to-End Execution Flow

```
1. CANONICAL GATE CHECK & IMMUTABLE SNAPSHOT
   (phase19_gate_snapshot.json created)
              │
              ▼
   [Is Real CTC Authorized?]
      ├── NO  ──► [NO-DATA FAST EXIT: STATE_B, REAL_CTC = BLOCKED]
      └── YES ──►
              │
              ▼
2. GENUINE DATASET CONSTRUCTION & FINGERPRINTING
   (SHA-256 verification, dataset_manifest.json, dataset_sha256)
              │
              ▼
3. GENUINE VOCABULARY GENERATION
   (<BLANK>=0, <UNK>=1, deterministic token mapping)
              │
              ▼
4. STRICT SPLIT HIERARCHY
   (SIGNER_INDEPENDENT ──► SESSION_INDEPENDENT ──► RANDOM [with explicit --allow-random-split warning])
              │
              ▼
5. SEQUENCE-LEVEL CTC FEASIBILITY
   (T_required = L + Σ I(y_i == y_{i+1}) <= T_features)
              │
              ▼
6. BiGRU CTC MODEL TRAINING & VERSIONED RUNS
   (models/experiments/phase21_real_ctc/run_<TIMESTAMP>_<TAG>/)
              │
              ▼
7. MULTI-STAGE LIVE MODEL INTEGRATION GATE
   (TRAINED ──► CHECKPOINT VERIFIED ──► HELD-OUT EVALUATED ──► INPUT SPEC MATCH ──► LIVE SMOKE TEST ──► LIVE_MODEL_AUTHORIZED)
```

---

## 3. Strict Split Hierarchy & Leakage Prevention

Phase 21 enforces the scientific split taxonomy:
1. **SIGNER_INDEPENDENT**: Required for research-grade publication claims. Zero signer overlap between train, val, and test partitions.
2. **SESSION_INDEPENDENT**: Fallback when multiple sessions exist per signer.
3. **RANDOM (WITH EXPLICIT WARNING)**: Permitted on limited pilot data only when `--allow-random-split` is explicitly passed. Marks scientific status as `LIMITED`.

---

## 4. Sequence CTC Feasibility Equation

For every sequential sample $i$, the CTC alignment constraint requires:
$$T_{\text{required}}^{(i)} = L^{(i)} + \sum_{j=1}^{L^{(i)}-1} \mathbb{I}(y_j^{(i)} == y_{j+1}^{(i)})$$
$$T_{\text{features}}^{(i)} \ge T_{\text{required}}^{(i)}$$

Where:
- $L^{(i)}$ is the number of tokens in the gloss target.
- $\mathbb{I}$ is the indicator function detecting consecutive identical tokens (which require an intermediate CTC blank token).
- $T_{\text{features}}^{(i)}$ is the number of valid temporal frames in the feature sequence.

---

## 5. Model Architecture & Training Specification

- **Input Dimension**: 150 (50 landmark coordinates: 33 Pose + 21 LH + 21 RH subset in `HANDS_POSE`)
- **Backbone**: 2-layer Bidirectional GRU (`hidden_dim=128`, `dropout=0.2`)
- **Loss**: PyTorch `nn.CTCLoss(blank=0, zero_infinity=True)`
- **Optimization**: Adam (`lr=1e-3`, `weight_decay=1e-4`, `gradient_clip=5.0`)
- **Model Input Specification**: Compatible with Phase 20 (`temporal_window=64`, `temporal_stride=16`, `landmark_topology=543`)

---

## 6. Live Model Abstention & Confidence Gating

To prevent hallucinations during live camera streaming, Phase 21 connects to Phase 20's `SignovaLiveRuntime` with dynamic confidence gating:
- **HIGH CONFIDENCE ($\ge 0.50$)**: Emit decoded gloss and translate to English sentence.
- **LOW CONFIDENCE ($< 0.50$)**: Emit `WAIT / UNCERTAIN` and abstain from translating.
- **LOW ACTIVITY / INACTIVE SIGNER**: Emit `—` and abstain from translation (`NO SIGN ACTIVITY`).

---

## 7. Python-First CLI Suite

```powershell
# 1. Non-modifying diagnostic inspection
python scripts/check_phase21.py

# 2. Genuine dataset preparation
python scripts/prepare_phase21_dataset.py

# 3. Genuine BiGRU CTC training
python scripts/train_phase21_ctc.py

# 4. Held-out test evaluation & error analysis
python scripts/evaluate_phase21_ctc.py

# 5. Checkpoint cryptographic provenance verification
python scripts/verify_phase21_checkpoint.py

# 6. Multi-stage live model authorization
python scripts/integrate_phase21_live_model.py

# 7. End-to-end Phase 21 verification report
python scripts/run_phase21_verification.py
```
