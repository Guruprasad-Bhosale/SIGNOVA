# SIGNOVA Experiment Tracking & Protocols

## 1. Experiment Lifecycle

Every machine learning experiment conducted under SIGNOVA must follow these reproducibility guidelines:

1. **Config-Driven Runs**: Every run must reference a specific YAML configuration under `configs/` or `models/experiments/`.
2. **Deterministic Seeds**: Random seeds are explicitly set in `configs/base.yaml` (default: 42) for PyTorch, NumPy, and Python standard library.
3. **Artifact Output Structure**:
   - Checkpoints saved to `models/checkpoints/{experiment_name}_best.pt`.
   - Metrics logs saved to `outputs/metrics/{experiment_name}_metrics.json`.
   - Predictions saved to `outputs/predictions/{experiment_name}_val_preds.csv`.

---

## 2. Planned Experiment Baseline Matrix

| Exp ID | Architecture | Dataset | Modality | Target Metric (WER) |
|---|---|---|---|---|
| `EXP-01` | ST-GCN Baseline | INCLUDE (263 signs) | Pose (33) + Hands (42) | < 25.0% (Isolated Top-1) |
| `EXP-02` | ST-GCN + Transformer | INCLUDE | Holistic (543) | < 20.0% (Isolated Top-1) |
| `EXP-03` | ST-GCN + Transformer + CTC | ISLTranslate (Continuous) | Holistic (543) | < 35.0% WER |
| `EXP-04` | Full Pipeline (CTC + Seq2Seq NMT) | ISLTranslate | Continuous Signs -> English | > 0.45 BLEU-4 |
