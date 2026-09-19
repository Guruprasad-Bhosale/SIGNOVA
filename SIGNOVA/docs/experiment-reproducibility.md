# SIGNOVA Experiment Reproducibility & Provenance Standard

## Overview

Every machine learning experiment conducted within SIGNOVA must be 100% reproducible from deterministic seeds, configuration files, manifest snapshots, and code revisions.

---

## 1. Environment & Software Stack

- **Operating System**: Windows 11 (PowerShell)
- **Python**: 3.12.10 (AMD64)
- **PyTorch**: 2.14.0
- **Scikit-learn**: 1.9.1
- **MediaPipe**: 0.10.35 (Holistic 543 Topology)
- **Target GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)
- **Random Seed**: 42 (enforced across Python `random`, NumPy `np.random`, and PyTorch `torch.manual_seed`)

---

## 2. Artifact Provenance Hierarchy

Every experiment run automatically produces an immutable artifact bundle under `models/experiments/<experiment_id>/`:

```
models/experiments/<experiment_id>/
├── config.yaml               # Complete hyperparameters, dataset paths, feature groups
├── model_summary.json        # Parameter counts, layer dimensions, memory estimates
├── training.log              # Timestamped epoch-by-epoch loss and evaluation log
├── metrics.json              # Final best-epoch validation metrics and test split results
├── test_confusion_matrix.csv # Confusion matrix on held-out test split
├── best.pt                   # Checkpoint weights at peak validation Macro F1
└── last.pt                   # Final epoch checkpoint weights
```

---

## 3. Command Reproduction Template

To reproduce any baseline experiment identically:

```powershell
# Reproduce Baseline A (BiGRU Full Body)
python scripts/train_recognizer.py `
    --model baseline_rnn `
    --landmark-group full `
    --batch-size 4 `
    --grad-accum 4 `
    --epochs 15 `
    --lr 0.0003 `
    --manifest data/manifests/phase3_manifest.csv `
    --exp-id repro_baseline_rnn_full

# Evaluate held-out test split
python scripts/evaluate_recognizer.py `
    --checkpoint models/experiments/repro_baseline_rnn_full/best.pt `
    --manifest data/manifests/phase3_manifest.csv `
    --split test
```
