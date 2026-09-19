# ISL Continuous Sign Language Translator

Continuous Indian Sign Language (ISL) recognition system using ST-GCN + Transformer with CTC decoding. Trained on the [INCLUDE dataset](https://zenodo.org/records/4010759) (263 word-level signs, 4,287 videos).

Achieved **30.77% WER** on continuous recognition using a pose-only pipeline (no raw video input).

## Architecture

1. **Keypoint extraction** - MediaPipe Holistic extracts 543 2D keypoints per frame (33 pose, 468 face, 42 hand).
2. **ST-GCN encoder** - Spatial graph convolutions over the skeleton structure.
3. **Transformer encoder** - Temporal sequence modeling across frames.
4. **CTC head** - Alignment-free gloss prediction via Connectionist Temporal Classification.
5. **IndicTrans2 (optional)** - Gloss-to-text translation (Hindi/English).

## Repository Structure

```
isl-translator/
├── src/
│   ├── preprocessing/       # Keypoint extraction, dataset loaders
│   ├── models/              # ST-GCN, Transformer, CTC decoder
│   └── training/            # Trainer, mixed-precision, metrics
├── notebooks/               # Kaggle training notebooks
├── scripts/                 # CLI entry points
└── configs/                 # Hyperparameter configs (YAML)
```

## Reproducibility

Training and keypoint extraction were run on Kaggle. Only the primary notebooks are included; resumption notebooks (due to Kaggle session timeouts) follow the same setup.

- `01_extract_keypoints.ipynb` - MediaPipe keypoint extraction from raw video.
- `02_pretraining_include.ipynb` - Full training pipeline on INCLUDE.
- `03_finetune_csltr.ipynb` - Fine-tuning for sentence-level translation.

## Local Inference

Requires a GPU with >= 8 GB VRAM (tested on RTX 4060).

```bash
git clone https://github.com/Sid-V5/isl-translator.git
cd isl-translator
pip install -r requirements.txt

python src/inference/demo.py --checkpoint checkpoints/best.pt
```

## Configuration

Hyperparameters are in `configs/config.yaml`. Training used cosine-annealed LR (down to 1e-6) with label smoothing disabled.

## License

MIT
