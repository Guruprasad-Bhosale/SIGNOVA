#!/usr/bin/env python
"""
Training Script for ISL Translator.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --resume checkpoints/latest.pt
"""

import argparse
import logging
from pathlib import Path
import sys
import yaml
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.dataset import ISLDataset, create_dataloaders, collate_fn
from src.preprocessing.data_augmentation import DataAugmenter
from src.models.isl_model import ISLTranslator, create_model
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train ISL Translator")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--finetune", type=str, default=None, help="Fine-tune from pre-trained encoder weights")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.data_dir:
        config["paths"]["processed_dir"] = args.data_dir
    
    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("Using CPU (training will be slow)")
    
    # Create data loaders
    logger.info("Loading datasets...")
    
    data_dir = config["paths"]["processed_dir"]
    train_annotations = Path(data_dir) / "train_annotations.json"
    val_annotations = Path(data_dir) / "val_annotations.json"
    
    # Check if data exists
    if not train_annotations.exists():
        logger.error(f"Training annotations missing at {train_annotations}. Run extract_keypoints.py first.")
        return
    
    # Augmentation config
    aug_config = config.get("augmentation", {})
    augmenter = None
    if aug_config.get("enabled", True):
        augmenter = DataAugmenter(
            speed_range=tuple(aug_config.get("speed_range", [0.8, 1.2])),
            rotation_range=tuple(aug_config.get("rotation_range", [-15, 15])),
            scale_range=tuple(aug_config.get("scale_range", [0.9, 1.1])),
            gaussian_noise_std=aug_config.get("gaussian_noise", 0.01),
            landmark_dropout_prob=aug_config.get("landmark_dropout", 0.1),
        )
    
    # Create datasets
    train_dataset = ISLDataset(
        data_dir=str(Path(data_dir) / "train"),
        annotations_file=str(train_annotations),
        augmenter=augmenter,
        mode="train",
    )
    
    val_dataset = ISLDataset(
        data_dir=str(Path(data_dir) / "val"),
        annotations_file=str(val_annotations),
        vocab=train_dataset.vocab,
        mode="val",
    )
    
    # Create dataloaders
    train_config = config.get("training", {})
    batch_size = train_config.get("batch_size", 8)
    num_workers = train_config.get("num_workers", 4)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Vocabulary size: {train_dataset.get_vocab_size()}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(train_dataset.vocab, config)
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=args.output_dir,
    )
    
    # Train
    logger.info("Starting training...")
    best_wer = trainer.train(resume_from=args.resume, finetune_from=args.finetune)
    
    logger.info(f"Training complete! Best WER: {best_wer:.2%}")
    logger.info(f"Best model saved to: {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
