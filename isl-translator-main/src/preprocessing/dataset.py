"""
PyTorch Dataset for ISL Sign Language Recognition.

Handles loading keypoint sequences and labels for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable
import json
import logging

from .data_augmentation import DataAugmenter

logger = logging.getLogger(__name__)


class ISLDataset(Dataset):
    """
    Dataset for Indian Sign Language keypoint sequences.
    
    Expects:
    - Keypoint files: .npz files with 'keypoints' array of shape (T, 543, 3)
    - Annotations: JSON file mapping video IDs to gloss sequences and text
    """
    
    def __init__(
        self,
        data_dir: str,
        annotations_file: str,
        vocab: Optional[Dict[str, int]] = None,
        max_seq_length: int = 512,
        min_seq_length: int = 16,
        augmenter: Optional[DataAugmenter] = None,
        mode: str = "train",
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing .npz keypoint files
            annotations_file: Path to JSON annotations file
            vocab: Gloss vocabulary mapping {gloss: index}
            max_seq_length: Maximum sequence length (pad/truncate)
            min_seq_length: Minimum sequence length (filter out shorter)
            augmenter: DataAugmenter instance for training augmentation
            mode: "train", "val", or "test"
        """
        self.data_dir = Path(data_dir)
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.augmenter = augmenter if mode == "train" else None
        self.mode = mode
        
        # Load annotations
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # Build or use provided vocabulary
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
        
        self.idx_to_gloss = {v: k for k, v in self.vocab.items()}
        
        # Filter samples
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {mode} split")
        logger.info(f"Vocabulary size: {len(self.vocab)}")
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from annotations."""
        vocab = {"<blank>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        
        for sample in self.annotations:
            glosses = sample.get("glosses", [])
            for gloss in glosses:
                if gloss not in vocab:
                    vocab[gloss] = len(vocab)
        
        return vocab
    
    def _load_samples(self) -> List[Dict]:
        """Load and filter valid samples."""
        samples = []
        
        for ann in self.annotations:
            video_id = ann["video_id"]
            keypoint_path = self.data_dir / f"{video_id}.npz"
            
            if not keypoint_path.exists():
                logger.warning(f"Keypoint file not found: {keypoint_path}")
                continue
            
            samples.append({
                "video_id": video_id,
                "keypoint_path": str(keypoint_path),
                "glosses": ann.get("glosses", []),
                "text_hindi": ann.get("text_hindi", ""),
                "text_english": ann.get("text_english", ""),
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load keypoints
        data = np.load(sample["keypoint_path"])
        keypoints = data["keypoints"]  # (T, 543, 3)
        
        # Apply augmentation if training
        if self.augmenter is not None:
            keypoints = self.augmenter(keypoints)
        
        # Get sequence length before padding
        seq_length = len(keypoints)
        
        # Truncate if too long
        if seq_length > self.max_seq_length:
            keypoints = keypoints[:self.max_seq_length]
            seq_length = self.max_seq_length
        
        # Pad if too short
        if seq_length < self.max_seq_length:
            padding = np.zeros(
                (self.max_seq_length - seq_length, keypoints.shape[1], keypoints.shape[2]),
                dtype=np.float32
            )
            keypoints = np.concatenate([keypoints, padding], axis=0)
        
        # Convert glosses to indices
        gloss_indices = [self.vocab.get(g, self.vocab["<unk>"]) for g in sample["glosses"]]
        gloss_length = len(gloss_indices)
        
        # Pad gloss indices
        max_gloss_length = 100  # Max gloss sequence length
        if len(gloss_indices) < max_gloss_length:
            gloss_indices = gloss_indices + [0] * (max_gloss_length - len(gloss_indices))
        else:
            gloss_indices = gloss_indices[:max_gloss_length]
            gloss_length = max_gloss_length
        
        return {
            "video_id": sample["video_id"],
            "keypoints": torch.tensor(keypoints, dtype=torch.float32),
            "seq_length": torch.tensor(seq_length, dtype=torch.long),
            "glosses": torch.tensor(gloss_indices, dtype=torch.long),
            "gloss_length": torch.tensor(gloss_length, dtype=torch.long),
            "text_hindi": sample["text_hindi"],
            "text_english": sample["text_english"],
        }
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)
    
    def decode_glosses(self, indices: torch.Tensor) -> List[str]:
        """Convert gloss indices back to gloss strings."""
        glosses = []
        for idx in indices:
            idx = idx.item()
            if idx == 0:  # <blank>
                continue
            glosses.append(self.idx_to_gloss.get(idx, "<unk>"))
        return glosses


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched tensors
    """
    video_ids = [sample["video_id"] for sample in batch]
    keypoints = torch.stack([sample["keypoints"] for sample in batch])
    seq_lengths = torch.stack([sample["seq_length"] for sample in batch])
    glosses = torch.stack([sample["glosses"] for sample in batch])
    gloss_lengths = torch.stack([sample["gloss_length"] for sample in batch])
    text_hindi = [sample["text_hindi"] for sample in batch]
    text_english = [sample["text_english"] for sample in batch]
    
    return {
        "video_ids": video_ids,
        "keypoints": keypoints,
        "seq_lengths": seq_lengths,
        "glosses": glosses,
        "gloss_lengths": gloss_lengths,
        "text_hindi": text_hindi,
        "text_english": text_english,
    }


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    train_annotations: str,
    val_annotations: str,
    batch_size: int = 8,
    num_workers: int = 4,
    augmentation_config: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_dir: Directory with training keypoints
        val_dir: Directory with validation keypoints
        train_annotations: Path to training annotations
        val_annotations: Path to validation annotations
        batch_size: Batch size
        num_workers: Number of data loading workers
        augmentation_config: Augmentation parameters
        
    Returns:
        train_loader, val_loader, vocab
    """
    # Create augmenter for training
    augmenter = None
    if augmentation_config:
        augmenter = DataAugmenter(**augmentation_config)
    
    # Create training dataset
    train_dataset = ISLDataset(
        data_dir=train_dir,
        annotations_file=train_annotations,
        augmenter=augmenter,
        mode="train",
    )
    
    # Create validation dataset with same vocab
    val_dataset = ISLDataset(
        data_dir=val_dir,
        annotations_file=val_annotations,
        vocab=train_dataset.vocab,
        mode="val",
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader, train_dataset.vocab


if __name__ == "__main__":
    # Test dataset loading
    print("Testing ISLDataset...")
    
    # Create dummy annotation file
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy keypoint file
        dummy_kp = np.random.rand(100, 543, 3).astype(np.float32)
        np.savez(os.path.join(tmpdir, "test_video.npz"), keypoints=dummy_kp)
        
        # Create dummy annotation
        annotations = [
            {
                "video_id": "test_video",
                "glosses": ["HELLO", "WORLD"],
                "text_hindi": "नमस्ते दुनिया",
                "text_english": "Hello World",
            }
        ]
        ann_path = os.path.join(tmpdir, "annotations.json")
        with open(ann_path, 'w') as f:
            json.dump(annotations, f)
        
        # Test dataset
        dataset = ISLDataset(
            data_dir=tmpdir,
            annotations_file=ann_path,
            mode="train",
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Vocab size: {dataset.get_vocab_size()}")
        
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Keypoints shape: {sample['keypoints'].shape}")
        print(f"Glosses: {dataset.decode_glosses(sample['glosses'])}")
        
        print("All tests passed!")
