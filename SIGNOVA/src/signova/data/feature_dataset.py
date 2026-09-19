"""
PyTorch Feature Dataset for SIGNOVA Landmark Sequences.

Implements lazy on-demand loading of compressed .npz skeletal landmark archives,
semantic landmark group selection, class label encoding, and session provenance preservation.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from signova.features.feature_groups import LandmarkGroup, slice_landmark_tensor
from signova.features.storage import load_landmark_features


class SignLanguageFeatureDataset(Dataset):
    """
    Lazy PyTorch Dataset for loading extracted .npz sign language landmark sequences.
    """

    def __init__(
        self,
        manifest: Union[str, Path, pd.DataFrame],
        split: Optional[str] = None,
        landmark_group: Union[str, LandmarkGroup] = LandmarkGroup.FULL,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            manifest: Path to manifest CSV or pre-loaded pandas DataFrame.
            split: Optional split filter ('train', 'val', 'test', or None for all).
            landmark_group: Semantic landmark group to extract (FULL, HANDS, HANDS_POSE, POSE, FACE).
            transform: Optional feature transform callable.
        """
        if isinstance(manifest, (str, Path)):
            self.df = pd.read_csv(manifest)
        elif isinstance(manifest, pd.DataFrame):
            self.df = manifest.copy()
        else:
            raise TypeError(f"Unsupported manifest type: {type(manifest)}")

        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.landmark_group = landmark_group
        self.transform = transform

        # Ensure required columns exist
        if "feature_path" not in self.df.columns:
            raise ValueError("Manifest DataFrame must contain a 'feature_path' column.")
        if "class_id" not in self.df.columns:
            raise ValueError("Manifest DataFrame must contain a 'class_id' column.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        feature_path = Path(row["feature_path"])

        if not feature_path.is_file():
            raise FileNotFoundError(f"Feature archive missing: {feature_path}")

        # Lazy load .npz container
        raw_lm, det_mask, timestamps, frame_indices, meta = load_landmark_features(feature_path)

        # Slice semantic landmark group
        sliced_lm = slice_landmark_tensor(raw_lm, group_name=self.landmark_group)

        # Convert to PyTorch tensors
        feat_tensor = torch.from_numpy(sliced_lm.astype(np.float32))     # (T, num_joints, 3)
        mask_tensor = torch.from_numpy(det_mask.astype(np.float32))      # (T, 4)
        ts_tensor = torch.from_numpy(timestamps.astype(np.float32))      # (T,)
        fi_tensor = torch.from_numpy(frame_indices.astype(np.int64))     # (T,)
        label_tensor = torch.tensor(int(row["class_id"]), dtype=torch.long)

        if self.transform is not None:
            feat_tensor = self.transform(feat_tensor)

        return {
            "features": feat_tensor,
            "detection_mask": mask_tensor,
            "timestamps_ms": ts_tensor,
            "frame_indices": fi_tensor,
            "label": label_tensor,
            "sample_id": str(row["sample_id"]),
            "session_id": str(row.get("session_id", "unknown")),
            "split": str(row.get("split", "unassigned")),
            "sequence_length": feat_tensor.shape[0],
            "class_name": str(row.get("class_name", "")),
        }
