"""
Continuous Sign Language Feature Dataset for SIGNOVA.

Implements lazy loading for variable-length continuous skeletal sequences,
supporting anatomical landmark groups, detection masks, timestamps, and
optional sequential targets for CTC verification.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from signova.features.feature_groups import LandmarkGroup, slice_landmark_tensor
from signova.features.storage import load_landmark_features


class ContinuousSignDataset(Dataset):
    """
    Lazy PyTorch Dataset for loading continuous sign language landmark sequences.
    """

    def __init__(
        self,
        manifest: Union[str, Path, pd.DataFrame],
        split: Optional[str] = None,
        landmark_group: Union[str, LandmarkGroup] = LandmarkGroup.FULL,
        transform: Optional[Callable] = None,
        require_targets: bool = False,
    ):
        """
        Args:
            manifest: Path to manifest CSV/JSON or pre-loaded DataFrame.
            split: Optional split filter ('train', 'val', 'test', or None).
            landmark_group: Semantic landmark group to extract (FULL, HANDS, HANDS_POSE, etc.).
            transform: Optional feature transform callable.
            require_targets: If True, raises ValueError when sequential targets are missing.
        """
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if manifest_path.suffix.lower() == ".json":
                self.df = pd.read_json(manifest_path)
            else:
                self.df = pd.read_csv(manifest_path)
        elif isinstance(manifest, pd.DataFrame):
            self.df = manifest.copy()
        else:
            raise TypeError(f"Unsupported manifest type: {type(manifest)}")

        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.landmark_group = landmark_group
        self.transform = transform
        self.require_targets = require_targets

        if "feature_path" not in self.df.columns:
            raise ValueError("Continuous manifest must contain a 'feature_path' column.")

        if self.require_targets and "target_sequence" not in self.df.columns:
            raise ValueError("Manifest missing required 'target_sequence' column for supervised sequential mode.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        feature_path = Path(row["feature_path"])

        if not feature_path.is_file():
            raise FileNotFoundError(f"Continuous feature archive missing: {feature_path}")

        # Lazy load .npz container
        raw_lm, det_mask, timestamps, frame_indices, meta = load_landmark_features(feature_path)

        # Slice semantic anatomical landmark group
        sliced_lm = slice_landmark_tensor(raw_lm, group_name=self.landmark_group)

        feat_tensor = torch.from_numpy(sliced_lm.astype(np.float32))     # (T, num_joints, 3)
        mask_tensor = torch.from_numpy(det_mask.astype(np.float32))      # (T, 4)
        ts_tensor = torch.from_numpy(timestamps.astype(np.float32))      # (T,)
        fi_tensor = torch.from_numpy(frame_indices.astype(np.int64))     # (T,)

        if self.transform is not None:
            feat_tensor = self.transform(feat_tensor)

        # Parse optional targets (e.g. for synthetic fixtures or future annotated datasets)
        target_seq = None
        if "target_sequence" in row and pd.notna(row["target_sequence"]):
            val = row["target_sequence"]
            if isinstance(val, str):
                try:
                    import ast
                    parsed = ast.literal_eval(val)
                    target_seq = torch.tensor(parsed, dtype=torch.long)
                except Exception:
                    # Comma-separated list of ints
                    target_seq = torch.tensor([int(x.strip()) for x in val.split(",") if x.strip().isdigit()], dtype=torch.long)
            elif isinstance(val, (list, tuple)):
                target_seq = torch.tensor(val, dtype=torch.long)

        # Optional boundary targets if genuinely available
        boundary_targets = None
        if "boundary_targets" in row and pd.notna(row["boundary_targets"]):
            bval = row["boundary_targets"]
            if isinstance(bval, (list, tuple)):
                boundary_targets = torch.tensor(bval, dtype=torch.float32)

        return {
            "features": feat_tensor,
            "detection_mask": mask_tensor,
            "timestamps_ms": ts_tensor,
            "frame_indices": fi_tensor,
            "sample_id": str(row.get("sample_id", f"sample_{index}")),
            "session_id": str(row.get("session_id", "unknown")),
            "signer_id": str(row.get("signer_id", "unverified")),
            "split": str(row.get("split", "unassigned")),
            "sequence_length": feat_tensor.shape[0],
            "target_sequence": target_seq,
            "boundary_targets": boundary_targets,
        }


class ContinuousPadCollate:
    """
    Collate function for continuous variable-length sequences.
    Pads features to max sequence length in the batch and constructs padding validity masks.
    """

    def __init__(self, max_length: Optional[int] = None, flatten_landmarks: bool = False):
        self.max_length = max_length
        self.flatten_landmarks = flatten_landmarks

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Empty batch passed to ContinuousPadCollate")

        raw_lengths = [item["features"].shape[0] for item in batch]
        if self.max_length is not None:
            lengths = [min(l, self.max_length) for l in raw_lengths]
        else:
            lengths = raw_lengths

        B = len(batch)
        T_max = max(lengths) if lengths else 0
        num_joints = batch[0]["features"].shape[1]
        coords = batch[0]["features"].shape[2]

        padded_features = torch.zeros((B, T_max, num_joints, coords), dtype=torch.float32)
        padded_det_masks = torch.zeros((B, T_max, 4), dtype=torch.float32)
        padded_timestamps = torch.zeros((B, T_max), dtype=torch.float32)
        padding_mask = torch.zeros((B, T_max), dtype=torch.bool)  # True = valid frame, False = padding

        sample_ids = []
        session_ids = []
        signer_ids = []
        target_sequences = []
        target_lengths = []
        has_targets = False

        for i, item in enumerate(batch):
            seq_len = min(item["features"].shape[0], T_max)
            padded_features[i, :seq_len] = item["features"][:seq_len]
            padded_det_masks[i, :seq_len] = item["detection_mask"][:seq_len]
            padded_timestamps[i, :seq_len] = item["timestamps_ms"][:seq_len]
            padding_mask[i, :seq_len] = True

            sample_ids.append(item["sample_id"])
            session_ids.append(item["session_id"])
            signer_ids.append(item["signer_id"])

            if item["target_sequence"] is not None:
                has_targets = True
                target_sequences.append(item["target_sequence"])
                target_lengths.append(len(item["target_sequence"]))

        if self.flatten_landmarks:
            padded_features = padded_features.view(B, T_max, num_joints * coords)

        result = {
            "features": padded_features,
            "padding_mask": padding_mask,
            "detection_mask": padded_det_masks,
            "timestamps_ms": padded_timestamps,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "sample_ids": sample_ids,
            "session_ids": session_ids,
            "signer_ids": signer_ids,
        }

        if has_targets and len(target_sequences) == B:
            # Concatenate targets for CTC loss or pack into padded tensor
            max_target_len = max(target_lengths) if target_lengths else 0
            padded_targets = torch.zeros((B, max_target_len), dtype=torch.long)
            for i, tgt in enumerate(target_sequences):
                padded_targets[i, :len(tgt)] = tgt
            result["targets"] = padded_targets
            result["target_lengths"] = torch.tensor(target_lengths, dtype=torch.long)
        else:
            result["targets"] = None
            result["target_lengths"] = None

        return result
