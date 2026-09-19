"""
Variable-Length Sequence Collation and Padding Masks for SIGNOVA.

Pads batch sequences dynamically to the batch maximum length T_max,
produces binary validity masks (B, T_max) to prevent models from interpreting
zero-padding as movement, and bundles metadata for tracking.
"""

from typing import Any, Dict, List, Optional
import torch


class PadCollate:
    """
    Collate function that pads variable-length sequences to the max length in the batch.
    """

    def __init__(self, max_length: Optional[int] = None, flatten_landmarks: bool = False):
        """
        Args:
            max_length: Optional upper bound on sequence length. If sequence exceeds this, it is truncated.
            flatten_landmarks: If True, flattens (num_joints, 3) to (num_joints * 3).
        """
        self.max_length = max_length
        self.flatten_landmarks = flatten_landmarks

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Empty batch passed to PadCollate")

        lengths = [item["features"].shape[0] for item in batch]
        if self.max_length is not None:
            lengths = [min(l, self.max_length) for l in lengths]

        B = len(batch)
        T_max = max(lengths) if lengths else 0
        num_joints = batch[0]["features"].shape[1]
        coords = batch[0]["features"].shape[2]

        # Allocate padded tensors
        padded_features = torch.zeros((B, T_max, num_joints, coords), dtype=torch.float32)
        padded_det_masks = torch.zeros((B, T_max, 4), dtype=torch.float32)
        padding_mask = torch.zeros((B, T_max), dtype=torch.bool)  # True = valid frame, False = padding

        labels = torch.zeros(B, dtype=torch.long)
        sample_ids = []
        session_ids = []
        class_names = []

        for i, item in enumerate(batch):
            seq_len = min(item["features"].shape[0], T_max)
            padded_features[i, :seq_len] = item["features"][:seq_len]
            padded_det_masks[i, :seq_len] = item["detection_mask"][:seq_len]
            padding_mask[i, :seq_len] = True

            labels[i] = item["label"]
            sample_ids.append(item["sample_id"])
            session_ids.append(item["session_id"])
            class_names.append(item.get("class_name", ""))

        if self.flatten_landmarks:
            # (B, T_max, num_joints * 3)
            padded_features = padded_features.view(B, T_max, num_joints * coords)

        return {
            "features": padded_features,          # (B, T_max, num_joints, 3) or (B, T_max, num_joints * 3)
            "padding_mask": padding_mask,        # (B, T_max) bool (True=valid, False=pad)
            "detection_mask": padded_det_masks,  # (B, T_max, 4) float32
            "labels": labels,                    # (B,) long
            "lengths": torch.tensor(lengths, dtype=torch.long), # (B,) long
            "sample_ids": sample_ids,
            "session_ids": session_ids,
            "class_names": class_names,
        }
