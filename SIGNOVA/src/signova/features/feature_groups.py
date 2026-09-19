"""
Centralized Landmark Group Definitions and Slicing Utilities for SIGNOVA.

Defines standardized anatomical joint subsets across the 543 MediaPipe Holistic topology:
- POSE: Indices 0..32 (33 joints)
- FACE: Indices 33..500 (468 joints)
- LEFT_HAND: Indices 501..521 (21 joints)
- RIGHT_HAND: Indices 522..542 (21 joints)
- HANDS: Indices 501..542 (42 joints)
- HANDS_POSE: Indices 0..32 + 501..542 (75 joints)
- FULL: Indices 0..542 (543 joints)
"""

from enum import Enum
from typing import Dict, List, Tuple, Union
import numpy as np
import torch

# Exact slice ranges and indices for the 543 Holistic topology
POSE_SLICE = slice(0, 33)        # 33 landmarks
FACE_SLICE = slice(33, 501)      # 468 landmarks
LEFT_HAND_SLICE = slice(501, 522)# 21 landmarks
RIGHT_HAND_SLICE = slice(522, 543)# 21 landmarks

# Index lists for composite groups
POSE_INDICES = list(range(0, 33))
FACE_INDICES = list(range(33, 501))
LEFT_HAND_INDICES = list(range(501, 522))
RIGHT_HAND_INDICES = list(range(522, 543))
HANDS_INDICES = LEFT_HAND_INDICES + RIGHT_HAND_INDICES                   # 42 landmarks
HANDS_POSE_INDICES = POSE_INDICES + LEFT_HAND_INDICES + RIGHT_HAND_INDICES # 75 landmarks
FULL_INDICES = list(range(0, 543))                                       # 543 landmarks


class LandmarkGroup(str, Enum):
    POSE = "pose"
    FACE = "face"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    HANDS = "hands"              # 42 joints
    HANDS_POSE = "hands_pose"    # 75 joints
    FULL = "full"                # 543 joints


LANDMARK_GROUP_INDICES: Dict[str, List[int]] = {
    LandmarkGroup.POSE.value: POSE_INDICES,
    LandmarkGroup.FACE.value: FACE_INDICES,
    LandmarkGroup.LEFT_HAND.value: LEFT_HAND_INDICES,
    LandmarkGroup.RIGHT_HAND.value: RIGHT_HAND_INDICES,
    LandmarkGroup.HANDS.value: HANDS_INDICES,
    LandmarkGroup.HANDS_POSE.value: HANDS_POSE_INDICES,
    LandmarkGroup.FULL.value: FULL_INDICES,
}

LANDMARK_GROUP_SIZES: Dict[str, int] = {
    LandmarkGroup.POSE.value: 33,
    LandmarkGroup.FACE.value: 468,
    LandmarkGroup.LEFT_HAND.value: 21,
    LandmarkGroup.RIGHT_HAND.value: 21,
    LandmarkGroup.HANDS.value: 42,
    LandmarkGroup.HANDS_POSE.value: 75,
    LandmarkGroup.FULL.value: 543,
}


def get_landmark_group_indices(group_name: Union[str, LandmarkGroup]) -> List[int]:
    """Return list of landmark indices for a given semantic group name."""
    name = group_name.value if isinstance(group_name, LandmarkGroup) else str(group_name).lower()
    if name not in LANDMARK_GROUP_INDICES:
        raise ValueError(
            f"Unknown landmark group '{name}'. Available groups: {list(LANDMARK_GROUP_INDICES.keys())}"
        )
    return LANDMARK_GROUP_INDICES[name]


def get_landmark_group_size(group_name: Union[str, LandmarkGroup]) -> int:
    """Return number of landmarks in a given semantic group."""
    name = group_name.value if isinstance(group_name, LandmarkGroup) else str(group_name).lower()
    if name not in LANDMARK_GROUP_SIZES:
        raise ValueError(
            f"Unknown landmark group '{name}'. Available groups: {list(LANDMARK_GROUP_SIZES.keys())}"
        )
    return LANDMARK_GROUP_SIZES[name]


def slice_landmark_tensor(
    tensor: Union[np.ndarray, torch.Tensor],
    group_name: Union[str, LandmarkGroup] = LandmarkGroup.FULL,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Extract specific landmark group from a landmark tensor.
    
    Supports:
    - (543, 3) single frame
    - (T, 543, 3) sequence
    - (B, T, 543, 3) batched sequence

    Returns sliced array/tensor with the selected number of joints along the landmark axis.
    """
    indices = get_landmark_group_indices(group_name)
    if isinstance(tensor, np.ndarray):
        if tensor.shape[-2] != 543:
            raise ValueError(f"Expected landmark axis size 543, got shape {tensor.shape}")
        return tensor[..., indices, :]
    elif isinstance(tensor, torch.Tensor):
        if tensor.shape[-2] != 543:
            raise ValueError(f"Expected landmark axis size 543, got shape {tensor.shape}")
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=tensor.device)
        return torch.index_select(tensor, dim=-2, index=idx_tensor)
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")
