"""
Deterministic dataset splitting routines.
"""

import hashlib
from typing import List, Tuple, TypeVar

T = TypeVar("T")


def deterministic_split(
    items: List[T],
    id_getter,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed_salt: str = "signova_v1",
) -> Tuple[List[T], List[T], List[T]]:
    """
    Split items into train, validation, and test sets deterministically using SHA-256 hash.
    Ensures that adding new data does not reshuffle existing splits.

    Args:
        items: List of objects to split.
        id_getter: Function that returns a unique string ID for an item.
        train_ratio: Proportion of data for training (e.g. 0.8).
        val_ratio: Proportion of data for validation (e.g. 0.1).
        test_ratio: Proportion of data for testing (e.g. 0.1).
        seed_salt: Salt string for hashing.

    Returns:
        (train_items, val_items, test_items)
    """
    total = train_ratio + val_ratio + test_ratio
    train_norm = train_ratio / total
    val_norm = val_ratio / total

    train_items: List[T] = []
    val_items: List[T] = []
    test_items: List[T] = []

    for item in items:
        uid = str(id_getter(item))
        hash_val = hashlib.sha256(f"{seed_salt}_{uid}".encode("utf-8")).hexdigest()
        # Convert first 8 hex chars to float between 0 and 1
        num = int(hash_val[:8], 16) / 0xFFFFFFFF

        if num < train_norm:
            train_items.append(item)
        elif num < train_norm + val_norm:
            val_items.append(item)
        else:
            test_items.append(item)

    return train_items, val_items, test_items
