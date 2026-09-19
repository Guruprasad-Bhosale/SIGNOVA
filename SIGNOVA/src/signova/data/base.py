"""
Abstract Base Class for Dataset Adapters in SIGNOVA.

Decouples the core preprocessing and training pipeline from specific external dataset
folder formats and raw annotation conventions.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from signova.data.manifest import Manifest, ManifestEntry


class DatasetAdapter(ABC):
    """
    Abstract adapter defining the lifecycle and boundary for external datasets.
    """

    def __init__(self, root_dir: Optional[Path] = None, name: str = "base_dataset"):
        self.root_dir = Path(root_dir).resolve() if root_dir else None
        self.name = name

    @abstractmethod
    def discover(self) -> List[Dict[str, Any]]:
        """
        Scan and discover all available raw samples/records in the dataset source.

        Returns:
            List of raw sample descriptor dictionaries.
        """
        pass

    @abstractmethod
    def validate(self) -> Dict[str, Any]:
        """
        Verify the integrity, existence, and readability of the dataset assets.

        Returns:
            Dictionary with validation status, error count, and missing file lists.
        """
        pass

    @abstractmethod
    def build_manifest(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Manifest:
        """
        Construct a normalized SIGNOVA canonical Manifest object with deterministic splits.

        Args:
            train_ratio: Ratio for training split.
            val_ratio: Ratio for validation split.
            test_ratio: Ratio for test split.

        Returns:
            Manifest instance.
        """
        pass

    @abstractmethod
    def get_sample(self, sample_id: str) -> Optional[ManifestEntry]:
        """
        Retrieve a single sample descriptor by its unique identifier.

        Args:
            sample_id: Unique string ID.

        Returns:
            ManifestEntry or None if not found.
        """
        pass

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        """
        Provide a concise overview of dataset metrics (sample count, vocabulary size, format).

        Returns:
            Dictionary of summary metrics.
        """
        pass
