"""
Dataset adapter for INCLUDE (Indian Sign Language Dataset on Zenodo).

Isolated word-level ISL dataset with 263 sign categories.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from signova.data.base import DatasetAdapter
from signova.data.manifest import Manifest, ManifestEntry
from signova.data.splits import deterministic_split


class INCLUDEAdapter(DatasetAdapter):
    """
    Adapter for the INCLUDE word-level sign dataset.
    """

    def __init__(self, root_dir: Optional[Path] = None, metadata_csv: Optional[Path] = None):
        super().__init__(root_dir=root_dir, name="INCLUDE")
        self.metadata_csv = Path(metadata_csv).resolve() if metadata_csv else None

    def discover(self) -> List[Dict[str, Any]]:
        if not self.root_dir or not self.root_dir.is_dir():
            return []
        # Discover video files (.mp4) in subfolders
        videos = list(self.root_dir.glob("**/*.mp4"))
        records = []
        for v in videos:
            gloss_name = v.parent.name
            records.append({
                "sample_id": f"include_{v.stem}",
                "gloss": gloss_name,
                "video_path": str(v),
            })
        return records

    def validate(self) -> Dict[str, Any]:
        if not self.root_dir or not self.root_dir.is_dir():
            return {
                "valid": False,
                "error": f"INCLUDE root directory not found or not configured: {self.root_dir}",
                "status": "unmounted",
            }
        videos = self.discover()
        return {
            "valid": True,
            "total_videos_found": len(videos),
            "root_dir": str(self.root_dir),
            "status": "ready",
        }

    def build_manifest(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Manifest:
        records = self.discover()
        train_set, val_set, test_set = deterministic_split(
            records,
            id_getter=lambda x: x["sample_id"],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        manifest = Manifest()
        for split_name, subset in [("train", train_set), ("val", val_set), ("test", test_set)]:
            for item in subset:
                entry = ManifestEntry(
                    sample_id=item["sample_id"],
                    dataset="INCLUDE",
                    split=split_name,
                    video_path=item["video_path"],
                    gloss=item["gloss"],
                    metadata={"task": "isolated_sign_recognition"},
                )
                manifest.append(entry)

        return manifest

    def get_sample(self, sample_id: str) -> Optional[ManifestEntry]:
        records = self.discover()
        for r in records:
            if r["sample_id"] == sample_id:
                return ManifestEntry(
                    sample_id=r["sample_id"],
                    dataset="INCLUDE",
                    split="unassigned",
                    video_path=r["video_path"],
                    gloss=r["gloss"],
                )
        return None

    def summary(self) -> Dict[str, Any]:
        val = self.validate()
        if not val.get("valid"):
            return val
        records = self.discover()
        unique_glosses = set(r["gloss"] for r in records)
        return {
            "dataset_name": "INCLUDE",
            "total_videos": len(records),
            "unique_glosses": len(unique_glosses),
            "task_type": "isolated_sign_recognition",
        }
