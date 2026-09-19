"""
Manifest data structures and schema definitions for SIGNOVA.

Canonical format for continuous and isolated sign language datasets.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class ManifestEntry:
    """
    A single sign language sample entry in the canonical SIGNOVA manifest.
    """
    sample_id: str
    dataset: str
    split: str  # "train", "val", "test"
    video_path: Optional[str] = None
    annotation_path: Optional[str] = None
    features_path: Optional[str] = None
    gloss: Optional[str] = None
    translation: Optional[str] = None
    signer_id: Optional[str] = None
    duration_sec: Optional[float] = None
    num_frames: Optional[int] = None
    fps: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary for serialization."""
        d = asdict(self)
        if isinstance(d.get("metadata"), dict) and len(d["metadata"]) > 0:
            d["metadata"] = json.dumps(d["metadata"])
        else:
            d["metadata"] = ""
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ManifestEntry":
        """Create entry from dictionary."""
        d_copy = dict(d)
        if "metadata" in d_copy and isinstance(d_copy["metadata"], str) and d_copy["metadata"].strip():
            try:
                d_copy["metadata"] = json.loads(d_copy["metadata"])
            except json.JSONDecodeError:
                d_copy["metadata"] = {}
        elif "metadata" not in d_copy or not isinstance(d_copy["metadata"], dict):
            d_copy["metadata"] = {}
        return cls(**d_copy)


@dataclass
class Manifest:
    """
    Collection of manifest entries with export, filtering, and summary capabilities.
    """
    entries: List[ManifestEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def append(self, entry: ManifestEntry) -> None:
        self.entries.append(entry)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all entries to a pandas DataFrame."""
        rows = [e.to_dict() for e in self.entries]
        return pd.DataFrame(rows)

    def to_csv(self, file_path: Path) -> None:
        """Export manifest to CSV file."""
        df = self.to_dataframe()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False, encoding="utf-8")

    def to_json(self, file_path: Path) -> None:
        """Export manifest to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2, ensure_ascii=False)

    @classmethod
    def from_csv(cls, file_path: Path) -> "Manifest":
        """Load manifest from CSV file."""
        df = pd.read_csv(file_path, dtype=str)
        entries = []
        for _, row in df.iterrows():
            d = row.to_dict()
            # Convert numeric fields
            if pd.notna(d.get("duration_sec")):
                try:
                    d["duration_sec"] = float(d["duration_sec"])
                except (ValueError, TypeError):
                    d["duration_sec"] = None
            if pd.notna(d.get("num_frames")):
                try:
                    d["num_frames"] = int(float(d["num_frames"]))
                except (ValueError, TypeError):
                    d["num_frames"] = None
            if pd.notna(d.get("fps")):
                try:
                    d["fps"] = float(d["fps"])
                except (ValueError, TypeError):
                    d["fps"] = None
            entries.append(ManifestEntry.from_dict(d))
        return cls(entries=entries)

    def filter_by_split(self, split: str) -> "Manifest":
        """Return a new manifest filtered by split (train, val, test)."""
        return Manifest(entries=[e for e in self.entries if e.split == split])

    def summary(self) -> Dict[str, Any]:
        """Compute basic summary statistics of the manifest."""
        splits_count: Dict[str, int] = {}
        for e in self.entries:
            splits_count[e.split] = splits_count.get(e.split, 0) + 1

        return {
            "total_samples": len(self.entries),
            "splits": splits_count,
            "has_translations": sum(1 for e in self.entries if e.translation),
            "has_glosses": sum(1 for e in self.entries if e.gloss),
            "has_videos": sum(1 for e in self.entries if e.video_path),
            "has_features": sum(1 for e in self.entries if e.features_path),
        }
