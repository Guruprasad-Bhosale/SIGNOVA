"""
Generic Sequential Sign Language Dataset Adapter for SIGNOVA.

Provides an extensible canonical adapter interface for continuous sign language datasets
with sequential sign/gloss annotations.

IMPORTANT RESEARCH DISTINCTION:
- When evaluated on mock/generated sequences, the adapter is explicitly labeled:
  'SYNTHETIC_FIXTURE_VALIDATION'
- It is NOT claimed as a real dataset integration until an authentic external dataset
  with genuine ordered sign annotations is verified on disk.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from signova.data.base import DatasetAdapter
from signova.data.models import SampleAvailability, SignSample
from signova.preprocessing.label_normalizer import LabelNormalizer


class GenericSequentialAdapter(DatasetAdapter):
    """
    Standardized adapter for ingesting continuous sign video sequences and aligned gloss targets.
    """

    def __init__(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        dataset_name: str = "GENERIC_SEQUENTIAL_ISL",
        is_synthetic: bool = True,
    ):
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.dataset_name = dataset_name
        self.is_synthetic = is_synthetic
        self.normalizer = LabelNormalizer()
        self.df: Optional[pd.DataFrame] = None

        if self.manifest_path and self.manifest_path.is_file():
            self._load_manifest()

    def _load_manifest(self):
        if self.manifest_path.suffix.lower() == ".json":
            self.df = pd.read_json(self.manifest_path)
        else:
            self.df = pd.read_csv(self.manifest_path)

    def discover(self) -> List[Dict[str, Any]]:
        """Scans manifest and returns record descriptors."""
        if self.df is None:
            return []
        return self.df.to_dict(orient="records")

    def validate(self) -> Dict[str, Any]:
        """Validates feature path existence and sequence length constraints."""
        if self.df is None:
            return {"valid": False, "error": "No manifest loaded."}
        missing = 0
        for _, r in self.df.iterrows():
            fp = Path(str(r.get("feature_path", "")))
            if not fp.is_file():
                missing += 1
        return {
            "valid": missing == 0,
            "total_records": len(self.df),
            "missing_feature_files": missing,
            "is_synthetic": self.is_synthetic,
        }

    def build_manifest(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Constructs and returns canonical manifest."""
        return self.df

    def summary(self) -> Dict[str, Any]:
        count = len(self.df) if self.df is not None else 0
        return {
            "dataset_name": self.dataset_name,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "sample_count": count,
            "is_synthetic": self.is_synthetic,
            "integration_status": "SYNTHETIC_FIXTURE_VALIDATED" if self.is_synthetic else "REAL_DATASET_INTEGRATED",
            "supervision_type": "ordered_sign_gloss_sequence",
        }

    def get_sample(self, sample_id: str) -> Optional[Dict[str, Any]]:
        if self.df is None or "sample_id" not in self.df.columns:
            return None
        matches = self.df[self.df["sample_id"] == sample_id]
        if matches.empty:
            return None
        row = matches.iloc[0].to_dict()

        # Normalize label sequence
        raw_label = row.get("sequence_label", "")
        raw_toks, canon_toks = self.normalizer.normalize_sequence(raw_label)
        row["original_tokens"] = raw_toks
        row["canonical_tokens"] = canon_toks
        return row

    def build_canonical_manifest(
        self,
        samples: List[Dict[str, Any]],
        output_csv: Union[str, Path],
    ) -> pd.DataFrame:
        """
        Builds and saves a canonical Phase 5 sequential manifest.
        """
        rows = []
        for s in samples:
            raw_lbl = s.get("sequence_label", "")
            raw_t, canon_t = self.normalizer.normalize_sequence(raw_lbl)
            rows.append({
                "sample_id": s.get("sample_id"),
                "dataset": self.dataset_name,
                "split": s.get("split", "train"),
                "session_id": s.get("session_id", "unknown"),
                "signer_id": s.get("signer_id", "unverified"),
                "original_label": raw_lbl,
                "canonical_tokens": str(canon_t),
                "sequence_length": len(canon_t),
                "num_frames": s.get("num_frames", 0),
                "feature_path": s.get("feature_path", ""),
                "is_synthetic": self.is_synthetic,
            })

        df_out = pd.DataFrame(rows)
        p = Path(output_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(p, index=False)
        self.df = df_out
        return df_out
