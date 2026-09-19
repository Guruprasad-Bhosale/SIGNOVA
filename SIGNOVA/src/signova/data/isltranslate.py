"""
Dataset adapter for ISLTranslate (ACL 2023).

Continuous Indian Sign Language sentence/phrase translation dataset with 31,222 pairs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from signova.data.base import DatasetAdapter
from signova.data.manifest import Manifest, ManifestEntry
from signova.data.models import SampleAvailability, SignSample
from signova.data.splits import deterministic_split


class ISLTranslateAdapter(DatasetAdapter):
    """
    Adapter for the ISLTranslate dataset with canonical SignSample contract.
    """

    def __init__(
        self,
        root_dir: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        video_root: Optional[Path] = None,
    ):
        super().__init__(root_dir=root_dir, name="ISLTranslate")
        if csv_path:
            self.csv_path = Path(csv_path).resolve()
        elif self.root_dir:
            self.csv_path = (self.root_dir / "data" / "ISLTranslate.csv").resolve()
        else:
            self.csv_path = None

        self.video_root = Path(video_root).resolve() if video_root else None
        self._cached_df: Optional[pd.DataFrame] = None

    def _load_df(self) -> pd.DataFrame:
        if self._cached_df is not None:
            return self._cached_df
        if not self.csv_path or not self.csv_path.is_file():
            raise FileNotFoundError(f"ISLTranslate CSV not found at: {self.csv_path}")

        df = pd.read_csv(self.csv_path, dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        self._cached_df = df
        return self._cached_df

    def discover(self) -> List[Dict[str, Any]]:
        df = self._load_df()
        records = []
        for _, row in df.iterrows():
            uid = str(row.get("uid", "")).strip()
            text = str(row.get("text", "")).strip()
            records.append({"uid": uid, "text": text})
        return records

    def validate(self) -> Dict[str, Any]:
        try:
            df = self._load_df()
            total = len(df)
            empty_uids = df["uid"].isna().sum() + (df["uid"] == "").sum()
            empty_texts = df["text"].isna().sum() + (df["text"].str.strip() == "").sum()
            return {
                "valid": True,
                "total_records": int(total),
                "empty_uids": int(empty_uids),
                "empty_translations": int(empty_texts),
                "csv_path": str(self.csv_path),
                "local_videos_available": bool(self.video_root and self.video_root.is_dir()),
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "csv_path": str(self.csv_path),
            }

    def build_manifest(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Manifest:
        df = self._load_df()
        raw_items = []
        seen_uids: Dict[str, int] = {}
        for _, row in df.iterrows():
            raw_uid = str(row.get("uid", "")).strip()
            text = str(row.get("text", "")).strip()
            if raw_uid in seen_uids:
                seen_uids[raw_uid] += 1
                unique_uid = f"{raw_uid}_dup{seen_uids[raw_uid]}"
            else:
                seen_uids[raw_uid] = 0
                unique_uid = raw_uid
            raw_items.append({"uid": unique_uid, "raw_uid": raw_uid, "text": text})

        train_set, val_set, test_set = deterministic_split(
            raw_items,
            id_getter=lambda x: x["uid"],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        manifest = Manifest()

        for split_name, subset in [("train", train_set), ("val", val_set), ("test", test_set)]:
            for item in subset:
                uid = item["uid"]
                raw_uid = item["raw_uid"]
                text = item["text"]
                session_id = raw_uid.split("--")[0] if "--" in raw_uid else (raw_uid.rsplit("-", 1)[0] if "-" in raw_uid else raw_uid)

                # Check if local video exists
                local_vid = None
                avail = SampleAvailability.REMOTE_ONLY
                if self.video_root:
                    potential_vid = self.video_root / f"{raw_uid}.mp4"
                    if potential_vid.is_file():
                        local_vid = str(potential_vid)
                        avail = SampleAvailability.BOTH

                entry = ManifestEntry(
                    sample_id=uid,
                    dataset="ISLTranslate",
                    split=split_name,
                    video_path=local_vid,
                    translation=text if text != "nan" else None,
                    signer_id=None,  # Explicitly None (unverified)
                    metadata={
                        "source": "ISLTranslate_v1",
                        "raw_uid": raw_uid,
                        "session_id": session_id,
                        "availability": avail.value,
                    },
                )
                manifest.append(entry)

        return manifest

    def get_sample(self, sample_id: str) -> Optional[SignSample]:
        df = self._load_df()
        raw_uid = sample_id.split("_dup")[0]
        matches = df[df["uid"] == raw_uid]
        if matches.empty:
            return None
        row = matches.iloc[0]
        uid = str(row["uid"]).strip()
        text = str(row["text"]).strip()
        session_id = uid.split("--")[0] if "--" in uid else (uid.rsplit("-", 1)[0] if "-" in uid else uid)

        return SignSample(
            sample_id=sample_id,
            dataset="ISLTranslate",
            video_reference=uid,
            split="unassigned",
            source_language="Indian Sign Language (ISL)",
            target_language="English",
            target_translation=text if text != "nan" else None,
            signer_id=None,
            session_id=session_id,
            availability=SampleAvailability.REMOTE_ONLY,
            metadata={"source": "ISLTranslate_v1"},
        )

    def get_split(self, split: str) -> List[SignSample]:
        manifest = self.build_manifest()
        filtered = manifest.filter_by_split(split)
        samples = []
        for e in filtered.entries:
            sample = self.get_sample(e.sample_id)
            if sample:
                sample.split = split
                samples.append(sample)
        return samples

    def summary(self) -> Dict[str, Any]:
        val = self.validate()
        if not val.get("valid"):
            return val
        df = self._load_df()
        unique_sessions = set()
        for uid in df["uid"].dropna():
            s = str(uid)
            sess = s.split("--")[0] if "--" in s else (s.rsplit("-", 1)[0] if "-" in s else s)
            unique_sessions.add(sess)

        return {
            "dataset_name": "ISLTranslate",
            "total_pairs": len(df),
            "unique_source_sessions": len(unique_sessions),
            "signer_metadata_status": "UNKNOWN (Unverified in raw CSV)",
            "task_type": "continuous_sentence_translation",
            "local_videos": 0,
            "remote_availability": "Hugging Face (Exploration-Lab/iSign)",
            "languages": {"source": "ISL", "target": "English"},
        }
