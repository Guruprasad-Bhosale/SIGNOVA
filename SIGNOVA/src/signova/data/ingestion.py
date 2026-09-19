"""
Phase 10 Canonical Data Ingestion Module for SIGNOVA.

Transforms verified raw video and annotation data into the canonical SIGNOVA representation.
Enforces:
1. CANONICAL_INGESTED != TRAINING_READY (State C remains active).
2. Explicit classification into VALID, INVALID, REQUIRES_REVIEW without silent error patching.
3. Raw data immutability (reads from data/raw/phase10/, writes derived records to data/interim/phase10/).
"""

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from signova.data.provenance import compute_sha256
from signova.data.quality_audit import validate_gloss_token


class IngestionStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


@dataclass
class IngestedSample:
    dataset_id: str
    sample_id: str
    signer_id: str
    session_id: str
    video_id: str
    duration_sec: float
    frame_count: int
    fps: float
    resolution: str
    gloss_count: int
    token_count: int
    glosses: List[str]
    english_text: str
    status: str  # IngestionStatus
    failure_reason: str
    quality_status: str
    leakage_status: str
    provenance_id: str
    raw_sha256: str
    derived_sha256: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "sample_id": self.sample_id,
            "signer_id": self.signer_id,
            "session_id": self.session_id,
            "video_id": self.video_id,
            "duration": self.duration_sec,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "resolution": self.resolution,
            "gloss_count": self.gloss_count,
            "token_count": self.token_count,
            "glosses": self.glosses,
            "english_text": self.english_text,
            "annotation_status": self.status,
            "failure_reason": self.failure_reason,
            "quality_status": self.quality_status,
            "leakage_status": self.leakage_status,
            "provenance_id": self.provenance_id,
            "sha256": self.derived_sha256,
        }


class CanonicalIngestionEngine:
    """
    Ingests and normalizes raw dataset records into canonical schema.
    """

    def __init__(self, interim_dir: Path | str = "data/interim/phase10"):
        self.interim_dir = Path(interim_dir)
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    def ingest_sample(
        self,
        raw_record: Dict[str, Any],
        parent_provenance_id: str = "root",
        raw_sha256: str = "N/A",
    ) -> IngestedSample:
        sample_id = raw_record.get("sample_id") or raw_record.get("video_id") or "unknown_sample"
        dataset_id = raw_record.get("dataset_id", "isl_dataset")
        signer_id = raw_record.get("signer_id", "unknown_signer")
        session_id = raw_record.get("session_id", "unknown_session")
        video_id = raw_record.get("video_id", sample_id)

        duration = float(raw_record.get("duration", raw_record.get("duration_sec", 0.0)))
        frame_count = int(raw_record.get("frame_count", raw_record.get("frames", 0)))
        fps = float(raw_record.get("fps", 30.0))
        resolution = raw_record.get("resolution", "1920x1080")

        glosses = raw_record.get("glosses") or raw_record.get("ordered_glosses") or []
        english_text = raw_record.get("english_text") or raw_record.get("translation") or ""

        provenance_id = f"ingest_{dataset_id}_{sample_id}"
        reasons = []

        # Validate video integrity
        if frame_count <= 0 and duration <= 0:
            reasons.append("MISSING_VIDEO_FRAMES")

        # Validate glosses
        if not glosses or len(glosses) == 0:
            reasons.append("EMPTY_GLOSS_SEQUENCE")
        else:
            malformed = [g for g in glosses if not validate_gloss_token(str(g))]
            if malformed:
                reasons.append(f"MALFORMED_GLOSS_TOKENS: {malformed[:3]}")

        # Validate CTC feasibility
        if len(glosses) > 0 and frame_count > 0 and frame_count < len(glosses):
            reasons.append("CTC_INFEASIBLE_FRAMES_LESS_THAN_GLOSSES")

        # Determine classification
        if not reasons:
            status = IngestionStatus.VALID.value
            failure_reason = ""
            quality_status = "VERIFIED"
        elif "EMPTY_GLOSS_SEQUENCE" in reasons or "MISSING_VIDEO_FRAMES" in reasons:
            status = IngestionStatus.INVALID.value
            failure_reason = "; ".join(reasons)
            quality_status = "WEAK"
        else:
            status = IngestionStatus.REQUIRES_REVIEW.value
            failure_reason = "; ".join(reasons)
            quality_status = "PARTIAL"

        # Derived artifact saving
        derived_record = {
            "dataset_id": dataset_id,
            "sample_id": sample_id,
            "signer_id": signer_id,
            "session_id": session_id,
            "video_id": video_id,
            "duration": duration,
            "frame_count": frame_count,
            "fps": fps,
            "resolution": resolution,
            "glosses": glosses,
            "english_text": english_text,
            "status": status,
            "failure_reason": failure_reason,
            "parent_provenance_id": parent_provenance_id,
        }
        derived_path = self.interim_dir / f"{dataset_id}_{sample_id}.json"
        derived_path.write_text(json.dumps(derived_record, indent=2), encoding="utf-8")
        derived_sha = compute_sha256(derived_path)

        leakage_status = "SIGNER_INDEPENDENT_SPLIT_POSSIBLE" if signer_id != "unknown_signer" else "SIGNER_INDEPENDENT_SPLIT_NOT_POSSIBLE"

        return IngestedSample(
            dataset_id=dataset_id,
            sample_id=sample_id,
            signer_id=signer_id,
            session_id=session_id,
            video_id=video_id,
            duration_sec=duration,
            frame_count=frame_count,
            fps=fps,
            resolution=resolution,
            gloss_count=len(glosses),
            token_count=len(glosses),
            glosses=glosses,
            english_text=english_text,
            status=status,
            failure_reason=failure_reason,
            quality_status=quality_status,
            leakage_status=leakage_status,
            provenance_id=provenance_id,
            raw_sha256=raw_sha256,
            derived_sha256=derived_sha,
            metadata=derived_record,
        )
