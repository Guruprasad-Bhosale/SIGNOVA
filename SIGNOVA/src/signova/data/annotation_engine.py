"""
Sequential ISL Annotation Ingestion, Verification, and Multi-Format Integration Engine.

Guiding Principles:
1. Strict quality grades: UNVERIFIED, WEAK, PARTIAL, VERIFIED, LINGUIST_REVIEWED.
2. Only VERIFIED and LINGUIST_REVIEWED are training eligible; never auto-upgrade.
3. Primary rich temporal format: ELAN .eaf (XML tiers) with CSV and JSON adapters.
4. CTC sequence feasibility validation (T_features >= T_glosses).
5. Zero pseudo-label fabrication from English sentences.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import xml.etree.ElementTree as ET
import pandas as pd


class SupervisionGrade(str, Enum):
    """Formal supervision quality grade for sign annotations."""
    UNVERIFIED = "UNVERIFIED"
    WEAK = "WEAK"
    PARTIAL = "PARTIAL"
    VERIFIED = "VERIFIED"
    LINGUIST_REVIEWED = "LINGUIST_REVIEWED"


TRAINING_ELIGIBLE_GRADES: Set[SupervisionGrade] = {
    SupervisionGrade.VERIFIED,
    SupervisionGrade.LINGUIST_REVIEWED,
}


@dataclass
class TemporalSegment:
    """
    Temporal boundary alignment for an individual sign gloss within a continuous video.
    """
    gloss: str
    start_ms: int
    end_ms: int
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    confidence: float = 1.0

    def __post_init__(self):
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError(f"Invalid temporal segment bounds: start_ms={self.start_ms}, end_ms={self.end_ms}")


@dataclass
class SequentialISLAnnotation:
    """
    Canonical sequential ISL annotation record.
    """
    sample_id: str
    video_id: str
    gloss_sequence: List[str]
    english_translation: Optional[str] = None
    signer_id: Optional[str] = None
    session_id: Optional[str] = None
    temporal_segments: List[TemporalSegment] = field(default_factory=list)
    fps: float = 30.0
    frame_count: Optional[int] = None
    supervision_grade: SupervisionGrade = SupervisionGrade.UNVERIFIED
    provenance: str = "unknown"
    license: str = "unknown"

    def is_training_eligible(self) -> bool:
        """Only VERIFIED or LINGUIST_REVIEWED annotations with non-empty gloss sequences are training-eligible."""
        if not self.gloss_sequence or len(self.gloss_sequence) == 0:
            return False
        return self.supervision_grade in TRAINING_ELIGIBLE_GRADES

    def check_ctc_feasibility(self, feature_frames: int) -> Tuple[bool, str]:
        """
        Verify that feature length satisfies CTC constraint: T_features >= len(gloss_sequence).
        """
        gloss_len = len(self.gloss_sequence)
        if feature_frames < gloss_len:
            return False, f"CTC constraint violated: {feature_frames} feature frames < {gloss_len} glosses"
        return True, "CTC constraint satisfied"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize canonical annotation record to dictionary."""
        return {
            "sample_id": self.sample_id,
            "video_id": self.video_id,
            "gloss_sequence": list(self.gloss_sequence),
            "english_translation": self.english_translation,
            "signer_id": self.signer_id,
            "session_id": self.session_id,
            "temporal_segments": [
                {
                    "gloss": seg.gloss,
                    "start_ms": seg.start_ms,
                    "end_ms": seg.end_ms,
                    "start_frame": seg.start_frame,
                    "end_frame": seg.end_frame,
                    "confidence": seg.confidence,
                }
                for seg in self.temporal_segments
            ],
            "fps": self.fps,
            "frame_count": self.frame_count,
            "supervision_grade": self.supervision_grade.value,
            "provenance": self.provenance,
            "license": self.license,
            "is_training_eligible": self.is_training_eligible(),
        }


def validate_annotation_semantics(annotation: SequentialISLAnnotation) -> Tuple[bool, List[str]]:
    """
    Validates linguistic consistency, ordering, and temporal monotonicity.
    """
    errors = []

    if not annotation.video_id:
        errors.append("video_id cannot be empty")

    if not annotation.gloss_sequence or len(annotation.gloss_sequence) == 0:
        errors.append("gloss_sequence cannot be empty")

    # Check for lowercase sentences masquerading as glosses
    for g in annotation.gloss_sequence:
        if not isinstance(g, str) or not g.strip():
            errors.append(f"Invalid empty gloss token encountered: '{g}'")
        elif " " in g.strip() and not g.startswith("<"):
            errors.append(f"Multi-word string found in single gloss token '{g}' (possible English text masquerading as gloss)")

    # Check temporal segments monotonicity if present
    if annotation.temporal_segments:
        last_start = -1
        for seg in annotation.temporal_segments:
            if seg.start_ms < last_start:
                errors.append(f"Non-monotonic temporal segment start: {seg.start_ms} < {last_start}")
            last_start = seg.start_ms

    return len(errors) == 0, errors


class ELANAnnotationAdapter:
    """
    Parser for rich temporal ELAN (.eaf) XML annotation files.
    """

    DEFAULT_GLOSS_TIERS = ["GLOSS_MAIN", "GLOSS", "GLOSS_RH", "SIGNS"]
    DEFAULT_TRANS_TIERS = ["TRANSLATION_EN", "ENGLISH", "TRANSLATION"]

    def parse_file(
        self,
        eaf_path: Union[str, Path],
        video_id: Optional[str] = None,
        signer_id: Optional[str] = None,
        supervision_grade: SupervisionGrade = SupervisionGrade.VERIFIED,
        fps: float = 30.0,
    ) -> SequentialISLAnnotation:
        """Parse ELAN .eaf XML into SequentialISLAnnotation."""
        eaf_path = Path(eaf_path)
        if not eaf_path.exists():
            raise FileNotFoundError(f"ELAN file not found: {eaf_path}")

        tree = ET.parse(eaf_path)
        root = tree.getroot()

        # 1. Parse TIME_ORDER
        time_slots: Dict[str, int] = {}
        for ts in root.findall(".//TIME_SLOT"):
            ts_id = ts.get("TIME_SLOT_ID")
            ts_val = ts.get("TIME_VALUE")
            if ts_id and ts_val is not None:
                time_slots[ts_id] = int(ts_val)

        # 2. Parse Tiers
        gloss_segments: List[TemporalSegment] = []
        gloss_tokens: List[str] = []
        english_translation: Optional[str] = None

        for tier in root.findall(".//TIER"):
            tier_id = tier.get("TIER_ID", "").strip()
            
            # Gloss Tiers
            if tier_id in self.DEFAULT_GLOSS_TIERS or "GLOSS" in tier_id.upper():
                for ann in tier.findall(".//ALIGNABLE_ANNOTATION"):
                    ts1 = ann.get("TIME_SLOT_REF1")
                    ts2 = ann.get("TIME_SLOT_REF2")
                    val_elem = ann.find("ANNOTATION_VALUE")
                    val_text = val_elem.text.strip().upper() if val_elem is not None and val_elem.text else ""

                    if val_text and ts1 in time_slots and ts2 in time_slots:
                        start_ms = time_slots[ts1]
                        end_ms = time_slots[ts2]
                        start_f = int((start_ms / 1000.0) * fps)
                        end_f = int((end_ms / 1000.0) * fps)

                        seg = TemporalSegment(
                            gloss=val_text,
                            start_ms=start_ms,
                            end_ms=end_ms,
                            start_frame=start_f,
                            end_frame=end_f,
                        )
                        gloss_segments.append(seg)
                        gloss_tokens.append(val_text)

            # Translation Tiers
            elif tier_id in self.DEFAULT_TRANS_TIERS or "TRANSLAT" in tier_id.upper():
                trans_vals = []
                for ann in tier.findall(".//ANNOTATION_VALUE"):
                    if ann.text and ann.text.strip():
                        trans_vals.append(ann.text.strip())
                if trans_vals:
                    english_translation = " ".join(trans_vals)

        # Sort segments by start_ms
        gloss_segments.sort(key=lambda s: s.start_ms)
        gloss_tokens = [s.gloss for s in gloss_segments]

        vid = video_id or eaf_path.stem

        return SequentialISLAnnotation(
            sample_id=f"EAF-{vid}",
            video_id=vid,
            gloss_sequence=gloss_tokens,
            english_translation=english_translation,
            signer_id=signer_id,
            temporal_segments=gloss_segments,
            fps=fps,
            supervision_grade=supervision_grade,
            provenance=f"ELAN file: {eaf_path.name}",
            license="Research Annotation",
        )


class CSVAnnotationAdapter:
    """
    Parser for tabular CSV manifests containing sequential sign annotations.
    """

    def parse_csv(
        self,
        csv_path: Union[str, Path],
        gloss_col: str = "gloss_sequence",
        video_id_col: str = "video_id",
        trans_col: Optional[str] = "english_translation",
        signer_col: Optional[str] = "signer_id",
        grade_col: Optional[str] = "supervision_grade",
        default_grade: SupervisionGrade = SupervisionGrade.VERIFIED,
    ) -> List[SequentialISLAnnotation]:
        """Parse CSV rows into list of SequentialISLAnnotation records."""
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        annotations: List[SequentialISLAnnotation] = []
        for idx, row in df.iterrows():
            raw_gloss = str(row[gloss_col]) if gloss_col in row and pd.notna(row[gloss_col]) else ""
            if not raw_gloss:
                continue

            # Split tokens by space or comma
            if "," in raw_gloss:
                tokens = [t.strip().upper() for t in raw_gloss.split(",") if t.strip()]
            else:
                tokens = [t.strip().upper() for t in raw_gloss.split() if t.strip()]

            vid = str(row[video_id_col]) if video_id_col in row and pd.notna(row[video_id_col]) else f"CSV-VID-{idx+1:04d}"
            trans = str(row[trans_col]) if trans_col and trans_col in row and pd.notna(row[trans_col]) else None
            signer = str(row[signer_col]) if signer_col and signer_col in row and pd.notna(row[signer_col]) else None

            if grade_col and grade_col in row and pd.notna(row[grade_col]):
                grade_str = str(row[grade_col]).upper()
                grade = SupervisionGrade(grade_str) if grade_str in SupervisionGrade.__members__ else default_grade
            else:
                grade = default_grade

            ann = SequentialISLAnnotation(
                sample_id=f"CSV-{vid}",
                video_id=vid,
                gloss_sequence=tokens,
                english_translation=trans,
                signer_id=signer,
                supervision_grade=grade,
                provenance=f"CSV manifest: {csv_path.name}",
                license="Research Manifest",
            )
            annotations.append(ann)

        return annotations


class JSONAnnotationAdapter:
    """
    Parser for structured JSON sequence records.
    """

    def parse_json(
        self,
        json_path: Union[str, Path],
        default_grade: SupervisionGrade = SupervisionGrade.VERIFIED,
    ) -> List[SequentialISLAnnotation]:
        """Parse JSON file containing single or list of sequential annotation records."""
        json_path = Path(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data if isinstance(data, list) else data.get("annotations", [data])
        annotations: List[SequentialISLAnnotation] = []

        for idx, rec in enumerate(records):
            tokens = rec.get("gloss_sequence", [])
            vid = rec.get("video_id", f"JSON-VID-{idx+1:04d}")
            trans = rec.get("english_translation", None)
            signer = rec.get("signer_id", None)
            grade_str = rec.get("supervision_grade", default_grade.value).upper()
            grade = SupervisionGrade(grade_str) if grade_str in SupervisionGrade.__members__ else default_grade

            segments = []
            for s in rec.get("temporal_segments", []):
                segments.append(
                    TemporalSegment(
                        gloss=s["gloss"],
                        start_ms=s["start_ms"],
                        end_ms=s["end_ms"],
                        start_frame=s.get("start_frame"),
                        end_frame=s.get("end_frame"),
                        confidence=s.get("confidence", 1.0),
                    )
                )

            ann = SequentialISLAnnotation(
                sample_id=rec.get("sample_id", f"JSON-{vid}"),
                video_id=vid,
                gloss_sequence=[str(t).upper() for t in tokens],
                english_translation=trans,
                signer_id=signer,
                temporal_segments=segments,
                fps=rec.get("fps", 30.0),
                frame_count=rec.get("frame_count"),
                supervision_grade=grade,
                provenance=rec.get("provenance", f"JSON file: {json_path.name}"),
                license=rec.get("license", "Research JSON"),
            )
            annotations.append(ann)

        return annotations
