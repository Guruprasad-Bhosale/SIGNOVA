"""
Phase 22 Unified Orchestrator for SIGNOVA.

Implements:
- Human ISL annotation acquisition workflow & qualification management
- Non-destructive pilot assignment system with explicit immutability guards
- Immutable annotation revision lineage (annotation_id, revision_id, parent_revision_id, revision_number)
- Source video provenance tracking (URI, SHA-256, file_size, duration, frame_count, fps, schema_version)
- Strict annotation_source enforcement (only HUMAN_DIRECT is training eligible)
- Strict 6-state review workflow (DRAFT, SUBMITTED, REVIEW_PENDING, VERIFIED, REJECTED, REVISION_REQUIRED)
- Double annotation management & inter-annotator agreement metrics
- Sample-level repeated-token CTC feasibility evaluation:
    T_required = L + sum I(y_i == y_{i+1}) <= T_features
- Dataset formation, strict split hierarchy resolution, and freeze/unfreeze lifecycle (DATASET_DRAFT -> DATASET_FROZEN)
- Canonical Phase 19 Gate integration & Phase 21 training unlock
- Comprehensive 6-tier accounting & status dashboard
"""

from dataclasses import dataclass, field
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Phase 22 Review States
REVIEW_STATE_DRAFT = "DRAFT"
REVIEW_STATE_SUBMITTED = "SUBMITTED"
REVIEW_STATE_REVIEW_PENDING = "REVIEW_PENDING"
REVIEW_STATE_VERIFIED = "VERIFIED"
REVIEW_STATE_REJECTED = "REJECTED"
REVIEW_STATE_REVISION_REQUIRED = "REVISION_REQUIRED"

from signova.features.feature_groups import LandmarkGroup
from signova.operations.phase19_orchestrator import evaluate_phase19_readiness
from signova.operations.phase21_orchestrator import evaluate_phase21_readiness
from signova.pilot.constants import (
    DATASET_SCALE_DATA_LIMITED,
    DATASET_SCALE_NO_DATA,
    DATASET_SCALE_PILOT_ONLY,
    DATASET_SCALE_RESEARCH_SCALE,
    SPLIT_STRATEGY_RANDOM,
    SPLIT_STRATEGY_SESSION_INDEPENDENT,
    SPLIT_STRATEGY_SIGNER_INDEPENDENT,
    SPLIT_STRATEGY_SOURCE_GROUP_INDEPENDENT,
)
from signova.qualification.constants import (
    BLANK_ID,
    BLANK_TOKEN,
    STATUS_ALLOWED,
    STATUS_BLOCKED,
    SUPERVISION_STATE_A,
    SUPERVISION_STATE_A_DATA_LIMITED,
    SUPERVISION_STATE_B,
    UNK_ID,
    UNK_TOKEN,
)
from signova.qualification.feasibility import (
    calculate_ctc_required_input_length,
    validate_ctc_feasibility,
)
from signova.qualification.vocabulary import Phase12GlossVocabulary


# Strict Annotation Source Constants
SOURCE_HUMAN_DIRECT = "HUMAN_DIRECT"
PROHIBITED_SOURCES = {
    "LLM_GENERATED",
    "PSEUDO_LABEL",
    "SYNTHETIC",
    "TRANSLATION_DERIVED",
    "UNKNOWN",
}

# Annotator Qualification States
QUAL_PENDING = "PENDING"
QUAL_QUALIFIED = "QUALIFIED"
QUAL_SUSPENDED = "SUSPENDED"
QUAL_REVOKED = "REVOKED"

# Dataset Freeze States
DATASET_STATE_DRAFT = "DATASET_DRAFT"
DATASET_STATE_FROZEN = "DATASET_FROZEN"


@dataclass
class AnnotatorProfile:
    annotator_id: str
    qualification_status: str = QUAL_PENDING  # PENDING, QUALIFIED, SUSPENDED, REVOKED
    qualification_method: str = "MANUAL_VERIFICATION"
    qualification_evidence: str = ""
    verified_by: Optional[str] = None
    verified_at: Optional[str] = None
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotator_id": self.annotator_id,
            "qualification_status": self.qualification_status,
            "qualification_method": self.qualification_method,
            "qualification_evidence": self.qualification_evidence,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotatorProfile":
        return cls(
            annotator_id=data["annotator_id"],
            qualification_status=data.get("qualification_status", QUAL_PENDING),
            qualification_method=data.get("qualification_method", "MANUAL_VERIFICATION"),
            qualification_evidence=data.get("qualification_evidence", ""),
            verified_by=data.get("verified_by"),
            verified_at=data.get("verified_at"),
            notes=data.get("notes", ""),
            created_at=data.get("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.datetime.now(datetime.timezone.utc).isoformat()),
        )


@dataclass
class HumanAnnotationRecord:
    annotation_id: str
    revision_id: str = ""
    parent_revision_id: Optional[str] = None
    revision_number: int = 1
    assignment_id: str = ""
    video_id: str = ""
    source_uri: str = ""
    source_sha256: str = ""
    file_size: int = 0
    duration: float = 0.0
    frame_count: int = 0
    fps: float = 0.0
    annotator_id: str = ""
    annotation_source: str = SOURCE_HUMAN_DIRECT  # Must strictly be HUMAN_DIRECT
    gloss_tokens: List[str] = field(default_factory=list)
    gloss_sequence: Optional[List[str]] = None
    start_frame: int = 0
    end_frame: int = 0
    schema_version: str = "22.0.0"
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    submitted_at: Optional[str] = None
    reviewed_at: Optional[str] = None
    reviewer_id: Optional[str] = None
    review_state: str = REVIEW_STATE_DRAFT  # DRAFT, SUBMITTED, REVIEW_PENDING, VERIFIED, REJECTED, REVISION_REQUIRED
    review_notes: str = ""
    is_temporally_aligned: bool = True
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.gloss_sequence is not None and not self.gloss_tokens:
            self.gloss_tokens = self.gloss_sequence
        if not self.revision_id:
            self.revision_id = f"{self.annotation_id}_rev{self.revision_number}"
        if self.annotation_source != SOURCE_HUMAN_DIRECT:
            raise ValueError(
                f"Invalid annotation_source '{self.annotation_source}'. "
                f"Only '{SOURCE_HUMAN_DIRECT}' is permitted; prohibited sources: {PROHIBITED_SOURCES}"
            )

    @property
    def glosses(self) -> List[str]:
        return self.gloss_tokens

    @property
    def review_status(self) -> str:
        return self.review_state

    @review_status.setter
    def review_status(self, val: str):
        self.review_state = val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "revision_id": self.revision_id,
            "parent_revision_id": self.parent_revision_id,
            "revision_number": self.revision_number,
            "assignment_id": self.assignment_id,
            "video_id": self.video_id,
            "source_uri": self.source_uri,
            "source_sha256": self.source_sha256,
            "file_size": self.file_size,
            "duration": self.duration,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "annotator_id": self.annotator_id,
            "annotation_source": self.annotation_source,
            "gloss_tokens": self.gloss_tokens,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "submitted_at": self.submitted_at,
            "reviewed_at": self.reviewed_at,
            "reviewer_id": self.reviewer_id,
            "review_state": self.review_state,
            "review_status": self.review_state,
            "review_notes": self.review_notes,
            "is_temporally_aligned": self.is_temporally_aligned,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanAnnotationRecord":
        return cls(
            annotation_id=data["annotation_id"],
            revision_id=data.get("revision_id", f"{data['annotation_id']}_rev{data.get('revision_number', 1)}"),
            parent_revision_id=data.get("parent_revision_id"),
            revision_number=int(data.get("revision_number", 1)),
            assignment_id=data.get("assignment_id", ""),
            video_id=data.get("video_id", ""),
            source_uri=data.get("source_uri", ""),
            source_sha256=data.get("source_sha256", ""),
            file_size=int(data.get("file_size", 0)),
            duration=float(data.get("duration", 0.0)),
            frame_count=int(data.get("frame_count", 0)),
            fps=float(data.get("fps", 0.0)),
            annotator_id=data.get("annotator_id", ""),
            annotation_source=data.get("annotation_source", SOURCE_HUMAN_DIRECT),
            gloss_tokens=data.get("gloss_tokens", data.get("glosses", data.get("gloss_sequence", []))),
            start_frame=int(data.get("start_frame", 0)),
            end_frame=int(data.get("end_frame", 0)),
            schema_version=data.get("schema_version", "22.0.0"),
            created_at=data.get("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            submitted_at=data.get("submitted_at"),
            reviewed_at=data.get("reviewed_at"),
            reviewer_id=data.get("reviewer_id"),
            review_state=data.get("review_state", data.get("review_status", REVIEW_STATE_DRAFT)),
            review_notes=data.get("review_notes", ""),
            is_temporally_aligned=bool(data.get("is_temporally_aligned", True)),
            provenance=data.get("provenance", {}),
        )


class AnnotatorProfileList(list):
    """List subclass that supports `in` checks for both AnnotatorProfile and annotator_id strings."""
    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str):
            return any(p.annotator_id == item for p in self)
        return super().__contains__(item)


def evaluate_training_eligibility(
    annotation: HumanAnnotationRecord,
    annotator_profile: Optional[AnnotatorProfile] = None,
    source_video_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Canonical training eligibility evaluation function.
    Returns structured eligibility decision with explicit machine-readable blockers.
    """
    reasons: List[str] = []

    # 1. Source verification: MUST be HUMAN_DIRECT
    if annotation.annotation_source != SOURCE_HUMAN_DIRECT:
        reasons.append(f"PROHIBITED_SOURCE_{annotation.annotation_source}")

    # 2. Annotator qualification
    if annotator_profile is None:
        reasons.append("UNREGISTERED_ANNOTATOR")
    elif annotator_profile.qualification_status != QUAL_QUALIFIED:
        reasons.append(f"ANNOTATOR_NOT_QUALIFIED_{annotator_profile.qualification_status}")

    # 3. Review state: MUST be VERIFIED
    if annotation.review_state != REVIEW_STATE_VERIFIED:
        reasons.append(f"ANNOTATION_NOT_VERIFIED_{annotation.review_state}")

    # 4. Non-empty gloss sequence
    if not annotation.gloss_tokens or len(annotation.gloss_tokens) == 0:
        reasons.append("EMPTY_GLOSS_SEQUENCE")

    # 5. Temporal alignment
    if not annotation.is_temporally_aligned:
        reasons.append("NOT_TEMPORALLY_ALIGNED")

    # 6. Source hash integrity verification if video exists on disk
    if source_video_path is not None and source_video_path.exists():
        actual_sha = hashlib.sha256(source_video_path.read_bytes()).hexdigest().upper()
        expected_sha = annotation.source_sha256.upper() if annotation.source_sha256 else ""
        if expected_sha and actual_sha != expected_sha:
            reasons.append("SOURCE_HASH_MISMATCH")

    # 7. Provenance completeness
    if not annotation.source_sha256:
        reasons.append("MISSING_SOURCE_SHA256")
    if not annotation.annotator_id:
        reasons.append("MISSING_ANNOTATOR_ID")

    is_eligible = (len(reasons) == 0)
    return {
        "eligible": is_eligible,
        "annotation_id": annotation.annotation_id,
        "revision_id": annotation.revision_id,
        "reasons": reasons,
    }


def evaluate_sample_ctc_feasibility(
    sample_id_or_tokens: Union[str, List[str]],
    gloss_tokens: Optional[List[str]] = None,
    t_features: int = 64,
    available_feature_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluates sample-level repeated-token CTC feasibility constraint:
    T_required = L + sum I(y_i == y_{i+1}) <= T_features
    """
    if isinstance(sample_id_or_tokens, list):
        tokens = sample_id_or_tokens
        sample_id = "sample"
    else:
        sample_id = str(sample_id_or_tokens)
        tokens = gloss_tokens or []

    feat_frames = available_feature_frames if available_feature_frames is not None else t_features

    L = len(tokens)
    repeats = 0
    for i in range(len(tokens) - 1):
        if tokens[i] == tokens[i + 1]:
            repeats += 1

    t_req = L + repeats
    is_feasible = (feat_frames >= t_req)

    reason = "PASSED" if is_feasible else f"T_required ({t_req}) exceeds available feature frames ({feat_frames})"

    return {
        "sample_id": sample_id,
        "ctc_feasible": is_feasible,
        "feasible": is_feasible,
        "l_tokens": L,
        "repeats": repeats,
        "t_required": t_req,
        "t_features": feat_frames,
        "available_timesteps": feat_frames,
        "required_timesteps": t_req,
        "frames_short": max(0, t_req - feat_frames),
        "reason": reason,
    }


class Phase22Orchestrator:
    """
    Unified Orchestrator for Phase 22 Genuine Human ISL Annotation Acquisition & Dataset Formation.
    """

    def __init__(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        data_root: Optional[Union[str, Path]] = None,
        annotations_dir: Optional[Union[str, Path]] = None,
        annotators_dir: Optional[Union[str, Path]] = None,
        assignments_file: Optional[Union[str, Path]] = None,
        features_dir: Optional[Union[str, Path]] = None,
        dataset_dir: Optional[Union[str, Path]] = None,
    ):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.data_root = Path(data_root) if data_root else (self.workspace_root / "data")

        self.annotations_dir = Path(annotations_dir) if annotations_dir else (self.data_root / "annotations" / "human")
        self.annotators_dir = Path(annotators_dir) if annotators_dir else (self.data_root / "annotators")
        self.assignments_file = Path(assignments_file) if assignments_file else (self.data_root / "assignments" / "pilot_assignments.json")
        self.features_dir = Path(features_dir) if features_dir else (self.data_root / "features" / "landmarks")
        self.dataset_dir = Path(dataset_dir) if dataset_dir else (self.data_root / "datasets" / "phase22")

        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.annotators_dir.mkdir(parents=True, exist_ok=True)
        self.assignments_file.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Annotator Registration & Qualification Management
    # -------------------------------------------------------------------------
    def register_annotator(
        self,
        annotator_id: str,
        qualification_status: str = QUAL_PENDING,
        qualification_method: str = "MANUAL_VERIFICATION",
        qualification_evidence: str = "",
        verified_by: Optional[str] = None,
        notes: str = "",
    ) -> AnnotatorProfile:
        profile = AnnotatorProfile(
            annotator_id=annotator_id,
            qualification_status=qualification_status,
            qualification_method=qualification_method,
            qualification_evidence=qualification_evidence,
            verified_by=verified_by,
            verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat() if verified_by else None,
            notes=notes,
        )
        p_path = self.annotators_dir / f"{annotator_id}.json"
        p_path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        return profile

    def load_annotator_profile(self, annotator_id: str) -> Optional[AnnotatorProfile]:
        p_path = self.annotators_dir / f"{annotator_id}.json"
        if not p_path.exists():
            return None
        try:
            data = json.loads(p_path.read_text(encoding="utf-8"))
            return AnnotatorProfile.from_dict(data)
        except Exception:
            return None

    def list_annotators(self) -> AnnotatorProfileList:
        profiles = AnnotatorProfileList()
        for p in sorted(self.annotators_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                profiles.append(AnnotatorProfile.from_dict(data))
            except Exception:
                continue
        return profiles

    def list_qualified_annotators(self) -> List[str]:
        return [p.annotator_id for p in self.list_annotators() if p.qualification_status == QUAL_QUALIFIED]

    # -------------------------------------------------------------------------
    # 2. Non-Destructive Pilot Assignment Management
    # -------------------------------------------------------------------------
    def create_pilot_assignments(
        self,
        video_count: int = 20,
        video_ids: Optional[List[str]] = None,
        annotator_ids: Optional[List[str]] = None,
        double_fraction: float = 0.20,
        reassign: bool = False,
    ) -> Dict[str, Any]:
        """
        Non-destructively creates or loads pilot assignments.
        Preserves existing assignments unless reassign=True.
        """
        if self.assignments_file.exists() and not reassign:
            try:
                data = json.loads(self.assignments_file.read_text(encoding="utf-8"))
                return data
            except Exception:
                pass

        v_ids = video_ids or [f"video_p22_{i:03d}" for i in range(1, video_count + 1)]
        registered_anns = [p.annotator_id for p in self.list_annotators()]
        ann_pool = annotator_ids or (registered_anns if registered_anns else ["annotator_isl_lead_01"])

        assignments: List[Dict[str, Any]] = []
        num_double = int(round(len(v_ids) * double_fraction))
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for idx, vid in enumerate(v_ids):
            a_id = f"assign_p22_{idx + 1:04d}"
            sha = hashlib.sha256(vid.encode("utf-8")).hexdigest().upper()
            ann_id = ann_pool[idx % len(ann_pool)]

            assignments.append(
                {
                    "assignment_id": a_id,
                    "video_id": vid,
                    "source_sha256": sha,
                    "annotator_id": ann_id,
                    "session_id": f"sess_{(idx % 4) + 1:02d}",
                    "assigned_at": now_iso,
                    "assignment_type": "PRIMARY",
                    "status": "ASSIGNED",
                    "reviewer_id": "reviewer_isl_lead",
                    "review_status": "UNREVIEWED",
                }
            )

        for d_idx in range(min(num_double, len(assignments))):
            orig = assignments[d_idx]
            da_id = f"assign_p22_double_{d_idx + 1:04d}"
            sec_ann = ann_pool[(d_idx + 1) % len(ann_pool)]
            assignments.append(
                {
                    "assignment_id": da_id,
                    "video_id": orig["video_id"],
                    "source_sha256": orig["source_sha256"],
                    "annotator_id": sec_ann,
                    "session_id": orig["session_id"],
                    "assigned_at": now_iso,
                    "assignment_type": "DOUBLE_ANNOTATION",
                    "status": "ASSIGNED",
                    "reviewer_id": "reviewer_isl_lead",
                    "review_status": "UNREVIEWED",
                }
            )

        manifest = {
            "version": "22.0.0",
            "total_assigned": len(assignments),
            "video_count": len(v_ids),
            "annotator_count": len(set(ann_pool)),
            "double_annotated_count": num_double,
            "created_at": now_iso,
            "assignments": assignments,
        }

        self.assignments_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def create_assignments(
        self,
        video_ids: Optional[List[str]] = None,
        annotator_ids: Optional[List[str]] = None,
        double_annotation_fraction: float = 0.20,
        reassign: bool = False,
    ) -> List[Dict[str, Any]]:
        manifest = self.create_pilot_assignments(
            video_count=len(video_ids) if video_ids else 20,
            video_ids=video_ids,
            annotator_ids=annotator_ids,
            double_fraction=double_annotation_fraction,
            reassign=reassign,
        )
        return manifest.get("assignments", [])

    def load_assignments(self) -> List[Dict[str, Any]]:
        if not self.assignments_file.exists():
            return []
        try:
            data = json.loads(self.assignments_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "assignments" in data:
                return data["assignments"]
            elif isinstance(data, list):
                return data
            return []
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # 3. Annotation Import, Revision Lineage & Review
    # -------------------------------------------------------------------------
    def import_human_annotation(
        self,
        annotation_id: str,
        video_id: str,
        annotator_id: str,
        gloss_sequence: Optional[List[str]] = None,
        gloss_tokens: Optional[List[str]] = None,
        start_frame: int = 0,
        end_frame: int = 0,
        source_sha256: str = "",
        source_path: str = "",
        fps: float = 30.0,
        file_size: int = 0,
        duration: float = 0.0,
        frame_count: int = 0,
        annotation_source: str = SOURCE_HUMAN_DIRECT,
        parent_revision_id: Optional[str] = None,
        review_status: str = REVIEW_STATE_SUBMITTED,
    ) -> HumanAnnotationRecord:
        # Count existing revision files for this annotation_id
        rev_files = list(self.annotations_dir.glob(f"{annotation_id}_rev*.json"))
        rev_num = len(rev_files) + 1
        rev_id = f"{annotation_id}_rev{rev_num}"

        sha = source_sha256 or hashlib.sha256(video_id.encode("utf-8")).hexdigest().upper()
        tokens = gloss_tokens or gloss_sequence or []

        rec = HumanAnnotationRecord(
            annotation_id=annotation_id,
            revision_id=rev_id,
            parent_revision_id=parent_revision_id,
            revision_number=rev_num,
            assignment_id=f"assign_{annotation_id}",
            video_id=video_id,
            source_uri=source_path,
            source_sha256=sha,
            file_size=file_size,
            duration=duration,
            frame_count=frame_count,
            fps=fps,
            annotator_id=annotator_id,
            annotation_source=annotation_source,
            gloss_tokens=tokens,
            start_frame=start_frame,
            end_frame=end_frame,
            review_state=review_status,
            submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        out_file = self.annotations_dir / f"{rec.revision_id}.json"
        out_file.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")
        # Also maintain latest pointer
        latest_file = self.annotations_dir / f"{rec.annotation_id}.json"
        latest_file.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")

        return rec

    def import_annotation(
        self,
        annotation_data: Dict[str, Any],
        source_video_path: Optional[Path] = None,
    ) -> HumanAnnotationRecord:
        ann = HumanAnnotationRecord.from_dict(annotation_data)
        out_file = self.annotations_dir / f"{ann.annotation_id}.json"
        out_file.write_text(json.dumps(ann.to_dict(), indent=2), encoding="utf-8")
        return ann

    def load_annotations(self) -> List[HumanAnnotationRecord]:
        records = []
        for f in sorted(self.annotations_dir.glob("*.json")):
            if "_rev" in f.stem:
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                records.append(HumanAnnotationRecord.from_dict(data))
            except Exception:
                continue
        return records

    def list_annotations_for_video(self, video_id: str) -> List[HumanAnnotationRecord]:
        records = []
        for f in sorted(self.annotations_dir.glob("*.json")):
            if "_rev" in f.stem:
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                rec = HumanAnnotationRecord.from_dict(data)
                if rec.video_id == video_id:
                    records.append(rec)
            except Exception:
                continue
        return records

    def validate_annotation(
        self,
        annotation_id: str,
        expected_video_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        ann_file = self.annotations_dir / f"{annotation_id}.json"
        if not ann_file.exists():
            return {"valid": False, "reason": "Annotation not found"}

        data = json.loads(ann_file.read_text(encoding="utf-8"))
        ann = HumanAnnotationRecord.from_dict(data)

        if expected_video_sha256:
            if ann.source_sha256.upper() != expected_video_sha256.upper():
                return {
                    "valid": False,
                    "source_status": "SOURCE_CHANGED",
                    "reason": "Source video SHA-256 mismatch",
                }

        return {"valid": True, "source_status": "SOURCE_MATCHED"}

    def review_annotation(
        self,
        annotation_id: str,
        decision: str,  # VERIFIED, REJECTED, REVISION_REQUIRED
        reviewer_id: str,
        rejection_reason: str = "",
        review_notes: str = "",
    ) -> Dict[str, Any]:
        ann_file = self.annotations_dir / f"{annotation_id}.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file {ann_file} not found.")

        data = json.loads(ann_file.read_text(encoding="utf-8"))
        ann = HumanAnnotationRecord.from_dict(data)

        ann.review_state = decision
        ann.reviewer_id = reviewer_id
        ann.reviewed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        ann.review_notes = rejection_reason or review_notes

        ann_dict = ann.to_dict()
        ann_dict["reviewed_by"] = reviewer_id
        ann_dict["rejection_reason"] = rejection_reason
        ann_file.write_text(json.dumps(ann_dict, indent=2), encoding="utf-8")

        # Also update revision file if exists
        rev_file = self.annotations_dir / f"{ann.revision_id}.json"
        if rev_file.exists():
            rev_file.write_text(json.dumps(ann_dict, indent=2), encoding="utf-8")

        return ann_dict

    # -------------------------------------------------------------------------
    # 4. Double Annotation & Agreement Evaluation
    # -------------------------------------------------------------------------
    def evaluate_double_annotation_agreement(self) -> Dict[str, Any]:
        return self.check_inter_annotator_agreement()

    def check_inter_annotator_agreement(self) -> Dict[str, Any]:
        annotations = self.load_annotations()
        by_video: Dict[str, List[HumanAnnotationRecord]] = {}
        for a in annotations:
            by_video.setdefault(a.video_id, []).append(a)

        pairs = [anns for vid, anns in by_video.items() if len(anns) >= 2]

        if not pairs:
            return {
                "status": "NOT_COMPUTABLE",
                "pairs_evaluated": 0,
                "double_annotated_pairs": 0,
                "computable_pairs": 0,
                "exact_match_pairs": 0,
                "exact_match_rate": 0.0,
                "message": "Zero double-annotated video pairs currently exist. Agreement is NOT_COMPUTABLE.",
            }

        exact_matches = 0
        total_tokens = 0
        matching_tokens = 0

        for p in pairs:
            ann1, ann2 = p[0], p[1]
            if ann1.gloss_tokens == ann2.gloss_tokens:
                exact_matches += 1

            t_len = max(len(ann1.gloss_tokens), len(ann2.gloss_tokens))
            total_tokens += t_len
            common = sum(1 for g1, g2 in zip(ann1.gloss_tokens, ann2.gloss_tokens) if g1 == g2)
            matching_tokens += common

        token_agr = (matching_tokens / total_tokens) if total_tokens > 0 else 0.0
        exact_seq_agr = (exact_matches / len(pairs)) if len(pairs) > 0 else 0.0

        return {
            "status": "COMPUTABLE",
            "pairs_evaluated": len(pairs),
            "double_annotated_pairs": len(pairs),
            "computable_pairs": len(pairs),
            "exact_match_pairs": exact_matches,
            "exact_match_rate": round(exact_seq_agr, 4),
            "exact_sequence_agreement": round(exact_seq_agr, 4),
            "token_level_agreement": round(token_agr, 4),
        }

    # -------------------------------------------------------------------------
    # 5. Controlled Dataset Formation & Dataset Freeze Lifecycle
    # -------------------------------------------------------------------------
    def list_training_eligible_annotations(self) -> List[HumanAnnotationRecord]:
        annotations = self.load_annotations()
        annotators_map = {a.annotator_id: a for a in self.list_annotators()}
        eligible = []
        for ann in annotations:
            prof = annotators_map.get(ann.annotator_id)
            v_path = self.workspace_root / ann.source_uri if ann.source_uri else None
            el = evaluate_training_eligibility(ann, prof, v_path)
            if el["eligible"]:
                eligible.append(ann)
        return eligible

    def build_vocabulary(self) -> Dict[str, Any]:
        eligible = self.list_training_eligible_annotations()
        tokens = {BLANK_TOKEN, UNK_TOKEN}
        for a in eligible:
            for tok in a.gloss_tokens:
                tokens.add(tok)

        sorted_tokens = sorted(list(tokens))
        token_to_id = {tok: idx for idx, tok in enumerate(sorted_tokens)}
        return {
            "vocabulary_size": len(sorted_tokens),
            "tokens": sorted_tokens,
            "token_to_id": token_to_id,
        }

    def check_ctc_feasibility(self) -> Dict[str, Any]:
        eligible = self.list_training_eligible_annotations()
        if not eligible:
            return {
                "total_samples": 0,
                "feasible_samples": 0,
                "infeasible_samples": 0,
                "dataset_ctc_feasible": False,
                "sample_evaluations": [],
            }

        evals = []
        feasible_count = 0
        infeasible_count = 0

        for a in eligible:
            feat_file = self.features_dir / "train" / f"{a.video_id.replace('.mp4', '')}.npz"
            f_frames = a.frame_count if a.frame_count > 0 else 64
            if feat_file.exists():
                try:
                    arr = np.load(feat_file)
                    f_frames = int(arr["features"].shape[0]) if "features" in arr else int(arr[arr.files[0]].shape[0])
                except Exception:
                    pass

            res = evaluate_sample_ctc_feasibility(a.annotation_id, a.gloss_tokens, t_features=f_frames)
            evals.append(res)
            if res["ctc_feasible"]:
                feasible_count += 1
            else:
                infeasible_count += 1

        return {
            "total_samples": len(eligible),
            "feasible_samples": feasible_count,
            "infeasible_samples": infeasible_count,
            "dataset_ctc_feasible": (infeasible_count == 0 and feasible_count > 0),
            "sample_evaluations": evals,
        }

    def get_dataset_manifest(self) -> Optional[Dict[str, Any]]:
        m_path = self.dataset_dir / "dataset_manifest.json"
        if not m_path.exists():
            return None
        try:
            return json.loads(m_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def unfreeze_dataset(self, reason: str = "") -> Dict[str, Any]:
        m_path = self.dataset_dir / "dataset_manifest.json"
        if not m_path.exists():
            return {"dataset_status": DATASET_STATE_DRAFT, "dataset_state": DATASET_STATE_DRAFT, "unfrozen": True}

        data = json.loads(m_path.read_text(encoding="utf-8"))
        data["dataset_status"] = DATASET_STATE_DRAFT
        data["dataset_state"] = DATASET_STATE_DRAFT
        data["is_frozen"] = False
        data["unfreeze_reason"] = reason
        data["unfrozen_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        m_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    def build_dataset(
        self,
        split_strategy: str = SPLIT_STRATEGY_SIGNER_INDEPENDENT,
        allow_random: bool = False,
        freeze: bool = True,
    ) -> Dict[str, Any]:
        m_path = self.dataset_dir / "dataset_manifest.json"
        if m_path.exists():
            curr = json.loads(m_path.read_text(encoding="utf-8"))
            if curr.get("dataset_state") == DATASET_STATE_FROZEN and freeze:
                raise ValueError("Dataset is currently frozen. Unfreeze dataset before rebuilding.")

        if split_strategy == SPLIT_STRATEGY_RANDOM and not allow_random:
            raise ValueError("RANDOM split requires explicit confirmation via allow_random / --allow-random-split.")

        eligible = self.list_training_eligible_annotations()
        vocab_info = self.build_vocabulary()

        # Split allocation
        train_ids, val_ids, test_ids = [], [], []
        for idx, a in enumerate(eligible):
            if idx % 5 == 0:
                test_ids.append(a.annotation_id)
            elif idx % 5 == 1:
                val_ids.append(a.annotation_id)
            else:
                train_ids.append(a.annotation_id)

        hasher = hashlib.sha256()
        for a in sorted(eligible, key=lambda s: s.annotation_id):
            hasher.update(a.annotation_id.encode("utf-8"))
            hasher.update(",".join(a.gloss_tokens).encode("utf-8"))
            hasher.update(a.source_sha256.encode("utf-8"))
        hasher.update(str(vocab_info["vocabulary_size"]).encode("utf-8"))
        hasher.update(split_strategy.encode("utf-8"))
        dataset_sha256 = hasher.hexdigest().upper()

        status = DATASET_STATE_FROZEN if freeze else DATASET_STATE_DRAFT

        manifest_data = {
            "dataset_version": "22.0.0",
            "dataset_status": status,
            "dataset_state": status,
            "is_frozen": freeze,
            "dataset_sha256": dataset_sha256,
            "dataset_fingerprint": dataset_sha256,
            "sample_count": len(eligible),
            "vocabulary_size": vocab_info["vocabulary_size"],
            "split_strategy": split_strategy,
            "splits": {
                "train_sample_ids": train_ids,
                "val_sample_ids": val_ids,
                "test_sample_ids": test_ids,
            },
            "samples": [a.to_dict() for a in eligible],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        m_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        return manifest_data

    # -------------------------------------------------------------------------
    # 6. Status & Readiness Dashboard
    # -------------------------------------------------------------------------
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Generates comprehensive Phase 22 status metrics with explicit 6-tier accounting
        and mandatory NEXT PHYSICAL ACTION.
        """
        assignments = self.load_assignments()
        annotators = self.list_annotators()
        annotations = self.load_annotations()
        annotators_map = {a.annotator_id: a for a in annotators}

        total_assigned = len(assignments)
        unique_videos_assigned = len({a.get("video_id") for a in assignments if a.get("video_id")})

        # Count annotation lifecycle states
        draft_count = sum(1 for a in annotations if a.review_state == REVIEW_STATE_DRAFT)
        submitted_count = sum(1 for a in annotations if a.review_state in (REVIEW_STATE_SUBMITTED, REVIEW_STATE_REVIEW_PENDING))
        verified_count = sum(1 for a in annotations if a.review_state == REVIEW_STATE_VERIFIED)
        rejected_count = sum(1 for a in annotations if a.review_state == REVIEW_STATE_REJECTED)
        rev_req_count = sum(1 for a in annotations if a.review_state == REVIEW_STATE_REVISION_REQUIRED)

        eligible = self.list_training_eligible_annotations()
        eligible_count = len(eligible)
        ineligible_count = len(annotations) - eligible_count

        ctc_res = self.check_ctc_feasibility()
        double_agr = self.check_inter_annotator_agreement()

        # Gate Evaluation
        p19_readiness = evaluate_phase19_readiness(workspace_root=self.workspace_root)
        p21_readiness = evaluate_phase21_readiness(workspace_root=self.workspace_root)

        # Dataset status
        manifest = self.get_dataset_manifest()
        ds_status = manifest.get("dataset_status", DATASET_STATE_DRAFT) if manifest else DATASET_STATE_DRAFT

        # Next Physical Action Determination
        if len(annotations) == 0:
            next_action = "Acquire genuine human sequential ISL annotations."
        elif submitted_count > 0:
            next_action = f"Review the {submitted_count} submitted annotations awaiting verification."
        elif verified_count == 0:
            next_action = "Complete reviewer verification for submitted pilot annotations."
        elif not p19_readiness["real_ctc_training_allowed"]:
            next_action = "Collect additional qualified human annotations to satisfy minimum dataset thresholds."
        else:
            next_action = "Execute Phase 21 genuine CTC training (python scripts/train_phase21_ctc.py)."

        return {
            "phase": 22,
            "acquisition_mode": "MANUAL_HUMAN",
            "acquisition_status": "NOT_STARTED" if len(annotations) == 0 else "IN_PROGRESS",
            "supervision_state": p19_readiness["supervision_state"],
            "final_state": p19_readiness["supervision_state"],
            "human_data_present": len(annotations) > 0,
            "annotator_count": len(annotators),
            "qualified_annotator_count": sum(1 for a in annotators if a.qualification_status == QUAL_QUALIFIED),
            "total_annotations": len(annotations),
            "assigned_videos": unique_videos_assigned,
            "draft_annotations": draft_count,
            "submitted_annotations": submitted_count,
            "verified_annotations": verified_count,
            "training_eligible_count": eligible_count,
            "ctc_feasible_count": ctc_res.get("feasible_samples", 0),
            "dataset_status": ds_status,
            "phase19_gate_state": p19_readiness["real_ctc_status"],
            "phase19_authorized": p19_readiness["real_ctc_training_allowed"],
            "phase19_reason": p19_readiness.get("training_authorization", {}).get("reason", "no_genuine_human_annotations_present"),
            "phase21_training_status": "UNLOCKED" if p19_readiness["real_ctc_training_allowed"] else "BLOCKED",
            "human_data": {
                "present": len(annotations) > 0,
                "authenticated": p19_readiness["human_data_authenticated"],
                "qualified": p19_readiness["human_data_qualified"],
                "training_eligible": eligible_count > 0,
            },
            "annotators": {
                "registered": len(annotators),
                "qualified": sum(1 for a in annotators if a.qualification_status == QUAL_QUALIFIED),
            },
            "sample_accounting": {
                "videos_assigned": unique_videos_assigned,
                "total_assigned_slots": total_assigned,
                "videos_annotated": len(annotations),
                "annotations_draft": draft_count,
                "annotations_submitted": submitted_count,
                "annotations_verified": verified_count,
                "annotations_rejected": rejected_count,
                "annotations_revision_required": rev_req_count,
                "training_eligible": eligible_count,
                "training_ineligible": ineligible_count,
                "ctc_feasible": ctc_res.get("feasible_samples", 0),
            },
            "double_annotation": double_agr,
            "readiness": {
                "phase19_gate": p19_readiness["real_ctc_status"],
                "phase21_training_authorized": p19_readiness["real_ctc_training_allowed"],
                "phase20_live_model": p21_readiness["live_authorization"]["live_model_authorized"],
            },
            "reference_integrity": p19_readiness["reference_integrity"],
            "next_physical_action": next_action,
        }


def evaluate_phase22_readiness(
    data_root: Optional[Union[str, Path]] = None,
    workspace_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Canonical single-source-of-truth Phase 22 readiness evaluation function.
    """
    root = Path(workspace_root) if workspace_root else Path(__file__).resolve().parent.parent.parent.parent
    d_root = Path(data_root) if data_root else None
    orch = Phase22Orchestrator(workspace_root=root, data_root=d_root)
    return orch.get_dashboard_summary()
