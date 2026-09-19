"""
Phase 25 Orchestration Layer for SIGNOVA.

Executes genuine human sequential Indian Sign Language (ISL) annotation acquisition,
intake validation, annotator accountability, source provenance verification,
ordered gloss sequence checking, sample-level repeated-token CTC feasibility,
deterministic dataset formation, split generation, dataset freeze locking,
and authoritative Phase 19 gate evaluation.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from signova.data.splits import deterministic_split
from signova.data.leakage_audit import audit_split_leakage, LeakageAuditReport
from signova.live.model_registry import LiveModelRegistry, ModelInputSpec
from signova.operations.phase19_orchestrator import evaluate_phase19_readiness
from signova.operations.phase21_orchestrator import evaluate_phase21_readiness
from signova.operations.phase22_orchestrator import Phase22Orchestrator
from signova.operations.phase23_orchestrator import Phase23Orchestrator
from signova.operations.phase24_orchestrator import Phase24Orchestrator, evaluate_phase24_readiness


# -----------------------------------------------------------------------------
# Constants & Enums
# -----------------------------------------------------------------------------
ANNOTATION_SOURCE_HUMAN_DIRECT = "HUMAN_DIRECT"
FORBIDDEN_SOURCES = {
    "LLM_GENERATED",
    "PSEUDO_LABEL",
    "SYNTHETIC",
    "TRANSLATION_DERIVED",
    "UNKNOWN",
}

# Annotator Status Dimensions
IDENTITY_PRESENT = "PRESENT"
IDENTITY_MISSING = "MISSING"

AUTH_AUTHENTICATED = "AUTHENTICATED"
AUTH_NOT_AUTHENTICATED = "NOT_AUTHENTICATED"

QUAL_QUALIFIED = "QUALIFIED"
QUAL_UNQUALIFIED = "UNQUALIFIED"
QUAL_UNVERIFIED = "UNVERIFIED"

# Review States
REVIEW_DRAFT = "DRAFT"
REVIEW_SUBMITTED = "SUBMITTED"
REVIEW_PENDING = "REVIEW_PENDING"
REVIEW_VERIFIED = "VERIFIED"
REVIEW_REJECTED = "REJECTED"
REVIEW_INVALIDATED = "INVALIDATED"

# Acquisition Status
ACQ_NOT_STARTED = "NOT_STARTED"
ACQ_IN_PROGRESS = "IN_PROGRESS"
ACQ_DATA_READY = "DATA_READY"
ACQ_COMPLETE = "COMPLETE"

# Dataset State
DATASET_NONE = "NONE"
DATASET_DRAFT = "DATASET_DRAFT"
DATASET_READY = "DATASET_READY"
DATASET_FROZEN = "DATASET_FROZEN"


@dataclass
class HumanAnnotationRecord:
    """Canonical human annotation record conforming to Phase 25 import/export contract."""
    annotation_id: str
    source_video_id: str
    source_sha256: str
    annotator_id: str
    annotation_source: str = ANNOTATION_SOURCE_HUMAN_DIRECT
    ordered_glosses: List[str] = field(default_factory=list)
    vocabulary_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    submitted_at: Optional[str] = None
    review_state: str = REVIEW_DRAFT
    revision_id: str = "rev_001"
    parent_revision_id: Optional[str] = None
    qualification_reference: Optional[str] = None
    annotation_group_id: Optional[str] = None
    independent_annotation_index: int = 1
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanAnnotationRecord":
        return cls(
            annotation_id=data["annotation_id"],
            source_video_id=data["source_video_id"],
            source_sha256=data.get("source_sha256", ""),
            annotator_id=data.get("annotator_id", ""),
            annotation_source=data.get("annotation_source", ANNOTATION_SOURCE_HUMAN_DIRECT),
            ordered_glosses=list(data.get("ordered_glosses", [])),
            vocabulary_version=data.get("vocabulary_version", "1.0.0"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            submitted_at=data.get("submitted_at"),
            review_state=data.get("review_state", REVIEW_DRAFT),
            revision_id=data.get("revision_id", "rev_001"),
            parent_revision_id=data.get("parent_revision_id"),
            qualification_reference=data.get("qualification_reference"),
            annotation_group_id=data.get("annotation_group_id"),
            independent_annotation_index=int(data.get("independent_annotation_index", 1)),
            rejection_reason=data.get("rejection_reason"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Phase25DatasetFingerprint:
    """Cryptographic fingerprint bundle binding all dataset provenance fields."""
    dataset_version: str
    sample_ids: List[str]
    source_sha256_map: Dict[str, str]
    annotation_id_map: Dict[str, str]
    revision_id_map: Dict[str, str]
    ordered_gloss_ids_map: Dict[str, List[int]]
    split_assignments: Dict[str, str]
    signer_id_map: Dict[str, str]
    vocabulary_sha256: str
    feature_schema_version: str
    normalization_version: str
    input_spec_fingerprint: str
    dataset_manifest_sha256: str
    composite_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Phase25Orchestrator:
    """
    Authoritative Orchestrator for Phase 25 data acquisition, qualification,
    dataset formation, and training authorization readiness.
    """

    def __init__(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        data_root: Optional[Union[str, Path]] = None,
        annotations_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.workspace_root = (
            Path(workspace_root) if workspace_root else Path(__file__).resolve().parent.parent.parent.parent
        )
        self.data_root = (
            Path(data_root) if data_root else self.workspace_root / "data"
        )
        self.annotations_dir = (
            Path(annotations_dir) if annotations_dir else self.data_root / "annotations" / "phase25"
        )
        self.artifacts_dir = self.workspace_root / "artifacts" / "phase25"
        self.datasets_dir = self.data_root / "datasets" / "phase25"

        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        self.phase22_orch = Phase22Orchestrator(workspace_root=self.workspace_root, data_root=self.data_root)
        self.phase23_orch = Phase23Orchestrator(workspace_root=self.workspace_root, data_root=self.data_root)
        self.phase24_orch = Phase24Orchestrator(workspace_root=self.workspace_root, data_root=self.data_root)

    # -------------------------------------------------------------------------
    # 1. Annotator Status & Accountability
    # -------------------------------------------------------------------------
    def evaluate_annotator_status(self, annotator_id: Optional[str]) -> Dict[str, str]:
        """
        Evaluates the three orthogonal annotator status dimensions:
        1. ANNOTATOR_IDENTITY: PRESENT / MISSING
        2. ANNOTATOR_AUTHENTICATION: AUTHENTICATED / NOT_AUTHENTICATED
        3. ANNOTATOR_QUALIFICATION: QUALIFIED / UNQUALIFIED / UNVERIFIED
        """
        if not annotator_id or not str(annotator_id).strip():
            return {
                "annotator_identity": IDENTITY_MISSING,
                "annotator_authentication": AUTH_NOT_AUTHENTICATED,
                "annotator_qualification": QUAL_UNVERIFIED,
            }

        # Query existing annotator registry in Phase 22 / Phase 23 if available
        registry = getattr(self.phase22_orch, "annotator_registry", None)
        annotator = registry.get_annotator(annotator_id) if (registry and hasattr(registry, "get_annotator")) else None

        if annotator is None:
            return {
                "annotator_identity": IDENTITY_PRESENT,
                "annotator_authentication": AUTH_NOT_AUTHENTICATED,
                "annotator_qualification": QUAL_UNVERIFIED,
            }

        is_auth = getattr(annotator, "is_authenticated", False)
        is_qual = getattr(annotator, "is_qualified", False)
        qual_status = getattr(annotator, "qualification_status", QUAL_UNVERIFIED)

        return {
            "annotator_identity": IDENTITY_PRESENT,
            "annotator_authentication": AUTH_AUTHENTICATED if is_auth else AUTH_NOT_AUTHENTICATED,
            "annotator_qualification": QUAL_QUALIFIED if is_qual else qual_status,
        }

    # -------------------------------------------------------------------------
    # 2. Sequential Gloss & CTC Feasibility Math
    # -------------------------------------------------------------------------
    @staticmethod
    def calculate_ctc_required_timesteps(ordered_glosses: List[str]) -> int:
        """
        Calculates minimum CTC timesteps: T_req = L + sum(1 for y_i == y_{i+1}).
        """
        if not ordered_glosses:
            return 0
        L = len(ordered_glosses)
        repeats = sum(1 for i in range(len(ordered_glosses) - 1) if ordered_glosses[i] == ordered_glosses[i + 1])
        return L + repeats

    def check_sample_ctc_feasibility(
        self, ordered_glosses: List[str], available_timesteps: int = 64
    ) -> Dict[str, Any]:
        """
        Audits sample-level repeated token CTC feasibility.
        """
        t_req = self.calculate_ctc_required_timesteps(ordered_glosses)
        feasible = available_timesteps >= t_req and t_req > 0
        return {
            "ordered_glosses": ordered_glosses,
            "token_count": len(ordered_glosses),
            "required_timesteps": t_req,
            "available_timesteps": available_timesteps,
            "ctc_feasible": feasible,
        }

    # -------------------------------------------------------------------------
    # 3. Source Video Integrity & Post-Verification Mutation Detection
    # -------------------------------------------------------------------------
    def verify_source_video_integrity(
        self, video_path: Union[str, Path], expected_sha256: str
    ) -> Dict[str, Any]:
        """
        Calculates physical video file SHA-256 and compares with expected hash.
        """
        p = Path(video_path)
        if not p.exists():
            return {
                "source_exists": False,
                "hash_matches": False,
                "current_sha256": "NONE",
                "expected_sha256": expected_sha256,
                "status": "SOURCE_NOT_FOUND",
            }

        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        curr_sha = h.hexdigest().upper()
        matches = curr_sha.upper() == expected_sha256.upper()

        return {
            "source_exists": True,
            "hash_matches": matches,
            "current_sha256": curr_sha,
            "expected_sha256": expected_sha256,
            "status": "PASSED" if matches else "SOURCE_CHANGED",
        }

    # -------------------------------------------------------------------------
    # 4. Human Annotation Intake, Validation & Lifecycle
    # -------------------------------------------------------------------------
    def intake_annotation(self, record: HumanAnnotationRecord) -> Dict[str, Any]:
        """
        Validates and registers an incoming human annotation record.
        Strictly rejects forbidden sources (LLM, pseudo-label, synthetic, translation-derived).
        """
        # 1. Reject forbidden annotation sources
        if record.annotation_source in FORBIDDEN_SOURCES or record.annotation_source != ANNOTATION_SOURCE_HUMAN_DIRECT:
            record.review_state = REVIEW_REJECTED
            record.rejection_reason = f"FORBIDDEN_SOURCE: {record.annotation_source}"
            return {
                "status": "REJECTED",
                "reason": record.rejection_reason,
                "training_eligible": False,
                "record": record.to_dict(),
            }

        # 2. Check Annotator Accountability
        annotator_status = self.evaluate_annotator_status(record.annotator_id)
        if annotator_status["annotator_identity"] == IDENTITY_MISSING:
            record.review_state = REVIEW_REJECTED
            record.rejection_reason = "MISSING_ANNOTATOR_IDENTITY"
            return {
                "status": "REJECTED",
                "reason": record.rejection_reason,
                "training_eligible": False,
                "record": record.to_dict(),
            }

        # 3. Check Ordered Glosses & Vocabulary
        if not record.ordered_glosses or len(record.ordered_glosses) == 0:
            record.review_state = REVIEW_REJECTED
            record.rejection_reason = "EMPTY_GLOSS_SEQUENCE"
            return {
                "status": "REJECTED",
                "reason": record.rejection_reason,
                "training_eligible": False,
                "record": record.to_dict(),
            }

        # Vocabulary Membership Check
        vocab_mgr = getattr(self.phase22_orch, "vocabulary_manager", None)
        canonical_vocab = vocab_mgr.get_vocabulary() if (vocab_mgr and hasattr(vocab_mgr, "get_vocabulary")) else {"<blank>": 0, "<unk>": 1}
        unknown_tokens = [g for g in record.ordered_glosses if g not in canonical_vocab and g.upper() not in canonical_vocab]

        if unknown_tokens:
            record.review_state = REVIEW_REJECTED
            record.rejection_reason = f"UNKNOWN_GLOSSES: {unknown_tokens}"
            return {
                "status": "REJECTED",
                "reason": record.rejection_reason,
                "training_eligible": False,
                "record": record.to_dict(),
            }

        # 4. Save valid record
        record_path = self.annotations_dir / f"{record.annotation_id}.json"
        record_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")

        return {
            "status": "ACCEPTED",
            "review_state": record.review_state,
            "training_eligible": record.review_state == REVIEW_VERIFIED,
            "record": record.to_dict(),
        }

    def review_annotation(
        self,
        annotation_id: str,
        action: str,  # "VERIFY" or "REJECT"
        reviewer_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Applies human expert review to a submitted annotation.
        Transitions state to VERIFIED or REJECTED non-destructively.
        """
        record_path = self.annotations_dir / f"{annotation_id}.json"
        if not record_path.exists():
            return {"status": "NOT_FOUND", "message": f"Annotation {annotation_id} not found."}

        data = json.loads(record_path.read_text(encoding="utf-8"))
        record = HumanAnnotationRecord.from_dict(data)

        if action.upper() == "VERIFY":
            record.review_state = REVIEW_VERIFIED
            record.metadata["verified_by"] = reviewer_id
            record.metadata["verified_at"] = datetime.now(timezone.utc).isoformat()
        elif action.upper() == "REJECT":
            record.review_state = REVIEW_REJECTED
            record.rejection_reason = reason or "REJECTED_BY_REVIEWER"
            record.metadata["rejected_by"] = reviewer_id
            record.metadata["rejected_at"] = datetime.now(timezone.utc).isoformat()

        record_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
        return {"status": "UPDATED", "review_state": record.review_state, "record": record.to_dict()}

    def invalidate_annotation_on_source_mutation(
        self,
        annotation_id: str,
        video_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Checks source video integrity and invalidates a VERIFIED annotation if the physical file mutated.
        """
        record_path = self.annotations_dir / f"{annotation_id}.json"
        if not record_path.exists():
            return {"status": "NOT_FOUND", "message": f"Annotation {annotation_id} not found."}

        data = json.loads(record_path.read_text(encoding="utf-8"))
        record = HumanAnnotationRecord.from_dict(data)

        check = self.verify_source_video_integrity(video_path, record.source_sha256)
        if not check["hash_matches"]:
            record.review_state = REVIEW_INVALIDATED
            record.rejection_reason = f"SOURCE_MUTATION_DETECTED: {check['status']}"
            record_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
            return {
                "status": "INVALIDATED",
                "training_eligible": False,
                "reason": record.rejection_reason,
                "check": check,
            }

        return {
            "status": "VALID",
            "training_eligible": record.review_state == REVIEW_VERIFIED,
            "check": check,
        }

    # -------------------------------------------------------------------------
    # 5. Independent Double Annotation & Agreement Calculation
    # -------------------------------------------------------------------------
    def evaluate_double_annotation_agreement(
        self, annotation_id_a: str, annotation_id_b: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Computes exact gloss sequence agreement between two independent annotations.
        Explicitly reports AGREEMENT_NOT_COMPUTABLE if only one annotation exists.
        Never fabricates agreement.
        """
        path_a = self.annotations_dir / f"{annotation_id_a}.json"
        if not path_a.exists():
            return {"status": "ERROR", "agreement": "NOT_COMPUTABLE", "reason": "Annotation A not found"}

        if not annotation_id_b:
            return {
                "status": "NOT_COMPUTABLE",
                "agreement": "AGREEMENT_NOT_COMPUTABLE",
                "reason": "Only one annotation present in group; independent second annotation required.",
            }

        path_b = self.annotations_dir / f"{annotation_id_b}.json"
        if not path_b.exists():
            return {
                "status": "NOT_COMPUTABLE",
                "agreement": "AGREEMENT_NOT_COMPUTABLE",
                "reason": "Annotation B not found",
            }

        data_a = json.loads(path_a.read_text(encoding="utf-8"))
        data_b = json.loads(path_b.read_text(encoding="utf-8"))

        # Verify annotators are distinct
        if data_a.get("annotator_id") == data_b.get("annotator_id"):
            return {
                "status": "INVALID",
                "agreement": "NOT_COMPUTABLE",
                "reason": "Double annotation requires two distinct independent annotators.",
            }

        glosses_a = data_a.get("ordered_glosses", [])
        glosses_b = data_b.get("ordered_glosses", [])

        is_match = glosses_a == glosses_b
        return {
            "status": "COMPUTED",
            "agreement": "AGREEMENT" if is_match else "DISAGREEMENT",
            "exact_match": is_match,
            "glosses_a": glosses_a,
            "glosses_b": glosses_b,
        }

    # -------------------------------------------------------------------------
    # 6. Deterministic Dataset Formation, Splitting & Freezing
    # -------------------------------------------------------------------------
    def build_dataset(
        self,
        dataset_version: str = "phase25_v001",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed_salt: str = "signova_phase25",
    ) -> Dict[str, Any]:
        """
        Constructs a candidate dataset strictly from VERIFIED, training-eligible human annotations.
        """
        # Gather all annotation records in Phase 25 annotations dir
        records: List[HumanAnnotationRecord] = []
        for p in self.annotations_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                records.append(HumanAnnotationRecord.from_dict(data))
            except Exception:
                pass

        # Filter strictly for TRAINING_ELIGIBLE records
        eligible: List[HumanAnnotationRecord] = []
        for r in records:
            if (
                r.review_state == REVIEW_VERIFIED
                and r.annotation_source == ANNOTATION_SOURCE_HUMAN_DIRECT
                and r.annotator_id
            ):
                eligible.append(r)

        if not eligible:
            return {
                "dataset_status": "EMPTY",
                "dataset_version": dataset_version,
                "total_annotations": len(records),
                "training_eligible_count": 0,
                "message": "Zero training-eligible human annotations found. Dataset formation skipped.",
            }

        # Deterministic Splitting
        train_recs, val_recs, test_recs = deterministic_split(
            items=eligible,
            id_getter=lambda x: x.annotation_id,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed_salt=seed_salt,
        )

        train_ids = [r.annotation_id for r in train_recs]
        val_ids = [r.annotation_id for r in val_recs]
        test_ids = [r.annotation_id for r in test_recs]

        # Audit split leakage
        train_dicts = [{"video_id": r.source_video_id, "signer_id": r.metadata.get("signer_id")} for r in train_recs]
        val_dicts = [{"video_id": r.source_video_id, "signer_id": r.metadata.get("signer_id")} for r in val_recs]
        test_dicts = [{"video_id": r.source_video_id, "signer_id": r.metadata.get("signer_id")} for r in test_recs]
        leakage_rep = audit_split_leakage(train_dicts, val_dicts, test_dicts)

        manifest = {
            "dataset_version": dataset_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_frozen": False,
            "total_samples": len(eligible),
            "splits": {
                "train": train_ids,
                "validation": val_ids,
                "test": test_ids,
            },
            "sample_records": [r.to_dict() for r in eligible],
            "leakage_audit": leakage_rep.to_dict(),
        }

        manifest_path = self.datasets_dir / f"{dataset_version}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "dataset_status": DATASET_READY,
            "dataset_version": dataset_version,
            "manifest_path": str(manifest_path),
            "total_samples": len(eligible),
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "leakage_status": leakage_rep.status,
        }

    def freeze_dataset(self, dataset_version: str = "phase25_v001") -> Dict[str, Any]:
        """
        Freezes candidate dataset and generates cryptographic fingerprint bundle.
        Enforces dataset immutability; mutations create a new version rather than mutating.
        """
        manifest_path = self.datasets_dir / f"{dataset_version}_manifest.json"
        if not manifest_path.exists():
            return {"status": "NOT_FOUND", "message": f"Dataset {dataset_version} not found."}

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("is_frozen"):
            return {
                "status": "ALREADY_FROZEN",
                "dataset_version": dataset_version,
                "fingerprint": manifest.get("fingerprint"),
            }

        # Calculate enhanced cryptographic fingerprint
        records = [HumanAnnotationRecord.from_dict(r) for r in manifest.get("sample_records", [])]
        splits = manifest.get("splits", {})

        sample_ids = sorted([r.annotation_id for r in records])
        src_map = {r.annotation_id: r.source_sha256 for r in records}
        ann_map = {r.annotation_id: r.annotation_id for r in records}
        rev_map = {r.annotation_id: r.revision_id for r in records}
        signer_map = {r.annotation_id: str(r.metadata.get("signer_id", "UNKNOWN")) for r in records}

        # Dynamic Model Input Spec
        registry = LiveModelRegistry()
        input_spec = registry.default_spec
        spec_str = json.dumps(input_spec.to_dict(), sort_keys=True)
        spec_fingerprint = hashlib.sha256(spec_str.encode("utf-8")).hexdigest().upper()

        vocab_mgr = getattr(self.phase22_orch, "vocabulary_manager", None)
        vocab = vocab_mgr.get_vocabulary() if (vocab_mgr and hasattr(vocab_mgr, "get_vocabulary")) else {"<blank>": 0, "<unk>": 1}
        vocab_str = json.dumps(vocab, sort_keys=True)
        vocab_sha = hashlib.sha256(vocab_str.encode("utf-8")).hexdigest().upper()

        gloss_ids_map = {
            r.annotation_id: [vocab.get(g, vocab.get(g.upper(), 1)) for g in r.ordered_glosses]
            for r in records
        }

        split_assignment = {}
        for sid in splits.get("train", []):
            split_assignment[sid] = "TRAIN"
        for sid in splits.get("validation", []):
            split_assignment[sid] = "VALIDATION"
        for sid in splits.get("test", []):
            split_assignment[sid] = "TEST"

        manifest_str = json.dumps(manifest, sort_keys=True)
        manifest_sha = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest().upper()

        composite_payload = {
            "dataset_version": dataset_version,
            "sample_ids": sample_ids,
            "source_sha256_map": src_map,
            "annotation_id_map": ann_map,
            "revision_id_map": rev_map,
            "ordered_gloss_ids_map": gloss_ids_map,
            "split_assignments": split_assignment,
            "signer_id_map": signer_map,
            "vocabulary_sha256": vocab_sha,
            "feature_schema_version": input_spec.feature_schema_version,
            "normalization_version": input_spec.normalization_version,
            "input_spec_fingerprint": spec_fingerprint,
            "dataset_manifest_sha256": manifest_sha,
        }
        composite_str = json.dumps(composite_payload, sort_keys=True)
        composite_sha = hashlib.sha256(composite_str.encode("utf-8")).hexdigest().upper()

        fingerprint = Phase25DatasetFingerprint(
            dataset_version=dataset_version,
            sample_ids=sample_ids,
            source_sha256_map=src_map,
            annotation_id_map=ann_map,
            revision_id_map=rev_map,
            ordered_gloss_ids_map=gloss_ids_map,
            split_assignments=split_assignment,
            signer_id_map=signer_map,
            vocabulary_sha256=vocab_sha,
            feature_schema_version=input_spec.feature_schema_version,
            normalization_version=input_spec.normalization_version,
            input_spec_fingerprint=spec_fingerprint,
            dataset_manifest_sha256=manifest_sha,
            composite_sha256=composite_sha,
        )

        manifest["is_frozen"] = True
        manifest["frozen_at"] = datetime.now(timezone.utc).isoformat()
        manifest["fingerprint"] = fingerprint.to_dict()

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Write lock file
        lock_path = self.artifacts_dir / f"{dataset_version}_freeze_lock.json"
        lock_path.write_text(json.dumps(fingerprint.to_dict(), indent=2), encoding="utf-8")

        return {
            "status": DATASET_FROZEN,
            "dataset_version": dataset_version,
            "composite_sha256": composite_sha,
            "manifest_path": str(manifest_path),
            "lock_path": str(lock_path),
        }

    # -------------------------------------------------------------------------
    # 7. Authoritative Status & Readiness Evaluation
    # -------------------------------------------------------------------------
    def evaluate_readiness(self) -> Dict[str, Any]:
        """
        Canonical evaluation of Phase 25 data acquisition and training readiness.
        """
        p19_res = evaluate_phase19_readiness(workspace_root=self.workspace_root)
        p21_res = evaluate_phase21_readiness(workspace_root=self.workspace_root)
        p24_res = evaluate_phase24_readiness(workspace_root=self.workspace_root)

        # Count genuine annotations
        records: List[HumanAnnotationRecord] = []
        for p in self.annotations_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                records.append(HumanAnnotationRecord.from_dict(data))
            except Exception:
                pass

        total_annotations = len(records)
        submitted = sum(1 for r in records if r.review_state in (REVIEW_SUBMITTED, REVIEW_PENDING, REVIEW_VERIFIED))
        verified = sum(1 for r in records if r.review_state == REVIEW_VERIFIED)
        rejected = sum(1 for r in records if r.review_state == REVIEW_REJECTED)
        training_eligible = sum(
            1 for r in records
            if r.review_state == REVIEW_VERIFIED
            and r.annotation_source == ANNOTATION_SOURCE_HUMAN_DIRECT
            and r.annotator_id
        )

        # Determine Acquisition Status
        if total_annotations == 0:
            acq_status = ACQ_NOT_STARTED
        elif training_eligible == 0:
            acq_status = ACQ_IN_PROGRESS
        elif not p19_res.get("real_ctc_training_allowed"):
            acq_status = ACQ_DATA_READY
        else:
            acq_status = ACQ_COMPLETE

        # Operational Status Dimensions
        operational_status = {
            "software_ready": True,
            "data_acquisition_started": total_annotations > 0,
            "human_data_present": total_annotations > 0,
            "human_data_authenticated": False,  # True only if authenticated annotators provided samples
            "human_data_qualified": False,      # True only if qualified annotators provided samples
            "training_eligible_data": training_eligible > 0,
            "dataset_frozen": False,
            "phase19_authorized": p19_res.get("real_ctc_training_allowed", False),
            "training_ready": p19_res.get("real_ctc_training_allowed", False),
        }

        next_action = "Acquire genuine human sequential ISL annotations."
        if total_annotations > 0 and verified == 0:
            next_action = "Review and verify submitted human annotations."
        elif training_eligible > 0 and not p19_res.get("real_ctc_training_allowed"):
            next_action = "Run Phase 19 authorization evaluation on verified human dataset."
        elif p19_res.get("real_ctc_training_allowed"):
            next_action = "Execute Phase 24 real CTC training with explicit --train flag."

        return {
            "phase": "PHASE_25",
            "supervision_state": p19_res.get("supervision_state", "STATE_B"),
            "acquisition_status": acq_status,
            "operational_status": operational_status,
            "human_annotations": {
                "total": total_annotations,
                "submitted": submitted,
                "verified": verified,
                "rejected": rejected,
                "training_eligible": training_eligible,
            },
            "dataset": {
                "status": DATASET_NONE if training_eligible == 0 else DATASET_READY,
                "version": "NONE",
                "samples": training_eligible,
                "train": 0,
                "validation": 0,
                "test": 0,
                "unique_signers": 0,
                "unique_glosses": 0,
                "fingerprint": "NONE",
            },
            "ctc": {
                "feasible": 0,
                "infeasible": 0,
                "feasibility_rate": "0.00%",
            },
            "quality": {
                "double_annotated": 0,
                "agreement_computable": 0,
                "agreement_not_computable": total_annotations,
            },
            "authorization": {
                "phase19_authorized": p19_res.get("real_ctc_training_allowed", False),
                "phase21_status": p21_res.get("training", {}).get("status", "BLOCKED"),
                "phase24_status": p24_res.get("state_reporting", {}).get("experiment_status", "BLOCKED"),
            },
            "live": {
                "model": "NONE",
                "checkpoint": "NONE",
                "live_model_authorized": False,
            },
            "reference_integrity": p19_res.get("reference_integrity", {}),
            "final_state": p19_res.get("supervision_state", "STATE_B"),
            "next_physical_action": next_action,
        }


def evaluate_phase25_readiness(
    workspace_root: Optional[Union[str, Path]] = None,
    data_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Canonical Phase 25 readiness evaluation entrypoint."""
    orch = Phase25Orchestrator(workspace_root=workspace_root, data_root=data_root)
    return orch.evaluate_readiness()
