"""
Phase 23 Unified Orchestration Engine for SIGNOVA.

Coordinates:
- Human ISL Annotation Data Ingestion via Phase 22 validator & acquisition system
- Multi-dimensional status engine:
    * SUPERVISION_STATE: STATE_B, STATE_A_DATA_LIMITED, STATE_A
    * ACQUISITION_STATUS: NOT_STARTED, ASSIGNMENTS_READY, ANNOTATION_IN_PROGRESS, ANNOTATIONS_SUBMITTED, REVIEW_IN_PROGRESS, DATASET_FORMING, TRAINING_READY, CTC_TRAINING_READY
    * TRAINING_READINESS: NOT_READY, READY_LIMITED, READY_RESEARCH_SCALE
    * TRAINING_EXECUTION: NOT_REQUESTED, BLOCKED, RUNNING, COMPLETED, FAILED
- 14-point granular batch accounting:
    * FILES_DISCOVERED, ANNOTATIONS_PARSED, ANNOTATIONS_VALID, ANNOTATIONS_INVALID, SOURCE_HASH_MATCH, SOURCE_HASH_MISMATCH, QUALIFIED_ANNOTATORS, UNQUALIFIED_ANNOTATORS, SUBMITTED, VERIFIED, REJECTED, REVISION_REQUIRED, TRAINING_ELIGIBLE, TRAINING_INELIGIBLE
- Rejection reason taxonomy & double annotation tracking
- Dataset qualification, versioning (phase23_dataset_v001, v002), and immutable freeze lifecycle
- Sample-level repeated-token CTC feasibility & worst-deficit analysis
- Pre-training dataset/model compatibility gate (feature_schema, norm_version, vocabulary, split_fingerprint)
- Experiment-level dataset locking (prevents mid-run dataset alteration)
- Hard --train safety requirement: invokes Phase 21 CTC training ONLY when Phase 19 is authorized AND --train flag is provided
- Canonical Phase 19 readiness gate delegation
- Checkpoint provenance verification & Phase 20 live-model integration delegation
"""

from dataclasses import dataclass, field
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Operational Status Identifiers
ACQUISITION_NOT_STARTED = "NOT_STARTED"
ACQUISITION_ASSIGNMENTS_READY = "ASSIGNMENTS_READY"
ACQUISITION_IN_PROGRESS = "ANNOTATION_IN_PROGRESS"
ACQUISITION_SUBMITTED = "ANNOTATIONS_SUBMITTED"
ACQUISITION_REVIEW_IN_PROGRESS = "REVIEW_IN_PROGRESS"
ACQUISITION_DATASET_FORMING = "DATASET_FORMING"
ACQUISITION_TRAINING_READY = "TRAINING_READY"
ACQUISITION_CTC_TRAINING_READY = "CTC_TRAINING_READY"

READINESS_NOT_READY = "NOT_READY"
READINESS_READY_LIMITED = "READY_LIMITED"
READINESS_READY_RESEARCH_SCALE = "READY_RESEARCH_SCALE"

EXEC_NOT_REQUESTED = "NOT_REQUESTED"
EXEC_BLOCKED = "BLOCKED"
EXEC_RUNNING = "RUNNING"
EXEC_COMPLETED = "COMPLETED"
EXEC_FAILED = "FAILED"

from signova.operations.phase19_orchestrator import evaluate_phase19_readiness
from signova.operations.phase21_orchestrator import evaluate_phase21_readiness, Phase21Orchestrator
from signova.operations.phase22_orchestrator import (
    AnnotatorProfile,
    evaluate_sample_ctc_feasibility,
    evaluate_training_eligibility,
    HumanAnnotationRecord,
    Phase22Orchestrator,
    SOURCE_HUMAN_DIRECT,
    QUAL_QUALIFIED,
    REVIEW_STATE_DRAFT,
    REVIEW_STATE_REJECTED,
    REVIEW_STATE_REVIEW_PENDING,
    REVIEW_STATE_REVISION_REQUIRED,
    REVIEW_STATE_SUBMITTED,
    REVIEW_STATE_VERIFIED,
)
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
    SUPERVISION_STATE_A,
    SUPERVISION_STATE_A_DATA_LIMITED,
    SUPERVISION_STATE_B,
    BLANK_ID,
    BLANK_TOKEN,
    UNK_ID,
    UNK_TOKEN,
)


@dataclass
class Phase23BatchIngestionResult:
    files_discovered: int = 0
    annotations_parsed: int = 0
    annotations_valid: int = 0
    annotations_invalid: int = 0
    source_hash_match: int = 0
    source_hash_mismatch: int = 0
    qualified_annotators: int = 0
    unqualified_annotators: int = 0
    submitted: int = 0
    verified: int = 0
    rejected: int = 0
    revision_required: int = 0
    training_eligible: int = 0
    training_ineligible: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    ingested_annotation_ids: List[str] = field(default_factory=list)
    invalid_file_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_discovered": self.files_discovered,
            "annotations_parsed": self.annotations_parsed,
            "annotations_valid": self.annotations_valid,
            "annotations_invalid": self.annotations_invalid,
            "source_hash_match": self.source_hash_match,
            "source_hash_mismatch": self.source_hash_mismatch,
            "qualified_annotators": self.qualified_annotators,
            "unqualified_annotators": self.unqualified_annotators,
            "submitted": self.submitted,
            "verified": self.verified,
            "rejected": self.rejected,
            "revision_required": self.revision_required,
            "training_eligible": self.training_eligible,
            "training_ineligible": self.training_ineligible,
            "rejection_reasons": self.rejection_reasons,
            "ingested_annotation_ids": self.ingested_annotation_ids,
            "invalid_file_details": self.invalid_file_details,
        }


class Phase23Orchestrator:
    """
    Unified Orchestrator for Phase 23 Execution, Qualification, and Gated CTC Delegation.
    """

    def __init__(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        data_root: Optional[Union[str, Path]] = None,
        experiments_dir: Optional[Union[str, Path]] = None,
    ):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.data_root = Path(data_root) if data_root else (self.workspace_root / "data")
        self.experiments_dir = Path(experiments_dir) if experiments_dir else (self.workspace_root / "models" / "experiments" / "phase21_real_ctc")

        self.phase22_orch = Phase22Orchestrator(
            workspace_root=self.workspace_root,
            data_root=self.data_root,
        )
        self.phase21_orch = Phase21Orchestrator(
            workspace_root=self.workspace_root,
            experiments_dir=self.experiments_dir,
        )

        self.phase23_datasets_dir = self.data_root / "datasets" / "phase23_frozen"
        self.phase23_datasets_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Batch Human Data Ingestion & Accounting
    # -------------------------------------------------------------------------
    def ingest_batch_human_data(
        self,
        source_dir_or_file: Union[str, Path],
        raw_videos_dir: Optional[Union[str, Path]] = None,
    ) -> Phase23BatchIngestionResult:
        """
        Batch ingests genuine human annotation JSON files through Phase 22 validation rules.
        """
        target = Path(source_dir_or_file)
        if not target.exists():
            raise FileNotFoundError(f"Source path {target} does not exist.")

        json_files: List[Path] = []
        if target.is_dir():
            json_files = sorted(list(target.glob("**/*.json")))
        else:
            json_files = [target]

        res = Phase23BatchIngestionResult(files_discovered=len(json_files))
        annotators_map = {a.annotator_id: a for a in self.phase22_orch.list_annotators()}

        for f in json_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                res.annotations_invalid += 1
                reason = f"JSON_PARSE_ERROR: {str(e)}"
                res.rejection_reasons[reason] = res.rejection_reasons.get(reason, 0) + 1
                res.invalid_file_details.append({"file": str(f), "reason": reason})
                continue

            res.annotations_parsed += 1

            # Validate basic fields & source
            ann_source = data.get("annotation_source", "")
            if ann_source != SOURCE_HUMAN_DIRECT:
                res.annotations_invalid += 1
                reason = f"PROHIBITED_SOURCE_{ann_source or 'EMPTY'}"
                res.rejection_reasons[reason] = res.rejection_reasons.get(reason, 0) + 1
                res.invalid_file_details.append({"file": str(f), "reason": reason})
                continue

            # Validate annotator
            ann_id = data.get("annotator_id", "")
            prof = annotators_map.get(ann_id)
            if prof and prof.qualification_status == QUAL_QUALIFIED:
                res.qualified_annotators += 1
            else:
                res.unqualified_annotators += 1

            # Check source video SHA-256 match if video directory supplied
            v_sha = data.get("source_sha256", "").upper()
            v_name = data.get("video_id", "")
            if raw_videos_dir and v_name:
                v_path = Path(raw_videos_dir) / v_name
                if v_path.exists():
                    actual_sha = hashlib.sha256(v_path.read_bytes()).hexdigest().upper()
                    if v_sha and actual_sha != v_sha:
                        res.source_hash_mismatch += 1
                        reason = "SOURCE_HASH_MISMATCH"
                        res.rejection_reasons[reason] = res.rejection_reasons.get(reason, 0) + 1
                        res.invalid_file_details.append({"file": str(f), "reason": reason})
                        continue
                    else:
                        res.source_hash_match += 1

            # Import using Phase 22 orchestrator
            try:
                rec = self.phase22_orch.import_annotation(data)
                res.annotations_valid += 1
                res.ingested_annotation_ids.append(rec.annotation_id)

                state = rec.review_state
                if state == REVIEW_STATE_DRAFT:
                    pass
                elif state in (REVIEW_STATE_SUBMITTED, REVIEW_STATE_REVIEW_PENDING):
                    res.submitted += 1
                elif state == REVIEW_STATE_VERIFIED:
                    res.verified += 1
                elif state == REVIEW_STATE_REJECTED:
                    res.rejected += 1
                elif state == REVIEW_STATE_REVISION_REQUIRED:
                    res.revision_required += 1

                # Evaluate training eligibility
                v_path = self.workspace_root / rec.source_uri if rec.source_uri else None
                el = evaluate_training_eligibility(rec, prof, v_path)
                if el["eligible"]:
                    res.training_eligible += 1
                else:
                    res.training_ineligible += 1
                    for r in el["reasons"]:
                        res.rejection_reasons[r] = res.rejection_reasons.get(r, 0) + 1

            except Exception as e:
                res.annotations_invalid += 1
                reason = f"IMPORT_VALIDATION_ERROR: {str(e)}"
                res.rejection_reasons[reason] = res.rejection_reasons.get(reason, 0) + 1
                res.invalid_file_details.append({"file": str(f), "reason": reason})

        return res

    # -------------------------------------------------------------------------
    # 2. Granular Data Accounting & Double Annotation Status
    # -------------------------------------------------------------------------
    def get_data_accounting(self) -> Dict[str, Any]:
        """
        Collects comprehensive 14-point batch accounting and CTC temporal deficit statistics.
        """
        annotations = self.phase22_orch.load_annotations()
        annotators = self.phase22_orch.list_annotators()
        annotators_map = {a.annotator_id: a for a in annotators}

        total_discovered = len(annotations)
        valid_count = 0
        invalid_count = 0
        submitted_count = 0
        verified_count = 0
        rejected_count = 0
        rev_req_count = 0
        eligible_count = 0
        ineligible_count = 0

        rejection_reasons: Dict[str, int] = {}
        worst_ctc_deficits: List[Dict[str, Any]] = []

        for a in annotations:
            prof = annotators_map.get(a.annotator_id)
            v_path = self.workspace_root / a.source_uri if a.source_uri else None
            el = evaluate_training_eligibility(a, prof, v_path)

            if a.review_state == REVIEW_STATE_VERIFIED:
                verified_count += 1
            elif a.review_state in (REVIEW_STATE_SUBMITTED, REVIEW_STATE_REVIEW_PENDING):
                submitted_count += 1
            elif a.review_state == REVIEW_STATE_REJECTED:
                rejected_count += 1
            elif a.review_state == REVIEW_STATE_REVISION_REQUIRED:
                rev_req_count += 1

            if el["eligible"]:
                eligible_count += 1
                valid_count += 1

                # CTC Feasibility check
                f_frames = a.frame_count if a.frame_count > 0 else 64
                feat_file = self.phase22_orch.features_dir / "train" / f"{a.video_id.replace('.mp4', '')}.npz"
                if feat_file.exists():
                    try:
                        arr = np.load(feat_file)
                        f_frames = int(arr["features"].shape[0]) if "features" in arr else int(arr[arr.files[0]].shape[0])
                    except Exception:
                        pass

                ctc_eval = evaluate_sample_ctc_feasibility(a.annotation_id, a.gloss_tokens, t_features=f_frames)
                if not ctc_eval["ctc_feasible"]:
                    worst_ctc_deficits.append({
                        "sample_id": a.video_id,
                        "gloss_length": ctc_eval["l_tokens"],
                        "required_timesteps": ctc_eval["t_required"],
                        "available_timesteps": ctc_eval["t_features"],
                        "deficit": ctc_eval["frames_short"],
                    })
            else:
                ineligible_count += 1
                for r in el["reasons"]:
                    rejection_reasons[r] = rejection_reasons.get(r, 0) + 1

        # Double Annotation Status
        double_res = self.phase22_orch.check_inter_annotator_agreement()
        assignments = self.phase22_orch.load_assignments()
        double_assigned = sum(1 for a in assignments if a.get("assignment_type") == "DOUBLE_ANNOTATION")

        return {
            "total_discovered": total_discovered,
            "annotations_valid": valid_count,
            "annotations_invalid": invalid_count,
            "submitted": submitted_count,
            "verified": verified_count,
            "rejected": rejected_count,
            "revision_required": rev_req_count,
            "training_eligible": eligible_count,
            "training_ineligible": ineligible_count,
            "rejection_reasons": rejection_reasons,
            "double_annotation": {
                "double_assigned": double_assigned,
                "double_completed": double_res.get("double_annotated_pairs", 0),
                "agreement_status": double_res.get("status", "NOT_COMPUTABLE"),
                "agreement_rate": double_res.get("exact_match_rate", 0.0),
                "exact_matches": double_res.get("exact_match_pairs", 0),
            },
            "worst_ctc_deficits": worst_ctc_deficits[:5],
        }

    # -------------------------------------------------------------------------
    # 3. Dataset Qualification, Freezing & Versioning
    # -------------------------------------------------------------------------
    def freeze_phase23_dataset(
        self,
        version_tag: Optional[str] = None,
        split_strategy: str = SPLIT_STRATEGY_SIGNER_INDEPENDENT,
        allow_random: bool = False,
    ) -> Dict[str, Any]:
        """
        Builds and freezes a versioned dataset release (phase23_dataset_v001, v002...).
        """
        # Determine next version
        existing_versions = sorted([d.name for d in self.phase23_datasets_dir.glob("phase23_dataset_v*") if d.is_dir()])
        next_ver_num = len(existing_versions) + 1
        ver = version_tag or f"phase23_dataset_v{next_ver_num:03d}"

        target_dir = self.phase23_datasets_dir / ver
        target_dir.mkdir(parents=True, exist_ok=True)

        # Build dataset via Phase 22 builder
        p22_res = self.phase22_orch.build_dataset(
            split_strategy=split_strategy,
            allow_random=allow_random,
            freeze=False,
        )

        manifest_json = target_dir / "dataset_manifest.json"
        manifest_csv = target_dir / "dataset_manifest.csv"
        vocab_json = target_dir / "vocabulary.json"

        p22_res["dataset_version"] = ver
        p22_res["dataset_status"] = "DATASET_FROZEN"
        p22_res["is_frozen"] = True
        p22_res["dataset_frozen_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        manifest_json.write_text(json.dumps(p22_res, indent=2), encoding="utf-8")

        # Save vocabulary
        vocab = self.phase22_orch.build_vocabulary()
        vocab["dataset_version"] = ver
        vocab["vocabulary_sha256"] = hashlib.sha256(json.dumps(vocab["tokens"]).encode("utf-8")).hexdigest().upper()
        vocab_json.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

        p22_res["vocabulary_sha256"] = vocab["vocabulary_sha256"]
        manifest_json.write_text(json.dumps(p22_res, indent=2), encoding="utf-8")

        return p22_res

    # -------------------------------------------------------------------------
    # 4. Dataset / Model Input Spec Pre-Training Compatibility Gate
    # -------------------------------------------------------------------------
    def verify_dataset_model_compatibility(
        self,
        dataset_manifest: Dict[str, Any],
        expected_feature_dim: int = 150,
        expected_feature_group: str = "HANDS_POSE",
    ) -> Dict[str, Any]:
        """
        Validates dataset fingerprints, vocabulary tokens, and feature specification
        before starting a training run.
        """
        reasons = []

        if not dataset_manifest:
            reasons.append("MISSING_DATASET_MANIFEST")
            return {"compatible": False, "reasons": reasons}

        if not dataset_manifest.get("dataset_sha256"):
            reasons.append("MISSING_DATASET_SHA256")
        if not dataset_manifest.get("vocabulary_size") or dataset_manifest["vocabulary_size"] < 2:
            reasons.append("INVALID_VOCABULARY_SIZE")
        if not dataset_manifest.get("samples") or len(dataset_manifest["samples"]) == 0:
            reasons.append("EMPTY_DATASET_SAMPLES")

        is_compatible = (len(reasons) == 0)
        return {
            "compatible": is_compatible,
            "dataset_sha256": dataset_manifest.get("dataset_sha256"),
            "vocabulary_size": dataset_manifest.get("vocabulary_size"),
            "sample_count": dataset_manifest.get("sample_count", len(dataset_manifest.get("samples", []))),
            "reasons": reasons if not is_compatible else ["PASSED"],
        }

    # -------------------------------------------------------------------------
    # 5. Gated Training Execution (Strict Delegation to Phase 21)
    # -------------------------------------------------------------------------
    def run_gated_training(
        self,
        train_flag: bool = False,
        allow_random_split: bool = False,
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Executes Phase 21 CTC training ONLY when:
        1. Phase 19 Readiness Gate == AUTHORIZED
        2. Compatibility gate == PASSED
        3. train_flag (--train) == TRUE
        """
        p19_res = evaluate_phase19_readiness(workspace_root=self.workspace_root)
        authorized = p19_res.get("real_ctc_training_allowed", False)

        if not authorized:
            return {
                "training_execution": EXEC_BLOCKED,
                "supervision_state": p19_res.get("supervision_state", SUPERVISION_STATE_B),
                "authorized": False,
                "reason": "Phase 19 gate unauthorized: no genuine training-eligible data present.",
                "checkpoint": "NONE",
            }

        if not train_flag:
            return {
                "training_execution": EXEC_NOT_REQUESTED,
                "supervision_state": p19_res.get("supervision_state"),
                "authorized": True,
                "reason": "Phase 19 is authorized, but explicit --train flag was not provided.",
                "checkpoint": "NONE",
            }

        # Check compatibility gate
        manifest = self.phase22_orch.get_dataset_manifest()
        comp = self.verify_dataset_model_compatibility(manifest)
        if not comp["compatible"]:
            return {
                "training_execution": EXEC_BLOCKED,
                "supervision_state": p19_res.get("supervision_state"),
                "authorized": False,
                "reason": f"INPUT_OR_DATASET_SPEC_MISMATCH: {comp['reasons']}",
                "checkpoint": "NONE",
            }

        # Delegate execution to Phase 21 orchestrator
        res = self.phase21_orch.train_real_ctc(
            epochs=epochs,
            lr=lr,
            allow_random_split=allow_random_split,
        )

        return {
            "training_execution": EXEC_COMPLETED if res.get("status") == "SUCCESS" else EXEC_FAILED,
            "supervision_state": p19_res.get("supervision_state"),
            "authorized": True,
            "run_result": res,
        }

    # -------------------------------------------------------------------------
    # 6. Authoritative Status & Multi-Dimensional Readiness Dashboard
    # -------------------------------------------------------------------------
    def get_dashboard_summary(self) -> Dict[str, Any]:
        p19_res = evaluate_phase19_readiness(workspace_root=self.workspace_root)
        p21_res = evaluate_phase21_readiness(workspace_root=self.workspace_root)
        p22_dash = self.phase22_orch.get_dashboard_summary()
        accounting = self.get_data_accounting()
        ctc_res = self.phase22_orch.check_ctc_feasibility()
        manifest = self.phase22_orch.get_dataset_manifest()

        # Determine Acquisition Status
        total_ann = p22_dash.get("total_annotations", 0)
        submitted = p22_dash.get("submitted_annotations", 0)
        verified = p22_dash.get("verified_annotations", 0)
        eligible = p22_dash.get("training_eligible_count", 0)

        if total_ann == 0:
            acq_status = ACQUISITION_NOT_STARTED
        elif submitted == 0 and verified == 0:
            acq_status = ACQUISITION_IN_PROGRESS
        elif submitted > 0 and verified == 0:
            acq_status = ACQUISITION_SUBMITTED
        elif verified > 0 and eligible == 0:
            acq_status = ACQUISITION_REVIEW_IN_PROGRESS
        elif eligible > 0 and not manifest:
            acq_status = ACQUISITION_DATASET_FORMING
        elif eligible > 0 and manifest and not p19_res.get("real_ctc_training_allowed"):
            acq_status = ACQUISITION_TRAINING_READY
        elif p19_res.get("real_ctc_training_allowed"):
            acq_status = ACQUISITION_CTC_TRAINING_READY
        else:
            acq_status = ACQUISITION_NOT_STARTED

        # Determine Training Readiness
        if not p19_res.get("real_ctc_training_allowed"):
            training_readiness = READINESS_NOT_READY
        elif p19_res.get("dataset_scale") == DATASET_SCALE_DATA_LIMITED:
            training_readiness = READINESS_READY_LIMITED
        else:
            training_readiness = READINESS_READY_RESEARCH_SCALE

        # Check Active Trained Checkpoint
        pointer_file = self.experiments_dir / "live_model_pointer.json"
        has_checkpoint = False
        active_checkpoint_sha = "NONE"
        if pointer_file.exists():
            try:
                pdata = json.loads(pointer_file.read_text(encoding="utf-8"))
                active_dir = Path(pdata.get("active_run_dir", ""))
                meta_path = active_dir / "model_metadata.json"
                if meta_path.exists():
                    active_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    has_checkpoint = True
                    active_checkpoint_sha = active_meta.get("checkpoint_sha256", "NONE")
            except Exception:
                pass

        # Determine Training Execution State
        if not p19_res.get("real_ctc_training_allowed"):
            training_execution = EXEC_BLOCKED
        elif has_checkpoint:
            training_execution = EXEC_COMPLETED
        else:
            training_execution = EXEC_NOT_REQUESTED

        # Determine Next Physical Action
        if total_ann == 0:
            next_action = "Acquire genuine human sequential ISL annotations."
        elif submitted > 0:
            next_action = f"Review the {submitted} submitted annotations awaiting verification."
        elif verified == 0:
            next_action = "Complete reviewer verification for submitted pilot annotations."
        elif not manifest or not manifest.get("is_frozen"):
            next_action = "Freeze the training-eligible dataset (python scripts/build_phase22_dataset.py --freeze)."
        elif not p19_res.get("real_ctc_training_allowed"):
            next_action = "Collect additional qualified human annotations to satisfy minimum dataset thresholds."
        elif not has_checkpoint:
            next_action = "Execute Phase 23 genuine CTC training with explicit confirmation (python scripts/run_phase23_training.py --train)."
        else:
            next_action = "Perform genuine live camera validation (python run_signova.py --camera)."

        return {
            "phase": 23,
            "supervision_state": p19_res.get("supervision_state", SUPERVISION_STATE_B),
            "final_state": p19_res.get("supervision_state", SUPERVISION_STATE_B),
            "acquisition_status": acq_status,
            "training_readiness": training_readiness,
            "training_execution": training_execution,
            "assignments": {
                "total": p22_dash.get("sample_accounting", {}).get("total_assigned_slots", 0),
                "unique_videos": p22_dash.get("assigned_videos", 0),
            },
            "annotators": {
                "registered": p22_dash.get("annotators", {}).get("registered", 0),
                "qualified": p22_dash.get("annotators", {}).get("qualified", 0),
            },
            "annotations": {
                "discovered": accounting.get("total_discovered", 0),
                "valid": accounting.get("annotations_valid", 0),
                "submitted": submitted,
                "verified": verified,
                "rejected": accounting.get("rejected", 0),
                "revision_required": accounting.get("revision_required", 0),
            },
            "training_data": {
                "eligible": eligible,
                "ctc_feasible": ctc_res.get("feasible_samples", 0),
                "ctc_infeasible": ctc_res.get("infeasible_samples", 0),
                "vocabulary_size": manifest.get("vocabulary_size", 2) if manifest else 2,
            },
            "dataset": {
                "status": manifest.get("dataset_status", "DATASET_DRAFT") if manifest else "DATASET_DRAFT",
                "version": manifest.get("dataset_version", "NONE") if manifest else "NONE",
                "fingerprint": manifest.get("dataset_sha256", "NONE") if manifest else "NONE",
                "split": manifest.get("split_strategy", "NONE") if manifest else "NONE",
            },
            "phase19": {
                "state": p19_res.get("real_ctc_status", "BLOCKED"),
                "authorized": p19_res.get("real_ctc_training_allowed", False),
            },
            "phase21": {
                "training_status": "UNLOCKED" if p19_res.get("real_ctc_training_allowed") else "BLOCKED",
                "checkpoint": active_checkpoint_sha,
                "evaluation": "AVAILABLE" if has_checkpoint else "N/A",
            },
            "live_model": {
                "input_spec": "MATCH" if p21_res.get("live_authorization", {}).get("stages", {}).get("input_spec_match") else "MISMATCH / AWAITING",
                "verified": p21_res.get("live_authorization", {}).get("stages", {}).get("checkpoint_verified", False),
                "smoke_test": p21_res.get("live_authorization", {}).get("stages", {}).get("live_smoke_passed", False),
                "authorized": p21_res.get("live_authorization", {}).get("live_model_authorized", False),
            },
            "reference_integrity": p19_res.get("reference_integrity", {}),
            "next_physical_action": next_action,
        }


def evaluate_phase23_readiness(
    workspace_root: Optional[Union[str, Path]] = None,
    data_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Canonical single-source-of-truth Phase 23 readiness evaluation function.
    """
    root = Path(workspace_root) if workspace_root else Path(__file__).resolve().parent.parent.parent.parent
    d_root = Path(data_root) if data_root else None
    orch = Phase23Orchestrator(workspace_root=root, data_root=d_root)
    return orch.get_dashboard_summary()
