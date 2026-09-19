"""
Phase 16 Unified Orchestrator for SIGNOVA.

Coordinates:
- Human data discovery & classification (UNVERIFIED, VALID, INVALID, REQUIRES_REVIEW, TRAINING_ELIGIBLE)
- Detailed sample accounting (discovered, genuine, verified, rejected, training_eligible, ctc_infeasible, leakage_excluded)
- Distinctions between HUMAN_DATA_PRESENT, HUMAN_DATA_QUALIFIED, and TRAINING_ELIGIBLE_DATA
- Canonical 12-condition supervision gate
- Repeated-token CTC feasibility (T_required = L + sum I(y_i == y_{i+1}))
- Preferred split hierarchy selection with explicit rationale
- Dynamic readiness summary generation and prohibited automation audit
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from signova.annotation.constants import REVIEW_STATE_VERIFIED, REVIEW_STATE_REJECTED
from signova.annotation.schema import VideoAnnotation
from signova.operations.annotator_qualification import validate_annotator_qualification
from signova.operations.audit_automation import audit_prohibited_automation_pathways
from signova.operations.double_annotation import DoubleAnnotationManager
from signova.pilot.constants import (
    CLAIMS_LIMITED,
    CLAIMS_NOT_READY,
    CLAIMS_PERMITTED_BY_EVIDENCE,
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
    STATUS_ALLOWED,
    STATUS_BLOCKED,
    STATUS_BLOCKED_HUMAN_RESOURCE,
    SUPERVISION_STATE_A,
    SUPERVISION_STATE_A_DATA_LIMITED,
    SUPERVISION_STATE_B,
)
from signova.qualification.feasibility import validate_ctc_feasibility
from signova.qualification.gate import Phase12SupervisionGate
from signova.qualification.ingestion import AnnotationIngestionEngine
from signova.qualification.leakage import audit_phase12_leakage
from signova.qualification.vocabulary import Phase12GlossVocabulary


class Phase16Orchestrator:
    """Orchestrates Phase 16 human dataset formation, qualification, and dynamic supervision gating."""

    def __init__(
        self,
        annotations_dir: Optional[Path] = None,
        features_dir: Optional[Path] = None,
    ):
        self.annotations_dir = annotations_dir or Path("data/annotations/phase11/human_gold")
        self.features_dir = features_dir or Path("data/features/landmarks")

    def run_full_qualification(self) -> Dict[str, Any]:
        # 1. Ingestion and discovery
        engine = AnnotationIngestionEngine(landmarks_dir=self.features_dir)
        ingest_res = engine.batch_ingest(self.annotations_dir)
        valid_annots: List[VideoAnnotation] = ingest_res["valid_annotations"]
        rejected_annots: List[VideoAnnotation] = ingest_res["rejected_annotations"]
        total_discovered = ingest_res["total_files"]

        # 2. Vocabulary
        vocab = Phase12GlossVocabulary.build_from_annotations(valid_annots)

        # 3. Leakage audit
        leakage_res = audit_phase12_leakage(valid_annots)

        # 4. CTC Feasibility (repeated token blanks)
        feasibility_res = validate_ctc_feasibility(valid_annots, features_dir=self.features_dir)

        # 5. Automation audit against prohibited pathways
        auto_audit = audit_prohibited_automation_pathways(self.annotations_dir)

        # 6. Double Annotation & Agreement
        double_mgr = DoubleAnnotationManager()
        ann_by_sample: Dict[str, List[VideoAnnotation]] = {}
        for a in valid_annots:
            ann_by_sample.setdefault(a.sample_id, []).append(a)
        agreement_res = double_mgr.evaluate_independent_agreement(ann_by_sample)

        # 7. Preferred Split Strategy with explicit rationale
        split_strategy, split_rationale = self._select_split_strategy(valid_annots)

        # 8. Scale Classification
        scale = self._classify_scale(len(valid_annots), vocab.size)

        # 9. Gate Evaluation
        gate_res = Phase12SupervisionGate.evaluate(annotations_dir=self.annotations_dir)

        # 10. Sample accounting
        verified_count = sum(1 for a in valid_annots if a.review_status == REVIEW_STATE_VERIFIED)
        rejected_count = len(rejected_annots) + sum(1 for a in valid_annots if a.review_status == REVIEW_STATE_REJECTED)
        ctc_infeasible_count = feasibility_res.rejected_samples
        leakage_excluded_count = len(leakage_res.duplicate_checksums)

        # Training eligible: verified, CTC feasible, and leakage clean
        training_eligible_count = max(0, verified_count - ctc_infeasible_count - leakage_excluded_count)

        human_data_present = total_discovered > 0
        human_data_qualified = len(valid_annots) > 0 and (auto_audit["audit_status"] == "PASSED")
        training_eligible_data = training_eligible_count > 0 and gate_res.real_ctc_status == STATUS_ALLOWED

        # Dynamic claims
        if gate_res.supervision_state == SUPERVISION_STATE_A:
            claims = CLAIMS_PERMITTED_BY_EVIDENCE
            pub_grade = "READY"
        elif gate_res.supervision_state == SUPERVISION_STATE_A_DATA_LIMITED:
            claims = CLAIMS_LIMITED
            pub_grade = "NOT_READY"
        else:
            claims = CLAIMS_NOT_READY
            pub_grade = "NOT_READY"

        sample_accounting = {
            "total_discovered": total_discovered,
            "genuine_annotations": len(valid_annots),
            "verified": verified_count,
            "rejected": rejected_count,
            "training_eligible": training_eligible_count,
            "ctc_infeasible": ctc_infeasible_count,
            "excluded_due_to_leakage": leakage_excluded_count,
        }

        return {
            "phase": 16,
            "sample_accounting": sample_accounting,
            "human_data_present": human_data_present,
            "human_data_qualified": human_data_qualified,
            "training_eligible_data": training_eligible_data,
            "total_annotations_found": total_discovered,
            "valid_training_eligible": training_eligible_count,
            "rejected_annotations": rejected_count,
            "vocabulary_size": vocab.size,
            "dataset_scale": scale,
            "split_strategy": split_strategy,
            "split_rationale": split_rationale,
            "leakage_status": leakage_res.audit_status,
            "ctc_feasibility_status": feasibility_res.feasibility_status,
            "automation_audit_status": auto_audit["audit_status"],
            "agreement_summary": agreement_res,
            "supervision_state": gate_res.supervision_state,
            "real_ctc_status": gate_res.real_ctc_status,
            "pilot_status": gate_res.pilot_status,
            "generalization_claims": claims,
            "publication_grade_evaluation": pub_grade,
            "conditions_satisfied": sum(1 for v in gate_res.conditions_satisfied.values() if v),
            "conditions_required": len(gate_res.conditions_satisfied),
            "failed_conditions": gate_res.failed_conditions,
        }

    def _select_split_strategy(self, annotations: List[VideoAnnotation]) -> Tuple[str, str]:
        if not annotations:
            return "NONE", "no_annotations_available"

        signers = {a.metadata.get("signer_id", "UNKNOWN") for a in annotations if a.metadata}
        sessions = {a.metadata.get("session_id", "UNKNOWN") for a in annotations if a.metadata}
        signers.discard("UNKNOWN")
        sessions.discard("UNKNOWN")

        if len(signers) >= 2:
            return SPLIT_STRATEGY_SIGNER_INDEPENDENT, "verified_signer_metadata_available"
        elif len(sessions) >= 2:
            return SPLIT_STRATEGY_SESSION_INDEPENDENT, "signer_metadata_unavailable_session_metadata_available"
        else:
            return SPLIT_STRATEGY_RANDOM, "signer_and_session_metadata_unavailable"

    def _classify_scale(self, total_samples: int, unique_glosses: int) -> str:
        if total_samples == 0:
            return DATASET_SCALE_NO_DATA
        elif total_samples < 5 or unique_glosses < 3:
            return DATASET_SCALE_PILOT_ONLY
        elif total_samples < 50 or unique_glosses < 20:
            return DATASET_SCALE_DATA_LIMITED
        else:
            return DATASET_SCALE_RESEARCH_SCALE
