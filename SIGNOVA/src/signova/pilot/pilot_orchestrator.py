"""
Phase 13 Pilot Orchestrator for SIGNOVA.

Coordinates:
- Human annotation pilot ingestion
- Dataset qualification
- Inter-annotator agreement evaluation
- Leakage auditing & Split strategy selection
- Repeated-token CTC feasibility
- Conditional real CTC training
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from signova.annotation.schema import VideoAnnotation
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
from signova.qualification.gate import GateEvaluationResult, Phase12SupervisionGate, RealCTCTrainingBlockedError
from signova.qualification.ingestion import AnnotationIngestionEngine
from signova.qualification.leakage import audit_phase12_leakage
from signova.qualification.vocabulary import Phase12GlossVocabulary


class Phase13PilotOrchestrator:
    """End-to-end pilot orchestrator for Phase 13."""

    def __init__(
        self,
        annotations_dir: Optional[Path] = None,
        features_dir: Optional[Path] = None,
    ):
        self.annotations_dir = annotations_dir or Path("data/annotations/phase11/human_gold")
        self.features_dir = features_dir or Path("data/features/landmarks")

    def classify_dataset_scale(self, total_samples: int, unique_glosses: int) -> str:
        """Classifies dataset scale based on sample count and vocabulary."""
        if total_samples == 0:
            return DATASET_SCALE_NO_DATA
        elif total_samples < 5 or unique_glosses < 3:
            return DATASET_SCALE_PILOT_ONLY
        elif total_samples < 50 or unique_glosses < 20:
            return DATASET_SCALE_DATA_LIMITED
        else:
            return DATASET_SCALE_RESEARCH_SCALE

    def determine_split_strategy(self, annotations: List[VideoAnnotation]) -> Tuple[str, str]:
        """Determines best supported split strategy based on metadata."""
        if not annotations:
            return "NONE", "NO_ANNOTATIONS"

        signers = {a.metadata.get("signer_id", "UNKNOWN") for a in annotations if a.metadata}
        sessions = {a.metadata.get("session_id", "UNKNOWN") for a in annotations if a.metadata}

        signers.discard("UNKNOWN")
        sessions.discard("UNKNOWN")

        if len(signers) >= 2:
            return SPLIT_STRATEGY_SIGNER_INDEPENDENT, "Signer IDs available for independent partition"
        elif len(sessions) >= 2:
            return SPLIT_STRATEGY_SESSION_INDEPENDENT, "Session IDs available for independent partition"
        else:
            return SPLIT_STRATEGY_RANDOM, "Signer and session identities unknown; signer independence NOT supported"

    def run_qualification_pipeline(self) -> Dict[str, Any]:
        """Runs complete qualification pipeline on genuine annotations."""
        ingestion_engine = AnnotationIngestionEngine(landmarks_dir=self.features_dir)
        ingest_res = ingestion_engine.batch_ingest(self.annotations_dir)
        valid_annots = ingest_res["valid_annotations"]

        # Build vocabulary
        vocab = Phase12GlossVocabulary.build_from_annotations(valid_annots)

        # Audit leakage
        leakage_res = audit_phase12_leakage(valid_annots)

        # CTC feasibility with repeated token blank separation
        feasibility_res = validate_ctc_feasibility(valid_annots, features_dir=self.features_dir)

        # Split strategy
        split_strat, split_limitation = self.determine_split_strategy(valid_annots)

        # Scale classification
        scale = self.classify_dataset_scale(len(valid_annots), vocab.size)

        # Gate evaluation
        gate_res = Phase12SupervisionGate.evaluate(annotations_dir=self.annotations_dir)

        return {
            "total_files": ingest_res["total_files"],
            "valid_annotations_count": len(valid_annots),
            "rejected_annotations_count": len(ingest_res["rejected_annotations"]),
            "vocabulary_size": vocab.size,
            "dataset_scale": scale,
            "split_strategy": split_strat,
            "split_limitation": split_limitation,
            "leakage_status": leakage_res.audit_status,
            "ctc_feasibility_status": feasibility_res.feasibility_status,
            "supervision_state": gate_res.supervision_state,
            "real_ctc_status": gate_res.real_ctc_status,
            "pilot_status": gate_res.pilot_status,
            "generalization_claims": (
                CLAIMS_PERMITTED_BY_EVIDENCE if gate_res.supervision_state == SUPERVISION_STATE_A
                else CLAIMS_LIMITED if gate_res.supervision_state == SUPERVISION_STATE_A_DATA_LIMITED
                else CLAIMS_NOT_READY
            ),
        }
