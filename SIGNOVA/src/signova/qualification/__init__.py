"""
Phase 12 Dataset Qualification and Supervision Gate Package for SIGNOVA.
"""

from signova.qualification.constants import (
    BLANK_ID,
    BLANK_TOKEN,
    EXACT_12_STATE_A_CONDITIONS,
    RECOMMENDED_DATASET_THRESHOLDS,
    STATUS_ALLOWED,
    STATUS_BLOCKED,
    STATUS_BLOCKED_HUMAN_RESOURCE,
    STATUS_LIMITED,
    SUPERVISION_STATE_A,
    SUPERVISION_STATE_A_DATA_LIMITED,
    SUPERVISION_STATE_B,
    SUPERVISION_STATE_C,
    TECHNICAL_MINIMUM_THRESHOLDS,
    UNK_ID,
    UNK_TOKEN,
)
from signova.qualification.gate import (
    GateEvaluationResult,
    Phase12SupervisionGate,
    RealCTCTrainingBlockedError,
)
from signova.qualification.ingestion import AnnotationIngestionEngine
from signova.qualification.manifest import generate_phase12_manifest, read_phase12_manifest
from signova.qualification.leakage import audit_phase12_leakage, LeakageAuditReport
from signova.qualification.feasibility import validate_ctc_feasibility, CTCFeasibilityReport
from signova.qualification.vocabulary import Phase12GlossVocabulary

__all__ = [
    "BLANK_ID",
    "BLANK_TOKEN",
    "EXACT_12_STATE_A_CONDITIONS",
    "RECOMMENDED_DATASET_THRESHOLDS",
    "STATUS_ALLOWED",
    "STATUS_BLOCKED",
    "STATUS_BLOCKED_HUMAN_RESOURCE",
    "STATUS_LIMITED",
    "SUPERVISION_STATE_A",
    "SUPERVISION_STATE_A_DATA_LIMITED",
    "SUPERVISION_STATE_B",
    "SUPERVISION_STATE_C",
    "TECHNICAL_MINIMUM_THRESHOLDS",
    "UNK_ID",
    "UNK_TOKEN",
    "GateEvaluationResult",
    "Phase12SupervisionGate",
    "RealCTCTrainingBlockedError",
    "AnnotationIngestionEngine",
    "generate_phase12_manifest",
    "read_phase12_manifest",
    "audit_phase12_leakage",
    "LeakageAuditReport",
    "validate_ctc_feasibility",
    "CTCFeasibilityReport",
    "Phase12GlossVocabulary",
]
