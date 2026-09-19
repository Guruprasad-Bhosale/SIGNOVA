"""
Phase 14-17 Annotation Operations Package for SIGNOVA.
"""

from signova.operations.constants import (
    AGREEMENT_COMPUTABLE,
    AGREEMENT_INSUFFICIENT_SAMPLE,
    AGREEMENT_NOT_COMPUTABLE,
    ALL_AGREEMENT_STATUSES,
    ARTIFACT_STATUS_BLOCKED,
    ARTIFACT_STATUS_CREATED,
    ARTIFACT_STATUS_PREEXISTING,
    DEFAULT_DOUBLE_ANNOTATION_FRACTION,
    DEFAULT_DOUBLE_ANNOTATION_MINIMUM,
    SPLIT_STRATEGY_RANDOM,
    SPLIT_STRATEGY_SESSION_INDEPENDENT,
    SPLIT_STRATEGY_SIGNER_INDEPENDENT,
    SPLIT_STRATEGY_SOURCE_GROUP_INDEPENDENT,
)
from signova.operations.annotator_qualification import (
    AnnotatorQualificationProfile,
    QUALIFICATION_DOCUMENTED,
    QUALIFICATION_NOT_DOCUMENTED,
    REQUIRES_ISL_EXPERT_REVIEW,
    validate_annotator_qualification,
)
from signova.operations.audit_automation import audit_prohibited_automation_pathways
from signova.operations.checkpoint_provenance import (
    CHECKPOINT_TYPE_REAL,
    CHECKPOINT_TYPE_SYNTHETIC,
    CheckpointProvenanceMetadata,
    create_real_checkpoint_provenance,
)
from signova.operations.double_annotation import DoubleAnnotationManager, IndependentAnnotationPair
from signova.operations.phase15_orchestrator import Phase15Orchestrator
from signova.operations.phase16_orchestrator import Phase16Orchestrator
from signova.operations.phase17_orchestrator import (
    Phase17Orchestrator,
    PILOT_STATUS_NOT_STARTED,
    PILOT_STATUS_BLOCKED_HUMAN_RESOURCE,
    PILOT_STATUS_IN_PROGRESS,
    PILOT_STATUS_COMPLETED,
    PILOT_STATUS_COMPLETED_INSUFFICIENT_DATA,
)
from signova.operations.assignment import Phase19PilotAssignmentManager
from signova.operations.phase18_orchestrator import (
    Phase18Orchestrator,
    evaluate_phase18_readiness,
)
from signova.operations.phase19_orchestrator import (
    Phase19Orchestrator,
    evaluate_phase19_readiness,
)
from signova.operations.pilot import Phase14PilotManifestGenerator
from signova.operations.revision import AnnotationRevision, AnnotationRevisionHistory

__all__ = [
    "AGREEMENT_COMPUTABLE",
    "AGREEMENT_INSUFFICIENT_SAMPLE",
    "AGREEMENT_NOT_COMPUTABLE",
    "ALL_AGREEMENT_STATUSES",
    "ARTIFACT_STATUS_BLOCKED",
    "ARTIFACT_STATUS_CREATED",
    "ARTIFACT_STATUS_PREEXISTING",
    "DEFAULT_DOUBLE_ANNOTATION_FRACTION",
    "DEFAULT_DOUBLE_ANNOTATION_MINIMUM",
    "SPLIT_STRATEGY_RANDOM",
    "SPLIT_STRATEGY_SESSION_INDEPENDENT",
    "SPLIT_STRATEGY_SIGNER_INDEPENDENT",
    "SPLIT_STRATEGY_SOURCE_GROUP_INDEPENDENT",
    "DoubleAnnotationManager",
    "IndependentAnnotationPair",
    "Phase14PilotManifestGenerator",
    "AnnotationRevision",
    "AnnotationRevisionHistory",
    "AnnotatorQualificationProfile",
    "QUALIFICATION_DOCUMENTED",
    "QUALIFICATION_NOT_DOCUMENTED",
    "REQUIRES_ISL_EXPERT_REVIEW",
    "validate_annotator_qualification",
    "audit_prohibited_automation_pathways",
    "CHECKPOINT_TYPE_REAL",
    "CHECKPOINT_TYPE_SYNTHETIC",
    "CheckpointProvenanceMetadata",
    "create_real_checkpoint_provenance",
    "Phase15Orchestrator",
    "Phase16Orchestrator",
    "Phase17Orchestrator",
    "Phase18Orchestrator",
    "evaluate_phase18_readiness",
    "Phase19Orchestrator",
    "evaluate_phase19_readiness",
    "Phase19PilotAssignmentManager",
    "PILOT_STATUS_NOT_STARTED",
    "PILOT_STATUS_BLOCKED_HUMAN_RESOURCE",
    "PILOT_STATUS_IN_PROGRESS",
    "PILOT_STATUS_COMPLETED",
    "PILOT_STATUS_COMPLETED_INSUFFICIENT_DATA",
]
