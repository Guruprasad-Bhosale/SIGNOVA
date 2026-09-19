"""
SIGNOVA Human Annotation Subsystem.
"""

from signova.annotation.annotator import AnnotatorProfile, AnnotatorRole
from signova.annotation.constants import (
    BLANK_ID,
    BLANK_TOKEN,
    EXACT_12_STATE_A_CONDITIONS,
    QUALITY_LINGUIST_REVIEWED,
    QUALITY_PARTIAL,
    QUALITY_UNVERIFIED,
    QUALITY_VERIFIED,
    QUALITY_WEAK,
    REVIEW_STATE_ANNOTATED,
    REVIEW_STATE_ANNOTATION_IN_PROGRESS,
    REVIEW_STATE_DISAGREEMENT,
    REVIEW_STATE_REJECTED,
    REVIEW_STATE_REVIEW_PENDING,
    REVIEW_STATE_SECOND_ANNOTATION_PENDING,
    REVIEW_STATE_UNANNOTATED,
    REVIEW_STATE_VERIFIED,
    STATUS_BLOCKED_HUMAN_RESOURCE,
    STATUS_OPERATIONAL,
    SUPERVISION_STATE_A,
    SUPERVISION_STATE_B,
    SUPERVISION_STATE_C,
    UNK_ID,
    UNK_TOKEN,
)
from signova.annotation.export import export_annotation_to_eaf, export_annotation_to_json, import_annotation_from_json
from signova.annotation.quality import calculate_annotation_quality_grade
from signova.annotation.review import AnnotationStateMachine, AnnotationWorkflowError
from signova.annotation.schema import AnnotationSet, GlossToken, TemporalSegment, VideoAnnotation
from signova.annotation.validation import evaluate_training_eligibility
from signova.annotation.vocabulary import ProjectGlossVocabulary

__all__ = [
    "SUPERVISION_STATE_C",
    "SUPERVISION_STATE_B",
    "SUPERVISION_STATE_A",
    "EXACT_12_STATE_A_CONDITIONS",
    "BLANK_TOKEN",
    "BLANK_ID",
    "UNK_TOKEN",
    "UNK_ID",
    "GlossToken",
    "TemporalSegment",
    "VideoAnnotation",
    "AnnotationSet",
    "AnnotatorProfile",
    "AnnotatorRole",
    "AnnotationStateMachine",
    "AnnotationWorkflowError",
    "calculate_annotation_quality_grade",
    "ProjectGlossVocabulary",
    "evaluate_training_eligibility",
    "export_annotation_to_json",
    "import_annotation_from_json",
    "export_annotation_to_eaf",
]
