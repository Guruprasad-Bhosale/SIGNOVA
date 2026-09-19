"""
Phase 12 Constants, Gate Definitions, and Configurable Research Thresholds for SIGNOVA.
"""

from typing import Dict, List, Set

# Supervision State Machine
SUPERVISION_STATE_C = "STATE_C"  # No sequential supervision available
SUPERVISION_STATE_B = "STATE_B"  # Annotation platform operational; human data collection not started / pending
SUPERVISION_STATE_A_DATA_LIMITED = "STATE_A_DATA_LIMITED"  # Genuine supervision verified, but below full research thresholds (exploratory baseline allowed, claims restricted)
SUPERVISION_STATE_A = "STATE_A"  # Genuine sequential ISL dataset available meeting all research thresholds

# Operational Status Codes
STATUS_OPERATIONAL = "OPERATIONAL"
STATUS_NOT_STARTED = "NOT_STARTED"
STATUS_BLOCKED_HUMAN_RESOURCE = "BLOCKED_HUMAN_RESOURCE"
STATUS_BLOCKED = "BLOCKED"
STATUS_ALLOWED = "ALLOWED"
STATUS_LIMITED = "LIMITED"
STATUS_NOT_READY = "NOT_READY"
STATUS_READY = "READY"

# Configurable Minimum Dataset Research Thresholds
RECOMMENDED_DATASET_THRESHOLDS: Dict[str, int] = {
    "MIN_TRAIN_SAMPLES": 50,
    "MIN_VALIDATION_SAMPLES": 10,
    "MIN_TEST_SAMPLES": 10,
    "MIN_UNIQUE_GLOSSES": 20,
    "MIN_ANNOTATED_SIGNERS": 3,
    "MIN_ANNOTATED_SESSIONS": 3,
    "MIN_TRAIN_GLOSS_FREQUENCY": 2,
}

# Technical Minimum Feasibility Thresholds (permits STATE_A_DATA_LIMITED for pipeline validation)
TECHNICAL_MINIMUM_THRESHOLDS: Dict[str, int] = {
    "MIN_TRAIN_SAMPLES": 3,
    "MIN_VALIDATION_SAMPLES": 1,
    "MIN_TEST_SAMPLES": 1,
    "MIN_UNIQUE_GLOSSES": 3,
    "MIN_ANNOTATED_SIGNERS": 1,
    "MIN_ANNOTATED_SESSIONS": 1,
    "MIN_TRAIN_GLOSS_FREQUENCY": 1,
}

# Reserved CTC / Tokenizer Vocabulary Tokens
BLANK_TOKEN = "<BLANK>"
BLANK_ID = 0
UNK_TOKEN = "<UNK>"
UNK_ID = 1

# Inherited 12 Authoritative STATE_A Conditions
EXACT_12_STATE_A_CONDITIONS: List[str] = [
    "GENUINE_SEQUENTIAL_ANNOTATIONS_EXIST",
    "VIDEO_ANNOTATION_PAIRING_VERIFIED",
    "ANNOTATION_SCHEMA_VALIDATION_PASSED",
    "VOCABULARY_DERIVED_FROM_GENUINE_ANNOTATIONS",
    "ANNOTATION_QUALITY_MEETS_TRAINING_THRESHOLD",
    "REQUIRED_HUMAN_REVIEW_COMPLETE",
    "NO_CRITICAL_LEAKAGE_DETECTED",
    "DATASET_PROVENANCE_RECORDED",
    "LICENSING_ACCESS_STATUS_RECORDED",
    "TRAIN_VAL_TEST_SPLIT_VALID",
    "SIGNER_SESSION_INDEPENDENCE_CHARACTERIZED",
    "MINIMUM_SAMPLE_THRESHOLD_SATISFIED",
]
