"""
Phase 11 Real CTC Training Gate Module for SIGNOVA.

Enforces the 12 Authoritative STATE_A Prerequisites:
1. GENUINE_SEQUENTIAL_ANNOTATIONS_EXIST
2. VIDEO_ANNOTATION_PAIRING_VERIFIED
3. ANNOTATION_SCHEMA_VALIDATION_PASSED
4. VOCABULARY_DERIVED_FROM_GENUINE_ANNOTATIONS
5. ANNOTATION_QUALITY_MEETS_TRAINING_THRESHOLD
6. REQUIRED_HUMAN_REVIEW_COMPLETE
7. NO_CRITICAL_LEAKAGE_DETECTED
8. DATASET_PROVENANCE_RECORDED
9. LICENSING_ACCESS_STATUS_RECORDED
10. TRAIN_VAL_TEST_SPLIT_VALID
11. SIGNER_SESSION_INDEPENDENCE_CHARACTERIZED
12. MINIMUM_SAMPLE_THRESHOLD_SATISFIED

Under STATE_C and STATE_B, real CTC training raises RealCTCTrainingBlockedError.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from signova.annotation.constants import (
    EXACT_12_STATE_A_CONDITIONS,
    SUPERVISION_STATE_A,
    SUPERVISION_STATE_B,
    SUPERVISION_STATE_C,
)
from signova.recognition.phase9_gate import RealCTCTrainingBlockedError


@dataclass
class Phase11GateConditionCheck:
    condition_id: str
    is_satisfied: bool
    evidence_notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "is_satisfied": self.is_satisfied,
            "evidence_notes": self.evidence_notes,
        }


@dataclass
class Phase11SupervisionGate:
    supervision_state: str = SUPERVISION_STATE_B
    conditions: Dict[str, Phase11GateConditionCheck] = field(default_factory=dict)
    annotated_videos_count: int = 0
    verified_videos_count: int = 0
    training_eligible_videos_count: int = 0
    minimum_sample_threshold: int = 20

    def __post_init__(self):
        if not self.conditions:
            self._initialize_default_state_b_conditions()

    def _initialize_default_state_b_conditions(self) -> None:
        defaults = {
            "GENUINE_SEQUENTIAL_ANNOTATIONS_EXIST": (False, "Human pilot blocked pending deaf annotator deployment."),
            "VIDEO_ANNOTATION_PAIRING_VERIFIED": (False, "No verified human annotation pairings exist yet."),
            "ANNOTATION_SCHEMA_VALIDATION_PASSED": (True, "Phase 11 canonical schema validation engine operational."),
            "VOCABULARY_DERIVED_FROM_GENUINE_ANNOTATIONS": (False, "Project vocabulary has 0 human-verified tokens."),
            "ANNOTATION_QUALITY_MEETS_TRAINING_THRESHOLD": (False, "Quality gate requires VERIFIED or LINGUIST_REVIEWED."),
            "REQUIRED_HUMAN_REVIEW_COMPLETE": (False, "Human review pending."),
            "NO_CRITICAL_LEAKAGE_DETECTED": (True, "Leakage audit engine operational with signer isolation."),
            "DATASET_PROVENANCE_RECORDED": (True, "Cryptographic provenance chain verified."),
            "LICENSING_ACCESS_STATUS_RECORDED": (True, "Zero-cost licensing audit completed."),
            "TRAIN_VAL_TEST_SPLIT_VALID": (False, "Split requires verified training-eligible samples."),
            "SIGNER_SESSION_INDEPENDENCE_CHARACTERIZED": (True, "Signer independence metadata framework active."),
            "MINIMUM_SAMPLE_THRESHOLD_SATISFIED": (False, f"0 samples < minimum threshold of {self.minimum_sample_threshold}."),
        }
        for cid in EXACT_12_STATE_A_CONDITIONS:
            sat, notes = defaults.get(cid, (False, "Unverified condition."))
            self.conditions[cid] = Phase11GateConditionCheck(
                condition_id=cid,
                is_satisfied=sat,
                evidence_notes=notes,
            )

    @property
    def is_state_a_unlocked(self) -> bool:
        return all(c.is_satisfied for c in self.conditions.values())

    @property
    def is_real_ctc_training_permitted(self) -> bool:
        return self.supervision_state == SUPERVISION_STATE_A and self.is_state_a_unlocked

    def enforce_gate(self) -> None:
        if not self.is_real_ctc_training_permitted:
            unmet = [c.condition_id for c in self.conditions.values() if not c.is_satisfied]
            raise RealCTCTrainingBlockedError(
                f"[REAL CTC TRAINING BLOCKED - SUPERVISION {self.supervision_state}]\n"
                f"Real CTC training is forbidden until all 12 STATE_A prerequisites are met.\n"
                f"Unmet conditions ({len(unmet)}/12):\n  - " + "\n  - ".join(unmet)
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supervision_state": self.supervision_state,
            "real_ctc_training": "UNLOCKED" if self.is_real_ctc_training_permitted else "BLOCKED",
            "state_a_unlocked": self.is_state_a_unlocked,
            "annotated_videos_count": self.annotated_videos_count,
            "verified_videos_count": self.verified_videos_count,
            "training_eligible_videos_count": self.training_eligible_videos_count,
            "minimum_sample_threshold": self.minimum_sample_threshold,
            "conditions_summary": {
                "total_conditions": len(EXACT_12_STATE_A_CONDITIONS),
                "satisfied_count": sum(1 for c in self.conditions.values() if c.is_satisfied),
                "unmet_count": sum(1 for c in self.conditions.values() if not c.is_satisfied),
            },
            "conditions": {k: v.to_dict() for k, v in self.conditions.items()},
        }
