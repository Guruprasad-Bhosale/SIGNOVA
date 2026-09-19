"""
Phase 9 Real CTC Training Gate Module for SIGNOVA.

Enforces:
Real CTC training is permitted ONLY when:
1. data_gate_state == "STATE A"
2. verified_sequential_glosses == True
3. verified_video_pairing == True
4. valid_license == True
5. leakage_audit_passed == True
6. pilot_validation_passed == True

Under STATE C / STATE B, real CTC training is strictly blocked.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class RealCTCTrainingBlockedError(RuntimeError):
    """Raised when real CTC training is attempted without passing all Phase 9 verification gates."""
    pass


@dataclass
class Phase9GateStatus:
    data_gate_state: str  # "STATE A", "STATE B", "STATE C"
    verified_sequential_glosses: bool
    verified_video_pairing: bool
    valid_license: bool
    leakage_audit_passed: bool
    pilot_validation_passed: bool
    budget_spend_inr: float = 0.0

    @property
    def is_training_permitted(self) -> bool:
        return (
            self.data_gate_state == "STATE A"
            and self.verified_sequential_glosses
            and self.verified_video_pairing
            and self.valid_license
            and self.leakage_audit_passed
            and self.pilot_validation_passed
        )

    def get_blocker_reasons(self) -> List[str]:
        reasons = []
        if self.data_gate_state != "STATE A":
            reasons.append(f"Data gate state is '{self.data_gate_state}', required 'STATE A'.")
        if not self.verified_sequential_glosses:
            reasons.append("Ordered sequential ISL gloss annotations are unverified / missing.")
        if not self.verified_video_pairing:
            reasons.append("Video-to-annotation synchronization is unverified.")
        if not self.valid_license:
            reasons.append("License does not permit training / research usage.")
        if not self.leakage_audit_passed:
            reasons.append("Signer or session split leakage audit has not passed.")
        if not self.pilot_validation_passed:
            reasons.append("Controlled real-dataset pilot validation has not been completed.")
        return reasons


class RealCTCTrainingGate:
    """
    Programmatic hard gate for SIGNOVA real CTC model training.
    """

    def __init__(self, gate_status: Optional[Phase9GateStatus] = None):
        self.status = gate_status or Phase9GateStatus(
            data_gate_state="STATE C",
            verified_sequential_glosses=False,
            verified_video_pairing=False,
            valid_license=False,
            leakage_audit_passed=False,
            pilot_validation_passed=False,
            budget_spend_inr=0.0,
        )

    def can_train(self) -> bool:
        return self.status.is_training_permitted

    def enforce_gate(self) -> None:
        """Raises RealCTCTrainingBlockedError if any gate condition is unsatisfied."""
        if not self.can_train():
            reasons = "\n  - ".join(self.status.get_blocker_reasons())
            raise RealCTCTrainingBlockedError(
                f"[REAL CTC TRAINING BLOCKED]\n"
                f"Real CTC training is forbidden until all Phase 9 verification criteria are met:\n"
                f"  - {reasons}"
            )
