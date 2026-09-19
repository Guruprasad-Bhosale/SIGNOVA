"""
Lightweight Diagnostic Check CLI for SIGNOVA Phase 19.

Safely inspects the workspace and dynamically reports:
- Current supervision state (STATE_B, STATE_A_DATA_LIMITED, STATE_A)
- Human-data state (PRESENT, AUTHENTICATED, QUALIFIED, TRAINING_ELIGIBLE)
- Pilot status (NOT_STARTED, IN_PROGRESS, BLOCKED_HUMAN_RESOURCE, COMPLETED, COMPLETED_INSUFFICIENT_DATA)
- Annotation activity tracking (ANNOTATION_ACTIVITY_STARTED)
- 7-tier explicit sample accounting
- Dataset-level CTC feasibility & feasibility rate
- Selected split strategy, fallback reason & leakage risk
- Real CTC training authorization status with explicit reason
- Checkpoint existence / invariant verification
- Protected reference repository integrity (44/44)
- Current blockers and required next actions

This script NEVER triggers training and NEVER produces model checkpoints.
"""

from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase19_orchestrator import evaluate_phase19_readiness


def main():
    print("===========================================================")
    print(" SIGNOVA PHASE 19 — LIGHTWEIGHT DIAGNOSTIC CHECK")
    print("===========================================================")

    readiness = evaluate_phase19_readiness(workspace_root=WORKSPACE_ROOT)
    accounting = readiness["sample_accounting"]
    feasibility = readiness["ctc_feasibility"]
    ref_integrity = readiness["reference_integrity"]
    auth = readiness["training_authorization"]

    print("\n[1] Supervision State:")
    print(f"  State: {readiness['supervision_state']}")
    print(f"  Real CTC Status: {readiness['real_ctc_status']}")
    print(f"  Conditions Satisfied: {readiness['conditions_satisfied']} / {readiness['conditions_required']}")

    print("\n[2] Human Data State:")
    print(f"  Present: {readiness['human_data_present']}")
    print(f"  Authenticated: {readiness['human_data_authenticated']}")
    print(f"  Qualified: {readiness['human_data_qualified']}")
    print(f"  Training Eligible: {readiness['training_eligible_data']}")
    print(f"  Vocabulary Size: {readiness['vocabulary_size']} tokens")

    print("\n[3] Pilot & Sample Accounting:")
    print(f"  Pilot Status: {readiness['pilot_status']}")
    print(f"  Annotation Activity Started: {readiness['annotation_activity_started']}")
    print(f"  Dataset Scale: {readiness['dataset_scale']}")
    print(f"  Total Annotations: {accounting['total_annotations']}")
    print(f"  Authenticated Annotations: {accounting['authenticated_annotations']}")
    print(f"  Verified Annotations: {accounting['verified_annotations']}")
    print(f"  Pending Review Samples: {accounting['pending_review_samples']}")
    print(f"  Rejected Samples: {accounting['rejected_samples']}")
    print(f"  Training Eligible Samples: {accounting['training_eligible_samples']}")

    print("\n[4] Dataset-Level CTC Feasibility:")
    print(f"  Feasibility Status: {feasibility['feasibility_status']}")
    print(f"  Feasibility Rate: {feasibility['feasibility_rate'] * 100:.1f}% ({feasibility['feasible_sequences']} / {feasibility['total_sequences']})")
    print(f"  Infeasible Sequences: {feasibility['infeasible_sequences']}")
    if feasibility["max_required_frames"] is not None:
        print(f"  Max Required Frames: {feasibility['max_required_frames']}")

    print("\n[5] Split Strategy & Leakage Audit:")
    print(f"  Selected Split: {readiness['split_strategy']}")
    print(f"  Fallback Reason: {readiness['split_rationale']}")
    if readiness.get("split_warning"):
        print(f"  Split Warning: {readiness['split_warning']}")
    print(f"  Identity Metadata Available: {readiness['identity_metadata_available']}")
    print(f"  Leakage Risk: {readiness['leakage_risk']}")
    print(f"  Leakage Status: {readiness['leakage_status']}")

    print("\n[6] Real CTC Training Authorization:")
    print(f"  Training Allowed: {readiness['real_ctc_training_allowed']}")
    print(f"  Authorization Reason: {auth['reason']}")
    print(f"  Training Executed: {readiness['real_ctc_training_executed']}")
    print(f"  Real Checkpoint Created: {readiness['real_checkpoint_created']}")
    print(f"  Generalization Claims: {readiness['generalization_claims']}")
    print(f"  Publication Grade Evaluation: {readiness['publication_grade_evaluation']}")

    print("\n[7] Protected Reference Integrity:")
    print(f"  Status: {ref_integrity['status']} ({ref_integrity['matching_files']}/{ref_integrity['total_files']} files matching SHA-256 baseline)")

    print("\n[8] Active Blockers & Next Actions:")
    if readiness["failed_conditions"]:
        print(f"  Failed Conditions: {', '.join(readiness['failed_conditions'])}")
    else:
        print("  Failed Conditions: None (All conditions satisfied)")

    if readiness["supervision_state"] == "STATE_B":
        if not readiness["annotation_activity_started"]:
            print("  Action Required: Assign pilot samples and initiate human annotation collection.")
        else:
            print("  Action Required: Genuine human annotations must be submitted and verified through the annotation platform.")
    else:
        print("  Action Permitted: Real CTC training is authorized.")

    print("\n===========================================================")
    print(f" Diagnostic check completed safely. Supervision State: {readiness['supervision_state']}")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
