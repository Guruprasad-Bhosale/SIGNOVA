"""
Lightweight Diagnostic Check CLI for SIGNOVA Phase 21.

Safely inspects the workspace and dynamically reports:
- Current supervision state (STATE_B, STATE_A_DATA_LIMITED, STATE_A)
- Human-data state (PRESENT, AUTHENTICATED, QUALIFIED, TRAINING_ELIGIBLE)
- Dataset scale & sample accounting
- Genuine vocabulary size
- Split strategy & leakage status
- Sequence CTC feasibility
- Real CTC training authorization status
- Checkpoint existence & provenance
- Phase 20 live runtime integration status
- Active blockers and required next actions

This script NEVER triggers training and NEVER produces model checkpoints.
"""

from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase21_orchestrator import evaluate_phase21_readiness


def main():
    print("===========================================================")
    print(" SIGNOVA PHASE 21 -- LIGHTWEIGHT DIAGNOSTIC CHECK")
    print("===========================================================")

    readiness = evaluate_phase21_readiness(workspace_root=WORKSPACE_ROOT)
    accounting = readiness["sample_accounting"]
    feasibility = readiness["ctc_feasibility"]
    ref_integrity = readiness["reference_integrity"]
    auth = readiness["training_authorization"]
    live_auth = readiness["live_authorization"]

    print("\n[1] Supervision State:")
    print(f"  Supervision State: {readiness['supervision_state']}")
    print(f"  Real CTC Status:   {readiness['real_ctc_status']}")
    print(f"  Pilot Status:      {readiness['pilot_status']}")
    print(f"  Dataset Scale:     {readiness['dataset_scale']}")

    print("\n[2] Human Data State:")
    print(f"  Present:           {readiness['human_data_present']}")
    print(f"  Authenticated:     {readiness['human_data_authenticated']}")
    print(f"  Qualified:         {readiness['human_data_qualified']}")
    print(f"  Training Eligible: {readiness['training_eligible_data']}")
    print(f"  Total Samples:     {accounting['total_annotations']}")
    print(f"  Verified Samples:  {accounting['verified_annotations']}")
    print(f"  Eligible Samples:  {accounting['training_eligible_samples']}")
    print(f"  Vocabulary Size:   {readiness['vocabulary_size']} tokens")

    print("\n[3] Split Strategy & CTC Feasibility:")
    print(f"  Split Strategy:    {readiness['split_strategy']}")
    print(f"  Feasibility Status:{feasibility['feasibility_status']}")
    print(f"  Feasibility Rate:  {feasibility['feasibility_rate'] * 100:.1f}% ({feasibility['feasible_sequences']} / {feasibility['total_sequences']})")

    print("\n[4] Real CTC Training Authorization:")
    print(f"  Training Allowed:  {readiness['real_ctc_training_allowed']}")
    print(f"  Gate Decision:     {auth['reason']}")
    if readiness["failed_conditions"]:
        print(f"  Failed Conditions: {', '.join(readiness['failed_conditions'])}")

    print("\n[5] Phase 20 Live Integration Status:")
    print(f"  Live Authorized:   {live_auth['live_model_authorized']}")
    print(f"  Current Stage:     {live_auth['current_stage']}")
    print(f"  Stage Breakdown:   {live_auth['stages']}")

    print("\n[6] Protected Reference Baseline:")
    print(f"  Baseline Match:    {ref_integrity['matching_files']}/{ref_integrity['total_files']} files ({ref_integrity['status']})")

    print("\n[7] Next Action:")
    if readiness["supervision_state"] == "STATE_B":
        print("  Acquire and verify genuine human sequential ISL annotations through the existing annotation platform.")
    else:
        print("  Execute genuine real CTC training: python scripts/train_phase21_ctc.py")

    print("\n===========================================================")
    print(f" Diagnostic check completed safely. State: {readiness['supervision_state']}")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
