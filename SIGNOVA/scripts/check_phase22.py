"""
Lightweight Diagnostic Check CLI for SIGNOVA Phase 22.

Safely inspects the workspace and dynamically reports:
- Human data acquisition status (present, authenticated, qualified, eligible)
- Annotator registration and qualification counts
- 6-tier sample and annotation accounting
- Independent double annotation and agreement status
- Dataset scale and CTC feasibility
- Canonical Phase 19/21 readiness gates
- Mandatory NEXT PHYSICAL ACTION
"""

from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase22_orchestrator import evaluate_phase22_readiness


def main():
    print("===========================================================")
    print(" SIGNOVA PHASE 22 -- HUMAN ANNOTATION ACQUISITION CHECK")
    print("===========================================================")

    readiness = evaluate_phase22_readiness(workspace_root=WORKSPACE_ROOT)
    hd = readiness["human_data"]
    ann_prof = readiness["annotators"]
    acc = readiness["sample_accounting"]
    double_agr = readiness["double_annotation"]
    gates = readiness["readiness"]

    print(f"\n[1] Acquisition State:")
    print(f"  Mode:                  {readiness['acquisition_mode']}")
    print(f"  Supervision State:     {readiness['supervision_state']}")

    print(f"\n[2] Human Data States:")
    print(f"  Present:               {hd['present']}")
    print(f"  Authenticated:         {hd['authenticated']}")
    print(f"  Qualified:             {hd['qualified']}")
    print(f"  Training Eligible:     {hd['training_eligible']}")

    print(f"\n[3] Annotators:")
    print(f"  Registered:            {ann_prof['registered']}")
    print(f"  Qualified:             {ann_prof['qualified']}")

    print(f"\n[4] Annotation & Sample Accounting:")
    print(f"  Videos Assigned:       {acc['videos_assigned']}")
    print(f"  Total Assigned Slots:  {acc['total_assigned_slots']}")
    print(f"  Videos Annotated:      {acc['videos_annotated']}")
    print(f"  Draft Annotations:     {acc['annotations_draft']}")
    print(f"  Submitted Annotations: {acc['annotations_submitted']}")
    print(f"  Verified Annotations:  {acc['annotations_verified']}")
    print(f"  Rejected Annotations:  {acc['annotations_rejected']}")
    print(f"  Revision Required:     {acc['annotations_revision_required']}")
    print(f"  Training Eligible:     {acc['training_eligible']}")
    print(f"  Training Ineligible:   {acc['training_ineligible']}")
    print(f"  CTC Feasible Samples:  {acc['ctc_feasible']}")

    print(f"\n[5] Double Annotation:")
    print(f"  Status:                {double_agr['status']}")
    print(f"  Double Annotated Pairs:{double_agr['double_annotated_pairs']}")
    if double_agr.get("exact_sequence_agreement") is not None:
        print(f"  Exact Sequence Agr:    {double_agr['exact_sequence_agreement'] * 100:.1f}%")

    print(f"\n[6] Readiness Gates:")
    print(f"  Phase 19 Gate:         {gates['phase19_gate']}")
    print(f"  Phase 21 CTC Training: {'AUTHORIZED' if gates['phase21_training_authorized'] else 'BLOCKED'}")
    print(f"  Phase 20 Live Model:   {'AVAILABLE' if gates['phase20_live_model'] else 'UNAVAILABLE (DIAGNOSTIC)'}")

    print(f"\n[7] Next Action:")
    print(f"  NEXT PHYSICAL ACTION: {readiness['next_physical_action']}")

    print("\n===========================================================")
    print(f" Phase 22 check completed safely. State: {readiness['supervision_state']}")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
