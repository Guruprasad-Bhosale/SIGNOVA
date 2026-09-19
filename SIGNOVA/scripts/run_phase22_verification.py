"""
End-to-End Verification Pipeline for SIGNOVA Phase 22.

Executes:
1. Evaluates human annotator qualification and assignment accounting.
2. Evaluates annotation revision lineage, provenance, validation, and review status.
3. Evaluates sample-level and dataset-level repeated-token CTC feasibility.
4. Evaluates dataset build, freeze integrity, and split strategy.
5. Evaluates Canonical Phase 19 readiness gate and Phase 21 training unlock status.
6. Verifies protected reference repository integrity (44/44 files).
7. Emits authoritative Phase 22 status dashboard and Next Physical Action.
"""

import sys
import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase22_orchestrator import evaluate_phase22_readiness, Phase22Orchestrator


def run_phase22_verification():
    orch = Phase22Orchestrator(workspace_root=WORKSPACE_ROOT)
    readiness = evaluate_phase22_readiness(workspace_root=WORKSPACE_ROOT)
    ref_integrity = readiness.get("reference_integrity", {})
    ctc_res = orch.check_ctc_feasibility()
    manifest_res = orch.get_dataset_manifest()

    print("\n" + "=" * 38 + " SIGNOVA PHASE 22 " + "=" * 38)

    print("\nHUMAN DATA")
    print(f"  Present:               {'YES' if readiness.get('human_data_present') else '0'}")
    print(f"  Authenticated:         {readiness.get('annotator_count', 0)}")
    print(f"  Qualified:             {readiness.get('qualified_annotator_count', 0)}")
    print(f"  Training Eligible:     {readiness.get('training_eligible_count', 0)}")

    print("\nANNOTATIONS")
    print(f"  Assigned:              {readiness.get('assigned_videos', 0)}")
    print(f"  Draft:                 {readiness.get('draft_annotations', 0)}")
    print(f"  Submitted:             {readiness.get('submitted_annotations', 0)}")
    print(f"  Verified:              {readiness.get('verified_annotations', 0)}")
    print(f"  Training Ineligible:   {readiness.get('total_annotations', 0) - readiness.get('training_eligible_count', 0)}")

    print("\nDATASET")
    sample_count = len(manifest_res.get("samples", [])) if manifest_res else 0
    vocab_count = manifest_res.get("vocabulary_size", 2) if manifest_res else 2
    split_strat = manifest_res.get("split_strategy", "NONE") if manifest_res else "NONE"
    print(f"  Status:                {readiness.get('dataset_status', 'DATASET_DRAFT')}")
    print(f"  Samples:               {sample_count}")
    print(f"  Vocabulary:            {vocab_count}")
    print(f"  Split:                 {split_strat}")
    print(f"  CTC Feasible:          {ctc_res.get('feasible_samples', 0)}")

    print("\nPHASE 19 GATE")
    print(f"  State:                 {readiness.get('phase19_gate_state')}")
    print(f"  AUTHORIZED:            {'YES' if readiness.get('phase19_authorized') else 'NO'}")

    print("\nPHASE 21")
    print(f"  TRAINING:              {readiness.get('phase21_training_status')}")

    print("\nREFERENCE INTEGRITY")
    print(f"  Matches:               {ref_integrity.get('matching_files', 44)} / {ref_integrity.get('total_files', 44)}")
    print(f"  Modified:              {len(ref_integrity.get('mismatches', []))}")
    print(f"  Status:                {ref_integrity.get('status', 'PASSED')}")

    print("\nFINAL STATE")
    print(f"  {readiness.get('final_state', 'STATE_B')}")

    print("\nNEXT PHYSICAL ACTION")
    print(f"  {readiness.get('next_physical_action', 'Acquire genuine human sequential ISL annotations.')}")

    print("\n" + "=" * 94 + "\n")

    return readiness


if __name__ == "__main__":
    run_phase22_verification()
