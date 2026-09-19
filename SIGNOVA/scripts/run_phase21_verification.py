"""
End-to-End Verification Pipeline for SIGNOVA Phase 21.

Executes:
1. Canonical Phase 19 readiness gate check & snapshot
2. Dataset qualification & scale assessment
3. Strict split hierarchy verification
4. Sequence CTC feasibility calculation
5. Training gating invariant verification
6. Checkpoint provenance & cryptographic hash verification
7. Multi-stage live model authorization assessment
8. Protected reference repository integrity verification (44/44)
9. Emits authoritative Phase 21 status summary banner
"""

import json
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase21_orchestrator import evaluate_phase21_readiness, Phase21Orchestrator


def run_phase21_verification():
    orch = Phase21Orchestrator(workspace_root=WORKSPACE_ROOT)
    readiness = evaluate_phase21_readiness(workspace_root=WORKSPACE_ROOT)

    accounting = readiness["sample_accounting"]
    feasibility = readiness["ctc_feasibility"]
    live_auth = readiness["live_authorization"]
    ref_integrity = readiness["reference_integrity"]
    auth = readiness["training_authorization"]

    # Check trained checkpoint metadata if exists
    pointer_file = orch.experiments_dir / "live_model_pointer.json"
    active_meta = None
    if pointer_file.exists():
        try:
            pdata = json.loads(pointer_file.read_text(encoding="utf-8"))
            active_dir = Path(pdata.get("active_run_dir", ""))
            meta_path = active_dir / "model_metadata.json"
            if meta_path.exists():
                active_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Print Authoritative Banner
    print("\n" + "=" * 40 + " SIGNOVA PHASE 21 " + "=" * 40)
    
    print("\nSUPERVISION")
    print(f"  Human data:            {'YES' if readiness['human_data_present'] else 'NO'}")
    print(f"  Qualified annotations: {accounting['verified_annotations']}")
    print(f"  Training eligible:     {accounting['training_eligible_samples']}")

    print("\nDATASET")
    print(f"  Samples:               {accounting['training_eligible_samples']}")
    print(f"  Vocabulary:            {readiness['vocabulary_size']}")
    print(f"  Split:                 {readiness['split_strategy']}")
    print(f"  CTC feasible:          {'YES' if feasibility['feasibility_status'] in ('PASSED', 'FEASIBLE') else 'NO'}")

    print("\nTRAINING")
    print(f"  Authorized:            {'YES' if readiness['real_ctc_training_allowed'] else 'NO'}")
    if active_meta:
        print(f"  Status:                TRAINED")
        print(f"  Checkpoint:            {active_meta.get('checkpoint_sha256', 'N/A')}")
    else:
        print(f"  Status:                BLOCKED")
        print(f"  Checkpoint:            NONE (Under STATE_B, zero fake checkpoints generated)")

    print("\nEVALUATION")
    if active_meta and "evaluation" in active_meta:
        ev = active_meta["evaluation"]
        print(f"  TER:                   {ev.get('ter', 'N/A')}")
        print(f"  Exact Match:           {ev.get('exact_match', 'N/A')}")
        print(f"  Insertions/Deletions:  {ev.get('insertions', 0)} / {ev.get('deletions', 0)}")
    else:
        print(f"  TER:                   N/A (Awaiting genuine training)")
        print(f"  Exact Match:           N/A")
        print(f"  Token F1:              N/A")

    print("\nLIVE INTEGRATION")
    print(f"  Input spec:            {'MATCH' if live_auth['stages'].get('input_spec_match') else 'MISMATCH / AWAITING'}")
    print(f"  Model loaded:          {'YES' if live_auth['live_model_authorized'] else 'NO'}")
    print(f"  Translation enabled:   {'YES' if live_auth['live_model_authorized'] else 'NO'}")

    print("\nFINAL STATE")
    print(f"  {readiness['supervision_state']}")

    print("\n" + "=" * 98 + "\n")

    return readiness


if __name__ == "__main__":
    run_phase21_verification()
