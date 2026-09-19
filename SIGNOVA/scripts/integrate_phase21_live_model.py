"""
Live Model Integration and Multi-Stage Readiness Gate CLI for SIGNOVA Phase 21.

Evaluates:
TRAINED -> CHECKPOINT VERIFIED -> HELD-OUT EVALUATION -> INPUT SPEC MATCH -> LIVE SMOKE TEST -> LIVE_MODEL_AUTHORIZED
"""

import argparse
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase21_orchestrator import Phase21Orchestrator


def main():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 21 Live Model Integration")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory to integrate")
    args = parser.parse_args()

    print("===========================================================")
    print(" SIGNOVA PHASE 21 -- LIVE MODEL INTEGRATION GATE")
    print("===========================================================")

    orch = Phase21Orchestrator(workspace_root=WORKSPACE_ROOT)
    res = orch.evaluate_live_authorization_pipeline(run_dir=args.run_dir)

    print("\n[+] Multi-Stage Live Readiness Assessment:")
    stages = res.get("stages", {})
    print(f"  [1] TRAINED:               {'PASSED' if stages.get('trained') else 'BLOCKED'}")
    print(f"  [2] CHECKPOINT VERIFIED:   {'PASSED' if stages.get('checkpoint_verified') else 'BLOCKED'}")
    print(f"  [3] HELD-OUT EVALUATION:   {'PASSED' if stages.get('held_out_evaluation') else 'BLOCKED'}")
    print(f"  [4] INPUT SPEC MATCH:      {'PASSED' if stages.get('input_spec_match') else 'BLOCKED'}")
    print(f"  [5] LIVE SMOKE TEST:       {'PASSED' if stages.get('live_smoke_test') else 'BLOCKED'}")

    print("\n[+] Live Authorization Decision:")
    print(f"  Current Status:    {res['status']}")
    print(f"  Current Stage:     {res['current_stage']}")
    print(f"  Live Authorized:   {res['live_model_authorized']}")
    if res.get("reason"):
        print(f"  Reason:            {res['reason']}")

    if res["live_model_authorized"]:
        print(f"\n[+] Active Run Integrated: {res.get('active_run_dir')}")
        print("    LiveModelRegistry updated with pointer: models/experiments/phase21_real_ctc/live_model_pointer.json")
        print("    Phase 20 live camera runtime will now load this model in LIVE mode.")
    else:
        print("\n[!] Live Model Integration: BLOCKED.")
        print("    Phase 20 live camera runtime remains in OPERATIONAL DIAGNOSTIC MODE (Zero hallucinations).")

    print("\n===========================================================")
    print(f" Integration gate evaluated safely.")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
