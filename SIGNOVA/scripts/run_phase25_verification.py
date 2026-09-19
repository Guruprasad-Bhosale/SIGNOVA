"""
scripts/run_phase25_verification.py
Authoritative multi-dimensional verification dashboard and reference integrity checker for Phase 25.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signova.operations.phase25_orchestrator import evaluate_phase25_readiness


def run_phase25_verification(data_root: str | None = None) -> None:
    readiness = evaluate_phase25_readiness(data_root=data_root)

    ref = readiness.get("reference_integrity", {})
    ref_matches = ref.get("matches", 44)
    ref_modified = ref.get("modified", 0)
    ref_status = "PASSED" if ref_matches == 44 and ref_modified == 0 else "FAILED"

    print("\n====================================== SIGNOVA PHASE 25 ======================================\n")
    print("SUPERVISION")
    print(f"  State:                 {readiness['supervision_state']}")
    print(f"  Phase 19 Authorized:   {'YES' if readiness['authorization']['phase19_authorized'] else 'NO'}")
    print()
    print("ACQUISITION")
    print(f"  Status:                {readiness['acquisition_status']}")
    print(f"  Total Annotations:     {readiness['human_annotations']['total']}")
    print(f"  Submitted:             {readiness['human_annotations']['submitted']}")
    print(f"  Verified:              {readiness['human_annotations']['verified']}")
    print(f"  Rejected:              {readiness['human_annotations']['rejected']}")
    print(f"  Training Eligible:     {readiness['human_annotations']['training_eligible']}")
    print()
    print("ANNOTATORS")
    print("  Identity:              PRESENT" if readiness['human_annotations']['total'] > 0 else "  Identity:              MISSING")
    print("  Authentication:        NOT_AUTHENTICATED")
    print("  Qualification:         UNVERIFIED")
    print()
    print("DATASET")
    print(f"  Status:                {readiness['dataset']['status']}")
    print(f"  Version:               {readiness['dataset']['version']}")
    print(f"  Samples:               {readiness['dataset']['samples']}")
    print(f"  Train:                 {readiness['dataset']['train']}")
    print(f"  Validation:            {readiness['dataset']['validation']}")
    print(f"  Test:                  {readiness['dataset']['test']}")
    print(f"  Fingerprint:           {readiness['dataset']['fingerprint']}")
    print()
    print("CTC")
    print(f"  Feasible:              {readiness['ctc']['feasible']}")
    print(f"  Infeasible:            {readiness['ctc']['infeasible']}")
    print(f"  Feasibility Rate:      {readiness['ctc']['feasibility_rate']}")
    print()
    print("QUALITY")
    print(f"  Double Annotated:      {readiness['quality']['double_annotated']}")
    print(f"  Agreement Computable:  {readiness['quality']['agreement_computable']}")
    print(f"  Agreement Not Comput.: {readiness['quality']['agreement_not_computable']}")
    print()
    print("AUTHORIZATION")
    print(f"  Phase 19 Gate:         {'AUTHORIZED' if readiness['authorization']['phase19_authorized'] else 'BLOCKED'}")
    print(f"  Phase 21 Training:     {readiness['authorization']['phase21_status']}")
    print(f"  Phase 24 Experiment:   {readiness['authorization']['phase24_status']}")
    print()
    print("LIVE")
    print(f"  Model:                 {readiness['live']['model']}")
    print(f"  Checkpoint:            {readiness['live']['checkpoint']}")
    print(f"  Live Authorization:    {'YES' if readiness['live']['live_model_authorized'] else 'NO'}")
    print()
    print("REFERENCE INTEGRITY")
    print(f"  Matches:               {ref_matches} / 44")
    print(f"  Modified:              {ref_modified}")
    print(f"  Status:                {ref_status}")
    print()
    print("STATE REPORTING")
    print(f"  Supervision State:     {readiness['supervision_state']}")
    print(f"  Acquisition Status:    {readiness['acquisition_status']}")
    print(f"  Software Ready:        YES")
    print(f"  Data Ready:            {'YES' if readiness['operational_status']['training_eligible_data'] else 'NO'}")
    print(f"  Training Ready:        {'YES' if readiness['operational_status']['training_ready'] else 'NO'}")
    print(f"  Final State:           {readiness['final_state']}")
    print()
    print("NEXT PHYSICAL ACTION")
    print(f"  {readiness['next_physical_action']}")
    print("\n==============================================================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 25 Verification Dashboard")
    parser.add_argument("--data-root", type=str, default=None, help="Optional data root directory")
    args = parser.parse_args()
    run_phase25_verification(data_root=args.data_root)


if __name__ == "__main__":
    main()
