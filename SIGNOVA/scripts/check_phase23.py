#!/usr/bin/env python3
"""
SIGNOVA Phase 23 - Status & Readiness Dashboard.

Outputs the standardized 8-section status dashboard:
1. ACQUISITION
2. ANNOTATIONS
3. TRAINING DATA
4. DATASET
5. PHASE 19 GATE
6. PHASE 21 CTC
7. LIVE MODEL
8. FINAL STATE & NEXT PHYSICAL ACTION
"""

import sys
import json
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase23_orchestrator import evaluate_phase23_readiness


def parse_args():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 23 Status Dashboard")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--json", action="store_true", help="Output JSON result")
    return parser.parse_args()


def main():
    args = parse_args()
    res = evaluate_phase23_readiness(workspace_root=WORKSPACE_ROOT, data_root=args.data_root)

    if args.json:
        print(json.dumps(res, indent=2))
        return

    print("=" * 65)
    print(" SIGNOVA PHASE 23 -- STATUS DASHBOARD")
    print("=" * 65)

    print("\nACQUISITION")
    print(f"  Status:                {res.get('acquisition_status')}")
    print(f"  Assignments:           {res.get('assignments', {}).get('total', 0)}")
    print(f"  Annotators:            {res.get('annotators', {}).get('registered', 0)}")
    print(f"  Qualified:             {res.get('annotators', {}).get('qualified', 0)}")

    print("\nANNOTATIONS")
    print(f"  Discovered:            {res.get('annotations', {}).get('discovered', 0)}")
    print(f"  Valid:                 {res.get('annotations', {}).get('valid', 0)}")
    print(f"  Submitted:             {res.get('annotations', {}).get('submitted', 0)}")
    print(f"  Verified:              {res.get('annotations', {}).get('verified', 0)}")
    print(f"  Rejected:              {res.get('annotations', {}).get('rejected', 0)}")
    print(f"  Revision Required:     {res.get('annotations', {}).get('revision_required', 0)}")

    print("\nTRAINING DATA")
    print(f"  Eligible:              {res.get('training_data', {}).get('eligible', 0)}")
    print(f"  CTC Feasible:          {res.get('training_data', {}).get('ctc_feasible', 0)}")
    print(f"  CTC Infeasible:        {res.get('training_data', {}).get('ctc_infeasible', 0)}")
    print(f"  Vocabulary:            {res.get('training_data', {}).get('vocabulary_size', 2)}")

    print("\nDATASET")
    print(f"  Status:                {res.get('dataset', {}).get('status')}")
    print(f"  Version:               {res.get('dataset', {}).get('version')}")
    print(f"  Fingerprint:           {res.get('dataset', {}).get('fingerprint')}")
    print(f"  Split:                 {res.get('dataset', {}).get('split')}")

    print("\nPHASE 19")
    print(f"  State:                 {res.get('phase19', {}).get('state')}")
    print(f"  Authorized:            {'YES' if res.get('phase19', {}).get('authorized') else 'NO'}")

    print("\nPHASE 21")
    print(f"  Training:              {res.get('phase21', {}).get('training_status')}")
    print(f"  Checkpoint:            {res.get('phase21', {}).get('checkpoint')}")
    print(f"  Evaluation:            {res.get('phase21', {}).get('evaluation')}")

    print("\nLIVE MODEL")
    print(f"  Input Spec:            {res.get('live_model', {}).get('input_spec')}")
    print(f"  Verified:              {'YES' if res.get('live_model', {}).get('verified') else 'NO'}")
    print(f"  Smoke Test:            {'YES' if res.get('live_model', {}).get('smoke_test') else 'NO'}")
    print(f"  Authorized:            {'YES' if res.get('live_model', {}).get('authorized') else 'NO'}")

    print("\nSTATE REPORTING")
    print(f"  Supervision State:     {res.get('supervision_state')}")
    print(f"  Training Readiness:    {res.get('training_readiness')}")
    print(f"  Training Execution:    {res.get('training_execution')}")
    print(f"  Final State:           {res.get('final_state')}")

    print("\nNEXT PHYSICAL ACTION")
    print(f"  {res.get('next_physical_action')}")

    print("=" * 65)


if __name__ == "__main__":
    main()
