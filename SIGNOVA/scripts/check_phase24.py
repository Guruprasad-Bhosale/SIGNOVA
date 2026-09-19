#!/usr/bin/env python3
"""
SIGNOVA Phase 24 - Status & Experiment Readiness Dashboard.

Outputs the standardized 8-section status dashboard:
1. SUPERVISION
2. DATASET
3. CTC
4. TRAINING
5. EVALUATION
6. MODEL READINESS
7. LIVE
8. STATE REPORTING & NEXT PHYSICAL ACTION
"""

import sys
import json
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase24_orchestrator import evaluate_phase24_readiness


def parse_args():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 24 Status Dashboard")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--json", action="store_true", help="Output JSON result")
    return parser.parse_args()


def main():
    args = parse_args()
    res = evaluate_phase24_readiness(workspace_root=WORKSPACE_ROOT, data_root=args.data_root)

    if args.json:
        print(json.dumps(res, indent=2))
        return

    print("=" * 65)
    print(" SIGNOVA PHASE 24 -- STATUS DASHBOARD")
    print("=" * 65)

    print("\nSUPERVISION")
    print(f"  State:                 {res.get('supervision', {}).get('state')}")
    print(f"  Phase 19 Authorized:   {'YES' if res.get('supervision', {}).get('phase19_authorized') else 'NO'}")

    print("\nDATASET")
    print(f"  Version:               {res.get('dataset', {}).get('version')}")
    print(f"  Samples:               {res.get('dataset', {}).get('samples', 0)}")
    print(f"  Train:                 {res.get('dataset', {}).get('train', 0)}")
    print(f"  Validation:            {res.get('dataset', {}).get('validation', 0)}")
    print(f"  Test:                  {res.get('dataset', {}).get('test', 0)}")
    print(f"  Fingerprint:           {res.get('dataset', {}).get('fingerprint')}")
    print(f"  Vocabulary:            {res.get('dataset', {}).get('vocabulary', 2)}")
    print(f"  Split:                 {res.get('dataset', {}).get('split')}")

    print("\nCTC")
    print(f"  Feasible:              {res.get('ctc', {}).get('feasible', 0)}")
    print(f"  Infeasible:            {res.get('ctc', {}).get('infeasible', 0)}")
    print(f"  Feasibility Rate:      {res.get('ctc', {}).get('feasibility_rate', 0.0):.2%}")

    print("\nTRAINING")
    print(f"  Action:                {res.get('training', {}).get('action')}")
    print(f"  Status:                {res.get('training', {}).get('status')}")
    print(f"  Device:                {res.get('training', {}).get('device')}")
    print(f"  Experiment:            {res.get('training', {}).get('experiment')}")
    print(f"  Checkpoint:            {res.get('training', {}).get('checkpoint')}")

    print("\nEVALUATION")
    print(f"  Status:                {res.get('evaluation', {}).get('status')}")
    print(f"  TER:                   {res.get('evaluation', {}).get('ter')}")
    print(f"  Exact Match:           {res.get('evaluation', {}).get('exact_match')}")
    print(f"  Token F1:              {res.get('evaluation', {}).get('token_f1')}")
    print(f"  Insertions:            {res.get('evaluation', {}).get('insertions')}")
    print(f"  Deletions:             {res.get('evaluation', {}).get('deletions')}")
    print(f"  Substitutions:         {res.get('evaluation', {}).get('substitutions')}")

    print("\nMODEL READINESS")
    print(f"  Trained:               {'YES' if res.get('model_readiness', {}).get('trained') else 'NO'}")
    print(f"  Checkpoint Verified:   {'YES' if res.get('model_readiness', {}).get('checkpoint_verified') else 'NO'}")
    print(f"  Held-Out Evaluated:    {'YES' if res.get('model_readiness', {}).get('held_out_evaluated') else 'NO'}")
    print(f"  Input Spec Match:      {'YES' if res.get('model_readiness', {}).get('input_spec_match') else 'NO'}")
    print(f"  Live Smoke Test:       {'YES' if res.get('model_readiness', {}).get('live_smoke_test') else 'NO'}")
    print(f"  Live Authorized:       {'YES' if res.get('model_readiness', {}).get('live_authorized') else 'NO'}")

    print("\nLIVE")
    print(f"  Model:                 {res.get('live', {}).get('model')}")
    print(f"  Tracking:              {res.get('live', {}).get('tracking')}")
    print(f"  Activity:              {res.get('live', {}).get('activity')}")
    print(f"  CTC:                   {res.get('live', {}).get('ctc')}")
    print(f"  Gloss:                 {res.get('live', {}).get('gloss')}")
    print(f"  English:               {res.get('live', {}).get('english')}")
    print(f"  Confidence:            {res.get('live', {}).get('confidence')}")
    print(f"  Abstention:            {res.get('live', {}).get('abstention')}")

    print("\nSTATE REPORTING")
    print(f"  Supervision State:     {res.get('state_reporting', {}).get('supervision_state')}")
    print(f"  Experiment Status:     {res.get('state_reporting', {}).get('experiment_status')}")
    print(f"  ISL Recognition:       {res.get('state_reporting', {}).get('isl_recognition_validation')}")
    print(f"  Gloss to English:      {res.get('state_reporting', {}).get('gloss_to_english_validation')}")
    print(f"  Final State:           {res.get('final_state')}")

    print("\nNEXT PHYSICAL ACTION")
    print(f"  {res.get('next_physical_action')}")

    print("=" * 65)


if __name__ == "__main__":
    main()
