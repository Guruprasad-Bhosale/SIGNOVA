"""
Annotator Qualification Validation CLI for SIGNOVA Phase 22.
"""

import argparse
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase22_orchestrator import Phase22Orchestrator, QUAL_QUALIFIED


def main():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 22 Annotator Validation")
    parser.add_argument("--id", type=str, required=True, help="Annotator ID to validate")
    args = parser.parse_args()

    orch = Phase22Orchestrator(workspace_root=WORKSPACE_ROOT)
    prof = orch.load_annotator_profile(args.id)

    print("===========================================================")
    print(" SIGNOVA PHASE 22 -- ANNOTATOR VALIDATION")
    print("===========================================================")

    if prof is None:
        print(f"\n[!] Annotator '{args.id}' NOT found in registry.")
        print("    Status: UNREGISTERED / INELIGIBLE")
        print("\n===========================================================\n")
        return

    print(f"\n[+] Profile for '{prof.annotator_id}':")
    print(f"  Qualification Status: {prof.qualification_status}")
    print(f"  Method:               {prof.qualification_method}")
    print(f"  Evidence:             {prof.qualification_evidence or 'None'}")
    print(f"  Verified By:          {prof.verified_by or 'Pending'}")
    print(f"  Training Eligible:    {'YES' if prof.qualification_status == QUAL_QUALIFIED else 'NO'}")
    print("\n===========================================================\n")


if __name__ == "__main__":
    main()
