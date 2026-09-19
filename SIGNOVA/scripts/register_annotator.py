"""
Annotator Registration CLI for SIGNOVA Phase 22.

Registers a human annotator with explicit qualification evidence.
"""

import argparse
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase22_orchestrator import Phase22Orchestrator, QUAL_PENDING, QUAL_QUALIFIED


def main():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 22 Annotator Registration")
    parser.add_argument("--id", type=str, required=True, help="Annotator ID (e.g. annotator_isl_01)")
    parser.add_argument("--status", type=str, default=QUAL_PENDING, choices=[QUAL_PENDING, QUAL_QUALIFIED, "SUSPENDED", "REVOKED"], help="Qualification status")
    parser.add_argument("--method", type=str, default="MANUAL_EVALUATION", help="Qualification method")
    parser.add_argument("--evidence", type=str, default="", help="Qualification evidence or credentials")
    parser.add_argument("--verified-by", type=str, default=None, help="Reviewer / Organization ID verifying qualification")
    parser.add_argument("--notes", type=str, default="", help="Additional notes")
    args = parser.parse_args()

    orch = Phase22Orchestrator(workspace_root=WORKSPACE_ROOT)
    prof = orch.register_annotator(
        annotator_id=args.id,
        qualification_status=args.status,
        qualification_method=args.method,
        qualification_evidence=args.evidence,
        verified_by=args.verified_by,
        notes=args.notes,
    )

    print("===========================================================")
    print(" SIGNOVA PHASE 22 -- ANNOTATOR REGISTRATION")
    print("===========================================================")
    print(f"\n[+] Annotator Registered Successfully:")
    print(f"  Annotator ID:         {prof.annotator_id}")
    print(f"  Qualification Status: {prof.qualification_status}")
    print(f"  Method:               {prof.qualification_method}")
    print(f"  Evidence:             {prof.qualification_evidence or 'None provided'}")
    print(f"  Verified By:          {prof.verified_by or 'Pending verification'}")
    print("\n===========================================================\n")


if __name__ == "__main__":
    main()
