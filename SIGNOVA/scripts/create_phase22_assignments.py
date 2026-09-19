"""
Pilot Assignment Manifest Generator for SIGNOVA Phase 22.

Creates deterministic video assignments without mutating existing assignments
unless --reassign is explicitly provided.
"""

import argparse
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase22_orchestrator import Phase22Orchestrator


def main():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 22 Assignment Generator")
    parser.add_argument("--reassign", action="store_true", help="Force regenerate assignments (destructive)")
    parser.add_argument("--num-videos", type=int, default=20, help="Number of pilot videos (default: 20)")
    parser.add_argument("--annotator-pool", type=str, nargs="+", default=["annotator_isl_lead_01", "annotator_isl_lead_02"], help="Pool of annotators")
    args = parser.parse_args()

    print("===========================================================")
    print(" SIGNOVA PHASE 22 -- PILOT ASSIGNMENT MANIFEST")
    print("===========================================================")

    orch = Phase22Orchestrator(workspace_root=WORKSPACE_ROOT)
    v_ids = [f"video_p22_{i:03d}" for i in range(1, args.num_videos + 1)]
    assignments = orch.create_assignments(
        video_ids=v_ids,
        annotator_ids=args.annotator_pool,
        reassign=args.reassign,
    )

    print(f"\n[+] Total Assigned Slots:    {len(assignments)}")
    print(f"  Unique Videos Assigned:  {len({a['video_id'] for a in assignments})}")
    print(f"  Double Annotation Slots: {sum(1 for a in assignments if a['assignment_type'] == 'DOUBLE_ANNOTATION')}")
    print(f"  Manifest File:           {orch.assignments_file}")

    print("\nSample Assignment Entries:")
    for a in assignments[:5]:
        print(f"  * {a['assignment_id']} -> Video: {a['video_id']} | Annotator: {a['annotator_id']} | Type: {a['assignment_type']}")

    print("\n===========================================================\n")


if __name__ == "__main__":
    main()
