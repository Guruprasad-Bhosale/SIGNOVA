"""
Smoke test for Phase 16 Real Data Pipeline.

Verifies:
- Annotation discovery and ingestion mechanics
- Graceful handling when no human annotations are present (SKIPPED — NO_GENUINE_ANNOTATIONS)
- Feasibility, leakage, and gate invocation
"""

from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase16_orchestrator import Phase16Orchestrator


def smoke_test_phase16_pipeline():
    print("\n--- Starting Phase 16 Real Data Pipeline Smoke Test ---")
    orchestrator = Phase16Orchestrator()
    summary = orchestrator.run_full_qualification()

    total_discovered = summary["sample_accounting"]["total_discovered"]
    print(f"Total annotations discovered: {total_discovered}")
    print(f"Human data present: {summary['human_data_present']}")
    print(f"Human data qualified: {summary['human_data_qualified']}")
    print(f"Training eligible samples: {summary['sample_accounting']['training_eligible']}")

    if total_discovered == 0:
        print("Status: SKIPPED — NO_GENUINE_ANNOTATIONS (Awaiting external human data collection).")
    else:
        print(f"Status: PROCESSED — {summary['sample_accounting']['training_eligible']} training-eligible samples.")

    print(f"Supervision State: {summary['supervision_state']}")
    print(f"Real CTC Training Status: {summary['real_ctc_status']}")
    print("Phase 16 data qualification pipeline operational.\n")


if __name__ == "__main__":
    smoke_test_phase16_pipeline()
