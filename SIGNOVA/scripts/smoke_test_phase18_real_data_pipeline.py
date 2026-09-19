"""
Smoke test for Phase 18 Real Data Pipeline.

Verifies:
- Annotation discovery, authenticity verification, and lifecycle checks
- Dynamic execution:
    * If NO annotations present -> SKIPPED — NO_GENUINE_ANNOTATIONS
    * If annotations present -> runs pipeline steps (video resolution, feature loading, gloss normalization, vocabulary, CTC feasibility, split membership, training eligibility) without training.
- Gate and CTC readiness status.
"""

from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase18_orchestrator import Phase18Orchestrator


def smoke_test_phase18_pipeline():
    print("\n--- Starting Phase 18 Real Data Pipeline Smoke Test ---")
    orchestrator = Phase18Orchestrator()
    summary = orchestrator.run_full_qualification()

    total_discovered = summary["sample_accounting"]["total_discovered"]
    print(f"Total annotations discovered: {total_discovered}")
    print(f"Human data present: {summary['human_data_present']}")
    print(f"Human data authenticated: {summary['human_data_authenticated']}")
    print(f"Human data qualified: {summary['human_data_qualified']}")
    print(f"Training eligible samples: {summary['sample_accounting']['training_eligible_samples']}")

    if total_discovered == 0:
        print("Status: SKIPPED — NO_GENUINE_ANNOTATIONS (Awaiting external human data collection).")
    else:
        print(f"Status: PROCESSED — {summary['sample_accounting']['training_eligible_samples']} training-eligible samples.")
        print(f"Vocabulary size: {summary['vocabulary_size']}")
        print(f"CTC Feasibility status: {summary['ctc_feasibility']['feasibility_status']} (Rate: {summary['ctc_feasibility']['feasibility_rate'] * 100:.1f}%)")
        print(f"Split Strategy: {summary['split_strategy']}")

    print(f"Supervision State: {summary['supervision_state']}")
    print(f"Real CTC Training Status: {summary['real_ctc_status']}")
    print(f"Pilot Status: {summary['pilot_status']}")
    print("Phase 18 data qualification pipeline operational.\n")


if __name__ == "__main__":
    smoke_test_phase18_pipeline()
