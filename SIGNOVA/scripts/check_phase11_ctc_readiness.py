"""
Phase 11 CTC Readiness Checker Script for SIGNOVA.

Evaluates the 12 Authoritative STATE_A Conditions and writes machine-readable report.
"""

import json
from pathlib import Path
import sys

# Ensure src is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signova.recognition.phase11_gate import Phase11SupervisionGate


def main():
    gate = Phase11SupervisionGate()
    report = gate.to_dict()

    output_path = Path("outputs/reports/phase11_ctc_readiness.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Supervision State: {report['supervision_state']}")
    print(f"Real CTC Training: {report['real_ctc_training']}")
    print(f"Satisfied Conditions: {report['conditions_summary']['satisfied_count']}/12")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
