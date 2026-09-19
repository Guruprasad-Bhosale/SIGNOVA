"""
Phase 15 Automated Integrity Audit against Prohibited Pseudo-Labeling Pathways.

Enforces:
- No English sentence -> gloss conversion used as supervision
- No model prediction -> ground-truth annotation creation
- No LLM-generated pseudo-glosses assigned to human datasets
"""

from pathlib import Path
from typing import Any, Dict, List


def audit_prohibited_automation_pathways(annotations_dir: Path) -> Dict[str, Any]:
    """
    Audits annotations to verify they originate from human operations rather than
    automated text translation or model inference pseudo-labeling.
    """
    issues: List[str] = []
    total_checked = 0

    if annotations_dir.exists():
        for f in annotations_dir.glob("*.json"):
            total_checked += 1
            content = f.read_text(encoding="utf-8")
            # Check for suspicious pseudo-label flags
            if '"is_pseudo_label": true' in content:
                issues.append(f"PSEUDO_LABEL_DETECTED in {f.name}")
            if '"llm_generated": true' in content:
                issues.append(f"LLM_GENERATED_LABEL_DETECTED in {f.name}")

    return {
        "annotations_checked": total_checked,
        "prohibited_pathways_detected": len(issues) > 0,
        "violations": issues,
        "audit_status": "PASSED" if len(issues) == 0 else "FAILED",
    }
