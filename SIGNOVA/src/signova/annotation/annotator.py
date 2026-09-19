"""
Phase 11 Annotator and Reviewer Metadata Models for SIGNOVA.

Strictly separates:
- Annotator ID (individual performing sequence annotation)
- Reviewer ID (linguist or senior reviewer verifying correctness)
- Signer ID (individual performing signs in the source video)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class AnnotatorRole(str, Enum):
    ANNOTATOR = "ANNOTATOR"
    REVIEWER = "REVIEWER"
    LEAD_LINGUIST = "LEAD_LINGUIST"


class AnnotatorQualificationState(str, Enum):
    UNQUALIFIED = "UNQUALIFIED"
    QUALIFIED = "QUALIFIED"
    SUSPENDED = "SUSPENDED"
    REVOKED = "REVOKED"


@dataclass
class AnnotatorProfile:
    annotator_id: str
    name: str
    role: AnnotatorRole = AnnotatorRole.ANNOTATOR
    qualification_status: AnnotatorQualificationState = AnnotatorQualificationState.UNQUALIFIED
    is_deaf_native: bool = True
    experience_years: float = 1.0
    institution: str = "Deaf Community / ISL Research Group"
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotator_id": self.annotator_id,
            "name": self.name,
            "role": self.role.value if isinstance(self.role, AnnotatorRole) else str(self.role),
            "qualification_status": (
                self.qualification_status.value
                if isinstance(self.qualification_status, AnnotatorQualificationState)
                else str(self.qualification_status)
            ),
            "is_deaf_native": self.is_deaf_native,
            "experience_years": self.experience_years,
            "institution": self.institution,
            "notes": self.notes,
            "metadata": self.metadata,
        }
