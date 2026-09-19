"""
Phase 14 Non-Destructive Annotation Revision Manager for SIGNOVA.

Enforces immutable annotation history:
- Historical annotations are never overwritten.
- Reviewer corrections and annotator edits generate versioned revisions.
"""

from dataclasses import dataclass, field
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from signova.annotation.schema import VideoAnnotation


@dataclass
class AnnotationRevision:
    revision_id: str
    sample_id: str
    revision_number: int
    parent_revision_id: Optional[str]
    author_id: str
    author_role: str  # "ANNOTATOR", "REVIEWER", "LINGUIST"
    glosses: List[str]
    is_temporally_aligned: bool
    review_status: str
    quality_grade: str
    change_reason: str
    provenance_id: str
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "sample_id": self.sample_id,
            "revision_number": self.revision_number,
            "parent_revision_id": self.parent_revision_id,
            "author_id": self.author_id,
            "author_role": self.author_role,
            "glosses": self.glosses,
            "is_temporally_aligned": self.is_temporally_aligned,
            "review_status": self.review_status,
            "quality_grade": self.quality_grade,
            "change_reason": self.change_reason,
            "provenance_id": self.provenance_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class AnnotationRevisionHistory:
    """Manages the full immutable timeline of revisions for a given sample."""

    def __init__(self, sample_id: str):
        self.sample_id = sample_id
        self.revisions: List[AnnotationRevision] = []

    def add_revision(
        self,
        author_id: str,
        author_role: str,
        glosses: List[str],
        change_reason: str,
        is_temporally_aligned: bool = False,
        review_status: str = "ANNOTATION_IN_PROGRESS",
        quality_grade: str = "UNVERIFIED",
        provenance_id: str = "prov_unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AnnotationRevision:
        rev_num = len(self.revisions) + 1
        parent_id = self.revisions[-1].revision_id if self.revisions else None
        rev_id = f"rev_{self.sample_id}_{rev_num:03d}"

        rev = AnnotationRevision(
            revision_id=rev_id,
            sample_id=self.sample_id,
            revision_number=rev_num,
            parent_revision_id=parent_id,
            author_id=author_id,
            author_role=author_role,
            glosses=list(glosses),
            is_temporally_aligned=is_temporally_aligned,
            review_status=review_status,
            quality_grade=quality_grade,
            change_reason=change_reason,
            provenance_id=provenance_id,
            metadata=metadata or {},
        )
        self.revisions.append(rev)
        return rev

    def get_latest_revision(self) -> Optional[AnnotationRevision]:
        return self.revisions[-1] if self.revisions else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "total_revisions": len(self.revisions),
            "revisions": [r.to_dict() for r in self.revisions],
        }

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
