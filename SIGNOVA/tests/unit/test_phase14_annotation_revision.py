"""
Phase 14 Non-Destructive Annotation Revision Unit Tests.
"""

from signova.operations.revision import AnnotationRevisionHistory


def test_annotation_revision_history_preservation():
    history = AnnotationRevisionHistory(sample_id="sample_rev_01")

    # 1. Initial draft revision
    r1 = history.add_revision(
        author_id="annot_01",
        author_role="ANNOTATOR",
        glosses=["HELLO"],
        change_reason="Initial draft",
    )
    assert r1.revision_number == 1
    assert r1.parent_revision_id is None

    # 2. Reviewer correction revision (non-destructive)
    r2 = history.add_revision(
        author_id="reviewer_01",
        author_role="REVIEWER",
        glosses=["HELLO", "FRIEND"],
        change_reason="Added omitted compound element",
    )
    assert r2.revision_number == 2
    assert r2.parent_revision_id == r1.revision_id
    assert len(history.revisions) == 2

    # Verify history maintains both versions intact
    assert history.revisions[0].glosses == ["HELLO"]
    assert history.revisions[1].glosses == ["HELLO", "FRIEND"]
