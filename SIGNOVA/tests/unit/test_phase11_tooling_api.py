"""Unit tests for Phase 11 Human Annotation Platform API."""

import pytest
from fastapi.testclient import TestClient
from apps.annotation.main import app, DRAFT_STORAGE_DIR, ANNOTATION_STORAGE_DIR
from signova.annotation.constants import (
    REVIEW_STATE_ANNOTATION_IN_PROGRESS,
    REVIEW_STATE_REVIEW_PENDING,
    REVIEW_STATE_VERIFIED,
    REVIEW_STATE_REJECTED,
)

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_test_dirs():
    DRAFT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Cleanup test files created during test
    for f in DRAFT_STORAGE_DIR.glob("test_sample_*.json"):
        f.unlink(missing_ok=True)
    for f in ANNOTATION_STORAGE_DIR.glob("test_sample_*.json"):
        f.unlink(missing_ok=True)


def test_list_videos_endpoint():
    resp = client.get("/api/videos")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "videos" in data
    assert isinstance(data["videos"], list)


def test_save_draft_endpoint():
    sample_id = "test_sample_001"
    payload = {
        "sample_id": sample_id,
        "annotator_id": "annot_01",
        "glosses": ["NAMASTE", "HELP"],
        "english_translation": "Greetings, need help",
        "notes": "Clear signing"
    }
    resp = client.post("/api/annotations/draft", json=payload)
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "DRAFT_SAVED"
    assert res["annotation"]["glosses"] == ["NAMASTE", "HELP"]
    assert res["annotation"]["review_status"] == REVIEW_STATE_ANNOTATION_IN_PROGRESS
    assert res["annotation"]["training_eligible"] is False


def test_get_annotation_workflow():
    sample_id = "test_sample_002"
    # Initially NEW
    resp = client.get(f"/api/annotations/{sample_id}")
    assert resp.status_code == 200
    assert resp.json()["type"] == "NEW"

    # Save draft
    client.post("/api/annotations/draft", json={
        "sample_id": sample_id,
        "annotator_id": "annot_01",
        "glosses": ["TEACHER", "STUDENT"],
    })

    # Now DRAFT
    resp2 = client.get(f"/api/annotations/{sample_id}")
    assert resp2.status_code == 200
    assert resp2.json()["type"] == "DRAFT"
    assert resp2.json()["data"]["glosses"] == ["TEACHER", "STUDENT"]


def test_submit_and_review_workflow():
    sample_id = "test_sample_003"
    # 1. Save draft
    client.post("/api/annotations/draft", json={
        "sample_id": sample_id,
        "annotator_id": "annot_01",
        "glosses": ["HOUSE", "BIG"],
    })

    # 2. Submit
    submit_resp = client.post("/api/annotations/submit", json={
        "sample_id": sample_id,
        "annotator_id": "annot_01",
    })
    assert submit_resp.status_code == 200
    sub_data = submit_resp.json()
    assert sub_data["status"] == "SUBMITTED_FOR_REVIEW"
    assert sub_data["annotation"]["review_status"] == REVIEW_STATE_REVIEW_PENDING
    assert sub_data["annotation"]["training_eligible"] is False

    # Draft should have been moved
    assert not (DRAFT_STORAGE_DIR / f"{sample_id}.json").exists()
    assert (ANNOTATION_STORAGE_DIR / f"{sample_id}.json").exists()

    # 3. Review - VERIFY
    review_resp = client.post("/api/annotations/review", json={
        "sample_id": sample_id,
        "reviewer_id": "rev_01",
        "decision": "VERIFY",
        "notes": "Accurate sequence",
        "is_linguist": True,
    })
    assert review_resp.status_code == 200
    rev_data = review_resp.json()
    assert rev_data["status"] == REVIEW_STATE_VERIFIED
    assert rev_data["training_eligible"] is True


def test_review_reject_workflow():
    sample_id = "test_sample_004"
    client.post("/api/annotations/draft", json={
        "sample_id": sample_id,
        "annotator_id": "annot_01",
        "glosses": ["WATER"],
    })
    client.post("/api/annotations/submit", json={
        "sample_id": sample_id,
        "annotator_id": "annot_01",
    })
    review_resp = client.post("/api/annotations/review", json={
        "sample_id": sample_id,
        "reviewer_id": "rev_01",
        "decision": "REJECT",
        "notes": "Handshape ambiguous",
    })
    assert review_resp.status_code == 200
    rev_data = review_resp.json()
    assert rev_data["status"] == REVIEW_STATE_REJECTED
    assert rev_data["training_eligible"] is False


def test_vocabulary_endpoint():
    resp = client.get("/api/vocabulary")
    assert resp.status_code == 200
    data = resp.json()
    assert "size" in data
    assert "reserved_tokens" in data
