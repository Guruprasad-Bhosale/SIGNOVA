"""
Phase 11 Lightweight Offline Human Annotation Server for SIGNOVA.

Supports:
- Video inspection and frame timeline
- Full session workflow: OPEN -> DRAFT -> SAVE -> RESUME -> SUBMIT -> REVIEW -> VERIFY/REJECT
- Canonical JSON export
- Strict rule: NO automatic ground truth gloss generation.
"""

from pathlib import Path
import json
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from signova.annotation.constants import (
    QUALITY_PARTIAL,
    QUALITY_UNVERIFIED,
    REVIEW_STATE_ANNOTATION_IN_PROGRESS,
    REVIEW_STATE_REJECTED,
    REVIEW_STATE_REVIEW_PENDING,
    REVIEW_STATE_UNANNOTATED,
    REVIEW_STATE_VERIFIED,
)
from signova.annotation.export import export_annotation_to_json, import_annotation_from_json
from signova.annotation.quality import calculate_annotation_quality_grade
from signova.annotation.review import AnnotationStateMachine
from signova.annotation.schema import TemporalSegment, VideoAnnotation
from signova.annotation.validation import evaluate_training_eligibility
from signova.annotation.vocabulary import ProjectGlossVocabulary

app = FastAPI(
    title="SIGNOVA Human Annotation Platform",
    description="Offline research tool for continuous Indian Sign Language (ISL) sequence annotation.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ANNOTATION_STORAGE_DIR = Path("data/annotations/phase11/human_gold")
DRAFT_STORAGE_DIR = Path("data/annotations/phase11/drafts")
ANNOTATION_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DRAFT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
async def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        from fastapi.responses import HTMLResponse
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return {"message": "SIGNOVA Human Annotation API", "status": "OPERATIONAL"}


class DraftPayload(BaseModel):
    sample_id: str
    annotator_id: str
    glosses: List[str]
    segments: Optional[List[Dict[str, Any]]] = None
    english_translation: Optional[str] = ""
    notes: Optional[str] = ""


class SubmitPayload(BaseModel):
    sample_id: str
    annotator_id: str


class ReviewPayload(BaseModel):
    sample_id: str
    reviewer_id: str
    decision: str  # "VERIFY" or "REJECT"
    notes: str = ""
    is_linguist: bool = True


@app.get("/api/videos")
async def list_videos():
    """Lists continuous ISL diagnostic videos available for annotation."""
    train_landmarks = list(Path("data/features/landmarks/train").glob("*.npz"))
    videos = [
        {
            "sample_id": p.stem,
            "video_file": f"{p.stem}.mp4",
            "landmark_file": p.name,
            "has_draft": (DRAFT_STORAGE_DIR / f"{p.stem}.json").exists(),
            "has_submitted": (ANNOTATION_STORAGE_DIR / f"{p.stem}.json").exists(),
        }
        for p in train_landmarks[:20]
    ]
    return {"total": len(videos), "videos": videos}


@app.get("/api/annotations/{sample_id}")
async def get_annotation(sample_id: str):
    """Fetches existing draft or verified annotation for a sample."""
    draft_file = DRAFT_STORAGE_DIR / f"{sample_id}.json"
    submitted_file = ANNOTATION_STORAGE_DIR / f"{sample_id}.json"

    if draft_file.exists():
        return {"type": "DRAFT", "data": json.loads(draft_file.read_text(encoding="utf-8"))}
    elif submitted_file.exists():
        return {"type": "SUBMITTED", "data": json.loads(submitted_file.read_text(encoding="utf-8"))}
    else:
        return {
            "type": "NEW",
            "data": {
                "sample_id": sample_id,
                "review_status": REVIEW_STATE_UNANNOTATED,
                "glosses": [],
                "segments": [],
                "english_translation": "",
            },
        }


@app.post("/api/annotations/draft")
async def save_draft(payload: DraftPayload):
    """Saves an in-progress draft (Session workflow: OPEN -> DRAFT -> SAVE)."""
    annot = VideoAnnotation(
        annotation_id=f"annot_{payload.sample_id}_{payload.annotator_id}",
        sample_id=payload.sample_id,
        annotator_id=payload.annotator_id,
        is_temporally_aligned=bool(payload.segments),
        glosses=[g.strip().upper() for g in payload.glosses if g.strip()],
        segments=[
            TemporalSegment(
                gloss=s["gloss"],
                start_frame=s.get("start_frame"),
                end_frame=s.get("end_frame"),
                confidence=s.get("confidence", "HIGH"),
                notes=s.get("notes", ""),
            )
            for s in (payload.segments or [])
        ],
        english_translation=payload.english_translation or "",
        review_status=REVIEW_STATE_ANNOTATION_IN_PROGRESS,
        quality_grade=QUALITY_PARTIAL if payload.glosses else QUALITY_UNVERIFIED,
        reviewer_notes=payload.notes or "",
        dataset_split="train",
        training_eligible=False,
    )
    draft_path = DRAFT_STORAGE_DIR / f"{payload.sample_id}.json"
    export_annotation_to_json(annot, draft_path)
    return {"status": "DRAFT_SAVED", "file": str(draft_path), "annotation": annot.to_dict()}


@app.post("/api/annotations/submit")
async def submit_annotation(payload: SubmitPayload):
    """Submits draft for reviewer verification (Session workflow: SAVE -> SUBMIT)."""
    draft_path = DRAFT_STORAGE_DIR / f"{payload.sample_id}.json"
    if not draft_path.exists():
        raise HTTPException(status_code=404, detail="No draft found to submit.")

    annot = import_annotation_from_json(draft_path)
    if not annot.glosses:
        raise HTTPException(status_code=400, detail="Cannot submit empty annotation.")

    annot.review_status = REVIEW_STATE_REVIEW_PENDING
    annot.quality_grade = calculate_annotation_quality_grade(annot)
    annot.training_eligible = False

    target_path = ANNOTATION_STORAGE_DIR / f"{payload.sample_id}.json"
    export_annotation_to_json(annot, target_path)
    draft_path.unlink(missing_ok=True)

    return {"status": "SUBMITTED_FOR_REVIEW", "file": str(target_path), "annotation": annot.to_dict()}


@app.post("/api/annotations/review")
async def review_annotation(payload: ReviewPayload):
    """Reviewer decision: VERIFY or REJECT (Session workflow: SUBMIT -> REVIEW -> VERIFY/REJECT)."""
    target_path = ANNOTATION_STORAGE_DIR / f"{payload.sample_id}.json"
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Annotation not found.")

    annot = import_annotation_from_json(target_path)

    if payload.decision == "VERIFY":
        AnnotationStateMachine.verify_by_reviewer(
            annot,
            reviewer_id=payload.reviewer_id,
            reviewer_notes=payload.notes,
            is_linguist=payload.is_linguist,
        )
        annot.quality_grade = calculate_annotation_quality_grade(annot, is_linguist_review=payload.is_linguist)
        evaluate_training_eligibility(annot)
    else:
        AnnotationStateMachine.reject_by_reviewer(
            annot,
            reviewer_id=payload.reviewer_id,
            rejection_reason=payload.notes,
        )

    export_annotation_to_json(annot, target_path)
    return {"status": annot.review_status, "training_eligible": annot.training_eligible, "annotation": annot.to_dict()}


@app.get("/api/vocabulary")
async def get_vocabulary():
    """Returns project vocabulary derived from genuine annotations."""
    vocab_file = Path("data/annotations/phase11/vocabulary.json")
    if vocab_file.exists():
        return json.loads(vocab_file.read_text(encoding="utf-8"))
    vocab = ProjectGlossVocabulary()
    return vocab.to_dict()


if __name__ == "__main__":
    import uvicorn
    print("Starting SIGNOVA Human Annotation Platform at http://127.0.0.1:8000 ...")
    uvicorn.run("apps.annotation.main:app", host="127.0.0.1", port=8000, reload=True)
