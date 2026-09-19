"""
Phase 11 Canonical JSON and Optional EAF Export/Import Module for SIGNOVA.

Canonical format: JSON
Optional export: EAF (ELAN XML)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET
from signova.annotation.schema import (
    AnnotationSet,
    TemporalSegment,
    VideoAnnotation,
)


def export_annotation_to_json(annotation: VideoAnnotation, output_path: Path | str) -> str:
    """Exports canonical VideoAnnotation to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(annotation.to_dict(), indent=2)
    path.write_text(json_str, encoding="utf-8")
    return str(path)


def import_annotation_from_json(json_path: Path | str) -> VideoAnnotation:
    """Loads a VideoAnnotation from a canonical JSON file."""
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        TemporalSegment(
            gloss=s["gloss"],
            start_frame=s.get("start_frame"),
            end_frame=s.get("end_frame"),
            start_time_ms=s.get("start_time_ms"),
            end_time_ms=s.get("end_time_ms"),
            confidence=s.get("confidence", "HIGH"),
            notes=s.get("notes", ""),
        )
        for s in data.get("segments", [])
    ]
    return VideoAnnotation(
        annotation_id=data["annotation_id"],
        sample_id=data["sample_id"],
        annotator_id=data["annotator_id"],
        is_temporally_aligned=data.get("is_temporally_aligned", False),
        glosses=data.get("glosses", []),
        segments=segments,
        english_translation=data.get("english_translation", ""),
        review_status=data.get("review_status", "UNANNOTATED"),
        quality_grade=data.get("quality_grade", "UNVERIFIED"),
        reviewer_id=data.get("reviewer_id"),
        reviewer_notes=data.get("reviewer_notes", ""),
        dataset_split=data.get("dataset_split", "unassigned"),
        training_eligible=data.get("training_eligible", False),
        provenance_id=data.get("provenance_id", "prov_unknown"),
        version=data.get("version", "0.1.0"),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        metadata=data.get("metadata", {}),
    )


def export_annotation_to_eaf(annotation: VideoAnnotation, output_path: Path | str) -> str:
    """
    Optional helper: exports canonical annotation to ELAN EAF XML structure.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("ANNOTATION_DOCUMENT", {
        "AUTHOR": annotation.annotator_id,
        "DATE": annotation.updated_at,
        "VERSION": "3.0",
        "FORMAT": "3.0",
    })

    header = ET.SubElement(root, "HEADER", {"MEDIA_FILE": f"{annotation.sample_id}.mp4", "TIME_UNITS": "milliseconds"})
    time_order = ET.SubElement(root, "TIME_ORDER")

    tier_gloss = ET.SubElement(root, "TIER", {"TIER_ID": "Sign-Gloss", "LINGUISTIC_TYPE_REF": "default-lt"})

    for i, gloss in enumerate(annotation.glosses):
        t1_id = f"ts{2*i+1}"
        t2_id = f"ts{2*i+2}"
        t1_val = str(i * 1000)
        t2_val = str((i + 1) * 1000)

        ET.SubElement(time_order, "TIME_SLOT", {"TIME_SLOT_ID": t1_id, "TIME_VALUE": t1_val})
        ET.SubElement(time_order, "TIME_SLOT", {"TIME_SLOT_ID": t2_id, "TIME_VALUE": t2_val})

        annot_elem = ET.SubElement(tier_gloss, "ANNOTATION")
        alignable = ET.SubElement(annot_elem, "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": f"a{i+1}",
            "TIME_SLOT_REF1": t1_id,
            "TIME_SLOT_REF2": t2_id,
        })
        val_elem = ET.SubElement(alignable, "ANNOTATION_VALUE")
        val_elem.text = gloss

    tree = ET.ElementTree(root)
    tree.write(str(path), encoding="utf-8", xml_declaration=True)
    return str(path)
