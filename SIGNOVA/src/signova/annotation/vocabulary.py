"""
Phase 11 Versioned Gloss Vocabulary Manager for SIGNOVA.

Enforces:
1. Vocabulary originates solely from genuine human annotations (never from English sentence words).
2. Reserved tokens: <BLANK> = 0, <UNK> = 1.
3. Preserves token index determinism across versions.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from signova.annotation.constants import (
    BLANK_ID,
    BLANK_TOKEN,
    UNK_ID,
    UNK_TOKEN,
)
from signova.annotation.schema import VideoAnnotation


@dataclass
class VocabularyEntry:
    token_id: int
    gloss: str
    first_seen_annotation_id: str
    first_seen_sample_id: str
    frequency: int = 1
    aliases: List[str] = field(default_factory=list)
    status: str = "VERIFIED_HUMAN_ORIGIN"  # Never synthetic

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "gloss": self.gloss,
            "first_seen_annotation_id": self.first_seen_annotation_id,
            "first_seen_sample_id": self.first_seen_sample_id,
            "frequency": self.frequency,
            "aliases": self.aliases,
            "status": self.status,
        }


class ProjectGlossVocabulary:
    """
    Manages project-wide lexical sign gloss vocabulary.
    """

    def __init__(self, version: str = "0.1.0"):
        self.version = version
        self.token_to_id: Dict[str, int] = {
            BLANK_TOKEN: BLANK_ID,
            UNK_TOKEN: UNK_ID,
        }
        self.id_to_token: Dict[int, str] = {
            BLANK_ID: BLANK_TOKEN,
            UNK_ID: UNK_TOKEN,
        }
        self.entries: Dict[str, VocabularyEntry] = {}

    def add_annotation_tokens(self, annotation: VideoAnnotation) -> List[str]:
        """Adds all novel gloss tokens from a verified human annotation."""
        new_tokens = []
        for gloss in annotation.glosses:
            norm_gloss = gloss.strip().upper()
            if not norm_gloss:
                continue

            if norm_gloss not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[norm_gloss] = new_id
                self.id_to_token[new_id] = norm_gloss
                self.entries[norm_gloss] = VocabularyEntry(
                    token_id=new_id,
                    gloss=norm_gloss,
                    first_seen_annotation_id=annotation.annotation_id,
                    first_seen_sample_id=annotation.sample_id,
                    frequency=1,
                )
                new_tokens.append(norm_gloss)
            else:
                if norm_gloss in self.entries:
                    self.entries[norm_gloss].frequency += 1

        return new_tokens

    def get_token_id(self, gloss: str) -> int:
        norm = gloss.strip().upper()
        return self.token_to_id.get(norm, UNK_ID)

    def get_token(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, UNK_TOKEN)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocabulary_version": self.version,
            "size": len(self.token_to_id),
            "reserved_tokens": {
                BLANK_TOKEN: BLANK_ID,
                UNK_TOKEN: UNK_ID,
            },
            "entries": [e.to_dict() for e in self.entries.values()],
        }

    def save_to_file(self, file_path: Path | str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_from_file(cls, file_path: Path | str) -> "ProjectGlossVocabulary":
        path = Path(file_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls(version=data.get("vocabulary_version", "0.1.0"))
        for entry_dict in data.get("entries", []):
            t_id = entry_dict["token_id"]
            gloss = entry_dict["gloss"]
            vocab.token_to_id[gloss] = t_id
            vocab.id_to_token[t_id] = gloss
            vocab.entries[gloss] = VocabularyEntry(
                token_id=t_id,
                gloss=gloss,
                first_seen_annotation_id=entry_dict["first_seen_annotation_id"],
                first_seen_sample_id=entry_dict["first_seen_sample_id"],
                frequency=entry_dict.get("frequency", 1),
                aliases=entry_dict.get("aliases", []),
                status=entry_dict.get("status", "VERIFIED_HUMAN_ORIGIN"),
            )
        return vocab
