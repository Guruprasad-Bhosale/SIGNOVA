"""
Phase 12 Real ISL Gloss Vocabulary Manager for SIGNOVA.

Constructs vocabulary strictly from training-eligible human annotations.
Preserves <BLANK>=0, <UNK>=1.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from signova.annotation.schema import VideoAnnotation
from signova.qualification.constants import BLANK_ID, BLANK_TOKEN, UNK_ID, UNK_TOKEN


class Phase12GlossVocabulary:
    """Manages Phase 12 ISL gloss vocabulary with singleton and OOV diagnostics."""

    def __init__(self, version: str = "0.1.0"):
        self.version = version
        self.token_to_id: Dict[str, int] = {BLANK_TOKEN: BLANK_ID, UNK_TOKEN: UNK_ID}
        self.id_to_token: Dict[int, str] = {BLANK_ID: BLANK_TOKEN, UNK_ID: UNK_TOKEN}
        self.frequencies: Dict[str, int] = {}
        self.split_vocabularies: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    @classmethod
    def build_from_annotations(
        cls,
        annotations: List[VideoAnnotation],
        version: str = "0.1.0",
    ) -> "Phase12GlossVocabulary":
        """Builds vocabulary using training-eligible samples."""
        vocab = cls(version=version)
        counts: Counter = Counter()

        for a in annotations:
            split = a.dataset_split
            glosses = [g.strip().upper() for g in a.glosses if g.strip()]

            if split in vocab.split_vocabularies:
                vocab.split_vocabularies[split].update(glosses)

            # Build vocab entries from train split (or all eligible if unassigned)
            if split == "train" or split == "unassigned":
                for g in glosses:
                    counts[g] += 1

        # Assign IDs to train tokens
        for token, count in sorted(counts.items()):
            if token not in vocab.token_to_id:
                new_id = len(vocab.token_to_id)
                vocab.token_to_id[token] = new_id
                vocab.id_to_token[new_id] = token
            vocab.frequencies[token] = count

        return vocab

    def encode(self, glosses: List[str]) -> List[int]:
        """Encodes gloss list to token IDs, substituting UNK_ID for out-of-vocabulary tokens."""
        return [self.token_to_id.get(g.strip().upper(), UNK_ID) for g in glosses]

    def decode(self, token_ids: List[int], remove_blank: bool = True) -> List[str]:
        """Decodes token IDs back to gloss strings."""
        result = []
        for tid in token_ids:
            if remove_blank and tid == BLANK_ID:
                continue
            token = self.id_to_token.get(tid, UNK_TOKEN)
            result.append(token)
        return result

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def get_singletons(self) -> List[str]:
        return [t for t, c in self.frequencies.items() if c == 1]

    def get_overlap_diagnostics(self) -> Dict[str, Any]:
        train_v = self.split_vocabularies["train"]
        val_v = self.split_vocabularies["val"]
        test_v = self.split_vocabularies["test"]

        val_oov = list(val_v - train_v) if train_v else []
        test_oov = list(test_v - train_v) if train_v else []

        return {
            "train_unique_tokens": len(train_v),
            "val_unique_tokens": len(val_v),
            "test_unique_tokens": len(test_v),
            "val_oov_count": len(val_oov),
            "test_oov_count": len(test_oov),
            "val_oov_tokens": val_oov,
            "test_oov_tokens": test_oov,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "size": self.size,
            "reserved_tokens": {BLANK_TOKEN: BLANK_ID, UNK_TOKEN: UNK_ID},
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "frequencies": self.frequencies,
            "singleton_count": len(self.get_singletons()),
            "overlap_diagnostics": self.get_overlap_diagnostics(),
        }

    def save(self, json_path: Path) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
