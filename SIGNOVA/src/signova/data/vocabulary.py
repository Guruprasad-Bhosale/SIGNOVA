"""
Sign Language Gloss Vocabulary and Token Management for SIGNOVA.

Manages token-to-index mappings for Connectionist Temporal Classification (CTC):
- Reserves BLANK token at index 0.
- Reserves optional UNK token at index 1.
- Tracks token frequencies across train, validation, and test splits.
- Validates Out-Of-Vocabulary (OOV) tokens without silent data corruption.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd


class SignVocabulary:
    """
    Vocabulary container for CTC sequential sign recognition.
    """

    BLANK_TOKEN = "<BLANK>"
    UNK_TOKEN = "<UNK>"

    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        blank_index: int = 0,
        unk_token: Optional[str] = "<UNK>",
    ):
        self.blank_index = blank_index
        self.unk_token = unk_token

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.frequencies: Dict[str, int] = {}
        self.split_frequencies: Dict[str, Dict[str, int]] = {
            "train": {}, "val": {}, "test": {}, "unassigned": {}
        }

        # Initialize special tokens
        self._add_token(self.BLANK_TOKEN, forced_id=self.blank_index)
        if self.unk_token:
            unk_id = 1 if self.blank_index == 0 else 0
            self._add_token(self.unk_token, forced_id=unk_id)

        if tokens:
            for t in tokens:
                self.add_token(t)

    def _add_token(self, token: str, forced_id: Optional[int] = None) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]

        if forced_id is not None:
            t_id = forced_id
        else:
            t_id = len(self.token_to_id)
            while t_id in self.id_to_token:
                t_id += 1

        self.token_to_id[token] = t_id
        self.id_to_token[t_id] = token
        self.frequencies[token] = 0
        return t_id

    def add_token(self, token: str, split: Optional[str] = None) -> int:
        t_id = self._add_token(token)
        self.frequencies[token] = self.frequencies.get(token, 0) + 1

        if split:
            sp = split if split in self.split_frequencies else "unassigned"
            self.split_frequencies[sp][token] = self.split_frequencies[sp].get(token, 0) + 1

        return t_id

    def encode(self, sequence: List[str], allow_oov: bool = True) -> List[int]:
        """
        Converts token sequence to list of integer IDs.
        """
        ids = []
        for t in sequence:
            if t in self.token_to_id:
                ids.append(self.token_to_id[t])
            elif allow_oov and self.unk_token in self.token_to_id:
                ids.append(self.token_to_id[self.unk_token])
            else:
                raise KeyError(f"Out-of-vocabulary token '{t}' encountered.")
        return ids

    def decode(self, ids: List[int], remove_blank: bool = True) -> List[str]:
        """
        Converts integer IDs to token strings.
        """
        tokens = []
        for i in ids:
            if remove_blank and i == self.blank_index:
                continue
            tokens.append(self.id_to_token.get(i, self.unk_token or f"<ID_{i}>"))
        return tokens

    def __len__(self) -> int:
        return len(self.token_to_id)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for t, tid in sorted(self.token_to_id.items(), key=lambda x: x[1]):
            rows.append({
                "token_id": tid,
                "token": t,
                "total_frequency": self.frequencies.get(t, 0),
                "train_frequency": self.split_frequencies["train"].get(t, 0),
                "val_frequency": self.split_frequencies["val"].get(t, 0),
                "test_frequency": self.split_frequencies["test"].get(t, 0),
            })
        return pd.DataFrame(rows)

    def save_csv(self, filepath: Union[str, Path]):
        df = self.to_dataframe()
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    @classmethod
    def load_csv(cls, filepath: Union[str, Path]) -> "SignVocabulary":
        df = pd.read_csv(filepath)
        vocab = cls()
        for _, row in df.iterrows():
            t = str(row["token"])
            tid = int(row["token_id"])
            vocab.token_to_id[t] = tid
            vocab.id_to_token[tid] = t
            vocab.frequencies[t] = int(row.get("total_frequency", 0))
        return vocab
