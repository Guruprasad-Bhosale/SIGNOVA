"""
Vocabulary management for Source Gloss Sequences and Target English Sentences.

Guiding Principles:
1. Complete separation of Source Gloss Vocabulary and Target English Vocabulary.
2. Explicit special tokens (<PAD>, <BOS>, <EOS>, <UNK>).
3. OOV rate tracking and deterministic serialization.
4. No fake real-world vocabularies constructed under State C.
"""

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union


class TranslationVocabulary:
    """
    Vocabulary representation mapping discrete string tokens to integer IDs and vice versa.
    """

    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(
        self,
        name: str = "vocab",
        is_synthetic: bool = False,
        min_freq: int = 1,
    ):
        self.name = name
        self.is_synthetic = is_synthetic
        self.min_freq = min_freq

        self.special_tokens = [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
        ]

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.frequencies: Dict[str, int] = {}

        # Initialize special tokens
        for idx, token in enumerate(self.special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            self.frequencies[token] = 0

        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.bos_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_id = self.token_to_id[self.EOS_TOKEN]
        self.unk_id = self.token_to_id[self.UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def add_token(self, token: str, count: int = 1) -> int:
        """Add a token to the vocabulary or increment its count."""
        if token in self.token_to_id:
            self.frequencies[token] += count
            return self.token_to_id[token]

        token_id = len(self.token_to_id)
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self.frequencies[token] = count
        return token_id

    def build_from_sequences(
        self,
        token_sequences: Iterable[Sequence[str]],
    ) -> "TranslationVocabulary":
        """Build vocabulary from an iterable of token lists."""
        counter: Counter = Counter()
        for seq in token_sequences:
            counter.update(seq)

        # Add tokens meeting min_freq
        for token, freq in counter.most_common():
            if token not in self.token_to_id and freq >= self.min_freq:
                self.add_token(token, count=freq)
            elif token in self.token_to_id:
                self.frequencies[token] += freq

        return self

    def encode(
        self,
        tokens: Union[Sequence[str], str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: Optional[int] = None,
    ) -> List[int]:
        """
        Encode a sequence of tokens into integer IDs.
        If a single string is passed, it is split by whitespace.
        """
        if isinstance(tokens, str):
            token_list = self.tokenize_text(tokens)
        else:
            token_list = list(tokens)

        encoded = []
        if add_bos:
            encoded.append(self.bos_id)

        for token in token_list:
            encoded.append(self.token_to_id.get(token, self.unk_id))

        if add_eos:
            encoded.append(self.eos_id)

        if max_len is not None:
            encoded = encoded[:max_len]

        return encoded

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special: bool = True,
        stop_at_eos: bool = True,
    ) -> List[str]:
        """
        Decode a sequence of integer IDs into tokens.
        """
        tokens = []
        for tid in token_ids:
            if stop_at_eos and tid == self.eos_id:
                break
            if skip_special and tid in (self.pad_id, self.bos_id, self.eos_id):
                continue
            token_str = self.id_to_token.get(tid, self.UNK_TOKEN)
            tokens.append(token_str)
        return tokens

    def decode_to_text(
        self,
        token_ids: Sequence[int],
        skip_special: bool = True,
        stop_at_eos: bool = True,
    ) -> str:
        """Decode IDs directly into a space-joined string."""
        tokens = self.decode(token_ids, skip_special=skip_special, stop_at_eos=stop_at_eos)
        return " ".join(tokens)

    def calculate_oov_rate(self, token_sequences: Iterable[Sequence[str]]) -> float:
        """Calculate Out-Of-Vocabulary rate on a sequence dataset."""
        total_tokens = 0
        oov_tokens = 0
        for seq in token_sequences:
            if isinstance(seq, str):
                seq = self.tokenize_text(seq)
            for token in seq:
                total_tokens += 1
                if token not in self.token_to_id:
                    oov_tokens += 1
        return (oov_tokens / total_tokens) if total_tokens > 0 else 0.0

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """Simple whitespace and punctuation-aware tokenization for English text."""
        # Separate punctuation into tokens
        text = re.sub(r"([.,!?;:\'\"()\[\]])", r" \1 ", text)
        return [t for t in text.strip().split() if t]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize vocabulary to dictionary."""
        return {
            "name": self.name,
            "is_synthetic": self.is_synthetic,
            "min_freq": self.min_freq,
            "vocab_size": len(self.token_to_id),
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
            "frequencies": self.frequencies,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranslationVocabulary":
        """Load vocabulary from dictionary."""
        vocab = cls(
            name=data.get("name", "vocab"),
            is_synthetic=data.get("is_synthetic", False),
            min_freq=data.get("min_freq", 1),
        )
        vocab.token_to_id = data["token_to_id"]
        vocab.id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        vocab.frequencies = data.get("frequencies", {})
        vocab.pad_id = vocab.token_to_id[vocab.PAD_TOKEN]
        vocab.bos_id = vocab.token_to_id[vocab.BOS_TOKEN]
        vocab.eos_id = vocab.token_to_id[vocab.EOS_TOKEN]
        vocab.unk_id = vocab.token_to_id[vocab.UNK_TOKEN]
        return vocab

    def save(self, filepath: Union[str, Path]) -> None:
        """Save vocabulary to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TranslationVocabulary":
        """Load vocabulary from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
