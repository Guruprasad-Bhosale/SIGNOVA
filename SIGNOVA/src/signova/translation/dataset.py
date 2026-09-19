"""
Dataset abstractions and collation utilities for Gloss-to-English translation.

Guiding Principles:
1. Clean separation: Source is discrete ISL gloss tokens; Target is English text.
2. Translation layer NEVER accepts video or landmark features.
3. Unknown metadata fields remain None (no manufactured signer IDs).
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


@dataclass
class TranslationSample:
    """
    Canonical representation of a translation sample.
    
    Attributes:
        sample_id: Unique identifier for the sample.
        source_tokens: List of discrete gloss/token strings (e.g., ['I', 'GO', 'COLLEGE']).
        target_text: English target sentence (raw or normalized).
        signer_id: Signer identifier if verified; otherwise None.
        session_id: Session identifier if verified; otherwise None.
        split: Dataset split ('train', 'val', 'test', etc.).
        provenance: Origin dataset or fixture name.
        license: Dataset license or 'Synthetic'.
        source_type: Type of source representation (e.g., 'isl_gloss', 'synthetic_gloss').
        target_type: Type of target representation (e.g., 'english_text').
    """
    sample_id: str
    source_tokens: List[str]
    target_text: str
    signer_id: Optional[str] = None
    session_id: Optional[str] = None
    split: str = "train"
    provenance: str = "unknown"
    license: str = "unknown"
    source_type: str = "isl_gloss"
    target_type: str = "english_text"

    def __post_init__(self):
        if not isinstance(self.source_tokens, list):
            self.source_tokens = list(self.source_tokens)
        if not isinstance(self.target_text, str):
            self.target_text = str(self.target_text)


class GlossTranslationDataset(Dataset):
    """
    PyTorch Dataset wrapping a sequence of TranslationSample objects.
    """

    def __init__(
        self,
        samples: Sequence[TranslationSample],
        source_vocab: Optional[Any] = None,
        target_vocab: Optional[Any] = None,
        target_normalizer: Optional[Callable[[str], str]] = None,
        max_source_len: int = 64,
        max_target_len: int = 64,
    ):
        self.samples = list(samples)
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.target_normalizer = target_normalizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        source_tokens = sample.source_tokens[:self.max_source_len]
        
        target_text = sample.target_text
        if self.target_normalizer is not None:
            target_text = self.target_normalizer(target_text)

        item: Dict[str, Any] = {
            "sample_id": sample.sample_id,
            "source_tokens": source_tokens,
            "target_text": target_text,
            "signer_id": sample.signer_id,
            "session_id": sample.session_id,
            "split": sample.split,
        }

        if self.source_vocab is not None:
            source_ids = self.source_vocab.encode(source_tokens, add_bos=False, add_eos=False)
            item["source_ids"] = torch.tensor(source_ids, dtype=torch.long)
            item["source_length"] = len(source_ids)

        if self.target_vocab is not None:
            target_ids = self.target_vocab.encode(
                target_text,
                add_bos=True,
                add_eos=True,
                max_len=self.max_target_len,
            )
            item["target_ids"] = torch.tensor(target_ids, dtype=torch.long)
            item["target_length"] = len(target_ids)

        return item


class TranslationPadCollate:
    """
    Collate function for dynamic batching of variable-length translation sequences.
    """

    def __init__(
        self,
        source_pad_id: int = 0,
        target_pad_id: int = 0,
    ):
        self.source_pad_id = source_pad_id
        self.target_pad_id = target_pad_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_ids = [item["sample_id"] for item in batch]
        source_tokens = [item["source_tokens"] for item in batch]
        target_texts = [item["target_text"] for item in batch]
        splits = [item.get("split", "train") for item in batch]

        output: Dict[str, Any] = {
            "sample_ids": sample_ids,
            "source_tokens": source_tokens,
            "target_texts": target_texts,
            "splits": splits,
        }

        if "source_ids" in batch[0]:
            source_ids_list = [item["source_ids"] for item in batch]
            source_lengths = torch.tensor([item["source_length"] for item in batch], dtype=torch.long)
            padded_source = torch.nn.utils.rnn.pad_sequence(
                source_ids_list,
                batch_first=True,
                padding_value=self.source_pad_id,
            )
            output["source_ids"] = padded_source
            output["source_lengths"] = source_lengths
            # Source mask: True for valid tokens, False for padding
            output["source_mask"] = (padded_source != self.source_pad_id)

        if "target_ids" in batch[0]:
            target_ids_list = [item["target_ids"] for item in batch]
            target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)
            padded_target = torch.nn.utils.rnn.pad_sequence(
                target_ids_list,
                batch_first=True,
                padding_value=self.target_pad_id,
            )
            output["target_ids"] = padded_target
            output["target_lengths"] = target_lengths
            output["target_mask"] = (padded_target != self.target_pad_id)

        return output
