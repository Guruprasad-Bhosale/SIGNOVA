"""
Syntactic Label Normalization Utilities for SIGNOVA.

Performs deterministic syntactic normalization of sign language gloss tokens:
- Whitespace stripping and normalization.
- Unicode NFKC canonical normalization.
- Standardized uppercase casing rules.
- Punctuation stripping.

IMPORTANT RESEARCH RULE:
- Semantic merging, synonym mapping, English translation, or automatic gloss generation
  is STRICTLY PROHIBITED.
- The original raw label is always preserved alongside the normalized canonical label.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union


class LabelNormalizer:
    """
    Deterministic syntactic normalizer for sign/gloss labels.
    """

    def __init__(self, strip_punctuation: bool = True, uppercase: bool = True):
        self.strip_punctuation = strip_punctuation
        self.uppercase = uppercase

    def normalize_token(self, token: str) -> str:
        """
        Normalizes a single sign token string.

        Example:
            '  hello-world! ' -> 'HELLO-WORLD' or 'HELLO_WORLD'
        """
        if not isinstance(token, str):
            return str(token)

        # 1. Unicode NFKC Normalization
        text = unicodedata.normalize("NFKC", token).strip()

        # 2. Casing
        if self.uppercase:
            text = text.upper()

        # 3. Punctuation handling
        if self.strip_punctuation:
            # Replace whitespace/punctuation variations with underscores
            text = re.sub(r"[^\w\-]", "_", text)
            text = re.sub(r"_+", "_", text).strip("_")

        return text

    def normalize_sequence(self, sequence: Union[str, List[str]]) -> Tuple[List[str], List[str]]:
        """
        Normalizes a sequence of sign/gloss tokens.

        Returns:
            Tuple of (original_tokens, canonical_tokens).
        """
        if isinstance(sequence, str):
            # Parse comma or whitespace separated string
            raw_tokens = [t.strip() for t in re.split(r"[,\s]+", sequence) if t.strip()]
        elif isinstance(sequence, (list, tuple)):
            raw_tokens = [str(t).strip() for t in sequence if str(t).strip()]
        else:
            raw_tokens = []

        canonical_tokens = [self.normalize_token(t) for t in raw_tokens if self.normalize_token(t)]
        return raw_tokens, canonical_tokens
