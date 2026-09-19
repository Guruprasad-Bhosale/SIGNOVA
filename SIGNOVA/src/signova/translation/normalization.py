"""
Deterministic text and token normalization for Gloss-to-English translation.

Guiding Principles:
1. No semantic rewriting, synonym substitution, or hallucinated corrections.
2. Deterministic whitespace, Unicode normalization, and punctuation cleaning.
3. Preserves both raw and normalized representations.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple, Union

Sequence_or_List = Union[List[str], Tuple[str, ...], Sequence[str]]


class TranslationNormalizer:
    """
    Deterministic normalizer for translation inputs (glosses) and outputs (English text).
    """

    def __init__(
        self,
        lowercase_target: bool = True,
        normalize_unicode: bool = True,
        strip_redundant_whitespace: bool = True,
        standardize_punctuation: bool = True,
        remove_punctuation_for_eval: bool = False,
    ):
        self.lowercase_target = lowercase_target
        self.normalize_unicode = normalize_unicode
        self.strip_redundant_whitespace = strip_redundant_whitespace
        self.standardize_punctuation = standardize_punctuation
        self.remove_punctuation_for_eval = remove_punctuation_for_eval

    def normalize_gloss_token(self, token: str) -> str:
        """
        Normalize an individual gloss token.
        Preserves uppercase convention for sign glosses while stripping extraneous noise.
        """
        if not isinstance(token, str):
            token = str(token)
        if self.normalize_unicode:
            token = unicodedata.normalize("NFC", token)
        token = token.strip().upper()
        # Collapse multiple spaces or dashes
        token = re.sub(r"\s+", " ", token)
        return token

    def normalize_gloss_sequence(self, glosses: Sequence_or_List) -> List[str]:
        """
        Normalize an ordered sequence of gloss tokens.
        Filters out empty tokens while preserving sign order.
        """
        normalized = []
        for g in glosses:
            norm_g = self.normalize_gloss_token(g)
            if norm_g:
                normalized.append(norm_g)
        return normalized

    def normalize_english_text(self, text: str) -> str:
        """
        Normalize English target text deterministically.
        """
        if not isinstance(text, str):
            text = str(text)

        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        # Standardize quotes and dashes
        if self.standardize_punctuation:
            text = text.replace("“", '"').replace("”", '"')
            text = text.replace("‘", "'").replace("’", "'")
            text = text.replace("—", "-").replace("–", "-")

        if self.lowercase_target:
            text = text.lower()

        if self.remove_punctuation_for_eval:
            text = re.sub(r"[^\w\s]", "", text)

        if self.strip_redundant_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def process_pair(
        self,
        source_glosses: List[str],
        target_text: str,
    ) -> Dict[str, Union[List[str], str]]:
        """
        Process a source-target pair and return both raw and normalized values.
        """
        return {
            "raw_source_tokens": list(source_glosses),
            "normalized_source_tokens": self.normalize_gloss_sequence(source_glosses),
            "raw_target_text": target_text,
            "normalized_target_text": self.normalize_english_text(target_text),
        }


# Type annotation helper
Sequence_or_List = Union[List[str], Tuple[str, ...]]
