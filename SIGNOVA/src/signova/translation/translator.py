"""
Production-facing Gloss-to-English Translator API and execution wrappers.

Guiding Principles:
1. Strict contract: Input is an already-recognized gloss sequence (list[str]).
2. Translation module NEVER accepts video or landmarks and NEVER performs visual recognition.
3. Deterministic handling of empty, unknown, malformed, and out-of-vocabulary gloss inputs.
4. Clean translate_gloss_sequence(gloss_tokens) top-level convenience interface.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import torch

from signova.translation.decoding import BeamSearchDecoder, GreedyDecoder
from signova.translation.model import SyntheticGlossToEnglishSeq2Seq
from signova.translation.normalization import TranslationNormalizer
from signova.translation.vocabulary import TranslationVocabulary

GlossSequence = List[str]


class GlossToEnglishTranslator:
    """
    Production-facing translation engine converting discrete ISL gloss sequences into English text.
    """

    def __init__(
        self,
        model: Optional[SyntheticGlossToEnglishSeq2Seq] = None,
        source_vocab: Optional[TranslationVocabulary] = None,
        target_vocab: Optional[TranslationVocabulary] = None,
        normalizer: Optional[TranslationNormalizer] = None,
        max_source_len: int = 64,
        max_target_len: int = 64,
        device: str = "cpu",
    ):
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.model = model.to(self.device) if model is not None else None
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.normalizer = normalizer or TranslationNormalizer()
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

        self.greedy_decoder = GreedyDecoder(max_len=max_target_len)
        self.beam_decoder = BeamSearchDecoder(max_len=max_target_len, beam_width=4)
        self.is_loaded = (self.model is not None and self.source_vocab is not None and self.target_vocab is not None)

    @classmethod
    def load_from_directory(
        cls,
        model_dir: Union[str, Path],
        device: str = "cpu",
    ) -> "GlossToEnglishTranslator":
        """
        Load translation model, source vocabulary, and target vocabulary from a directory.
        """
        model_dir = Path(model_dir)
        ckpt_path = model_dir / "model_checkpoint.pt"
        src_vocab_path = model_dir / "source_vocab.json"
        tgt_vocab_path = model_dir / "target_vocab.json"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        if not src_vocab_path.exists():
            raise FileNotFoundError(f"Source vocabulary not found at: {src_vocab_path}")
        if not tgt_vocab_path.exists():
            raise FileNotFoundError(f"Target vocabulary not found at: {tgt_vocab_path}")

        source_vocab = TranslationVocabulary.load(src_vocab_path)
        target_vocab = TranslationVocabulary.load(tgt_vocab_path)
        model = SyntheticGlossToEnglishSeq2Seq.load_checkpoint(ckpt_path, device=device)

        return cls(
            model=model,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            device=device,
        )

    def translate_glosses(
        self,
        glosses: Sequence[str],
        beam_width: int = 1,
    ) -> str:
        """
        Translate a sequence of discrete ISL gloss tokens to natural English text.

        Args:
            glosses: List of strings (e.g., ['I', 'GO', 'COLLEGE']).
            beam_width: Beam search width (1 = Greedy decoding).

        Returns:
            Translated English sentence.
        """
        # 1. Edge Case: Empty sequence
        if not glosses or len(glosses) == 0:
            return ""

        # 2. Normalize gloss sequence
        norm_glosses = self.normalizer.normalize_gloss_sequence(glosses)
        if not norm_glosses:
            return ""

        # 3. Truncation guard for excessively long sequence
        if len(norm_glosses) > self.max_source_len:
            norm_glosses = norm_glosses[:self.max_source_len]

        if not self.is_loaded or self.model is None or self.source_vocab is None or self.target_vocab is None:
            # Fallback if uninitialized
            return " ".join(g.lower() for g in norm_glosses)

        # 4. Check for all-unknown sequence
        encoded_ids = self.source_vocab.encode(norm_glosses, add_bos=False, add_eos=False)
        unk_count = sum(1 for tid in encoded_ids if tid == self.source_vocab.unk_id)
        if unk_count == len(encoded_ids):
            # All tokens are unknown; return clean fallback
            return ""

        # 5. Model decoding
        src_tensor = torch.tensor([encoded_ids], dtype=torch.long, device=self.device)
        src_mask = (src_tensor != self.source_vocab.pad_id)
        src_len = torch.tensor([len(encoded_ids)], dtype=torch.long, device=self.device)

        if beam_width > 1:
            self.beam_decoder.beam_width = beam_width
            result = self.beam_decoder.decode_single(
                model=self.model,
                source_ids=src_tensor,
                target_vocab=self.target_vocab,
                source_lengths=src_len,
                source_mask=src_mask,
            )
        else:
            result = self.greedy_decoder.decode_single(
                model=self.model,
                source_ids=src_tensor,
                target_vocab=self.target_vocab,
                source_lengths=src_len,
                source_mask=src_mask,
            )

        return result.text

    def get_status(self) -> Dict[str, Any]:
        """Return translator health and configuration status."""
        return {
            "module": "GlossToEnglishTranslator",
            "is_loaded": self.is_loaded,
            "device": self.device,
            "parameter_count": self.model.count_parameters() if self.model else 0,
            "is_synthetic_model": getattr(self.model, "is_synthetic", True) if self.model else True,
            "source_vocab_size": len(self.source_vocab) if self.source_vocab else 0,
            "target_vocab_size": len(self.target_vocab) if self.target_vocab else 0,
            "max_source_len": self.max_source_len,
            "max_target_len": self.max_target_len,
        }


# Global default translator instance (lazy loaded)
_DEFAULT_TRANSLATOR: Optional[GlossToEnglishTranslator] = None


def translate_gloss_sequence(
    gloss_tokens: Sequence[str],
    beam_width: int = 1,
    translator: Optional[GlossToEnglishTranslator] = None,
) -> str:
    """
    Top-level convenience API for translating discrete ISL gloss sequences into English.

    Args:
        gloss_tokens: Sequence of string gloss tokens (e.g., ['NAME', 'YOUR', 'WHAT']).
        beam_width: 1 for Greedy decoding, > 1 for optional Beam search.
        translator: Optional GlossToEnglishTranslator instance.

    Returns:
        Translated English string.
    """
    if translator is not None:
        return translator.translate_glosses(gloss_tokens, beam_width=beam_width)

    global _DEFAULT_TRANSLATOR
    if _DEFAULT_TRANSLATOR is None:
        _DEFAULT_TRANSLATOR = GlossToEnglishTranslator()

    return _DEFAULT_TRANSLATOR.translate_glosses(gloss_tokens, beam_width=beam_width)
