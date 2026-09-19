"""
Translation baselines for Gloss-to-English translation.

Guiding Principles:
1. Baseline A: Exact memorization / frequency lookup baseline for controlled verification.
2. Baseline B: Attentive Seq2Seq BiGRU neural baseline.
3. Strict separation of training pairs and unseen test queries.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch

from signova.translation.dataset import TranslationSample
from signova.translation.decoding import GreedyDecoder
from signova.translation.model import SyntheticGlossToEnglishSeq2Seq
from signova.translation.vocabulary import TranslationVocabulary


class MemorizationBaseline:
    """
    Baseline A — Frequency / Exact Memorization Lookup Baseline.
    Maps known gloss sequences to their most frequent target translation.
    """

    def __init__(self, fallback_strategy: str = "most_frequent_target"):
        self.fallback_strategy = fallback_strategy
        self.pair_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.token_to_word_map: Dict[str, Counter] = defaultdict(Counter)
        self.global_target_counter: Counter = Counter()
        self.is_fitted: bool = False

    def fit(self, samples: Sequence[TranslationSample]) -> "MemorizationBaseline":
        """Fit lookup table on training translation samples."""
        self.pair_counts.clear()
        self.token_to_word_map.clear()
        self.global_target_counter.clear()

        for s in samples:
            key = tuple(s.source_tokens)
            target = s.target_text.strip()
            self.pair_counts[key][target] += 1
            self.global_target_counter[target] += 1

            # Build simple token unigram map
            target_words = target.split()
            for token in s.source_tokens:
                for w in target_words:
                    self.token_to_word_map[token][w.lower()] += 1

        self.is_fitted = True
        return self

    def predict(self, source_tokens: Sequence[str]) -> str:
        """Predict target text from source gloss sequence."""
        if not self.is_fitted:
            return ""

        key = tuple(source_tokens)
        if key in self.pair_counts and len(self.pair_counts[key]) > 0:
            return self.pair_counts[key].most_common(1)[0][0]

        # Fallback for unseen sequence
        if self.fallback_strategy == "unigram_mapping":
            words = []
            for token in source_tokens:
                if token in self.token_to_word_map and len(self.token_to_word_map[token]) > 0:
                    words.append(self.token_to_word_map[token].most_common(1)[0][0])
                else:
                    words.append(token.lower())
            return " ".join(words)

        if len(self.global_target_counter) > 0:
            return self.global_target_counter.most_common(1)[0][0]

        return ""

    def evaluate_exact_match(self, samples: Sequence[TranslationSample]) -> float:
        """Calculate exact match ratio on evaluated samples."""
        if not samples or not self.is_fitted:
            return 0.0
        matches = sum(1 for s in samples if self.predict(s.source_tokens) == s.target_text.strip())
        return matches / len(samples)


class Seq2SeqNeuralBaseline:
    """
    Baseline B — Attentive BiGRU Seq2Seq Neural Model wrapper.
    """

    def __init__(
        self,
        model: SyntheticGlossToEnglishSeq2Seq,
        source_vocab: TranslationVocabulary,
        target_vocab: TranslationVocabulary,
        decoder: Optional[GreedyDecoder] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.decoder = decoder or GreedyDecoder()
        self.device = device

    def predict(self, source_tokens: Sequence[str]) -> str:
        """Translate gloss sequence to English text."""
        source_ids = self.source_vocab.encode(source_tokens, add_bos=False, add_eos=False)
        if not source_ids:
            return ""

        src_tensor = torch.tensor([source_ids], dtype=torch.long, device=self.device)
        result = self.decoder.decode_single(
            model=self.model,
            source_ids=src_tensor,
            target_vocab=self.target_vocab,
        )
        return result.text
