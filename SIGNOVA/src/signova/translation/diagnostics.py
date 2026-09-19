"""
Inference and training diagnostics for sequence-to-sequence translation.

Guiding Principles:
1. Teacher-forcing loss alone is insufficient; autoregressive metrics are explicitly profiled.
2. Tracks EOS termination rates, repetition frequency, UNK emission rate, and decoding latency.
3. Records model parameter count and memory resource footprint.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Sequence
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from signova.translation.decoding import GreedyDecoder
from signova.translation.metrics import compute_sentence_bleu
from signova.translation.model import SyntheticGlossToEnglishSeq2Seq
from signova.translation.vocabulary import TranslationVocabulary


@dataclass
class TranslationDiagnosticsReport:
    """
    Structured diagnostics for translation model validation.
    """
    total_samples: int
    teacher_forced_loss: float
    autoregressive_bleu4_mean: float
    autoregressive_exact_match_pct: float
    eos_termination_rate: float
    repetition_rate: float
    unk_token_rate: float
    avg_decoding_latency_ms: float
    model_param_count: int
    model_size_mb: float
    is_synthetic_evaluation: bool = True


class TranslationDiagnostics:
    """
    Runs diagnostic validation across validation/test dataloaders.
    """

    def __init__(
        self,
        model: SyntheticGlossToEnglishSeq2Seq,
        source_vocab: TranslationVocabulary,
        target_vocab: TranslationVocabulary,
        loss_fn: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.device = device
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=model.tgt_pad_idx)
        self.decoder = GreedyDecoder(max_len=50)

    @torch.no_grad()
    def evaluate_diagnostics(
        self,
        dataloader: DataLoader,
        is_synthetic: bool = True,
    ) -> TranslationDiagnosticsReport:
        """
        Compute full diagnostics over a dataloader.
        """
        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        bleu_scores: List[float] = []
        exact_matches = 0
        eos_terminations = 0
        repeated_sequences = 0
        total_gen_tokens = 0
        total_unk_tokens = 0
        total_latency_ms = 0.0
        total_samples = 0

        for batch in dataloader:
            source_ids = batch["source_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            source_lengths = batch.get("source_lengths", None)
            if source_lengths is not None:
                source_lengths = source_lengths.to(self.device)
            source_mask = batch.get("source_mask", None)
            if source_mask is not None:
                source_mask = source_mask.to(self.device)

            # 1. Teacher-forced forward pass loss
            # outputs: (batch_size, tgt_len - 1, tgt_vocab)
            outputs, _ = self.model(
                source_ids=source_ids,
                target_ids=target_ids,
                source_lengths=source_lengths,
                source_mask=source_mask,
                teacher_forcing_ratio=1.0,
            )
            # targets: (batch_size, tgt_len - 1)
            targets = target_ids[:, 1:].contiguous()
            loss = self.loss_fn(outputs.view(-1, self.model.tgt_vocab_size), targets.view(-1))
            total_loss += loss.item()
            total_batches += 1

            # 2. Autoregressive decoding per sample
            batch_size = source_ids.size(0)
            for i in range(batch_size):
                total_samples += 1
                single_src = source_ids[i : i + 1]
                single_len = source_lengths[i : i + 1] if source_lengths is not None else None
                single_mask = source_mask[i : i + 1] if source_mask is not None else None

                t0 = time.perf_counter()
                result = self.decoder.decode_single(
                    model=self.model,
                    source_ids=single_src,
                    target_vocab=self.target_vocab,
                    source_lengths=single_len,
                    source_mask=single_mask,
                )
                lat_ms = (time.perf_counter() - t0) * 1000.0
                total_latency_ms += lat_ms

                # EOS status
                if result.terminated_by_eos:
                    eos_terminations += 1

                # Repetition check (consecutive identical words)
                gen_words = result.tokens
                has_repeat = any(gen_words[j] == gen_words[j + 1] for j in range(len(gen_words) - 1)) if len(gen_words) > 1 else False
                if has_repeat:
                    repeated_sequences += 1

                # UNK count
                total_gen_tokens += len(result.token_ids)
                total_unk_tokens += sum(1 for tid in result.token_ids if tid == self.model.tgt_vocab_size or tid == self.target_vocab.unk_id)

                # Reference text comparison
                target_raw = batch["target_texts"][i]
                target_words = target_raw.strip().split()
                s_bleu = compute_sentence_bleu(gen_words, target_words)
                bleu_scores.append(s_bleu)

                if result.text.strip().lower() == target_raw.strip().lower():
                    exact_matches += 1

        avg_loss = (total_loss / total_batches) if total_batches > 0 else 0.0
        avg_bleu = (sum(bleu_scores) / total_samples) if total_samples > 0 else 0.0
        exact_pct = (exact_matches / total_samples * 100.0) if total_samples > 0 else 0.0
        eos_rate = (eos_terminations / total_samples * 100.0) if total_samples > 0 else 0.0
        rep_rate = (repeated_sequences / total_samples * 100.0) if total_samples > 0 else 0.0
        unk_rate = (total_unk_tokens / max(1, total_gen_tokens) * 100.0)
        avg_lat = (total_latency_ms / total_samples) if total_samples > 0 else 0.0

        param_count = self.model.count_parameters()
        model_size_mb = (param_count * 4) / (1024 * 1024)

        return TranslationDiagnosticsReport(
            total_samples=total_samples,
            teacher_forced_loss=round(avg_loss, 4),
            autoregressive_bleu4_mean=round(avg_bleu, 2),
            autoregressive_exact_match_pct=round(exact_pct, 2),
            eos_termination_rate=round(eos_rate, 2),
            repetition_rate=round(rep_rate, 2),
            unk_token_rate=round(unk_rate, 2),
            avg_decoding_latency_ms=round(avg_lat, 2),
            model_param_count=param_count,
            model_size_mb=round(model_size_mb, 2),
            is_synthetic_evaluation=is_synthetic,
        )
