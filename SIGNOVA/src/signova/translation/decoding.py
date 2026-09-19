"""
Decoding strategies for sequence-to-sequence translation.

Guiding Principles:
1. Greedy decoding is the core verified decoder.
2. Beam search is an optional experimental decoder.
3. Latency, EOS termination, and generated length are tracked per decoding run.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F


@dataclass
class DecodingResult:
    """
    Standardized result for single-sample or batch decoding.
    """
    token_ids: List[int]
    tokens: List[str]
    text: str
    terminated_by_eos: bool
    decoding_latency_ms: float
    length: int
    score: float = 0.0


class GreedyDecoder:
    """
    Greedy autoregressive decoder for Seq2Seq translation.
    """

    def __init__(
        self,
        max_len: int = 50,
        repetition_penalty: float = 1.0,
    ):
        self.max_len = max_len
        self.repetition_penalty = repetition_penalty

    @torch.no_grad()
    def decode_single(
        self,
        model: Any,
        source_ids: torch.Tensor,
        target_vocab: Any,
        source_lengths: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
    ) -> DecodingResult:
        """
        Greedy decode a single sequence.
        
        Args:
            model: Seq2Seq model instance
            source_ids: Tensor of shape (1, src_len) or (src_len,)
            target_vocab: TranslationVocabulary instance
            source_lengths: Optional lengths tensor
            source_mask: Optional mask tensor
        """
        start_time = time.perf_counter()
        model.eval()

        if source_ids.dim() == 1:
            source_ids = source_ids.unsqueeze(0)

        device = source_ids.device
        if source_mask is None:
            source_mask = (source_ids != model.src_pad_idx)

        encoder_outputs, decoder_hidden = model.encoder(source_ids, source_lengths)

        current_token = torch.tensor([model.tgt_bos_idx], dtype=torch.long, device=device)
        generated_ids: List[int] = []
        terminated_by_eos = False

        for _ in range(self.max_len):
            logits, decoder_hidden, _ = model.decoder(
                input_token_id=current_token,
                decoder_hidden=decoder_hidden,
                encoder_outputs=encoder_outputs,
                source_mask=source_mask,
            )
            
            # Repetition penalty if configured
            if self.repetition_penalty > 1.0 and len(generated_ids) > 0:
                for past_id in set(generated_ids):
                    if logits[0, past_id] > 0:
                        logits[0, past_id] /= self.repetition_penalty
                    else:
                        logits[0, past_id] *= self.repetition_penalty

            next_token_id = int(logits.argmax(dim=-1).item())

            if next_token_id == model.tgt_eos_idx:
                terminated_by_eos = True
                break

            generated_ids.append(next_token_id)
            current_token = torch.tensor([next_token_id], dtype=torch.long, device=device)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        tokens = target_vocab.decode(generated_ids, skip_special=True)
        text = target_vocab.decode_to_text(generated_ids, skip_special=True)

        return DecodingResult(
            token_ids=generated_ids,
            tokens=tokens,
            text=text,
            terminated_by_eos=terminated_by_eos,
            decoding_latency_ms=elapsed_ms,
            length=len(generated_ids),
        )


class BeamSearchDecoder:
    """
    Optional experimental Beam Search decoder.
    """

    def __init__(
        self,
        beam_width: int = 4,
        max_len: int = 50,
        length_penalty_alpha: float = 0.6,
    ):
        self.beam_width = beam_width
        self.max_len = max_len
        self.length_penalty_alpha = length_penalty_alpha

    @torch.no_grad()
    def decode_single(
        self,
        model: Any,
        source_ids: torch.Tensor,
        target_vocab: Any,
        source_lengths: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
    ) -> DecodingResult:
        """
        Beam search decoding for a single sample.
        """
        start_time = time.perf_counter()
        model.eval()

        if source_ids.dim() == 1:
            source_ids = source_ids.unsqueeze(0)

        device = source_ids.device
        if source_mask is None:
            source_mask = (source_ids != model.src_pad_idx)

        encoder_outputs, decoder_hidden = model.encoder(source_ids, source_lengths)

        # Candidate beams: (log_prob, token_ids, hidden_state, ended)
        beams: List[Tuple[float, List[int], torch.Tensor, bool]] = [
            (0.0, [], decoder_hidden, False)
        ]

        for _ in range(self.max_len):
            all_candidates = []
            all_ended = True

            for log_prob, tokens, hidden, ended in beams:
                if ended:
                    all_candidates.append((log_prob, tokens, hidden, True))
                    continue

                all_ended = False
                last_token = tokens[-1] if len(tokens) > 0 else model.tgt_bos_idx
                input_token = torch.tensor([last_token], dtype=torch.long, device=device)

                logits, new_hidden, _ = model.decoder(
                    input_token_id=input_token,
                    decoder_hidden=hidden,
                    encoder_outputs=encoder_outputs,
                    source_mask=source_mask,
                )

                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (vocab_size,)
                topk_probs, topk_indices = torch.topk(log_probs, self.beam_width)

                for k in range(self.beam_width):
                    cand_id = int(topk_indices[k].item())
                    cand_log_prob = log_prob + float(topk_probs[k].item())
                    cand_tokens = tokens + [cand_id]
                    cand_ended = (cand_id == model.tgt_eos_idx)
                    all_candidates.append((cand_log_prob, cand_tokens, new_hidden, cand_ended))

            if all_ended:
                break

            # Score candidates with length penalty
            def score_candidate(cand: Tuple[float, List[int], torch.Tensor, bool]) -> float:
                lp, toks, _, _ = cand
                length = max(1, len(toks))
                penalty = ((5.0 + length) / 6.0) ** self.length_penalty_alpha
                return lp / penalty

            # Keep top beam_width candidates
            all_candidates.sort(key=score_candidate, reverse=True)
            beams = all_candidates[:self.beam_width]

        # Select top beam
        best_log_prob, best_tokens, _, best_ended = beams[0]
        # Filter out EOS if at the end
        if len(best_tokens) > 0 and best_tokens[-1] == model.tgt_eos_idx:
            best_tokens_clean = best_tokens[:-1]
            terminated_by_eos = True
        else:
            best_tokens_clean = best_tokens
            terminated_by_eos = best_ended

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        tokens_str = target_vocab.decode(best_tokens_clean, skip_special=True)
        text = target_vocab.decode_to_text(best_tokens_clean, skip_special=True)

        return DecodingResult(
            token_ids=best_tokens_clean,
            tokens=tokens_str,
            text=text,
            terminated_by_eos=terminated_by_eos,
            decoding_latency_ms=elapsed_ms,
            length=len(best_tokens_clean),
            score=best_log_prob,
        )
