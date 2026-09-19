"""
CTC Decoding and Error Rate Evaluation Utilities for SIGNOVA.

Provides greedy CTC sequence decoding, blank removal, duplicate collapse,
and Levenshtein Token Error Rate (TER / CER) calculations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class CTCDecoder:
    """
    Greedy and Collapse utilities for CTC logit predictions.
    """

    def __init__(self, blank_idx: int = 0, id_to_label: Optional[Dict[int, str]] = None):
        self.blank_idx = blank_idx
        self.id_to_label = id_to_label or {}

    def decode_greedy_tensor(
        self,
        log_probs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            log_probs: Tensor of shape (B, T, num_classes) or (T, num_classes).
            lengths: Optional tensor (B,) with valid frame counts.

        Returns:
            List of decoded sequence dicts.
        """
        if log_probs.ndim == 2:
            log_probs = log_probs.unsqueeze(0)

        B, T, C = log_probs.shape
        best_tokens = torch.argmax(log_probs, dim=-1).cpu().numpy()
        probs = torch.exp(log_probs).cpu().numpy()

        results = []
        for b in range(B):
            seq_len = int(lengths[b].item()) if lengths is not None else T
            raw_tokens = best_tokens[b, :seq_len]
            token_probs = probs[b, :seq_len]

            collapsed = []
            collapsed_labels = []
            confidences = []
            prev_tok = None

            for t in range(seq_len):
                tok = int(raw_tokens[t])
                if tok != self.blank_idx:
                    if tok != prev_tok:
                        collapsed.append(tok)
                        collapsed_labels.append(self.id_to_label.get(tok, str(tok)))
                        confidences.append(float(token_probs[t, tok]))
                prev_tok = tok

            results.append({
                "raw_tokens": [int(x) for x in raw_tokens],
                "tokens": collapsed,
                "labels": collapsed_labels,
                "confidences": [round(c, 4) for c in confidences],
                "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
                "sequence_length": seq_len,
            })

        return results

    def decode_beam_search(
        self,
        log_probs: torch.Tensor,
        beam_width: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Optional beam search interface for future exploration.
        Defaults to greedy decoding if specialized CTC beam decoder is uninstalled.
        """
        # Fallback to greedy decoding
        return self.decode_greedy_tensor(log_probs)

    @staticmethod
    def compute_token_error_rate(
        reference: List[Union[int, str]],
        hypothesis: List[Union[int, str]],
    ) -> Dict[str, Any]:
        """
        Calculates Levenshtein Token Error Rate (TER / CER) between reference and hypothesis sequences.

        Returns:
            Dict containing TER, substitution count, deletion count, insertion count, and exact match bool.
        """
        r_len = len(reference)
        h_len = len(hypothesis)

        if r_len == 0:
            return {
                "token_error_rate": 1.0 if h_len > 0 else 0.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": h_len,
                "exact_match": (h_len == 0),
                "reference_length": 0,
                "hypothesis_length": h_len,
            }

        # DP Table for Levenshtein Distance
        dp = np.zeros((r_len + 1, h_len + 1), dtype=int)
        for i in range(r_len + 1):
            dp[i, 0] = i
        for j in range(h_len + 1):
            dp[0, j] = j

        for i in range(1, r_len + 1):
            for j in range(1, h_len + 1):
                if reference[i - 1] == hypothesis[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1]
                else:
                    dp[i, j] = 1 + min(
                        dp[i - 1, j],     # Deletion
                        dp[i, j - 1],     # Insertion
                        dp[i - 1, j - 1]  # Substitution
                    )

        # Backtrace to count S, D, I
        i, j = r_len, h_len
        subs, dels, ins = 0, 0, 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
                subs += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
                dels += 1
                i -= 1
            elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
                ins += 1
                j -= 1
            else:
                break

        edit_dist = dp[r_len, h_len]
        ter = float(edit_dist) / r_len
        exact_match = (reference == hypothesis)

        return {
            "token_error_rate": round(ter, 4),
            "edit_distance": int(edit_dist),
            "substitutions": subs,
            "deletions": dels,
            "insertions": ins,
            "exact_match": exact_match,
            "reference_length": r_len,
            "hypothesis_length": h_len,
        }
