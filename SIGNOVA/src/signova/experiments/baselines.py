"""
Phase 12 Experimental Baselines for Real Continuous ISL Recognition.

Baselines:
- Baseline A: Majority / Frequency-based Sequence Baseline
- Baseline B: Temporal Average Pooling Sequence Classifier
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from signova.qualification.vocabulary import Phase12GlossVocabulary


class BaselineAMajoritySequence:
    """Predicts the most frequent training sequence or token sequence for any input."""

    def __init__(self):
        self.most_common_sequence: List[str] = []

    def fit(self, train_sequences: List[List[str]]) -> None:
        if not train_sequences:
            self.most_common_sequence = []
            return
        # Convert each sequence to tuple to count
        seq_tuples = [tuple(seq) for seq in train_sequences]
        counts = Counter(seq_tuples)
        self.most_common_sequence = list(counts.most_common(1)[0][0])

    def predict(self, num_samples: int) -> List[List[str]]:
        return [list(self.most_common_sequence) for _ in range(num_samples)]


class BaselineBTemporalPoolingClassifier(nn.Module):
    """Simple temporal average-pooling sequence classifier."""

    def __init__(self, input_dim: int, vocab_size: int, max_seq_len: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size * max_seq_len),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (batch_size, time_steps, input_dim)
        # Average over time
        pooled = x.mean(dim=1)  # (batch_size, input_dim)
        logits = self.fc(pooled)  # (batch_size, vocab_size * max_seq_len)
        return logits.view(x.size(0), self.max_seq_len, self.vocab_size)

    def predict_sequences(self, x: torch.Tensor, vocab: Phase12GlossVocabulary) -> List[List[str]]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            pred_ids = logits.argmax(dim=-1).cpu().numpy()  # (batch_size, max_seq_len)

        results = []
        for row in pred_ids:
            decoded = vocab.decode(list(row), remove_blank=True)
            results.append(decoded)
        return results
