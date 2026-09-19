"""
Training utilities and losses for ISL Translator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLoss(nn.Module):
    """CTC Loss wrapper with additional features."""
    
    def __init__(self, blank: int = 0, reduction: str = "mean"):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=True)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (N, T, vocab_size)
            targets: (N, S) or (sum(target_lengths),)
            input_lengths: (N,)
            target_lengths: (N,)
        """
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        return self.ctc(log_probs, targets, input_lengths, target_lengths)


class TranslationLoss(nn.Module):
    """Cross-entropy loss for translation with label smoothing."""
    
    def __init__(self, vocab_size: int, label_smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (N, T, vocab_size)
            targets: (N, T)
        """
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        return self.loss(logits, targets)


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss implementation."""
    
    def __init__(self, size: int, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.size = size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, vocab_size)
            targets: (N,)
        """
        vocab_size = logits.size(-1)
        
        # Create smoothed labels
        with torch.no_grad():
            smooth_target = torch.zeros_like(logits)
            smooth_target.fill_(self.smoothing / (vocab_size - 2))
            smooth_target.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
            
            # Mask padding
            mask = targets == self.ignore_index
            smooth_target[mask] = 0
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(smooth_target * log_probs, dim=-1)
        loss = loss.masked_fill(mask, 0)
        
        return loss.sum() / (~mask).sum()
