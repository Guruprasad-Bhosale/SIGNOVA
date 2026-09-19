"""
CTC Decoder for Gloss Recognition.

Uses Connectionist Temporal Classification for alignment-free training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class CTCDecoder(nn.Module):
    """
    CTC-based decoder for sign language gloss recognition.
    
    Maps encoder features to gloss probabilities and performs CTC decoding.
    """
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        blank_token: int = 0,
    ):
        """
        Args:
            input_dim: Input feature dimension
            vocab_size: Size of gloss vocabulary (including blank)
            hidden_dim: Hidden layer dimension
            num_layers: Number of projection layers
            dropout: Dropout rate
            blank_token: Index of CTC blank token
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.blank_token = blank_token
        
        # Projection layers
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [vocab_size]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation/dropout on last layer
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        self.projection = nn.Sequential(*layers)
        
        # CTC loss
        self.ctc_loss = nn.CTCLoss(blank=blank_token, reduction='mean', zero_infinity=True)
    
    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute logits for CTC.
        
        Args:
            features: Encoder output (N, T, input_dim)
            lengths: Feature lengths (N,)
            
        Returns:
            logits: (N, T, vocab_size)
        """
        return self.projection(features)
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss.
        
        Args:
            logits: (N, T, vocab_size)
            targets: Target gloss indices (N, S) or (sum(target_lengths),)
            input_lengths: (N,)
            target_lengths: (N,)
            
        Returns:
            CTC loss value
        """
        # CTC expects (T, N, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    
    def decode_greedy(
        self,
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """
        Greedy CTC decoding (best path).
        
        Args:
            logits: (N, T, vocab_size)
            lengths: (N,)
            
        Returns:
            List of decoded index sequences
        """
        # Get best path
        predictions = logits.argmax(dim=-1)  # (N, T)
        
        decoded = []
        for i, pred in enumerate(predictions):
            length = lengths[i].item() if lengths is not None else len(pred)
            pred = pred[:length].tolist()
            
            # Remove blanks and consecutive duplicates
            output = []
            prev = self.blank_token
            for p in pred:
                if p != self.blank_token and p != prev:
                    output.append(p)
                prev = p
            
            decoded.append(output)
        
        return decoded
    
    def decode_beam(
        self,
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        beam_width: int = 5,
    ) -> List[List[int]]:
        """
        Beam search CTC decoding.
        
        Args:
            logits: (N, T, vocab_size)
            lengths: (N,)
            beam_width: Number of beams
            
        Returns:
            List of decoded index sequences
        """
        log_probs = F.log_softmax(logits, dim=-1)
        batch_size = logits.size(0)
        
        decoded = []
        for i in range(batch_size):
            length = lengths[i].item() if lengths is not None else logits.size(1)
            seq_probs = log_probs[i, :length]  # (T, vocab_size)
            
            # Simple beam search
            beams = [([], 0.0)]  # (sequence, score)
            
            for t in range(length):
                new_beams = []
                for seq, score in beams:
                    for v in range(self.vocab_size):
                        new_score = score + seq_probs[t, v].item()
                        new_seq = seq.copy()
                        
                        if v != self.blank_token:
                            if len(new_seq) == 0 or new_seq[-1] != v:
                                new_seq.append(v)
                        
                        new_beams.append((new_seq, new_score))
                
                # Keep top beams
                new_beams.sort(key=lambda x: x[1], reverse=True)
                
                # Merge beams with same sequence
                merged = {}
                for seq, score in new_beams:
                    key = tuple(seq)
                    if key not in merged or score > merged[key]:
                        merged[key] = score
                
                beams = [(list(k), v) for k, v in merged.items()]
                beams.sort(key=lambda x: x[1], reverse=True)
                beams = beams[:beam_width]
            
            decoded.append(beams[0][0] if beams else [])
        
        return decoded


class CTCHead(nn.Module):
    """
    Complete CTC head with temporal downsampling and projection.
    """
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        hidden_dim: int = 256,
        downsample_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Temporal downsampling
        self.downsample = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=downsample_factor, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.decoder = CTCDecoder(
            input_dim=hidden_dim,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        self.downsample_factor = downsample_factor
    
    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (N, T, input_dim)
            lengths: (N,)
            
        Returns:
            logits: (N, T', vocab_size)
            lengths: Updated lengths (N,)
        """
        # Temporal downsampling
        x = features.permute(0, 2, 1)  # (N, C, T)
        x = self.downsample(x)
        x = x.permute(0, 2, 1)  # (N, T', C)
        
        # Update lengths
        if lengths is not None:
            lengths = (lengths + self.downsample_factor - 1) // self.downsample_factor
        else:
            lengths = torch.full((features.size(0),), x.size(1), device=x.device)
        
        # Get logits
        logits = self.decoder(x, lengths)
        
        return logits, lengths


if __name__ == "__main__":
    # Test CTC decoder
    print("Testing CTCDecoder...")
    
    decoder = CTCDecoder(
        input_dim=256,
        vocab_size=1100,
        hidden_dim=256,
    )
    
    features = torch.randn(2, 50, 256)
    lengths = torch.tensor([50, 40])
    
    logits = decoder(features, lengths)
    print(f"Logits shape: {logits.shape}")
    
    # Test decoding
    decoded = decoder.decode_greedy(logits, lengths)
    print(f"Decoded lengths: {[len(d) for d in decoded]}")
    
    # Test loss computation
    targets = torch.randint(1, 100, (2, 10))
    target_lengths = torch.tensor([10, 8])
    
    loss = decoder.compute_loss(logits, targets, lengths, target_lengths)
    print(f"CTC Loss: {loss.item():.4f}")
    
    print("\nTesting CTCHead...")
    head = CTCHead(input_dim=256, vocab_size=1100)
    logits, out_lengths = head(features, lengths)
    print(f"Head output shape: {logits.shape}")
    print(f"Output lengths: {out_lengths}")
    
    print("\nAll tests passed!")
