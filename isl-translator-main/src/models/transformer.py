"""
Transformer Encoder for Sign Language Recognition.

Processes keypoint sequences with self-attention for temporal modeling.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position information.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (N, T, d_model)
            
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class KeypointEmbedding(nn.Module):
    """
    Embed raw keypoints into a feature space suitable for Transformer.
    """
    
    def __init__(
        self,
        num_keypoints: int = 543,
        keypoint_dim: int = 3,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.input_dim = num_keypoints * keypoint_dim
        
        # Project from flattened keypoints to embed_dim
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Keypoints (N, T, num_keypoints, keypoint_dim)
            
        Returns:
            Embeddings (N, T, embed_dim)
        """
        N, T, V, C = x.size()
        x = x.view(N, T, V * C)  # Flatten keypoints
        return self.projection(x)


class SignTransformer(nn.Module):
    """
    Transformer Encoder for Sign Language Recognition.
    
    Takes keypoint sequences and outputs contextualized representations.
    """
    
    def __init__(
        self,
        num_keypoints: int = 543,
        keypoint_dim: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_position: int = 1024,
    ):
        """
        Args:
            num_keypoints: Number of input keypoints
            keypoint_dim: Dimension per keypoint (x, y, conf)
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer layers
            dim_feedforward: Feed-forward network dimension
            dropout: Dropout rate
            max_position: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Keypoint embedding
        self.embedding = KeypointEmbedding(
            num_keypoints=num_keypoints,
            keypoint_dim=keypoint_dim,
            embed_dim=d_model,
            dropout=dropout,
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_position, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )
        
        # Output layer norm
        self.output_norm = nn.LayerNorm(d_model)
    
    def _generate_padding_mask(
        self,
        lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """
        Generate attention mask for padded sequences.
        
        Args:
            lengths: Sequence lengths (N,)
            max_len: Maximum sequence length
            
        Returns:
            Padding mask (N, max_len) where True = ignore
        """
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
        mask = mask >= lengths.unsqueeze(1)
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Keypoint sequence (N, T, num_keypoints, keypoint_dim)
            lengths: Sequence lengths (N,)
            
        Returns:
            features: Contextualized features (N, T, d_model)
            lengths: Same as input lengths
        """
        N, T, V, C = x.size()
        
        # Embed keypoints
        x = self.embedding(x)  # (N, T, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Generate padding mask
        if lengths is not None:
            padding_mask = self._generate_padding_mask(lengths, T)
        else:
            padding_mask = None
            lengths = torch.full((N,), T, device=x.device)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Output normalization
        x = self.output_norm(x)
        
        return x, lengths
    
    def get_output_dim(self) -> int:
        """Return output feature dimension."""
        return self.d_model


class HybridEncoder(nn.Module):
    """
    Hybrid encoder combining ST-GCN (spatial) and Transformer (temporal).
    
    ST-GCN captures local body part relationships.
    Transformer captures global temporal dependencies.
    """
    
    def __init__(
        self,
        num_keypoints: int = 543,
        keypoint_dim: int = 3,
        stgcn_hidden: int = 64,
        stgcn_layers: int = 2,
        d_model: int = 256,
        nhead: int = 8,
        num_transformer_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Import STGCN here to avoid circular import
        from .stgcn import STGCN
        
        self.d_model = d_model
        
        # ST-GCN for spatial features
        self.stgcn = STGCN(
            in_channels=keypoint_dim,
            hidden_channels=stgcn_hidden,
            out_channels=d_model,
            num_layers=stgcn_layers,
            dropout=dropout,
        )
        
        # Transformer for temporal modeling
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )
        
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Keypoints (N, T, V, C)
            lengths: Sequence lengths
            
        Returns:
            features: (N, T', d_model)
            lengths: Updated lengths
        """
        # ST-GCN spatial encoding
        x, lengths = self.stgcn(x, lengths)  # (N, T', d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Generate padding mask
        if lengths is not None:
            T = x.size(1)
            padding_mask = self._generate_padding_mask(lengths, T, x.device)
        else:
            padding_mask = None
        
        # Transformer temporal encoding
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.output_norm(x)
        
        return x, lengths
    
    def _generate_padding_mask(
        self,
        lengths: torch.Tensor,
        max_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=device).expand(batch_size, max_len)
        mask = mask >= lengths.unsqueeze(1)
        return mask
    
    def get_output_dim(self) -> int:
        return self.d_model


if __name__ == "__main__":
    # Test modules
    print("Testing SignTransformer...")
    
    model = SignTransformer(
        num_keypoints=543,
        keypoint_dim=3,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
    )
    
    x = torch.randn(2, 100, 543, 3)
    lengths = torch.tensor([100, 80])
    
    features, out_lengths = model(x, lengths)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting HybridEncoder...")
    
    hybrid = HybridEncoder(
        num_keypoints=543,
        d_model=256,
        stgcn_layers=2,
        num_transformer_layers=4,
    )
    
    features, out_lengths = hybrid(x, lengths)
    print(f"Hybrid output shape: {features.shape}")
    print(f"Hybrid parameters: {sum(p.numel() for p in hybrid.parameters()):,}")
    
    print("\nAll tests passed!")
