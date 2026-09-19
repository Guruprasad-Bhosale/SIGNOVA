"""
Spatio-Temporal Graph Convolutional Network (ST-GCN) for Sign Language.

Based on: "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition"
Adapted for sign language with hand, face, and body keypoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer.
    
    Applies spatial convolution on graph-structured data.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(1, 1),
            bias=bias,
        )
    
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (N, C, T, V)
            A: Adjacency matrix (K, V, V) where K = kernel_size
            
        Returns:
            Output features (N, C_out, T, V)
        """
        x = self.conv(x)  # (N, C_out * K, T, V)
        
        N, KC, T, V = x.size()
        x = x.view(N, self.kernel_size, KC // self.kernel_size, T, V)
        
        # Aggregate over kernel partitions
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        
        return x.contiguous()


class TemporalConvolution(nn.Module):
    """
    Temporal Convolution Layer.
    
    Applies 1D convolution along the temporal dimension.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1),
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (N, C, T, V)
            
        Returns:
            Output (N, C_out, T', V)
        """
        return self.bn(self.conv(x))


class STGCNBlock(nn.Module):
    """
    Spatio-Temporal Graph Convolution Block.
    
    Combines spatial graph convolution with temporal convolution.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: (temporal_kernel, spatial_kernel)
            stride: Temporal stride
            dropout: Dropout rate
            residual: Whether to use residual connection
        """
        super().__init__()
        
        temporal_kernel, spatial_kernel = kernel_size
        
        # Spatial convolution
        self.gcn = GraphConvolution(in_channels, out_channels, spatial_kernel)
        
        # Temporal convolution
        self.tcn = TemporalConvolution(
            out_channels, out_channels,
            kernel_size=temporal_kernel,
            stride=stride,
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (N, C, T, V)
            A: Adjacency matrix (K, V, V)
            
        Returns:
            Output (N, C_out, T', V)
        """
        res = self.residual(x)
        
        x = self.gcn(x, A)
        x = self.bn(x)
        x = self.relu(x)
        x = self.tcn(x)
        x = self.dropout(x)
        
        return self.relu(x + res)


class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for Sign Language Recognition.
    
    Takes keypoint sequences and outputs feature representations.
    """
    
    # Simplified graph structure for sign language
    # Full 543 landmarks would be too large, so we use a subset
    NUM_KEYPOINTS = 75  # 33 pose + 21 left hand + 21 right hand (skip face for now)
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        out_channels: int = 256,
        num_layers: int = 4,
        temporal_kernel: int = 9,
        dropout: float = 0.2,
        num_keypoints: int = 75,
    ):
        """
        Args:
            in_channels: Input channels per keypoint (x, y, conf)
            hidden_channels: Hidden layer channels
            out_channels: Output feature dimension
            num_layers: Number of ST-GCN blocks
            temporal_kernel: Temporal convolution kernel size
            dropout: Dropout rate
            num_keypoints: Number of keypoints to use
        """
        super().__init__()
        
        self.num_keypoints = num_keypoints
        
        # Build adjacency matrix
        self.register_buffer('A', self._build_adjacency_matrix())
        
        # Input batch normalization
        self.data_bn = nn.BatchNorm1d(in_channels * num_keypoints)
        
        # Build ST-GCN layers
        channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(
                STGCNBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size=(temporal_kernel, 3),  # 3 partition kernel
                    stride=1 if i < num_layers - 1 else 2,  # Temporal downsampling on last layer
                    dropout=dropout,
                )
            )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
    
    def _build_adjacency_matrix(self) -> torch.Tensor:
        """
        Build graph adjacency matrix with 3 partitions:
        - Identity (self-loop)
        - Inward edges (towards center)
        - Outward edges (away from center)
        """
        V = self.num_keypoints
        A = torch.zeros(3, V, V)
        
        # Partition 0: Identity
        A[0] = torch.eye(V)
        
        # Partition 1 & 2: Skeletal connections
        # Define edges for pose + hands
        edges = self._get_skeleton_edges()
        
        for src, dst in edges:
            if src < V and dst < V:
                A[1, src, dst] = 1  # Inward
                A[2, dst, src] = 1  # Outward
        
        # Normalize
        for i in range(3):
            D = torch.sum(A[i], dim=1, keepdim=True)
            D = torch.clamp(D, min=1e-6)
            A[i] = A[i] / D
        
        return A
    
    def _get_skeleton_edges(self):
        """Define skeleton edges for pose + hands."""
        edges = []
        
        # Pose edges (simplified MediaPipe pose)
        pose_edges = [
            (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),  # Face
            (0, 11), (0, 12),  # Shoulders
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),  # Hips
            (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
        ]
        edges.extend(pose_edges)
        
        # Left hand edges (start at index 33)
        left_hand_offset = 33
        hand_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17),  # Palm
        ]
        edges.extend([(s + left_hand_offset, d + left_hand_offset) for s, d in hand_edges])
        
        # Right hand edges (start at index 54)
        right_hand_offset = 54
        edges.extend([(s + right_hand_offset, d + right_hand_offset) for s, d in hand_edges])
        
        # Connect hands to wrists
        edges.append((15, 33))  # Left wrist to left hand
        edges.append((16, 54))  # Right wrist to right hand
        
        return edges
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Keypoint sequence (N, T, V, C)
            lengths: Sequence lengths (N,)
            
        Returns:
            features: Output features (N, T', C_out)
            lengths: Output lengths (N,)
        """
        N, T, V, C = x.size()
        
        # Select subset of keypoints if needed
        if V > self.num_keypoints:
            # Take pose (0-33) + left hand (501-522) + right hand (522-543)
            # For MediaPipe: pose + hands (skip face for efficiency)
            pose = x[:, :, :33, :]
            left_hand = x[:, :, 501:522, :]
            right_hand = x[:, :, 522:543, :]
            x = torch.cat([pose, left_hand, right_hand], dim=2)
            V = self.num_keypoints
        
        # Reshape for batch norm: (N, C*V, T)
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, C, T, V)
        x = x.view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, T, V)
        
        # Apply ST-GCN layers
        for layer in self.layers:
            x = layer(x, self.A)
        
        # Global spatial pooling
        x = self.pool(x)  # (N, C_out, T', 1)
        x = x.squeeze(-1)  # (N, C_out, T')
        x = x.permute(0, 2, 1)  # (N, T', C_out)
        
        # Update lengths for temporal downsampling
        if lengths is not None:
            # Account for stride in last layer
            lengths = (lengths + 1) // 2
        else:
            lengths = torch.full((N,), x.size(1), device=x.device)
        
        return x, lengths


if __name__ == "__main__":
    # Test ST-GCN
    print("Testing STGCN...")
    
    model = STGCN(
        in_channels=3,
        hidden_channels=64,
        out_channels=256,
        num_layers=4,
    )
    
    # Dummy input: batch of 2, 100 frames, 543 keypoints, 3 coords
    x = torch.randn(2, 100, 543, 3)
    lengths = torch.tensor([100, 80])
    
    features, out_lengths = model(x, lengths)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output lengths: {out_lengths}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("Test passed!")
