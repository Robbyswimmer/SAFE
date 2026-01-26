"""
Point cloud encoders for SAFE architecture.

Mirrors the interface of audio_encoders.py (CLAPAudioEncoder) to enable
modality-agnostic projector and fusion reuse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional, Union, List, Tuple
from pathlib import Path


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) for point cloud subsampling.

    Args:
        xyz: Point cloud (B, N, 3) or (N, 3)
        npoint: Number of points to sample

    Returns:
        Sampled point cloud (B, npoint, 3) or (npoint, 3)
    """
    single_batch = xyz.dim() == 2
    if single_batch:
        xyz = xyz.unsqueeze(0)

    B, N, C = xyz.shape
    device = xyz.device

    if N <= npoint:
        # If we have fewer points than needed, pad with duplicates
        if N == 0:
            return torch.zeros(B, npoint, C, device=device)
        indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        # Pad by repeating
        pad_size = npoint - N
        pad_indices = torch.randint(0, N, (B, pad_size), device=device)
        indices = torch.cat([indices, pad_indices], dim=1)
        sampled = torch.gather(xyz, 1, indices.unsqueeze(-1).expand(-1, -1, C))
    else:
        # FPS algorithm
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), device=device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[torch.arange(B, device=device), farthest, :].unsqueeze(1)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.argmax(distance, dim=-1)

        sampled = torch.gather(xyz, 1, centroids.unsqueeze(-1).expand(-1, -1, C))

    if single_batch:
        sampled = sampled.squeeze(0)

    return sampled


class PointBERTEncoder(nn.Module):
    """
    Point cloud encoder using PointBERT or similar pre-trained model.

    Mirrors CLAPAudioEncoder interface for modality-agnostic integration.

    For initial implementation, uses a simple PointNet-style encoder
    that can be swapped for pre-trained PointBERT weights.
    """

    def __init__(
        self,
        model_name: str = "pointbert-base",
        freeze: bool = True,
        num_points: int = 1024,
        embed_dim: int = 768,
        use_pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize point cloud encoder.

        Args:
            model_name: Model variant ("pointbert-base", "pointnet", etc.)
            freeze: Whether to freeze encoder weights
            num_points: Number of points to sample from input
            embed_dim: Output embedding dimension
            use_pretrained: Whether to load pre-trained weights
            checkpoint_path: Path to checkpoint file (optional)
        """
        super().__init__()

        self.model_name = model_name
        self.num_points = num_points
        self.pointcloud_embed_dim = embed_dim
        self.debug_logging = False

        print(f"[PointCloud] Initializing point cloud encoder: {model_name}...", flush=True)

        # Build encoder based on model type
        if "pointbert" in model_name.lower():
            self.encoder = self._build_pointbert_encoder(embed_dim, checkpoint_path)
        else:
            # Default to simple PointNet-style encoder
            self.encoder = self._build_pointnet_encoder(embed_dim)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            print(f"[PointCloud] Encoder frozen ({sum(p.numel() for p in self.encoder.parameters())} params)", flush=True)

        print(f"[PointCloud] ✓ Encoder initialized: embed_dim={embed_dim}, num_points={num_points}", flush=True)

    def _build_pointnet_encoder(self, embed_dim: int) -> nn.Module:
        """
        Build a simple PointNet-style encoder.

        Architecture:
        - Shared MLPs: 3 -> 64 -> 128 -> 256 -> 512
        - Global max pooling
        - FC layers: 512 -> embed_dim
        """
        return nn.Sequential(
            # Point-wise feature extraction
            PointNetSetAbstraction(
                in_channels=3,
                mlp_channels=[64, 128, 256],
            ),
            # Global feature
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            # Project to embedding dimension
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def _build_pointbert_encoder(self, embed_dim: int, checkpoint_path: Optional[str]) -> nn.Module:
        """
        Build PointBERT-style transformer encoder.

        Uses a simplified transformer architecture that can load
        pre-trained PointBERT weights.
        """
        encoder = PointTransformerEncoder(
            in_channels=3,
            embed_dim=embed_dim,
            depth=12,
            num_heads=12,
            num_groups=64,  # Number of point groups/patches
            group_size=32,  # Points per group
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[PointCloud] Loading checkpoint: {checkpoint_path}", flush=True)
            # weights_only=False needed for PointBERT checkpoints (contain numpy arrays)
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            # Handle different checkpoint formats
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Try to load (may need key remapping for different checkpoint formats)
            try:
                encoder.load_state_dict(state_dict, strict=False)
                print(f"[PointCloud] ✓ Checkpoint loaded", flush=True)
            except Exception as e:
                print(f"[PointCloud] Warning: Could not load checkpoint: {e}", flush=True)

        return encoder

    def set_debug_logging(self, enabled: bool) -> None:
        """Enable or disable verbose logging."""
        self.debug_logging = enabled

    def preprocess_pointcloud(
        self,
        points: Union[torch.Tensor, np.ndarray, str],
    ) -> torch.Tensor:
        """
        Preprocess point cloud to standard format.

        Args:
            points: Point cloud as tensor (N, 3+), numpy array, or .npy/.ply file path

        Returns:
            Preprocessed point cloud (num_points, 3), normalized to unit sphere
        """
        # Load from file if path
        if isinstance(points, str):
            path = Path(points)
            if path.suffix == ".npy":
                points = np.load(path)
            elif path.suffix == ".ply":
                try:
                    import trimesh
                    mesh = trimesh.load(path)
                    points = np.array(mesh.vertices)
                except ImportError:
                    raise ImportError("trimesh required for .ply files: pip install trimesh")
            elif path.suffix in [".pcd", ".xyz"]:
                points = np.loadtxt(path)
            else:
                raise ValueError(f"Unsupported point cloud format: {path.suffix}")

        # Convert to tensor
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()

        # Ensure float
        points = points.float()

        # Take only xyz coordinates (ignore color/normal if present)
        if points.shape[-1] > 3:
            points = points[..., :3]

        # Handle different input shapes
        if points.dim() == 1:
            points = points.reshape(-1, 3)

        # Subsample to num_points using FPS
        points = farthest_point_sample(points, self.num_points)

        # Center the point cloud
        centroid = points.mean(dim=0, keepdim=True)
        points = points - centroid

        # Normalize to unit sphere
        max_dist = points.norm(dim=-1).max()
        if max_dist > 1e-6:
            points = points / max_dist

        if self.debug_logging:
            print(f"[PointCloud] Preprocessed: shape={list(points.shape)}, "
                  f"range=[{points.min():.3f}, {points.max():.3f}]", flush=True)

        return points

    def forward(
        self,
        pointclouds: Union[torch.Tensor, List[Any]],
    ) -> torch.Tensor:
        """
        Extract point cloud embeddings.

        Args:
            pointclouds: Batch of point clouds (various formats supported)

        Returns:
            Point cloud embeddings (batch_size, pointcloud_embed_dim)
        """
        if isinstance(pointclouds, list):
            # Process batch of point clouds
            processed = []
            for idx, pc in enumerate(pointclouds):
                p = self.preprocess_pointcloud(pc)
                processed.append(p)
                if idx < 2 and self.debug_logging:
                    print(f"[PointCloud] preprocess[{idx}]: shape={list(p.shape)}", flush=True)

            batch = torch.stack(processed)  # (B, num_points, 3)
        else:
            # Handle tensor input
            batch = pointclouds
            if batch.dim() == 2:
                batch = batch.unsqueeze(0)  # (1, N, 3)

            # Match preprocess_pointcloud behavior for tensor inputs:
            # - take xyz
            # - FPS to num_points
            # - center + unit-sphere normalize
            if batch.shape[-1] > 3:
                batch = batch[..., :3]

            # Subsample/pad to expected num_points
            if batch.shape[1] != self.num_points:
                batch = farthest_point_sample(batch, self.num_points)  # (B, num_points, 3)

            # Center the point cloud
            centroid = batch.mean(dim=1, keepdim=True)
            batch = batch - centroid

            # Normalize to unit sphere (per-sample)
            max_dist = batch.norm(dim=-1).amax(dim=1, keepdim=True)  # (B, 1)
            max_dist = max_dist.clamp(min=1e-6).unsqueeze(-1)  # (B, 1, 1)
            batch = batch / max_dist

        # Ensure on correct device
        device = next(self.encoder.parameters()).device
        batch = batch.to(device)

        # Get embeddings
        with torch.no_grad():
            # Transpose for conv layers: (B, N, 3) -> (B, 3, N)
            if hasattr(self.encoder, 'expects_channels_first') and self.encoder.expects_channels_first:
                batch = batch.transpose(1, 2)

            embeddings = self.encoder(batch)

            # Ensure output is (B, embed_dim)
            if embeddings.dim() == 3:
                # Pool over sequence dimension
                embeddings = embeddings.mean(dim=1)

        if self.debug_logging:
            print(f"[PointCloud] Output: shape={list(embeddings.shape)}, "
                  f"norm={embeddings.norm(dim=-1).mean():.4f}", flush=True)

        return embeddings


class PointNetSetAbstraction(nn.Module):
    """
    PointNet-style set abstraction layer with shared MLPs.
    """

    def __init__(self, in_channels: int, mlp_channels: List[int]):
        super().__init__()

        layers = []
        prev_channels = in_channels
        for out_channels in mlp_channels:
            layers.extend([
                nn.Conv1d(prev_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            ])
            prev_channels = out_channels

        self.mlp = nn.Sequential(*layers)
        self.expects_channels_first = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, N) point cloud
        Returns:
            (B, C, N) point features
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[1] != 3:
            x = x.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)
        return self.mlp(x)


class PointTransformerEncoder(nn.Module):
    """
    Simplified Point Transformer encoder (PointBERT-style).

    Groups points into patches, embeds each patch, and processes
    with transformer layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_groups: int = 64,
        group_size: int = 32,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.num_groups = num_groups
        self.group_size = group_size
        self.embed_dim = embed_dim
        self.expects_channels_first = False

        # Point patch embedding
        self.patch_embed = nn.Sequential(
            nn.Linear(in_channels * group_size, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding for groups
        self.pos_embed = nn.Parameter(torch.zeros(1, num_groups + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _group_points(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Group points into patches using FPS + KNN.

        Args:
            xyz: (B, N, 3) point cloud

        Returns:
            (B, num_groups, group_size * 3) grouped point features
        """
        B, N, C = xyz.shape

        # Get group centers via FPS
        centers = farthest_point_sample(xyz, self.num_groups)  # (B, num_groups, 3)

        # For each center, find nearest neighbors
        # Simplified: just reshape if N = num_groups * group_size
        if N == self.num_groups * self.group_size:
            groups = xyz.reshape(B, self.num_groups, self.group_size, C)
        else:
            # Use random grouping as fallback
            indices = torch.randint(0, N, (B, self.num_groups, self.group_size), device=xyz.device)
            groups = torch.gather(
                xyz.unsqueeze(1).expand(-1, self.num_groups, -1, -1),
                2,
                indices.unsqueeze(-1).expand(-1, -1, -1, C)
            )

        # Flatten group features
        groups = groups.reshape(B, self.num_groups, -1)  # (B, num_groups, group_size * 3)

        return groups

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) point cloud

        Returns:
            (B, embed_dim) global embedding
        """
        if xyz.dim() == 2:
            xyz = xyz.unsqueeze(0)

        B = xyz.shape[0]

        # Group points into patches
        groups = self._group_points(xyz)  # (B, num_groups, group_size * 3)

        # Embed patches
        x = self.patch_embed(groups)  # (B, num_groups, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_groups + 1, embed_dim)

        # Add position embedding
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Normalize
        x = self.norm(x)

        # Return CLS token embedding
        return x[:, 0]  # (B, embed_dim)


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))

        return x
