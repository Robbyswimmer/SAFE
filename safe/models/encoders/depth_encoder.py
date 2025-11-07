from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DepthEncoderOutput:
    """Container for depth encoder results."""

    embeddings: torch.Tensor
    feature_map: Optional[torch.Tensor]


class DepthMapEncoder(nn.Module):
    """Simple depth encoder producing a fixed-size embedding."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_size: int = 128,
        embed_dim: int = 256,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        self.normalize = normalize
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, embed_dim),
        )

    def forward(
        self,
        depth: torch.Tensor,
        *,
        return_feature_map: bool = False,
    ) -> DepthEncoderOutput:
        if depth.ndim not in (3, 4):
            raise ValueError("Depth tensor must be shaped (B, H, W) or (B, C, H, W)")

        if depth.ndim == 3:
            depth = depth.unsqueeze(1)

        depth = depth.float()
        if self.normalize and torch.isfinite(depth).all():
            max_val = depth.max()
            if max_val > 0:
                depth = depth / max_val

        pooled = self.pool(depth).view(depth.size(0), -1)
        embeddings = self.mlp(pooled)

        feature_map = depth if return_feature_map else None
        return DepthEncoderOutput(embeddings=embeddings, feature_map=feature_map)
