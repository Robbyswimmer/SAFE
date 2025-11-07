from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import (
    BaseProjector,
    init_linear_stack,
    register_projector,
    resolve_activation,
)


@register_projector("video")
class VideoProjector(BaseProjector):
    """Project pooled video embeddings into the LLM token space."""

    def __init__(
        self,
        video_embed_dim: int,
        llm_hidden_size: int,
        num_video_tokens: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        bottleneck_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.video_embed_dim = video_embed_dim
        self.llm_hidden_size = llm_hidden_size
        self.num_video_tokens = num_video_tokens

        if bottleneck_dim is None:
            bottleneck_dim = min(1024, llm_hidden_size // 4)
        self.bottleneck_dim = bottleneck_dim

        self.activation = resolve_activation(activation)
        self.input_norm = nn.LayerNorm(video_embed_dim, eps=1e-6)

        self.projector = nn.Sequential(
            nn.Linear(video_embed_dim, bottleneck_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, llm_hidden_size * num_video_tokens),
            nn.Tanh(),
        )

        self.output_norm = nn.LayerNorm(llm_hidden_size, eps=1e-6)
        init_linear_stack(self.projector.modules())

    def forward(
        self, video_features: torch.Tensor, out_dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        batch_size = video_features.shape[0]

        normalized_input = self.input_norm(video_features)
        projected = self.projector(normalized_input)
        projected = torch.tanh(projected) * 2.0

        video_tokens = projected.view(batch_size, self.num_video_tokens, self.llm_hidden_size)
        video_tokens = self.output_norm(video_tokens)

        if out_dtype is not None:
            video_tokens = video_tokens.to(out_dtype)
        return video_tokens
