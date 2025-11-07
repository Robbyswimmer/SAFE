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


@register_projector("audio")
class AudioProjector(BaseProjector):
    """
    Trainable projector that maps audio embeddings to the LLM token space.

    The architecture matches the historical SAFE implementation: a bottlenecked
    two-layer MLP that emits a fixed number of tokens, followed by tanh scaling
    and layer-normalisation for stability.
    """

    def __init__(
        self,
        audio_embed_dim: int,
        llm_hidden_size: int,
        num_audio_tokens: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        bottleneck_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.audio_embed_dim = audio_embed_dim
        self.llm_hidden_size = llm_hidden_size
        self.num_audio_tokens = num_audio_tokens

        if bottleneck_dim is None:
            bottleneck_dim = min(1024, llm_hidden_size // 4)
        self.bottleneck_dim = bottleneck_dim

        self.activation = resolve_activation(activation)
        self.input_norm = nn.LayerNorm(audio_embed_dim, eps=1e-6)

        self.projector = nn.Sequential(
            nn.Linear(audio_embed_dim, bottleneck_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, llm_hidden_size * num_audio_tokens),
            nn.Tanh(),
        )

        self.output_norm = nn.LayerNorm(llm_hidden_size, eps=1e-6)
        init_linear_stack(self.projector.modules())

    def forward(
        self, audio_features: torch.Tensor, out_dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        batch_size = audio_features.shape[0]

        x = torch.nan_to_num(audio_features, nan=0.0, posinf=0.0, neginf=0.0)
        if x.dtype != torch.float32:
            x = x.float()

        normalized_input = self.input_norm(x)
        projected = self.projector(normalized_input)
        projected = torch.tanh(projected) * 2.0

        audio_tokens = projected.view(batch_size, self.num_audio_tokens, self.llm_hidden_size)
        audio_tokens = self.output_norm(audio_tokens)

        if out_dtype is not None:
            audio_tokens = audio_tokens.to(out_dtype)
        return audio_tokens


class AdaptiveAudioProjector(BaseProjector):
    """Adaptive projector that can emit a variable number of audio tokens."""

    def __init__(
        self,
        audio_embed_dim: int,
        llm_hidden_size: int,
        max_audio_tokens: int = 12,
        min_audio_tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.audio_embed_dim = audio_embed_dim
        self.llm_hidden_size = llm_hidden_size
        self.max_audio_tokens = max_audio_tokens
        self.min_audio_tokens = min_audio_tokens

        self.input_norm = nn.LayerNorm(audio_embed_dim, eps=1e-6)

        self.feature_extractor = nn.Sequential(
            nn.Linear(audio_embed_dim, llm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.token_predictor = nn.Sequential(
            nn.Linear(llm_hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, max_audio_tokens - min_audio_tokens + 1),
        )

        self.token_generators = nn.ModuleDict()
        for k in range(min_audio_tokens, max_audio_tokens + 1):
            self.token_generators[str(k)] = nn.Sequential(
                nn.Linear(llm_hidden_size, llm_hidden_size * k),
                nn.Tanh(),
            )

        self.output_norm = nn.LayerNorm(llm_hidden_size, eps=1e-6)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for generator in self.token_generators.values():
            for module in generator.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=1e-5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        audio_features: torch.Tensor,
        num_tokens: Optional[int] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        batch_size = audio_features.shape[0]

        x = torch.nan_to_num(audio_features, nan=0.0, posinf=0.0, neginf=0.0)
        if x.dtype != torch.float32:
            x = x.float()

        normalized_input = self.input_norm(x)
        features = self.feature_extractor(normalized_input)

        if num_tokens is None:
            token_logits = self.token_predictor(features)
            token_probs = torch.softmax(token_logits, dim=-1)

            token_weights = torch.arange(
                self.min_audio_tokens,
                self.max_audio_tokens + 1,
                device=features.device,
                dtype=features.dtype,
            )
            expected_tokens = torch.sum(token_probs * token_weights.unsqueeze(0), dim=-1)
            num_tokens = torch.round(expected_tokens).int()

        if isinstance(num_tokens, int):
            num_tokens = torch.full((batch_size,), num_tokens, device=features.device)

        most_common_tokens = torch.mode(num_tokens).values.item()
        most_common_tokens = max(self.min_audio_tokens, min(self.max_audio_tokens, most_common_tokens))

        generator = self.token_generators[str(most_common_tokens)]
        projected = generator(features)
        projected = projected * 2.0

        audio_tokens = projected.view(batch_size, most_common_tokens, self.llm_hidden_size)
        audio_tokens = self.output_norm(audio_tokens)

        if out_dtype is not None:
            audio_tokens = audio_tokens.to(out_dtype)
        return audio_tokens
