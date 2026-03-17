from __future__ import annotations

import torch
import torch.nn as nn


class AudioTokenEmbeddingProxy(nn.Module):
    """
    Wrap a token embedding table and let the owner replace reserved audio token
    positions with dynamic projected audio embeddings at runtime.
    """

    def __init__(self, base_embedding: nn.Module, owner) -> None:
        super().__init__()
        self.base_embedding = base_embedding
        self.owner = owner

    @property
    def weight(self):
        return self.base_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.base_embedding(input_ids)
        return self.owner._apply_pending_audio_embeddings(input_ids, embeddings)
