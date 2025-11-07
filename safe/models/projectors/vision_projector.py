from __future__ import annotations

import torch.nn as nn

from .base import BaseProjector, register_projector


@register_projector("vision_base")
class VisionProjector(BaseProjector):
    """Baseline vision projector used by the frozen VL backbone."""

    def __init__(
        self,
        vision_embed_dim: int,
        llm_hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(vision_embed_dim, llm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, vision_features):
        return self.projector(vision_features)
