"""
Simple point cloud classifier: frozen point cloud encoder + MLP head.

This mirrors `train_ave_classifier.py`'s CLAP+MLP baseline, but for point clouds.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from safe.models.pointcloud_encoders import PointBERTEncoder


class PointCloudClassifier(nn.Module):
    """
    Point cloud -> encoder (frozen) -> MLP classifier.

    Intended as a non-generative baseline for ModelNet40-style tasks.
    """

    def __init__(
        self,
        num_classes: int,
        encoder_model_name: str = "pointbert-base",
        encoder_num_points: int = 1024,
        encoder_embed_dim: int = 768,
        encoder_checkpoint_path: Optional[str] = None,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.encoder = PointBERTEncoder(
            model_name=encoder_model_name,
            freeze=True,
            num_points=encoder_num_points,
            embed_dim=encoder_embed_dim,
            use_pretrained=True,
            checkpoint_path=encoder_checkpoint_path,
        )

        self.classifier = nn.Sequential(
            nn.Linear(encoder_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pointclouds: Any) -> torch.Tensor:
        features = self.encoder(pointclouds)  # (B, D)
        return self.classifier(features)  # (B, num_classes)

