"""
Point cloud LLM probe classifier.

Mirrors `train_audio_llm_probe.py` but for point clouds:
  pointcloud -> PointBERT (frozen) -> projector (trainable) -> SAFE fusion (trainable) -> frozen LLM
  pooled LLM hidden state -> classification head (trainable)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List

import torch
import torch.nn as nn

from safe.models.safe_pointcloud_model import SAFEPointCloudModel


@dataclass
class PointCloudProbeOutput:
    logits: torch.Tensor
    pooled: torch.Tensor


class SAFEPointCloudLLMProbe(nn.Module):
    def __init__(
        self,
        safe_config: Dict[str, Any],
        num_classes: int,
        head_type: str = "linear",  # "linear" or "mlp"
        pooling: str = "last",  # "last" or "mean"
    ) -> None:
        super().__init__()

        self.pooling = pooling

        constructor_keys = {
            "llm_model_name",
            "vision_model_name",
            "pointcloud_encoder_type",
            "pointcloud_encoder_config",
            "projector_type",
            "num_tokens",
            "projector_config",
            "fusion_type",
            "fusion_layer_indices",
            "lora_rank",
            "fusion_config",
            "freeze_base_vl",
            "freeze_pointcloud_encoder",
            "label_smoothing",
            "llm_hidden_size",
            "pointcloud_embed_dim",
        }
        constructor_config = {k: v for k, v in safe_config.items() if k in constructor_keys}
        self.safe_model = SAFEPointCloudModel(**constructor_config)
        self.safe_model.enable_pointcloud_training()

        hidden_size = int(safe_config.get("llm_hidden_size", 5120))

        if head_type == "mlp":
            mlp_hidden = min(1024, max(64, hidden_size // 4))
            self.head = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden, num_classes),
            )
        else:
            self.head = nn.Linear(hidden_size, num_classes)

        # Encourage HF models to actually return hidden states.
        try:
            llm = self.safe_model.base_vl.llm
            for candidate in [llm, getattr(llm, "language_model", None), getattr(llm, "model", None)]:
                if candidate is None or not hasattr(candidate, "config"):
                    continue
                try:
                    candidate.config.output_hidden_states = True
                except Exception:
                    pass
                try:
                    candidate.config.return_dict = True
                except Exception:
                    pass
        except Exception:
            pass

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = list(self.safe_model.get_trainable_parameters())
        params.extend(list(self.head.parameters()))
        return params

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pointcloud: Optional[Union[torch.Tensor, List[Any]]] = None,
        pointcloud_tokens: Optional[torch.Tensor] = None,
    ) -> PointCloudProbeOutput:
        out = self.safe_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            pointcloud=pointcloud,
            pointcloud_tokens=pointcloud_tokens,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden = out.get("hidden_states")
        if hidden is None or not torch.is_tensor(hidden):
            raise RuntimeError("SAFEPointCloudModel did not return hidden_states; probe requires them.")

        if self.pooling == "mean":
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.to(dtype=hidden.dtype, device=hidden.device).unsqueeze(-1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                pooled = (hidden * mask).sum(dim=1) / denom
        else:
            # "last" = last non-pad token if attention_mask provided, else last position
            if attention_mask is None:
                pooled = hidden[:, -1, :]
            else:
                lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths, :]

        logits = self.head(pooled.float())
        return PointCloudProbeOutput(logits=logits, pooled=pooled)

