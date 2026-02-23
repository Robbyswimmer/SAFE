import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SetMixerBlock(nn.Module):
    """Permutation-invariant transformer block over modality set tokens."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=max(1, int(num_heads)),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        key_padding_mask = None
        if active_mask is not None:
            key_padding_mask = ~active_mask.bool()
        q = self.ln1(x)
        attn_out, _ = self.attn(
            q,
            q,
            q,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class InteractionCorrectionMixer(nn.Module):
    """
    Composable interaction sidecar:
    - Consumes a set of modality summary vectors (order agnostic).
    - Produces a small hidden-space correction with an adaptive gate.
    - Supports missing modalities via per-sample active masks.
    """

    def __init__(
        self,
        hidden_size: int,
        mixer_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        gate_init: float = -2.0,
        min_modalities_to_apply: int = 2,
        util_target_entropy: float = 0.7,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.mixer_dim = int(mixer_dim)
        self.min_modalities_to_apply = max(1, int(min_modalities_to_apply))
        self.util_target_entropy = float(util_target_entropy)

        self.in_proj = nn.Linear(self.hidden_size, self.mixer_dim)
        self.blocks = nn.ModuleList(
            [_SetMixerBlock(self.mixer_dim, num_heads=num_heads, dropout=dropout) for _ in range(max(1, int(num_layers)))]
        )
        self.final_ln = nn.LayerNorm(self.mixer_dim)

        # Modality contribution scores (for weighted set pooling + utilization entropy).
        self.score_head = nn.Linear(self.mixer_dim, 1)

        self.correction_head = nn.Sequential(
            nn.Linear(self.mixer_dim, self.mixer_dim),
            nn.GELU(),
            nn.Linear(self.mixer_dim, self.hidden_size),
        )
        self.gate_head = nn.Linear(self.mixer_dim, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_init))

    def forward(
        self,
        modality_tokens: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            modality_tokens: (B, M, H) modality summary vectors.
            active_mask: (B, M) bool, True for active modality tokens.
        Returns:
            Dict containing correction vectors, gate stats, utilization entropy.
        """
        if modality_tokens.dim() != 3:
            raise ValueError(f"Expected modality_tokens [B,M,H], got {tuple(modality_tokens.shape)}")

        bsz, mcnt, _ = modality_tokens.shape
        if active_mask is None:
            active_mask = torch.ones((bsz, mcnt), device=modality_tokens.device, dtype=torch.bool)
        else:
            active_mask = active_mask.to(device=modality_tokens.device, dtype=torch.bool)

        x = self.in_proj(modality_tokens.float())
        for block in self.blocks:
            x = block(x, active_mask=active_mask)
        x = self.final_ln(x)

        # Masked scoring / pooling across modality set elements.
        raw_scores = self.score_head(x).squeeze(-1)  # (B, M)
        masked_scores = raw_scores.masked_fill(~active_mask, float("-inf"))
        # Handle rows where all modalities are inactive.
        all_inactive = ~active_mask.any(dim=1)
        if all_inactive.any():
            masked_scores = masked_scores.clone()
            masked_scores[all_inactive] = 0.0

        weights = F.softmax(masked_scores, dim=-1)
        weights = weights * active_mask.float()
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom

        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)  # (B, D)
        gate = torch.sigmoid(self.gate_head(pooled))  # (B, 1)
        raw_correction = gate * self.correction_head(pooled)  # (B, H)

        active_count = active_mask.sum(dim=-1)  # (B,)
        apply_mask = (active_count >= self.min_modalities_to_apply).float().unsqueeze(-1)  # (B,1)
        applied_correction = raw_correction * apply_mask

        # Normalized entropy in [0,1] on active modalities (1=balanced).
        logw = torch.log(weights.clamp_min(1e-8))
        entropy = -(weights * logw).sum(dim=-1)  # (B,)
        max_entropy = torch.log(active_count.float().clamp_min(2.0))
        entropy_norm = torch.where(
            active_count >= 2,
            entropy / max_entropy.clamp_min(1e-6),
            torch.ones_like(entropy),
        )

        return {
            "raw_correction": raw_correction,
            "correction": applied_correction,
            "gate": gate,
            "weights": weights,
            "active_mask": active_mask,
            "active_count": active_count.float(),
            "entropy_norm": entropy_norm,
            "util_target_entropy": torch.full_like(entropy_norm, float(self.util_target_entropy)),
        }
