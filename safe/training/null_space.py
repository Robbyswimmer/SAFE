"""Null-space projection utilities for SAFE training.

This module implements a simple null-space editing routine that protects
vision-language retention directions by projecting gradients away from a
reference subspace. The reference subspace is estimated from retention-only
batches (no audio) and refreshed periodically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NullSpaceConfig:
    """Configuration for the null-space projector."""

    max_rank: int = 8
    min_samples: int = 32
    max_samples: int = 128
    audio_ratio_threshold: float = 0.01  # Collect from VL-only batches (0% audio, with small tolerance)
    refresh_interval: Optional[int] = 2000
    verbose: bool = False


class NullSpaceProjector:
    """Tracks a protected gradient subspace and projects future gradients away."""

    def __init__(
        self,
        params: Dict[str, nn.Parameter],
        config: Optional[NullSpaceConfig] = None,
    ) -> None:
        if not params:
            raise ValueError("NullSpaceProjector requires at least one trainable parameter")

        self.params = params
        self.config = config or NullSpaceConfig()

        # Gradient sample buffers per parameter
        self._buffers: Dict[str, List[torch.Tensor]] = {name: [] for name in self.params}
        self._bases: Dict[str, torch.Tensor] = {}

        self._samples_collected: int = 0
        self._last_refresh_step: int = 0
        self._ready: bool = False

    def reset(self, step: int = 0) -> None:
        """Reset buffers and bases (e.g. when refreshing the subspace)."""
        self._buffers = {name: [] for name in self.params}
        self._bases = {}
        self._samples_collected = 0
        self._ready = False
        self._last_refresh_step = step

        if self.config.verbose:
            print("[NullSpace] Resetting protected subspace")

    def _collectable(self, has_audio: torch.Tensor) -> bool:
        """
        Determine if this batch should be used for collecting VL retention gradients.

        We want to protect VL-only directions, so we collect from VL-only batches
        (audio_ratio ≈ 0), NOT from mixed batches.
        """
        if has_audio is None or has_audio.numel() == 0:
            # Empty batch - treat as VL-only for safety
            return True
        audio_ratio = has_audio.float().mean().item()
        # Collect only from VL-only batches (with small tolerance for floating point)
        return audio_ratio <= self.config.audio_ratio_threshold

    def _maybe_refresh(self, step: int) -> None:
        if not self.config.refresh_interval:
            return
        if step - self._last_refresh_step >= self.config.refresh_interval:
            self.reset(step=step)

    def observe(self, *, step: int, has_audio: Optional[torch.Tensor]) -> None:
        """Record the current gradient snapshot for subspace estimation."""
        self._maybe_refresh(step)

        if not self._collectable(has_audio):
            return

        collected_this_step = False
        for name, param in self.params.items():
            grad = param.grad
            if grad is None:
                continue
            flat = grad.detach().view(-1)
            if torch.count_nonzero(flat).item() == 0:
                continue

            self._buffers[name].append(flat.to(dtype=torch.float32, device="cpu"))
            if len(self._buffers[name]) > self.config.max_samples:
                self._buffers[name].pop(0)
            collected_this_step = True

        if collected_this_step:
            self._samples_collected += 1

        if not self._ready and self._samples_collected >= self.config.min_samples:
            self._build_bases()

    def _build_bases(self) -> None:
        """Compute orthonormal bases for each parameter buffer."""
        new_bases: Dict[str, torch.Tensor] = {}
        rank = self.config.max_rank

        for name, samples in self._buffers.items():
            if not samples:
                continue
            stacked = torch.stack(samples, dim=0)  # (num_samples, dim)
            try:
                # Work in float32 on CPU for numerical stability
                stacked = stacked.to(dtype=torch.float32, device="cpu")
                # Use SVD to obtain orthonormal directions associated with largest variance
                u, s, vh = torch.linalg.svd(stacked, full_matrices=False)
                effective_rank = min(rank, vh.size(0))
                basis = vh[:effective_rank]
                basis = F.normalize(basis, dim=1, eps=1e-6)
                new_bases[name] = basis
            except RuntimeError as exc:
                if self.config.verbose:
                    print(f"[NullSpace] SVD failed for {name}: {exc}")

        if new_bases:
            self._bases = new_bases
            self._buffers = {name: [] for name in self.params}
            self._ready = True
            if self.config.verbose:
                active = ", ".join(f"{k}:{v.size(0)}" for k, v in self._bases.items())
                print(f"[NullSpace] Built bases with ranks {active}")

    def project(self) -> None:
        """Project current gradients onto the null space (remove protected directions)."""
        if not self._ready:
            return

        for name, param in self.params.items():
            grad = param.grad
            if grad is None or name not in self._bases:
                continue
            basis = self._bases[name]
            if basis.numel() == 0:
                continue

            device = grad.device
            flat_grad = grad.view(-1)
            basis_device = basis.to(device=device, dtype=flat_grad.dtype)

            # Remove components along protected directions
            coeffs = torch.matmul(basis_device, flat_grad)
            correction = torch.matmul(basis_device.t(), coeffs)
            flat_grad = flat_grad - correction
            grad.copy_(flat_grad.view_as(grad))

    @property
    def ready(self) -> bool:
        return self._ready
