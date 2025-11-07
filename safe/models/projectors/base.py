"""Shared utilities for modality projectors.

This module centralizes boilerplate used by the audio and video projectors so
that modality-specific implementations remain lightweight.  It exposes a simple
registration decorator that we will later use when wiring a formal modality
registry inside ``SAFEModel``.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Type, TypeVar

import torch.nn as nn


T = TypeVar("T", bound="BaseProjector")


def resolve_activation(name: str) -> nn.Module:
    """Return an activation module given a config string."""
    normalized = name.lower()
    activations: Dict[str, Callable[[], nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }

    try:
        return activations[normalized]()
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported activation: {name}") from exc


def init_linear_stack(modules: Iterable[nn.Module], last_layer_std: float = 1e-5) -> None:
    """Initialize linear layers in a stack with sensible defaults.

    We apply Xavier uniform to every linear layer for stable gradients and then
    retune the last layer with a tiny normal init so that fusion tokens start
    close to zero (matching the behaviour that evolved in SAFE's audio path).
    """
    last_linear: nn.Linear | None = None

    for module in modules:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            last_linear = module

    if last_linear is not None:
        nn.init.normal_(last_linear.weight, mean=0.0, std=last_layer_std)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)


_PROJECTOR_REGISTRY: Dict[str, Type[T]] = {}


def register_projector(name: str) -> Callable[[Type[T]], Type[T]]:
    """Decorator that registers a projector implementation by modality name."""

    def _decorator(cls: Type[T]) -> Type[T]:
        _PROJECTOR_REGISTRY[name] = cls
        return cls

    return _decorator


class BaseProjector(nn.Module):
    """Common mixin providing debug logging controls for projectors."""

    def __init__(self) -> None:
        super().__init__()
        self.debug_logging: bool = False
        self._log_limit = 5
        self._logs_emitted = 0

    def set_debug_logging(self, enabled: bool, log_limit: int = 5) -> None:
        self.debug_logging = bool(enabled)
        self._log_limit = int(max(0, log_limit))
        self._logs_emitted = 0

    def _maybe_log(self, message: str) -> None:
        if self.debug_logging and self._logs_emitted < self._log_limit:
            print(message, flush=True)
            self._logs_emitted += 1


def get_registered_projectors() -> Dict[str, Type[T]]:
    """Return a copy of the global projector registry."""
    return dict(_PROJECTOR_REGISTRY)
