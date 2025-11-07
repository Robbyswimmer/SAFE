from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, MutableMapping, Optional

import torch
import torch.nn as nn

from .projectors import BaseProjector


@dataclass
class ModalityComponents:
    """Bundle encoder/projector objects alongside runtime metadata."""

    name: str
    encoder: nn.Module
    projector: BaseProjector
    gate: float = 0.0
    gate_warmup_steps: Optional[int] = None
    dtype: Optional[torch.dtype] = None
    token_key: str = "tokens"
    attention_key: Optional[str] = None
    metadata: MutableMapping[str, object] = field(default_factory=dict)

    def set_gate(self, value: float) -> None:
        self.gate = float(value)

    def set_dtype(self, dtype: torch.dtype) -> None:
        self.dtype = dtype


class ModalityRegistry:
    """Lightweight container tracking SAFE modalities at runtime."""

    def __init__(self) -> None:
        self._modalities: "OrderedDict[str, ModalityComponents]" = OrderedDict()

    # ------------------------------------------------------------------
    def register(self, components: ModalityComponents, *, overwrite: bool = False) -> None:
        if components.name in self._modalities and not overwrite:
            raise ValueError(f"Modality '{components.name}' already registered")
        self._modalities[components.name] = components

    def unregister(self, name: str) -> None:
        self._modalities.pop(name, None)

    def get(self, name: str) -> ModalityComponents:
        try:
            return self._modalities[name]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Modality '{name}' not registered") from exc

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial
        return name in self._modalities

    def items(self) -> Iterable[tuple[str, ModalityComponents]]:
        return self._modalities.items()

    def values(self) -> Iterable[ModalityComponents]:
        return self._modalities.values()

    def names(self) -> Iterable[str]:
        return self._modalities.keys()

    def __iter__(self) -> Iterator[ModalityComponents]:
        return iter(self._modalities.values())

    # ------------------------------------------------------------------
    def gates(self) -> Dict[str, float]:
        return {name: comp.gate for name, comp in self._modalities.items()}

    def set_gate(self, name: str, value: float) -> None:
        self.get(name).set_gate(value)

    def set_all_gates(self, value: float) -> None:
        for comp in self._modalities.values():
            comp.set_gate(value)

    def set_dtype(self, name: str, dtype: torch.dtype) -> None:
        self.get(name).set_dtype(dtype)

    def configure_gate_warmup(self, name: str, steps: Optional[int]) -> None:
        self.get(name).gate_warmup_steps = steps

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Dict[str, object]]:
        """Serialise the registry into a plain dict for logging/debug."""
        output: Dict[str, Dict[str, object]] = {}
        for name, comp in self._modalities.items():
            output[name] = {
                "gate": comp.gate,
                "gate_warmup_steps": comp.gate_warmup_steps,
                "dtype": str(comp.dtype) if comp.dtype is not None else None,
                "token_key": comp.token_key,
                "attention_key": comp.attention_key,
                "metadata_keys": sorted(comp.metadata.keys()),
            }
        return output
