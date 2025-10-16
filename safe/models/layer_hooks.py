import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union

from .fusion_adapter import MultiLayerFusionAdapter


def _is_module_list(obj: Any) -> bool:
    return isinstance(obj, (list, nn.ModuleList, tuple))


class FusionHook:
    """Forward hook that injects modality fusion at a specific decoder layer."""

    def __init__(
        self,
        layer_idx: int,
        fusion_adapter: MultiLayerFusionAdapter,
        modalities: List[str],
        modality_tokens: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
        gate: Any = 1.0,
        supervised_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self.layer_idx = layer_idx
        self.fusion_adapter = fusion_adapter
        self.modalities = modalities
        self.modality_tokens = modality_tokens
        self.modality_masks = modality_masks or {}
        self.gate = gate
        self.supervised_mask = supervised_mask

    def __call__(self, module: nn.Module, inputs: tuple, output: Any) -> Any:
        if not self.modalities:
            return output

        hidden_states, remainder = self._unpack_output(output)
        if hidden_states is None:
            return output

        fused = self.fusion_adapter.apply_fusion_at_layer(
            layer_idx=self.layer_idx,
            hidden_states=hidden_states,
            modality_tokens=self.modality_tokens,
            modality_masks=self.modality_masks,
            gate=self.gate,
            supervised_mask=self.supervised_mask,
        )

        return self._repack_output(fused, remainder, output_type=type(output))

    @staticmethod
    def _unpack_output(output: Any) -> tuple:
        if isinstance(output, torch.Tensor):
            return output, None

        if isinstance(output, (list, tuple)) and len(output) > 0:
            hidden_states = output[0]
            remainder = output[1:]
            return hidden_states, remainder

        if hasattr(output, "last_hidden_state"):
            hidden_states = output.last_hidden_state
            remainder = output
            return hidden_states, remainder

        return None, None

    @staticmethod
    def _repack_output(new_hidden: torch.Tensor, remainder: Any, output_type: type) -> Any:
        if remainder is None:
            return new_hidden

        if isinstance(remainder, (list, tuple)):
            if output_type is tuple:
                return (new_hidden, *remainder)
            return [new_hidden, *remainder]

        if hasattr(remainder, "__dict__"):
            setattr(remainder, "last_hidden_state", new_hidden)
            return remainder

        return new_hidden


class LayerHookManager:
    """Manages registration and cleanup of decoder layer fusion hooks."""

    def __init__(
        self,
        model: nn.Module,
        fusion_adapter: MultiLayerFusionAdapter,
        fusion_layers: Union[Dict[str, List[int]], List[int]],
    ) -> None:
        self.model = model
        self.fusion_adapter = fusion_adapter
        self.fusion_layers = self._normalize_layer_mapping(fusion_layers)
        self.layer_modules = self._discover_layer_modules(model)
        self.layer_to_modalities = self._invert_layer_mapping(self.fusion_layers)
        self._handles: List[Any] = []

    def register_hooks(
        self,
        modality_tokens: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
        gate: Any = 1.0,
        supervised_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self.remove_hooks()
        for idx, module in self.layer_modules.items():
            modalities = self.layer_to_modalities.get(idx, [])
            if not modalities:
                continue

            hook = FusionHook(
                layer_idx=idx,
                fusion_adapter=self.fusion_adapter,
                modalities=modalities,
                modality_tokens=modality_tokens,
                modality_masks=modality_masks,
                gate=gate,
                supervised_mask=supervised_mask,
            )
            handle = module.register_forward_hook(hook)
            self._handles.append(handle)

    def remove_hooks(self) -> None:
        if not self._handles:
            return
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _discover_layer_modules(self, model: nn.Module) -> Dict[int, nn.Module]:
        # Common HuggingFace decoder layouts
        candidates = [
            getattr(model, "model", None),
            getattr(model, "decoder", None),
            getattr(model, "transformer", None),
            model,
        ]

        for candidate in candidates:
            if candidate is None:
                continue

            if hasattr(candidate, "layers") and _is_module_list(candidate.layers):
                return {i: layer for i, layer in enumerate(candidate.layers)}

            if hasattr(candidate, "h") and _is_module_list(candidate.h):
                return {i: layer for i, layer in enumerate(candidate.h)}

        raise ValueError("Unable to locate decoder layers for fusion hooks")

    @staticmethod
    def _invert_layer_mapping(mapping: Dict[str, List[int]]) -> Dict[int, List[str]]:
        layer_to_modalities: Dict[int, List[str]] = {}
        for modality, indices in mapping.items():
            for idx in indices:
                layer_to_modalities.setdefault(idx, []).append(modality)
        return layer_to_modalities

    @staticmethod
    def _normalize_layer_mapping(
        mapping: Union[Dict[str, List[int]], List[int]]
    ) -> Dict[str, List[int]]:
        if isinstance(mapping, dict):
            return {key: list(value) for key, value in mapping.items()}
        if isinstance(mapping, (list, tuple)):
            return {"audio": list(mapping)}
        raise ValueError("fusion_layers must be a dict or list of layer indices")

    def __del__(self) -> None:
        self.remove_hooks()
