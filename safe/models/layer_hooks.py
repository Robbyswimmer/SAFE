import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union

from .fusion_adapter import MultiLayerFusionAdapter


def _is_module_list(obj: Any) -> bool:
    return isinstance(obj, (list, nn.ModuleList, tuple))


class FusionHook:
    """Forward hook that injects modality fusion at a specific decoder layer output."""

    def __init__(
        self,
        layer_idx: int,
        fusion_adapter: MultiLayerFusionAdapter,
        modalities: List[str],
        modality_tokens: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
        gate: Any = 1.0,
        supervised_mask: Optional[torch.Tensor] = None,
        debug_fusion: bool = False,
        debug_fusion_log_every: int = 50,
    ) -> None:
        self.layer_idx = layer_idx
        self.fusion_adapter = fusion_adapter
        self.modalities = modalities
        self.modality_tokens = modality_tokens
        self.modality_masks = modality_masks or {}
        self.gate = gate
        self.supervised_mask = supervised_mask
        self.debug_fusion = bool(debug_fusion)
        self.debug_fusion_log_every = int(debug_fusion_log_every)
        self._call_idx = 0

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

        if self.debug_fusion and self.debug_fusion_log_every > 0:
            self._call_idx += 1
            if (self._call_idx % self.debug_fusion_log_every) == 0:
                try:
                    hs = hidden_states.detach().float()
                    fu = fused.detach().float()
                    delta = fu - hs
                    hs_norm = hs.norm(dim=-1).mean().item()
                    delta_norm = delta.norm(dim=-1).mean().item()
                    ratio = delta_norm / (hs_norm + 1e-6)
                    delta_max = delta.abs().max().item()
                    hs_max = hs.abs().max().item()
                    finite = torch.isfinite(fu).all().item()

                    gate_repr = self.gate
                    if isinstance(self.gate, dict):
                        gate_repr = {k: float(v) if isinstance(v, (int, float)) else v for k, v in self.gate.items()}
                    print(
                        f"[FUSION] layer={self.layer_idx} point={type(module).__name__} "
                        f"hs_norm={hs_norm:.4f} delta_norm={delta_norm:.4f} ratio={ratio:.4f} "
                        f"hs_max={hs_max:.3g} delta_max={delta_max:.3g} finite={finite} gate={gate_repr}",
                        flush=True,
                    )
                except Exception:
                    pass

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


class PreFFNFusionHook:
    """
    Forward pre-hook that injects modality fusion at the input to a layer's
    feed-forward (MLP) block. This allows fusion *before* the FFN, i.e.,
    after self-attention + first Add&Norm.
    """

    def __init__(
        self,
        layer_idx: int,
        fusion_adapter: MultiLayerFusionAdapter,
        modalities: Optional[List[str]],
        modality_tokens: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
        gate: Any = 1.0,
        supervised_mask: Optional[torch.Tensor] = None,
        debug_fusion: bool = False,
        debug_fusion_log_every: int = 50,
    ) -> None:
        self.layer_idx = layer_idx
        self.fusion_adapter = fusion_adapter
        self.modalities = modalities or []
        self.modality_tokens = modality_tokens
        self.modality_masks = modality_masks or {}
        self.gate = gate
        self.supervised_mask = supervised_mask
        self.debug_fusion = bool(debug_fusion)
        self.debug_fusion_log_every = int(debug_fusion_log_every)
        self._call_idx = 0

    def __call__(self, module: nn.Module, inputs: tuple) -> tuple:
        if not inputs:
            return inputs

        hidden_states = inputs[0]
        if not torch.is_tensor(hidden_states):
            return inputs

        fused = self.fusion_adapter.apply_fusion_at_layer(
            layer_idx=self.layer_idx,
            hidden_states=hidden_states,
            modality_tokens=self.modality_tokens,
            modality_masks=self.modality_masks,
            gate=self.gate,
            supervised_mask=self.supervised_mask,
        )

        if self.debug_fusion and self.debug_fusion_log_every > 0:
            self._call_idx += 1
            if (self._call_idx % self.debug_fusion_log_every) == 0:
                try:
                    hs = hidden_states.detach().float()
                    fu = fused.detach().float()
                    delta = fu - hs
                    hs_norm = hs.norm(dim=-1).mean().item()
                    delta_norm = delta.norm(dim=-1).mean().item()
                    ratio = delta_norm / (hs_norm + 1e-6)
                    delta_max = delta.abs().max().item()
                    hs_max = hs.abs().max().item()
                    finite = torch.isfinite(fu).all().item()
                    gate_repr = self.gate
                    if isinstance(self.gate, dict):
                        gate_repr = {k: float(v) if isinstance(v, (int, float)) else v for k, v in self.gate.items()}
                    print(
                        f"[FUSION] layer={self.layer_idx} point=pre_ffn "
                        f"hs_norm={hs_norm:.4f} delta_norm={delta_norm:.4f} ratio={ratio:.4f} "
                        f"hs_max={hs_max:.3g} delta_max={delta_max:.3g} finite={finite} gate={gate_repr}",
                        flush=True,
                    )
                except Exception:
                    pass

        # Replace the first positional arg (hidden states) with fused version
        if len(inputs) == 1:
            return (fused,)
        return (fused, *inputs[1:])


class LayerHookManager:
    """Manages registration and cleanup of decoder layer fusion hooks."""

    def __init__(
        self,
        model: nn.Module,
        fusion_adapter: MultiLayerFusionAdapter,
        fusion_layers: Union[Dict[str, List[int]], List[int]],
        injection_point: str = "post_layer",
    ) -> None:
        self.model = model
        self.fusion_adapter = fusion_adapter
        self.injection_point = injection_point
        self.fusion_layers = self._normalize_layer_mapping(fusion_layers)
        self.layer_modules = self._discover_layer_modules(model)
        self.layer_to_modalities = self._invert_layer_mapping(self.fusion_layers)
        self._handles: List[Any] = []

    def register_hooks(
        self,
        modality_tokens: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
        gate: Any = 1.0,
        active_layers: Optional[Union[List[int], set]] = None,
        supervised_mask: Optional[torch.Tensor] = None,
        debug_fusion: bool = False,
        debug_fusion_log_every: int = 50,
    ) -> None:
        self.remove_hooks()
        active_set = set(active_layers) if active_layers is not None else None
        requested_layers = {idx for indices in self.fusion_layers.values() for idx in indices}
        available_layers = set(self.layer_modules.keys())
        missing_layers = sorted(requested_layers - available_layers)
        if missing_layers and not getattr(self, "_warned_missing_layers", False):
            print(
                f"[LayerHookManager] Warning: requested fusion layers not found in model: {missing_layers}. "
                f"Available layer indices span [{min(available_layers)}..{max(available_layers)}] "
                f"({len(available_layers)} total).",
                flush=True,
            )
            self._warned_missing_layers = True
        # Inject either at the layer output (post_layer) or before FFN (pre_ffn)
        for idx, layer_module in self.layer_modules.items():
            modalities = self.layer_to_modalities.get(idx, [])
            if not modalities:
                continue
            if active_set is not None and idx not in active_set:
                continue

            if self.injection_point == "pre_ffn":
                # Try to locate the FFN/MLP submodule within the decoder layer
                ffn_module = getattr(layer_module, "mlp", None)
                if ffn_module is None:
                    # If we can't find an FFN, skip this layer for pre-FFN injection
                    continue
                hook = PreFFNFusionHook(
                    layer_idx=idx,
                    fusion_adapter=self.fusion_adapter,
                    modalities=modalities,
                    modality_tokens=modality_tokens,
                    modality_masks=modality_masks,
                    gate=gate,
                    supervised_mask=supervised_mask,
                    debug_fusion=debug_fusion,
                    debug_fusion_log_every=debug_fusion_log_every,
                )
                handle = ffn_module.register_forward_pre_hook(hook)
                self._handles.append(handle)
            else:
                hook = FusionHook(
                    layer_idx=idx,
                    fusion_adapter=self.fusion_adapter,
                    modalities=modalities,
                    modality_tokens=modality_tokens,
                    modality_masks=modality_masks,
                    gate=gate,
                    supervised_mask=supervised_mask,
                    debug_fusion=debug_fusion,
                    debug_fusion_log_every=debug_fusion_log_every,
                )
                handle = layer_module.register_forward_hook(hook)
                self._handles.append(handle)

    @property
    def num_hooks(self) -> int:
        return len(self._handles)

    def remove_hooks(self) -> None:
        if not self._handles:
            return
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _discover_layer_modules(self, model: nn.Module) -> Dict[int, nn.Module]:
        """
        Locate the decoder layer ModuleList for a variety of HF model wrappers.

        LLaVA (and other multimodal wrappers) can nest the actual language model
        several levels deep (e.g., `.model.language_model.model.layers`), so we
        do a bounded graph walk over common attribute names.
        """

        def _try_extract(candidate: Any) -> Optional[Dict[int, nn.Module]]:
            if candidate is None:
                return None
            if hasattr(candidate, "layers") and _is_module_list(getattr(candidate, "layers")):
                layers = getattr(candidate, "layers")
                return {i: layer for i, layer in enumerate(layers)}
            if hasattr(candidate, "h") and _is_module_list(getattr(candidate, "h")):
                layers = getattr(candidate, "h")
                return {i: layer for i, layer in enumerate(layers)}
            return None

        # Seed with typical top-level containers.
        seeds: List[Any] = [
            model,
            getattr(model, "model", None),
            getattr(model, "language_model", None),
            getattr(model, "decoder", None),
            getattr(model, "transformer", None),
        ]

        seen: set = set()
        queue: List[Any] = [s for s in seeds if s is not None]
        max_visits = 50  # bounded to avoid pathological graphs

        # Walk down common wrapper attributes.
        expand_attrs = ("model", "language_model", "decoder", "transformer")

        visits = 0
        while queue and visits < max_visits:
            candidate = queue.pop(0)
            visits += 1
            key = id(candidate)
            if key in seen:
                continue
            seen.add(key)

            extracted = _try_extract(candidate)
            if extracted is not None:
                return extracted

            for attr in expand_attrs:
                child = getattr(candidate, attr, None)
                if child is not None and id(child) not in seen:
                    queue.append(child)

        raise ValueError(
            "Unable to locate decoder layers for fusion hooks. "
            f"Tried {visits} candidates starting from type={type(model).__name__}."
        )

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
