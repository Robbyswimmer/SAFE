"""
SAFEMultiModalModel: Unified multi-modal model supporting audio + point cloud.

Combines CLAP (audio) and PointBERT (point cloud) encoders with shared fusion
adapters to test modality composition. Designed for the composition ablation
study comparing pre-FFN vs KV augmentation architectures.
"""

from __future__ import annotations

import sys
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_vl import BaseVLModel
from .audio_encoders import CLAPAudioEncoder
from .pointcloud_encoders import PointBERTEncoder
from .projectors import AudioProjector, TokenSetProjector
from .fusion_adapter import MultiLayerFusionAdapter
from .kv_augmentation import (
    KVAugmentationHookManager,
    KVAugmentationAdapter,
    MultiModalKVAugmentationHookManager,
)
from .layer_hooks import LayerHookManager


class SAFEMultiModalModel(nn.Module):
    """
    SAFE architecture with both audio and point cloud modalities.

    Architecture:
        Audio → CLAP → Projector ─────┐
                                      ├─→ Fusion → LLM → Output
        PointCloud → PointBERT → Projector ─┘

    Supports:
    - Training individual modality adapters separately
    - Loading both adapters for composition evaluation
    - Pre-FFN residual fusion OR KV augmentation
    """

    def __init__(
        self,
        # Base VL configuration
        llm_model_name: str = "llava-hf/llava-1.5-13b-hf",
        vision_model_name: str = "openai/clip-vit-large-patch14",

        # Audio configuration
        audio_encoder_type: str = "clap",
        audio_encoder_config: Optional[Dict] = None,
        audio_embed_dim: int = 512,
        num_audio_tokens: int = 8,

        # Point cloud configuration
        pointcloud_encoder_type: str = "pointbert",
        pointcloud_encoder_config: Optional[Dict] = None,
        pointcloud_embed_dim: int = 768,
        num_pointcloud_tokens: int = 8,

        # Projector configuration
        projector_type: str = "standard",
        projector_config: Optional[Dict] = None,

        # Fusion configuration
        fusion_type: str = "multilayer",
        fusion_layer_indices: Optional[List[int]] = None,
        lora_rank: int = 8,
        fusion_config: Optional[Dict] = None,

        # Training configuration
        freeze_base_vl: bool = True,
        freeze_audio_encoder: bool = True,
        freeze_pointcloud_encoder: bool = True,
        label_smoothing: float = 0.0,

        # Model dimensions
        llm_hidden_size: int = 5120,
    ):
        super().__init__()

        self.llm_hidden_size = llm_hidden_size
        self.audio_embed_dim = audio_embed_dim
        self.pointcloud_embed_dim = pointcloud_embed_dim
        self.num_audio_tokens = num_audio_tokens
        self.num_pointcloud_tokens = num_pointcloud_tokens
        self.label_smoothing = label_smoothing

        # Initialize optional components
        self.kv_adapters = None
        self.kv_hook_manager = None
        self.fusion_adapter = None

        # Track which modalities are active
        self.audio_enabled = True
        self.pointcloud_enabled = True

        # Debug settings
        self.debug_fusion_stats = False
        self.debug_fusion_log_every = 50

        # =====================================================================
        # Initialize Base VL Model
        # =====================================================================
        print(f"[SAFE-MM] Initializing BaseVLModel...", flush=True)
        self.base_vl = BaseVLModel(
            llm_model_name=llm_model_name,
            vision_model_name=vision_model_name,
            llm_hidden_size=llm_hidden_size,
            freeze_vision=freeze_base_vl,
            freeze_llm=freeze_base_vl,
        )
        print(f"[SAFE-MM] ✓ BaseVLModel initialized", flush=True)

        # =====================================================================
        # Initialize Audio Encoder (CLAP)
        # =====================================================================
        print(f"[SAFE-MM] Initializing audio encoder ({audio_encoder_type})...", flush=True)
        audio_encoder_config = audio_encoder_config or {}

        if audio_encoder_type == "clap":
            self.audio_encoder = CLAPAudioEncoder(
                freeze=freeze_audio_encoder,
                **audio_encoder_config
            )
            self.audio_embed_dim = self.audio_encoder.audio_embed_dim
        else:
            raise ValueError(f"Unsupported audio encoder: {audio_encoder_type}")

        print(f"[SAFE-MM] ✓ Audio encoder initialized (embed_dim={self.audio_embed_dim})", flush=True)

        # =====================================================================
        # Initialize Point Cloud Encoder (PointBERT)
        # =====================================================================
        print(f"[SAFE-MM] Initializing point cloud encoder ({pointcloud_encoder_type})...", flush=True)
        pointcloud_encoder_config = pointcloud_encoder_config or {}

        if pointcloud_encoder_type in ["pointbert", "pointnet"]:
            self.pointcloud_encoder = PointBERTEncoder(
                freeze=freeze_pointcloud_encoder,
                **pointcloud_encoder_config
            )
            self.pointcloud_embed_dim = self.pointcloud_encoder.pointcloud_embed_dim
        else:
            raise ValueError(f"Unsupported point cloud encoder: {pointcloud_encoder_type}")

        print(f"[SAFE-MM] ✓ Point cloud encoder initialized (embed_dim={self.pointcloud_embed_dim})", flush=True)

        # =====================================================================
        # Check Fusion Mode
        # =====================================================================
        fusion_config = fusion_config or {}
        fusion_layer_indices = fusion_layer_indices or [1, 5, 9, 13, 17, 21]
        self.fusion_layer_indices = fusion_layer_indices
        self.fusion_mode = fusion_config.get("fusion_mode", "residual")
        self.enable_kv_augmentation = (self.fusion_mode == "kv_augment")
        self.fusion_injection_point = fusion_config.get("injection_point", "pre_ffn")

        # Determine projector output dimension
        is_kv_augment = self.enable_kv_augmentation
        projector_config = dict(projector_config) if projector_config else {}
        projector_output_dim = projector_config.pop("output_dim", None)

        # =====================================================================
        # Initialize Audio Projector
        # =====================================================================
        print(f"[SAFE-MM] Initializing audio projector...", flush=True)
        audio_output_dim = projector_output_dim or (self.audio_embed_dim if is_kv_augment else llm_hidden_size)

        self.audio_projector = AudioProjector(
            audio_embed_dim=self.audio_embed_dim,
            llm_hidden_size=llm_hidden_size,
            num_audio_tokens=num_audio_tokens,
            output_dim=audio_output_dim,
            **projector_config
        )
        print(f"[SAFE-MM] ✓ Audio projector initialized (output_dim={audio_output_dim})", flush=True)

        # =====================================================================
        # Initialize Point Cloud Projector
        # =====================================================================
        print(f"[SAFE-MM] Initializing point cloud projector...", flush=True)
        pc_output_dim = projector_output_dim or (self.pointcloud_embed_dim if is_kv_augment else llm_hidden_size)

        self.pointcloud_projector = AudioProjector(
            audio_embed_dim=self.pointcloud_embed_dim,
            llm_hidden_size=llm_hidden_size,
            num_audio_tokens=num_pointcloud_tokens,
            output_dim=pc_output_dim,
            **projector_config
        )
        print(f"[SAFE-MM] ✓ Point cloud projector initialized (output_dim={pc_output_dim})", flush=True)

        # =====================================================================
        # Initialize Fusion Adapter (Multi-Modal)
        # =====================================================================
        print(f"[SAFE-MM] Initializing fusion adapter ({fusion_type}, mode={self.fusion_mode})...", flush=True)

        # Get modalities config - supports both audio and pointcloud
        modalities = fusion_config.get("modalities", {
            "audio": {"layer_indices": fusion_layer_indices},
            "pointcloud": {"layer_indices": fusion_layer_indices},
        })

        if not self.enable_kv_augmentation:
            # Pre-FFN or post-layer residual fusion
            self.fusion_adapter = MultiLayerFusionAdapter(
                hidden_size=llm_hidden_size,
                fusion_layer_indices=fusion_layer_indices,
                modalities=modalities,
                lora_rank=lora_rank,
                num_attention_heads=fusion_config.get("num_attention_heads", 40),
                lora_alpha=fusion_config.get("lora_alpha", 16.0),
                lora_dropout=fusion_config.get("lora_dropout", 0.1),
                attention_dropout=fusion_config.get("attention_dropout", 0.1),
                use_tokenwise_gate=fusion_config.get("use_tokenwise_gate", False),
                use_bottleneck=fusion_config.get("use_bottleneck", True),
                bottleneck_dim=fusion_config.get("bottleneck_dim", 256),
                fusion_mode=fusion_config.get("fusion_mode", "residual"),
                use_ffn=fusion_config.get("use_ffn", True),
                ffn_expansion=fusion_config.get("ffn_expansion", 2.0),
                use_pre_norm=fusion_config.get("use_pre_norm", True),
            )
            print(f"[SAFE-MM] ✓ MultiLayerFusionAdapter initialized", flush=True)
            print(f"[SAFE-MM]   Modalities: {list(modalities.keys())}", flush=True)
            print(f"[SAFE-MM]   Fusion adapters: {list(self.fusion_adapter.fusion_adapters.keys())}", flush=True)
        else:
            # KV Augmentation mode
            print(f"[SAFE-MM] Initializing KV Augmentation adapters...", flush=True)
            self._init_kv_augmentation(fusion_config, modalities)

        self.enable_midlayer_fusion = (fusion_type == "multilayer") and not self.enable_kv_augmentation

        print(f"[SAFE-MM] ✓ SAFEMultiModalModel initialization complete", flush=True)
        sys.stdout.flush()

    def _init_kv_augmentation(self, fusion_config: Dict, modalities: Dict):
        """Initialize KV augmentation adapters for both modalities."""
        # Get layer indices from modalities config
        audio_layers = modalities.get("audio", {}).get("layer_indices", self.fusion_layer_indices)
        pc_layers = modalities.get("pointcloud", {}).get("layer_indices", self.fusion_layer_indices)
        all_layers = sorted(set(audio_layers + pc_layers))

        # Get attention config
        num_attention_heads = fusion_config.get("num_attention_heads", 40)
        head_dim = fusion_config.get("head_dim", 128)
        num_kv_heads = fusion_config.get("num_key_value_heads", num_attention_heads)

        # Create KV adapters for each modality at each layer
        self.kv_adapters = nn.ModuleDict()
        self._kv_fusion_layers = all_layers
        self._audio_kv_layers = audio_layers
        self._pc_kv_layers = pc_layers

        for layer_idx in all_layers:
            # Audio adapter at this layer
            if layer_idx in audio_layers:
                self.kv_adapters[f"audio:{layer_idx}"] = KVAugmentationAdapter(
                    hidden_size=self.llm_hidden_size,
                    num_heads=num_attention_heads,
                    head_dim=head_dim,
                    num_key_value_heads=num_kv_heads,
                    bottleneck_dim=fusion_config.get("bottleneck_dim", 64),
                    dropout=fusion_config.get("dropout", 0.1),
                    query_adapter_rank=fusion_config.get("query_adapter_rank", 16),
                    input_dim=self.audio_embed_dim,
                )

            # Point cloud adapter at this layer
            if layer_idx in pc_layers:
                self.kv_adapters[f"pointcloud:{layer_idx}"] = KVAugmentationAdapter(
                    hidden_size=self.llm_hidden_size,
                    num_heads=num_attention_heads,
                    head_dim=head_dim,
                    num_key_value_heads=num_kv_heads,
                    bottleneck_dim=fusion_config.get("bottleneck_dim", 64),
                    dropout=fusion_config.get("dropout", 0.1),
                    query_adapter_rank=fusion_config.get("query_adapter_rank", 16),
                    input_dim=self.pointcloud_embed_dim,
                )

        print(f"[SAFE-MM] ✓ KV Augmentation adapters initialized", flush=True)
        print(f"[SAFE-MM]   Audio layers: {audio_layers}", flush=True)
        print(f"[SAFE-MM]   Point cloud layers: {pc_layers}", flush=True)
        print(f"[SAFE-MM]   KV adapters: {list(self.kv_adapters.keys())}", flush=True)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to projected tokens."""
        if not self.audio_enabled:
            return None

        # Encode with CLAP
        audio_features = self.audio_encoder(audio)

        # Project to token space
        audio_tokens = self.audio_projector(audio_features)

        return audio_tokens

    def encode_pointcloud(self, pointcloud: torch.Tensor) -> torch.Tensor:
        """Encode point cloud to projected tokens."""
        if not self.pointcloud_enabled:
            return None

        # Encode with PointBERT
        pc_features = self.pointcloud_encoder(pointcloud)

        # Project to token space
        pc_tokens = self.pointcloud_projector(pc_features)

        return pc_tokens

    def forward(
        self,
        text: Optional[str] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        pointcloud: Optional[torch.Tensor] = None,
        audio_tokens: Optional[torch.Tensor] = None,
        pointcloud_tokens: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        pointcloud_attention_mask: Optional[torch.Tensor] = None,
        gate: Union[float, Dict[str, float]] = 1.0,
        return_hidden_states: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass with optional audio and/or point cloud inputs.

        Args:
            text: Optional text input
            input_ids: Token IDs
            attention_mask: Attention mask for tokens
            audio: Raw audio waveform (will be encoded)
            pointcloud: Raw point cloud (will be encoded)
            audio_tokens: Pre-encoded audio tokens (skips encoding)
            pointcloud_tokens: Pre-encoded point cloud tokens (skips encoding)
            audio_attention_mask: Mask for audio tokens
            pointcloud_attention_mask: Mask for point cloud tokens
            gate: Gating factor (scalar or per-modality dict)
            return_hidden_states: Whether to return hidden states

        Returns:
            Dict with 'logits', 'hidden_states' (if requested), etc.
        """
        # Encode modalities if raw inputs provided
        if audio is not None and audio_tokens is None:
            audio_tokens = self.encode_audio(audio)

        if pointcloud is not None and pointcloud_tokens is None:
            pointcloud_tokens = self.encode_pointcloud(pointcloud)

        # Handle text input
        if text is not None and input_ids is None:
            tokenizer = getattr(self.base_vl, "tokenizer", None) or getattr(self.base_vl, "processor", None)
            if tokenizer is not None:
                encoded = tokenizer(text, return_tensors="pt", padding=True)
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")

        # Get device
        device = next(self.parameters()).device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Build modality tokens dict
        modality_tokens = {}
        modality_masks = {}

        if audio_tokens is not None and self.audio_enabled:
            audio_tokens = audio_tokens.to(device)
            modality_tokens["audio"] = audio_tokens
            if audio_attention_mask is not None:
                modality_masks["audio"] = audio_attention_mask.to(device)

        if pointcloud_tokens is not None and self.pointcloud_enabled:
            pointcloud_tokens = pointcloud_tokens.to(device)
            modality_tokens["pointcloud"] = pointcloud_tokens
            if pointcloud_attention_mask is not None:
                modality_masks["pointcloud"] = pointcloud_attention_mask.to(device)

        # Convert scalar gate to dict
        if isinstance(gate, (int, float)):
            gate_dict = {mod: float(gate) for mod in modality_tokens.keys()}
        else:
            gate_dict = gate

        # Prepare LLM inputs
        llm_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": return_hidden_states,
        }

        # Apply fusion
        if self.enable_midlayer_fusion and modality_tokens:
            # Pre-FFN or post-layer fusion via hooks
            fusion_layers = self._resolve_fusion_layers()
            hook_manager = LayerHookManager(
                model=self.base_vl.llm,
                fusion_adapter=self.fusion_adapter,
                fusion_layers=fusion_layers,
                injection_point=self.fusion_injection_point,
            )
            hook_manager.register_hooks(
                modality_tokens=modality_tokens,
                modality_masks=modality_masks if modality_masks else None,
                gate=gate_dict,
                debug_fusion=self.debug_fusion_stats,
                debug_fusion_log_every=self.debug_fusion_log_every,
            )
            try:
                outputs = self.base_vl.llm(**llm_inputs)
            finally:
                hook_manager.remove_hooks()

        elif self.enable_kv_augmentation and modality_tokens:
            # KV augmentation fusion
            outputs = self._forward_kv_augmentation(
                llm_inputs=llm_inputs,
                modality_tokens=modality_tokens,
                modality_masks=modality_masks,
                gate_dict=gate_dict,
            )
        else:
            # No modality tokens - standard LLM forward
            outputs = self.base_vl.llm(**llm_inputs)

        # Build result
        result = {
            "logits": outputs.logits if hasattr(outputs, "logits") else outputs[0],
        }

        if return_hidden_states and hasattr(outputs, "hidden_states"):
            result["hidden_states"] = outputs.hidden_states
            # Also return last hidden state for convenience
            if outputs.hidden_states:
                result["last_hidden_state"] = outputs.hidden_states[-1]

        return result

    def _forward_kv_augmentation(
        self,
        llm_inputs: Dict,
        modality_tokens: Dict[str, torch.Tensor],
        modality_masks: Dict[str, torch.Tensor],
        gate_dict: Dict[str, float],
    ):
        """
        Forward pass with KV augmentation for multiple modalities.

        Uses MultiModalKVAugmentationHookManager which handles N modalities,
        each with its own adapter. Each modality gets:
        - Its own K,V projections
        - Its own query adapter (ΔQ)
        - Independent gating

        The outputs are combined as:
            output = text_output + sum(gate_i * modality_i_output)
        """
        if self.kv_adapters is None:
            raise RuntimeError("KV adapters not initialized")

        # Initialize multi-modal hook manager if needed
        if self.kv_hook_manager is None:
            self.kv_hook_manager = MultiModalKVAugmentationHookManager(
                model=self.base_vl.llm,
                kv_adapters=self.kv_adapters,
            )
            self.kv_hook_manager.wrap_attention_modules()

        # Filter to only active modalities with tokens
        active_tokens = {k: v for k, v in modality_tokens.items() if v is not None}

        if not active_tokens:
            # No modality tokens - standard forward
            return self.base_vl.llm(**llm_inputs)

        # Inject modality tokens
        self.kv_hook_manager.inject_modality_tokens(
            modality_tokens=active_tokens,
            modality_masks=modality_masks if modality_masks else None,
            modality_gates=gate_dict,
        )

        try:
            outputs = self.base_vl.llm(**llm_inputs)
        finally:
            # Clear tokens after forward
            self.kv_hook_manager.clear_modality_tokens()

        return outputs

    def _resolve_fusion_layers(self) -> Dict[str, List[int]]:
        """Get fusion layer indices for each modality."""
        if self.fusion_adapter is not None and hasattr(self.fusion_adapter, "fusion_layers"):
            return {
                modality: list(indices)
                for modality, indices in self.fusion_adapter.fusion_layers.items()
            }
        return {
            "audio": self.fusion_layer_indices,
            "pointcloud": self.fusion_layer_indices,
        }

    def enable_modality(self, modality: str, enabled: bool = True):
        """Enable or disable a specific modality."""
        if modality == "audio":
            self.audio_enabled = enabled
        elif modality == "pointcloud":
            self.pointcloud_enabled = enabled
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        params = []

        # Audio projector
        params.extend(p for p in self.audio_projector.parameters() if p.requires_grad)

        # Point cloud projector
        params.extend(p for p in self.pointcloud_projector.parameters() if p.requires_grad)

        # Fusion adapter
        if self.fusion_adapter is not None:
            params.extend(p for p in self.fusion_adapter.parameters() if p.requires_grad)

        # KV adapters
        if self.kv_adapters is not None:
            params.extend(p for p in self.kv_adapters.parameters() if p.requires_grad)

        return params

    def save_adapters(self, path: str, modality: Optional[str] = None):
        """
        Save adapter weights.

        Args:
            path: Path to save checkpoint
            modality: If specified, only save adapters for this modality.
                      Otherwise save all adapters.
        """
        state_dict = {}

        if modality is None or modality == "audio":
            state_dict["audio_projector"] = self.audio_projector.state_dict()

        if modality is None or modality == "pointcloud":
            state_dict["pointcloud_projector"] = self.pointcloud_projector.state_dict()

        if self.fusion_adapter is not None:
            if modality is None:
                state_dict["fusion_adapter"] = self.fusion_adapter.state_dict()
            else:
                # Save only adapters for specified modality
                adapter_state = {}
                for key, module in self.fusion_adapter.fusion_adapters.items():
                    if key.startswith(f"{modality}:"):
                        adapter_state[key] = module.state_dict()
                state_dict[f"fusion_adapter_{modality}"] = adapter_state

        if self.kv_adapters is not None:
            if modality is None:
                state_dict["kv_adapters"] = self.kv_adapters.state_dict()
            else:
                # Save only KV adapters for specified modality
                kv_state = {}
                for key, module in self.kv_adapters.items():
                    if key.startswith(f"{modality}:"):
                        kv_state[key] = module.state_dict()
                state_dict[f"kv_adapters_{modality}"] = kv_state

        torch.save(state_dict, path)
        print(f"[SAFE-MM] Saved adapters to {path}", flush=True)

    def load_adapters(self, path: str, modality: Optional[str] = None, strict: bool = False):
        """
        Load adapter weights.

        Args:
            path: Path to checkpoint
            modality: If specified, only load adapters for this modality.
            strict: Whether to require exact key matching
        """
        state_dict = torch.load(path, map_location="cpu")

        if modality is None or modality == "audio":
            if "audio_projector" in state_dict:
                self.audio_projector.load_state_dict(state_dict["audio_projector"], strict=strict)
                print(f"[SAFE-MM] Loaded audio projector from {path}", flush=True)

        if modality is None or modality == "pointcloud":
            if "pointcloud_projector" in state_dict:
                self.pointcloud_projector.load_state_dict(state_dict["pointcloud_projector"], strict=strict)
                print(f"[SAFE-MM] Loaded pointcloud projector from {path}", flush=True)

        if self.fusion_adapter is not None:
            if "fusion_adapter" in state_dict:
                self.fusion_adapter.load_state_dict(state_dict["fusion_adapter"], strict=strict)
                print(f"[SAFE-MM] Loaded fusion adapter from {path}", flush=True)
            elif f"fusion_adapter_{modality}" in state_dict:
                # Load modality-specific adapters
                adapter_state = state_dict[f"fusion_adapter_{modality}"]
                for key, module_state in adapter_state.items():
                    if key in self.fusion_adapter.fusion_adapters:
                        self.fusion_adapter.fusion_adapters[key].load_state_dict(module_state, strict=strict)
                print(f"[SAFE-MM] Loaded {modality} fusion adapters from {path}", flush=True)

        if self.kv_adapters is not None:
            if "kv_adapters" in state_dict:
                self.kv_adapters.load_state_dict(state_dict["kv_adapters"], strict=strict)
                print(f"[SAFE-MM] Loaded KV adapters from {path}", flush=True)
            elif f"kv_adapters_{modality}" in state_dict:
                # Load modality-specific KV adapters
                kv_state = state_dict[f"kv_adapters_{modality}"]
                for key, module_state in kv_state.items():
                    if key in self.kv_adapters:
                        self.kv_adapters[key].load_state_dict(module_state, strict=strict)
                print(f"[SAFE-MM] Loaded {modality} KV adapters from {path}", flush=True)
