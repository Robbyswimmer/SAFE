"""
SAFEPointCloudModel: Point cloud variant of SAFE architecture.

Proves the SAFE architecture generalizes to other modalities by swapping
the encoder (CLAP → PointBERT) while reusing the projector and fusion.
"""

from __future__ import annotations

import sys
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_vl import BaseVLModel
from .pointcloud_encoders import PointBERTEncoder
from .projectors import AudioProjector, AdaptiveAudioProjector, TokenSetProjector
from .fusion_adapter import (
    MultiLayerFusionAdapter,
    LoRAFusionAdapter,
    GatedFusionAdapter,
)
from .kv_augmentation import KVAugmentationHookManager, KVAugmentationAdapter
from .layer_hooks import LayerHookManager


class SAFEPointCloudModel(nn.Module):
    """
    SAFE architecture adapted for point cloud modality.

    Architecture:
        PointCloud → Encoder → Projector → Fusion → LLM → Output

    Mirrors SAFEModel but replaces audio encoder with point cloud encoder.
    Reuses the same projector and fusion adapter classes (they're modality-agnostic).
    """

    def __init__(
        self,
        # Base VL configuration
        llm_model_name: str = "llava-hf/llava-1.5-13b-hf",
        vision_model_name: str = "openai/clip-vit-large-patch14",

        # Point cloud configuration
        pointcloud_encoder_type: str = "pointbert",
        pointcloud_encoder_config: Optional[Dict] = None,

        # Projector configuration (reuses AudioProjector)
        projector_type: str = "standard",
        num_tokens: int = 8,
        projector_config: Optional[Dict] = None,

        # Fusion configuration (reuses fusion adapters)
        fusion_type: str = "multilayer",
        fusion_layer_indices: Optional[List[int]] = None,
        lora_rank: int = 8,
        fusion_config: Optional[Dict] = None,

        # Training configuration
        freeze_base_vl: bool = True,
        freeze_pointcloud_encoder: bool = True,
        label_smoothing: float = 0.0,

        # Model dimensions
        llm_hidden_size: int = 5120,
        pointcloud_embed_dim: int = 768,
    ):
        super().__init__()

        # Initialize optional components to None (may be set later)
        self.kv_adapters = None
        self.kv_hook_manager = None
        self.fusion_adapter = None

        self.pointcloud_encoder_type = pointcloud_encoder_type
        self.projector_type = projector_type
        self.fusion_type = fusion_type
        self.num_tokens = num_tokens
        self.llm_hidden_size = llm_hidden_size
        self.pointcloud_embed_dim = pointcloud_embed_dim
        self.label_smoothing = label_smoothing
        self.debug_fusion_stats = False
        self.debug_fusion_log_every = 50

        # Initialize base VL model (same as SAFEModel)
        print(f"[SAFE-PC] Initializing BaseVLModel...", flush=True)
        sys.stdout.flush()
        self.base_vl = BaseVLModel(
            llm_model_name=llm_model_name,
            vision_model_name=vision_model_name,
            llm_hidden_size=llm_hidden_size,
            freeze_vision=freeze_base_vl,
            freeze_llm=freeze_base_vl,
        )
        print(f"[SAFE-PC] ✓ BaseVLModel initialized", flush=True)

        # Initialize point cloud encoder
        print(f"[SAFE-PC] Initializing point cloud encoder ({pointcloud_encoder_type})...", flush=True)
        pointcloud_encoder_config = pointcloud_encoder_config or {}

        if pointcloud_encoder_type in ["pointbert", "pointnet"]:
            self.pointcloud_encoder = PointBERTEncoder(
                freeze=freeze_pointcloud_encoder,
                **pointcloud_encoder_config
            )
            pointcloud_embed_dim = self.pointcloud_encoder.pointcloud_embed_dim
        else:
            raise ValueError(f"Unsupported point cloud encoder: {pointcloud_encoder_type}")

        self.pointcloud_embed_dim = pointcloud_embed_dim
        print(f"[SAFE-PC] ✓ Point cloud encoder initialized (embed_dim={pointcloud_embed_dim})", flush=True)

        # Check fusion mode
        fusion_config = fusion_config or {}
        is_kv_augment = fusion_config.get("fusion_mode") == "kv_augment"

        # Initialize projector (REUSE AudioProjector - it's modality-agnostic!)
        print(f"[SAFE-PC] Initializing projector ({projector_type})...", flush=True)
        projector_config = dict(projector_config) if projector_config else {}
        use_group_tokens = bool(pointcloud_encoder_config.get("return_group_tokens", False))

        # For KV augmentation, output to embed_dim space (smaller)
        projector_output_dim = projector_config.pop("output_dim", None)
        if projector_output_dim is None:
            # Default:
            # - KV augmentation: keep tokens small in encoder space
            # - Standard fusion: project into LLM hidden space for maximum capacity
            projector_output_dim = pointcloud_embed_dim if is_kv_augment else llm_hidden_size

        if projector_type == "standard":
            if use_group_tokens:
                # Token-set projector: (B, G, D) -> (B, num_tokens, output_dim)
                self.pointcloud_projector = TokenSetProjector(
                    input_dim=pointcloud_embed_dim,
                    num_tokens=num_tokens,
                    output_dim=projector_output_dim,
                    **projector_config,
                )
            else:
                # Note: We reuse AudioProjector - the name is misleading but it's generic
                self.pointcloud_projector = AudioProjector(
                    audio_embed_dim=pointcloud_embed_dim,  # Input from PC encoder
                    llm_hidden_size=llm_hidden_size,
                    num_audio_tokens=num_tokens,
                    output_dim=projector_output_dim,
                    **projector_config
                )
        elif projector_type == "adaptive":
            self.pointcloud_projector = AdaptiveAudioProjector(
                audio_embed_dim=pointcloud_embed_dim,
                llm_hidden_size=llm_hidden_size,
                max_audio_tokens=num_tokens,
                **projector_config
            )
        else:
            raise ValueError(f"Unsupported projector type: {projector_type}")

        actual_output_dim = getattr(self.pointcloud_projector, 'output_dim', llm_hidden_size)
        print(f"[SAFE-PC] ✓ Projector initialized (output_dim={actual_output_dim})", flush=True)

        # Safety: Fusion adapters expect modality tokens in LLM hidden space.
        # If projector outputs a different dim (e.g., 768), add a small adapter to map -> llm_hidden_size.
        self.pointcloud_token_adapter: Optional[nn.Module] = None
        if (not is_kv_augment) and int(actual_output_dim) != int(llm_hidden_size):
            print(
                f"[SAFE-PC] Warning: projector output_dim={actual_output_dim} != llm_hidden_size={llm_hidden_size}. "
                f"Adding pointcloud_token_adapter to map -> {llm_hidden_size}.",
                flush=True,
            )
            self.pointcloud_token_adapter = nn.Linear(int(actual_output_dim), int(llm_hidden_size), bias=False)

        # Initialize fusion adapter (REUSE from audio SAFE)
        self.fusion_mode = fusion_config.get("fusion_mode", "residual")
        self.enable_kv_augmentation = (self.fusion_mode == "kv_augment")
        self.fusion_layer_indices = fusion_layer_indices or [1]
        self.fusion_injection_point = fusion_config.get("injection_point", "pre_ffn")

        if self.enable_kv_augmentation:
            print(f"[SAFE-PC] Initializing KV Augmentation...", flush=True)
            self.fusion_adapter = None
            self.enable_midlayer_fusion = False
            self._setup_kv_augmentation(
                self.fusion_layer_indices,
                fusion_config,
            )
        else:
            print(f"[SAFE-PC] Initializing fusion adapter ({fusion_type})...", flush=True)
            if fusion_type == "multilayer":
                # Use "pointcloud" as modality key for this adapter
                modalities = {"pointcloud": {"layer_indices": self.fusion_layer_indices}}
                self.fusion_adapter = MultiLayerFusionAdapter(
                    hidden_size=llm_hidden_size,
                    modalities=modalities,
                    lora_rank=lora_rank,
                    num_attention_heads=fusion_config.get("num_attention_heads", 40),
                    lora_alpha=fusion_config.get("lora_alpha", 16.0),
                    lora_dropout=fusion_config.get("lora_dropout", 0.1),
                    attention_dropout=fusion_config.get("attention_dropout", 0.1),
                    use_bottleneck=fusion_config.get("use_bottleneck", False),
                    bottleneck_dim=fusion_config.get("bottleneck_dim", 32),
                    fusion_mode=fusion_config.get("fusion_mode", "residual"),
                )
                self.enable_midlayer_fusion = True
            elif fusion_type == "lora":
                self.fusion_adapter = LoRAFusionAdapter(
                    hidden_size=llm_hidden_size,
                    num_attention_heads=fusion_config.get("num_attention_heads", 40),
                    lora_rank=lora_rank,
                )
                self.enable_midlayer_fusion = False
            else:
                raise ValueError(f"Unsupported fusion type: {fusion_type}")

            print(f"[SAFE-PC] ✓ Fusion adapter initialized (midlayer={self.enable_midlayer_fusion})", flush=True)

        print(f"[SAFE-PC] ✓ Model initialization complete", flush=True)

    def set_fusion_debug(self, enabled: bool, log_every: int = 50) -> None:
        """Enable periodic fusion strength logging (delta vs hidden norms)."""
        self.debug_fusion_stats = bool(enabled)
        self.debug_fusion_log_every = int(log_every)

    def _setup_kv_augmentation(
        self,
        layer_indices: List[int],
        fusion_config: Dict,
    ) -> None:
        """Set up KV augmentation hooks (mirrors SAFEModel)."""
        # Get LLM attention config
        llm_config = self.base_vl.llm.config
        text_config = getattr(llm_config, 'text_config', llm_config)
        num_attention_heads = getattr(text_config, 'num_attention_heads', 40)
        num_key_value_heads = getattr(text_config, 'num_key_value_heads', num_attention_heads)
        head_dim = self.llm_hidden_size // num_attention_heads

        # Create KV adapters
        kv_adapters = nn.ModuleDict()
        for layer_idx in layer_indices:
            adapter = KVAugmentationAdapter(
                input_dim=self.pointcloud_embed_dim,  # Point cloud dim
                hidden_size=self.llm_hidden_size,
                num_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                bottleneck_dim=fusion_config.get("bottleneck_dim", 64),
                query_adapter_rank=fusion_config.get("query_adapter_rank", 16),
            )
            kv_adapters[str(layer_idx)] = adapter

        self.kv_adapters = kv_adapters

        # Create hook manager
        self.kv_hook_manager = KVAugmentationHookManager(
            model=self.base_vl.llm,
            kv_adapters=kv_adapters,
            fusion_layer_indices=layer_indices,
        )

        print(f"[SAFE-PC] ✓ KV Augmentation initialized at layers {layer_indices}", flush=True)

    def encode_pointcloud(
        self,
        pointcloud: Union[torch.Tensor, List[Any]],
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode point cloud to token representations.

        Args:
            pointcloud: Point cloud input (batch of tensors or list)
            num_tokens: Number of output tokens (optional override)

        Returns:
            Point cloud tokens (batch_size, num_tokens, hidden_dim)
        """
        # Get point cloud features from encoder
        # - (B, embed_dim) global vector (CLS)
        # - or (B, G, embed_dim) group tokens if return_group_tokens=True
        pc_features = self.pointcloud_encoder(pointcloud)

        # Debug: check for inf/nan in encoder features once
        if not hasattr(self, "_pc_feat_debug_logged"):
            self._pc_feat_debug_logged = True
            feat_finite = torch.isfinite(pc_features).all()
            feat_min = pc_features.min().item()
            feat_max = pc_features.max().item()
            feat_mean = pc_features.mean().item()
            print(
                f"[DEBUG] PC feats: shape={tuple(pc_features.shape)}, "
                f"finite={feat_finite}, min={feat_min:.4f}, max={feat_max:.4f}, mean={feat_mean:.4f}",
                flush=True,
            )

        # Get device and target dtype (LLM dtype, likely fp16)
        device = next(self.pointcloud_projector.parameters()).device
        target_dtype = next(self.base_vl.llm.parameters()).dtype

        # Keep in fp32 for projection to avoid overflow
        pc_features = pc_features.to(device=device, dtype=torch.float32)

        # Project to token space
        if self.projector_type == "adaptive":
            # Adaptive projector only supports vector inputs; fall back to mean-pool if tokens are provided.
            if pc_features.dim() == 3:
                pc_vec = pc_features.mean(dim=1)
            else:
                pc_vec = pc_features
            pc_tokens = self.pointcloud_projector(
                pc_vec,
                num_tokens=num_tokens or self.num_tokens,
            )
        else:
            # TokenSetProjector supports (B,G,D); AudioProjector expects (B,D)
            if pc_features.dim() == 3 and isinstance(self.pointcloud_projector, AudioProjector):
                pc_features = pc_features.mean(dim=1)
            pc_tokens = self.pointcloud_projector(pc_features)

        # Debug: check projector output once
        if not hasattr(self, "_pc_proj_debug_logged"):
            self._pc_proj_debug_logged = True
            proj_finite = torch.isfinite(pc_tokens).all()
            proj_min = pc_tokens.min().item()
            proj_max = pc_tokens.max().item()
            proj_mean = pc_tokens.mean().item()
            print(
                f"[DEBUG] PC proj: shape={tuple(pc_tokens.shape)}, "
                f"finite={proj_finite}, min={proj_min:.4f}, max={proj_max:.4f}, mean={proj_mean:.4f}",
                flush=True,
            )

        # Clamp before converting to fp16 to avoid overflow (fp16 max ~65504)
        if target_dtype == torch.float16:
            pc_tokens = torch.clamp(pc_tokens.float(), min=-65000, max=65000)

        # Ensure tokens match LLM hidden size when using fusion adapters
        if self.pointcloud_token_adapter is not None:
            pc_tokens = self.pointcloud_token_adapter(pc_tokens.float())

        return pc_tokens.to(dtype=target_dtype)

    def get_image_prompt_prefix_length(self) -> int:
        """Number of synthetic image placeholder tokens prepended for InternVL."""
        if self.base_vl.model_type != "internvl":
            return 0
        return int(getattr(self.base_vl.llm.config, "image_seq_length", 256))

    def _prepare_internvl_image_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Align InternVL image inputs by prepending <IMG_CONTEXT> placeholders."""
        if self.base_vl.model_type != "internvl" or pixel_values is None:
            return input_ids, attention_mask, labels, pixel_values

        model_dtype = next(self.base_vl.llm.parameters()).dtype
        if pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(dtype=model_dtype)

        img_token_id = getattr(self.base_vl.llm, "img_context_token_id", None)
        if img_token_id is None:
            img_token_id = getattr(self.base_vl.llm.config, "image_token_id", 151671)
        num_img_tokens = self.get_image_prompt_prefix_length()

        bsz = input_ids.size(0)
        device = input_ids.device
        img_ids = torch.full(
            (bsz, num_img_tokens),
            img_token_id,
            dtype=input_ids.dtype,
            device=device,
        )
        input_ids = torch.cat([img_ids, input_ids], dim=1)

        if attention_mask is not None:
            img_mask = torch.ones(
                (bsz, num_img_tokens),
                dtype=attention_mask.dtype,
                device=device,
            )
            attention_mask = torch.cat([img_mask, attention_mask], dim=1)

        if labels is not None:
            img_labels = torch.full(
                (bsz, num_img_tokens),
                -100,
                dtype=labels.dtype,
                device=device,
            )
            labels = torch.cat([img_labels, labels], dim=1)

        if not hasattr(self, "_internvl_img_prefix_logged"):
            self._internvl_img_prefix_logged = True
            print(
                f"[SAFE-PC] Prepending {num_img_tokens} <IMG_CONTEXT> tokens "
                f"(id={img_token_id}) to input_ids for InternVL vision",
                flush=True,
            )

        return input_ids, attention_mask, labels, pixel_values

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pointcloud: Optional[Union[torch.Tensor, List]] = None,
        pointcloud_tokens: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        gate: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass with point cloud fusion.

        Supports three modes:
        - PC only: input_ids + pointcloud (text-only input, PC fusion)
        - Image only: input_ids + pixel_values (LLaVA native, no PC fusion)
        - Both (composition): input_ids + pixel_values + pointcloud
          LLaVA processes image+text, SAFE adds PC residuals via fusion

        Args:
            input_ids: Text input token IDs
            attention_mask: Attention mask
            labels: Target labels for loss computation
            pointcloud: Raw point cloud input
            pointcloud_tokens: Pre-computed point cloud tokens
            pixel_values: Image pixel values for LLaVA (enables composition)

        Returns:
            Dictionary with logits and optional loss
        """
        # Get point cloud tokens
        if pointcloud_tokens is None and pointcloud is not None:
            pointcloud_tokens = self.encode_pointcloud(pointcloud)

        # Debug: check for inf/nan in pointcloud tokens
        if pointcloud_tokens is not None and not hasattr(self, '_pc_debug_logged'):
            self._pc_debug_logged = True
            pc_finite = torch.isfinite(pointcloud_tokens).all()
            pc_min = pointcloud_tokens.min().item()
            pc_max = pointcloud_tokens.max().item()
            pc_mean = pointcloud_tokens.mean().item()
            print(f"[DEBUG] PC tokens: shape={tuple(pointcloud_tokens.shape)}, "
                  f"finite={pc_finite}, min={pc_min:.4f}, max={pc_max:.4f}, mean={pc_mean:.4f}", flush=True)

        # Determine if we're doing composition (image + PC)
        use_composition = pixel_values is not None

        # Determine fusion mode
        use_midlayer_hooks = (
            pointcloud_tokens is not None
            and self.enable_midlayer_fusion
            and self.fusion_adapter is not None
            and hasattr(self.fusion_adapter, "apply_fusion_at_layer")
        )

        if use_midlayer_hooks:
            # Set up fusion layers mapping
            fusion_layers = {"pointcloud": self.fusion_layer_indices}
            modality_tokens = {"pointcloud": pointcloud_tokens}

            # Create hook manager and register hooks
            hook_manager = LayerHookManager(
                model=self.base_vl.llm,
                fusion_adapter=self.fusion_adapter,
                fusion_layers=fusion_layers,
                injection_point=self.fusion_injection_point,
            )

            # Align PC tokens dtype with model
            target_dtype = next(self.base_vl.llm.parameters()).dtype
            pointcloud_tokens = pointcloud_tokens.to(
                device=input_ids.device,
                dtype=target_dtype,
            )
            modality_tokens = {"pointcloud": pointcloud_tokens}

            hook_manager.register_hooks(
                modality_tokens=modality_tokens,
                modality_masks=None,
                gate={"pointcloud": gate},
                debug_fusion=self.debug_fusion_stats,
                debug_fusion_log_every=self.debug_fusion_log_every,
            )

            # Debug: log hook info once
            if not hasattr(self, '_hook_debug_logged'):
                self._hook_debug_logged = True
                mode = "composition (image+PC)" if use_composition else "PC-only"
                print(f"[DEBUG] Hooks registered ({mode}): num_hooks={hook_manager.num_hooks}, "
                      f"layers={fusion_layers}, injection={self.fusion_injection_point}", flush=True)

            try:
                if use_composition:
                    # COMPOSITION MODE: Let VLM process image+text natively,
                    # while SAFE hooks inject PC tokens as residuals
                    comp_input_ids, comp_attn_mask, comp_labels, comp_pixel_values = (
                        self._prepare_internvl_image_inputs(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            pixel_values=pixel_values,
                        )
                    )

                    fwd_kwargs = dict(
                        input_ids=comp_input_ids,
                        attention_mask=comp_attn_mask,
                        pixel_values=comp_pixel_values,
                        labels=comp_labels,
                        use_cache=False, **kwargs,
                    )
                    if self.base_vl.model_type == "internvl" and pixel_values is not None:
                        batch = pixel_values.size(0)
                        fwd_kwargs["image_flags"] = torch.ones(
                            (batch, 1), dtype=torch.long, device=pixel_values.device
                        )
                    outputs = self.base_vl.llm(**fwd_kwargs)
                else:
                    # PC-ONLY MODE: text only (no image), hooks inject PC residuals.
                    # InternVL's forward() requires pixel_values as a positional arg,
                    # so bypass it and call the inner language_model directly.
                    inputs_embeds = self.base_vl.llm.get_input_embeddings()(input_ids)
                    lm = getattr(self.base_vl.llm, "language_model", self.base_vl.llm)
                    outputs = lm(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                        **kwargs,
                    )
            finally:
                hook_manager.remove_hooks()

        elif pointcloud_tokens is not None and self.enable_kv_augmentation:
            # KV augmentation: replace attention modules and inject tokens as K,V
            if self.kv_hook_manager is None:
                raise RuntimeError("KV augmentation enabled but kv_hook_manager is None")

            # Wrap attention modules once (no-op if already wrapped)
            self.kv_hook_manager.wrap_attention_modules()

            # NOTE: Do NOT clear tokens after forward; gradient checkpointing replays forward in backward.
            # Tokens persist until the next inject call.
            self.kv_hook_manager.inject_audio(
                audio_tokens=pointcloud_tokens,
                audio_mask=None,
                gate=1.0,
            )

            # KV augmentation wrapper does not support caching in forward.
            if use_composition:
                fwd_input_ids, fwd_attn_mask, fwd_labels, fwd_pixel_values = self._prepare_internvl_image_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=pixel_values,
                )
                fwd_kwargs = dict(
                    input_ids=fwd_input_ids, attention_mask=fwd_attn_mask,
                    pixel_values=fwd_pixel_values, labels=fwd_labels,
                    use_cache=False, **kwargs,
                )
                if self.base_vl.model_type == "internvl" and pixel_values is not None:
                    batch = pixel_values.size(0)
                    fwd_kwargs["image_flags"] = torch.ones(
                        (batch, 1), dtype=torch.long, device=pixel_values.device
                    )
                outputs = self.base_vl.llm(**fwd_kwargs)
            else:
                # PC-only with KV augmentation, no image.
                # Bypass InternVL wrapper (requires pixel_values) and call
                # the inner language_model directly.
                inputs_embeds = self.base_vl.llm.get_input_embeddings()(input_ids)
                lm = getattr(self.base_vl.llm, "language_model", self.base_vl.llm)
                outputs = lm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                    **kwargs,
                )
        else:
            # No PC fusion - just process image+text or text-only
            if use_composition:
                # Image + text through LLaVA (no PC)
                fwd_input_ids, fwd_attn_mask, fwd_labels, fwd_pixel_values = self._prepare_internvl_image_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=pixel_values,
                )
                fwd_kwargs = dict(
                    input_ids=fwd_input_ids, attention_mask=fwd_attn_mask,
                    pixel_values=fwd_pixel_values, labels=fwd_labels,
                    use_cache=False, **kwargs,
                )
                if self.base_vl.model_type == "internvl" and pixel_values is not None:
                    batch = pixel_values.size(0)
                    fwd_kwargs["image_flags"] = torch.ones(
                        (batch, 1), dtype=torch.long, device=pixel_values.device
                    )
                outputs = self.base_vl.llm(**fwd_kwargs)
            else:
                # Text-only (no PC, no image).
                # Bypass InternVL wrapper and call inner language_model.
                inputs_embeds = self.base_vl.llm.get_input_embeddings()(input_ids)
                lm = getattr(self.base_vl.llm, "language_model", self.base_vl.llm)
                outputs = lm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                    **kwargs,
                )

        # Debug: check outputs
        if not hasattr(self, '_output_debug_logged'):
            self._output_debug_logged = True
            logits_finite = torch.isfinite(outputs.logits).all()
            logits_min = outputs.logits.min().item()
            logits_max = outputs.logits.max().item()
            print(f"[DEBUG] Logits: shape={tuple(outputs.logits.shape)}, "
                  f"finite={logits_finite}, min={logits_min:.4f}, max={logits_max:.4f}", flush=True)
            if outputs.loss is not None:
                print(f"[DEBUG] HF Loss: {outputs.loss.item():.4f}", flush=True)

        result = {"logits": outputs.logits}

        if labels is not None:
            # Use HF's built-in loss computation (handles ignore_index properly)
            loss = outputs.loss
            result["loss"] = loss

        # Expose last hidden state for downstream probes/classifiers.
        hidden_state_out = None
        try:
            hs = getattr(outputs, "hidden_states", None)
            if isinstance(hs, (list, tuple)) and len(hs) > 0 and torch.is_tensor(hs[-1]):
                hidden_state_out = hs[-1]
            else:
                lhs = getattr(outputs, "last_hidden_state", None)
                if torch.is_tensor(lhs):
                    hidden_state_out = lhs
        except Exception:
            hidden_state_out = None

        if torch.is_tensor(hidden_state_out):
            result["hidden_states"] = hidden_state_out

        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pointcloud: Optional[Union[torch.Tensor, List]] = None,
        pointcloud_tokens: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        num_beams: int = 5,
        gate: float = 1.0,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Generate text conditioned on point cloud (and optionally images).

        Supports:
        - PC only: input_ids + pointcloud
        - Composition: input_ids + pixel_values + pointcloud
          (LLaVA processes image+text, SAFE injects PC residuals)

        Args:
            input_ids: Input token IDs (prompt)
            attention_mask: Attention mask
            pointcloud: Raw point cloud input
            pointcloud_tokens: Pre-computed tokens
            pixel_values: Image pixel values for composition mode
            max_new_tokens: Maximum new tokens to generate
            num_beams: Beam search width

        Returns:
            Generated token IDs
        """
        # Get point cloud tokens
        if pointcloud_tokens is None and pointcloud is not None:
            pointcloud_tokens = self.encode_pointcloud(pointcloud)

        # Determine if we're doing composition (image + PC)
        use_composition = pixel_values is not None

        # Determine fusion mode
        use_midlayer_hooks = (
            pointcloud_tokens is not None
            and self.enable_midlayer_fusion
            and self.fusion_adapter is not None
            and hasattr(self.fusion_adapter, "apply_fusion_at_layer")
        )

        if use_midlayer_hooks:
            # Align dtypes
            target_dtype = next(self.base_vl.llm.parameters()).dtype
            pointcloud_tokens = pointcloud_tokens.to(
                device=input_ids.device,
                dtype=target_dtype,
            )

            # Set up fusion layers mapping
            fusion_layers = {"pointcloud": self.fusion_layer_indices}
            modality_tokens = {"pointcloud": pointcloud_tokens}

            # Create hook manager and register hooks
            hook_manager = LayerHookManager(
                model=self.base_vl.llm,
                fusion_adapter=self.fusion_adapter,
                fusion_layers=fusion_layers,
                injection_point=self.fusion_injection_point,
            )
            hook_manager.register_hooks(
                modality_tokens=modality_tokens,
                modality_masks=None,
                gate={"pointcloud": gate},
                debug_fusion=self.debug_fusion_stats,
                debug_fusion_log_every=self.debug_fusion_log_every,
            )

            try:
                if use_composition:
                    # COMPOSITION: VLM processes image+text, SAFE adds PC residuals
                    gen_input_ids, gen_attn_mask, _, gen_pixel_values = self._prepare_internvl_image_inputs(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                    )

                    gen_kwargs = dict(
                        input_ids=gen_input_ids, attention_mask=gen_attn_mask,
                        pixel_values=gen_pixel_values, max_new_tokens=max_new_tokens,
                        num_beams=num_beams, **generate_kwargs,
                    )
                    # Note: do NOT pass image_flags to generate() — InternVL's
                    # generate() consumes it internally but then leaks remaining
                    # kwargs to language_model.generate() which rejects unknowns.
                    outputs = self.base_vl.llm.generate(**gen_kwargs)
                else:
                    # PC-only: generate from token IDs; custom InternVL generate()
                    # expects input_ids and may ignore inputs_embeds.
                    outputs = self.base_vl.llm.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        **generate_kwargs,
                    )
            finally:
                hook_manager.remove_hooks()

        elif pointcloud_tokens is not None and self.enable_kv_augmentation:
            # KV augmentation: wrap attention modules and inject tokens as K,V
            if self.kv_hook_manager is None:
                raise RuntimeError("KV augmentation enabled but kv_hook_manager is None")

            self.kv_hook_manager.wrap_attention_modules()
            self.kv_hook_manager.inject_audio(
                audio_tokens=pointcloud_tokens,
                audio_mask=None,
                gate=1.0,
            )

            # Disable caching for KV augmentation unless explicitly set.
            if "use_cache" not in generate_kwargs:
                generate_kwargs = dict(generate_kwargs)
                generate_kwargs["use_cache"] = False

            if use_composition:
                gen_input_ids, gen_attn_mask, _, gen_pixel_values = self._prepare_internvl_image_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )
                gen_kwargs = dict(
                    input_ids=gen_input_ids, attention_mask=gen_attn_mask,
                    pixel_values=gen_pixel_values, max_new_tokens=max_new_tokens,
                    num_beams=num_beams, **generate_kwargs,
                )
                # Note: do NOT pass image_flags — InternVL's generate() builds
                # it internally but leaks remaining kwargs to language_model.generate().
                outputs = self.base_vl.llm.generate(**gen_kwargs)
            else:
                # Custom InternVL generate() expects input_ids rather than inputs_embeds.
                outputs = self.base_vl.llm.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    **generate_kwargs,
                )
        else:
            # No PC fusion
            if use_composition:
                # Image + text through LLaVA (no PC)
                gen_input_ids, gen_attn_mask, _, gen_pixel_values = self._prepare_internvl_image_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )
                gen_kwargs = dict(
                    input_ids=gen_input_ids, attention_mask=gen_attn_mask,
                    pixel_values=gen_pixel_values, max_new_tokens=max_new_tokens,
                    num_beams=num_beams, **generate_kwargs,
                )
                # Note: do NOT pass image_flags — InternVL's generate() builds
                # it internally but leaks remaining kwargs to language_model.generate().
                outputs = self.base_vl.llm.generate(**gen_kwargs)
            else:
                # Text only
                # Custom InternVL generate() expects input_ids rather than inputs_embeds.
                outputs = self.base_vl.llm.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    **generate_kwargs,
                )

        return outputs

    def enable_pointcloud_training(self) -> None:
        """Enable training for point cloud components only."""
        # Freeze base VL
        for param in self.base_vl.parameters():
            param.requires_grad = False

        # Freeze point cloud encoder unless it has explicitly-unfrozen params
        if not any(p.requires_grad for p in self.pointcloud_encoder.parameters()):
            for param in self.pointcloud_encoder.parameters():
                param.requires_grad = False

        # Enable projector training
        for param in self.pointcloud_projector.parameters():
            param.requires_grad = True

        # Enable fusion adapter training
        if self.fusion_adapter is not None:
            for param in self.fusion_adapter.parameters():
                param.requires_grad = True

        # Enable KV adapters training (only set in kv_augment mode)
        if self.kv_adapters is not None:
            for param in self.kv_adapters.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[SAFE-PC] Training enabled: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)", flush=True)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
