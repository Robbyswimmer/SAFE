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
from .projectors import AudioProjector, AdaptiveAudioProjector
from .fusion_adapter import (
    MultiLayerFusionAdapter,
    LoRAFusionAdapter,
    GatedFusionAdapter,
)
from .kv_augmentation import KVAugmentationHookManager, KVAugmentationAdapter


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

        self.pointcloud_encoder_type = pointcloud_encoder_type
        self.projector_type = projector_type
        self.fusion_type = fusion_type
        self.num_tokens = num_tokens
        self.llm_hidden_size = llm_hidden_size
        self.pointcloud_embed_dim = pointcloud_embed_dim
        self.label_smoothing = label_smoothing

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

        # For KV augmentation, output to embed_dim space (smaller)
        projector_output_dim = projector_config.pop("output_dim", None)
        if projector_output_dim is None:
            projector_output_dim = pointcloud_embed_dim if is_kv_augment else None

        if projector_type == "standard":
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

        # Initialize fusion adapter (REUSE from audio SAFE)
        self.fusion_mode = fusion_config.get("fusion_mode", "residual")
        self.enable_kv_augmentation = (self.fusion_mode == "kv_augment")

        if self.enable_kv_augmentation:
            print(f"[SAFE-PC] Initializing KV Augmentation...", flush=True)
            self.fusion_adapter = None
            self._setup_kv_augmentation(
                fusion_layer_indices or [12, 24, 36],
                fusion_config,
            )
        else:
            print(f"[SAFE-PC] Initializing fusion adapter ({fusion_type})...", flush=True)
            if fusion_type == "multilayer":
                self.fusion_adapter = MultiLayerFusionAdapter(
                    hidden_size=llm_hidden_size,
                    fusion_layer_indices=fusion_layer_indices,
                    lora_rank=lora_rank,
                    num_attention_heads=fusion_config.get("num_attention_heads", 40),
                    lora_alpha=fusion_config.get("lora_alpha", 16.0),
                    lora_dropout=fusion_config.get("lora_dropout", 0.1),
                    attention_dropout=fusion_config.get("attention_dropout", 0.1),
                    use_bottleneck=fusion_config.get("use_bottleneck", False),
                    bottleneck_dim=fusion_config.get("bottleneck_dim", 32),
                    fusion_mode=fusion_config.get("fusion_mode", "residual"),
                )
            elif fusion_type == "lora":
                self.fusion_adapter = LoRAFusionAdapter(
                    hidden_size=llm_hidden_size,
                    num_attention_heads=fusion_config.get("num_attention_heads", 40),
                    lora_rank=lora_rank,
                )
            else:
                raise ValueError(f"Unsupported fusion type: {fusion_type}")

            print(f"[SAFE-PC] ✓ Fusion adapter initialized", flush=True)

        self.kv_hook_manager = None
        print(f"[SAFE-PC] ✓ Model initialization complete", flush=True)

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
        pc_features = self.pointcloud_encoder(pointcloud)  # (B, embed_dim)

        # Get device/dtype from projector
        device = next(self.pointcloud_projector.parameters()).device
        dtype = next(self.pointcloud_projector.parameters()).dtype

        pc_features = pc_features.to(device=device, dtype=dtype)

        # Project to token space
        if self.projector_type == "adaptive":
            pc_tokens = self.pointcloud_projector(
                pc_features,
                num_tokens=num_tokens or self.num_tokens,
            )
        else:
            pc_tokens = self.pointcloud_projector(pc_features)

        return pc_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pointcloud: Optional[Union[torch.Tensor, List]] = None,
        pointcloud_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass with point cloud fusion.

        Args:
            input_ids: Text input token IDs
            attention_mask: Attention mask
            labels: Target labels for loss computation
            pointcloud: Raw point cloud input
            pointcloud_tokens: Pre-computed point cloud tokens

        Returns:
            Dictionary with logits and optional loss
        """
        # Get point cloud tokens
        if pointcloud_tokens is None and pointcloud is not None:
            pointcloud_tokens = self.encode_pointcloud(pointcloud)

        # Get text embeddings
        inputs_embeds = self.base_vl.llm.get_input_embeddings()(input_ids)

        # Apply fusion
        if pointcloud_tokens is not None and self.enable_kv_augmentation:
            # KV augmentation: inject tokens via hook manager
            if self.kv_hook_manager is not None:
                self.kv_hook_manager.set_audio_tokens(pointcloud_tokens)
        elif pointcloud_tokens is not None and self.fusion_adapter is not None:
            # Traditional fusion: cross-attention
            inputs_embeds = self.fusion_adapter(
                hidden_states=inputs_embeds,
                audio_tokens=pointcloud_tokens,
            )

        # Forward through LLM
        outputs = self.base_vl.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

        # Clear KV hooks
        if self.kv_hook_manager is not None:
            self.kv_hook_manager.clear_audio_tokens()

        result = {"logits": outputs.logits}

        if labels is not None:
            if self.label_smoothing > 0:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                loss = loss_fct(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    labels.view(-1),
                )
            else:
                loss = outputs.loss

            result["loss"] = loss

        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pointcloud: Optional[Union[torch.Tensor, List]] = None,
        pointcloud_tokens: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        num_beams: int = 5,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Generate text conditioned on point cloud.

        Args:
            input_ids: Input token IDs (prompt)
            attention_mask: Attention mask
            pointcloud: Raw point cloud input
            pointcloud_tokens: Pre-computed tokens
            max_new_tokens: Maximum new tokens to generate
            num_beams: Beam search width

        Returns:
            Generated token IDs
        """
        # Get point cloud tokens
        if pointcloud_tokens is None and pointcloud is not None:
            pointcloud_tokens = self.encode_pointcloud(pointcloud)

        # Set up KV augmentation
        if pointcloud_tokens is not None and self.kv_hook_manager is not None:
            self.kv_hook_manager.set_audio_tokens(pointcloud_tokens)

        # Get embeddings
        inputs_embeds = self.base_vl.llm.get_input_embeddings()(input_ids)

        # Apply traditional fusion if not using KV augment
        if pointcloud_tokens is not None and self.fusion_adapter is not None:
            inputs_embeds = self.fusion_adapter(
                hidden_states=inputs_embeds,
                audio_tokens=pointcloud_tokens,
            )

        # Generate
        outputs = self.base_vl.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            **generate_kwargs,
        )

        # Clear hooks
        if self.kv_hook_manager is not None:
            self.kv_hook_manager.clear_audio_tokens()

        return outputs

    def enable_pointcloud_training(self) -> None:
        """Enable training for point cloud components only."""
        # Freeze base VL
        for param in self.base_vl.parameters():
            param.requires_grad = False

        # Freeze point cloud encoder
        for param in self.pointcloud_encoder.parameters():
            param.requires_grad = False

        # Enable projector training
        for param in self.pointcloud_projector.parameters():
            param.requires_grad = True

        # Enable fusion adapter training
        if self.fusion_adapter is not None:
            for param in self.fusion_adapter.parameters():
                param.requires_grad = True

        # Enable KV adapters training
        if self.kv_adapters is not None:
            for param in self.kv_adapters.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[SAFE-PC] Training enabled: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)", flush=True)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
