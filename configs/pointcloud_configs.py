# -*- coding: utf-8 -*-
"""
Point cloud model configurations for SAFE architecture.

These configs mirror the audio configs but for 3D point cloud modality.
Proves the architecture generalizes to other modalities.
"""

import os
from typing import Dict, Any

# ModelNet40 Classification Config
# Use this first to validate the architecture works for point clouds
MODELNET40_CONFIG: Dict[str, Any] = {
    "name": "modelnet40_cls",
    "description": "Point cloud classification on ModelNet40 to validate architecture",

    # Base VL Model (same as audio SAFE)
    "llm_model_name": "llava-hf/llava-1.5-13b-hf",
    "vision_model_name": "openai/clip-vit-large-patch14",

    # Point cloud encoder configuration
    "pointcloud_encoder_type": "pointbert",
    "pointcloud_encoder_config": {
        "model_name": "pointbert-base",
        "num_points": 1024,
        "embed_dim": 768,
        "use_pretrained": True,
        "checkpoint_path": None,  # Set to path if you have pre-trained weights
        # Use PointTransformer group tokens (not just CLS) for richer fusion.
        "return_group_tokens": True,
    },

    # Model dimensions
    "llm_hidden_size": 5120,  # LLaVA 13B hidden size
    "pointcloud_embed_dim": 768,  # PointBERT output dimension

    # Projector configuration (reuses AudioProjector)
    "projector_type": "standard",
    "num_tokens": 32,  # More tokens helps for 40-way classification
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 2048,
        # Explicitly project into LLM hidden space (prevents dim mismatch).
        "output_dim": 5120,
        # TokenSetProjector ignores unknown keys like use_swiglu.
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    # Fusion configuration (reuses MultiLayerFusionAdapter)
    # Uses pre-FFN residual fusion (same as audio LLM probe)
    "fusion_type": "multilayer",
    "fusion_layer_indices": [12, 24, 36],  # Stronger signal via multi-layer fusion
    "lora_rank": 8,
    "fusion_config": {
        "num_attention_heads": 40,  # LLaVA 13B heads
        "attention_dropout": 0.1,
        "fusion_mode": "residual",  # Switch to "kv_augment" to test KV augmentation
        "use_bottleneck": True,
        "bottleneck_dim": 256,
    },

    # Training configuration
    "freeze_base_vl": True,
    "freeze_pointcloud_encoder": True,
    "label_smoothing": 0.1,

    # Dataset configuration
    "dataset": "modelnet40",
    "num_points": 1024,
    "num_classes": 40,

    # Memory and compute
    "expected_vram_gb": 35,
    "recommended_batch_size": 8,  # Larger than audio (no audio loading overhead)
    "gradient_accumulation_steps": 8,
}


# Cap3D Captioning Config
# Use this after classification works to test generation
CAP3D_CONFIG: Dict[str, Any] = {
    "name": "cap3d_captioning",
    "description": "Point cloud captioning on Cap3D/Objaverse dataset",

    # Base VL Model
    "llm_model_name": "llava-hf/llava-1.5-13b-hf",
    "vision_model_name": "openai/clip-vit-large-patch14",

    # Point cloud encoder configuration
    "pointcloud_encoder_type": "pointbert",
    "pointcloud_encoder_config": {
        "model_name": "pointbert-base",
        "num_points": 2048,  # More points for detailed captioning
        "embed_dim": 768,
        "use_pretrained": True,
        "checkpoint_path": None,
        "return_group_tokens": True,
    },

    # Model dimensions
    "llm_hidden_size": 5120,
    "pointcloud_embed_dim": 768,

    # Projector configuration - more tokens for captioning
    "projector_type": "standard",
    "num_tokens": 16,  # More tokens for detailed descriptions
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    # Fusion configuration (pre-FFN residual for captioning)
    "fusion_type": "multilayer",
    "fusion_layer_indices": [12, 24, 36],  # Multi-layer for generation
    "lora_rank": 8,
    "fusion_config": {
        "num_attention_heads": 40,
        "attention_dropout": 0.1,
        "fusion_mode": "residual",  # Pre-FFN residual addition
        "use_bottleneck": True,
        "bottleneck_dim": 64,
    },

    # Training configuration
    "freeze_base_vl": True,
    "freeze_pointcloud_encoder": True,
    "label_smoothing": 0.1,

    # Dataset configuration
    "dataset": "cap3d",
    "num_points": 2048,

    # Memory and compute
    "expected_vram_gb": 40,
    "recommended_batch_size": 2,
    "gradient_accumulation_steps": 32,
}


# Demo/Debug Config - lightweight for testing
POINTCLOUD_DEMO_CONFIG: Dict[str, Any] = {
    "name": "pointcloud_demo",
    "description": "Lightweight point cloud config for testing",

    # Smaller base model
    "llm_model_name": "Salesforce/blip2-opt-2.7b",
    "vision_model_name": "openai/clip-vit-base-patch32",

    # Point cloud encoder
    "pointcloud_encoder_type": "pointnet",  # Simpler encoder
    "pointcloud_encoder_config": {
        "model_name": "pointnet-simple",
        "num_points": 512,
        "embed_dim": 256,
        "use_pretrained": False,
    },

    # Model dimensions
    "llm_hidden_size": 2560,  # OPT 2.7B
    "pointcloud_embed_dim": 256,

    # Projector configuration
    "projector_type": "standard",
    "num_tokens": 4,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 512,
    },

    # Fusion configuration (pre-FFN residual)
    "fusion_type": "multilayer",
    "fusion_layer_indices": [1],  # Single layer for demo
    "lora_rank": 4,
    "fusion_config": {
        "num_attention_heads": 20,  # OPT 2.7B
        "attention_dropout": 0.1,
        "fusion_mode": "residual",  # Pre-FFN residual addition
        "use_bottleneck": True,
        "bottleneck_dim": 32,
    },

    # Training
    "freeze_base_vl": True,
    "freeze_pointcloud_encoder": True,

    # Memory
    "expected_vram_gb": 12,
    "recommended_batch_size": 16,
    "gradient_accumulation_steps": 4,
}


# ScanQA Joint Config — InternVL 8B with PointBERT + multi-layer fusion
# Joint training: point cloud sees vision + text (modality=both)
SCANQA_INTERNVL_CONFIG: Dict[str, Any] = {
    "name": "scanqa_internvl",
    "description": "ScanQA joint: PointBERT on InternVL 8B with native vision active",

    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/OpenGVLab_InternVL3_5-8B"),
    "vision_model_name": "built-in",

    "pointcloud_encoder_type": "pointbert",
    "pointcloud_encoder_config": {
        "model_name": "pointbert-base",
        "num_points": 8192,
        "embed_dim": 768,
        "use_pretrained": True,
        "checkpoint_path": None,
        "return_group_tokens": False,
    },

    "llm_hidden_size": 4096,        # Qwen3-8B
    "pointcloud_embed_dim": 768,    # PointBERT output

    "projector_type": "standard",
    "num_tokens": 8,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    "fusion_type": "multilayer",
    "fusion_layer_indices": [1, 5, 9, 13, 17, 21],
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "num_attention_heads": 32,
        "dropout": 0.1,
    },

    "freeze_base_vl": True,
    "freeze_pointcloud_encoder": True,
    "label_smoothing": 0.1,
    "dataset": "scanqa",
    "num_points": 8192,
    "expected_vram_gb": 42,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 8,
}


# Config registry
POINTCLOUD_CONFIGS: Dict[str, Dict[str, Any]] = {
    "modelnet40": MODELNET40_CONFIG,
    "modelnet40_cls": MODELNET40_CONFIG,
    "cap3d": CAP3D_CONFIG,
    "cap3d_captioning": CAP3D_CONFIG,
    "demo": POINTCLOUD_DEMO_CONFIG,
    "pointcloud_demo": POINTCLOUD_DEMO_CONFIG,
    "scanqa_internvl": SCANQA_INTERNVL_CONFIG,
}


def get_pointcloud_config(name: str) -> Dict[str, Any]:
    """
    Get a point cloud configuration by name.

    Args:
        name: Config name ("modelnet40", "cap3d", "demo", etc.)

    Returns:
        Configuration dictionary (copy)
    """
    if name not in POINTCLOUD_CONFIGS:
        available = list(POINTCLOUD_CONFIGS.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")

    return POINTCLOUD_CONFIGS[name].copy()


def list_pointcloud_configs() -> list:
    """List all available point cloud configurations."""
    return list(POINTCLOUD_CONFIGS.keys())
