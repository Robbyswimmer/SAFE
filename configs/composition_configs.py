# -*- coding: utf-8 -*-
"""
Composition ablation configs for multi-modal (audio + point cloud) experiments.
Tests whether pre-FFN vs KV augmentation handles modality composition differently.
"""

import os

# Base LLM config (shared)
_BASE_LLM_CONFIG = {
    "llm_model_name": "llava-hf/llava-1.5-13b-hf",
    "vision_model_name": "openai/clip-vit-large-patch14",
    "llm_hidden_size": 5120,
    "freeze_base_vl": True,
}

# Audio encoder config (CLAP)
_AUDIO_ENCODER_CONFIG = {
    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0
    },
    "audio_embed_dim": 512,
    "num_audio_tokens": 8,
}

# Point cloud encoder config (PointBERT)
# Note: num_pointcloud_tokens=32 to match ModelNet40 training config
_POINTCLOUD_ENCODER_CONFIG = {
    "pointcloud_encoder_type": "pointbert",
    "pointcloud_encoder_config": {
        "model_name": "pointbert-pretrained",
        "num_points": 1024,
    },
    "pointcloud_embed_dim": 768,
    "num_pointcloud_tokens": 32,
}

# Projector config (shared for both modalities)
_PROJECTOR_CONFIG = {
    "projector_type": "standard",
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },
}

# Layer indices for fusion (LLaVA 1.5 13B has 40 layers)
# Using same layers as layer ablation: start at 1, stride 4
_FUSION_LAYERS = [1, 5, 9, 13, 17, 21]


# =============================================================================
# Pre-FFN Composition Config
# =============================================================================
COMPOSITION_PREFFN_CONFIG = {
    "name": "composition_preffn",
    "description": "Multi-modal composition with pre-FFN residual fusion (audio + point cloud)",

    # Base LLM
    **_BASE_LLM_CONFIG,

    # Both encoders
    **_AUDIO_ENCODER_CONFIG,
    **_POINTCLOUD_ENCODER_CONFIG,
    **_PROJECTOR_CONFIG,

    # Fusion configuration - Pre-FFN residual
    "fusion_type": "multilayer",
    "fusion_layer_indices": _FUSION_LAYERS,
    "lora_rank": 16,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "num_attention_heads": 40,
        "attention_dropout": 0.1,

        # Multi-modal: both audio and point cloud at same layers
        "modalities": {
            "audio": {
                "layer_indices": _FUSION_LAYERS,
                "num_tokens": 8,
            },
            "pointcloud": {
                "layer_indices": _FUSION_LAYERS,
                "num_tokens": 32,  # Match ModelNet40 training config
            },
        },

        # Bottleneck cross-attention settings
        "use_bottleneck": True,
        "bottleneck_dim": 256,
        "use_ffn": True,
        "ffn_expansion": 2.0,
        "use_pre_norm": True,
        "use_tokenwise_gate": True,
    },

    # Training
    "freeze_audio_encoder": True,
    "freeze_pointcloud_encoder": True,
    "label_smoothing": 0.1,

    # Memory
    "expected_vram_gb": 45,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16,
}


# =============================================================================
# KV Augmentation Composition Config
# =============================================================================
COMPOSITION_KVAUG_CONFIG = {
    "name": "composition_kvaug",
    "description": "Multi-modal composition with KV augmentation fusion (audio + point cloud)",

    # Base LLM
    **_BASE_LLM_CONFIG,

    # Both encoders
    **_AUDIO_ENCODER_CONFIG,
    **_POINTCLOUD_ENCODER_CONFIG,
    **_PROJECTOR_CONFIG,

    # Fusion configuration - KV Augmentation
    "fusion_type": "multilayer",
    "fusion_layer_indices": _FUSION_LAYERS,
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "kv_augment",

        # LLaVA 13B attention configuration
        "num_attention_heads": 40,
        "head_dim": 128,

        # Bottleneck for K,V projections
        "bottleneck_dim": 64,
        "use_bottleneck": True,
        "dropout": 0.1,

        # Query adapter for attending to modality K,V
        "query_adapter_rank": 16,

        # Multi-modal: both audio and point cloud
        "modalities": {
            "audio": {
                "layer_indices": _FUSION_LAYERS,
                "num_tokens": 8,
            },
            "pointcloud": {
                "layer_indices": _FUSION_LAYERS,
                "num_tokens": 32,  # Match ModelNet40 training config
            },
        },
    },

    # Training
    "freeze_audio_encoder": True,
    "freeze_pointcloud_encoder": True,
    "label_smoothing": 0.1,

    # Memory
    "expected_vram_gb": 40,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 8,
}


# =============================================================================
# Single-modality configs for training adapters separately
# =============================================================================

# Audio-only Pre-FFN (for training audio adapter)
AUDIO_ONLY_PREFFN_CONFIG = {
    "name": "audio_only_preffn",
    "description": "Audio-only pre-FFN config for training audio adapter",

    **_BASE_LLM_CONFIG,
    **_AUDIO_ENCODER_CONFIG,
    **_PROJECTOR_CONFIG,

    "fusion_type": "multilayer",
    "fusion_layer_indices": _FUSION_LAYERS,
    "lora_rank": 16,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "num_attention_heads": 40,
        "attention_dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": _FUSION_LAYERS,
                "num_tokens": 8,
            },
        },
        "use_bottleneck": True,
        "bottleneck_dim": 256,
        "use_ffn": True,
        "ffn_expansion": 2.0,
        "use_pre_norm": True,
        "use_tokenwise_gate": True,
    },

    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,
}

# Audio-only KV-Aug (for training audio adapter)
AUDIO_ONLY_KVAUG_CONFIG = {
    "name": "audio_only_kvaug",
    "description": "Audio-only KV augmentation config for training audio adapter",

    **_BASE_LLM_CONFIG,
    **_AUDIO_ENCODER_CONFIG,
    **_PROJECTOR_CONFIG,

    "fusion_type": "multilayer",
    "fusion_layer_indices": _FUSION_LAYERS,
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "kv_augment",
        "num_attention_heads": 40,
        "head_dim": 128,
        "bottleneck_dim": 64,
        "use_bottleneck": True,
        "dropout": 0.1,
        "query_adapter_rank": 16,
        "modalities": {
            "audio": {
                "layer_indices": _FUSION_LAYERS,
                "num_tokens": 8,
            },
        },
    },

    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,
}

# Point cloud-only Pre-FFN (for training PC adapter)
POINTCLOUD_ONLY_PREFFN_CONFIG = {
    "name": "pointcloud_only_preffn",
    "description": "Point cloud-only pre-FFN config for training PC adapter",

    **_BASE_LLM_CONFIG,
    **_POINTCLOUD_ENCODER_CONFIG,
    **_PROJECTOR_CONFIG,

    "fusion_type": "multilayer",
    "fusion_layer_indices": _FUSION_LAYERS,
    "lora_rank": 16,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "num_attention_heads": 40,
        "attention_dropout": 0.1,
        "modalities": {
            "pointcloud": {
                "layer_indices": _FUSION_LAYERS,
                "num_tokens": 32,  # Match ModelNet40 training config
            },
        },
        "use_bottleneck": True,
        "bottleneck_dim": 256,
        "use_ffn": True,
        "ffn_expansion": 2.0,
        "use_pre_norm": True,
        "use_tokenwise_gate": True,
    },

    "freeze_pointcloud_encoder": True,
    "label_smoothing": 0.1,
}

# Point cloud-only KV-Aug (for training PC adapter)
POINTCLOUD_ONLY_KVAUG_CONFIG = {
    "name": "pointcloud_only_kvaug",
    "description": "Point cloud-only KV augmentation config for training PC adapter",

    **_BASE_LLM_CONFIG,
    **_POINTCLOUD_ENCODER_CONFIG,
    **_PROJECTOR_CONFIG,

    "fusion_type": "multilayer",
    "fusion_layer_indices": _FUSION_LAYERS,
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "kv_augment",
        "num_attention_heads": 40,
        "head_dim": 128,
        "bottleneck_dim": 64,
        "use_bottleneck": True,
        "dropout": 0.1,
        "query_adapter_rank": 16,
        "modalities": {
            "pointcloud": {
                "layer_indices": _FUSION_LAYERS,
                "num_tokens": 32,  # Match ModelNet40 training config
            },
        },
    },

    "freeze_pointcloud_encoder": True,
    "label_smoothing": 0.1,
}


# =============================================================================
# Config registry
# =============================================================================
COMPOSITION_CONFIGS = {
    # Full multi-modal
    "composition_preffn": COMPOSITION_PREFFN_CONFIG,
    "composition_kvaug": COMPOSITION_KVAUG_CONFIG,

    # Single-modality for training
    "audio_preffn": AUDIO_ONLY_PREFFN_CONFIG,
    "audio_kvaug": AUDIO_ONLY_KVAUG_CONFIG,
    "pointcloud_preffn": POINTCLOUD_ONLY_PREFFN_CONFIG,
    "pointcloud_kvaug": POINTCLOUD_ONLY_KVAUG_CONFIG,
}


def get_composition_config(name: str) -> dict:
    """Get composition config by name."""
    if name not in COMPOSITION_CONFIGS:
        available = ", ".join(COMPOSITION_CONFIGS.keys())
        raise ValueError(f"Unknown composition config '{name}'. Available: {available}")
    return COMPOSITION_CONFIGS[name].copy()
