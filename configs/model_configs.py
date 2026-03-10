# -*- coding: utf-8 -*-
"""
Model configurations for SAFE training.
Provides both demo and full production configurations.
"""

import os
import copy

# Demo configuration - lightweight for testing
DEMO_CONFIG = {
    "name": "demo",
    "description": "Lightweight configuration for testing and development with smaller VL model",
    
    # Base VL Model - Using smaller but proper vision-language model
    "llm_model_name": "Salesforce/blip2-opt-2.7b",  # Smaller VL model for demo
    "vision_model_name": "openai/clip-vit-base-patch32",  # Smaller vision encoder
    
    # Audio configuration
    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/smaller_clap_general",
        "sample_rate": 48000,
        "max_length": 10.0
    },
    
    # Model dimensions - Updated for BLIP2 OPT 2.7B
    "llm_hidden_size": 2560,  # OPT 2.7B hidden size
    "audio_embed_dim": 512,
    "vision_embed_dim": 512,  # CLIP Base
    
    # Projector configuration
    "projector_type": "standard",
    "num_audio_tokens": 8,
    "projector_config": {
        "dropout": 0.1
    },
    
    # Fusion configuration - Updated for OPT 2.7B architecture
    "fusion_type": "multilayer",
    "fusion_layer_indices": [6, 16, 24],  # Early, mid, and late fusion for OPT 2.7B
    "lora_rank": 8,
    "fusion_config": {
        "num_attention_heads": 20,  # OPT 2.7B attention heads
        "attention_dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [6, 16, 24],
                "num_tokens": 8
            }
        }
    },
    
    # Training configuration
    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    
    # Memory and compute - More reasonable for demo
    "expected_vram_gb": 8,  # More accessible for testing
    "recommended_batch_size": 4,
    "gradient_accumulation_steps": 2
}

# Full production configuration - as per SAFE paper specification
FULL_CONFIG = {
    "name": "full",
    "description": "Full production configuration matching SAFE paper specification",
    
    # Base VL Model - Full LLaVA model for production
    "llm_model_name": "llava-hf/llava-1.5-13b-hf",  # LLaVA 13B for full model
    "vision_model_name": "openai/clip-vit-large-patch14",  # CLIP ViT-L
    
    # Audio configuration
    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0
    },
    
    # Model dimensions (LLaVA 13B)
    "llm_hidden_size": 5120,   # LLaVA 13B hidden size
    "audio_embed_dim": 512,
    "vision_embed_dim": 1024,  # CLIP ViT-L
    
    # Projector configuration
    "projector_type": "standard",
    "num_audio_tokens": 16,  # Increased from 8 for more detailed audio representations
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024  # Bottleneck for 80% parameter reduction (212M → 42M)
    },

    # Fusion configuration
    "fusion_type": "multilayer",
    "fusion_layer_indices": [6, 12, 24],   # Add early fusion tap-in for richer alignment
    "lora_rank": 8,
    "fusion_config": {
        "num_attention_heads": 40,  # LLaVA 13B attention heads
        "attention_dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [6, 12, 24],  # Match fusion_layer_indices
                "num_tokens": 16  # Match num_audio_tokens
            }
        }
    },
    
    # Training configuration
    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    
    # Memory and compute
    "expected_vram_gb": 32,  # LLaVA 13B requires substantial memory
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 8
}

# Multimodal configuration with Whisper support
MULTIMODAL_CONFIG = {
    "name": "multimodal",
    "description": "Full configuration with both CLAP and Whisper audio encoders",

    # Base VL Model - Same as full for multimodal
    "llm_model_name": "llava-hf/llava-1.5-13b-hf",
    "vision_model_name": "openai/clip-vit-large-patch14",

    # Audio configuration - both CLAP and Whisper
    "audio_encoder_type": "multimodal",
    "audio_encoder_config": {
        "use_clap": True,
        "use_whisper": True,
        "clap_config": {
            "model_name": "laion/larger_clap_music_and_speech",
            "sample_rate": 48000,
            "max_length": 10.0
        },
        "whisper_config": {
            "model_name": "whisper-small",
            "extract_transcript": True,
            "sample_rate": 16000,
            "max_length": 30.0
        }
    },

    # Model dimensions
    "llm_hidden_size": 5120,  # LLaVA 13B hidden size
    "audio_embed_dim": 512 + 768,  # CLAP + Whisper combined
    "vision_embed_dim": 1024,

    # Projector configuration
    "projector_type": "adaptive",  # Adaptive projector for variable tokens
    "num_audio_tokens": 12,  # More tokens for multimodal audio
    "projector_config": {
        "max_audio_tokens": 12,
        "min_audio_tokens": 4,
        "dropout": 0.1
    },

    # Fusion configuration
    "fusion_type": "multilayer",
    "fusion_layer_indices": [15, 25, 35],  # Multiple fusion layers for LLaVA 13B
    "lora_rank": 16,  # Higher rank for more complex fusion
    "fusion_config": {
        "num_attention_heads": 40,  # LLaVA 13B attention heads
        "attention_dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [15, 25, 35],
                "num_tokens": 12
            }
        }
    },

    # Training configuration
    "freeze_base_vl": True,
    "freeze_audio_encoder": True,

    # Memory and compute
    "expected_vram_gb": 40,  # Multimodal with LLaVA 13B requires even more memory
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16
}

# Phase 1: Signal Verification Configuration
# This config tests if the frozen LLM can learn to attend to audio by removing
# all capacity bottlenecks identified in the architectural review.
PHASE1_CONFIG = {
    "name": "phase1",
    "description": "Phase 1: Signal verification - prove frozen LLM can listen to audio",

    # Base VL Model - Same as full
    "llm_model_name": "llava-hf/llava-1.5-13b-hf",
    "vision_model_name": "openai/clip-vit-large-patch14",

    # Audio configuration - Same as full
    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0
    },

    # Model dimensions
    "llm_hidden_size": 5120,
    "audio_embed_dim": 512,
    "vision_embed_dim": 1024,

    # Projector configuration
    "projector_type": "standard",
    # 8 tokens for efficient audio representation
    "num_audio_tokens": 8,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,  # Keep for Phase 1, remove in Phase 2
        # NEW: SwiGLU activation for better gradient flow (used in LLaMA/PaLM)
        "use_swiglu": True,
        # NEW: Positional embeddings for temporal structure in audio tokens
        "use_positional_embedding": True,
    },

    # Fusion configuration - MULTI-LAYER + HIGHER RANK
    "fusion_type": "multilayer",
    # Phase 1: 6-layer injection distributed across early-to-mid layers
    "fusion_layer_indices": [1, 5, 9, 13, 17, 21],
    # Keep LoRA rank modest; training full cross-attention matrices is too large.
    "lora_rank": 16,
    "fusion_config": {
        "num_attention_heads": 40,
        "attention_dropout": 0.1,
        "modalities": {
            "audio": {
                # Match fusion_layer_indices and num_audio_tokens
                "layer_indices": [12, 24, 36],
                "num_tokens": 16
            }
        },
        # Use bottleneck cross-attention instead of LoRA (simpler, no PEFT dependency)
        # bottleneck_dim=256 for increased capacity
        "use_bottleneck": True,
        "bottleneck_dim": 256,
        # NEW: Add FFN after cross-attention (standard transformer pattern)
        # This provides crucial non-linear transformation capacity
        "use_ffn": True,
        "ffn_expansion": 2.0,
        # NEW: Use pre-norm for more stable training (like modern LLMs)
        "use_pre_norm": True,
        # Legacy LoRA settings (only used if use_bottleneck=False)
        "target_modules": ["query", "key", "value", "output_dense"],
        "train_base_cross_attention": False,
        # Default fusion injection point for Phase 1: inject BEFORE FFN
        "injection_point": "pre_ffn",
        # Enable token-wise gating of audio residuals in Phase 1 experiments
        "use_tokenwise_gate": True,
    },

    # Training configuration
    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,  # Prevents overconfidence, improves generalization

    # Memory and compute (increased due to more tokens/rank)
    "expected_vram_gb": 40,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16  # Target effective batch size of 128
}

# KV Augmentation Configuration
# Audio tokens are injected as additional K,V in LLM self-attention
# This makes audio "un-ignorable" by the frozen LLM
KV_AUGMENT_CONFIG = {
    "name": "kv_augment",
    "description": "KV Augmentation - audio injected into LLM self-attention as additional K,V",
    # Eval prompt: keep it direct and avoid meta-explanations.
    "eval_prompt": "Describe what you hear in one short sentence.",

    # Base VL Model - LLaVA 13B
    "llm_model_name": "llava-hf/llava-1.5-13b-hf",
    "vision_model_name": "openai/clip-vit-large-patch14",

    # Audio configuration
    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0
    },

    # Model dimensions
    "llm_hidden_size": 5120,
    "audio_embed_dim": 512,
    "vision_embed_dim": 1024,

    # Projector configuration
    "projector_type": "standard",
    "num_audio_tokens": 8,  # Fewer tokens for classification
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    # Fusion configuration - KV AUGMENTATION MODE
    "fusion_type": "multilayer",
    # Layer indices for LLaVA-13B (40 layers total):
    # Try {16, 24, 32} first - mid layers where classification info forms
    # Alternative: {12, 20, 28} if above doesn't work
    # Avoid ultra-late layers (36+) until traction is seen
    "fusion_layer_indices": [16, 24, 32],
    "lora_rank": 8,  # Not used in kv_augment mode but kept for compatibility
    "fusion_config": {
        # KEY: Enable KV augmentation mode
        "fusion_mode": "kv_augment",

        # LLaVA 13B attention configuration
        "num_attention_heads": 40,
        "head_dim": 128,

        # Bottleneck for K,V projections (reduces params)
        "bottleneck_dim": 64,
        "use_bottleneck": True,
        "dropout": 0.1,

        # Query adapter for audio attention (LoRA-style ΔQ)
        # This allows frozen queries to attend to audio K,V
        "query_adapter_rank": 16,

        # Minimum attention regularization - ENABLED for captioning
        # Forces audio attention early to prevent language-prior shortcuts
        # Use a step-based curriculum:
        # - steps < warmup: disabled (weight=0, min_attention=0)
        # - warmup → warmup+ramp: linearly ramp to targets
        # This avoids early training being dominated by the regularizer.
        "min_audio_attention_start": 0.0,
        "min_audio_attention_weight_start": 0.0,
        "min_audio_attention": 0.02,
        "min_audio_attention_weight": 0.05,
        "min_audio_attention_warmup_steps": 1000,
        "min_audio_attention_ramp_steps": 4000,
        # Attention-shape regularization (stability against collapse/dominance)
        "min_audio_attention_entropy_min": 0.20,
        "min_audio_attention_entropy_max": 0.98,
        "max_audio_token_attention": 0.95,
        "audio_attention_entropy_weight": 1.0,
        "audio_attention_dominance_weight": 0.5,
        # Modality dropout (randomly disable audio rows during training)
        "kv_modality_dropout_prob": 0.10,
        # Alert thresholds for KV diagnostics (logged via kv_hook_manager)
        "kv_alerts_enabled": True,
        "kv_alert_log_every": 100,
        "kv_alert_rms_ratio_low": 0.01,
        "kv_alert_rms_ratio_high": 0.40,
        "kv_alert_entropy_low": 0.15,
        "kv_alert_entropy_high": 0.98,

        # Modality configuration (for compatibility with existing code)
        "modalities": {
            "audio": {
                "layer_indices": [16, 24, 32],  # Match fusion_layer_indices
                "num_tokens": 8
            }
        },
    },

    # Training configuration
    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,

    # Memory and compute
    "expected_vram_gb": 35,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 8
}

# Qwen-3 8B Configuration
# Uses Qwen-3 8B as base LLM instead of LLaVA for potentially better performance
# IMPORTANT: Run with these env vars:
#   SAFE_QWEN_QUANT=none SAFE_GRAD_CKPT=0 FP16=0 BATCH_SIZE=1
QWEN3_14B_CONFIG = {
    "name": "qwen3_8b",
    "description": "Qwen-3 8B base LLM with residual fusion - modern LLM for audio captioning",
    "eval_prompt": "Describe what you hear in one short sentence.",

    # Base LLM - Qwen3-8B (no vision, audio-only)
    # Uses local path by default; set LLM_MODEL_PATH env var to override
    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/Qwen_Qwen3-8B"),
    "vision_model_name": None,  # No vision encoder needed for audio-only

    # Audio configuration (same as KV_AUGMENT)
    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0
    },

    # Model dimensions - Qwen-3 8B specs
    "llm_hidden_size": 4096,
    "audio_embed_dim": 512,
    "vision_embed_dim": 1024,

    # Projector configuration (same as KV_AUGMENT)
    "projector_type": "standard",
    "num_audio_tokens": 8,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    # Fusion configuration - KV AUGMENTATION MODE
    "fusion_type": "multilayer",
    # Qwen-3 8B has 32 layers
    # Proportionally: [12,24,36] * 32/40 = [10,19,29]
    "fusion_layer_indices": [10, 19, 29],
    "lora_rank": 8,
    "fusion_config": {
        # Use simpler residual fusion for now (KV augment has compatibility issues)
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",  # Match LLaVA Phase1 config
        "num_attention_heads": 32,
        "dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [10, 19, 29],
                "num_tokens": 8
            }
        },
    },

    # Training configuration
    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,

    # Memory and compute - Qwen3-8B in bf16 requires ~32GB VRAM
    # Must use: SAFE_QWEN_QUANT=none SAFE_GRAD_CKPT=0 FP16=0 BATCH_SIZE=1
    "expected_vram_gb": 32,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16,
    # Qwen-specific settings (set via env vars)
    "qwen_quant": "none",  # bf16 only, no quantization
    "gradient_checkpointing": False,
    "fp16_mixed_precision": False,  # bf16 incompatible with fp16 scaler
}

# InternVL 3.5-8B Configuration
# Uses InternVL 3.5-8B as base VLM (InternViT-300M + Qwen3-8B backbone)
# This is a vision-language model with built-in vision encoder.
# IMPORTANT: Run with these env vars:
#   SAFE_QWEN_QUANT=none SAFE_GRAD_CKPT=0 FP16=0 BATCH_SIZE=1
INTERNVL_CONFIG = {
    "name": "internvl",
    "description": "InternVL 3.5-8B VLM with SAFE audio adapters - vision+audio composition",
    "eval_prompt": "Describe what you hear in one short sentence.",

    # Base VL Model - InternVL 3.5-8B (has built-in InternViT-300M vision encoder)
    # Uses local path by default; set LLM_MODEL_PATH env var to override
    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/OpenGVLab_InternVL3_5-8B"),
    "vision_model_name": "built-in",  # InternVL has built-in InternViT-300M (not None)

    # InternVL vision pipeline defaults (from InternVL 3.5-8B config)
    "image_token_id": 151667,       # Token ID for image placeholder in vocabulary
    "image_seq_length": 256,        # Number of placeholder tokens per image
    "downsample_ratio": 0.5,        # Pixel shuffle downsample ratio
    "image_size": [448, 448],       # Input image resolution

    # Audio configuration (same as Qwen/KV_AUGMENT)
    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0
    },

    # Model dimensions - Qwen3-8B backbone inside InternVL
    "llm_hidden_size": 4096,
    "audio_embed_dim": 512,
    "num_audio_tokens": 8,

    # Projector configuration (same as Qwen config)
    "projector_type": "standard",
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    # Fusion configuration - residual mode.
    # Default schedule is aligned with Phase1/LLaVA recipe for parity studies.
    "fusion_type": "multilayer",
    "fusion_layer_indices": [1, 5, 9, 13, 17, 21],
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "num_attention_heads": 32,
        "dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [1, 5, 9, 13, 17, 21],
                "num_tokens": 8
            }
        },
    },

    # Training configuration
    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,

    # Memory and compute - InternVL 8B (InternViT + Qwen3-8B) in bf16
    "expected_vram_gb": 35,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 8,
}

# Symmetric Composition Configuration
# Validates the hypothesis that SAFE aligns modalities with underlying textual
# spaces by training separate vision and audio SAFE adapter branches on
# text-only Qwen3-8B, then composing both at inference time without joint training.
COMPOSITION_CONFIG = {
    "name": "composition",
    "description": "Symmetric composition: CLIP + CLAP as SAFE adapters on text-only Qwen3-8B",
    "eval_prompt": "Answer with a single word or number.",

    # Base LLM - Qwen3-8B (text-only, no native vision)
    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/Qwen_Qwen3-8B"),
    # CLIP loaded by BaseVL as separate vision encoder (not built-in)
    "vision_model_name": "openai/clip-vit-large-patch14",

    # Audio configuration (same as Qwen config)
    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0,
    },

    # Model dimensions
    "llm_hidden_size": 4096,
    "audio_embed_dim": 512,
    "vision_embed_dim": 1024,  # CLIP ViT-L hidden dim

    # Audio projector configuration (same as Qwen config)
    "projector_type": "standard",
    "num_audio_tokens": 8,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    # Vision projector settings (TokenSetProjector for CLIP spatial tokens)
    "num_vision_tokens": 8,
    "vision_projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_positional_embedding": True,
    },

    # Fusion: audio AND vision at same decoder layers
    "fusion_type": "multilayer",
    # Qwen-3 8B has 32 layers; proportionally [10, 19, 29]
    "fusion_layer_indices": [10, 19, 29],
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "use_bottleneck": True,
        "bottleneck_dim": 256,
        "num_attention_heads": 32,
        "dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [10, 19, 29],
                "num_tokens": 8,
            },
            "vision": {
                "layer_indices": [10, 19, 29],
                "num_tokens": 8,
            },
        },
    },

    # Training configuration
    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,

    # Memory and compute
    "expected_vram_gb": 38,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16,
}

# Composition Study Configuration
# For deeper composition analysis:
# - denser mid-layer placement
# - learned per-layer scalar gates enabled by default
COMPOSITION_STUDY_CONFIG = {
    "name": "composition_study",
    "description": "Composition study: mid-dense layers + learned per-layer gates on Qwen3-8B",
    "eval_prompt": "Answer with a single word or number.",

    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/Qwen_Qwen3-8B"),
    "vision_model_name": "openai/clip-vit-large-patch14",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0,
    },

    "llm_hidden_size": 4096,
    "audio_embed_dim": 512,
    "vision_embed_dim": 1024,

    "projector_type": "standard",
    "num_audio_tokens": 8,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    "num_vision_tokens": 8,
    "vision_projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_positional_embedding": True,
    },

    "fusion_type": "multilayer",
    "fusion_layer_indices": [8, 14, 20, 26],
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "use_bottleneck": True,
        "bottleneck_dim": 256,
        "num_attention_heads": 32,
        "dropout": 0.1,
        "use_tokenwise_gate": False,
        "use_learned_gate": True,
        "learned_gate_init": 0.2,
        "modalities": {
            "audio": {
                "layer_indices": [8, 14, 20, 26],
                "num_tokens": 8,
            },
            "vision": {
                "layer_indices": [8, 14, 20, 26],
                "num_tokens": 8,
            },
        },
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,

    "expected_vram_gb": 40,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16,
}

# Independent Composition Configuration
# Intended for zero-shot composition of separately trained adapters:
# - staggered modality layers to reduce residual interference
# - fixed scalar gate control (no learned per-layer gates by default)
COMPOSITION_INDEPENDENT_CONFIG = {
    "name": "composition_independent",
    "description": "Independent composition: staggered audio/vision layers with fixed gating on Qwen3-8B",
    "eval_prompt": "Answer with a single word or number.",

    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/Qwen_Qwen3-8B"),
    "vision_model_name": "openai/clip-vit-large-patch14",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0,
    },

    "llm_hidden_size": 4096,
    "audio_embed_dim": 512,
    "vision_embed_dim": 1024,

    "projector_type": "standard",
    "num_audio_tokens": 8,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    "num_vision_tokens": 8,
    "vision_projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_positional_embedding": True,
    },

    "fusion_type": "multilayer",
    # Union of modality layer sets. MultiLayerFusionAdapter uses per-modality map below.
    "fusion_layer_indices": [8, 10, 14, 16, 20, 22, 26, 28],
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "use_bottleneck": True,
        "bottleneck_dim": 256,
        "num_attention_heads": 32,
        "dropout": 0.1,
        "use_tokenwise_gate": False,
        "use_learned_gate": False,
        "modalities": {
            "audio": {
                "layer_indices": [8, 14, 20, 26],
                "num_tokens": 8,
            },
            "vision": {
                "layer_indices": [10, 16, 22, 28],
                "num_tokens": 8,
            },
        },
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,

    "expected_vram_gb": 40,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16,
}

# TTC-focused independent composition.
# Same staggered modality layout as composition_independent, but named explicitly
# for the unimodal-train -> compose -> test-time-compute pipeline.
COMPOSITION_TTC_CONFIG = {
    **COMPOSITION_STUDY_CONFIG,
    "name": "composition_ttc",
    "description": "Independent composition with shared fusion layers for gate/interactions TTC on MUSIC-AVQA",
}
COMPOSITION_TTC_CONFIG["fusion_config"] = {
    **COMPOSITION_STUDY_CONFIG["fusion_config"],
    "use_learned_gate": False,
}

# Operator-ladder baselines trained with shared-model interleaved unimodal batches.
# These keep the same shared fusion layers but replace raw residual addition with a
# learned composition operator over modality-specific residual updates.
COMPOSITION_AFFINE_CONFIG = {
    **COMPOSITION_STUDY_CONFIG,
    "name": "composition_affine",
    "description": "Shared-layer affine composition operator on MUSIC-AVQA",
}
COMPOSITION_AFFINE_CONFIG["fusion_config"] = {
    **COMPOSITION_STUDY_CONFIG["fusion_config"],
    "fusion_mode": "affine",
    "use_learned_gate": False,
    "affine_rank": 64,
}

COMPOSITION_FIXED_POINT_CONFIG = {
    **COMPOSITION_STUDY_CONFIG,
    "name": "composition_fixed_point",
    "description": "Shared-layer fixed-point composition operator on MUSIC-AVQA",
}
COMPOSITION_FIXED_POINT_CONFIG["fusion_config"] = {
    **COMPOSITION_STUDY_CONFIG["fusion_config"],
    "fusion_mode": "fixed_point",
    "use_learned_gate": False,
    "fixed_point_state_dim": 256,
    "fixed_point_steps": 3,
    "fixed_point_dropout": 0.1,
}

COMPOSITION_FIXED_POINT_T1_CONFIG = {
    **COMPOSITION_FIXED_POINT_CONFIG,
    "name": "composition_fixed_point_t1",
    "description": "Shared-layer fixed-point composition operator with one step (T=1) on MUSIC-AVQA",
}
COMPOSITION_FIXED_POINT_T1_CONFIG["fusion_config"] = {
    **COMPOSITION_FIXED_POINT_CONFIG["fusion_config"],
    "fixed_point_steps": 1,
}

# Disjoint composition: contiguous blocks with maximum layer separation.
# Audio injects at early-mid layers [6,8,10], vision at mid layers [16,18,20].
# 6 intervening layers (10→16) for Jacobian chain to absorb audio perturbation
# before vision injects — strongest separation of any disjoint config.
COMPOSITION_DISJOINT_CONFIG = {
    "name": "composition_disjoint",
    "description": "Disjoint composition: contiguous audio [6,8,10] / vision [16,18,20] blocks on Qwen3-8B",
    "eval_prompt": "Answer with a single word or number.",

    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/Qwen_Qwen3-8B"),
    "vision_model_name": "openai/clip-vit-large-patch14",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0,
    },

    "llm_hidden_size": 4096,
    "audio_embed_dim": 512,
    "vision_embed_dim": 1024,

    "projector_type": "standard",
    "num_audio_tokens": 8,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    "num_vision_tokens": 8,
    "vision_projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_positional_embedding": True,
    },

    "fusion_type": "multilayer",
    "fusion_layer_indices": [6, 8, 10, 16, 18, 20],
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "use_bottleneck": True,
        "bottleneck_dim": 256,
        "num_attention_heads": 32,
        "dropout": 0.1,
        "use_tokenwise_gate": False,
        "use_learned_gate": False,
        "modalities": {
            "audio": {
                "layer_indices": [6, 8, 10],
                "num_tokens": 8,
            },
            "vision": {
                "layer_indices": [16, 18, 20],
                "num_tokens": 8,
            },
        },
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,

    "expected_vram_gb": 40,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16,
}

# RKCA (Input Token Concatenation): modality tokens prepended to text embeddings
# instead of mid-layer fusion. Audio and vision projectors trained independently
# (interleaved), then compose at eval via simple concatenation.
RKCA_CONFIG = {
    "name": "rkca",
    "description": "RKCA: input token concatenation (no mid-layer fusion) on Qwen3-8B",
    "eval_prompt": "Answer with a single word or number.",

    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/Qwen_Qwen3-8B"),
    "vision_model_name": "openai/clip-vit-large-patch14",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0,
    },

    "llm_hidden_size": 4096,
    "audio_embed_dim": 512,
    "vision_embed_dim": 1024,

    "projector_type": "standard",
    "num_audio_tokens": 8,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    "num_vision_tokens": 8,
    "vision_projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_positional_embedding": True,
    },

    "fusion_type": "concat",
    "fusion_layer_indices": [],
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "concat",
        "injection_point": "pre_ffn",
        "use_bottleneck": False,
        "bottleneck_dim": 256,
        "num_attention_heads": 32,
        "dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [],
                "num_tokens": 8,
            },
            "vision": {
                "layer_indices": [],
                "num_tokens": 8,
            },
        },
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,

    "expected_vram_gb": 38,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16,
}

RKCA_SUBSPACE_CONFIG = {
    **RKCA_CONFIG,
    "name": "rkca_subspace",
    "description": "RKCA with subspace avoidance between modality projectors",
}

RKCA_JOINT_CONFIG = {
    "name": "rkca_joint",
    "description": "RKCA joint: concat audio on InternVL 8B with native vision always active",
    "eval_prompt": "Answer with a single word or number.",

    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/OpenGVLab_InternVL3_5-8B"),
    "vision_model_name": "built-in",

    "image_token_id": 151667,
    "image_seq_length": 256,
    "downsample_ratio": 0.5,
    "image_size": [448, 448],

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0,
    },

    "llm_hidden_size": 4096,
    "audio_embed_dim": 512,
    "num_audio_tokens": 8,
    "projector_type": "standard",
    "projector_config": {
        "dropout": 0.1, "bottleneck_dim": 1024,
        "use_swiglu": True, "use_positional_embedding": True,
    },

    # RKCA concat for audio only — NO vision in modalities
    "fusion_type": "concat",
    "fusion_layer_indices": [],
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "concat",
        "modalities": {
            "audio": {"layer_indices": [], "num_tokens": 8},
        },
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,
    "expected_vram_gb": 42,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 8,
}

# Higher-capacity InternVL 3.5-8B concat configuration for caption pre-alignment
# and downstream AVQA. The main increase is projector capacity:
# - 16 audio tokens instead of 8
# - 2048 projector bottleneck instead of 1024
# This keeps the simple concat interface but gives the projector more room to
# preserve description-level audio information before MUSIC-AVQA fine-tuning.
RKCA_JOINT_CAPTION16_CONFIG = {
    **RKCA_JOINT_CONFIG,
    "name": "rkca_joint_caption16",
    "description": "InternVL 3.5-8B concat audio with higher-capacity projector for AudioCaps -> MUSIC-AVQA",
    "num_audio_tokens": 16,
    "projector_config": {
        **RKCA_JOINT_CONFIG["projector_config"],
        "bottleneck_dim": 2048,
    },
}
RKCA_JOINT_CAPTION16_CONFIG["fusion_config"] = {
    **RKCA_JOINT_CONFIG["fusion_config"],
    "modalities": {
        "audio": {"layer_indices": [], "num_tokens": 16},
    },
}

# InternVL 3.5-14B: Qwen3-14B backbone (40 layers, hidden=5120)
INTERNVL_14B_CONFIG = {
    "name": "internvl_14b",
    "description": "InternVL 3.5-14B VLM with SAFE audio adapters",
    "eval_prompt": "Describe what you hear in one short sentence.",

    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/OpenGVLab_InternVL3_5-14B"),
    "vision_model_name": "built-in",

    # InternVL vision pipeline defaults
    "image_token_id": 151667,
    "image_seq_length": 256,
    "downsample_ratio": 0.5,
    "image_size": [448, 448],

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0,
    },

    # Model dimensions - Qwen3-14B backbone (40 layers)
    "llm_hidden_size": 5120,
    "audio_embed_dim": 512,
    "num_audio_tokens": 8,

    "projector_type": "standard",
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    # Fusion: 6 layers spread across 40-layer decoder, mid-heavy per gradient attribution
    "fusion_type": "multilayer",
    "fusion_layer_indices": [5, 11, 17, 23, 29, 35],
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "num_attention_heads": 40,
        "dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [5, 11, 17, 23, 29, 35],
                "num_tokens": 8,
            }
        },
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,

    "expected_vram_gb": 55,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16,
}

# InternVL 3.5-38B: Qwen3-32B backbone (64 layers, hidden=5120)
INTERNVL_38B_CONFIG = {
    "name": "internvl_38b",
    "description": "InternVL 3.5-38B VLM with SAFE audio adapters",
    "eval_prompt": "Describe what you hear in one short sentence.",

    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/OpenGVLab_InternVL3_5-38B"),
    "vision_model_name": "built-in",

    # InternVL vision pipeline defaults
    "image_token_id": 151667,
    "image_seq_length": 256,
    "downsample_ratio": 0.5,
    "image_size": [448, 448],

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0,
    },

    # Model dimensions - Qwen3-32B backbone (64 layers)
    "llm_hidden_size": 5120,
    "audio_embed_dim": 512,
    "num_audio_tokens": 8,

    "projector_type": "standard",
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    # Fusion: 6 layers spread across 64-layer decoder, mid-heavy per gradient attribution
    "fusion_type": "multilayer",
    "fusion_layer_indices": [8, 18, 28, 38, 48, 56],
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "num_attention_heads": 64,
        "dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [8, 18, 28, 38, 48, 56],
                "num_tokens": 8,
            }
        },
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,

    "expected_vram_gb": 90,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16,
}

# LoRA Baseline Configuration
# Demonstrates degradation when adding modalities via LoRA weight modification
# (Table 1 comparison): prepend tokens + LoRA on Qwen3-8B self-attention
LORA_BASELINE_CONFIG = {
    "name": "lora_baseline",
    "description": "LoRA baseline: prepend tokens + LoRA on Qwen3-8B attention",
    "eval_prompt": "Answer with exactly one short answer token (single word or number).",

    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/Qwen_Qwen3-8B"),
    "vision_model_name": "openai/clip-vit-large-patch14",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0,
    },

    "llm_hidden_size": 4096,
    "audio_embed_dim": 512,
    "vision_embed_dim": 1024,
    "num_audio_tokens": 8,
    "num_vision_tokens": 8,

    # LoRA hyperparameters (rank 8 on q_proj, v_proj = ~4.2M params)
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05,

    "freeze_audio_encoder": True,
    "freeze_vision_encoder": True,
    "label_smoothing": 0.1,

    "expected_vram_gb": 38,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16,
}

# InternVL 3.5-8B Modality Binding Configuration
# Audio-only training on InternVL (which has built-in InternViT vision).
# At eval, pass vision inputs to test whether InternVL's native vision
# composes with the trained audio adapter through the frozen backbone
# ("modality binding" hypothesis).
# Uses bottleneck_dim=756 projector (756/4096 = 18.5% ratio).
INTERNVL_BINDING_CONFIG = {
    "name": "internvl_binding",
    "description": "InternVL 8B modality binding: audio-only training, test vision composition at eval",
    "eval_prompt": "Answer with a single word or number.",

    # Base VL Model - InternVL 3.5-8B (has built-in InternViT-300M vision encoder)
    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/OpenGVLab_InternVL3_5-8B"),
    "vision_model_name": "built-in",  # InternVL has built-in InternViT-300M

    # InternVL vision pipeline defaults (frozen, used at eval only)
    "image_token_id": 151667,
    "image_seq_length": 256,
    "downsample_ratio": 0.5,
    "image_size": [448, 448],

    # Audio configuration
    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0,
    },

    # Model dimensions - Qwen3-8B backbone inside InternVL
    "llm_hidden_size": 4096,
    "audio_embed_dim": 512,
    "num_audio_tokens": 8,

    # Projector configuration - bottleneck_dim=756 (18.5% ratio)
    "projector_type": "standard",
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 756,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    # Fusion configuration - audio adapters only
    "fusion_type": "multilayer",
    "fusion_layer_indices": [1, 5, 9, 13, 17, 21],
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "use_bottleneck": True,
        "bottleneck_dim": 256,
        "num_attention_heads": 32,
        "dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [1, 5, 9, 13, 17, 21],
                "num_tokens": 8,
            },
        },
    },

    # Training configuration
    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,

    # Memory and compute
    "expected_vram_gb": 35,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 8,
}

# Qwen3-4B Composition: Vision-First Disjoint
# All vision layers BEFORE all audio layers → audio→vision transport = 0 by construction.
COMPOSITION_4B_VFIRST_CONFIG = {
    "name": "composition_4b_vfirst",
    "description": "Qwen3-4B vision-first disjoint: V={8,14,20} A={26,30,34}, structural transport elimination",
    "eval_prompt": "Answer with a single word or number.",

    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/Qwen_Qwen3-4B"),
    "vision_model_name": "openai/clip-vit-large-patch14",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0,
    },

    "llm_hidden_size": 2560,
    "audio_embed_dim": 512,
    "vision_embed_dim": 1024,

    "projector_type": "standard",
    "num_audio_tokens": 8,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 256,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    "num_vision_tokens": 8,
    "vision_projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 256,
        "use_positional_embedding": True,
    },

    "fusion_type": "multilayer",
    "fusion_layer_indices": [8, 14, 20, 26, 30, 34],
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "use_bottleneck": True,
        "bottleneck_dim": 256,
        "num_attention_heads": 32,
        "dropout": 0.1,
        "use_tokenwise_gate": False,
        "use_learned_gate": False,
        "modalities": {
            "audio": {
                "layer_indices": [26, 30, 34],
                "num_tokens": 8,
            },
            "vision": {
                "layer_indices": [8, 14, 20],
                "num_tokens": 8,
            },
        },
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,

    "expected_vram_gb": 25,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16,
}

# Qwen3-4B Composition: Tightened Staggered
# Interleaved staggered layout adapted for 36 layers + transport reg + from scratch.
COMPOSITION_4B_STAGGERED_CONFIG = {
    "name": "composition_4b_staggered",
    "description": "Qwen3-4B staggered: V={9,15,21,27} A={11,17,23,29}, transport reg + from scratch",
    "eval_prompt": "Answer with a single word or number.",

    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/Qwen_Qwen3-4B"),
    "vision_model_name": "openai/clip-vit-large-patch14",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0,
    },

    "llm_hidden_size": 2560,
    "audio_embed_dim": 512,
    "vision_embed_dim": 1024,

    "projector_type": "standard",
    "num_audio_tokens": 8,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 256,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    "num_vision_tokens": 8,
    "vision_projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 256,
        "use_positional_embedding": True,
    },

    "fusion_type": "multilayer",
    "fusion_layer_indices": [9, 11, 15, 17, 21, 23, 27, 29],
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "use_bottleneck": True,
        "bottleneck_dim": 256,
        "num_attention_heads": 32,
        "dropout": 0.1,
        "use_tokenwise_gate": False,
        "use_learned_gate": False,
        "modalities": {
            "audio": {
                "layer_indices": [11, 17, 23, 29],
                "num_tokens": 8,
            },
            "vision": {
                "layer_indices": [9, 15, 21, 27],
                "num_tokens": 8,
            },
        },
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,

    "expected_vram_gb": 25,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16,
}

# Qwen3-4B Composition: Vision-First Pairs
# Vision-first ordering within each pair + all soft objectives (kitchen sink).
COMPOSITION_4B_VFPAIRS_CONFIG = {
    "name": "composition_4b_vfpairs",
    "description": "Qwen3-4B vision-first pairs: V={8,16,24,32} A={10,18,26,34}, all soft objectives",
    "eval_prompt": "Answer with a single word or number.",

    "llm_model_name": os.environ.get("LLM_MODEL_PATH", "models/Qwen_Qwen3-4B"),
    "vision_model_name": "openai/clip-vit-large-patch14",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0,
    },

    "llm_hidden_size": 2560,
    "audio_embed_dim": 512,
    "vision_embed_dim": 1024,

    "projector_type": "standard",
    "num_audio_tokens": 8,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 256,
        "use_swiglu": True,
        "use_positional_embedding": True,
    },

    "num_vision_tokens": 8,
    "vision_projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 256,
        "use_positional_embedding": True,
    },

    "fusion_type": "multilayer",
    "fusion_layer_indices": [8, 10, 16, 18, 24, 26, 32, 34],
    "lora_rank": 8,
    "fusion_config": {
        "fusion_mode": "residual",
        "injection_point": "pre_ffn",
        "use_bottleneck": True,
        "bottleneck_dim": 256,
        "num_attention_heads": 32,
        "dropout": 0.1,
        "use_tokenwise_gate": False,
        "use_learned_gate": False,
        "modalities": {
            "audio": {
                "layer_indices": [10, 18, 26, 34],
                "num_tokens": 8,
            },
            "vision": {
                "layer_indices": [8, 16, 24, 32],
                "num_tokens": 8,
            },
        },
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,
    "label_smoothing": 0.1,

    "expected_vram_gb": 25,
    "recommended_batch_size": 1,
    "gradient_accumulation_steps": 16,
}

# Available configurations
CONFIGS = {
    "demo": DEMO_CONFIG,
    "full": FULL_CONFIG,
    "multimodal": MULTIMODAL_CONFIG,
    "phase1": PHASE1_CONFIG,
    "kv_augment": KV_AUGMENT_CONFIG,
    "qwen3_14b": QWEN3_14B_CONFIG,  # Legacy alias (actually 8B now)
    "qwen3_8b": QWEN3_14B_CONFIG,
    "qwen": QWEN3_14B_CONFIG,  # Short alias
    "internvl": INTERNVL_CONFIG,
    "internvl3.5": INTERNVL_CONFIG,
    "internvl_8b": INTERNVL_CONFIG,
    "internvl_14b": INTERNVL_14B_CONFIG,
    "internvl3.5-14b": INTERNVL_14B_CONFIG,
    "internvl_38b": INTERNVL_38B_CONFIG,
    "internvl3.5-38b": INTERNVL_38B_CONFIG,
    "composition": COMPOSITION_CONFIG,
    "composition_study": COMPOSITION_STUDY_CONFIG,
    "composition_independent": COMPOSITION_INDEPENDENT_CONFIG,
    "composition_ttc": COMPOSITION_TTC_CONFIG,
    "composition_affine": COMPOSITION_AFFINE_CONFIG,
    "composition_fixed_point": COMPOSITION_FIXED_POINT_CONFIG,
    "composition_fixed_point_t1": COMPOSITION_FIXED_POINT_T1_CONFIG,
    "composition_staggered": COMPOSITION_INDEPENDENT_CONFIG,
    "composition_disjoint": COMPOSITION_DISJOINT_CONFIG,
    "rkca": RKCA_CONFIG,
    "rkca_subspace": RKCA_SUBSPACE_CONFIG,
    "rkca_joint": RKCA_JOINT_CONFIG,
    "rkca_joint_caption16": RKCA_JOINT_CAPTION16_CONFIG,
    "lora_baseline": LORA_BASELINE_CONFIG,
    "internvl_binding": INTERNVL_BINDING_CONFIG,
    "composition_4b_vfirst": COMPOSITION_4B_VFIRST_CONFIG,
    "composition_4b_staggered": COMPOSITION_4B_STAGGERED_CONFIG,
    "composition_4b_vfpairs": COMPOSITION_4B_VFPAIRS_CONFIG,
}

def get_config(config_name: str):
    """Get configuration by name."""
    if config_name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return copy.deepcopy(CONFIGS[config_name])

def print_config_info():
    """Print information about available configurations."""
    print("Available SAFE Model Configurations:")
    print("=" * 50)
    
    for name, config in CONFIGS.items():
        print(f"\n📋 {name.upper()} Configuration")
        print(f"   Description: {config['description']}")
        print(f"   LLM: {config['llm_model_name']}")
        print(f"   Vision: {config['vision_model_name']}")
        print(f"   Audio: {config['audio_encoder_type']}")
        print(f"   Hidden Size: {config['llm_hidden_size']}")
        print(f"   Audio Tokens: {config['num_audio_tokens']}")
        print(f"   Expected VRAM: {config['expected_vram_gb']}GB")
        print(f"   Recommended Batch Size: {config['recommended_batch_size']}")

if __name__ == "__main__":
    print_config_info()
