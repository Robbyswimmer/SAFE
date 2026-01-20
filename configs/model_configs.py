"""
Model configurations for SAFE training.
Provides both demo and full production configurations.
"""

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
    # Increased tokens for more temporal detail
    "num_audio_tokens": 16,
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
    # Phase 1: 3-layer injection at 30%/60%/90% depth for concentrated gradients
    "fusion_layer_indices": [12, 24, 36],
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
        # bottleneck_dim=64 for increased capacity
        "use_bottleneck": True,
        "bottleneck_dim": 64,
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
    # Eval prompt: keep it direct and "caption-like" to avoid refusal templates and
    # meta-explanations (e.g., defining "sound source").
    "eval_prompt": "Audio caption (one short sentence; no opinions; no definitions):",

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

# Available configurations
CONFIGS = {
    "demo": DEMO_CONFIG,
    "full": FULL_CONFIG,
    "multimodal": MULTIMODAL_CONFIG,
    "phase1": PHASE1_CONFIG,
    "kv_augment": KV_AUGMENT_CONFIG,
}

def get_config(config_name: str):
    """Get configuration by name."""
    if config_name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return CONFIGS[config_name].copy()

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
