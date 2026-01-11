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

# ============================================================================
# LLaVA 1.5 7B Configuration (Smaller variant)
# ============================================================================

LLAVA_7B_CONFIG = {
    "name": "llava-7b",
    "description": "LLaVA 1.5 7B - Smaller, faster variant for reduced memory usage",

    "llm_model_name": "llava-hf/llava-1.5-7b-hf",
    "vision_model_name": "openai/clip-vit-large-patch14",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0
    },

    "llm_hidden_size": 4096,
    "audio_embed_dim": 512,
    "vision_embed_dim": 1024,

    "projector_type": "standard",
    "num_audio_tokens": 12,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 768
    },

    "fusion_type": "multilayer",
    "fusion_layer_indices": [10, 20, 28],
    "lora_rank": 8,
    "fusion_config": {
        "num_attention_heads": 32,
        "attention_dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [10, 20, 28],
                "num_tokens": 12
            }
        }
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,

    "expected_vram_gb": 18,
    "recommended_batch_size": 2,
    "gradient_accumulation_steps": 4
}

# ============================================================================
# Text-Only Model Configurations
# ============================================================================

# Llama 3.1 8B Configuration
LLAMA_3_1_8B_CONFIG = {
    "name": "llama-3.1-8b",
    "description": "Llama 3.1 8B - Modern text-only LLM with extended context",

    "llm_model_name": "meta-llama/Meta-Llama-3.1-8B",
    "vision_model_name": None,
    "model_type": "text_only",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0
    },

    "llm_hidden_size": 4096,
    "audio_embed_dim": 512,

    "projector_type": "standard",
    "num_audio_tokens": 16,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 768
    },

    "fusion_type": "multilayer",
    "fusion_layer_indices": [10, 20, 28],
    "lora_rank": 16,
    "fusion_config": {
        "num_attention_heads": 32,
        "attention_dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [10, 20, 28],
                "num_tokens": 16
            }
        }
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,

    "expected_vram_gb": 20,
    "recommended_batch_size": 2,
    "gradient_accumulation_steps": 4
}

# Llama 3.1 8B Instruct Configuration
LLAMA_3_1_8B_INSTRUCT_CONFIG = {
    **LLAMA_3_1_8B_CONFIG,
    "name": "llama-3.1-8b-instruct",
    "description": "Llama 3.1 8B Instruct - Fine-tuned for instruction following",
    "llm_model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

# Llama 3.2 3B Configuration
LLAMA_3_2_3B_CONFIG = {
    "name": "llama-3.2-3b",
    "description": "Llama 3.2 3B - Smaller efficient model for fast iteration",

    "llm_model_name": "meta-llama/Llama-3.2-3B",
    "vision_model_name": None,
    "model_type": "text_only",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0
    },

    "llm_hidden_size": 3072,
    "audio_embed_dim": 512,

    "projector_type": "standard",
    "num_audio_tokens": 8,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 512
    },

    "fusion_type": "multilayer",
    "fusion_layer_indices": [8, 16, 24],
    "lora_rank": 8,
    "fusion_config": {
        "num_attention_heads": 24,
        "attention_dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [8, 16, 24],
                "num_tokens": 8
            }
        }
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,

    "expected_vram_gb": 10,
    "recommended_batch_size": 4,
    "gradient_accumulation_steps": 2
}

# Mistral 7B Configuration
MISTRAL_7B_CONFIG = {
    "name": "mistral-7b",
    "description": "Mistral 7B v0.3 - Strong open model with sliding window attention",

    "llm_model_name": "mistralai/Mistral-7B-v0.3",
    "vision_model_name": None,
    "model_type": "text_only",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0
    },

    "llm_hidden_size": 4096,
    "audio_embed_dim": 512,

    "projector_type": "standard",
    "num_audio_tokens": 12,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 768
    },

    "fusion_type": "multilayer",
    "fusion_layer_indices": [10, 20, 28],
    "lora_rank": 8,
    "fusion_config": {
        "num_attention_heads": 32,
        "attention_dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [10, 20, 28],
                "num_tokens": 12
            }
        }
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,

    "expected_vram_gb": 18,
    "recommended_batch_size": 2,
    "gradient_accumulation_steps": 4
}

# Qwen 2.5 7B Configuration
QWEN_2_5_7B_CONFIG = {
    "name": "qwen-2.5-7b",
    "description": "Qwen 2.5 7B - Strong multilingual capabilities",

    "llm_model_name": "Qwen/Qwen2.5-7B",
    "vision_model_name": None,
    "model_type": "text_only",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0
    },

    "llm_hidden_size": 3584,
    "audio_embed_dim": 512,

    "projector_type": "standard",
    "num_audio_tokens": 12,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 640
    },

    "fusion_type": "multilayer",
    "fusion_layer_indices": [8, 16, 24],
    "lora_rank": 8,
    "fusion_config": {
        "num_attention_heads": 28,
        "attention_dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [8, 16, 24],
                "num_tokens": 12
            }
        }
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,

    "expected_vram_gb": 18,
    "recommended_batch_size": 2,
    "gradient_accumulation_steps": 4
}

# Gemma 2 9B Configuration
GEMMA_2_9B_CONFIG = {
    "name": "gemma-2-9b",
    "description": "Gemma 2 9B - Google's open model",

    "llm_model_name": "google/gemma-2-9b",
    "vision_model_name": None,
    "model_type": "text_only",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0
    },

    "llm_hidden_size": 3584,
    "audio_embed_dim": 512,

    "projector_type": "standard",
    "num_audio_tokens": 12,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 640
    },

    "fusion_type": "multilayer",
    "fusion_layer_indices": [12, 26, 38],
    "lora_rank": 8,
    "fusion_config": {
        "num_attention_heads": 16,
        "attention_dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [12, 26, 38],
                "num_tokens": 12
            }
        }
    },

    "freeze_base_vl": True,
    "freeze_audio_encoder": True,

    "expected_vram_gb": 22,
    "recommended_batch_size": 2,
    "gradient_accumulation_steps": 4
}

# ============================================================================
# API-Based Model Configurations
# ============================================================================

GEMINI_FLASH_CONFIG = {
    "name": "gemini-flash",
    "description": "Gemini 1.5 Flash - Fast Google API model",

    "llm_model_name": "gemini-1.5-flash",
    "vision_model_name": None,
    "model_type": "api",
    "api_provider": "google",
    "api_env_var": "GOOGLE_API_KEY",

    "audio_encoder_type": "clap",
    "audio_encoder_config": {
        "model_name": "laion/larger_clap_music_and_speech",
        "sample_rate": 48000,
        "max_length": 10.0
    },

    "audio_embed_dim": 512,
    "max_tokens": 2000,
    "temperature": 0.7,

    "expected_vram_gb": 2,
    "recommended_batch_size": 1,
}

# ============================================================================
# Configuration Registry
# ============================================================================

CONFIGS = {
    # Vision-Language Models
    "demo": DEMO_CONFIG,
    "full": FULL_CONFIG,
    "multimodal": MULTIMODAL_CONFIG,
    "llava-7b": LLAVA_7B_CONFIG,

    # Text-Only Models
    "llama-3.1-8b": LLAMA_3_1_8B_CONFIG,
    "llama-3.1-8b-instruct": LLAMA_3_1_8B_INSTRUCT_CONFIG,
    "llama-3.2-3b": LLAMA_3_2_3B_CONFIG,
    "mistral-7b": MISTRAL_7B_CONFIG,
    "qwen-2.5-7b": QWEN_2_5_7B_CONFIG,
    "gemma-2-9b": GEMMA_2_9B_CONFIG,

    # API-Based Models
    "gemini-flash": GEMINI_FLASH_CONFIG,
}

# Convenience aliases
CONFIGS["llava-13b"] = FULL_CONFIG
CONFIGS["llama-8b"] = LLAMA_3_1_8B_CONFIG
CONFIGS["llama-3b"] = LLAMA_3_2_3B_CONFIG


def get_config(config_name: str):
    """Get configuration by name."""
    if config_name not in CONFIGS:
        available = ", ".join(sorted(CONFIGS.keys()))
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")

    return CONFIGS[config_name].copy()


def list_configs(model_type=None):
    """List available configurations, optionally filtered by model type."""
    configs = []
    for name, config in CONFIGS.items():
        cfg_type = config.get("model_type", "vision_language")
        if model_type is None or cfg_type == model_type:
            configs.append(name)
    return sorted(set(configs))


def print_config_info():
    """Print information about available configurations."""
    print("Available SAFE Model Configurations:")
    print("=" * 70)

    # Group by type
    vl_configs = []
    text_configs = []
    api_configs = []

    for name, config in CONFIGS.items():
        cfg_type = config.get("model_type", "vision_language")
        if cfg_type == "api":
            api_configs.append((name, config))
        elif cfg_type == "text_only":
            text_configs.append((name, config))
        else:
            vl_configs.append((name, config))

    print("\nVISION-LANGUAGE MODELS:")
    print("-" * 40)
    for name, config in sorted(vl_configs, key=lambda x: x[0]):
        vram = config.get('expected_vram_gb', 0)
        print(f"  {name:25s} | {vram:4.0f}GB | {config['description'][:30]}...")

    print("\nTEXT-ONLY MODELS:")
    print("-" * 40)
    for name, config in sorted(text_configs, key=lambda x: x[0]):
        vram = config.get('expected_vram_gb', 0)
        print(f"  {name:25s} | {vram:4.0f}GB | {config['description'][:30]}...")

    print("\nAPI-BASED MODELS:")
    print("-" * 40)
    for name, config in sorted(api_configs, key=lambda x: x[0]):
        print(f"  {name:25s} | API   | {config['description'][:30]}...")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    print_config_info()
