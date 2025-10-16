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
    "fusion_layer_indices": [16, 24],  # Mid-layers for OPT 2.7B (32 layers total)
    "lora_rank": 8,
    "fusion_config": {
        "num_attention_heads": 20,  # OPT 2.7B attention heads
        "attention_dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [16, 24],
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
    "num_audio_tokens": 8,
    "projector_config": {
        "dropout": 0.1,
        "bottleneck_dim": 1024  # Bottleneck for 80% parameter reduction (212M → 42M)
    },

    # Fusion configuration
    "fusion_type": "multilayer",
    "fusion_layer_indices": [20, 30],   # Mid-layers for LLaVA 13B (40 layers total)
    "lora_rank": 8,
    "fusion_config": {
        "num_attention_heads": 40,  # LLaVA 13B attention heads
        "attention_dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": [20, 30],
                "num_tokens": 8
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

# Available configurations
CONFIGS = {
    "demo": DEMO_CONFIG,
    "full": FULL_CONFIG,
    "multimodal": MULTIMODAL_CONFIG
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
