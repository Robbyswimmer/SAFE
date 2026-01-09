"""
Model Registry for SAFE framework.
Provides centralized registration and configuration for different backbone models.
Supports vision-language models, text-only models, and API-based models.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Type
from enum import Enum


class ModelType(Enum):
    """Types of models supported by SAFE."""
    VISION_LANGUAGE = "vision_language"  # Llava, BLIP2, etc.
    TEXT_ONLY = "text_only"              # Llama, Mistral, etc.
    API = "api"                          # Gemini, GPT-4, Claude, etc.


class ModelFamily(Enum):
    """Model families for architecture-specific handling."""
    LLAVA = "llava"
    BLIP2 = "blip2"
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    PHI = "phi"
    GEMMA = "gemma"
    GEMINI = "gemini"
    GPT = "gpt"
    CLAUDE = "claude"
    CUSTOM = "custom"


@dataclass
class ModelSpec:
    """Specification for a model in the registry."""
    # Identification
    name: str
    model_id: str  # HuggingFace ID or API model name
    model_type: ModelType
    model_family: ModelFamily

    # Architecture details
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    vocab_size: int
    max_position_embeddings: int = 4096

    # Vision details (for VL models)
    has_vision: bool = False
    vision_hidden_size: Optional[int] = None
    vision_encoder: Optional[str] = None

    # Resource requirements
    expected_vram_gb: float = 8.0
    supports_flash_attention: bool = True
    supports_quantization: bool = True

    # API details (for API models)
    api_provider: Optional[str] = None
    api_env_var: Optional[str] = None

    # Recommended training settings
    recommended_batch_size: int = 1
    recommended_fusion_layers: List[int] = field(default_factory=list)
    recommended_lora_rank: int = 8

    # Additional metadata
    description: str = ""
    license: str = ""
    paper_url: Optional[str] = None

    def get_fusion_layer_indices(self, num_layers_fraction: List[float] = None) -> List[int]:
        """Get fusion layer indices based on layer fractions or defaults."""
        if self.recommended_fusion_layers:
            return self.recommended_fusion_layers

        # Default: inject at 30%, 60%, 90% of layers
        fractions = num_layers_fraction or [0.3, 0.6, 0.9]
        return [int(f * self.num_layers) for f in fractions]


# ============================================================================
# Model Registry - All supported models
# ============================================================================

MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> ModelSpec:
    """Register a model specification in the registry."""
    MODEL_REGISTRY[spec.name] = spec
    return spec


def get_model_spec(name: str) -> ModelSpec:
    """Get model specification by name."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name]


def list_models(model_type: Optional[ModelType] = None,
                model_family: Optional[ModelFamily] = None) -> List[str]:
    """List available models, optionally filtered by type or family."""
    models = []
    for name, spec in MODEL_REGISTRY.items():
        if model_type and spec.model_type != model_type:
            continue
        if model_family and spec.model_family != model_family:
            continue
        models.append(name)
    return models


# ============================================================================
# Register Vision-Language Models
# ============================================================================

# LLaVA 1.5 13B (Original SAFE target)
register_model(ModelSpec(
    name="llava-1.5-13b",
    model_id="llava-hf/llava-1.5-13b-hf",
    model_type=ModelType.VISION_LANGUAGE,
    model_family=ModelFamily.LLAVA,
    hidden_size=5120,
    num_layers=40,
    num_attention_heads=40,
    vocab_size=32000,
    max_position_embeddings=4096,
    has_vision=True,
    vision_hidden_size=1024,
    vision_encoder="openai/clip-vit-large-patch14",
    expected_vram_gb=32.0,
    recommended_batch_size=1,
    recommended_fusion_layers=[12, 24, 36],
    recommended_lora_rank=8,
    description="LLaVA 1.5 13B - Original SAFE paper target model",
    license="llama2",
    paper_url="https://arxiv.org/abs/2310.03744"
))

# LLaVA 1.5 7B (Smaller variant)
register_model(ModelSpec(
    name="llava-1.5-7b",
    model_id="llava-hf/llava-1.5-7b-hf",
    model_type=ModelType.VISION_LANGUAGE,
    model_family=ModelFamily.LLAVA,
    hidden_size=4096,
    num_layers=32,
    num_attention_heads=32,
    vocab_size=32000,
    max_position_embeddings=4096,
    has_vision=True,
    vision_hidden_size=1024,
    vision_encoder="openai/clip-vit-large-patch14",
    expected_vram_gb=16.0,
    recommended_batch_size=2,
    recommended_fusion_layers=[10, 20, 28],
    recommended_lora_rank=8,
    description="LLaVA 1.5 7B - Smaller, faster variant",
    license="llama2"
))

# LLaVA-NeXT variants
register_model(ModelSpec(
    name="llava-next-7b",
    model_id="llava-hf/llava-v1.6-mistral-7b-hf",
    model_type=ModelType.VISION_LANGUAGE,
    model_family=ModelFamily.LLAVA,
    hidden_size=4096,
    num_layers=32,
    num_attention_heads=32,
    vocab_size=32064,
    max_position_embeddings=32768,
    has_vision=True,
    vision_hidden_size=1024,
    vision_encoder="openai/clip-vit-large-patch14-336",
    expected_vram_gb=18.0,
    recommended_batch_size=2,
    recommended_fusion_layers=[10, 20, 28],
    recommended_lora_rank=8,
    description="LLaVA-NeXT 7B with Mistral backbone - improved visual reasoning",
    license="apache-2.0"
))

# BLIP2 (Demo/lightweight)
register_model(ModelSpec(
    name="blip2-opt-2.7b",
    model_id="Salesforce/blip2-opt-2.7b",
    model_type=ModelType.VISION_LANGUAGE,
    model_family=ModelFamily.BLIP2,
    hidden_size=2560,
    num_layers=32,
    num_attention_heads=32,
    vocab_size=50272,
    max_position_embeddings=2048,
    has_vision=True,
    vision_hidden_size=1408,
    vision_encoder="eva-clip",
    expected_vram_gb=8.0,
    recommended_batch_size=4,
    recommended_fusion_layers=[8, 16, 24],
    recommended_lora_rank=8,
    description="BLIP2 with OPT 2.7B - Lightweight for testing",
    license="mit"
))

# ============================================================================
# Register Text-Only Models (Llama family)
# ============================================================================

# Llama 3.1 8B
register_model(ModelSpec(
    name="llama-3.1-8b",
    model_id="meta-llama/Meta-Llama-3.1-8B",
    model_type=ModelType.TEXT_ONLY,
    model_family=ModelFamily.LLAMA,
    hidden_size=4096,
    num_layers=32,
    num_attention_heads=32,
    vocab_size=128256,
    max_position_embeddings=131072,
    has_vision=False,
    expected_vram_gb=18.0,
    supports_flash_attention=True,
    recommended_batch_size=2,
    recommended_fusion_layers=[10, 20, 28],
    recommended_lora_rank=16,
    description="Llama 3.1 8B - Modern text-only LLM with extended context",
    license="llama3.1",
    paper_url="https://ai.meta.com/research/publications/the-llama-3-herd-of-models/"
))

# Llama 3.1 8B Instruct
register_model(ModelSpec(
    name="llama-3.1-8b-instruct",
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_type=ModelType.TEXT_ONLY,
    model_family=ModelFamily.LLAMA,
    hidden_size=4096,
    num_layers=32,
    num_attention_heads=32,
    vocab_size=128256,
    max_position_embeddings=131072,
    has_vision=False,
    expected_vram_gb=18.0,
    supports_flash_attention=True,
    recommended_batch_size=2,
    recommended_fusion_layers=[10, 20, 28],
    recommended_lora_rank=16,
    description="Llama 3.1 8B Instruct - Fine-tuned for instruction following",
    license="llama3.1"
))

# Llama 3.2 3B (Smaller, efficient)
register_model(ModelSpec(
    name="llama-3.2-3b",
    model_id="meta-llama/Llama-3.2-3B",
    model_type=ModelType.TEXT_ONLY,
    model_family=ModelFamily.LLAMA,
    hidden_size=3072,
    num_layers=28,
    num_attention_heads=24,
    vocab_size=128256,
    max_position_embeddings=131072,
    has_vision=False,
    expected_vram_gb=8.0,
    supports_flash_attention=True,
    recommended_batch_size=4,
    recommended_fusion_layers=[8, 16, 24],
    recommended_lora_rank=8,
    description="Llama 3.2 3B - Smaller efficient model",
    license="llama3.2"
))

# Llama 3.2 1B (Very small)
register_model(ModelSpec(
    name="llama-3.2-1b",
    model_id="meta-llama/Llama-3.2-1B",
    model_type=ModelType.TEXT_ONLY,
    model_family=ModelFamily.LLAMA,
    hidden_size=2048,
    num_layers=16,
    num_attention_heads=32,
    vocab_size=128256,
    max_position_embeddings=131072,
    has_vision=False,
    expected_vram_gb=4.0,
    supports_flash_attention=True,
    recommended_batch_size=8,
    recommended_fusion_layers=[4, 8, 12],
    recommended_lora_rank=8,
    description="Llama 3.2 1B - Tiny model for fast iteration",
    license="llama3.2"
))

# Llama 2 7B (Legacy)
register_model(ModelSpec(
    name="llama-2-7b",
    model_id="meta-llama/Llama-2-7b-hf",
    model_type=ModelType.TEXT_ONLY,
    model_family=ModelFamily.LLAMA,
    hidden_size=4096,
    num_layers=32,
    num_attention_heads=32,
    vocab_size=32000,
    max_position_embeddings=4096,
    has_vision=False,
    expected_vram_gb=16.0,
    recommended_batch_size=2,
    recommended_fusion_layers=[10, 20, 28],
    recommended_lora_rank=8,
    description="Llama 2 7B - Legacy model",
    license="llama2"
))

# ============================================================================
# Register Other Text-Only Models
# ============================================================================

# Mistral 7B v0.3
register_model(ModelSpec(
    name="mistral-7b-v0.3",
    model_id="mistralai/Mistral-7B-v0.3",
    model_type=ModelType.TEXT_ONLY,
    model_family=ModelFamily.MISTRAL,
    hidden_size=4096,
    num_layers=32,
    num_attention_heads=32,
    vocab_size=32768,
    max_position_embeddings=32768,
    has_vision=False,
    expected_vram_gb=16.0,
    supports_flash_attention=True,
    recommended_batch_size=2,
    recommended_fusion_layers=[10, 20, 28],
    recommended_lora_rank=8,
    description="Mistral 7B v0.3 - Strong open model with sliding window attention",
    license="apache-2.0"
))

# Qwen 2.5 7B
register_model(ModelSpec(
    name="qwen-2.5-7b",
    model_id="Qwen/Qwen2.5-7B",
    model_type=ModelType.TEXT_ONLY,
    model_family=ModelFamily.QWEN,
    hidden_size=3584,
    num_layers=28,
    num_attention_heads=28,
    vocab_size=152064,
    max_position_embeddings=131072,
    has_vision=False,
    expected_vram_gb=16.0,
    supports_flash_attention=True,
    recommended_batch_size=2,
    recommended_fusion_layers=[8, 16, 24],
    recommended_lora_rank=8,
    description="Qwen 2.5 7B - Strong multilingual capabilities",
    license="apache-2.0"
))

# Phi-3 Mini
register_model(ModelSpec(
    name="phi-3-mini-4k",
    model_id="microsoft/Phi-3-mini-4k-instruct",
    model_type=ModelType.TEXT_ONLY,
    model_family=ModelFamily.PHI,
    hidden_size=3072,
    num_layers=32,
    num_attention_heads=32,
    vocab_size=32064,
    max_position_embeddings=4096,
    has_vision=False,
    expected_vram_gb=8.0,
    supports_flash_attention=True,
    recommended_batch_size=4,
    recommended_fusion_layers=[10, 20, 28],
    recommended_lora_rank=8,
    description="Phi-3 Mini - Microsoft's efficient small model",
    license="mit"
))

# Gemma 2 9B
register_model(ModelSpec(
    name="gemma-2-9b",
    model_id="google/gemma-2-9b",
    model_type=ModelType.TEXT_ONLY,
    model_family=ModelFamily.GEMMA,
    hidden_size=3584,
    num_layers=42,
    num_attention_heads=16,
    vocab_size=256000,
    max_position_embeddings=8192,
    has_vision=False,
    expected_vram_gb=20.0,
    supports_flash_attention=True,
    recommended_batch_size=2,
    recommended_fusion_layers=[12, 26, 38],
    recommended_lora_rank=8,
    description="Gemma 2 9B - Google's open model",
    license="gemma"
))

# Gemma 2 2B
register_model(ModelSpec(
    name="gemma-2-2b",
    model_id="google/gemma-2-2b",
    model_type=ModelType.TEXT_ONLY,
    model_family=ModelFamily.GEMMA,
    hidden_size=2304,
    num_layers=26,
    num_attention_heads=8,
    vocab_size=256000,
    max_position_embeddings=8192,
    has_vision=False,
    expected_vram_gb=6.0,
    supports_flash_attention=True,
    recommended_batch_size=4,
    recommended_fusion_layers=[8, 16, 22],
    recommended_lora_rank=8,
    description="Gemma 2 2B - Small efficient model",
    license="gemma"
))

# ============================================================================
# Register API-Based Models
# ============================================================================

# Gemini 1.5 Flash
register_model(ModelSpec(
    name="gemini-1.5-flash",
    model_id="gemini-1.5-flash",
    model_type=ModelType.API,
    model_family=ModelFamily.GEMINI,
    hidden_size=0,  # Not applicable for API models
    num_layers=0,
    num_attention_heads=0,
    vocab_size=0,
    has_vision=True,  # Gemini Flash supports multimodal
    api_provider="google",
    api_env_var="GOOGLE_API_KEY",
    expected_vram_gb=0.0,  # API, no local VRAM
    description="Gemini 1.5 Flash - Fast, efficient Google API model",
    license="proprietary"
))

# Gemini 1.5 Pro
register_model(ModelSpec(
    name="gemini-1.5-pro",
    model_id="gemini-1.5-pro",
    model_type=ModelType.API,
    model_family=ModelFamily.GEMINI,
    hidden_size=0,
    num_layers=0,
    num_attention_heads=0,
    vocab_size=0,
    has_vision=True,
    api_provider="google",
    api_env_var="GOOGLE_API_KEY",
    expected_vram_gb=0.0,
    description="Gemini 1.5 Pro - Larger Google API model with 1M context",
    license="proprietary"
))

# GPT-4o
register_model(ModelSpec(
    name="gpt-4o",
    model_id="gpt-4o",
    model_type=ModelType.API,
    model_family=ModelFamily.GPT,
    hidden_size=0,
    num_layers=0,
    num_attention_heads=0,
    vocab_size=0,
    has_vision=True,
    api_provider="openai",
    api_env_var="OPENAI_API_KEY",
    expected_vram_gb=0.0,
    description="GPT-4o - OpenAI's multimodal flagship model",
    license="proprietary"
))

# GPT-4o Mini
register_model(ModelSpec(
    name="gpt-4o-mini",
    model_id="gpt-4o-mini",
    model_type=ModelType.API,
    model_family=ModelFamily.GPT,
    hidden_size=0,
    num_layers=0,
    num_attention_heads=0,
    vocab_size=0,
    has_vision=True,
    api_provider="openai",
    api_env_var="OPENAI_API_KEY",
    expected_vram_gb=0.0,
    description="GPT-4o Mini - Smaller, faster OpenAI model",
    license="proprietary"
))

# Claude 3.5 Sonnet
register_model(ModelSpec(
    name="claude-3.5-sonnet",
    model_id="claude-3-5-sonnet-20241022",
    model_type=ModelType.API,
    model_family=ModelFamily.CLAUDE,
    hidden_size=0,
    num_layers=0,
    num_attention_heads=0,
    vocab_size=0,
    has_vision=True,
    api_provider="anthropic",
    api_env_var="ANTHROPIC_API_KEY",
    expected_vram_gb=0.0,
    description="Claude 3.5 Sonnet - Anthropic's balanced model",
    license="proprietary"
))


# ============================================================================
# Utility Functions
# ============================================================================

def get_model_config_for_safe(model_name: str) -> Dict[str, Any]:
    """
    Generate a SAFE-compatible configuration for a registered model.

    Returns a dictionary that can be merged with other SAFE configs.
    """
    spec = get_model_spec(model_name)

    config = {
        "name": f"safe-{model_name}",
        "description": f"SAFE configuration for {spec.description}",
        "llm_model_name": spec.model_id,
        "llm_hidden_size": spec.hidden_size,
        "model_type": spec.model_type.value,
        "model_family": spec.model_family.value,
        "expected_vram_gb": spec.expected_vram_gb,
        "recommended_batch_size": spec.recommended_batch_size,
    }

    # Vision settings
    if spec.has_vision and spec.vision_encoder:
        config["vision_model_name"] = spec.vision_encoder
        config["vision_embed_dim"] = spec.vision_hidden_size

    # Fusion settings
    fusion_layers = spec.get_fusion_layer_indices()
    config["fusion_layer_indices"] = fusion_layers
    config["lora_rank"] = spec.recommended_lora_rank
    config["fusion_config"] = {
        "num_attention_heads": spec.num_attention_heads,
        "attention_dropout": 0.1,
        "modalities": {
            "audio": {
                "layer_indices": fusion_layers,
                "num_tokens": 8 if spec.hidden_size < 4096 else 16
            }
        }
    }

    # API settings
    if spec.model_type == ModelType.API:
        config["api_provider"] = spec.api_provider
        config["api_env_var"] = spec.api_env_var

    return config


def print_model_registry():
    """Print all registered models with their details."""
    print("=" * 80)
    print("SAFE Model Registry")
    print("=" * 80)

    # Group by type
    for model_type in ModelType:
        models = list_models(model_type=model_type)
        if not models:
            continue

        print(f"\n{model_type.value.upper()} MODELS:")
        print("-" * 40)

        for name in sorted(models):
            spec = MODEL_REGISTRY[name]
            vram_str = f"{spec.expected_vram_gb:.0f}GB" if spec.expected_vram_gb > 0 else "API"
            print(f"  {name:30s} | {vram_str:6s} | {spec.description[:40]}...")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_model_registry()
