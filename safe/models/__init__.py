"""
SAFE Models Package

Provides model abstractions for the SAFE framework:
- Vision-Language models (LLaVA, BLIP2)
- Text-only models (Llama, Mistral, Qwen, etc.)
- API-based models (Gemini, GPT-4, Claude)
"""

from .base_vl import BaseVLModel, _detect_model_type
from .safe_model import SAFEModel
from .audio_encoders import CLAPAudioEncoder, WhisperAudioEncoder, MultiModalAudioEncoder
from .projectors import AudioProjector, AdaptiveAudioProjector
from .fusion_adapter import LoRAFusionAdapter, MultiLayerFusionAdapter, GatedFusionAdapter
from .layer_hooks import LayerHookManager

# Multi-model support
from .model_registry import (
    ModelSpec,
    ModelType,
    ModelFamily,
    MODEL_REGISTRY,
    register_model,
    get_model_spec,
    list_models,
    get_model_config_for_safe,
    print_model_registry,
)

from .base_model import (
    BaseModel,
    TextOnlyModel,
    VisionLanguageModel,
    APIModel,
    create_base_model,
    get_model_hidden_size,
    get_model_num_layers,
)

__all__ = [
    # Original exports
    "BaseVLModel",
    "_detect_model_type",
    "SAFEModel",
    "CLAPAudioEncoder",
    "WhisperAudioEncoder",
    "MultiModalAudioEncoder",
    "AudioProjector",
    "AdaptiveAudioProjector",
    "LoRAFusionAdapter",
    "MultiLayerFusionAdapter",
    "GatedFusionAdapter",
    "LayerHookManager",

    # Model registry
    "ModelSpec",
    "ModelType",
    "ModelFamily",
    "MODEL_REGISTRY",
    "register_model",
    "get_model_spec",
    "list_models",
    "get_model_config_for_safe",
    "print_model_registry",

    # Base model abstractions
    "BaseModel",
    "TextOnlyModel",
    "VisionLanguageModel",
    "APIModel",
    "create_base_model",
    "get_model_hidden_size",
    "get_model_num_layers",
]
