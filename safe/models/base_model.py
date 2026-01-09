"""
Abstract Base Model for SAFE framework.
Provides a unified interface for different backbone types:
- Vision-Language models (Llava, BLIP2)
- Text-only models (Llama, Mistral, etc.)
- API-based models (Gemini, GPT-4)
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

from .model_registry import (
    ModelSpec,
    ModelType,
    ModelFamily,
    get_model_spec,
    MODEL_REGISTRY,
)


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all SAFE backbone models.

    Provides a unified interface for:
    - Token embedding
    - Forward pass
    - Generation
    - Model introspection
    """

    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
        device_dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self._device_dtype = device_dtype

        # Will be set by subclasses
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.config = None
        self.model_spec: Optional[ModelSpec] = None

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the hidden dimension of the model."""
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        pass

    @property
    @abstractmethod
    def num_attention_heads(self) -> int:
        """Return the number of attention heads."""
        pass

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        if self.tokenizer is not None:
            return len(self.tokenizer)
        return 0

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the model."""
        return next(self.parameters()).dtype

    @abstractmethod
    def get_input_embeddings(self) -> nn.Embedding:
        """Return the input embedding layer."""
        pass

    @abstractmethod
    def get_decoder_layers(self) -> nn.ModuleList:
        """Return the list of decoder/transformer layers for hook injection."""
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs
    ) -> torch.Tensor:
        """Generate tokens."""
        pass

    def configure_tokenizer(self):
        """Configure tokenizer with proper padding settings."""
        if self.tokenizer is None:
            return

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[BaseModel] Set pad_token to eos_token", flush=True)

        # Use left padding for decoder-only models
        self.tokenizer.padding_side = "left"
        print(f"[BaseModel] Set padding_side='left'", flush=True)

    def freeze_model(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        print(f"[BaseModel] Froze all parameters", flush=True)

    def get_layer_by_index(self, idx: int) -> nn.Module:
        """Get a specific decoder layer by index."""
        layers = self.get_decoder_layers()
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"Layer index {idx} out of range [0, {len(layers)})")
        return layers[idx]


class TextOnlyModel(BaseModel):
    """
    Base model for text-only LLMs (Llama, Mistral, etc.)
    These models don't have vision encoders.
    """

    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
        device_dtype: Optional[torch.dtype] = None,
        use_flash_attention: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ):
        super().__init__(model_name, freeze, device_dtype, **kwargs)

        self.use_flash_attention = use_flash_attention
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        # Try to get model spec from registry
        try:
            self.model_spec = get_model_spec(model_name)
            model_id = self.model_spec.model_id
        except ValueError:
            # Not in registry, use model_name as HuggingFace ID directly
            model_id = model_name
            self.model_spec = None

        print(f"[TextOnlyModel] Loading model: {model_id}", flush=True)

        # Determine dtype
        if device_dtype is None:
            device_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._device_dtype = device_dtype

        # Load config first
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        # Prepare loading kwargs
        load_kwargs = {
            "torch_dtype": device_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        # Flash attention
        if use_flash_attention and torch.cuda.is_available():
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print(f"[TextOnlyModel] Using Flash Attention 2", flush=True)

        # Quantization
        if load_in_8bit or load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_compute_dtype=device_dtype if load_in_4bit else None,
                )
                load_kwargs["quantization_config"] = quant_config
                print(f"[TextOnlyModel] Using {'4-bit' if load_in_4bit else '8-bit'} quantization", flush=True)
            except ImportError:
                print("[TextOnlyModel] bitsandbytes not available, skipping quantization", flush=True)

        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                use_safetensors=True,
                **load_kwargs
            )
        except Exception as e:
            print(f"[TextOnlyModel] Flash attention failed ({e}), falling back", flush=True)
            load_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                use_safetensors=True,
                **load_kwargs
            )

        print(f"[TextOnlyModel] Model loaded successfully", flush=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.configure_tokenizer()
        print(f"[TextOnlyModel] Tokenizer loaded", flush=True)

        # Freeze if requested
        if freeze:
            self.freeze_model()

        # Store model architecture info
        self._hidden_size = self.config.hidden_size
        self._num_layers = self.config.num_hidden_layers
        self._num_attention_heads = self.config.num_attention_heads

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_attention_heads(self) -> int:
        return self._num_attention_heads

    @property
    def has_vision(self) -> bool:
        return False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def get_decoder_layers(self) -> nn.ModuleList:
        """Get decoder layers - handles different model architectures."""
        model = self.model

        # Try common attribute names for the decoder stack
        layer_attrs = [
            "model.layers",          # Llama, Mistral, Qwen
            "transformer.h",         # GPT-2 style
            "gpt_neox.layers",       # GPT-NeoX
            "decoder.layers",        # Some encoder-decoder models
            "layers",                # Direct attribute
        ]

        for attr_path in layer_attrs:
            obj = model
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.ModuleList):
                    return obj
            except AttributeError:
                continue

        raise AttributeError(f"Could not find decoder layers in model {type(model)}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the text-only model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs
        )

        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
            "hidden_states": getattr(outputs, "hidden_states", None),
        }

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs
    ) -> torch.Tensor:
        """Generate tokens from the model."""
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }

        if inputs_embeds is not None:
            gen_kwargs["inputs_embeds"] = inputs_embeds
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
        else:
            gen_kwargs["input_ids"] = input_ids
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask

        return self.model.generate(**gen_kwargs)


class VisionLanguageModel(BaseModel):
    """
    Base model for vision-language models (Llava, BLIP2, etc.)
    """

    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
        device_dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        super().__init__(model_name, freeze, device_dtype, **kwargs)

        # Try to get model spec from registry
        try:
            self.model_spec = get_model_spec(model_name)
            model_id = self.model_spec.model_id
            model_family = self.model_spec.model_family
        except ValueError:
            # Not in registry, infer from name
            model_id = model_name
            if "llava" in model_name.lower():
                model_family = ModelFamily.LLAVA
            elif "blip2" in model_name.lower():
                model_family = ModelFamily.BLIP2
            else:
                model_family = ModelFamily.CUSTOM
            self.model_spec = None

        self.model_family = model_family
        print(f"[VisionLanguageModel] Loading model: {model_id} (family: {model_family.value})", flush=True)

        # Determine dtype
        if device_dtype is None:
            device_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._device_dtype = device_dtype

        # Load based on model family
        if model_family == ModelFamily.LLAVA:
            self._load_llava(model_id, device_dtype)
        elif model_family == ModelFamily.BLIP2:
            self._load_blip2(model_id, device_dtype)
        else:
            raise ValueError(f"Unsupported VL model family: {model_family}")

        # Configure tokenizer
        self.configure_tokenizer()

        # Freeze if requested
        if freeze:
            self.freeze_model()

    def _load_llava(self, model_id: str, device_dtype: torch.dtype):
        """Load LLaVA model."""
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=device_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.processor = LlavaProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.config = self.model.config

        # Get language model config
        if hasattr(self.config, "text_config"):
            lm_config = self.config.text_config
        else:
            lm_config = self.config

        self._hidden_size = lm_config.hidden_size
        self._num_layers = lm_config.num_hidden_layers
        self._num_attention_heads = lm_config.num_attention_heads

        print(f"[VisionLanguageModel] LLaVA loaded: hidden={self._hidden_size}, layers={self._num_layers}", flush=True)

    def _load_blip2(self, model_id: str, device_dtype: torch.dtype):
        """Load BLIP2 model."""
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=device_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.config = self.model.config

        # Get language model config
        if hasattr(self.config, "text_config"):
            lm_config = self.config.text_config
        else:
            lm_config = self.config

        self._hidden_size = lm_config.hidden_size
        self._num_layers = lm_config.num_hidden_layers
        self._num_attention_heads = lm_config.num_attention_heads

        print(f"[VisionLanguageModel] BLIP2 loaded: hidden={self._hidden_size}, layers={self._num_layers}", flush=True)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_attention_heads(self) -> int:
        return self._num_attention_heads

    @property
    def has_vision(self) -> bool:
        return True

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embeddings from the language model component."""
        if self.model_family == ModelFamily.LLAVA:
            return self.model.language_model.get_input_embeddings()
        elif self.model_family == ModelFamily.BLIP2:
            return self.model.language_model.get_input_embeddings()
        return self.model.get_input_embeddings()

    def get_decoder_layers(self) -> nn.ModuleList:
        """Get decoder layers from the language model component."""
        if self.model_family == ModelFamily.LLAVA:
            # LLaVA uses Llama-style architecture
            return self.model.language_model.model.layers
        elif self.model_family == ModelFamily.BLIP2:
            # BLIP2 uses OPT
            return self.model.language_model.model.decoder.layers
        raise AttributeError(f"Could not find decoder layers for {self.model_family}")

    def get_language_model(self) -> nn.Module:
        """Get the underlying language model component."""
        if hasattr(self.model, "language_model"):
            return self.model.language_model
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the VL model."""
        model_kwargs = {
            "attention_mask": attention_mask,
            "labels": labels,
            **kwargs
        }

        if inputs_embeds is not None:
            model_kwargs["inputs_embeds"] = inputs_embeds
        else:
            model_kwargs["input_ids"] = input_ids

        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values

        outputs = self.model(**model_kwargs)

        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
            "hidden_states": getattr(outputs, "hidden_states", None),
        }

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs
    ) -> torch.Tensor:
        """Generate tokens from the VL model."""
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }

        if inputs_embeds is not None:
            gen_kwargs["inputs_embeds"] = inputs_embeds
        else:
            gen_kwargs["input_ids"] = input_ids

        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values

        return self.model.generate(**gen_kwargs)


class APIModel(BaseModel):
    """
    Base model for API-based models (Gemini, GPT-4, Claude).
    These models run inference through API calls, not local weights.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        # Don't call nn.Module init for API models
        self.model_name = model_name
        self.freeze = True
        self._device_dtype = None

        # Get model spec from registry
        try:
            self.model_spec = get_model_spec(model_name)
            model_id = self.model_spec.model_id
            api_provider = self.model_spec.api_provider
            api_env_var = self.model_spec.api_env_var
        except ValueError:
            raise ValueError(f"API model '{model_name}' not found in registry")

        self.model_id = model_id
        self.api_provider = api_provider

        # Get API key
        if api_key is None:
            import os
            api_key = os.environ.get(api_env_var)
            if api_key is None:
                raise ValueError(f"API key not found. Set {api_env_var} environment variable.")

        self.api_key = api_key
        self._client = None

        print(f"[APIModel] Initialized {model_name} with {api_provider} API", flush=True)

    def _get_client(self):
        """Lazily initialize the API client."""
        if self._client is not None:
            return self._client

        if self.api_provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model_id)
            except ImportError:
                raise ImportError("Install google-generativeai: pip install google-generativeai")

        elif self.api_provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")

        elif self.api_provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")

        return self._client

    @property
    def hidden_size(self) -> int:
        return 0  # Not applicable

    @property
    def num_layers(self) -> int:
        return 0  # Not applicable

    @property
    def num_attention_heads(self) -> int:
        return 0  # Not applicable

    @property
    def has_vision(self) -> bool:
        return self.model_spec.has_vision if self.model_spec else False

    def get_input_embeddings(self) -> nn.Embedding:
        raise NotImplementedError("API models don't have local embeddings")

    def get_decoder_layers(self) -> nn.ModuleList:
        raise NotImplementedError("API models don't have local layers")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("API models use generate() method only")

    def generate(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        audio: Optional[Any] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using the API."""
        client = self._get_client()

        if self.api_provider == "google":
            return self._generate_google(client, prompt, images, max_tokens, temperature)
        elif self.api_provider == "openai":
            return self._generate_openai(client, prompt, images, max_tokens, temperature)
        elif self.api_provider == "anthropic":
            return self._generate_anthropic(client, prompt, images, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown API provider: {self.api_provider}")

    def _generate_google(self, client, prompt, images, max_tokens, temperature):
        """Generate using Google Gemini API."""
        content = [prompt]
        if images:
            # Handle image inputs for multimodal Gemini
            for img in images:
                if hasattr(img, 'read'):  # File-like
                    content.append({"mime_type": "image/jpeg", "data": img.read()})
                elif isinstance(img, str):  # Path
                    with open(img, 'rb') as f:
                        content.append({"mime_type": "image/jpeg", "data": f.read()})

        response = client.generate_content(
            content,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return response.text

    def _generate_openai(self, client, prompt, images, max_tokens, temperature):
        """Generate using OpenAI API."""
        messages = [{"role": "user", "content": prompt}]

        if images and self.has_vision:
            # Build multimodal message
            content = [{"type": "text", "text": prompt}]
            for img in images:
                if isinstance(img, str):
                    # Assume URL or base64
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img}
                    })
            messages = [{"role": "user", "content": content}]

        response = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def _generate_anthropic(self, client, prompt, images, max_tokens, temperature):
        """Generate using Anthropic API."""
        content = [{"type": "text", "text": prompt}]

        if images and self.has_vision:
            import base64
            for img in images:
                if isinstance(img, str):
                    with open(img, 'rb') as f:
                        img_data = base64.standard_b64encode(f.read()).decode("utf-8")
                    content.insert(0, {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_data,
                        }
                    })

        response = client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text

    def parameters(self):
        """Yield no parameters (API model has no local parameters)."""
        return iter([])


# ============================================================================
# Factory Functions
# ============================================================================

def create_base_model(
    model_name: str,
    freeze: bool = True,
    device_dtype: Optional[torch.dtype] = None,
    **kwargs
) -> BaseModel:
    """
    Factory function to create the appropriate model type.

    Args:
        model_name: Name of the model (from registry) or HuggingFace model ID
        freeze: Whether to freeze model weights
        device_dtype: Data type for model weights
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Appropriate BaseModel subclass instance
    """
    # Check if model is in registry
    try:
        spec = get_model_spec(model_name)
        model_type = spec.model_type
    except ValueError:
        # Not in registry, infer type from name
        name_lower = model_name.lower()
        if any(vl in name_lower for vl in ["llava", "blip2", "blip-2"]):
            model_type = ModelType.VISION_LANGUAGE
        elif any(api in name_lower for api in ["gpt-4", "gemini", "claude"]):
            model_type = ModelType.API
        else:
            model_type = ModelType.TEXT_ONLY

    # Create appropriate model
    if model_type == ModelType.VISION_LANGUAGE:
        return VisionLanguageModel(model_name, freeze=freeze, device_dtype=device_dtype, **kwargs)
    elif model_type == ModelType.TEXT_ONLY:
        return TextOnlyModel(model_name, freeze=freeze, device_dtype=device_dtype, **kwargs)
    elif model_type == ModelType.API:
        return APIModel(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_hidden_size(model_name: str) -> int:
    """Get the hidden size for a model without loading it."""
    try:
        spec = get_model_spec(model_name)
        return spec.hidden_size
    except ValueError:
        # Load config to get hidden size
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return config.hidden_size


def get_model_num_layers(model_name: str) -> int:
    """Get the number of layers for a model without loading it."""
    try:
        spec = get_model_spec(model_name)
        return spec.num_layers
    except ValueError:
        # Load config to get layers
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return config.num_hidden_layers
