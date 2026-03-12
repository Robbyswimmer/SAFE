import torch
import torch.nn as nn
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    AutoModelForImageTextToText,
    AutoImageProcessor,
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    AutoProcessor
)
from typing import Optional, Dict, Any, Tuple
import os


class BaseVLModel(nn.Module):
    """
    Base Vision-Language model following LLaVA-style architecture.
    
    Components:
    - Frozen CLIP vision encoder
    - Vision projector (trainable)
    - Frozen LLM backbone
    """
    
    def __init__(
        self,
        llm_model_name: str = "microsoft/DialoGPT-medium",
        vision_model_name: str = "openai/clip-vit-large-patch14",
        vision_hidden_size: int = 1024,
        llm_hidden_size: int = 1024,
        num_vision_tokens: int = 256,
        freeze_vision: bool = True,
        freeze_llm: bool = True,
        enable_gradient_checkpointing: Optional[bool] = None,
        prefer_flash_attention_2: bool = True,
        qwen_quantization: str = "auto",  # "auto" | "4bit" | "8bit" | "none"
    ):
        super().__init__()
        
        self.llm_model_name = llm_model_name
        self.vision_model_name = vision_model_name
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.num_vision_tokens = num_vision_tokens
        self._input_require_grads_hook = None

        # Optional multi-GPU sharding controls (HF accelerate device_map path).
        # Env examples:
        #   SAFE_DEVICE_MAP=auto
        #   SAFE_MAX_MEMORY=0=46GiB,1=46GiB,2=46GiB,cpu=120GiB
        #   SAFE_OFFLOAD_FOLDER=/path/to/offload
        env_device_map = os.environ.get("SAFE_DEVICE_MAP", "").strip()
        env_max_memory = os.environ.get("SAFE_MAX_MEMORY", "").strip()
        env_offload_folder = os.environ.get("SAFE_OFFLOAD_FOLDER", "").strip()
        env_multi_gpu = os.environ.get("SAFE_MULTI_GPU", "").strip().lower()

        resolved_device_map = self._resolve_device_map_spec(env_device_map)
        if (
            resolved_device_map is None
            and env_multi_gpu in {"1", "true", "yes", "on"}
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
        ):
            resolved_device_map = "auto"

        resolved_max_memory = self._parse_max_memory_spec(env_max_memory)
        load_device_kwargs: Dict[str, Any] = {}
        if resolved_device_map is not None:
            load_device_kwargs["device_map"] = resolved_device_map
        if resolved_max_memory:
            load_device_kwargs["max_memory"] = resolved_max_memory
        if env_offload_folder:
            os.makedirs(env_offload_folder, exist_ok=True)
            load_device_kwargs["offload_folder"] = env_offload_folder
            load_device_kwargs["offload_state_dict"] = True

        if load_device_kwargs:
            print(
                "[BaseVL] Sharded load enabled: "
                f"device_map={load_device_kwargs.get('device_map')} "
                f"max_memory={load_device_kwargs.get('max_memory')} "
                f"offload_folder={load_device_kwargs.get('offload_folder', None)}",
                flush=True,
            )

        # Load vision encoder (frozen) - skip if None (e.g., audio-only Qwen)
        # or "built-in" (e.g., InternVL with integrated InternViT)
        import sys
        if vision_model_name and vision_model_name != "built-in":
            print(f"[BaseVL] Loading vision encoder: {vision_model_name}...", flush=True)
            sys.stdout.flush()
            self.vision_encoder = CLIPVisionModel.from_pretrained(
                vision_model_name,
                use_safetensors=True
            )
            print(f"[BaseVL] ✓ Vision encoder loaded", flush=True)
            sys.stdout.flush()
            print(f"[BaseVL] Loading image processor: {vision_model_name}...", flush=True)
            sys.stdout.flush()
            self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
            print(f"[BaseVL] ✓ Image processor loaded", flush=True)
            sys.stdout.flush()

            if freeze_vision:
                for param in self.vision_encoder.parameters():
                    param.requires_grad = False
        else:
            if vision_model_name == "built-in":
                print(f"[BaseVL] Skipping separate vision encoder (built-in to main model)", flush=True)
            else:
                print(f"[BaseVL] Skipping vision encoder (audio-only mode)", flush=True)
            sys.stdout.flush()
            self.vision_encoder = None
            self.image_processor = None
        
        # Determine appropriate dtype based on device availability
        # Use float16 for GPU, float32 for CPU to avoid LayerNorm issues
        device_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load LLM (frozen) - handle different VL models
        print(f"[BaseVL] Loading LLM: {llm_model_name}...", flush=True)
        sys.stdout.flush()
        if "llava" in llm_model_name.lower():
            print(f"[BaseVL] Detected LLaVA model type", flush=True)
            sys.stdout.flush()
            self.llm = LlavaForConditionalGeneration.from_pretrained(
                llm_model_name,
                torch_dtype=device_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                **load_device_kwargs,
            )
            print(f"[BaseVL] ✓ LLM model loaded", flush=True)
            sys.stdout.flush()
            self.processor = LlavaProcessor.from_pretrained(llm_model_name)
            self.tokenizer = self.processor.tokenizer
            self.model_type = "llava"
        elif "blip2" in llm_model_name.lower():
            print(f"[BaseVL] Detected BLIP2 model type", flush=True)
            sys.stdout.flush()
            self.llm = Blip2ForConditionalGeneration.from_pretrained(
                llm_model_name,
                torch_dtype=device_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                **load_device_kwargs,
            )
            print(f"[BaseVL] ✓ LLM model loaded", flush=True)
            sys.stdout.flush()
            self.processor = Blip2Processor.from_pretrained(llm_model_name)
            self.tokenizer = self.processor.tokenizer
            self.model_type = "blip2"
        elif "qwen" in llm_model_name.lower():
            print(f"[BaseVL] Detected Qwen model type", flush=True)
            sys.stdout.flush()
            # Qwen: prefer bf16 and optionally flash-attn + (4/8)-bit quantization to fit memory.
            qwen_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            # Env overrides (avoid plumbing flags through every script/config)
            env_quant = os.environ.get("SAFE_QWEN_QUANT", "").strip().lower()
            if env_quant:
                qwen_quantization = env_quant
            env_ckpt = os.environ.get("SAFE_GRAD_CKPT", "").strip()
            if env_ckpt:
                enable_gradient_checkpointing = env_ckpt not in ["0", "false", "False", "no", "NO"]
            if enable_gradient_checkpointing is None:
                enable_gradient_checkpointing = torch.cuda.is_available()
            env_flash = os.environ.get("SAFE_PREFER_FLASH2", "").strip()
            if env_flash:
                prefer_flash_attention_2 = env_flash not in ["0", "false", "False", "no", "NO"]

            # Allow env override for attention impl
            attn_impl = None
            if prefer_flash_attention_2 and torch.cuda.is_available():
                attn_impl = os.environ.get("SAFE_ATTN_IMPL", "flash_attention_2")

            def _try_load(quant_cfg: Optional[Any], torch_dtype: Optional[torch.dtype], attn_implementation: Optional[str]):
                kwargs: Dict[str, Any] = {
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                }
                kwargs.update(load_device_kwargs)
                if quant_cfg is not None:
                    kwargs["quantization_config"] = quant_cfg
                if torch_dtype is not None:
                    kwargs["torch_dtype"] = torch_dtype
                if attn_implementation is not None:
                    kwargs["attn_implementation"] = attn_implementation
                return AutoModelForCausalLM.from_pretrained(llm_model_name, **kwargs)

            quant_mode = (qwen_quantization or "auto").lower()
            tried: list = []
            llm = None

            quant_cfgs: list = []
            try:
                from transformers import BitsAndBytesConfig
                if quant_mode in ["auto", "4bit"]:
                    quant_cfgs.append(("4bit", BitsAndBytesConfig(load_in_4bit=True)))
                if quant_mode in ["auto", "8bit"]:
                    quant_cfgs.append(("8bit", BitsAndBytesConfig(load_in_8bit=True)))
            except Exception:
                quant_cfgs = []

            # Try: flash2+quant, quant, flash2+bf16, bf16
            attempts: list = []
            for name, cfg in quant_cfgs:
                attempts.append((f"{name}+{attn_impl or 'noattn'}", cfg, None, attn_impl))
                attempts.append((name, cfg, None, None))
            attempts.append((f"bf16+{attn_impl or 'noattn'}", None, qwen_dtype, attn_impl))
            attempts.append(("bf16", None, qwen_dtype, None))
            # Fallback for environments where bf16 path fails (driver/accelerate/runtime).
            if torch.cuda.is_available():
                attempts.append((f"fp16+{attn_impl or 'noattn'}", None, torch.float16, attn_impl))
                attempts.append(("fp16", None, torch.float16, None))

            for name, cfg, td, ai in attempts:
                try:
                    print(f"[BaseVL] Qwen load attempt: {name}", flush=True)
                    llm = _try_load(cfg, td, ai)
                    break
                except Exception as e:
                    tried.append(f"{name}: {type(e).__name__}")
                    continue

            if llm is None:
                raise RuntimeError(f"[BaseVL] Failed to load Qwen after attempts: {tried}")

            self.llm = llm

            # Strong defaults for adapter training on large Qwen:
            # - disable KV cache (saves memory, required for checkpointing)
            # - enable hidden states (probe training)
            try:
                self.llm.config.use_cache = False
            except Exception:
                pass
            try:
                self.llm.config.output_hidden_states = True
            except Exception:
                pass

            # Enable gradient checkpointing if requested
            if enable_gradient_checkpointing:
                try:
                    self.llm.gradient_checkpointing_enable()
                    try:
                        self.llm.config.use_cache = False
                    except Exception:
                        pass
                    self._enable_input_require_grads("Qwen")
                    print(f"[BaseVL] Qwen gradient checkpointing enabled", flush=True)
                except Exception as e:
                    print(f"[BaseVL] Warning: could not enable gradient checkpointing: {e}", flush=True)

            print(f"[BaseVL] ✓ LLM model loaded", flush=True)
            sys.stdout.flush()

            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
            print(f"[BaseVL] ✓ Tokenizer loaded", flush=True)
            sys.stdout.flush()
            self.model_type = "qwen"
        elif "internvl" in llm_model_name.lower():
            print(f"[BaseVL] Detected InternVL model type", flush=True)
            sys.stdout.flush()
            # InternVL: prefer built-in InternVLForConditionalGeneration via
            # AutoModelForImageTextToText. This class natively handles
            # input_ids + pixel_values + labels and has built-in weight key
            # conversion (_checkpoint_conversion_mapping).
            # Fallback: AutoModel with trust_remote_code for older weights.
            internvl_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            # Env overrides (same as Qwen for consistency)
            env_quant = os.environ.get("SAFE_QWEN_QUANT", "").strip().lower()
            if env_quant:
                qwen_quantization = env_quant
            env_ckpt = os.environ.get("SAFE_GRAD_CKPT", "").strip()
            if env_ckpt:
                enable_gradient_checkpointing = env_ckpt not in ["0", "false", "False", "no", "NO"]
            if enable_gradient_checkpointing is None:
                enable_gradient_checkpointing = torch.cuda.is_available()
            env_flash = os.environ.get("SAFE_PREFER_FLASH2", "").strip()
            if env_flash:
                prefer_flash_attention_2 = env_flash not in ["0", "false", "False", "no", "NO"]

            attn_impl = None
            if prefer_flash_attention_2 and torch.cuda.is_available():
                attn_impl = os.environ.get("SAFE_ATTN_IMPL", "flash_attention_2")

            def _try_load_internvl_builtin(quant_cfg, torch_dtype, attn_implementation):
                """Load via built-in InternVLForConditionalGeneration."""
                kwargs = {
                    "low_cpu_mem_usage": True,
                }
                kwargs.update(load_device_kwargs)
                if quant_cfg is not None:
                    kwargs["quantization_config"] = quant_cfg
                if torch_dtype is not None:
                    kwargs["torch_dtype"] = torch_dtype
                if attn_implementation is not None:
                    kwargs["attn_implementation"] = attn_implementation
                return AutoModelForImageTextToText.from_pretrained(llm_model_name, **kwargs)

            def _try_load_internvl_custom(quant_cfg, torch_dtype, attn_implementation):
                """Fallback: load via AutoModel with trust_remote_code (legacy path)."""
                kwargs = {
                    # The custom InternVL path is more brittle under meta-tensor
                    # initialization on this cluster/transformers combination.
                    # Use the safer non-meta load path for eval/inference.
                    "low_cpu_mem_usage": False,
                    "trust_remote_code": True,
                }
                # Do not pass sharded/device_map kwargs through the legacy custom
                # path; in this environment they can force meta-tensor loading and
                # break trust_remote_code models during eval.
                if quant_cfg is not None:
                    kwargs["quantization_config"] = quant_cfg
                if torch_dtype is not None:
                    kwargs["torch_dtype"] = torch_dtype
                if attn_implementation is not None:
                    kwargs["attn_implementation"] = attn_implementation
                return AutoModel.from_pretrained(llm_model_name, **kwargs)

            quant_mode = (qwen_quantization or "auto").lower()
            tried = []
            llm = None

            quant_cfgs = []
            try:
                from transformers import BitsAndBytesConfig
                if quant_mode in ["auto", "4bit"]:
                    quant_cfgs.append(("4bit", BitsAndBytesConfig(load_in_4bit=True)))
                if quant_mode in ["auto", "8bit"]:
                    quant_cfgs.append(("8bit", BitsAndBytesConfig(load_in_8bit=True)))
            except Exception:
                quant_cfgs = []

            # Build attempt list: flash2+quant, quant, flash2+bf16, bf16
            attempt_params = []
            for name, cfg in quant_cfgs:
                attempt_params.append((f"{name}+{attn_impl or 'noattn'}", cfg, None, attn_impl))
                attempt_params.append((name, cfg, None, None))
            attempt_params.append((f"bf16+{attn_impl or 'noattn'}", None, internvl_dtype, attn_impl))
            attempt_params.append(("bf16", None, internvl_dtype, None))

            # Try built-in classes first, then fallback to trust_remote_code
            for loader_name, loader_fn in [
                ("built-in", _try_load_internvl_builtin),
                ("custom(trust_remote_code)", _try_load_internvl_custom),
            ]:
                if llm is not None:
                    break
                for name, cfg, td, ai in attempt_params:
                    try:
                        label = f"{loader_name}/{name}"
                        print(f"[BaseVL] InternVL load attempt: {label}", flush=True)
                        llm = loader_fn(cfg, td, ai)
                        print(f"[BaseVL] InternVL loaded via {label}", flush=True)
                        break
                    except Exception as e:
                        tried.append(f"{loader_name}/{name}: {type(e).__name__}: {e}")
                        continue

            if llm is None:
                raise RuntimeError(f"[BaseVL] Failed to load InternVL after attempts: {tried}")

            self.llm = llm

            # Verify vision tower accessibility (built-in classes expose get_image_features)
            if hasattr(self.llm, 'get_image_features'):
                print(f"[BaseVL] ✓ InternVL vision tower accessible (get_image_features)", flush=True)
            else:
                print(f"[BaseVL] InternVL loaded without built-in vision pipeline "
                      f"(type: {type(self.llm).__name__})", flush=True)

            # Disable KV cache and enable hidden states for adapter training
            try:
                self.llm.config.use_cache = False
            except Exception:
                pass
            try:
                self.llm.config.output_hidden_states = True
            except Exception:
                pass

            # Enable gradient checkpointing if requested
            if enable_gradient_checkpointing:
                try:
                    self.llm.gradient_checkpointing_enable()
                    try:
                        self.llm.config.use_cache = False
                    except Exception:
                        pass
                    self._enable_input_require_grads("InternVL")
                    print(f"[BaseVL] InternVL gradient checkpointing enabled", flush=True)
                except Exception as e:
                    print(f"[BaseVL] Warning: could not enable gradient checkpointing: {e}", flush=True)

            print(f"[BaseVL] ✓ LLM model loaded", flush=True)
            sys.stdout.flush()

            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
            print(f"[BaseVL] ✓ Tokenizer loaded", flush=True)
            sys.stdout.flush()

            # Set img_context_token_id for custom InternVL generate().
            # The custom model's generate() asserts this is not None.
            # Use image_token_id from config (151667), or look up <IMG_CONTEXT> in tokenizer.
            if hasattr(self.llm, 'img_context_token_id') and self.llm.img_context_token_id is None:
                img_token_id = getattr(self.llm.config, 'image_token_id', None)
                if img_token_id is None:
                    # Try tokenizer lookup
                    try:
                        img_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
                        if img_token_id == self.tokenizer.unk_token_id:
                            img_token_id = None
                    except Exception:
                        pass
                if img_token_id is not None:
                    self.llm.img_context_token_id = img_token_id
                    print(f"[BaseVL] ✓ Set img_context_token_id={img_token_id}", flush=True)
                else:
                    print(f"[BaseVL] Warning: could not determine img_context_token_id", flush=True)

            # Load InternVL's image processor via AutoImageProcessor (avoids tokenizer
            # dependency that causes "Qwen2TokenizerFast has no attribute
            # start_image_token" when using AutoProcessor).
            try:
                self.internvl_image_processor = AutoImageProcessor.from_pretrained(
                    llm_model_name
                )
                print(f"[BaseVL] ✓ InternVL image processor loaded (AutoImageProcessor)", flush=True)
            except Exception as e:
                print(f"[BaseVL] Warning: AutoImageProcessor failed: {e}", flush=True)
                # Fallback: try AutoProcessor with trust_remote_code
                try:
                    self.internvl_image_processor = AutoProcessor.from_pretrained(
                        llm_model_name, trust_remote_code=True
                    )
                    print(f"[BaseVL] ✓ InternVL image processor loaded (AutoProcessor fallback)", flush=True)
                except Exception as e2:
                    print(f"[BaseVL] Warning: could not load InternVL image processor: {e2}", flush=True)
                    self.internvl_image_processor = None

            self.model_type = "internvl"
        else:
            print(f"[BaseVL] Using AutoModel for custom LLM", flush=True)
            sys.stdout.flush()
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                use_safetensors=True,
                **load_device_kwargs,
            )
            print(f"[BaseVL] ✓ LLM model loaded", flush=True)
            sys.stdout.flush()
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            print(f"[BaseVL] ✓ Tokenizer loaded", flush=True)
            sys.stdout.flush()
            self.model_type = "custom"
        print(f"[BaseVL] ✓ All LLM components loaded (type: {self.model_type})", flush=True)
        sys.stdout.flush()
        
        # Configure all tokenizers comprehensively
        self._configure_tokenizers()
            
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
                
        # Vision projector - only needed for custom models (not LLaVA, BLIP2, or Qwen)
        if self.model_type == "custom":
            vision_output_dim = self.vision_encoder.config.hidden_size
            self.vision_projector = nn.Sequential(
                nn.Linear(vision_output_dim, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
            # Always freeze vision projector parameters
            for param in self.vision_projector.parameters():
                param.requires_grad = False
        else:
            # LLaVA, BLIP2, Qwen: no vision projector needed
            # (LLaVA/BLIP2 have built-in vision; Qwen is audio-only in SAFE)
            self.vision_projector = None
        
        # Special tokens - only for custom models with vision
        if self.model_type == "custom":
            self.vision_start_token = "<img>"
            self.vision_end_token = "</img>"

            # Add special tokens to tokenizer
            special_tokens = [self.vision_start_token, self.vision_end_token]
            self.tokenizer.add_tokens(special_tokens)
            self.llm.resize_token_embeddings(len(self.tokenizer))
        else:
            # LLaVA, BLIP2, Qwen: no special vision tokens needed
            # (Qwen uses audio fusion via SAFE, not vision tokens)
            self.vision_start_token = None
            self.vision_end_token = None

    @staticmethod
    def _resolve_device_map_spec(spec: str):
        if not spec:
            return None
        s = spec.strip()
        lower = s.lower()
        if lower in {"none", "off", "false", "0"}:
            return None
        if lower in {"auto", "balanced", "balanced_low_0", "sequential"}:
            return lower
        if s.isdigit():
            return {"": int(s)}
        if lower.startswith("cuda:") or lower in {"cpu", "disk", "mps"}:
            return {"": s}
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return s

    def _enable_input_require_grads(self, model_label: str) -> None:
        """
        Ensure at least one forward input requires grad when gradient checkpointing is on.

        This is required for frozen-backbone adapter training, especially when models are
        called with input_ids (no inputs_embeds path), as in InternVL vision+text forward.
        """
        try:
            if hasattr(self.llm, "enable_input_require_grads"):
                self.llm.enable_input_require_grads()
                print(f"[BaseVL] {model_label} input-require-grads enabled", flush=True)
                return
        except Exception as e:
            print(
                f"[BaseVL] Warning: {model_label} enable_input_require_grads() failed: {e}",
                flush=True,
            )

        try:
            emb = self.llm.get_input_embeddings()
            if emb is None:
                return

            if self._input_require_grads_hook is None:
                def _make_output_require_grad(_module, _inputs, output):
                    if torch.is_tensor(output) and not output.requires_grad:
                        output.requires_grad_(True)

                self._input_require_grads_hook = emb.register_forward_hook(_make_output_require_grad)
                print(
                    f"[BaseVL] {model_label} input-require-grads hook registered",
                    flush=True,
                )
        except Exception as e:
            print(
                f"[BaseVL] Warning: could not register {model_label} input-require-grads hook: {e}",
                flush=True,
            )

    @staticmethod
    def _parse_max_memory_spec(spec: str) -> Optional[Dict[Any, str]]:
        if not spec:
            return None
        out: Dict[Any, str] = {}
        max_cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for raw_item in spec.split(","):
            item = raw_item.strip()
            if not item:
                continue
            if "=" not in item:
                continue
            key_raw, val = item.split("=", 1)
            key = key_raw.strip()
            val = val.strip()
            if not key or not val:
                continue
            lower = key.lower()
            parsed_key: Any
            if key.isdigit():
                parsed_key = int(key)
            elif lower.startswith("cuda:") and key.split(":", 1)[1].isdigit():
                parsed_key = int(key.split(":", 1)[1])
            elif lower in {"cpu", "disk"}:
                parsed_key = lower
            else:
                parsed_key = key
            # Guard against invalid GPU ids in max_memory, which can happen
            # when scripts assume 3 GPUs but Slurm allocated only 1.
            if isinstance(parsed_key, int):
                if parsed_key < 0:
                    continue
                if parsed_key >= max_cuda_devices:
                    print(
                        f"[BaseVL] Ignoring max_memory entry for unavailable cuda:{parsed_key} "
                        f"(visible_gpus={max_cuda_devices})",
                        flush=True,
                    )
                    continue
            out[parsed_key] = val
        return out or None
    
    def _set_padding_side_left(self, tokenizer, context: str) -> bool:
        """Utility to set padding_side to left if needed."""
        if tokenizer is None:
            return False

        if getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
            print(f"[BaseVL] Set {context} padding_side='left'", flush=True)
            return True
        return False

    def ensure_left_padding(self) -> bool:
        """Ensure all tokenizers use left padding when required."""
        is_decoder_only = not getattr(getattr(self.llm, "config", {}), "is_encoder_decoder", False)
        if not is_decoder_only:
            return False

        changed = False

        # Main tokenizer
        changed |= self._set_padding_side_left(self.tokenizer, "main tokenizer")

        # Processor tokenizer and potential nested tokenizers
        if hasattr(self, "processor"):
            processor_tokenizer = getattr(self.processor, "tokenizer", None)
            if processor_tokenizer is self.tokenizer:
                if processor_tokenizer is not None:
                    pass  # Processor tokenizer is same instance as main tokenizer
            else:
                if processor_tokenizer is not None and getattr(processor_tokenizer, "pad_token", None) is None:
                    processor_tokenizer.pad_token = processor_tokenizer.eos_token
                    print("[BaseVL] Set pad_token for processor tokenizer", flush=True)
                changed |= self._set_padding_side_left(processor_tokenizer, "processor tokenizer")

            # Nested processor tokenizers (image/text processors)
            nested_tokenizers = []
            if hasattr(self.processor, "image_processor"):
                nested_tokenizers.append((getattr(self.processor.image_processor, "tokenizer", None), "image processor tokenizer"))
            if hasattr(self.processor, "text_processor"):
                nested_tokenizers.append((getattr(self.processor.text_processor, "tokenizer", None), "text processor tokenizer"))

            for nested_tok, context in nested_tokenizers:
                changed |= self._set_padding_side_left(nested_tok, context)

            if hasattr(self.processor, "padding_side") and getattr(self.processor, "padding_side", None) != "left":
                self.processor.padding_side = "left"
                print("[BaseVL] Set processor padding_side='left'", flush=True)
                changed = True

        return changed

    def _configure_tokenizers(self):
        """Configure all tokenizer instances consistently."""
        print(f"[BaseVL] Configuring tokenizers for {self.model_type}...", flush=True)

        # 1. Configure main tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[BaseVL] Set pad_token to eos_token for main tokenizer", flush=True)

        # 2. Ensure left padding when required
        self.ensure_left_padding()
        if hasattr(self.llm, "config"):
            self.llm.config.pad_token_id = self.tokenizer.pad_token_id
            self.llm.config.eos_token_id = self.tokenizer.eos_token_id
        if hasattr(self.llm, "generation_config"):
            self.llm.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.llm.generation_config.eos_token_id = self.tokenizer.eos_token_id

        # 3. Verify configuration
        self._verify_tokenizer_config()
    
    def _verify_tokenizer_config(self):
        """Verify tokenizer configuration is correct."""
        print(f"[TokenizerVerify] Main tokenizer padding_side: {getattr(self.tokenizer, 'padding_side', 'NOT_SET')}", flush=True)
        print(f"[TokenizerVerify] Main tokenizer pad_token: {self.tokenizer.pad_token}", flush=True)
        
        if hasattr(self, 'processor') and hasattr(self.processor, 'tokenizer'):
            proc_tok = self.processor.tokenizer
            if proc_tok is not self.tokenizer:
                print(f"[TokenizerVerify] Processor tokenizer padding_side: {getattr(proc_tok, 'padding_side', 'NOT_SET')}", flush=True)
                print(f"[TokenizerVerify] Processor tokenizer pad_token: {proc_tok.pad_token}", flush=True)
        
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using CLIP vision encoder and project to LLM space.
        
        Args:
            images: (batch_size, 3, H, W) tensor of images
            
        Returns:
            vision_features: (batch_size, num_vision_tokens, llm_hidden_size)
        """
        if self.vision_encoder is None:
            # InternVL/Qwen: vision encoder not loaded separately (built-in or audio-only)
            return None
        if self.model_type in ["llava", "blip2", "internvl"]:
            # For LLaVA/BLIP2/InternVL, we'll let the model handle vision encoding internally
            # This method is mainly for compatibility
            with torch.no_grad():
                vision_outputs = self.vision_encoder(pixel_values=images)
                vision_features = vision_outputs.last_hidden_state  # (B, seq_len, vision_hidden_size)
            return vision_features
        else:
            with torch.no_grad():
                vision_outputs = self.vision_encoder(pixel_values=images)
                vision_features = vision_outputs.last_hidden_state  # (B, seq_len, vision_hidden_size)
            
            # Project to LLM space
            vision_features = self.vision_projector(vision_features)  # (B, seq_len, llm_hidden_size)
            
            return vision_features
    
    def prepare_inputs_for_training(
        self,
        text: str,
        images: Optional[torch.Tensor] = None,
        device: str = "cuda"
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for training by combining text and vision tokens.
        
        Args:
            text: Input text string
            images: Optional image tensor (batch_size, 3, H, W) or PIL Images
            device: Target device
            
        Returns:
            Dictionary with input_ids, attention_mask, labels, pixel_values
        """
        # Re-assert tokenizer padding configuration before tokenization
        if self.ensure_left_padding():
            print("[PrepareInputs] Re-applied left padding configuration", flush=True)

        # Check if we have valid images (not None and not a list of all None values)
        has_valid_images = False
        if images is not None:
            if isinstance(images, list):
                has_valid_images = any(img is not None for img in images)
            else:
                has_valid_images = True
        
        if self.model_type in ["llava", "blip2", "internvl"] and has_valid_images:
            # For BLIP-2, we need to handle tokenization more carefully
            if self.model_type == "blip2":
                # BLIP-2 expects text-only tokenization + separate pixel_values
                # Don't let the processor create excessive image tokens
                print(f"[PrepareInputs] Using tokenizer directly, padding_side: {getattr(self.tokenizer, 'padding_side', 'NOT_SET')}", flush=True)
                text_inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                inputs = {
                    "input_ids": text_inputs["input_ids"],
                    "attention_mask": text_inputs["attention_mask"]
                }
                
                # Process images only if we have valid images
                valid_images = images
                if isinstance(images, list):
                    valid_images = [img for img in images if img is not None]
                
                if valid_images:  # Only process if we have valid images after filtering
                    # Process images separately to get pixel_values
                    if hasattr(self.processor, 'image_processor'):
                        image_inputs = self.processor.image_processor(
                            valid_images,
                            return_tensors="pt"
                        )
                        inputs["pixel_values"] = image_inputs["pixel_values"]
                    else:
                        # Fallback: process images with the full processor but ignore input_ids
                        temp_inputs = self.processor(
                            images=valid_images,
                            return_tensors="pt"
                        )
                        inputs["pixel_values"] = temp_inputs["pixel_values"]
            else:
                # LLaVA: Use processor normally (it handles multimodal correctly)
                print(f"[PrepareInputs] Using processor, processor.tokenizer padding_side: {getattr(self.processor.tokenizer, 'padding_side', 'NOT_SET')}", flush=True)
                inputs = self.processor(
                    text=text,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
            
            # Move to device
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(device)
            
            # Add labels for training
            inputs["labels"] = inputs["input_ids"].clone()
            
            return inputs
        else:
            # Handle custom models or text-only inputs
            if images is not None and self.model_type == "custom":
                # Insert vision placeholders
                text_with_vision = f"{self.vision_start_token}{self.vision_end_token} {text}"
            else:
                text_with_vision = text
                
            inputs = self.tokenizer(
                text_with_vision,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone()
            }
            
            if images is not None and self.model_type == "custom":
                images = images.to(device)
                vision_features = self.encode_images(images)  # (B, seq_len, hidden_size)
                result["vision_features"] = vision_features
            
            return result
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VL model.
        
        Args:
            input_ids: (batch_size, seq_len) token ids
            attention_mask: (batch_size, seq_len) attention mask
            pixel_values: (batch_size, 3, H, W) pixel values for LLaVA
            vision_features: (batch_size, vision_seq_len, hidden_size) vision features
            labels: (batch_size, seq_len) labels for loss computation
            
        Returns:
            Dictionary with logits, loss, etc.
        """
        if self.model_type in ["llava", "blip2", "qwen", "internvl"]:
            # Use native forward pass (works for LLaVA, BLIP2, Qwen, and InternVL)
            llm_kwargs = dict(attention_mask=attention_mask, labels=labels, **kwargs)
            if self.model_type == "blip2" and pixel_values is None:
                base = input_ids if input_ids is not None else inputs_embeds
                if base is None:
                    raise ValueError("BLIP-2 forward requires input_ids or inputs_embeds")
                batch_size = base.size(0)
                device = base.device
                pixel_values = torch.zeros((batch_size, 3, 224, 224), dtype=torch.float32, device=device)
            if pixel_values is not None:
                llm_kwargs["pixel_values"] = pixel_values
                if self.model_type == "internvl" and "image_flags" not in llm_kwargs:
                    flags = None
                    try:
                        image_token_id = getattr(self.llm, "img_context_token_id", None)
                        if not isinstance(image_token_id, int) or image_token_id < 0:
                            image_token_id = getattr(self.llm.config, "image_token_id", None)
                        if (
                            isinstance(image_token_id, int)
                            and image_token_id >= 0
                            and input_ids is not None
                            and input_ids.dim() == 2
                            and input_ids.size(0) == pixel_values.size(0)
                        ):
                            flags = (input_ids == image_token_id).any(dim=1).to(dtype=torch.long)
                    except Exception:
                        flags = None
                    if flags is None:
                        flags = torch.ones(
                            (pixel_values.size(0),),
                            dtype=torch.long,
                            device=pixel_values.device,
                        )
                    else:
                        flags = flags.to(device=pixel_values.device, dtype=torch.long)
                    llm_kwargs["image_flags"] = flags.unsqueeze(-1)

            if input_ids is not None:
                outputs = self.llm(input_ids=input_ids, **llm_kwargs)
            elif inputs_embeds is not None:
                outputs = self.llm(inputs_embeds=inputs_embeds, **llm_kwargs)
            else:
                raise ValueError("BaseVLModel.forward requires input_ids or inputs_embeds")
        else:
            # Handle custom models
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("Custom model forward requires input_ids or inputs_embeds")
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            
            if vision_features is not None:
                # Placeholder: vision fusion for custom models would go here
                pass

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None
        }
    
    def generate(
        self,
        text: str,
        images: Optional[torch.Tensor] = None,
        max_length: int = 100,
        **generation_kwargs
    ) -> str:
        """
        Generate text response given input text and optional images.
        
        Args:
            text: Input text prompt
            images: Optional image tensor or PIL Images
            max_length: Maximum generation length
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        inputs = self.prepare_inputs_for_training(text, images)
        
        with torch.no_grad():
            if self.model_type in ["llava", "blip2", "internvl"] and images is not None:
                generated = self.llm.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs.get("pixel_values"),
                    max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs
                )
            else:
                generated = self.llm.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs
                )
        
        # Decode only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_text = self.tokenizer.decode(
            generated[0][input_length:], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
