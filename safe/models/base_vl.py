import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
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
    ):
        super().__init__()
        
        self.llm_model_name = llm_model_name
        self.vision_model_name = vision_model_name
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.num_vision_tokens = num_vision_tokens
        
        # Load vision encoder (frozen) - use safetensors to avoid PyTorch security issue
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            vision_model_name,
            use_safetensors=True
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
        
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        # Determine appropriate dtype based on device availability
        # Use float16 for GPU, float32 for CPU to avoid LayerNorm issues
        device_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load LLM (frozen) - handle different VL models
        if "llava" in llm_model_name.lower():
            self.llm = LlavaForConditionalGeneration.from_pretrained(
                llm_model_name,
                torch_dtype=device_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            self.processor = LlavaProcessor.from_pretrained(llm_model_name)
            self.tokenizer = self.processor.tokenizer
            self.model_type = "llava"
        elif "blip2" in llm_model_name.lower():
            self.llm = Blip2ForConditionalGeneration.from_pretrained(
                llm_model_name,
                torch_dtype=device_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            self.processor = Blip2Processor.from_pretrained(llm_model_name)
            self.tokenizer = self.processor.tokenizer
            self.model_type = "blip2"
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                use_safetensors=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.model_type = "custom"
        
        # Configure all tokenizers comprehensively
        self._configure_tokenizers()
            
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
                
        # Vision projector - only needed for custom models
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
            # LLaVA and BLIP2 already have vision integration, we'll use them directly
            self.vision_projector = None
        
        # Special tokens - only for custom models
        if self.model_type == "custom":
            self.vision_start_token = "<img>"
            self.vision_end_token = "</img>"
            
            # Add special tokens to tokenizer
            special_tokens = [self.vision_start_token, self.vision_end_token]
            self.tokenizer.add_tokens(special_tokens)
            self.llm.resize_token_embeddings(len(self.tokenizer))
        else:
            # LLaVA and BLIP2 use their own vision tokens
            self.vision_start_token = None
            self.vision_end_token = None
    
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
        if self.model_type in ["llava", "blip2"]:
            # For LLaVA/BLIP2, we'll let the model handle vision encoding internally
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
        
        if self.model_type in ["llava", "blip2"] and has_valid_images:
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
        if self.model_type in ["llava", "blip2"]:
            # Use native multimodal forward pass
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
            if self.model_type in ["llava", "blip2"] and images is not None:
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
