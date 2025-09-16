import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union, Tuple
from .base_vl import BaseVLModel
from .audio_encoders import CLAPAudioEncoder, WhisperAudioEncoder, MultiModalAudioEncoder
from .projectors import AudioProjector, AdaptiveAudioProjector
from .fusion_adapter import LoRAFusionAdapter, MultiLayerFusionAdapter, GatedFusionAdapter


class SAFEModel(nn.Module):
    """
    SAFE: Simple, Adaptive, Failure-Proof Audio Addition to a VL Model
    
    This model extends a base VL model with audio capabilities while preserving
    VL performance through careful architectural choices and training procedures.
    """
    
    def __init__(
        self,
        # Base VL configuration
        llm_model_name: str = "microsoft/DialoGPT-medium",
        vision_model_name: str = "openai/clip-vit-large-patch14",
        
        # Audio configuration
        audio_encoder_type: str = "clap",  # "clap", "whisper", "multimodal"
        audio_encoder_config: dict = None,
        
        # Projector configuration
        projector_type: str = "standard",  # "standard", "adaptive"
        num_audio_tokens: int = 8,
        projector_config: dict = None,
        
        # Fusion configuration
        fusion_type: str = "lora",  # "lora", "multilayer", "gated"
        fusion_layer_indices: List[int] = None,
        lora_rank: int = 8,
        fusion_config: dict = None,
        
        # Training configuration
        freeze_base_vl: bool = True,
        freeze_audio_encoder: bool = True,
        
        # Model dimensions
        llm_hidden_size: int = 1024,
        audio_embed_dim: int = 512,
    ):
        super().__init__()
        
        self.audio_encoder_type = audio_encoder_type
        self.projector_type = projector_type
        self.fusion_type = fusion_type
        self.num_audio_tokens = num_audio_tokens
        self.llm_hidden_size = llm_hidden_size
        
        # Initialize base VL model
        self.base_vl = BaseVLModel(
            llm_model_name=llm_model_name,
            vision_model_name=vision_model_name,
            llm_hidden_size=llm_hidden_size,
            freeze_vision=freeze_base_vl,
            freeze_llm=freeze_base_vl
        )
        
        # Initialize audio encoder
        audio_encoder_config = audio_encoder_config or {}
        if audio_encoder_type == "clap":
            self.audio_encoder = CLAPAudioEncoder(
                freeze=freeze_audio_encoder,
                **audio_encoder_config
            )
            audio_embed_dim = self.audio_encoder.audio_embed_dim
        elif audio_encoder_type == "whisper":
            self.audio_encoder = WhisperAudioEncoder(
                freeze=freeze_audio_encoder,
                **audio_encoder_config
            )
            audio_embed_dim = self.audio_encoder.audio_embed_dim
        elif audio_encoder_type == "multimodal":
            self.audio_encoder = MultiModalAudioEncoder(**audio_encoder_config)
            audio_embed_dim = self.audio_encoder.total_embed_dim
        else:
            raise ValueError(f"Unsupported audio encoder type: {audio_encoder_type}")
        
        # Initialize audio projector
        projector_config = projector_config or {}
        if projector_type == "standard":
            self.audio_projector = AudioProjector(
                audio_embed_dim=audio_embed_dim,
                llm_hidden_size=llm_hidden_size,
                num_audio_tokens=num_audio_tokens,
                **projector_config
            )
        elif projector_type == "adaptive":
            self.audio_projector = AdaptiveAudioProjector(
                audio_embed_dim=audio_embed_dim,
                llm_hidden_size=llm_hidden_size,
                max_audio_tokens=num_audio_tokens,
                **projector_config
            )
        else:
            raise ValueError(f"Unsupported projector type: {projector_type}")
        
        # Initialize fusion adapter
        fusion_config = fusion_config or {}
        if fusion_type == "lora":
            self.fusion_adapter = LoRAFusionAdapter(
                hidden_size=llm_hidden_size,
                lora_rank=lora_rank,
                **fusion_config
            )
        elif fusion_type == "multilayer":
            self.fusion_adapter = MultiLayerFusionAdapter(
                hidden_size=llm_hidden_size,
                fusion_layer_indices=fusion_layer_indices,
                lora_rank=lora_rank,
                **fusion_config
            )
        elif fusion_type == "gated":
            self.fusion_adapter = GatedFusionAdapter(
                hidden_size=llm_hidden_size,
                lora_rank=lora_rank,
                **fusion_config
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
        # Special tokens for audio
        self.audio_start_token = "<audio>"
        self.audio_end_token = "</audio>"
        
        # Store original vocabulary size to preserve base VL model
        self.original_vocab_size = len(self.base_vl.tokenizer)
        
        # Add audio tokens to tokenizer but DON'T resize embeddings yet
        special_tokens = [self.audio_start_token, self.audio_end_token]
        num_added = self.base_vl.tokenizer.add_tokens(special_tokens)
        
        # Store audio token IDs for later use
        self.audio_start_token_id = self.base_vl.tokenizer.convert_tokens_to_ids(self.audio_start_token)
        self.audio_end_token_id = self.base_vl.tokenizer.convert_tokens_to_ids(self.audio_end_token)
        
        # Create a separate embedding layer for audio tokens to avoid corrupting base model
        if num_added > 0:
            # Get the actual embedding dimension from the base model
            base_embeddings = self.base_vl.llm.get_input_embeddings()
            actual_hidden_size = base_embeddings.weight.size(1)
            
            self.audio_token_embeddings = nn.Embedding(
                num_added, 
                actual_hidden_size,
                dtype=base_embeddings.weight.dtype
            )
            # Initialize audio token embeddings with mean of existing embeddings
            with torch.no_grad():
                mean_embedding = base_embeddings.weight.mean(dim=0)
                for i in range(num_added):
                    self.audio_token_embeddings.weight[i] = mean_embedding
            
            # Update our stored hidden size to match the actual model
            self.llm_hidden_size = actual_hidden_size
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get input embeddings, handling both original tokens and new audio tokens.
        
        Args:
            input_ids: Token IDs
            
        Returns:
            Embeddings tensor
        """
        # Get base embeddings for original vocabulary
        base_embeddings = self.base_vl.llm.get_input_embeddings()
        
        # Check if we have any audio tokens
        has_audio_tokens = (input_ids >= self.original_vocab_size).any()
        
        if not has_audio_tokens:
            # No audio tokens, use base embeddings directly
            return base_embeddings(input_ids)
        
        # Handle mixed tokens (original + audio)
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Create output tensor using actual embedding dimension
        actual_hidden_size = base_embeddings.weight.size(1)
        embeddings = torch.zeros(
            batch_size, seq_len, actual_hidden_size,
            dtype=base_embeddings.weight.dtype,
            device=device
        )
        
        # Mask for original tokens (within original vocabulary)
        original_mask = input_ids < self.original_vocab_size
        audio_mask = input_ids >= self.original_vocab_size
        
        # Fill in original token embeddings
        if original_mask.any():
            original_ids = input_ids.masked_fill(~original_mask, 0)  # Use 0 for masked positions
            original_embeds = base_embeddings(original_ids)
            embeddings[original_mask] = original_embeds[original_mask]
        
        # Fill in audio token embeddings
        if audio_mask.any() and hasattr(self, 'audio_token_embeddings'):
            # Map audio token IDs to indices in audio_token_embeddings
            audio_ids = input_ids[audio_mask] - self.original_vocab_size
            audio_embeds = self.audio_token_embeddings(audio_ids)
            embeddings[audio_mask] = audio_embeds
        
        return embeddings
    
    def get_trainable_parameters(self):
        """
        Get trainable parameters for Stage A training.
        
        Returns:
            Iterator of trainable parameters
        """
        # Audio projector parameters
        for param in self.audio_projector.parameters():
            yield param
        
        # Fusion adapter parameters
        for param in self.fusion_adapter.parameters():
            yield param
        
        # Audio token embeddings (new addition)
        if hasattr(self, 'audio_token_embeddings'):
            for param in self.audio_token_embeddings.parameters():
                yield param
    
    def enable_audio_training(self):
        """Enable training mode for audio components while keeping base VL frozen."""
        # Set audio components to training mode
        self.audio_projector.train()
        self.fusion_adapter.train()
        if hasattr(self, 'audio_token_embeddings'):
            self.audio_token_embeddings.train()
        
        # Ensure base VL model stays frozen and in eval mode
        self.base_vl.eval()
        for param in self.base_vl.parameters():
            param.requires_grad = False
    
    def eval(self):
        """Override eval to handle both base VL and audio components."""
        super().eval()
        self.base_vl.eval()
        return self
    
    def encode_audio(
        self, 
        audio: Union[torch.Tensor, List[str], List], 
        num_tokens: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[List[str]]]:
        """
        Encode audio into token representations.
        
        Args:
            audio: Audio input (various formats supported)
            num_tokens: Optional override for number of tokens
            
        Returns:
            Tuple of (audio_tokens, transcripts)
        """
        # Extract audio features
        if self.audio_encoder_type in ["clap", "multimodal"]:
            audio_features, transcripts = self.audio_encoder(audio), None
        else:  # whisper
            audio_features, transcripts = self.audio_encoder(audio)
            
        # Project to token space
        if self.projector_type == "adaptive":
            audio_tokens = self.audio_projector(audio_features, num_tokens)
        else:
            audio_tokens = self.audio_projector(audio_features)
            
        return audio_tokens, transcripts
    
    def prepare_multimodal_inputs(
        self,
        text: Union[str, List[str]],
        images: Optional[Union[torch.Tensor, List]] = None,
        audio: Optional[Union[torch.Tensor, List[str]]] = None,
        device: str = "cuda",
        include_audio_tokens: bool = True,
        num_audio_tokens: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for multimodal processing.
        
        Args:
            text: Input text
            images: Optional images
            audio: Optional audio
            device: Target device
            include_audio_tokens: Whether to include audio token placeholders
            num_audio_tokens: Override for number of audio tokens
            
        Returns:
            Dictionary with input_ids, attention_mask, audio_tokens, etc.
        """
        # For LLaVA/BLIP2, use proper multimodal input preparation
        if self.base_vl.model_type == "llava":
            # LLaVA-specific handling with chat templates and proper <image> token insertion
            result = self._prepare_llava_inputs(text, images, device)
        elif self.base_vl.model_type == "blip2":
            # BLIP2-specific handling
            result = self._prepare_blip2_inputs(text, images, device)
        else:
            # For custom models, use the original approach
            text_with_modalities = text
            
            if images is not None:
                text_with_modalities = f"{self.base_vl.vision_start_token}{self.base_vl.vision_end_token} {text_with_modalities}"
                
            if audio is not None and include_audio_tokens:
                text_with_modalities = f"{self.audio_start_token}{self.audio_end_token} {text_with_modalities}"
            
            # Tokenize text
            inputs = self.base_vl.tokenizer(
                text_with_modalities,
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
            
            # Process vision if provided
            if images is not None:
                images = images.to(device)
                vision_features = self.base_vl.encode_images(images)
                result["vision_features"] = vision_features
        
        # Process audio if provided (same for all model types)
        if audio is not None:
            audio_tokens, transcripts = self.encode_audio(audio, num_audio_tokens)
            audio_tokens = audio_tokens.to(device)
            result["audio_tokens"] = audio_tokens
            if transcripts is not None:
                result["transcripts"] = transcripts
                
        return result
    
    def _prepare_llava_inputs(
        self, 
        text: Union[str, List[str]], 
        images: Optional[Union[torch.Tensor, List]] = None, 
        device: str = "cuda"
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for LLaVA with proper chat templates and <image> tokens.
        """
        from PIL import Image
        from pathlib import Path
        
        processor = self.base_vl.processor
        
        # Ensure we have lists for batch processing
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Convert images to PIL format if needed
        pil_images = []
        if images is not None:
            if isinstance(images, list):
                for img in images:
                    pil_images.append(self._convert_to_pil(img))
            else:
                # Single image or batch tensor
                if isinstance(images, torch.Tensor) and images.dim() == 4:
                    # Batch of images
                    for i in range(images.shape[0]):
                        pil_images.append(self._convert_to_pil(images[i]))
                else:
                    # Single image
                    pil_images = [self._convert_to_pil(images)]
        
        # Build LLaVA chat conversations with proper <image> token placement
        conversations = []
        for i, question in enumerate(texts):
            if i < len(pil_images) and pil_images[i] is not None:
                # Multimodal conversation
                conversations.append([
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"}, 
                            {"type": "text", "text": question}
                        ]
                    }
                ])
            else:
                # Text-only conversation
                conversations.append([
                    {"role": "user", "content": question}
                ])
        
        # Apply chat template to get proper prompts with <image> tokens
        try:
            prompts = []
            for conv in conversations:
                prompt = processor.apply_chat_template(
                    conv, 
                    add_generation_prompt=True, 
                    tokenize=False
                )
                prompts.append(prompt)
        except (AttributeError, NotImplementedError):
            # Fallback: manually insert image token
            image_token = getattr(processor, 'image_token', '<image>')
            prompts = []
            for i, question in enumerate(texts):
                if i < len(pil_images) and pil_images[i] is not None:
                    prompts.append(f"USER: {image_token}\n{question}\nASSISTANT:")
                else:
                    prompts.append(f"USER: {question}\nASSISTANT:")
        
        # Process with LLaVA processor
        if pil_images and any(img is not None for img in pil_images):
            # Ensure we have same number of images as prompts (pad with None if needed)
            while len(pil_images) < len(prompts):
                pil_images.append(None)
            
            # Filter out None images and corresponding prompts
            valid_pairs = [(p, img) for p, img in zip(prompts, pil_images) if img is not None]
            if valid_pairs:
                valid_prompts, valid_images = zip(*valid_pairs)
                inputs = processor(
                    text=list(valid_prompts),
                    images=list(valid_images),
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
            else:
                # No valid images, process as text-only
                inputs = processor(
                    text=prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
        else:
            # Text-only processing
            inputs = processor(
                text=prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        
        # Move to device
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].to(device)
        
        # Add labels for training
        inputs["labels"] = inputs["input_ids"].clone()
        
        # Sanity checks for LLaVA
        if pil_images and any(img is not None for img in pil_images):
            image_token_id = self._get_image_token_id()
            if not (inputs["input_ids"] == image_token_id).any():
                print(f"WARNING: No <image> token (ID: {image_token_id}) found in input_ids for LLaVA!")
                print(f"Prompt sample: {prompts[0][:200]}...")
            
            if "pixel_values" not in inputs or inputs["pixel_values"] is None:
                print(f"WARNING: pixel_values missing for LLaVA with images!")
        
        return inputs
    
    def _prepare_blip2_inputs(
        self, 
        text: Union[str, List[str]], 
        images: Optional[Union[torch.Tensor, List]] = None, 
        device: str = "cuda"
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for BLIP2 (simpler - no chat templates needed).
        """
        # Use the existing base VL preparation for BLIP2
        return self.base_vl.prepare_inputs_for_training(text, images, device)
    
    def _convert_to_pil(self, image):
        """
        Convert various image formats to PIL Image.
        """
        from PIL import Image
        from pathlib import Path
        
        if image is None:
            return None
        
        if isinstance(image, Image.Image):
            return image
        
        if isinstance(image, (str, type(Path(".")))):
            try:
                return Image.open(image).convert("RGB")
            except:
                return None
        
        if isinstance(image, torch.Tensor):
            img_tensor = image
            
            # Handle different tensor dimensions
            if img_tensor.dim() == 4:  # (B, C, H, W) - take first
                img_tensor = img_tensor[0]
            elif img_tensor.dim() == 2:  # (H, W) - add channels
                img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
            elif img_tensor.dim() != 3:
                return None
            
            # Convert to numpy and PIL format
            if img_tensor.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                img_tensor = img_tensor.permute(1, 2, 0)
            elif img_tensor.shape[2] == 3:  # Already (H, W, C)
                pass
            else:
                return None
            
            # Ensure values are in [0, 255] range
            if img_tensor.max() <= 1.0:
                img_tensor = (img_tensor * 255).clamp(0, 255)
            
            import numpy as np
            img_np = img_tensor.cpu().numpy().astype(np.uint8)
            return Image.fromarray(img_np)
        
        return None
    
    def _get_image_token_id(self):
        """Get the image token ID for LLaVA models."""
        # Be defensive across processors/tokenizers  
        bv = self.base_vl
        tok = getattr(bv, "tokenizer", None)
        proc = getattr(bv, "processor", None)

        image_token = None
        if proc is not None and hasattr(proc, "image_token"):
            image_token = proc.image_token  # e.g., "<image>"
        if image_token is None:
            # Common default in Llava; still try tokenizer first
            image_token = "<image>"

        if tok is not None and hasattr(tok, "convert_tokens_to_ids"):
            # Handle both AddedToken objects and string tokens
            if hasattr(image_token, 'content'):
                # It's an AddedToken, get the string content
                token_str = image_token.content
            else:
                # It's already a string
                token_str = str(image_token)
            
            # convert_tokens_to_ids expects a list of tokens
            token_ids = tok.convert_tokens_to_ids([token_str])
            return token_ids[0] if token_ids else None

        # Last-resort fallback (LLaVA often uses 32000)
        return getattr(bv, "image_token_id", 32000)

    # Rest of the methods remain the same as original...
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        audio_tokens: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        gate: float = 1.0,
        fusion_layer: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through SAFE model."""
        # For BLIP2/LLaVA models, always use base VL model as they handle multimodal fusion internally
        if self.base_vl.model_type in ["blip2", "llava"]:
            # Check if input_ids contain audio tokens
            has_audio_tokens = (input_ids >= self.original_vocab_size).any()
            
            if has_audio_tokens:
                # Replace audio tokens with a generic token to avoid index errors
                unk_token_id = self.base_vl.tokenizer.unk_token_id or 0
                clean_input_ids = input_ids.clone()
                clean_input_ids[input_ids >= self.original_vocab_size] = unk_token_id
            else:
                clean_input_ids = input_ids
            
            base_inputs = {
                "input_ids": clean_input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
            if "pixel_values" in kwargs:
                base_inputs["pixel_values"] = kwargs["pixel_values"]
            
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["pixel_values"]}
            return self.base_vl(**base_inputs, **filtered_kwargs)
        
        # For other model types, use custom implementation
        if audio_tokens is None or gate == 0.0:
            has_audio_tokens = (input_ids >= self.original_vocab_size).any()
            
            if not has_audio_tokens:
                base_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "vision_features": vision_features
                }
                return self.base_vl(**base_inputs, **kwargs)
            else:
                inputs_embeds = self.get_input_embeddings(input_ids)
                base_inputs = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "vision_features": vision_features
                }
                return self.base_vl(**base_inputs, **kwargs)
        
        # Get token embeddings using our custom method
        inputs_embeds = self.get_input_embeddings(input_ids)
        
        # Forward through LLM with audio fusion (simplified implementation)
        if hasattr(self.base_vl.llm, 'transformer'):
            hidden_states = inputs_embeds
            
            for i, layer in enumerate(self.base_vl.llm.transformer.h):
                hidden_states = layer(hidden_states)[0]
                
                # Apply audio fusion at appropriate layers
                if self.fusion_type == "multilayer":
                    hidden_states = self.fusion_adapter(
                        hidden_states=hidden_states,
                        audio_tokens=audio_tokens,
                        layer_idx=i,
                        attention_mask=None,
                        gate=gate,
                        active_fusion_layer=fusion_layer
                    )
                elif i == len(self.base_vl.llm.transformer.h) // 2:  # Mid-layer fusion
                    if self.fusion_type == "gated":
                        hidden_states, _ = self.fusion_adapter(
                            hidden_states=hidden_states,
                            audio_tokens=audio_tokens,
                            force_gate=gate if gate != 1.0 else None
                        )
                    else:
                        hidden_states = self.fusion_adapter(
                            hidden_states,
                            audio_tokens,
                            None,
                            gate
                        )
            
            hidden_states = self.base_vl.llm.transformer.ln_f(hidden_states)
            logits = self.base_vl.llm.lm_head(hidden_states)
        else:
            if self.base_vl.model_type in ["blip2", "llava"]:
                base_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
                if "pixel_values" in kwargs:
                    base_inputs["pixel_values"] = kwargs["pixel_values"]
                    
                outputs = self.base_vl(**base_inputs, **{k: v for k, v in kwargs.items() if k not in ["pixel_values"]})
            else:
                outputs = self.base_vl.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs
                )
            logits = outputs.logits
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1)
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states if 'hidden_states' in locals() else None
        }
    
    def generate(
        self,
        text: Optional[str] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[Union[torch.Tensor, List[str]]] = None,
        max_length: int = 100,
        gate: float = 1.0,
        num_audio_tokens: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **generation_kwargs
    ) -> Union[str, torch.Tensor]:
        """Generate text response given multimodal inputs."""
        # Low-level API: direct tensor inputs
        if input_ids is not None:
            base_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **generation_kwargs
            }
            if pixel_values is not None:
                base_inputs["pixel_values"] = pixel_values
                
            return self.base_vl.llm.generate(**base_inputs)
        
        # High-level API: text and multimodal inputs
        if text is None:
            raise ValueError("Either 'text' or 'input_ids' must be provided")
            
        inputs = self.prepare_multimodal_inputs(
            text=text,
            images=images,
            audio=audio,
            include_audio_tokens=(audio is not None),
            num_audio_tokens=num_audio_tokens
        )
        
        with torch.no_grad():
            if audio is not None and gate > 0.0:
                generated = self.base_vl.llm.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    pad_token_id=self.base_vl.tokenizer.pad_token_id,
                    eos_token_id=self.base_vl.tokenizer.eos_token_id,
                    **generation_kwargs
                )
            else:
                generated = self.base_vl.generate(
                    text=text,
                    images=images,
                    max_length=max_length,
                    **generation_kwargs
                )
                return generated
        
        input_length = inputs["input_ids"].shape[1]
        generated_text = self.base_vl.tokenizer.decode(
            generated[0][input_length:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def get_base_vl_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Get logits from base VL model for retention loss computation."""
        with torch.no_grad():
            outputs = self.base_vl(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_features=vision_features,
                **kwargs
            )
        return outputs["logits"]
    
    def enable_audio_training(self):
        """Enable training mode for audio components only."""
        self.audio_projector.train()
        self.fusion_adapter.train()
        
        # Freeze base components
        self.base_vl.eval()
        self.audio_encoder.eval()
        
    def enable_full_training(self):
        """Enable training mode for all components."""
        self.train()
        
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (audio projector + fusion adapter)."""
        params = []
        params.extend(list(self.audio_projector.parameters()))
        params.extend(list(self.fusion_adapter.parameters()))
        return [p for p in params if p.requires_grad]