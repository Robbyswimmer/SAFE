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
        
        # Add audio tokens to tokenizer
        special_tokens = [self.audio_start_token, self.audio_end_token]
        self.base_vl.tokenizer.add_tokens(special_tokens)
        self.base_vl.llm.resize_token_embeddings(len(self.base_vl.tokenizer))
        
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
        text: str,
        images: Optional[torch.Tensor] = None,
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
        # For LLaVA/BLIP2, use the base VL model's proper input preparation
        if self.base_vl.model_type in ["llava", "blip2"]:
            # Handle case where images might be a list (from batch processing)
            if images is not None and isinstance(images, list):
                # Process each text-image pair individually and then batch
                from PIL import Image
                import numpy as np
                
                
                batch_results = []
                texts = text if isinstance(text, list) else [text]
                
                for i, (single_text, single_image) in enumerate(zip(texts, images)):
                    if single_image is not None:
                        # Convert tensor to PIL
                        if isinstance(single_image, torch.Tensor):
                            img_tensor = single_image
                            
                            # Handle different tensor dimensions from real datasets
                            if img_tensor.dim() == 4:  # (B, C, H, W) - take first sample
                                img_tensor = img_tensor[0]
                            elif img_tensor.dim() == 2:  # (H, W) - add channel dim
                                img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
                            elif img_tensor.dim() != 3:
                                raise ValueError(f"Unexpected image tensor dimensions: {img_tensor.shape}")
                            
                            # Convert to numpy and PIL format
                            if img_tensor.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                                img_tensor = img_tensor.permute(1, 2, 0)
                            elif img_tensor.shape[2] == 3:  # Already (H, W, C)
                                pass  # No permutation needed
                            else:
                                raise ValueError(f"Unexpected channel dimension in tensor: {img_tensor.shape}")
                            
                            # Ensure values are in [0, 255] range
                            if img_tensor.max() <= 1.0:
                                img_tensor = (img_tensor * 255).clamp(0, 255)
                            
                            img_np = img_tensor.cpu().numpy().astype(np.uint8)
                            pil_image = Image.fromarray(img_np)
                        else:
                            pil_image = single_image
                        
                        # Process this text-image pair
                        single_result = self.base_vl.prepare_inputs_for_training(
                            text=single_text,
                            images=pil_image,
                            device=device
                        )
                    else:
                        # Text-only for this sample
                        single_result = self.base_vl.prepare_inputs_for_training(
                            text=single_text,
                            images=None,
                            device=device
                        )
                    
                    batch_results.append(single_result)
                
                # Combine results into a batch
                if batch_results:
                    result = {}
                    # Get all unique keys across all batch results
                    all_keys = set()
                    for br in batch_results:
                        all_keys.update(br.keys())
                    
                    for key in all_keys:
                        # Only collect values where the key exists
                        values = [br[key] for br in batch_results if key in br]
                        
                        # Skip if no values found for this key
                        if not values:
                            continue
                        # Special handling for pixel_values - include them even in mixed batches
                        if key == "pixel_values":
                            if len(values) > 0:
                                # We have at least some pixel_values, concatenate them
                                result[key] = torch.cat(values, dim=0) if len(values) > 1 else values[0]
                            continue
                        
                        if isinstance(values[0], torch.Tensor):
                            # Handle tensors with different sequence lengths by padding
                            if len(values) > 1 and values[0].dim() > 1:
                                # Find max length in the sequence dimension (usually dim 1)
                                max_length = max(v.shape[1] for v in values)
                                padded_values = []
                                
                                for v in values:
                                    if v.shape[1] < max_length:
                                        # Pad tensor to max length
                                        pad_size = max_length - v.shape[1]
                                        if key in ["input_ids", "labels"]:
                                            # Use pad token ID for input_ids and labels
                                            pad_value = 0  # Most tokenizers use 0 for padding
                                        elif key == "attention_mask":
                                            # Use 0 for attention mask padding
                                            pad_value = 0
                                        else:
                                            # Use 0 for other tensors
                                            pad_value = 0
                                        
                                        # Create padding tensor
                                        pad_shape = list(v.shape)
                                        pad_shape[1] = pad_size
                                        padding = torch.full(pad_shape, pad_value, dtype=v.dtype, device=v.device)
                                        
                                        # Concatenate with padding
                                        padded_v = torch.cat([v, padding], dim=1)
                                        padded_values.append(padded_v)
                                    else:
                                        padded_values.append(v)
                                
                                result[key] = torch.cat(padded_values, dim=0)
                            else:
                                # For single sample or tensors without sequence dimension
                                result[key] = torch.cat(values, dim=0)
                        else:
                            result[key] = values
                else:
                    # Fallback to text-only
                    result = self.base_vl.prepare_inputs_for_training(
                        text=text,
                        images=None,
                        device=device
                    )
            elif images is not None:
                # Single image or tensor batch
                if isinstance(images, torch.Tensor):
                    # Convert from tensor to PIL Image format expected by processor
                    from PIL import Image
                    import numpy as np
                    
                    if images.dim() == 4:  # (B, C, H, W) - batch of images
                        # Process all images in the batch
                        pil_images = []
                        for i in range(images.shape[0]):
                            img_tensor = images[i]  # (C, H, W)
                            
                            # Handle different tensor dimensions
                            if img_tensor.dim() == 4:  # Nested batch - take first
                                img_tensor = img_tensor[0]
                            elif img_tensor.dim() == 2:  # (H, W) - add channels
                                img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
                            
                            # Convert to numpy and PIL format
                            if img_tensor.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                                img_tensor = img_tensor.permute(1, 2, 0)
                            elif img_tensor.shape[2] == 3:  # Already (H, W, C)
                                pass  # No permutation needed
                            else:
                                raise ValueError(f"Unexpected channel dimension in batched tensor: {img_tensor.shape}")
                            
                            # Ensure values are in [0, 255] range
                            if img_tensor.max() <= 1.0:
                                img_tensor = (img_tensor * 255).clamp(0, 255)
                            
                            img_np = img_tensor.cpu().numpy().astype(np.uint8)
                            pil_images.append(Image.fromarray(img_np))
                        
                        # Use base VL model's proper input preparation with all images
                        result = self.base_vl.prepare_inputs_for_training(
                            text=text,
                            images=pil_images,
                            device=device
                        )
                    else:
                        # Single image - handle various dimensions
                        img_tensor = images
                        
                        # Handle different tensor dimensions
                        if img_tensor.dim() == 4:  # (1, C, H, W) - remove batch dim
                            img_tensor = img_tensor.squeeze(0)
                        elif img_tensor.dim() == 2:  # (H, W) - add channels
                            img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
                        elif img_tensor.dim() != 3:
                            raise ValueError(f"Unexpected single image tensor dimensions: {img_tensor.shape}")
                        
                        # Convert to numpy and PIL format
                        if img_tensor.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                            img_tensor = img_tensor.permute(1, 2, 0)
                        elif img_tensor.shape[2] == 3:  # Already (H, W, C)
                            pass  # No permutation needed
                        else:
                            raise ValueError(f"Unexpected channel dimension in single tensor: {img_tensor.shape}")
                        
                        # Ensure values are in [0, 255] range
                        if img_tensor.max() <= 1.0:
                            img_tensor = (img_tensor * 255).clamp(0, 255)
                        
                        img_np = img_tensor.cpu().numpy().astype(np.uint8)
                        pil_image = Image.fromarray(img_np)
                        
                        # Use base VL model's proper input preparation
                        result = self.base_vl.prepare_inputs_for_training(
                            text=text,
                            images=pil_image,
                            device=device
                        )
                else:
                    # Already PIL images
                    result = self.base_vl.prepare_inputs_for_training(
                        text=text,
                        images=images,
                        device=device
                    )
            else:
                # Text-only input
                result = self.base_vl.prepare_inputs_for_training(
                    text=text,
                    images=None,
                    device=device
                )
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
        """
        Forward pass through SAFE model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            vision_features: Optional vision features
            audio_tokens: Optional audio tokens
            labels: Optional labels for loss computation
            gate: Audio fusion gate (0.0 = no audio, 1.0 = full audio)
            fusion_layer: Specific layer for fusion (if supported)
            
        Returns:
            Dictionary with logits, loss, etc.
        """
        # If no audio, gate is 0, or using BLIP2/LLaVA (which don't support manual fusion), use base VL model
        if audio_tokens is None or gate == 0.0 or self.base_vl.model_type in ["blip2", "llava"]:
            base_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            # For BLIP2/LLaVA, pass pixel_values; for others, pass vision_features
            if self.base_vl.model_type in ["blip2", "llava"]:
                if "pixel_values" in kwargs:
                    base_inputs["pixel_values"] = kwargs["pixel_values"]
            else:
                base_inputs["vision_features"] = vision_features
            
            # Filter out pixel_values from kwargs since we already added it to base_inputs if needed
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["pixel_values"]}
            return self.base_vl(**base_inputs, **filtered_kwargs)
        
        # Get token embeddings
        inputs_embeds = self.base_vl.llm.get_input_embeddings()(input_ids)
        
        # For a full implementation, you would need to properly insert
        # vision and audio features at the correct positions in the sequence.
        # This is a simplified version that shows the overall architecture.
        
        # Forward through LLM with potential audio fusion
        if hasattr(self.base_vl.llm, 'transformer'):
            # For GPT-style models
            hidden_states = inputs_embeds
            
            for i, layer in enumerate(self.base_vl.llm.transformer.h):
                hidden_states = layer(hidden_states)[0]
                
                # Apply audio fusion at appropriate layers
                if self.fusion_type == "multilayer":
                    hidden_states = self.fusion_adapter(
                        hidden_states=hidden_states,
                        audio_tokens=audio_tokens,
                        layer_idx=i,
                        attention_mask=None,  # Simplified
                        gate=gate,
                        active_fusion_layer=fusion_layer
                    )
                elif i == len(self.base_vl.llm.transformer.h) // 2:  # Mid-layer fusion
                    if self.fusion_type == "gated":
                        hidden_states, computed_gate = self.fusion_adapter(
                            hidden_states=hidden_states,
                            audio_tokens=audio_tokens,
                            force_gate=gate if gate != 1.0 else None
                        )
                    else:
                        # Only pass expected arguments to fusion adapter (no kwargs)
                        hidden_states = self.fusion_adapter(
                            hidden_states,
                            audio_tokens,
                            None,  # attention_mask
                            gate
                        )
            
            # Final layer norm and output projection
            hidden_states = self.base_vl.llm.transformer.ln_f(hidden_states)
            logits = self.base_vl.llm.lm_head(hidden_states)
        else:
            # For BLIP2 and other multimodal architectures, use the base VL forward
            if self.base_vl.model_type in ["blip2", "llava"]:
                # Use the base VL model's forward method which handles BLIP2 properly
                base_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
                if "pixel_values" in kwargs:
                    base_inputs["pixel_values"] = kwargs["pixel_values"]
                    
                outputs = self.base_vl(**base_inputs, **{k: v for k, v in kwargs.items() if k not in ["pixel_values"]})
            else:
                # For other model architectures, use inputs_embeds
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
        text: str,
        images: Optional[torch.Tensor] = None,
        audio: Optional[Union[torch.Tensor, List[str]]] = None,
        max_length: int = 100,
        gate: float = 1.0,
        num_audio_tokens: Optional[int] = None,
        **generation_kwargs
    ) -> str:
        """
        Generate text response given multimodal inputs.
        
        Args:
            text: Input text prompt
            images: Optional images
            audio: Optional audio
            max_length: Maximum generation length
            gate: Audio fusion gate
            num_audio_tokens: Override for number of audio tokens
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        inputs = self.prepare_multimodal_inputs(
            text=text,
            images=images,
            audio=audio,
            include_audio_tokens=(audio is not None),
            num_audio_tokens=num_audio_tokens
        )
        
        with torch.no_grad():
            if audio is not None and gate > 0.0:
                # Use SAFE model for generation with audio
                generated = self.base_vl.llm.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    pad_token_id=self.base_vl.tokenizer.pad_token_id,
                    eos_token_id=self.base_vl.tokenizer.eos_token_id,
                    **generation_kwargs
                )
            else:
                # Use base VL model without audio
                generated = self.base_vl.generate(
                    text=text,
                    images=images,
                    max_length=max_length,
                    **generation_kwargs
                )
                return generated
        
        # Decode generated text
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
        """
        Get logits from base VL model for retention loss computation.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            vision_features: Optional vision features
            
        Returns:
            Base VL model logits
        """
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