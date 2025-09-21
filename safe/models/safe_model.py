import torch
import torch.nn as nn
from collections import Counter
from typing import Any, Optional, Dict, List, Sequence, Union, Tuple
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

    def _select_training_answer(self, answer: Any) -> str:
        if answer is None:
            return ""
        if isinstance(answer, str):
            return answer
        if isinstance(answer, dict):
            return answer.get("answer", "")
        if isinstance(answer, (list, tuple)):
            if not answer:
                return ""
            if isinstance(answer[0], dict):
                counts = Counter(
                    item.get("answer", "") for item in answer if item.get("answer")
                )
                if counts:
                    return counts.most_common(1)[0][0]
                return ""
            counts = Counter(str(a) for a in answer)
            return counts.most_common(1)[0][0]
        return str(answer)

    def _apply_answers_to_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        answers: Optional[Union[str, Sequence[Any]]],
        device: torch.device,
    ) -> None:
        if answers is None:
            return

        tokenizer = self.base_vl.tokenizer
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        eos_id = tokenizer.eos_token_id

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        batch_size = input_ids.size(0)

        if isinstance(answers, str):
            answer_list = [answers] * batch_size
        else:
            answer_list = list(answers) if answers is not None else []

        if len(answer_list) == 0:
            answer_list = [""] * batch_size
        elif len(answer_list) == 1 and batch_size > 1:
            answer_list = answer_list * batch_size
        elif len(answer_list) != batch_size:
            if len(answer_list) < batch_size:
                answer_list.extend([""] * (batch_size - len(answer_list)))
            else:
                answer_list = answer_list[:batch_size]

        new_input_ids: List[torch.Tensor] = []
        new_attention: List[torch.Tensor] = []
        new_labels: List[torch.Tensor] = []
        max_length = 0

        for i, raw_answer in enumerate(answer_list):
            answer_text = self._select_training_answer(raw_answer)
            prompt_len = int(attention_mask[i].sum().item())
            prompt_tokens = input_ids[i, :prompt_len]

            answer_tokens = tokenizer.encode(
                answer_text,
                add_special_tokens=False,
            )
            if eos_id is not None:
                if not answer_tokens or answer_tokens[-1] != eos_id:
                    answer_tokens.append(eos_id)

            if not answer_tokens:
                # Ensure there is at least one target token to supervise on
                answer_tokens = [tokenizer.pad_token_id or pad_id]

            answer_tensor = torch.tensor(
                answer_tokens,
                dtype=prompt_tokens.dtype,
                device=device,
            )

            combined = torch.cat([prompt_tokens, answer_tensor], dim=0)
            labels_prefix = torch.full(
                (prompt_len,),
                -100,
                dtype=torch.long,
                device=device,
            )
            labels = torch.cat([labels_prefix, answer_tensor], dim=0)

            attention = torch.cat(
                [
                    torch.ones(prompt_len, dtype=attention_mask.dtype, device=device),
                    torch.ones(answer_tensor.size(0), dtype=attention_mask.dtype, device=device),
                ],
                dim=0,
            )

            new_input_ids.append(combined)
            new_labels.append(labels)
            new_attention.append(attention)
            max_length = max(max_length, combined.size(0))

        padded_ids = torch.full(
            (len(new_input_ids), max_length),
            pad_id,
            dtype=input_ids.dtype,
            device=device,
        )
        padded_attention = torch.zeros(
            (len(new_attention), max_length),
            dtype=new_attention[0].dtype,
            device=device,
        )
        padded_labels = torch.full(
            (len(new_labels), max_length),
            -100,
            dtype=torch.long,
            device=device,
        )

        for i, (seq, mask, label) in enumerate(zip(new_input_ids, new_attention, new_labels)):
            length = seq.size(0)
            padded_ids[i, :length] = seq
            padded_attention[i, :length] = mask
            padded_labels[i, :length] = label

        inputs["input_ids"] = padded_ids
        inputs["attention_mask"] = padded_attention
        inputs["labels"] = padded_labels
    
    def prepare_multimodal_inputs(
        self,
        text: Union[str, List[str]],
        images: Optional[Union[torch.Tensor, List]] = None,
        audio: Optional[Union[torch.Tensor, List[str]]] = None,
        answers: Optional[Union[str, Sequence[Any]]] = None,
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
            answers: Optional answers for supervised training
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
                images = self._ensure_image_batch(images)
                if images is not None:
                    images = images.to(device)
                    vision_features = self.base_vl.encode_images(images)
                    result["vision_features"] = vision_features
        
        # Process audio if provided (same for all model types)
        audio_to_encode = audio
        audio_indices = None
        batch_size = None

        if isinstance(audio, list):
            batch_size = len(audio)
            non_null = [(idx, a) for idx, a in enumerate(audio) if a is not None]
            if not non_null:
                audio_to_encode = None
            else:
                audio_indices, audio_list = zip(*non_null)
                audio_to_encode = list(audio_list)

        if audio_to_encode is not None:
            audio_tokens, transcripts = self.encode_audio(audio_to_encode, num_audio_tokens)

            if isinstance(audio, list) and audio_indices is not None:
                audio_tokens = self._scatter_audio_tokens(audio_tokens, audio_indices, batch_size)
                if transcripts is not None:
                    transcripts = self._scatter_transcripts(transcripts, audio_indices, batch_size)

            audio_tokens = audio_tokens.to(device)
            result["audio_tokens"] = audio_tokens
            if transcripts is not None:
                result["transcripts"] = transcripts

        # Attach ground truth answers for supervised training
        if answers is not None and "input_ids" in result and "attention_mask" in result:
            dev = result["input_ids"].device if torch.is_tensor(result["input_ids"]) else torch.device(device)
            self._apply_answers_to_inputs(result, answers, dev)
        elif "labels" not in result and "input_ids" in result:
            # Default labels mirror the input ids (e.g., for inference)
            result["labels"] = result["input_ids"].clone()

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
        
        num_samples = len(prompts)
        if not pil_images:
            pil_images = [None] * num_samples
        elif len(pil_images) < num_samples:
            pil_images.extend([None] * (num_samples - len(pil_images)))
        elif len(pil_images) > num_samples:
            pil_images = pil_images[:num_samples]

        sample_encodings = []
        for prompt, image in zip(prompts, pil_images):
            if image is not None:
                encoded = processor(
                    text=[prompt],
                    images=[image],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
            else:
                encoded = processor(
                    text=[prompt],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
            sample_encodings.append(encoded)

        # Collate textual tensors while preserving batch order
        pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(processor.tokenizer, "eos_token_id", 0)

        input_id_seqs = [enc["input_ids"].squeeze(0) for enc in sample_encodings]
        attention_masks = [enc["attention_mask"].squeeze(0) for enc in sample_encodings]
        max_len = max(seq.size(0) for seq in input_id_seqs)

        input_ids = torch.full(
            (num_samples, max_len),
            pad_token_id,
            dtype=input_id_seqs[0].dtype
        )
        attention_mask = torch.zeros(
            (num_samples, max_len),
            dtype=attention_masks[0].dtype
        )

        for idx, (seq, mask) in enumerate(zip(input_id_seqs, attention_masks)):
            length = seq.size(0)
            input_ids[idx, :length] = seq
            attention_mask[idx, :length] = mask

        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device)
        }

        # Collate optional tokenizer keys (e.g., position_ids) if present
        auxiliary_keys = set().union(*(enc.keys() for enc in sample_encodings))
        auxiliary_keys.discard("input_ids")
        auxiliary_keys.discard("attention_mask")
        auxiliary_keys.discard("pixel_values")

        for key in auxiliary_keys:
            values = []
            for enc in sample_encodings:
                if key in enc:
                    values.append(enc[key].squeeze(0))
                else:
                    values.append(None)

            reference = next((v for v in values if v is not None), None)
            if reference is None:
                continue

            if reference.dim() == 1:
                padded = torch.zeros((num_samples, max_len), dtype=reference.dtype)
                for idx, val in enumerate(values):
                    if val is None:
                        continue
                    padded[idx, :val.size(0)] = val
                inputs[key] = padded.to(device)
            else:
                stacked = []
                for val in values:
                    if val is None:
                        val = torch.zeros_like(reference)
                    stacked.append(val.unsqueeze(0))
                inputs[key] = torch.cat(stacked, dim=0).to(device)

        # Handle pixel values while keeping alignment with batch
        pixel_values = [enc.get("pixel_values", None) for enc in sample_encodings]
        if any(pv is not None for pv in pixel_values):
            first_pixel = next(pv for pv in pixel_values if pv is not None)
            pixel_shape = first_pixel.shape[1:]
            pixel_dtype = first_pixel.dtype
            stacked_pixels = torch.zeros(
                (num_samples,) + pixel_shape,
                dtype=pixel_dtype
            )
            for idx, pv in enumerate(pixel_values):
                if pv is not None:
                    stacked_pixels[idx] = pv.squeeze(0)
            inputs["pixel_values"] = stacked_pixels.to(device)

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

    def _ensure_image_batch(self, images: Union[torch.Tensor, List]) -> Optional[torch.Tensor]:
        """Normalize different image container types into a batched tensor.

        Args:
            images: Either a tensor or a list of tensors/PIL images/paths.

        Returns:
            Torch tensor with shape (B, C, H, W) or None if conversion fails.
        """
        if images is None:
            return None

        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                return images.unsqueeze(0)
            return images

        # Handle list/tuple inputs
        if isinstance(images, (list, tuple)):
            tensors = []
            for img in images:
                if img is None:
                    continue
                if isinstance(img, torch.Tensor):
                    tensor = img.unsqueeze(0) if img.dim() == 3 else img
                else:
                    pil = self._convert_to_pil(img)
                    if pil is None:
                        continue
                    tensor = self._process_image_to_tensor(pil)
                    if tensor is None:
                        continue
                tensors.append(tensor)

            if not tensors:
                return None

            images_tensor = torch.cat(tensors, dim=0)
            return images_tensor

        # Fallback: try converting single item
        pil_image = self._convert_to_pil(images)
        if pil_image is None:
            return None
        return self._process_image_to_tensor(pil_image)

    def _process_image_to_tensor(self, pil_image):
        """Apply the model's vision preprocessing pipeline to a PIL image."""
        if pil_image is None:
            return None

        processor = getattr(self.base_vl, "processor", None)
        image_processor = getattr(self.base_vl, "image_processor", None)

        try:
            if self.base_vl.model_type in {"llava", "blip2"} and processor is not None:
                processed = processor(images=pil_image, return_tensors="pt")
                if "pixel_values" in processed:
                    return processed["pixel_values"]
                if hasattr(processor, "image_processor"):
                    tmp = processor.image_processor(images=pil_image, return_tensors="pt")
                    if "pixel_values" in tmp:
                        return tmp["pixel_values"]

            if image_processor is not None:
                processed = image_processor(images=pil_image, return_tensors="pt")
                if "pixel_values" in processed:
                    return processed["pixel_values"]
        except Exception:
            pass

        # Conservative fallback: resize + ToTensor
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        return transform(pil_image).unsqueeze(0)

    def _scatter_audio_tokens(self, audio_tokens: torch.Tensor, indices, batch_size: int) -> torch.Tensor:
        """Place encoded audio tokens back into their original batch positions."""
        if batch_size is None:
            return audio_tokens

        if audio_tokens.dim() == 2:
            audio_tokens = audio_tokens.unsqueeze(1)

        device = audio_tokens.device
        dtype = audio_tokens.dtype
        full = torch.zeros((batch_size,) + audio_tokens.shape[1:], device=device, dtype=dtype)

        for pos, token in zip(indices, audio_tokens):
            full[pos] = token

        return full

    def _scatter_transcripts(self, transcripts, indices, batch_size: int):
        if batch_size is None or transcripts is None:
            return transcripts

        full = [None] * batch_size
        for pos, tr in zip(indices, transcripts):
            full[pos] = tr
        return full

    def _sanitize_input_ids_for_base(self, input_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if input_ids is None:
            return None
        if not torch.is_tensor(input_ids):
            return input_ids
        original_vocab = getattr(self, "original_vocab_size", None)
        if original_vocab is None:
            return input_ids
        mask = input_ids >= original_vocab
        if not mask.any():
            return input_ids
        sanitized = input_ids.clone()
        tokenizer = getattr(self.base_vl, "tokenizer", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = 0
        sanitized[mask] = pad_id
        return sanitized

    def sanitize_input_ids_for_base(self, input_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return self._sanitize_input_ids_for_base(input_ids)

    def sanitize_labels_for_base(self, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Mask out audio-token targets so the base model never sees out-of-vocab labels."""
        if labels is None or not torch.is_tensor(labels):
            return labels

        original_vocab = getattr(self, "original_vocab_size", None)
        if original_vocab is None:
            return labels

        sanitized = labels.clone()
        mask = (sanitized >= original_vocab) & (sanitized != -100)
        if mask.any():
            sanitized[mask] = -100
        return sanitized

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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        audio_tokens: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        gate: float = 1.0,
        fusion_layer: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through SAFE model."""
        # For BLIP2/LLaVA models, fuse audio by prefixing projected tokens
        if self.base_vl.model_type in ["blip2", "llava"]:
            pixel_values = kwargs.pop("pixel_values", None)
            filtered_kwargs = kwargs

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            inputs_embeds = self.get_input_embeddings(input_ids)
            if audio_tokens is not None and gate > 0.0:
                audio_tokens = audio_tokens.to(inputs_embeds.dtype)
                if gate != 1.0:
                    audio_tokens = audio_tokens * gate

                audio_mask = torch.ones(
                    audio_tokens.size(0),
                    audio_tokens.size(1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                inputs_embeds = torch.cat([audio_tokens, inputs_embeds], dim=1)
                attention_mask = torch.cat([audio_mask, attention_mask], dim=1)

                if labels is not None:
                    audio_label_pad = torch.full(
                        (labels.size(0), audio_tokens.size(1)),
                        -100,
                        dtype=labels.dtype,
                        device=labels.device,
                    )
                    labels = torch.cat([audio_label_pad, labels], dim=1)

            model_inputs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "labels": labels,
                **filtered_kwargs,
            }

            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values

            outputs = self.base_vl.llm(**model_inputs)
            logits = outputs.logits
            loss = outputs.loss if labels is not None else None
            return {"logits": logits, "loss": loss, "hidden_states": None}
        
        resolved_input_ids = input_ids if input_ids is not None else kwargs.pop("input_ids", None)
        inputs_embeds_kw = kwargs.pop("inputs_embeds", None)

        if resolved_input_ids is None and inputs_embeds_kw is None:
            raise ValueError("SAFEModel.forward requires input_ids or inputs_embeds")

        # For other model types, use custom implementation
        if audio_tokens is None or gate == 0.0:
            has_audio_tokens = (
                resolved_input_ids is not None
                and self.original_vocab_size is not None
                and (resolved_input_ids >= self.original_vocab_size).any()
            )

            if not has_audio_tokens:
                sanitized_ids = self._sanitize_input_ids_for_base(resolved_input_ids)
                base_inputs = {
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "vision_features": vision_features,
                }
                if resolved_input_ids is not None:
                    base_inputs["input_ids"] = sanitized_ids
                else:
                    base_inputs["inputs_embeds"] = inputs_embeds_kw
                return self.base_vl(**base_inputs, **kwargs)
            else:
                if resolved_input_ids is not None:
                    inputs_embeds = self.get_input_embeddings(resolved_input_ids)
                else:
                    inputs_embeds = inputs_embeds_kw
                base_inputs = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "vision_features": vision_features,
                }
                return self.base_vl(**base_inputs, **kwargs)

        # Get token embeddings using our custom method
        if resolved_input_ids is not None:
            inputs_embeds = self.get_input_embeddings(resolved_input_ids)
        else:
            inputs_embeds = inputs_embeds_kw
        
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
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                if resolved_input_ids is not None:
                    base_inputs["input_ids"] = self._sanitize_input_ids_for_base(resolved_input_ids)
                else:
                    base_inputs["inputs_embeds"] = inputs_embeds
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
            
            # Ensure shapes match after flattening
            flat_logits = shift_logits.view(-1, shift_logits.size(-1)).float()
            flat_labels = shift_labels.view(-1)
            
            # Only compute loss on valid tokens (ignore -100 padding)
            if flat_logits.size(0) != flat_labels.size(0):
                min_size = min(flat_logits.size(0), flat_labels.size(0))
                flat_logits = flat_logits[:min_size]
                flat_labels = flat_labels[:min_size]
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(flat_logits, flat_labels)
        
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
        audio_tokens: Optional[torch.Tensor] = None,
        **generation_kwargs
    ) -> Union[str, torch.Tensor]:
        """Generate text response given multimodal inputs."""
        # Low-level API: direct tensor inputs
        if input_ids is not None:
            base_inputs = {**generation_kwargs}
            if audio_tokens is not None and gate > 0.0:
                embeds = self.get_input_embeddings(input_ids)
                audio_tokens = audio_tokens.to(embeds.dtype)
                if gate != 1.0:
                    audio_tokens = audio_tokens * gate
                embeds = torch.cat([audio_tokens, embeds], dim=1)

                if attention_mask is None:
                    attention_mask = torch.ones(
                        input_ids.size(0),
                        input_ids.size(1),
                        dtype=torch.long,
                        device=input_ids.device,
                    )

                audio_mask = torch.ones(
                    audio_tokens.size(0),
                    audio_tokens.size(1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([audio_mask, attention_mask], dim=1)

                base_inputs["inputs_embeds"] = embeds
            else:
                base_inputs["input_ids"] = input_ids

            if attention_mask is not None:
                base_inputs["attention_mask"] = attention_mask
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

        audio_tokens_tensor = inputs.pop("audio_tokens", None)
        labels = inputs.pop("labels", None)
        if labels is not None:
            inputs["labels"] = labels  # keep alignment for recursion safety

        return self.generate(
            text=None,
            max_length=max_length,
            gate=gate,
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            audio_tokens=audio_tokens_tensor,
            **generation_kwargs,
        )
    
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
    
