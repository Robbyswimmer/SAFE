import torch
import torch.nn as nn
from collections import Counter
from typing import Any, Optional, Dict, List, Sequence, Union, Tuple
from .base_vl import BaseVLModel
from .audio_encoders import CLAPAudioEncoder, WhisperAudioEncoder, MultiModalAudioEncoder
from .projectors import AudioProjector, AdaptiveAudioProjector
from .fusion_adapter import LoRAFusionAdapter, MultiLayerFusionAdapter, GatedFusionAdapter
from .layer_hooks import LayerHookManager


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
        print(f"[SAFE] Initializing BaseVLModel (LLM: {llm_model_name}, Vision: {vision_model_name})...", flush=True)
        import sys
        sys.stdout.flush()
        self.base_vl = BaseVLModel(
            llm_model_name=llm_model_name,
            vision_model_name=vision_model_name,
            llm_hidden_size=llm_hidden_size,
            freeze_vision=freeze_base_vl,
            freeze_llm=freeze_base_vl
        )
        print(f"[SAFE] ✓ BaseVLModel initialized", flush=True)
        sys.stdout.flush()
        
        # Initialize audio encoder
        print(f"[SAFE] Initializing audio encoder ({audio_encoder_type})...", flush=True)
        sys.stdout.flush()
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
        print(f"[SAFE] ✓ Audio encoder initialized (embed_dim={audio_embed_dim})", flush=True)
        sys.stdout.flush()
        
        # Initialize audio projector
        print(f"[SAFE] Initializing audio projector ({projector_type})...", flush=True)
        sys.stdout.flush()
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
        print(f"[SAFE] ✓ Audio projector initialized", flush=True)
        sys.stdout.flush()
        
        # Initialize fusion adapter
        print(f"[SAFE] Initializing fusion adapter ({fusion_type})...", flush=True)
        sys.stdout.flush()
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
        print(f"[SAFE] ✓ Fusion adapter initialized", flush=True)
        sys.stdout.flush()

        self.enable_midlayer_fusion = (fusion_type == "multilayer")

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

        print(f"[SAFE] ✓ SAFEModel.__init__() complete", flush=True)
        sys.stdout.flush()

        self.debug_logging = False
        
        # Gate default and setter for warmup
        self._default_gate = 1.0
        self._audio_silence_threshold = 1e-4

    def set_gate(self, value: float) -> None:
        """Set default gate value for fusion (useful for warmup)."""
        self._default_gate = float(value)
    
    def set_gate_warmup(self, global_step: int, warmup_steps: int = 2000) -> None:
        """Set gate using warmup schedule: 0 → 1 over warmup_steps."""
        gate = min(1.0, global_step / warmup_steps)
        self.set_gate(gate)

    def set_debug_logging(self, enabled: bool) -> None:
        """Enable or disable verbose debugging across SAFE components."""

        self.debug_logging = bool(enabled)

        # Enable projector debugging
        if hasattr(self.audio_projector, "set_debug_logging"):
            self.audio_projector.set_debug_logging(self.debug_logging)
        elif hasattr(self.audio_projector, "debug_logging"):
            self.audio_projector.debug_logging = self.debug_logging

        if hasattr(self.audio_encoder, "set_debug_logging") and not self.debug_logging:
            # Reset waveform logging counter when disabling global debug
            self.audio_encoder.set_debug_logging(False)

        if hasattr(self.fusion_adapter, "set_debug_logging"):
            self.fusion_adapter.set_debug_logging(self.debug_logging)

    def configure_audio_debug(
        self,
        waveform_stats: bool = False,
        waveform_log_limit: int = 5,
    ) -> None:
        """Configure waveform statistics logging for the audio encoder."""

        if hasattr(self.audio_encoder, "set_debug_logging"):
            self.audio_encoder.set_debug_logging(waveform_stats, waveform_log_limit)

    def configure_attention_probe(self, enabled: bool, log_limit: int = 5) -> None:
        if hasattr(self.fusion_adapter, "configure_attention_probe"):
            self.fusion_adapter.configure_attention_probe(enabled, log_limit)

    def get_last_attention_summary(self) -> Optional[dict]:
        return getattr(self.fusion_adapter, "last_attention_summary", None)
    
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
        target_dtype = base_embeddings.weight.dtype
        embeddings = torch.zeros(
            batch_size, seq_len, actual_hidden_size,
            dtype=target_dtype,
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
            # Ensure audio embeddings match target dtype
            audio_embeds = audio_embeds.to(dtype=target_dtype)
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
            audio: Audio input - waveform tensor for encoding
            num_tokens: Optional override for number of tokens

        Returns:
            Tuple of (audio_tokens, transcripts)
        """
        # Log first few audio encoding calls
        if not hasattr(self, '_encode_audio_count'):
            self._encode_audio_count = 0

        if self._encode_audio_count < 3:
            audio_type = type(audio).__name__
            if isinstance(audio, list):
                audio_info = f"list[{len(audio)}], first_type={type(audio[0]).__name__ if audio else 'empty'}"
            elif isinstance(audio, torch.Tensor):
                audio_info = f"Tensor{list(audio.shape)}"
            else:
                audio_info = audio_type
            print(f"[AudioEncode] Call {self._encode_audio_count + 1}: input={audio_info}, encoder={self.audio_encoder_type}", flush=True)
            self._encode_audio_count += 1

        # Extract audio features from waveform
        if self.audio_encoder_type in ["clap", "multimodal"]:
            audio_features, transcripts = self.audio_encoder(audio), None
        else:  # whisper
            audio_features, transcripts = self.audio_encoder(audio)

        # Establish consistent dtype/device pipeline
        embedding_weight = self.base_vl.llm.get_input_embeddings().weight
        target_dtype = embedding_weight.dtype
        target_device = embedding_weight.device
        
        # Stage 1: Convert to tensor and ensure finite values
        if not isinstance(audio_features, torch.Tensor):
            audio_features = torch.tensor(audio_features, dtype=torch.float32)
        else:
            audio_features = audio_features.float()  # Start with float32 for numerical stability
        
        # Clean NaNs/Infs early in pipeline
        audio_features = torch.nan_to_num(audio_features, nan=0.0, posinf=1.0, neginf=-1.0)
        

        # Stage 2: Projector processing with consistent dtype
        projector_dtype = next(self.audio_projector.parameters()).dtype
        projector_device = next(self.audio_projector.parameters()).device
        
        # Convert to projector's dtype/device
        audio_features = audio_features.to(device=projector_device, dtype=projector_dtype)
        
        # Handle dimensionality for standard projector
        if self.projector_type != "adaptive":
            if audio_features.dim() > 2:
                audio_features = audio_features.mean(dim=1)
            elif audio_features.dim() == 1:
                audio_features = audio_features.unsqueeze(0)
        

        # Apply projector with target dtype for LM compatibility
        if self.projector_type == "adaptive":
            audio_tokens = self.audio_projector(audio_features, num_tokens=num_tokens, out_dtype=target_dtype)
        else:
            audio_tokens = self.audio_projector(audio_features, out_dtype=target_dtype)

        # Stage 3: Final dtype/device conversion with validation
        # Ensure tokens are finite before final conversion
        if not torch.isfinite(audio_tokens).all():
            audio_tokens = torch.nan_to_num(audio_tokens, nan=0.0, posinf=0.1, neginf=-0.1)

        # Convert to target dtype/device
        audio_tokens = audio_tokens.to(device=target_device, dtype=target_dtype)

        # Final validation and logging
        if self._encode_audio_count <= 3:
            print(f"[AudioEncode] Output: tokens.shape={audio_tokens.shape}, dtype={audio_tokens.dtype}, device={audio_tokens.device}", flush=True)

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
        
        non_empty_answers = [a for a in answer_list if a and str(a).strip()]

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
            start = max_length - length
            padded_ids[i, start:] = seq
            padded_attention[i, start:] = mask
            padded_labels[i, start:] = label

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
        num_audio_tokens: Optional[int] = None,
        training_mode: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for multimodal processing.
        
        Args:
            text: Input text
            images: Optional images
            audio: Optional audio
            answers: Optional answers for supervised training (only used if training_mode=True)
            device: Target device
            include_audio_tokens: Whether to include audio token placeholders
            num_audio_tokens: Override for number of audio tokens
            training_mode: If True, apply answers for training. If False, ignore answers for inference.

        Returns:
            Dictionary with input_ids, attention_mask, audio_tokens, etc.
        """
        # Ensure tokenizer padding configuration remains correct before encoding
        if self.base_vl.ensure_left_padding():
            print("[SAFEModel] Re-applied left padding configuration before multimodal prep", flush=True)

        # Removed excessive PrepDebug logging

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

            raw_levels_by_batch: Optional[List[Optional[float]]] = None
            if isinstance(audio, torch.Tensor) and audio.dim() >= 2:
                flattened = audio.reshape(audio.shape[0], -1)
                raw_levels_by_batch = [float(flat.abs().max().item()) for flat in flattened]
            elif isinstance(audio, list):
                raw_levels_by_batch = []
                for item in audio:
                    waveform = self._extract_waveform_from_input(item)
                    if waveform is None or waveform.numel() == 0:
                        raw_levels_by_batch.append(None)
                    else:
                        raw_levels_by_batch.append(float(torch.abs(waveform).max().item()))

            if isinstance(audio, list) and audio_indices is not None:
                audio_tokens = self._scatter_audio_tokens(audio_tokens, audio_indices, batch_size)
                if transcripts is not None:
                    transcripts = self._scatter_transcripts(transcripts, audio_indices, batch_size)

                if raw_levels_by_batch is not None:
                    scattered_levels = [None] * batch_size
                    for storage_index, level in zip(audio_indices, raw_levels_by_batch):
                        scattered_levels[int(storage_index)] = level
                    raw_levels_by_batch = scattered_levels

            # Gentle finite guards and build audio attention mask
            if torch.is_tensor(audio_tokens):
                assert torch.isfinite(audio_tokens).all(), "Non-finite audio_tokens after encode_audio"

                audio_tokens = torch.nan_to_num(audio_tokens, nan=0.0, posinf=0.0, neginf=0.0)

                B, T, _ = audio_tokens.shape
                audio_attention_mask = torch.ones(
                    B,
                    T,
                    dtype=torch.long,
                    device=audio_tokens.device,
                )

                token_absmax = audio_tokens.abs().amax(dim=(1, 2))
                per_sample_levels: List[float] = []
                for idx in range(B):
                    raw_level = None
                    if raw_levels_by_batch is not None and idx < len(raw_levels_by_batch):
                        raw_level = raw_levels_by_batch[idx]
                    if raw_level is None:
                        raw_level = float(token_absmax[idx].item())
                    per_sample_levels.append(raw_level)

                level_tensor = torch.tensor(
                    per_sample_levels,
                    dtype=audio_tokens.dtype,
                    device=audio_tokens.device,
                )
                silent = level_tensor < self._audio_silence_threshold
                if silent.any():
                    audio_attention_mask[silent] = 0
                    if getattr(self, "debug_logging", False):
                        silent_count = int(silent.sum().item())
                        print(
                            f"[AudioMask] Masked {silent_count}/{B} samples (threshold={self._audio_silence_threshold:.1e})",
                            flush=True,
                        )
                elif getattr(self, "debug_logging", False):
                    preview = per_sample_levels[: min(B, 3)]
                    print(f"[AudioMask] Audio levels OK (sample max={preview})", flush=True)

                result["audio_attention_mask"] = audio_attention_mask

            audio_tokens = audio_tokens.to(device)
            result["audio_tokens"] = audio_tokens
            if transcripts is not None:
                result["transcripts"] = transcripts

        # Apply ground truth answers ONLY during training mode to prevent gold answer leakage
        if training_mode and answers is not None and "input_ids" in result and "attention_mask" in result:
            dev = result["input_ids"].device if torch.is_tensor(result["input_ids"]) else torch.device(device)
            self._apply_answers_to_inputs(result, answers, dev)
        elif "labels" not in result and "input_ids" in result:
            # Default labels mirror the input ids (for inference or when no answers provided)
            result["labels"] = result["input_ids"].clone()
            
        # Add contamination detection
        if not training_mode and answers is not None:
            print("[WARNING] Answers provided in non-training mode - they will be ignored to prevent contamination", flush=True)

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

        # Removed excessive debug logging - enable only when debugging LLaVA issues
        
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
        
        # Build simple prompts directly (chat templates are broken/not configured)
        prompts = []
        image_token = getattr(processor, 'image_token', '<image>')

        for i, question in enumerate(texts):
            # Determine if this is audio-only (no image) - for audio captioning vs VQA
            # Audio captioning needs full descriptions, VQA needs short answers
            has_image = i < len(pil_images) and pil_images[i] is not None

            if has_image:
                # VQA task: Add instruction for short-form answers (critical for VQA accuracy)
                instruction = "Answer in one word or a number."
                full_question = f"{instruction} Question: {question}"
                prompt = f"USER: {image_token}\n{full_question} ASSISTANT:"
            else:
                # Audio-only task: No short-answer instruction (audio captioning needs descriptions)
                # Format: USER: Question: <question> ASSISTANT:
                prompt = f"USER: Question: {question} ASSISTANT:"

            prompts.append(prompt)
        
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
        tokenizer = getattr(processor, "tokenizer", None)
        tokenizer_max_len = None
        if tokenizer is not None:
            tokenizer_max_len = getattr(tokenizer, "model_max_length", None)
        if not tokenizer_max_len and getattr(self.base_vl, "tokenizer", None) is not None:
            tokenizer_max_len = getattr(self.base_vl.tokenizer, "model_max_length", None)
        if tokenizer_max_len and tokenizer_max_len > 0:
            if max_len > tokenizer_max_len:
                print(
                    f"[LLaVAPrep] Clamping padded sequence length from {max_len} to tokenizer max {tokenizer_max_len}",
                    flush=True,
                )
            max_len = min(max_len, tokenizer_max_len)
        max_len = max(1, int(max_len))

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
            if length > max_len:
                seq = seq[-max_len:]
                mask = mask[-max_len:]
                length = max_len
            start = max_len - length
            input_ids[idx, start:] = seq
            attention_mask[idx, start:] = mask

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
                    length = val.size(0)
                    if length > max_len:
                        val = val[-max_len:]
                        length = max_len
                    start = max_len - length
                    padded[idx, start:] = val
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
                print(f"WARNING: No <image> token (ID: {image_token_id}) found in input_ids for LLaVA!", flush=True)
                print(f"Prompt sample: {prompts[0][:200]}...", flush=True)
            
            if "pixel_values" not in inputs or inputs["pixel_values"] is None:
                print(f"WARNING: pixel_values missing for LLaVA with images!", flush=True)
        
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
        
        # Ensure we have a tensor of indices on the correct device
        if isinstance(indices, torch.Tensor):
            index_tensor = indices.to(device=device, dtype=torch.long)
        else:
            index_tensor = torch.tensor(list(indices), device=device, dtype=torch.long)

        # Sanitize tokens but keep gradient connectivity
        safe_tokens = torch.nan_to_num(audio_tokens, nan=0.0, posinf=1e4, neginf=-1e4)

        # Scatter the tokens back into their original batch slots using a differentiable op
        full = torch.zeros((batch_size,) + safe_tokens.shape[1:], device=device, dtype=dtype)
        full.index_copy_(0, index_tensor, safe_tokens)

        return full

    def _scatter_transcripts(self, transcripts, indices, batch_size: int):
        if batch_size is None or transcripts is None:
            return transcripts

        full = [None] * batch_size
        for pos, tr in zip(indices, transcripts):
            full[pos] = tr
        return full

    def _extract_waveform_from_input(self, audio_input: Any) -> Optional[torch.Tensor]:
        """Attempt to recover a waveform tensor from arbitrary audio inputs."""

        if isinstance(audio_input, torch.Tensor):
            return audio_input

        if isinstance(audio_input, tuple) and audio_input:
            candidate = audio_input[0]
            if isinstance(candidate, torch.Tensor):
                return candidate

        if isinstance(audio_input, dict):
            candidate = audio_input.get("waveform")
            if isinstance(candidate, torch.Tensor):
                return candidate

        return None

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
        gate: Optional[float] = None,
        fusion_layer: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through SAFE model."""
        # Use default gate if none provided
        if gate is None:
            gate = self._default_gate
        # For BLIP2/LLaVA models, fuse audio by prefixing projected tokens
        if self.base_vl.model_type in ["blip2", "llava"]:
            pixel_values = kwargs.pop("pixel_values", None)
            audio_attention_mask = kwargs.pop("audio_attention_mask", None)
            filtered_kwargs = kwargs

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            # LLaVA requires <image> tokens to accompany pixel inputs; drop them if missing
            if (
                self.base_vl.model_type == "llava"
                and pixel_values is not None
            ):
                has_image_tokens = False
                try:
                    image_token_id = self._get_image_token_id()
                    if input_ids is not None:
                        has_image_tokens = (input_ids == image_token_id).any()
                    else:
                        has_image_tokens = True  # Fallback when we cannot inspect tokens directly
                except Exception:
                    has_image_tokens = True

                if not has_image_tokens:
                    if self.training:
                        print(
                            "[SAFEModel] Dropping pixel_values during training because "
                            "no <image> tokens were found in the batch.",
                            flush=True,
                        )
                    else:
                        print(
                            "[SAFEModel] Warning: pixel_values provided without <image> tokens. "
                            "Removing pixel_values to avoid LLaVA placeholder mismatch during inference.",
                            flush=True,
                        )
                    pixel_values = None

            # ==================== VL PASSTHROUGH CHECK ====================
            # If no audio is present, use true VL passthrough:
            # Call base model with input_ids directly to avoid embedding contamination
            # Gate value is irrelevant when there's no audio to fuse
            no_audio = (audio_tokens is None or audio_tokens.numel() == 0)

            # DEBUG: Log passthrough decision
            if self.debug_logging:
                print(
                    f"[PassthroughDebug-forward] audio_tokens type: {type(audio_tokens)}, is None: {audio_tokens is None}, numel: {audio_tokens.numel() if audio_tokens is not None else 'N/A'}, no_audio: {no_audio}, gate: {gate}",
                    flush=True,
                )

            if no_audio:
                # TRUE VL PASSTHROUGH: Use base embeddings + pixel_values (matches working fusion path)
                # Get embeddings from BASE model's embedding layer (not custom get_input_embeddings)
                # This avoids contamination while using the proven working path
                base_embeddings_layer = self.base_vl.llm.get_input_embeddings()
                inputs_embeds = base_embeddings_layer(input_ids)  # Clean base embeddings, no sanitization

                # Ensure correct dtype
                base_dtype = next(self.base_vl.llm.parameters()).dtype
                inputs_embeds = inputs_embeds.to(base_dtype)

                # Use same format as fusion path: inputs_embeds + pixel_values
                base_inputs = {
                    "inputs_embeds": inputs_embeds,  # Use embeddings (not input_ids) for proper vision merge
                    "attention_mask": attention_mask,
                    "labels": labels,
                    **filtered_kwargs,
                }
                if pixel_values is not None:
                    base_inputs["pixel_values"] = pixel_values

                # Call LlavaForConditionalGeneration with embeddings + vision (same as fusion path)
                outputs = self.base_vl.llm(**base_inputs)
                logits = outputs.logits
                loss = outputs.loss if labels is not None else None
                return {"logits": logits, "loss": loss, "hidden_states": None}
            # ==================== END VL PASSTHROUGH ====================

            # FUSION PATH: Convert to inputs_embeds only when audio fusion is needed
            inputs_embeds = self.get_input_embeddings(input_ids)

            # Determine base dtype from the language model weights
            base_dtype = next(self.base_vl.llm.parameters()).dtype

            # Ensure inputs_embeds matches base dtype
            inputs_embeds = inputs_embeds.to(base_dtype)

            supervised_mask = None
            if labels is not None:
                supervised_mask = (labels != -100).to(inputs_embeds.device)

            model_inputs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "labels": labels,
                **filtered_kwargs,
            }

            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values

            use_midlayer_hooks = (
                audio_tokens is not None
                and gate > 0.0
                and self.enable_midlayer_fusion
                and hasattr(self.fusion_adapter, "apply_fusion_at_layer")
            )

            language_model = None
            fusion_layers = None
            modality_tokens = None
            modality_masks = None

            if use_midlayer_hooks:
                assert torch.isfinite(audio_tokens).all(), "Non-finite audio_tokens before fusion"

                audio_tokens = audio_tokens.to(
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
                if audio_attention_mask is not None:
                    audio_attention_mask = audio_attention_mask.to(inputs_embeds.device)

                language_model = self._resolve_language_model(self.base_vl.llm)
                fusion_layers = self._resolve_fusion_layers()
                if not any(fusion_layers.values()):
                    use_midlayer_hooks = False
                else:
                    modality_tokens = {"audio": audio_tokens}
                    modality_masks = (
                        {"audio": audio_attention_mask}
                        if audio_attention_mask is not None
                        else None
                    )

            else:
                if audio_tokens is not None:
                    audio_tokens = audio_tokens.to(
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )
                if audio_attention_mask is not None:
                    audio_attention_mask = audio_attention_mask.to(inputs_embeds.device)

            def run_with_hooks(run_inputs: Dict[str, torch.Tensor]) -> Any:
                hook_manager = LayerHookManager(
                    model=language_model,
                    fusion_adapter=self.fusion_adapter,
                    fusion_layers=fusion_layers,
                )
                hook_manager.register_hooks(
                    modality_tokens=modality_tokens,
                    modality_masks=modality_masks,
                    gate={"audio": gate},
                    supervised_mask=supervised_mask,
                )
                try:
                    return self.base_vl.llm(**run_inputs)
                finally:
                    hook_manager.remove_hooks()

            def run_without_hooks(run_inputs: Dict[str, torch.Tensor]) -> Any:
                if (
                    audio_tokens is not None
                    and gate > 0.0
                    and not self.enable_midlayer_fusion
                ):
                    fused_embeds = self.fusion_adapter(
                        hidden_states=run_inputs["inputs_embeds"],
                        audio_tokens=audio_tokens,
                        attention_mask=audio_attention_mask,
                        gate=gate,
                        supervised_mask=supervised_mask,
                    )
                    updated_inputs = dict(run_inputs)
                    updated_inputs["inputs_embeds"] = fused_embeds
                    run_inputs = updated_inputs
                return self.base_vl.llm(**run_inputs)

            import sys
            sys.stdout.flush()

            try:
                if use_midlayer_hooks:
                    outputs = run_with_hooks(model_inputs)
                else:
                    outputs = run_without_hooks(model_inputs)
            except ValueError as exc:
                if (
                    self.base_vl.model_type == "llava"
                    and "Image features and image tokens do not match" in str(exc)
                    and "pixel_values" in model_inputs
                ):
                    print(
                        "[SAFEModel] Caught placeholder mismatch from LLaVA; retrying forward pass without pixel_values.",
                        flush=True,
                    )
                    retry_inputs = dict(model_inputs)
                    retry_inputs.pop("pixel_values", None)
                    if use_midlayer_hooks:
                        outputs = run_with_hooks(retry_inputs)
                    else:
                        outputs = run_without_hooks(retry_inputs)
                else:
                    raise
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
                result = self.base_vl(**base_inputs, **kwargs)
                return result
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
                result = self.base_vl(**base_inputs, **kwargs)
                return result

        # Get token embeddings using our custom method
        if resolved_input_ids is not None:
            inputs_embeds = self.get_input_embeddings(resolved_input_ids)
        else:
            inputs_embeds = inputs_embeds_kw
        
        # Forward through LLM with audio fusion (simplified implementation)
        if hasattr(self.base_vl.llm, 'transformer'):
            hidden_states = inputs_embeds
            supervised_mask = None
            if labels is not None:
                supervised_mask = (labels != -100).to(hidden_states.device)

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
                        active_fusion_layer=fusion_layer,
                        supervised_mask=supervised_mask,
                    )
                elif i == len(self.base_vl.llm.transformer.h) // 2:  # Mid-layer fusion
                    if self.fusion_type == "gated":
                        hidden_states, _ = self.fusion_adapter(
                            hidden_states=hidden_states,
                            audio_tokens=audio_tokens,
                            force_gate=gate if gate != 1.0 else None,
                            supervised_mask=supervised_mask,
                        )
                    else:
                        hidden_states = self.fusion_adapter(
                            hidden_states,
                            audio_tokens,
                            None,
                            gate,
                            supervised_mask=supervised_mask,
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

    def _resolve_language_model(self, llm: nn.Module) -> nn.Module:
        candidates = [
            getattr(llm, "language_model", None),
            getattr(llm, "model", None),
            getattr(llm, "decoder", None),
        ]

        for candidate in candidates:
            if candidate is not None:
                return candidate

        return llm

    def _resolve_fusion_layers(self) -> Dict[str, List[int]]:
        if hasattr(self.fusion_adapter, "fusion_layers"):
            return {
                modality: list(indices)
                for modality, indices in self.fusion_adapter.fusion_layers.items()
            }

        if hasattr(self.fusion_adapter, "fusion_layer_indices"):
            indices = self.fusion_adapter.fusion_layer_indices
            if isinstance(indices, dict):
                return {
                    modality: list(values)
                    for modality, values in indices.items()
                }
            return {"audio": list(indices)}

        return {"audio": []}
    
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
        audio_attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs
    ) -> Union[str, torch.Tensor]:
        """Generate text response given multimodal inputs."""
        # Low-level API: direct tensor inputs
        if input_ids is not None:
            # VL PASSTHROUGH CHECK for generation (same as forward)
            no_audio = (audio_tokens is None) or (audio_tokens is not None and audio_tokens.numel() == 0)

            # DEBUG: Log passthrough decision
            if self.debug_logging:
                print(
                    f"[PassthroughDebug-generate] audio_tokens type: {type(audio_tokens)}, is None: {audio_tokens is None}, numel: {audio_tokens.numel() if audio_tokens is not None else 'N/A'}, no_audio: {no_audio}",
                    flush=True,
                )

            if no_audio:
                # TRUE VL PASSTHROUGH: Use base_vl.llm.generate directly with input_ids
                # This bypasses SAFE's contaminated embedding layer entirely
                # Using inputs_embeds causes HF generate to return ONLY new tokens (breaks decoding)
                base_inputs = {
                    "input_ids": input_ids,
                    **generation_kwargs
                }
                if attention_mask is not None:
                    base_inputs["attention_mask"] = attention_mask
                if pixel_values is not None:
                    base_inputs["pixel_values"] = pixel_values

                # Generate with base model directly (no sanitization needed for VL)
                return self.base_vl.llm.generate(**base_inputs)

            # AUDIO PATH: Use custom embeddings and fusion (existing logic)
            base_inputs = {**generation_kwargs}
            sanitized_ids = self.sanitize_input_ids_for_base(input_ids)
            if sanitized_ids is not None:
                sanitized_ids = sanitized_ids.to(input_ids.device)

            token_source = input_ids
            if sanitized_ids is not None and input_ids is not None and not torch.equal(sanitized_ids, input_ids):
                token_source = input_ids
            elif sanitized_ids is not None:
                token_source = sanitized_ids

            base_dtype = next(self.base_vl.llm.parameters()).dtype
            embeds = self.get_input_embeddings(token_source).to(base_dtype)

            if sanitized_ids is not None:
                base_inputs["input_ids"] = sanitized_ids
            else:
                base_inputs["input_ids"] = input_ids

            base_inputs["inputs_embeds"] = embeds
            if attention_mask is not None:
                base_inputs["attention_mask"] = attention_mask
            if pixel_values is not None:
                base_inputs["pixel_values"] = pixel_values

            use_midlayer_hooks = (
                audio_tokens is not None
                and gate > 0.0
                and self.enable_midlayer_fusion
                and hasattr(self.fusion_adapter, "apply_fusion_at_layer")
            )

            language_model = None
            fusion_layers = None
            modality_tokens = None
            modality_masks = None

            if use_midlayer_hooks:
                assert torch.isfinite(audio_tokens).all(), "Non-finite audio_tokens before generation fusion"

                audio_tokens = audio_tokens.to(device=embeds.device, dtype=base_dtype)
                audio_attention = None
                if audio_attention_mask is not None:
                    audio_attention = audio_attention_mask.to(embeds.device)

                language_model = self._resolve_language_model(self.base_vl.llm)
                fusion_layers = self._resolve_fusion_layers()
                if not any(fusion_layers.values()):
                    use_midlayer_hooks = False
                else:
                    modality_tokens = {"audio": audio_tokens}
                    modality_masks = (
                        {"audio": audio_attention}
                        if audio_attention is not None
                        else None
                    )
            else:
                if audio_tokens is not None:
                    audio_tokens = audio_tokens.to(device=embeds.device, dtype=base_dtype)
                if audio_attention_mask is not None:
                    audio_attention_mask = audio_attention_mask.to(embeds.device)

            if use_midlayer_hooks:
                hook_manager = LayerHookManager(
                    model=language_model,
                    fusion_adapter=self.fusion_adapter,
                    fusion_layers=fusion_layers,
                )
                hook_manager.register_hooks(
                    modality_tokens=modality_tokens,
                    modality_masks=modality_masks,
                    gate={"audio": gate},
                )
                try:
                    return self.base_vl.llm.generate(**base_inputs)
                finally:
                    hook_manager.remove_hooks()

            if audio_tokens is not None and gate > 0.0 and not self.enable_midlayer_fusion:
                fused_embeds = self.fusion_adapter(
                    hidden_states=embeds,
                    audio_tokens=audio_tokens,
                    attention_mask=audio_attention_mask,
                    gate=gate,
                )
                base_inputs["inputs_embeds"] = fused_embeds

            try:
                return self.base_vl.llm.generate(**base_inputs)
            except ValueError as exc:
                if (
                    self.base_vl.model_type == "llava"
                    and "Image features and image tokens do not match" in str(exc)
                    and "pixel_values" in base_inputs
                ):
                    print(
                        "[SAFEModel] Generation fallback: removing pixel_values after LLaVA mismatch.",
                        flush=True,
                    )
                    retry_inputs = dict(base_inputs)
                    retry_inputs.pop("pixel_values", None)
                    if use_midlayer_hooks:
                        hook_manager = LayerHookManager(
                            model=language_model,
                            fusion_adapter=self.fusion_adapter,
                            fusion_layers=fusion_layers,
                        )
                        hook_manager.register_hooks(
                            modality_tokens=modality_tokens,
                            modality_masks=modality_masks,
                            gate={"audio": gate},
                        )
                        try:
                            return self.base_vl.llm.generate(**retry_inputs)
                        finally:
                            hook_manager.remove_hooks()
                    return self.base_vl.llm.generate(**retry_inputs)
                raise

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
        audio_attention_mask_tensor = inputs.pop("audio_attention_mask", None)
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
            audio_attention_mask=audio_attention_mask_tensor,
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
    
    def to_device(self, device):
        """Properly move all model components to specified device."""
        print(f"[SAFEModel] Moving all components to device: {device}", flush=True)
        
        # Use .to() instead of .cuda() for proper device management
        device = torch.device(device)
        
        # Move main module
        self.to(device)
        
        # Explicitly move all subcomponents with verification
        # Move all components to device
        self.base_vl = self.base_vl.to(device)
        
        # Determine target dtype from base model embeddings for consistency
        base_embeddings = self.base_vl.llm.get_input_embeddings()
        target_dtype = base_embeddings.weight.dtype
        
        self.audio_encoder = self.audio_encoder.to(device)
        self.audio_projector = self.audio_projector.to(device=device)  # Device only, keep fp32
        
        if hasattr(self, 'fusion_adapter') and self.fusion_adapter is not None:
            self.fusion_adapter = self.fusion_adapter.to(device=device)  # Device only, keep fp32

        if hasattr(self, 'audio_token_embeddings') and self.audio_token_embeddings is not None:
            # Ensure audio token embeddings match base model dtype
            self.audio_token_embeddings = self.audio_token_embeddings.to(device=device, dtype=target_dtype)

        # Final consistency check to ensure everything shares the target dtype
        self.ensure_dtype_consistency()
        
        print(f"Model moved to device: {device}", flush=True)
        return self
    
    def ensure_dtype_consistency(self):
        """Ensure all model components use consistent dtypes."""
        # Ensure dtype consistency
        base_embeddings = self.base_vl.llm.get_input_embeddings()
        target_dtype = base_embeddings.weight.dtype
        
        # Ensure audio components match target dtype
        if hasattr(self, 'audio_token_embeddings') and self.audio_token_embeddings is not None:
            current_dtype = self.audio_token_embeddings.weight.dtype
            if current_dtype != target_dtype:
                self.audio_token_embeddings = self.audio_token_embeddings.to(dtype=target_dtype)
        return self
    
    def verify_device_placement(self, expected_device):
        """Verify all components are on the expected device."""
        # Check device placement
        main_device = next(self.parameters()).device
        base_vl_device = next(self.base_vl.parameters()).device
        audio_enc_device = next(self.audio_encoder.parameters()).device
        proj_device = next(self.audio_projector.parameters()).device
        
        devices = [main_device, base_vl_device, audio_enc_device, proj_device]
        expected = torch.device(expected_device)
        
        all_correct = all(d == expected for d in devices)
        
        if not all_correct:
            component_names = ['main', 'base_vl', 'audio_encoder', 'audio_projector']
            for i, d in enumerate(devices):
                if d != expected:
                    print(f"ERROR: {component_names[i]} on wrong device: {d} (expected {expected})", flush=True)
            return False
        
        return True
    
