import torch
import torch.nn as nn
from typing import Optional


class AudioProjector(nn.Module):
    """
    Trainable projector that maps audio features to LLM token space.
    
    Architecture: 2-layer MLP that converts CLAP/audio features to d_model of LLM,
    emitting k audio tokens where k ∈ {0, 4, 8, 12}.
    """
    
    def __init__(
        self,
        audio_embed_dim: int,
        llm_hidden_size: int,
        num_audio_tokens: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        bottleneck_dim: Optional[int] = None  # New parameter for bottleneck
    ):
        super().__init__()

        self.audio_embed_dim = audio_embed_dim
        self.llm_hidden_size = llm_hidden_size
        self.num_audio_tokens = num_audio_tokens

        # Default bottleneck to 1024 if not specified (80% param reduction)
        if bottleneck_dim is None:
            bottleneck_dim = min(1024, llm_hidden_size // 4)
        self.bottleneck_dim = bottleneck_dim

        # Activation function
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Input normalization for stability
        self.input_norm = nn.LayerNorm(audio_embed_dim, eps=1e-6)

        # 2-layer MLP with bottleneck for parameter efficiency
        # Architecture: audio_embed_dim → bottleneck_dim → llm_hidden_size * num_audio_tokens
        self.projector = nn.Sequential(
            nn.Linear(audio_embed_dim, bottleneck_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, llm_hidden_size * num_audio_tokens),
            nn.Tanh()  # Soft bounding to prevent saturation
        )

        # Output normalization
        self.output_norm = nn.LayerNorm(llm_hidden_size, eps=1e-6)

        # Learnable scale to match LLM embedding magnitude
        # Initialize to 1.0 since LayerNorm already produces ~10-15 magnitude
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        # Debug logging
        self.debug_logging = False
        self._projector_log_limit = 5
        self._projector_logs_emitted = 0
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projector weights."""
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Force last linear of projector to tiny init to allow gradient flow
        last = None
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                last = m
        if last is not None:
            nn.init.normal_(last.weight, mean=0.0, std=1e-5)  # Safer tiny init
            if last.bias is not None:
                nn.init.zeros_(last.bias)
    
    def set_debug_logging(self, enabled: bool, log_limit: int = 5) -> None:
        """Enable or disable projector debug logging."""
        self.debug_logging = bool(enabled)
        self._projector_log_limit = int(max(0, log_limit))
        self._projector_logs_emitted = 0

    def forward(self, audio_features: torch.Tensor, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Project audio features to LLM token space.
        
        Args:
            audio_features: (batch_size, audio_embed_dim) audio embeddings
            out_dtype: Target dtype for output tokens (to match LM weights)
            
        Returns:
            audio_tokens: (batch_size, num_audio_tokens, llm_hidden_size)
        """
        batch_size = audio_features.shape[0]
        
        # Sanitize inputs and compute in fp32 for numerical stability
        x = torch.nan_to_num(audio_features, nan=0.0, posinf=0.0, neginf=0.0)
        if x.dtype != torch.float32:
            x = x.float()
        
        # Normalize input for stability
        normalized_input = self.input_norm(x)
        
        
        # Project through MLP with soft bounding
        projected = self.projector(normalized_input)  # (batch_size, llm_hidden_size * num_audio_tokens)

        # Phase 1.5 fix: Remove tanh*10.0 saturation, use LayerNorm instead
        # The projector already has tanh in the Sequential, adding clean normalization here

        # Reshape to token format
        audio_tokens = projected.view(
            batch_size,
            self.num_audio_tokens,
            self.llm_hidden_size
        )

        # Apply output normalization per token (centers around 0, variance 1)
        # This naturally aligns with LLM embedding distribution without saturation
        audio_tokens = self.output_norm(audio_tokens)

        # Apply learnable scale to match LLM embedding magnitude
        # Model learns optimal scale during training (initialized to 5.0)
        audio_tokens = audio_tokens * self.output_scale

        # Log embedding norms for debugging
        if self.debug_logging and self._projector_logs_emitted < self._projector_log_limit:
            with torch.no_grad():
                token_norms = torch.norm(audio_tokens, dim=-1).mean().item()
                print(f"[AudioProjector] Token norms (mean L2): {token_norms:.3f}, Scale: {self.output_scale.item():.3f}", flush=True)
                self._projector_logs_emitted += 1

        # Cast to requested/output dtype (the LM/base dtype)
        if out_dtype is not None:
            audio_tokens = audio_tokens.to(out_dtype)

        return audio_tokens


class AdaptiveAudioProjector(nn.Module):
    """
    Adaptive projector that can output variable number of tokens based on input complexity.
    """
    
    def __init__(
        self,
        audio_embed_dim: int,
        llm_hidden_size: int,
        max_audio_tokens: int = 12,
        min_audio_tokens: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.audio_embed_dim = audio_embed_dim
        self.llm_hidden_size = llm_hidden_size
        self.max_audio_tokens = max_audio_tokens
        self.min_audio_tokens = min_audio_tokens
        
        # Input normalization for stability
        self.input_norm = nn.LayerNorm(audio_embed_dim, eps=1e-6)
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(audio_embed_dim, llm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Token count predictor
        self.token_predictor = nn.Sequential(
            nn.Linear(llm_hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, max_audio_tokens - min_audio_tokens + 1)  # Predict offset from min
        )
        
        # Token generators for each possible count with soft bounding
        self.token_generators = nn.ModuleDict()
        for k in range(min_audio_tokens, max_audio_tokens + 1):
            self.token_generators[str(k)] = nn.Sequential(
                nn.Linear(llm_hidden_size, llm_hidden_size * k),
                nn.Tanh()  # Soft bounding to prevent saturation
            )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(llm_hidden_size, eps=1e-6)

        # Learnable scale to match LLM embedding magnitude
        # Initialize to 1.0 since LayerNorm already produces ~10-15 magnitude
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        # Debug logging
        self.debug_logging = False
        self._projector_log_limit = 5
        self._projector_logs_emitted = 0

        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Tiny-init each generator's final linear to allow gradient flow
        for generator in self.token_generators.values():
            for m in generator.modules():
                if isinstance(m, nn.Linear):
                    # This is the final layer (since generator is Sequential with one Linear)
                    nn.init.normal_(m.weight, mean=0.0, std=1e-5)  # Safer tiny init
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def set_debug_logging(self, enabled: bool, log_limit: int = 5) -> None:
        """Enable or disable projector debug logging."""
        self.debug_logging = bool(enabled)
        self._projector_log_limit = int(max(0, log_limit))
        self._projector_logs_emitted = 0

    def forward(
        self, 
        audio_features: torch.Tensor, 
        num_tokens: Optional[int] = None,
        out_dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Project audio features with adaptive token count.
        
        Args:
            audio_features: (batch_size, audio_embed_dim)
            num_tokens: Fixed number of tokens (if None, predict automatically)
            out_dtype: Target dtype for output tokens (to match LM weights)
            
        Returns:
            audio_tokens: (batch_size, num_tokens, llm_hidden_size)
        """
        batch_size = audio_features.shape[0]
        
        # Sanitize inputs and compute in fp32 for numerical stability
        x = torch.nan_to_num(audio_features, nan=0.0, posinf=0.0, neginf=0.0)
        if x.dtype != torch.float32:
            x = x.float()
        
        # Normalize input for stability
        normalized_input = self.input_norm(x)
        
        # Extract shared features
        features = self.feature_extractor(normalized_input)  # (batch_size, llm_hidden_size)
        
        if num_tokens is None:
            # Predict optimal token count
            token_logits = self.token_predictor(features)  # (batch_size, num_classes)
            token_probs = torch.softmax(token_logits, dim=-1)
            
            # Use expected value for differentiable token count
            token_weights = torch.arange(
                self.min_audio_tokens, 
                self.max_audio_tokens + 1, 
                device=features.device, 
                dtype=features.dtype
            )
            expected_tokens = torch.sum(token_probs * token_weights.unsqueeze(0), dim=-1)
            
            # Round to nearest integer for actual generation
            num_tokens = torch.round(expected_tokens).int()
        
        # Generate tokens for each sample
        if isinstance(num_tokens, int):
            # All samples use same token count
            num_tokens = torch.full((batch_size,), num_tokens, device=features.device)
        
        # For simplicity in batched processing, use the most common token count
        # In practice, you might want more sophisticated batching
        most_common_tokens = torch.mode(num_tokens).values.item()
        most_common_tokens = max(self.min_audio_tokens, 
                                min(self.max_audio_tokens, most_common_tokens))
        
        # Generate tokens with soft bounding
        generator = self.token_generators[str(most_common_tokens)]
        projected = generator(features)  # (batch_size, llm_hidden_size * k)

        # Phase 1.5 fix: Remove *10.0 saturation, rely on LayerNorm instead
        # Generator already has tanh, LayerNorm will align with LLM embedding distribution

        audio_tokens = projected.view(
            batch_size,
            most_common_tokens,
            self.llm_hidden_size
        )

        # Apply output normalization per token (centers around 0, variance 1)
        # This naturally aligns with LLM embedding distribution without saturation
        audio_tokens = self.output_norm(audio_tokens)

        # Apply learnable scale to match LLM embedding magnitude
        # Model learns optimal scale during training (initialized to 5.0)
        audio_tokens = audio_tokens * self.output_scale

        # Log embedding norms for debugging
        if self.debug_logging and self._projector_logs_emitted < self._projector_log_limit:
            with torch.no_grad():
                token_norms = torch.norm(audio_tokens, dim=-1).mean().item()
                print(f"[AdaptiveAudioProjector] Token norms (mean L2): {token_norms:.3f}, Scale: {self.output_scale.item():.3f}", flush=True)
                self._projector_logs_emitted += 1

        # Cast to requested/output dtype (the LM/base dtype)
        if out_dtype is not None:
            audio_tokens = audio_tokens.to(out_dtype)

        return audio_tokens


class VisionProjector(nn.Module):
    """
    Standard vision projector for the base VL model (for reference/comparison).
    """
    
    def __init__(
        self,
        vision_embed_dim: int,
        llm_hidden_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(vision_embed_dim, llm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to LLM space.
        
        Args:
            vision_features: (batch_size, seq_len, vision_embed_dim)
            
        Returns:
            projected_features: (batch_size, seq_len, llm_hidden_size)
        """
        return self.projector(vision_features)
