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
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.audio_embed_dim = audio_embed_dim
        self.llm_hidden_size = llm_hidden_size
        self.num_audio_tokens = num_audio_tokens
        
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
        
        # 2-layer MLP with improved stability
        self.projector = nn.Sequential(
            nn.Linear(audio_embed_dim, llm_hidden_size),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(llm_hidden_size, llm_hidden_size * num_audio_tokens),
            nn.Tanh()  # Soft bounding to prevent saturation
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(llm_hidden_size, eps=1e-6)
        
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
        
        # Force last linear of projector to zero init to prevent inf updates
        last = None
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                last = m
        if last is not None:
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)
    
    def set_debug_logging(self, enabled: bool, log_limit: int = 5) -> None:
        """Enable or disable projector debug logging."""
        self.debug_logging = bool(enabled)
        self._projector_log_limit = int(max(0, log_limit))
        self._projector_logs_emitted = 0

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Project audio features to LLM token space.
        
        Args:
            audio_features: (batch_size, audio_embed_dim) audio embeddings
            
        Returns:
            audio_tokens: (batch_size, num_audio_tokens, llm_hidden_size)
        """
        batch_size = audio_features.shape[0]
        
        # Normalize input for stability
        normalized_input = self.input_norm(audio_features)
        
        # Log input statistics for debugging
        if self.debug_logging and self._projector_logs_emitted < self._projector_log_limit:
            input_norm = normalized_input.norm(dim=-1).mean().item()
            input_max = normalized_input.abs().max().item()
            print(f"[ProjectorDebug] Input norm={input_norm:.6f}, max_abs={input_max:.6f}", flush=True)
            self._projector_logs_emitted += 1
        
        # Project through MLP with soft bounding
        projected = self.projector(normalized_input)  # (batch_size, llm_hidden_size * num_audio_tokens)

        # Scale to reasonable range (tanh outputs [-1,1], scale to match typical embedding ranges)
        scale_factor = 0.1  # Conservative scaling to prevent saturation
        projected = projected * scale_factor

        # Reshape to token format
        audio_tokens = projected.view(
            batch_size, 
            self.num_audio_tokens, 
            self.llm_hidden_size
        )
        
        # Apply output normalization per token
        audio_tokens = self.output_norm(audio_tokens)
        
        # Log output statistics for debugging
        if self.debug_logging and self._projector_logs_emitted < self._projector_log_limit:
            output_norm = audio_tokens.norm(dim=-1).mean().item()
            output_max = audio_tokens.abs().max().item()
            print(f"[ProjectorDebug] Output norm={output_norm:.6f}, max_abs={output_max:.6f}", flush=True)
            self._projector_logs_emitted += 1
        
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
        
        # Zero-init each generator's final linear to prevent inf updates
        for generator in self.token_generators.values():
            for m in generator.modules():
                if isinstance(m, nn.Linear):
                    # This is the final layer (since generator is Sequential with one Linear)
                    nn.init.zeros_(m.weight)
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
        num_tokens: Optional[int] = None
    ) -> torch.Tensor:
        """
        Project audio features with adaptive token count.
        
        Args:
            audio_features: (batch_size, audio_embed_dim)
            num_tokens: Fixed number of tokens (if None, predict automatically)
            
        Returns:
            audio_tokens: (batch_size, num_tokens, llm_hidden_size)
        """
        batch_size = audio_features.shape[0]
        
        # Normalize input for stability
        normalized_input = self.input_norm(audio_features)
        
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

        # Scale to reasonable range (tanh outputs [-1,1], scale to match typical embedding ranges)
        scale_factor = 0.1  # Conservative scaling to prevent saturation
        projected = projected * scale_factor

        audio_tokens = projected.view(
            batch_size,
            most_common_tokens,
            self.llm_hidden_size
        )
        
        # Apply output normalization per token
        audio_tokens = self.output_norm(audio_tokens)
        
        # Log output statistics for debugging
        if self.debug_logging and self._projector_logs_emitted < self._projector_log_limit:
            output_norm = audio_tokens.norm(dim=-1).mean().item()
            output_max = audio_tokens.abs().max().item()
            print(f"[AdaptiveProjectorDebug] Output norm={output_norm:.6f}, max_abs={output_max:.6f}", flush=True)
            self._projector_logs_emitted += 1
        
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
