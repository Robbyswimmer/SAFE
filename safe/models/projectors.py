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
        
        # 2-layer MLP
        self.projector = nn.Sequential(
            nn.Linear(audio_embed_dim, llm_hidden_size),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(llm_hidden_size, llm_hidden_size * num_audio_tokens)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projector weights."""
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Project audio features to LLM token space.
        
        Args:
            audio_features: (batch_size, audio_embed_dim) audio embeddings
            
        Returns:
            audio_tokens: (batch_size, num_audio_tokens, llm_hidden_size)
        """
        batch_size = audio_features.shape[0]
        
        # Project through MLP
        projected = self.projector(audio_features)  # (batch_size, llm_hidden_size * num_audio_tokens)

        # Guard against numerical blow-ups producing NaNs/Infs
        torch.nan_to_num_(projected, nan=0.0, posinf=1e3, neginf=-1e3)
        projected = projected.clamp_(-1e3, 1e3)

        # Reshape to token format
        audio_tokens = projected.view(
            batch_size, 
            self.num_audio_tokens, 
            self.llm_hidden_size
        )
        
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
        
        # Token generators for each possible count
        self.token_generators = nn.ModuleDict()
        for k in range(min_audio_tokens, max_audio_tokens + 1):
            self.token_generators[str(k)] = nn.Linear(
                llm_hidden_size, 
                llm_hidden_size * k
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
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
        
        # Extract shared features
        features = self.feature_extractor(audio_features)  # (batch_size, llm_hidden_size)
        
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
        
        # Generate tokens
        generator = self.token_generators[str(most_common_tokens)]
        projected = generator(features)  # (batch_size, llm_hidden_size * k)

        torch.nan_to_num_(projected, nan=0.0, posinf=1e3, neginf=-1e3)
        projected = projected.clamp_(-1e3, 1e3)

        audio_tokens = projected.view(
            batch_size,
            most_common_tokens,
            self.llm_hidden_size
        )
        
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
