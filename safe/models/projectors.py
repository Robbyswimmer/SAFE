import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """
    SwiGLU activation as used in LLaMA, PaLM, etc.
    Provides better gradient flow than simple GELU/ReLU.

    SwiGLU(x) = (x @ W1) * SiLU(x @ W_gate)
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w_gate = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))


class AudioProjector(nn.Module):
    """
    Trainable projector that maps audio features to LLM token space.

    Architecture: 2-layer MLP that converts CLAP/audio features to d_model of LLM,
    emitting k audio tokens where k ∈ {0, 4, 8, 12}.

    Supports SwiGLU activation (use_swiglu=True) for better gradient flow,
    and learnable positional embeddings (use_positional_embedding=True) for
    temporal structure in audio tokens.
    """

    def __init__(
        self,
        audio_embed_dim: int,
        llm_hidden_size: int,
        num_audio_tokens: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        bottleneck_dim: Optional[int] = None,
        use_swiglu: bool = False,
        use_positional_embedding: bool = False,
        output_dim: Optional[int] = None,  # If set, output in this dim instead of llm_hidden_size
        disable_output_norm: bool = False,  # If True, skip output LayerNorm (for classification probing)
        disable_input_norm: bool = False,  # If True, skip input LayerNorm (for classification probing)
        disable_scale: bool = False,  # If True, skip learnable output_scale (for classification probing)
        identity_mode: bool = False,  # If True, skip ALL transformations - just reshape CLAP to tokens
    ):
        super().__init__()

        self.audio_embed_dim = audio_embed_dim
        self.llm_hidden_size = llm_hidden_size
        self.num_audio_tokens = num_audio_tokens
        self.use_swiglu = use_swiglu
        self.use_positional_embedding = use_positional_embedding
        self.disable_output_norm = disable_output_norm
        self.disable_input_norm = disable_input_norm
        self.disable_scale = disable_scale
        self.identity_mode = identity_mode

        # Output dimension: use output_dim if specified, otherwise llm_hidden_size
        # For KV-augment mode, output_dim=audio_embed_dim keeps tokens small (~4M vs 42M params)
        self.output_dim = output_dim if output_dim is not None else llm_hidden_size
        output_size = self.output_dim * num_audio_tokens

        # Default bottleneck to 2048 if not specified.
        # Previous 1024 was too aggressive (5x compression for LLaVA's 5120 hidden size),
        # creating an information bottleneck that choked gradient flow.
        # 2048 provides better capacity while still reducing parameters.
        if bottleneck_dim is None:
            bottleneck_dim = min(2048, self.output_dim // 2) if self.output_dim > 1024 else min(512, self.output_dim)
        self.bottleneck_dim = bottleneck_dim

        # Input normalization for stability
        self.input_norm = nn.LayerNorm(audio_embed_dim, eps=1e-6)

        if use_swiglu:
            # SwiGLU-based projector (better gradient flow, used in LLaMA/PaLM)
            self.projector = nn.Sequential(
                SwiGLU(audio_embed_dim, bottleneck_dim, bottleneck_dim),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, output_size),
            )
        else:
            # Standard MLP projector
            # Activation function
            if activation.lower() == "gelu":
                act_fn = nn.GELU()
            elif activation.lower() == "relu":
                act_fn = nn.ReLU()
            elif activation.lower() == "silu":
                act_fn = nn.SiLU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            self.activation = act_fn

            # 2-layer MLP with bottleneck for parameter efficiency
            self.projector = nn.Sequential(
                nn.Linear(audio_embed_dim, bottleneck_dim),
                act_fn,
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, output_size),
            )

        # Learnable positional embeddings for audio tokens
        # Helps the model understand temporal structure in audio
        if use_positional_embedding:
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, num_audio_tokens, self.output_dim)
            )
            # Initialize with small values for stability
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        else:
            self.pos_embedding = None

        # Output normalization (use output_dim, not llm_hidden_size)
        self.output_norm = nn.LayerNorm(self.output_dim, eps=1e-6)

        # Trainable scale applied after LayerNorm.
        # With LayerNorm, per-token L2 norm is typically ~sqrt(hidden_size), so
        # an init of 1.0 is a safe default; training can adjust as needed.
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
        
        # Use Xavier init on final layer for proper gradient flow
        # (Tiny-init at 1e-5 was killing audio signal strength)
        last = None
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                last = m
        if last is not None:
            nn.init.xavier_uniform_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def set_debug_logging(self, enabled: bool, log_limit: int = 5) -> None:
        """Enable or disable projector debug logging."""
        self.debug_logging = bool(enabled)
        self._projector_log_limit = int(max(0, log_limit))
        self._projector_logs_emitted = 0

    def get_diagnostics(
        self,
        encoder_outputs: torch.Tensor,
        projector_outputs: Optional[torch.Tensor] = None,
    ) -> "ProjectorDiagnostics":
        """
        Compute comprehensive diagnostics for projector behavior.

        Args:
            encoder_outputs: Raw CLAP embeddings (B, encoder_dim)
            projector_outputs: Optional pre-computed projector outputs.
                              If None, will compute via forward pass.

        Returns:
            ProjectorDiagnostics dataclass with all metrics
        """
        from .projector_diagnostics import ProjectorDiagnosticsComputer

        if projector_outputs is None:
            with torch.no_grad():
                projector_outputs = self.forward(encoder_outputs)

        computer = ProjectorDiagnosticsComputer(
            variance_threshold=1e-6,
            sensitivity_perturbation_scale=0.01,
            top_k_singular_values=10,
        )

        return computer.compute_all(
            encoder_outputs=encoder_outputs,
            projector_outputs=projector_outputs,
            projector=self,
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
        text_embeds_for_calib: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Project audio features to token space.

        Args:
            audio_features: (batch_size, audio_embed_dim) audio embeddings
            out_dtype: Target dtype for output tokens (to match LM weights)
            text_embeds_for_calib: (batch_size, seq_len, hidden_size) text embeddings for norm calibration

        Returns:
            audio_tokens: (batch_size, num_audio_tokens, output_dim)
        """
        batch_size = audio_features.shape[0]

        # Sanitize inputs and compute in fp32 for numerical stability
        x = torch.nan_to_num(audio_features, nan=0.0, posinf=0.0, neginf=0.0)
        if x.dtype != torch.float32:
            x = x.float()

        # IDENTITY MODE: Skip all transformations, just pass CLAP embeddings through
        # This is for debugging - if identity works but MLP doesn't, the MLP is the problem
        if self.identity_mode:
            # Just reshape to (batch, 1, audio_embed_dim) - no transformation
            audio_tokens = x.unsqueeze(1)  # (batch, 1, 512)
            if self.debug_logging and self._projector_logs_emitted < self._projector_log_limit:
                with torch.no_grad():
                    audio_norm = audio_tokens.norm(dim=-1).mean().item()
                    print(f"[AudioProjector] IDENTITY MODE: audio_norm={audio_norm:.2f} (raw CLAP)", flush=True)
                    self._projector_logs_emitted += 1
            if out_dtype is not None:
                audio_tokens = audio_tokens.to(out_dtype)
            return audio_tokens

        # Normalize input for stability (can be disabled for classification probing)
        if not self.disable_input_norm:
            x = self.input_norm(x)

        # Project through MLP
        projected = self.projector(x)  # (batch_size, output_dim * num_audio_tokens)

        # Reshape to token format
        audio_tokens = projected.view(
            batch_size,
            self.num_audio_tokens,
            self.output_dim
        )

        # Add positional embeddings if enabled
        # This helps the model understand temporal structure in audio tokens
        if self.pos_embedding is not None:
            audio_tokens = audio_tokens + self.pos_embedding

        # Apply output normalization per token (centers around 0, variance 1)
        # This naturally aligns with LLM embedding distribution without saturation
        # NOTE: LayerNorm can destroy magnitude information needed for classification!
        # Use disable_output_norm=True when training classification probes.
        if not self.disable_output_norm:
            audio_tokens = self.output_norm(audio_tokens)

        # Apply learnable scale to match LLM embedding magnitude.
        # Use warmup minimum that increases over training to prevent early collapse
        # while still forcing strong audio signal later.
        # Can be disabled for classification probing to match baseline MLP behavior.
        if not self.disable_scale:
            min_scale = getattr(self, '_scale_min', 0.5)  # Default 0.5, can be set externally
            clamped_scale = torch.clamp(self.output_scale, min_scale, 10.0)
            audio_tokens = audio_tokens * clamped_scale

        # Log embedding norms for debugging
        if self.debug_logging and self._projector_logs_emitted < self._projector_log_limit:
            with torch.no_grad():
                audio_norm = audio_tokens.norm(dim=-1).mean().item()
                scale_val = self.output_scale.item() if not self.disable_scale else 0.0
                print(f"[AudioProjector] audio_norm={audio_norm:.2f}, scale={scale_val:.3f} (disabled={self.disable_scale})", flush=True)
                self._projector_logs_emitted += 1

        # Cast to requested/output dtype (the LM/base dtype)
        if out_dtype is not None:
            audio_tokens = audio_tokens.to(out_dtype)

        return audio_tokens


class TokenSetProjector(nn.Module):
    """
    Project a set of modality tokens into LLM token space.

    Input:  (B, G, input_dim)  where G is number of encoder/group tokens
    Output: (B, num_tokens, output_dim)

    Uses learned queries to attend over encoder tokens and produce exactly
    `num_tokens` pooled tokens, then an MLP to map to output_dim.
    """

    def __init__(
        self,
        input_dim: int,
        num_tokens: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        bottleneck_dim: Optional[int] = None,
        use_positional_embedding: bool = False,
        output_dim: Optional[int] = None,
        disable_output_norm: bool = False,
        disable_input_norm: bool = False,
        disable_scale: bool = False,
        **_unused: object,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.num_tokens = int(num_tokens)
        self.output_dim = int(output_dim) if output_dim is not None else self.input_dim
        self.disable_output_norm = bool(disable_output_norm)
        self.disable_input_norm = bool(disable_input_norm)
        self.disable_scale = bool(disable_scale)

        if bottleneck_dim is None:
            bottleneck_dim = min(2048, self.output_dim // 2) if self.output_dim > 1024 else min(512, self.output_dim)
        self.bottleneck_dim = int(bottleneck_dim)

        if activation.lower() == "gelu":
            act_fn = nn.GELU()
        elif activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "silu":
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.key_norm = nn.LayerNorm(self.input_dim, eps=1e-6)
        self.query_norm = nn.LayerNorm(self.input_dim, eps=1e-6)

        # Learned queries that pool G encoder tokens down to num_tokens outputs
        self.queries = nn.Parameter(torch.zeros(1, self.num_tokens, self.input_dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

        # MLP that maps pooled tokens to output_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.bottleneck_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(self.bottleneck_dim, self.output_dim),
        )

        self.pos_embedding = None
        if use_positional_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, self.output_dim))
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        self.output_norm = nn.LayerNorm(self.output_dim, eps=1e-6)
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, token_features: torch.Tensor, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Args:
            token_features: (B, G, input_dim) or (B, input_dim)
        Returns:
            tokens: (B, num_tokens, output_dim)
        """
        if token_features.dim() == 2:
            token_features = token_features.unsqueeze(1)
        if token_features.dim() != 3:
            raise ValueError(f"token_features must be (B,G,D) or (B,D), got {tuple(token_features.shape)}")

        x = torch.nan_to_num(token_features, nan=0.0, posinf=0.0, neginf=0.0)
        if x.dtype != torch.float32:
            x = x.float()

        if not self.disable_input_norm:
            k = self.key_norm(x)
        else:
            k = x

        q = self.queries.expand(k.shape[0], -1, -1)
        q = self.query_norm(q)

        # Attention pooling: (B, T, D) x (B, D, G) -> (B, T, G)
        attn_logits = torch.matmul(q, k.transpose(1, 2)) / (self.input_dim ** 0.5)
        attn = torch.softmax(attn_logits, dim=-1)
        pooled = torch.matmul(attn, x)  # (B, T, input_dim)

        tokens = self.mlp(pooled)  # (B, T, output_dim)

        if self.pos_embedding is not None:
            tokens = tokens + self.pos_embedding

        if not self.disable_output_norm:
            tokens = self.output_norm(tokens)

        if not self.disable_scale:
            min_scale = getattr(self, "_scale_min", 0.5)
            clamped_scale = torch.clamp(self.output_scale, min_scale, 10.0)
            tokens = tokens * clamped_scale

        if out_dtype is not None:
            tokens = tokens.to(out_dtype)
        return tokens


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
        
        # Token generators for each possible count
        # NOTE: No Tanh - LayerNorm provides sufficient normalization
        self.token_generators = nn.ModuleDict()
        for k in range(min_audio_tokens, max_audio_tokens + 1):
            self.token_generators[str(k)] = nn.Linear(llm_hidden_size, llm_hidden_size * k)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(llm_hidden_size, eps=1e-6)

        # Trainable scale to match LLM hidden state magnitudes
        # Init to ~8 since LLaMA hidden states typically have norm in 8-11 range
        self.output_scale = nn.Parameter(torch.tensor(8.0))

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
        
        # Use Xavier init on generator layers for proper gradient flow
        for generator in self.token_generators.values():
            if isinstance(generator, nn.Linear):
                nn.init.xavier_uniform_(generator.weight)
                if generator.bias is not None:
                    nn.init.zeros_(generator.bias)
            else:
                for m in generator.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
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
