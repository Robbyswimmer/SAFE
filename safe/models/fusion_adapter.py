import torch
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, LoraModel


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block for fusing audio tokens with LLM hidden states.
    Query comes from LLM hidden states, Key/Value from audio tokens.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        output_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = max(num_attention_heads, 1)  # Prevent division by zero
        self.attention_head_size = max(hidden_size // self.num_attention_heads, 1)  # Ensure >= 1
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.debug_logging = False
        self._last_attention_summary: Optional[dict] = None
        self._attention_log_limit = 5
        self._attention_logs_emitted = 0
        
        # Query projection (from LLM hidden states)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        
        # Key and Value projections (from audio tokens)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = nn.Dropout(output_dropout)
        
        # Optional stabilization on residual update (kept for checkpoint compatibility)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Attention dropout
        self.attention_dropout = nn.Dropout(attention_dropout)

        # Residual scaling for fusion (trainable with clamp)
        # Start at 0.5 to provide meaningful gradient signal from epoch 1.
        # Previous 0.1 init caused gradient starvation - audio contribution was
        # too weak to generate useful gradients, leading to training plateau at ~3 epochs.
        # The model can still learn to decrease this if needed.
        self.residual_scale = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.register_buffer("residual_scale_max", torch.tensor(5.0), persistent=False)

        # Initialize output projection with Xavier for proper gradient flow.
        # Zero-init causes gradient starvation - the fusion block produces near-zero
        # gradients, preventing the model from learning to use audio information.
        # Xavier provides balanced initialization that allows meaningful gradients
        # from the start while the residual_scale (starting at 0.5) controls the
        # actual contribution magnitude.
        nn.init.xavier_uniform_(self.output_dense.weight)
        if self.output_dense.bias is not None:
            nn.init.zeros_(self.output_dense.bias)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for multi-head attention computation."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        supervised_mask: Optional[torch.Tensor] = None,
        **kwargs,  # Accept and ignore extra kwargs to prevent PEFT errors
    ) -> torch.Tensor:
        """
        Forward pass of cross-attention.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size) - LLM hidden states (Query)
            audio_tokens: (batch_size, num_audio_tokens, hidden_size) - Audio tokens (Key/Value)
            attention_mask: Optional mask for audio tokens in standard 0/1 format:
                          - 1 = attend to this token
                          - 0 = ignore/mask this token
                          Can be boolean (True=attend, False=ignore) or float

        Returns:
            output: (batch_size, seq_len, hidden_size) - Fused representations
        """
        # Remember incoming dtype for final output
        orig_dtype = hidden_states.dtype
        
        # Clean inputs and upcast to fp32 for numerically stable computation
        hs = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)
        at = torch.nan_to_num(audio_tokens, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # Upcast to fp32 for stable attention math
        if hs.dtype != torch.float32:
            hs = hs.float()
        if at.dtype != torch.float32:
            at = at.float()

        input_dtype = torch.float32  # All computation now in fp32

        # Compute query, key, value (using cleaned fp32 inputs)
        query_layer = self.transpose_for_scores(self.query(hs))  # (B, H, seq_len, d)
        key_layer = self.transpose_for_scores(self.key(at))       # (B, H, audio_len, d)
        value_layer = self.transpose_for_scores(self.value(at))   # (B, H, audio_len, d)

        # Cast to float32 for numerically stable attention computation
        query_layer = torch.nan_to_num(query_layer, nan=0.0, posinf=1e4, neginf=-1e4).to(torch.float32)
        key_layer = torch.nan_to_num(key_layer, nan=0.0, posinf=1e4, neginf=-1e4).to(torch.float32)
        value_layer = torch.nan_to_num(value_layer, nan=0.0, posinf=1e4, neginf=-1e4).to(torch.float32)

        # Compute attention scores with numerical stability
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        # Prevent division by zero and ensure numerical stability
        scale_factor = max(math.sqrt(self.attention_head_size), 1e-8)
        attention_scores = attention_scores / scale_factor
        
        # Clamp scores to prevent softmax overflow (-50 to 50 is safe range for float32)
        attention_scores = torch.clamp(attention_scores, min=-50.0, max=50.0)
        attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=50.0, neginf=-50.0)
        
        # Apply attention mask with standardized format
        # STANDARD FORMAT: attention_mask uses 0/1 where 1=attend, 0=ignore
        if attention_mask is not None:
            # Standardize mask to boolean tensor
            mask = attention_mask.to(device=attention_scores.device)

            if mask.dtype == torch.bool:
                # Already boolean: True=attend, False=ignore
                attend_mask = mask
            else:
                # Convert to float and clean NaNs
                mask = mask.to(attention_scores.dtype)
                mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
                # Standard interpretation: 1=attend, 0=ignore
                attend_mask = mask > 0.5

            # Expand to (B,1,1,L) for broadcasting across heads and sequence
            while attend_mask.dim() < attention_scores.dim():
                attend_mask = attend_mask.unsqueeze(-2)

            # Mask out positions that should be ignored (attend_mask=False)
            mask_bool = ~attend_mask

            # 3) Use moderate negative, then clamp back to safe softmax range
            attention_scores = attention_scores.masked_fill(mask_bool, -1e4)
            attention_scores = torch.clamp(attention_scores, min=-50.0, max=50.0)

            # Log mask statistics for debugging
            if self.debug_logging and self._attention_logs_emitted < self._attention_log_limit:
                masked_tokens = mask_bool.sum().item()
                total_tokens = mask_bool.numel()
                print(f"[AttentionMask] Masked {masked_tokens}/{total_tokens} tokens", flush=True)
                self._attention_logs_emitted += 1

        # Stable softmax computation
        attention_probs = F.softmax(attention_scores, dim=-1)
        # Clean any NaN/Inf that might have appeared and ensure valid probability range
        attention_probs = torch.nan_to_num(attention_probs, nan=0.0, posinf=1.0, neginf=0.0)
        # Renormalize to ensure probabilities sum to 1 after cleaning
        attention_probs = attention_probs / (attention_probs.sum(dim=-1, keepdim=True) + 1e-8)
        attention_probs = self.attention_dropout(attention_probs)

        # Add attention diagnostics (sample to avoid spam)
        with torch.no_grad():
            if (
                getattr(self, "debug_logging", False)
                and self.training
                and torch.rand(1).item() < 0.01
            ):  # 1% sample rate
                # How much total attention flows INTO audio per head
                attn_to_audio = attention_probs.sum(dim=-1).mean().item()  # average over audio dim, then global mean
                # Entropy of attention over audio tokens (high = diffuse; low = peaky)
                p = attention_probs.clamp_min(1e-9)
                ent = (-p * p.log()).sum(dim=-1).mean().item()
                print(f"[AttnDiag] to_audio={attn_to_audio:.4f} entropy={ent:.4f}", flush=True)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.to(input_dtype)
        
        # Reshape and transpose back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        delta = self.output_dense(context_layer)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=1e4, neginf=-1e4)
        delta = self.output_dropout(delta)
        delta = delta.to(input_dtype)

        # Use warmup minimum that increases over training to prevent early collapse
        # while still forcing strong audio signal later.
        min_scale = getattr(self, '_scale_min', 0.5)  # Default 0.5, can be set externally
        residual_scale = torch.clamp(self.residual_scale, min_scale, float(self.residual_scale_max))
        if getattr(self, "debug_logging", False):
            print(
                f"[ResidualScale] scale={float(residual_scale.item()):.4f}",
                flush=True,
            )
        delta = residual_scale * delta

        if self.layer_norm is not None:
            delta = self.layer_norm(delta)

        delta = delta.to(orig_dtype)

        if getattr(self, "debug_logging", False):
            summary = {
                "overall_mean": float(attention_probs.mean().item()),
                "overall_max": float(attention_probs.max().item()),
            }

            if supervised_mask is not None:
                mask = supervised_mask
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1).unsqueeze(-1)
                elif mask.dim() == 3:
                    mask = mask.unsqueeze(-1)
                mask = mask.to(attention_probs.device, attention_probs.dtype)
                denom = mask.sum(dim=(1, 2, 3)).clamp_min(1e-6)
                weighted = (attention_probs * mask).sum(dim=(1, 2, 3)) / denom
                summary["supervised_mean_per_sample"] = weighted.detach().cpu()
                summary["supervised_mean"] = float(weighted.mean().item())
            else:
                per_sample = attention_probs.mean(dim=(1, 2, 3))
                summary["per_sample_mean"] = per_sample.detach().cpu()

            self._last_attention_summary = summary
            if self._attention_logs_emitted < self._attention_log_limit:
                message = (
                    f"[AttentionProbe] mean={summary['overall_mean']:.6f} "
                    f"max={summary['overall_max']:.6f}"
                )
                if "supervised_mean" in summary:
                    message += f" supervised_mean={summary['supervised_mean']:.6f}"
                print(message, flush=True)
                self._attention_logs_emitted += 1
        else:
            self._last_attention_summary = None

        return delta


class BottleneckCrossAttentionBlock(nn.Module):
    """
    Bottleneck cross-attention block for efficient audio fusion.
    Uses small projections instead of full hidden_size, equivalent to LoRA in parameter count
    but simpler (no PEFT dependency, no frozen random base layers).

    Now follows standard transformer pattern: CrossAttention → FFN
    This provides the non-linear transformation capacity that was missing.

    Parameter equivalence:
    - bottleneck_dim=32 ≈ LoRA rank-16 (655K params per layer)
    - bottleneck_dim=16 ≈ LoRA rank-8 (328K params per layer)
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: int = 32,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        output_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        use_ffn: bool = True,
        ffn_expansion: float = 2.0,
        use_pre_norm: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.num_attention_heads = max(num_attention_heads, 1)
        self.use_ffn = use_ffn
        self.use_pre_norm = use_pre_norm

        # Ensure bottleneck_dim is divisible by num_heads
        if bottleneck_dim % self.num_attention_heads != 0:
            # Adjust num_heads to divide evenly
            for n in [4, 2, 1]:
                if bottleneck_dim % n == 0:
                    self.num_attention_heads = n
                    break

        self.attention_head_size = bottleneck_dim // self.num_attention_heads
        self.all_head_size = bottleneck_dim  # This is the bottleneck dimension

        self.debug_logging = False
        self._attention_log_limit = 5
        self._attention_logs_emitted = 0

        # Pre-norm layer (applied before Q/K/V projections if use_pre_norm=True)
        # Pre-norm is more stable for training (used in GPT-2+, LLaMA, etc.)
        if use_pre_norm:
            self.pre_norm_q = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            self.pre_norm_kv = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # Bottleneck projections - small dimensions
        # Q: hidden_size → bottleneck_dim
        self.query = nn.Linear(hidden_size, bottleneck_dim)
        # K: hidden_size → bottleneck_dim
        self.key = nn.Linear(hidden_size, bottleneck_dim)
        # V: hidden_size → bottleneck_dim
        self.value = nn.Linear(hidden_size, bottleneck_dim)
        # O: bottleneck_dim → hidden_size
        self.output_dense = nn.Linear(bottleneck_dim, hidden_size)

        self.output_dropout = nn.Dropout(output_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

        # Layer norm on residual (applied to hidden_size)
        # For pre-norm, this becomes the post-attention norm
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # FFN block in BOTTLENECK space (not full hidden_size!)
        # This provides non-linear transformation capacity with minimal parameters.
        # Operating in bottleneck: 64 → 128 → 64 = ~16K params
        # vs full hidden: 5120 → 10240 → 5120 = ~105M params
        if use_ffn:
            ffn_hidden = int(bottleneck_dim * ffn_expansion)
            self.ffn = nn.Sequential(
                nn.LayerNorm(bottleneck_dim, eps=layer_norm_eps),
                nn.Linear(bottleneck_dim, ffn_hidden),
                nn.GELU(),
                nn.Dropout(output_dropout),
                nn.Linear(ffn_hidden, bottleneck_dim),
                nn.Dropout(output_dropout),
            )
            # Initialize FFN output to small values for stable residual
            nn.init.normal_(self.ffn[-2].weight, std=0.02)
            nn.init.zeros_(self.ffn[-2].bias)

        # Residual scaling - start at 0.5 for meaningful gradient flow
        # (0.1 was too conservative, causing gradient starvation)
        self.residual_scale = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.register_buffer("residual_scale_max", torch.tensor(5.0), persistent=False)

        # Initialize with Xavier for good gradient flow
        self._init_weights()

    def _init_weights(self):
        for module in [self.query, self.key, self.value]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # Initialize output projection with Xavier for proper gradient flow.
        # Zero-init causes gradient starvation - prevents learning audio fusion.
        nn.init.xavier_uniform_(self.output_dense.weight)
        if self.output_dense.bias is not None:
            nn.init.zeros_(self.output_dense.bias)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention: (B, L, bottleneck) → (B, H, L, head_size)"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        supervised_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of bottleneck cross-attention with optional FFN.

        Args:
            hidden_states: (B, seq_len, hidden_size) - LLM hidden states (Query source)
            audio_tokens: (B, audio_len, hidden_size) - Audio tokens (Key/Value source)
            attention_mask: Optional mask for audio tokens (1=attend, 0=ignore)

        Returns:
            delta: (B, seq_len, hidden_size) - Residual to add to hidden states
        """
        orig_dtype = hidden_states.dtype

        # Clean and upcast to fp32
        hs = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4).float()
        at = torch.nan_to_num(audio_tokens, nan=0.0, posinf=1e4, neginf=-1e4).float()

        # Apply pre-norm if enabled (more stable training, used in modern LLMs)
        if self.use_pre_norm:
            hs_normed = self.pre_norm_q(hs)
            at_normed = self.pre_norm_kv(at)
        else:
            hs_normed = hs
            at_normed = at

        # Project to bottleneck dimension
        query_layer = self.transpose_for_scores(self.query(hs_normed))  # (B, H, seq_len, head_size)
        key_layer = self.transpose_for_scores(self.key(at_normed))       # (B, H, audio_len, head_size)
        value_layer = self.transpose_for_scores(self.value(at_normed))   # (B, H, audio_len, head_size)

        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scale_factor = max(math.sqrt(self.attention_head_size), 1e-8)
        attention_scores = attention_scores / scale_factor
        attention_scores = torch.clamp(attention_scores, min=-50.0, max=50.0)

        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.to(device=attention_scores.device)
            if mask.dtype == torch.bool:
                attend_mask = mask
            else:
                mask = mask.float()
                attend_mask = mask > 0.5
            while attend_mask.dim() < attention_scores.dim():
                attend_mask = attend_mask.unsqueeze(-2)
            attention_scores = attention_scores.masked_fill(~attend_mask, -1e4)

        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = torch.nan_to_num(attention_probs, nan=0.0, posinf=1.0, neginf=0.0)
        attention_probs = self.attention_dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)  # (B, H, seq_len, head_size)

        # Reshape back: (B, H, seq_len, head_size) → (B, seq_len, bottleneck_dim)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size(0), context_layer.size(1), self.bottleneck_dim)

        # Apply FFN in BOTTLENECK space (before output projection)
        # This provides non-linear transformation with minimal parameters:
        # bottleneck(64) → ffn(128) → bottleneck(64) = ~16K params
        # vs hidden(5120) → ffn(10240) → hidden(5120) = ~105M params
        if self.use_ffn:
            context_layer = context_layer + self.ffn(context_layer)

        # Project back to hidden_size
        delta = self.output_dense(context_layer)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=1e4, neginf=-1e4)
        delta = self.output_dropout(delta)

        # Use warmup minimum that increases over training to prevent early collapse
        min_scale = getattr(self, '_scale_min', 0.5)  # Default 0.5, can be set externally
        residual_scale = torch.clamp(self.residual_scale, min_scale, float(self.residual_scale_max))
        delta = residual_scale * delta

        # Layer norm (post-attention)
        if self.layer_norm is not None:
            delta = self.layer_norm(delta)

        return delta.to(orig_dtype)


class SimpleFusionAdapter(nn.Module):
    """
    Simple fusion adapter using bottleneck cross-attention.
    No PEFT/LoRA - just straightforward small cross-attention blocks.
    All parameters are trainable.

    Now supports:
    - use_ffn: Add FFN after cross-attention (standard transformer pattern)
    - use_pre_norm: Use pre-norm instead of post-norm (more stable training)
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: int = 32,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        use_tokenwise_gate: bool = False,
        use_ffn: bool = True,
        ffn_expansion: float = 2.0,
        use_pre_norm: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.use_tokenwise_gate = bool(use_tokenwise_gate)
        self.debug_logging = False

        # Bottleneck cross-attention with optional FFN and pre-norm
        self.cross_attention = BottleneckCrossAttentionBlock(
            hidden_size=hidden_size,
            bottleneck_dim=bottleneck_dim,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            use_ffn=use_ffn,
            ffn_expansion=ffn_expansion,
            use_pre_norm=use_pre_norm,
        )

        # Optional token-wise gating
        if self.use_tokenwise_gate:
            self.token_gate = nn.Linear(hidden_size * 2, 1)

    def set_debug_logging(self, enabled: bool, log_limit: int = 5) -> None:
        self.debug_logging = bool(enabled)
        self.cross_attention.debug_logging = enabled
        self.cross_attention._attention_log_limit = log_limit

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        gate: float = 1.0,
        supervised_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional gating.

        Args:
            hidden_states: (B, seq_len, hidden_size) - LLM hidden states
            audio_tokens: (B, audio_len, hidden_size) - Audio tokens
            attention_mask: Optional mask for audio tokens
            gate: Scalar gating factor (0.0 = no audio, 1.0 = full audio)

        Returns:
            output: (B, seq_len, hidden_size) - hidden_states + gated residual
        """
        orig_dtype = hidden_states.dtype

        # Debug: log fusion inputs (limited)
        if self.debug_logging and not hasattr(self, '_fusion_forward_logged'):
            print(f"[FUSION FWD] hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}", flush=True)
            print(f"[FUSION FWD] audio_tokens shape: {audio_tokens.shape}, dtype: {audio_tokens.dtype}", flush=True)
            print(f"[FUSION FWD] hidden_states norm: {hidden_states.norm().item():.2f}", flush=True)
            print(f"[FUSION FWD] audio_tokens norm: {audio_tokens.norm().item():.2f}", flush=True)
            self._fusion_forward_logged = True

        # Get cross-attention residual
        delta = self.cross_attention(
            hidden_states=hidden_states,
            audio_tokens=audio_tokens,
            attention_mask=attention_mask,
            supervised_mask=supervised_mask,
        )

        # Debug: log delta (limited)
        if self.debug_logging and not hasattr(self, '_fusion_delta_logged'):
            print(f"[FUSION FWD] delta norm: {delta.norm().item():.2f}, delta range: [{delta.min().item():.3f}, {delta.max().item():.3f}]", flush=True)
            self._fusion_delta_logged = True

        # Apply gating
        if self.use_tokenwise_gate:
            batch_size, seq_len, _ = hidden_states.size()
            pooled_audio = audio_tokens.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
            gate_input = torch.cat([hidden_states, pooled_audio], dim=-1)
            # Cast to token_gate weight dtype (may be fp32 while input is fp16)
            gate_input = gate_input.to(self.token_gate.weight.dtype)
            g = torch.sigmoid(self.token_gate(gate_input))
            # Cast gate back to original dtype for multiplication
            g = g.to(orig_dtype)
            if isinstance(gate, torch.Tensor):
                g = g * gate.to(g.device, g.dtype).view(-1, 1, 1)
            else:
                g = g * float(gate)
            output = hidden_states + g * delta
        else:
            if isinstance(gate, torch.Tensor):
                gate_tensor = gate.to(hidden_states.device, hidden_states.dtype)
                while gate_tensor.dim() < delta.dim():
                    gate_tensor = gate_tensor.unsqueeze(-1)
                output = hidden_states + gate_tensor * delta
            else:
                output = hidden_states + float(gate) * delta

        return output.to(orig_dtype)


class LoRAFusionAdapter(nn.Module):
    """
    LoRA-based fusion adapter that adds audio cross-attention to LLM layers.
    Uses Low-Rank Adaptation on query and value projections.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,  # Disabled for stable gradient flow during bring-up
        attention_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        train_base_cross_attention: bool = False,
        use_tokenwise_gate: bool = False,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.debug_logging = False
        self.last_attention_summary: Optional[dict] = None
        self._attention_log_limit = 5
        self._attention_logs_emitted = 0
        self.use_tokenwise_gate = bool(use_tokenwise_gate)
        
        # Base cross-attention block
        self.cross_attention = CrossAttentionBlock(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout
        )
        
        # LoRA configuration
        if target_modules is None:
            # Default: adapt all projections for random-init cross-attention.
            # (If you only LoRA Q/V but keep K/O frozen random, fusion can be weak.)
            target_modules = ["query", "key", "value", "output_dense"]
            
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )

        # Apply LoRA to cross-attention
        self.cross_attention = get_peft_model(self.cross_attention, self.lora_config)

        # CrossAttentionBlock is randomly initialized. Training its full 5120×5120
        # projection matrices is extremely parameter-heavy; by default we train only
        # LoRA weights + a small residual scale. Enable full training explicitly.
        #
        # IMPORTANT: We need to selectively freeze/unfreeze:
        # - LoRA weights (lora_A, lora_B) should ALWAYS be trainable
        # - Base layer weights should only be trainable if train_base_cross_attention=True
        # - residual_scale should always be trainable
        base_model = getattr(self.cross_attention, "base_model", None)
        if base_model is not None:
            for name, param in base_model.named_parameters():
                # LoRA weights should always be trainable
                if "lora_" in name:
                    param.requires_grad = True
                # residual_scale should always be trainable
                elif "residual_scale" in name:
                    param.requires_grad = True
                # Base layer weights are only trainable if explicitly requested
                else:
                    param.requires_grad = bool(train_base_cross_attention)
        else:
            # Fallback: if PEFT wrapper doesn't expose base_model, do not attempt
            # to unfreeze everything.
            pass

        # Token-wise gating head (optional)
        if self.use_tokenwise_gate:
            # Gate takes [hidden; pooled_audio] → scalar gate per token
            self.token_gate = nn.Linear(hidden_size * 2, 1)

    def set_debug_logging(self, enabled: bool, log_limit: int = 5) -> None:
        self.debug_logging = bool(enabled)
        self._attention_log_limit = int(max(0, log_limit))
        self._attention_logs_emitted = 0

        base_model = getattr(self.cross_attention, "base_model", None)
        if base_model is not None:
            base_model.debug_logging = self.debug_logging
            base_model._attention_log_limit = self._attention_log_limit
            base_model._attention_logs_emitted = 0

    def configure_attention_probe(self, enabled: bool, log_limit: int = 5) -> None:
        self.set_debug_logging(enabled, log_limit)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        gate: float = 1.0,
        supervised_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with LoRA fusion and gating.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) - LLM hidden states
            audio_tokens: (batch_size, num_audio_tokens, hidden_size) - Audio tokens
            attention_mask: Optional attention mask for audio tokens
            gate: Gating factor (0.0 = no audio, 1.0 = full audio)
            
        Returns:
            fused_states: (batch_size, seq_len, hidden_size) - Fused representations
        """
        # Remember incoming dtype for final output
        orig_dtype = hidden_states.dtype
        weight = self.cross_attention.base_model.query.weight
        target_dtype = weight.dtype
        target_device = weight.device
        if hidden_states.dtype != target_dtype or hidden_states.device != target_device:
            hidden_states = hidden_states.to(device=target_device, dtype=target_dtype)
        if audio_tokens.dtype != target_dtype or audio_tokens.device != target_device:
            audio_tokens = audio_tokens.to(device=target_device, dtype=target_dtype)

        # Apply cross-attention with LoRA (returns hidden_states + attention_output)
        delta_states = self.cross_attention(
            hidden_states=hidden_states,
            audio_tokens=audio_tokens,
            attention_mask=attention_mask,
            supervised_mask=supervised_mask,
        )

        base_model = getattr(self.cross_attention, "base_model", None)
        if base_model is not None:
            self.last_attention_summary = getattr(base_model, "_last_attention_summary", None)
        else:
            self.last_attention_summary = None
        if self.debug_logging and self.last_attention_summary is not None:
            if self._attention_logs_emitted < self._attention_log_limit:
                summary = self.last_attention_summary
                msg = "[AttentionProbe]"
                overall_mean = summary.get("overall_mean")
                overall_max = summary.get("overall_max")
                if overall_mean is not None:
                    msg += f" mean={overall_mean:.6f}"
                if overall_max is not None:
                    msg += f" max={overall_max:.6f}"
                supervised_mean = summary.get("supervised_mean")
                if supervised_mean is not None:
                    msg += f" supervised_mean={supervised_mean:.6f}"
                print(msg, flush=True)
                self._attention_logs_emitted += 1

        # Apply gating on residual update
        if self.use_tokenwise_gate:
            # Token-wise gate: g ∈ [0,1]^{B×L×1}, based on hidden + pooled audio
            batch_size, seq_len, _ = hidden_states.size()
            pooled_audio = audio_tokens.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
            gate_input = torch.cat([hidden_states, pooled_audio], dim=-1)  # (B, L, 2H)
            g = torch.sigmoid(self.token_gate(gate_input))  # (B, L, 1)

            # Combine with scalar gate if provided
            if isinstance(gate, torch.Tensor):
                gate_tensor = gate.to(device=hidden_states.device, dtype=hidden_states.dtype)
                while gate_tensor.dim() < g.dim():
                    gate_tensor = gate_tensor.unsqueeze(-1)
                g = g * gate_tensor
            else:
                g = g * float(gate)

            output = hidden_states + g * delta_states
        else:
            if isinstance(gate, torch.Tensor):
                gate_tensor = gate.to(device=hidden_states.device, dtype=hidden_states.dtype)
                while gate_tensor.dim() < delta_states.dim():
                    gate_tensor = gate_tensor.unsqueeze(-1)
                output = hidden_states + gate_tensor * delta_states
            else:
                gate_value = float(gate)
                output = hidden_states + gate_value * delta_states

        # Cast back to the original dtype expected by the LM stack
        output = output.to(orig_dtype)

        return output


class MultiLayerFusionAdapter(nn.Module):
    """
    Multi-layer fusion adapter that can insert modality fusion at configurable decoder layers.
    Supports multiple modalities sharing the same adapter instance.

    Supports two modes:
    - LoRA mode (use_bottleneck=False): Uses LoRAFusionAdapter with PEFT
    - Bottleneck mode (use_bottleneck=True): Uses SimpleFusionAdapter with small cross-attention

    New architectural options (bottleneck mode only):
    - use_ffn: Add FFN after cross-attention (standard transformer pattern, default=True)
    - ffn_expansion: FFN hidden size multiplier (default=2.0)
    - use_pre_norm: Use pre-norm instead of post-norm (default=False)

    Parameter equivalence for bottleneck_dim:
    - bottleneck_dim=32 ≈ LoRA rank-16 (~655K params per layer)
    - bottleneck_dim=16 ≈ LoRA rank-8 (~328K params per layer)
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        fusion_layer_indices: Union[Dict[str, List[int]], List[int], None] = None,
        num_attention_heads: int = 8,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        modalities: Optional[Dict[str, Any]] = None,
        use_tokenwise_gate: bool = False,
        use_bottleneck: bool = False,
        bottleneck_dim: int = 32,
        use_ffn: bool = True,
        ffn_expansion: float = 2.0,
        use_pre_norm: bool = False,
        **unused_kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_tokenwise_gate = bool(use_tokenwise_gate)
        self.use_bottleneck = bool(use_bottleneck)
        self.bottleneck_dim = bottleneck_dim
        self.use_ffn = use_ffn
        self.ffn_expansion = ffn_expansion
        self.use_pre_norm = use_pre_norm
        # Last recorded attention summary from any inner fusion adapter
        self.last_attention_summary: Optional[dict] = None
        self.extra_config = dict(unused_kwargs)

        layer_mapping_source = modalities if modalities is not None else fusion_layer_indices
        self.modality_configs = modalities or {}
        self.fusion_layers = self._normalize_layer_mapping(layer_mapping_source)
        self.fusion_layer_indices = sorted({idx for indices in self.fusion_layers.values() for idx in indices})
        self.layer_modalities = self._invert_layer_mapping(self.fusion_layers)
        self.fusion_adapters = nn.ModuleDict()
        target_modules = unused_kwargs.get("target_modules", None)
        train_base_cross_attention = bool(unused_kwargs.get("train_base_cross_attention", False))

        for modality, indices in self.fusion_layers.items():
            for layer_idx in indices:
                key = self._adapter_key(modality, layer_idx)
                if self.use_bottleneck:
                    # Use simple bottleneck cross-attention (no PEFT)
                    # Now includes FFN and pre-norm options
                    self.fusion_adapters[key] = SimpleFusionAdapter(
                        hidden_size=hidden_size,
                        bottleneck_dim=bottleneck_dim,
                        num_attention_heads=min(num_attention_heads, bottleneck_dim),
                        attention_dropout=attention_dropout,
                        use_tokenwise_gate=self.use_tokenwise_gate,
                        use_ffn=use_ffn,
                        ffn_expansion=ffn_expansion,
                        use_pre_norm=use_pre_norm,
                    )
                else:
                    # Use LoRA-based fusion adapter (original behavior)
                    self.fusion_adapters[key] = LoRAFusionAdapter(
                        hidden_size=hidden_size,
                        num_attention_heads=num_attention_heads,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        attention_dropout=attention_dropout,
                        target_modules=target_modules,
                        train_base_cross_attention=train_base_cross_attention,
                        use_tokenwise_gate=self.use_tokenwise_gate,
                    )

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_tokens: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        gate: float = 1.0,
        active_fusion_layer: Optional[int] = None,
        supervised_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if active_fusion_layer is not None and layer_idx != active_fusion_layer:
            return hidden_states

        if layer_idx not in self.layer_modalities:
            return hidden_states

        modality_tokens = {"audio": audio_tokens}
        modality_masks = {"audio": attention_mask} if attention_mask is not None else None

        return self.apply_fusion_at_layer(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            modality_tokens=modality_tokens,
            modality_masks=modality_masks,
            gate=gate,
            supervised_mask=supervised_mask,
        )

    def apply_fusion_at_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        modality_tokens: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
        gate: Union[float, Dict[str, float]] = 1.0,
        supervised_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        modalities = self.layer_modalities.get(layer_idx, [])
        if not modalities:
            return hidden_states

        output = hidden_states
        target_batch = hidden_states.size(0)
        for modality in modalities:
            tokens = modality_tokens.get(modality)
            if tokens is None:
                continue
            mask = None
            if modality_masks is not None:
                mask = modality_masks.get(modality)

            tokens = self._align_batch(tokens, target_batch)
            if mask is not None:
                mask = self._align_batch(mask, target_batch)

            adapter_key = self._adapter_key(modality, layer_idx)
            if adapter_key not in self.fusion_adapters:
                continue
            adapter = self.fusion_adapters[adapter_key]

            modality_gate: Union[float, torch.Tensor]
            if isinstance(gate, dict):
                modality_gate = gate.get(modality, 1.0)
            else:
                modality_gate = gate

            output = adapter(
                hidden_states=output,
                audio_tokens=tokens,
                attention_mask=mask,
                gate=modality_gate,
                supervised_mask=supervised_mask,
            )

            # Capture latest attention diagnostics for external inspection
            if hasattr(adapter, "last_attention_summary"):
                self.last_attention_summary = adapter.last_attention_summary

        return output

    def _align_batch(self, tensor: torch.Tensor, target_batch: int):
        if tensor is None or not torch.is_tensor(tensor):
            return tensor
        current = tensor.size(0)
        if current == target_batch:
            return tensor
        if current == 1:
            return tensor.expand(target_batch, *tensor.shape[1:])
        if target_batch % current == 0:
            repeat = target_batch // current
            return tensor.repeat_interleave(repeat, dim=0)
        raise ValueError(
            f"Cannot align tensor batch dimension from {current} to {target_batch}"
        )

    def set_debug_logging(self, enabled: bool, log_limit: int = 5) -> None:
        for adapter in self.fusion_adapters.values():
            adapter.set_debug_logging(enabled, log_limit)

    def configure_attention_probe(self, enabled: bool, log_limit: int = 5) -> None:
        self.set_debug_logging(enabled, log_limit)

    @staticmethod
    def _adapter_key(modality: str, layer_idx: int) -> str:
        return f"{modality}:{layer_idx}"

    def _normalize_layer_mapping(
        self,
        mapping: Union[Dict[str, List[int]], List[int], None],
    ) -> Dict[str, List[int]]:
        if mapping is None:
            default_indices = [self.num_layers // 3, 2 * self.num_layers // 3]
            return {"audio": default_indices}

        if isinstance(mapping, dict):
            normalized: Dict[str, List[int]] = {}
            for modality, value in mapping.items():
                if value is None:
                    continue
                if isinstance(value, dict):
                    indices = value.get("layer_indices")
                else:
                    indices = value
                if indices is None:
                    continue
                normalized[modality] = sorted({int(idx) for idx in list(indices)})
            return normalized

        if isinstance(mapping, (list, tuple)):
            return {"audio": [int(idx) for idx in mapping]}

        raise ValueError("fusion_layer_indices must be None, a list, or a modality -> layers mapping")

    @staticmethod
    def _invert_layer_mapping(mapping: Dict[str, List[int]]) -> Dict[int, List[str]]:
        layer_to_modalities: Dict[int, List[str]] = {}
        for modality, indices in mapping.items():
            for idx in indices:
                layer_to_modalities.setdefault(idx, []).append(modality)
        return layer_to_modalities


class GatedFusionAdapter(nn.Module):
    """
    Fusion adapter with learnable gating mechanism.
    Gate is computed based on input features and can be trained end-to-end.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        lora_rank: int = 8,
        gate_hidden_size: int = 64,
        gate_init_bias: float = -2.5,  # Initialize gate toward OFF
        target_modules: Optional[List[str]] = None,
        train_base_cross_attention: bool = False,
    ):
        super().__init__()
        
        # Core fusion adapter
        self.fusion_adapter = LoRAFusionAdapter(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            lora_rank=lora_rank,
            target_modules=target_modules,
            train_base_cross_attention=train_base_cross_attention,
        )
        
        # Learnable gate network
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size * 2, gate_hidden_size),  # Concat LLM + audio features
            nn.GELU(),
            nn.Linear(gate_hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Initialize gate bias toward OFF
        with torch.no_grad():
            self.gate_network[-2].bias.fill_(gate_init_bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        force_gate: Optional[float] = None,
        supervised_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with learnable gating.
        
        Args:
            hidden_states: LLM hidden states
            audio_tokens: Audio tokens
            attention_mask: Optional attention mask
            force_gate: If provided, use this gate value instead of computing
            
        Returns:
            Tuple of (fused_states, gate_value)
        """
        if force_gate is not None:
            gate = torch.tensor(force_gate, device=hidden_states.device)
        else:
            # Compute gate from input features
            # Pool features for gate computation
            llm_pooled = hidden_states.mean(dim=1)  # (batch_size, hidden_size)
            audio_pooled = audio_tokens.mean(dim=1)  # (batch_size, hidden_size)
            
            # Concatenate and compute gate
            gate_input = torch.cat([llm_pooled, audio_pooled], dim=-1)
            gate = self.gate_network(gate_input).squeeze(-1)  # (batch_size,)
        
        # Apply fusion with computed gate
        if isinstance(gate, torch.Tensor) and gate.dim() > 0:
            # Batch-wise gating
            fused_states = []
            for i in range(hidden_states.shape[0]):
                sample_attention = (
                    attention_mask[i:i+1]
                    if attention_mask is not None
                    else None
                )
                sample_mask = supervised_mask[i:i+1] if supervised_mask is not None else None
                sample_fused = self.fusion_adapter(
                    hidden_states=hidden_states[i:i+1],
                    audio_tokens=audio_tokens[i:i+1],
                    attention_mask=sample_attention,
                    gate=gate[i].item(),
                    supervised_mask=sample_mask,
                )
                fused_states.append(sample_fused)
            fused_states = torch.cat(fused_states, dim=0)
        else:
            # Scalar gating
            fused_states = self.fusion_adapter(
                hidden_states=hidden_states,
                audio_tokens=audio_tokens,
                attention_mask=attention_mask,
                gate=gate.item() if isinstance(gate, torch.Tensor) else gate,
                supervised_mask=supervised_mask,
            )

        return fused_states, gate

    def set_debug_logging(self, enabled: bool, log_limit: int = 5) -> None:
        self.fusion_adapter.set_debug_logging(enabled, log_limit)

    def configure_attention_probe(self, enabled: bool, log_limit: int = 5) -> None:
        self.set_debug_logging(enabled, log_limit)
