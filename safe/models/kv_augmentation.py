# -*- coding: utf-8 -*-
"""
KV Augmentation for SAFE Audio-Visual Model

Instead of additive residual fusion (H' = H + gate * delta), KV augmentation
injects audio tokens directly into the LLM's self-attention as additional
keys and values:

    K' = [K_text; K_audio]
    V' = [V_text; V_audio]
    Attention(Q, K', V')

This makes audio "un-ignorable" - it's part of the attention context,
not an additive residual that the frozen LLM can suppress.

Key design decisions:
- Audio K,V do NOT get rotary position embeddings (position-agnostic like prefix tokens)
- Audio tokens are visible to all query positions (non-causal)
- Includes optional minimum attention regularization loss
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioQueryAdapter(nn.Module):
    """
    Low-rank adapter that produces ΔQ for audio attention.

    LoRA-style: hidden → rank → num_heads * head_dim
    Initialized near-zero so initial behavior ≈ no adapter.

    This allows frozen LLM queries to attend to audio K,V by learning
    a small delta: Q_audio = Q_frozen + ΔQ_adapter(H)
    """

    def __init__(
        self,
        hidden_size: int = 5120,
        num_heads: int = 40,
        head_dim: int = 128,
        rank: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rank = rank

        # LoRA-style down/up projection
        self.down_proj = nn.Linear(hidden_size, rank, bias=False)
        self.up_proj = nn.Linear(rank, num_heads * head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Learnable scale - start small to prevent ΔQ explosion
        # With 5e-3 LR and 0.5 scale, ΔQ/Q exploded to 300%+ and destroyed signal
        # Start at 0.1, let it grow naturally if needed
        self.scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        """Initialize near-zero for stable start."""
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)  # Start with zero output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute ΔQ for audio attention.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            delta_q: (batch, seq_len, num_heads * head_dim)
        """
        # Cast to match weights dtype if needed
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.down_proj.weight.dtype)

        delta = self.down_proj(hidden_states)
        delta = F.gelu(delta)
        delta = self.dropout(delta)
        delta = self.up_proj(delta)
        delta = delta * self.scale

        # Cast back to input dtype
        return delta.to(input_dtype)


class KVAugmentationAdapter(nn.Module):
    """
    Trainable projections for audio tokens → K,V space, plus query adapter.

    Projects audio tokens to keys and values compatible with LLM's self-attention.
    Uses bottleneck architecture to reduce parameters while maintaining expressivity.

    Also includes an AudioQueryAdapter that produces ΔQ for audio attention,
    allowing frozen queries to attend to audio K,V.
    """

    def __init__(
        self,
        hidden_size: int = 5120,
        num_heads: int = 40,
        head_dim: int = 128,
        num_key_value_heads: Optional[int] = None,  # For GQA models (e.g., LLaMA)
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        use_bottleneck: bool = True,
        query_adapter_rank: int = 16,
        input_dim: Optional[int] = None,  # Audio token input dim (if smaller than hidden_size)
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        # For GQA: K,V have fewer heads than Q
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        self.total_query_size = num_heads * head_dim  # For Q adapter
        self.total_kv_size = self.num_key_value_heads * head_dim  # For K,V projections
        self.use_bottleneck = use_bottleneck
        self.query_adapter_rank = query_adapter_rank

        # Input dimension for audio tokens (can be smaller than hidden_size for param efficiency)
        self.input_dim = input_dim if input_dim is not None else hidden_size

        if use_bottleneck:
            # Bottleneck projection: input_dim → bottleneck → kv_size
            self.audio_k_proj = nn.Sequential(
                nn.Linear(self.input_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, self.total_kv_size),
            )
            self.audio_v_proj = nn.Sequential(
                nn.Linear(self.input_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, self.total_kv_size),
            )
        else:
            # Direct projection (more parameters)
            self.audio_k_proj = nn.Linear(self.input_dim, self.total_kv_size)
            self.audio_v_proj = nn.Linear(self.input_dim, self.total_kv_size)

        # Learnable scaling factor for audio K,V (no gate here - gate applied at combine step only)
        # Start at 0.5 for meaningful gradients
        self.audio_scale = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("scale_min", torch.tensor(0.1))
        self.register_buffer("scale_max", torch.tensor(2.0))

        # LayerNorm for audio tokens BEFORE K,V projection
        # Critical: audio token norms can be huge (~500+), causing softmax saturation
        # Use input_dim (not hidden_size) since audio tokens may be in smaller space
        self.audio_norm = nn.LayerNorm(self.input_dim)

        # Audio Query Adapter: produces ΔQ for audio attention
        # This allows frozen queries to attend to audio K,V
        self.audio_query_adapter = AudioQueryAdapter(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            rank=query_adapter_rank,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize projections for stable training."""
        for module in [self.audio_k_proj, self.audio_v_proj]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        audio_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project audio tokens to K,V space.

        NOTE: Gate is NOT applied here. Gate is applied once at the combine step
        in _forward_with_audio_kv to avoid double-gating.

        Args:
            audio_tokens: (batch_size, num_audio_tokens, hidden_size)

        Returns:
            audio_keys: (batch_size, num_audio_tokens, total_head_size)
            audio_values: (batch_size, num_audio_tokens, total_head_size)
        """
        # Cast audio tokens to match projection weights dtype
        input_dtype = audio_tokens.dtype
        weight_dtype = self.audio_k_proj[0].weight.dtype if self.use_bottleneck else self.audio_k_proj.weight.dtype
        audio_tokens = audio_tokens.to(weight_dtype)

        # Normalize audio tokens BEFORE projection
        # Critical: audio token norms can be huge (~500+), causing softmax saturation
        audio_tokens = self.audio_norm(audio_tokens)

        # Clamp scale to reasonable range (no gate here - applied at combine step)
        scale = torch.clamp(self.audio_scale, self.scale_min, self.scale_max)

        # Project to K,V space with learned scale
        audio_keys = self.audio_k_proj(audio_tokens) * scale
        audio_values = self.audio_v_proj(audio_tokens) * scale

        # Cast back to input dtype for compatibility with attention
        audio_keys = audio_keys.to(input_dtype)
        audio_values = audio_values.to(input_dtype)

        return audio_keys, audio_values

    def reshape_for_attention(
        self,
        tensor: torch.Tensor,
        for_kv: bool = True,
    ) -> torch.Tensor:
        """
        Reshape projected tensor for multi-head attention.

        Args:
            tensor: (batch_size, seq_len, total_size)
            for_kv: If True, use num_key_value_heads (for K,V). If False, use num_heads (for Q).

        Returns:
            reshaped: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = tensor.shape
        num_heads = self.num_key_value_heads if for_kv else self.num_heads
        tensor = tensor.view(batch_size, seq_len, num_heads, self.head_dim)
        return tensor.transpose(1, 2)  # (batch, heads, seq, head_dim)


class KVAugmentedAttention(nn.Module):
    """
    Wrapper around LlamaAttention that injects audio K,V.

    This module wraps the original attention and extends it to include
    audio tokens in the key-value pairs. Audio tokens are:
    - NOT rotated by RoPE (position-agnostic like prefix tokens)
    - Visible to all query positions (non-causal)
    """

    def __init__(
        self,
        original_attention: nn.Module,
        kv_adapter: KVAugmentationAdapter,
        layer_idx: int,
    ):
        super().__init__()
        self.original_attention = original_attention
        self.kv_adapter = kv_adapter
        self.layer_idx = layer_idx

        # Copy attributes from original attention for compatibility
        # Try to infer dimensions from projection layers as fallback (more reliable for Qwen)
        q_proj = getattr(original_attention, 'q_proj', None)
        k_proj = getattr(original_attention, 'k_proj', None)

        # Get head_dim first (usually reliable from config or infer from hidden_size/num_heads)
        self.head_dim = getattr(original_attention, 'head_dim', 128)

        # Get num_heads - try attribute first, then infer from q_proj
        num_heads = getattr(original_attention, 'num_heads', None)
        if num_heads is None and q_proj is not None and hasattr(q_proj, 'out_features'):
            num_heads = q_proj.out_features // self.head_dim
        self.num_heads = num_heads if num_heads is not None else 32  # Default for Qwen3-8B

        # Get num_key_value_heads - try attribute first, then infer from k_proj
        num_kv_heads = getattr(original_attention, 'num_key_value_heads', None)
        if num_kv_heads is None and k_proj is not None and hasattr(k_proj, 'out_features'):
            num_kv_heads = k_proj.out_features // self.head_dim
        self.num_key_value_heads = num_kv_heads if num_kv_heads is not None else self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # Debug: log GQA config detection (once per class)
        if not hasattr(self.__class__, '_gqa_logged'):
            self.__class__._gqa_logged = True
            print(f"[KVWrapper] Detected GQA config: num_heads={self.num_heads}, "
                  f"num_kv_heads={self.num_key_value_heads}, head_dim={self.head_dim}, "
                  f"kv_groups={self.num_key_value_groups}", flush=True)

        # Storage for current forward pass audio data
        self._audio_tokens: Optional[torch.Tensor] = None
        self._audio_mask: Optional[torch.Tensor] = None
        self._gate: float = 1.0

        # For returning attention weights (used in regularization)
        self._last_attention_weights: Optional[torch.Tensor] = None  # Detached, for logging
        self._live_attention_weights: Optional[torch.Tensor] = None  # With gradients, for reg loss
        self._return_attention_weights: bool = False

        # Diagnostics storage (RMS ratios, attention mass, etc.)
        self._last_diagnostics: Optional[Dict[str, float]] = None

        # Return format expected by the downstream decoder layer.
        # This varies across HF/transformers versions (some layers unpack 2 values, some 3).
        # We detect it lazily by calling the original attention once on first augmented forward.
        self._return_format: Optional[int] = None

    def set_audio(
        self,
        audio_tokens: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        gate: float = 1.0,
    ):
        """Set audio tokens for the current forward pass."""
        self._audio_tokens = audio_tokens
        self._audio_mask = audio_mask
        self._gate = gate

    def clear_audio(self, preserve_attention_weights: bool = False):
        """Clear audio tokens after forward pass.

        Args:
            preserve_attention_weights: If True, keep attention weights for later access
        """
        self._audio_tokens = None
        self._audio_mask = None
        self._gate = 1.0
        # Always clear live weights (they should only exist for one forward pass)
        self._live_attention_weights = None
        if not preserve_attention_weights:
            self._last_attention_weights = None

    def set_return_attention_weights(self, return_weights: bool):
        """Enable/disable attention weight capture for regularization."""
        self._return_attention_weights = return_weights

    def get_last_attention_weights(self) -> Optional[torch.Tensor]:
        """Get detached attention weights from last forward pass (for logging/pooling)."""
        return self._last_attention_weights

    def get_live_attention_weights(self) -> Optional[torch.Tensor]:
        """Get live attention weights with gradients for reg loss computation.

        IMPORTANT: This returns the tensor and clears it immediately to prevent
        graph retention across steps. Call this only once per forward pass.
        """
        weights = self._live_attention_weights
        self._live_attention_weights = None  # Clear immediately to prevent memory leak
        return weights

    def get_diagnostics(self) -> Optional[Dict[str, float]]:
        """Get diagnostics from last forward pass (RMS ratios, attention mass, etc.)."""
        return self._last_diagnostics

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with optional audio KV augmentation.

        If no audio tokens are set, delegates to original attention.
        Otherwise, computes attention with audio K,V concatenated.
        """
        # If no audio, pass through to original
        if self._audio_tokens is None:
            return self.original_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        # Compute augmented attention with audio
        return self._forward_with_audio_kv(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions or self._return_attention_weights,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    def _forward_with_audio_kv(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor]],
        output_attentions: bool,
        use_cache: bool,
        cache_position: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute attention with SEPARATE text and audio branches.

        Key architecture:
        1. TEXT ATTENTION: Completely frozen, identical to original LlamaAttention
        2. AUDIO ATTENTION: Uses adapted queries (Q + ΔQ) to attend to audio K,V
        3. COMBINE: text_output + gate * audio_output

        This preserves the base model's text processing while adding audio capability.
        """
        bsz, q_len, _ = hidden_states.size()
        n_audio = self._audio_tokens.size(1)
        orig_attn = self.original_attention

        # Lazily determine how many values the original attention returns so we can
        # match what the decoder layer expects to unpack.
        if self._return_format is None:
            try:
                with torch.no_grad():
                    probe = orig_attn(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        **kwargs,
                    )
                if isinstance(probe, tuple):
                    self._return_format = len(probe)
                else:
                    self._return_format = 1
            except Exception:
                # Most modern LLaMA decoder layers unpack 2 values.
                self._return_format = 2

        # ============================================================
        # 1. TEXT ATTENTION (frozen, identical to original LlamaAttention)
        # ============================================================
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        query_states = orig_attn.q_proj(hidden_states)
        key_states = orig_attn.k_proj(hidden_states)
        value_states = orig_attn.v_proj(hidden_states)

        # Apply QK norms if present (Qwen3 uses RMSNorm on Q,K heads)
        if hasattr(orig_attn, 'q_norm'):
            query_states = orig_attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if hasattr(orig_attn, 'k_norm'):
            key_states = orig_attn.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        else:
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to text Q, K (standard LLaMA)
        cos, sin = None, None
        if hasattr(orig_attn, 'rotary_emb'):
            cos, sin = orig_attn.rotary_emb(value_states, position_ids)
            query_states, key_states = self._apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Update KV cache for generation (concatenates past K,V with current)
        if past_key_value is not None:
            cache_kwargs = {}
            if sin is not None:
                cache_kwargs["sin"] = sin
                cache_kwargs["cos"] = cos
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            layer_idx = getattr(orig_attn, 'layer_idx', getattr(self, 'layer_idx', 0))
            key_states, value_states = past_key_value.update(
                key_states, value_states, layer_idx, cache_kwargs
            )

        # GQA expansion for text K, V
        key_states_expanded = key_states
        value_states_expanded = value_states
        if self.num_key_value_groups > 1:
            key_states_expanded = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states_expanded = self._repeat_kv(value_states, self.num_key_value_groups)

        # Text attention (standard causal)
        text_attn_weights = torch.matmul(query_states, key_states_expanded.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            text_attn_weights = text_attn_weights + attention_mask
        text_attn_weights = torch.clamp(text_attn_weights, min=-50.0, max=50.0)
        text_attn_weights = F.softmax(text_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        text_output = torch.matmul(text_attn_weights, value_states_expanded)

        # ============================================================
        # 2. AUDIO ATTENTION (trainable adapter)
        # ============================================================
        # Adapted query for audio: Q_audio = Q_frozen + ΔQ_adapter(H)
        # CRITICAL: Apply RoPE to ΔQ to match query_states coordinate space
        delta_q = self.kv_adapter.audio_query_adapter(hidden_states)
        delta_q = delta_q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to delta_q (same rotation as query_states)
        # This ensures ΔQ is in the same coordinate space as post-RoPE queries
        if cos is not None and sin is not None:
            # Use the same cos, sin that was applied to query_states
            delta_q = self._apply_rotary_pos_emb_single(delta_q, cos, sin)

        query_for_audio = query_states + delta_q  # Q + ΔQ (both post-RoPE)

        # Audio K, V (no RoPE - position agnostic, no gate - applied at combine step)
        audio_keys, audio_values = self.kv_adapter(self._audio_tokens)
        # Use adapter's num_key_value_heads (may differ from LLM's for GQA)
        adapter_num_kv_heads = self.kv_adapter.num_key_value_heads

        # Handle batch size mismatch between audio and hidden_states
        # This can happen with beam search expansion OR 8-bit quantization + gradient checkpointing
        audio_bsz = audio_keys.size(0)
        if bsz != audio_bsz:
            if bsz > audio_bsz and bsz % audio_bsz == 0:
                # Beam search expansion: audio batch smaller, expand to match
                num_beams = bsz // audio_bsz
                audio_keys = audio_keys.unsqueeze(1).expand(-1, num_beams, -1, -1).reshape(bsz, n_audio, -1)
                audio_values = audio_values.unsqueeze(1).expand(-1, num_beams, -1, -1).reshape(bsz, n_audio, -1)
            elif audio_bsz > bsz and audio_bsz % bsz == 0:
                # 8-bit quantization quirk: hidden_states has smaller batch, slice audio
                # This happens during gradient checkpointing recompute with quantized models
                audio_keys = audio_keys[:bsz]
                audio_values = audio_values[:bsz]
            else:
                # Unexpected mismatch - log warning and try to continue
                if not hasattr(self, '_batch_mismatch_warned'):
                    self._batch_mismatch_warned = True
                    print(f"[KVWrapper] Warning: batch size mismatch: hidden_states={bsz}, audio={audio_bsz}", flush=True)
                # Use audio batch size for reshape
                bsz = audio_bsz

        audio_keys = audio_keys.view(bsz, n_audio, adapter_num_kv_heads, self.head_dim).transpose(1, 2)
        audio_values = audio_values.view(bsz, n_audio, adapter_num_kv_heads, self.head_dim).transpose(1, 2)

        # Cast audio K,V to match query dtype (handles fp16/bf16 models)
        audio_keys = audio_keys.to(query_for_audio.dtype)
        audio_values = audio_values.to(query_for_audio.dtype)

        # GQA expansion for audio K, V
        # Compute groups based on adapter's num_kv_heads (not LLM's)
        adapter_kv_groups = self.num_heads // adapter_num_kv_heads
        if adapter_kv_groups > 1:
            audio_keys = self._repeat_kv(audio_keys, adapter_kv_groups)
            audio_values = self._repeat_kv(audio_values, adapter_kv_groups)

        # Audio attention (non-causal, all positions can attend to all audio)
        audio_attn_weights = torch.matmul(query_for_audio, audio_keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # Apply audio mask if provided: 1=attend, 0=mask.
        if self._audio_mask is not None:
            audio_mask = self._audio_mask.to(device=audio_attn_weights.device)
            # Handle batch size mismatch (same logic as K/V handling above)
            if audio_mask.dim() == 2:
                mask_bsz = audio_mask.size(0)
                if mask_bsz != bsz:
                    if bsz > mask_bsz and bsz % mask_bsz == 0:
                        # Beam search expansion
                        num_beams = bsz // mask_bsz
                        audio_mask = audio_mask.unsqueeze(1).expand(-1, num_beams, -1).reshape(bsz, -1)
                    elif mask_bsz > bsz:
                        # 8-bit quantization quirk: slice to match
                        audio_mask = audio_mask[:bsz]
                audio_mask_expanded = audio_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,Ta)
                audio_attn_weights = audio_attn_weights.masked_fill(audio_mask_expanded <= 0.5, float("-inf"))
        audio_attn_weights = torch.clamp(audio_attn_weights, min=-50.0, max=50.0)
        audio_attn_weights = F.softmax(audio_attn_weights, dim=-1, dtype=torch.float32).to(query_for_audio.dtype)
        audio_output = torch.matmul(audio_attn_weights, audio_values)

        # Store audio attention weights for regularization and logging
        # - Detached copy for logging/pooling (safe, no graph retention)
        # - Live copy for reg loss computation (cleared immediately after use)
        if self._return_attention_weights:
            self._last_attention_weights = audio_attn_weights.detach()  # For logging
            self._live_attention_weights = audio_attn_weights  # For reg loss (has gradients)

        # ============================================================
        # 3. COMBINE: text_output + gate * audio_output
        # ============================================================
        gated_audio_output = self._gate * audio_output
        combined_output = text_output + gated_audio_output

        # ============================================================
        # DIAGNOSTICS: Store RMS values for monitoring
        # ============================================================
        with torch.no_grad():
            text_rms = torch.sqrt(torch.mean(text_output.float() ** 2)).item()
            audio_rms = torch.sqrt(torch.mean(gated_audio_output.float() ** 2)).item()
            rms_ratio = audio_rms / (text_rms + 1e-8)

            # NOTE: In separate-branch architecture, audio attention always sums to 1.0
            # (softmax over audio tokens). This is meaningless as a metric.
            # Instead, use ENTROPY (should decrease with training) and ΔQ/Q ratio.

            # Audio attention entropy (H/log(n) normalized)
            # ~1.0 = uniform (bad if stays), <1.0 = focused (good)
            attn_dist = audio_attn_weights.mean(dim=(0, 1))  # Average over batch, heads -> (seq, n_audio)
            attn_dist = attn_dist.clamp(min=1e-10)
            entropy_per_pos = -(attn_dist * attn_dist.log()).sum(dim=-1)  # (seq,)
            mean_entropy = entropy_per_pos.mean().item()
            max_entropy = math.log(n_audio) if n_audio > 1 else 1.0
            normalized_entropy = mean_entropy / max_entropy

            # ΔQ/Q ratio for checking if query adapter is meaningful
            delta_q_rms = torch.sqrt(torch.mean(delta_q.float() ** 2)).item()
            q_rms = torch.sqrt(torch.mean(query_states.float() ** 2)).item()

            # Store for external access
            self._last_diagnostics = {
                "text_rms": text_rms,
                "audio_rms": audio_rms,
                "rms_ratio": rms_ratio,
                "normalized_entropy": normalized_entropy,  # Replaces meaningless audio_attn_mass
                "gate": self._gate,
                # ΔQ/Q diagnostics
                "delta_q_rms": delta_q_rms,
                "q_rms": q_rms,
                # Store attention weights for external entropy calculation if needed
                "audio_attn_weights": audio_attn_weights.detach().clone(),
            }

        # Reshape and output projection (frozen)
        combined_output = combined_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = orig_attn.o_proj(combined_output)

        # Return format for LlamaDecoderLayer
        if output_attentions:
            # Return audio attention weights (text weights available as text_attn_weights)
            attn_weights_out = audio_attn_weights
        else:
            attn_weights_out = None

        # Match original attention return format expected by the decoder.
        if self._return_format == 1:
            return attn_output
        if self._return_format == 2:
            return (attn_output, attn_weights_out)
        # Default: 3-tuple (attn_output, attn_weights, past_key_value)
        return (attn_output, attn_weights_out, None)

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to Q and K."""
        # Standard LLaMA rotary embedding application
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _apply_rotary_pos_emb_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary position embeddings to a single tensor (for ΔQ)."""
        return (x * cos) + (self._rotate_half(x) * sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for grouped-query attention."""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def _extend_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        n_audio: int,
        bsz: int,
        q_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Extend attention mask to include audio positions.

        Audio tokens should be visible to ALL query positions (non-causal).
        Text tokens maintain causal masking.

        Args:
            attention_mask: Original mask (bsz, 1, q_len, q_len) or (bsz, 1, 1, q_len)
            n_audio: Number of audio tokens

        Returns:
            Extended mask (bsz, 1, q_len, q_len + n_audio)
        """
        # Create audio attention extension: all positions can see all audio tokens
        # 0 = attend, large negative = mask (additive mask format)
        audio_extension = torch.zeros(
            bsz, 1, q_len, n_audio, device=device, dtype=dtype
        )

        # Apply audio mask if provided (mask out padded/invalid audio)
        if self._audio_mask is not None:
            # audio_mask: (bsz, n_audio) where 1=attend, 0=mask
            audio_mask_expanded = self._audio_mask.unsqueeze(1).unsqueeze(2)  # (bsz, 1, 1, n_audio)
            audio_mask_expanded = audio_mask_expanded.expand(-1, -1, q_len, -1)
            # Convert to additive mask: 0 where attend, -inf where mask
            audio_extension = torch.where(
                audio_mask_expanded > 0.5,
                torch.zeros_like(audio_extension),
                torch.full_like(audio_extension, float('-inf'))
            )

        if attention_mask is None:
            # No original mask, just return audio extension with text causal mask
            # Create causal mask for text
            causal_mask = torch.triu(
                torch.full((q_len, q_len), float('-inf'), device=device, dtype=dtype),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1)
            return torch.cat([causal_mask, audio_extension], dim=-1)

        # Concatenate original mask with audio extension
        return torch.cat([attention_mask, audio_extension], dim=-1)


class KVAugmentationHookManager:
    """
    Manages attention module replacement for KV augmentation.

    Unlike the current LayerHookManager which uses forward hooks,
    this manager replaces attention modules entirely to enable
    modification during the attention computation.
    """

    def __init__(
        self,
        model: nn.Module,
        kv_adapters: nn.ModuleDict,
        fusion_layer_indices: List[int],
    ):
        self.model = model
        self.kv_adapters = kv_adapters
        self.fusion_layers = fusion_layer_indices
        self.layer_modules = self._discover_layer_modules(model)
        self.original_attentions: Dict[int, nn.Module] = {}
        self.wrapped_attentions: Dict[int, KVAugmentedAttention] = {}
        self._is_wrapped = False
        # Diagnostics alerting (helps catch collapse/dominance early)
        self._alerts_enabled: bool = True
        self._alert_log_every: int = 100
        self._alert_counter: int = 0
        self._rms_ratio_low: float = 0.01
        self._rms_ratio_high: float = 0.40
        self._entropy_low: float = 0.15
        self._entropy_high: float = 0.98

    def configure_alerts(
        self,
        *,
        enabled: bool = True,
        log_every: int = 100,
        rms_ratio_low: float = 0.01,
        rms_ratio_high: float = 0.40,
        entropy_low: float = 0.15,
        entropy_high: float = 0.98,
    ) -> None:
        """Configure threshold-based diagnostics alerts."""
        self._alerts_enabled = bool(enabled)
        self._alert_log_every = max(1, int(log_every))
        self._rms_ratio_low = float(rms_ratio_low)
        self._rms_ratio_high = float(rms_ratio_high)
        self._entropy_low = float(entropy_low)
        self._entropy_high = float(entropy_high)

    def _discover_layer_modules(self, model: nn.Module) -> Dict[int, nn.Module]:
        """
        Locate decoder layer ModuleList.

        Handles various model wrapper structures (LLaVA, LLaMA, etc.)
        """
        def _is_module_list(obj):
            return isinstance(obj, (list, nn.ModuleList, tuple))

        def _try_extract(candidate):
            if candidate is None:
                return None
            if hasattr(candidate, "layers") and _is_module_list(getattr(candidate, "layers")):
                return {i: layer for i, layer in enumerate(getattr(candidate, "layers"))}
            if hasattr(candidate, "h") and _is_module_list(getattr(candidate, "h")):
                return {i: layer for i, layer in enumerate(getattr(candidate, "h"))}
            return None

        # Search through common wrapper structures
        seeds = [
            model,
            getattr(model, "model", None),
            getattr(model, "language_model", None),
            getattr(model, "decoder", None),
            getattr(model, "transformer", None),
        ]

        seen = set()
        queue = [s for s in seeds if s is not None]
        expand_attrs = ("model", "language_model", "decoder", "transformer")
        max_visits = 50

        visits = 0
        while queue and visits < max_visits:
            candidate = queue.pop(0)
            visits += 1
            key = id(candidate)
            if key in seen:
                continue
            seen.add(key)

            extracted = _try_extract(candidate)
            if extracted is not None:
                return extracted

            for attr in expand_attrs:
                child = getattr(candidate, attr, None)
                if child is not None and id(child) not in seen:
                    queue.append(child)

        raise ValueError(
            f"Unable to locate decoder layers. Tried {visits} candidates."
        )

    def wrap_attention_modules(self):
        """Replace LlamaAttention with KVAugmentedAttention at fusion layers."""
        if self._is_wrapped:
            return

        for idx in self.fusion_layers:
            if idx not in self.layer_modules:
                print(f"[KVAugment] Warning: layer {idx} not found, skipping", flush=True)
                continue

            layer = self.layer_modules[idx]

            # Find the attention module (self_attn)
            if hasattr(layer, 'self_attn'):
                original_attn = layer.self_attn
            elif hasattr(layer, 'attention'):
                original_attn = layer.attention
            else:
                print(f"[KVAugment] Warning: no attention found in layer {idx}", flush=True)
                continue

            # Store original
            self.original_attentions[idx] = original_attn

            # Create wrapped attention
            adapter = self.kv_adapters[str(idx)]
            wrapped = KVAugmentedAttention(original_attn, adapter, idx)
            self.wrapped_attentions[idx] = wrapped

            # Replace in layer
            if hasattr(layer, 'self_attn'):
                layer.self_attn = wrapped
            else:
                layer.attention = wrapped

        self._is_wrapped = True

    def unwrap_attention_modules(self):
        """Restore original attention modules."""
        if not self._is_wrapped:
            return

        for idx, original in self.original_attentions.items():
            layer = self.layer_modules[idx]
            if hasattr(layer, 'self_attn'):
                layer.self_attn = original
            else:
                layer.attention = original

        self.original_attentions.clear()
        self.wrapped_attentions.clear()
        self._is_wrapped = False

    def inject_audio(
        self,
        audio_tokens: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        gate: float = 1.0,
    ):
        """Set audio tokens for current forward pass in all wrapped attentions.

        NOTE: We do NOT clear audio tokens after forward anymore because gradient
        checkpointing recomputes forward during backward. If audio tokens are cleared
        after forward but before backward, the recomputed forward won't have audio
        and gradients won't flow to kv_adapters.

        Audio tokens persist until the next inject_audio() call.
        """
        for wrapped in self.wrapped_attentions.values():
            wrapped.set_audio(audio_tokens, audio_mask, gate)

    def clear_audio(self, preserve_attention_weights: bool = False):
        """Clear audio tokens after forward pass.

        Args:
            preserve_attention_weights: If True, keep attention weights for later access
        """
        for wrapped in self.wrapped_attentions.values():
            wrapped.clear_audio(preserve_attention_weights=preserve_attention_weights)

    def set_return_attention_weights(self, return_weights: bool):
        """Enable/disable attention weight capture."""
        for wrapped in self.wrapped_attentions.values():
            wrapped.set_return_attention_weights(return_weights)

    def get_attention_weights(self) -> Dict[int, torch.Tensor]:
        """Get detached attention weights from all layers (for logging/pooling)."""
        weights = {}
        for idx, wrapped in self.wrapped_attentions.items():
            w = wrapped.get_last_attention_weights()
            if w is not None:
                weights[idx] = w
        return weights

    def get_live_attention_weights(self) -> Dict[int, torch.Tensor]:
        """Get live attention weights with gradients for reg loss computation.

        IMPORTANT: This returns the tensors and clears them immediately to prevent
        graph retention across steps. Call this only once per forward pass, right
        before computing the regularization loss.

        Returns:
            Dict mapping layer_idx to attention weight tensors (with gradients)
        """
        weights = {}
        for idx, wrapped in self.wrapped_attentions.items():
            w = wrapped.get_live_attention_weights()  # Gets and clears
            if w is not None:
                weights[idx] = w
        return weights

    def get_diagnostics(self) -> Dict[int, Dict[str, float]]:
        """
        Get diagnostics from all layers.

        Returns dict mapping layer_idx to diagnostics dict with:
        - text_rms: RMS of text attention output
        - audio_rms: RMS of gated audio output
        - rms_ratio: audio_rms / text_rms (want 1%-10%)
        - normalized_entropy: mean normalized entropy of audio attention
        - gate: current gate value
        """
        diagnostics = {}
        for idx, wrapped in self.wrapped_attentions.items():
            d = wrapped.get_diagnostics()
            if d is not None:
                diagnostics[idx] = d
        return diagnostics

    def log_diagnostics(self, prefix: str = "") -> Dict[str, float]:
        """
        Get flattened diagnostics dict suitable for logging to wandb/tensorboard.

        Returns dict with keys like:
        - {prefix}layer_12/rms_ratio
        - {prefix}layer_12/normalized_entropy
        - {prefix}layer_12/delta_q_ratio
        - {prefix}mean/rms_ratio
        - {prefix}mean/normalized_entropy
        - {prefix}mean/delta_q_ratio
        """
        all_diag = self.get_diagnostics()
        if not all_diag:
            return {}

        log_dict = {}
        rms_ratios = []
        entropies = []
        dq_ratios = []

        for layer_idx, diag in sorted(all_diag.items()):
            layer_prefix = f"{prefix}layer_{layer_idx}/"
            log_dict[f"{layer_prefix}text_rms"] = diag["text_rms"]
            log_dict[f"{layer_prefix}audio_rms"] = diag["audio_rms"]
            log_dict[f"{layer_prefix}rms_ratio"] = diag["rms_ratio"]
            rms_ratios.append(diag["rms_ratio"])

            # Normalized entropy (replaces meaningless audio_attn_mass)
            if "normalized_entropy" in diag:
                log_dict[f"{layer_prefix}normalized_entropy"] = diag["normalized_entropy"]
                entropies.append(diag["normalized_entropy"])

            # ΔQ/Q ratio
            if "delta_q_rms" in diag and "q_rms" in diag:
                q_rms = max(diag["q_rms"], 1e-8)
                dq_ratio = diag["delta_q_rms"] / q_rms
                log_dict[f"{layer_prefix}delta_q_ratio"] = dq_ratio
                dq_ratios.append(dq_ratio)

        # Averages across layers
        if rms_ratios:
            log_dict[f"{prefix}mean/rms_ratio"] = sum(rms_ratios) / len(rms_ratios)
        if entropies:
            log_dict[f"{prefix}mean/normalized_entropy"] = sum(entropies) / len(entropies)
        if dq_ratios:
            log_dict[f"{prefix}mean/delta_q_ratio"] = sum(dq_ratios) / len(dq_ratios)

        # Alert metrics and sparse console warnings
        self._alert_counter += 1
        mean_rms = log_dict.get(f"{prefix}mean/rms_ratio")
        mean_entropy = log_dict.get(f"{prefix}mean/normalized_entropy")
        if mean_rms is not None:
            log_dict[f"{prefix}alert/rms_too_low"] = float(mean_rms < self._rms_ratio_low)
            log_dict[f"{prefix}alert/rms_too_high"] = float(mean_rms > self._rms_ratio_high)
        if mean_entropy is not None:
            log_dict[f"{prefix}alert/entropy_too_low"] = float(mean_entropy < self._entropy_low)
            log_dict[f"{prefix}alert/entropy_too_high"] = float(mean_entropy > self._entropy_high)

        if (
            self._alerts_enabled
            and (self._alert_counter % self._alert_log_every) == 0
            and mean_rms is not None
            and mean_entropy is not None
        ):
            warn_parts = []
            if mean_rms < self._rms_ratio_low:
                warn_parts.append(f"rms_ratio too low ({mean_rms:.4f} < {self._rms_ratio_low:.4f})")
            if mean_rms > self._rms_ratio_high:
                warn_parts.append(f"rms_ratio too high ({mean_rms:.4f} > {self._rms_ratio_high:.4f})")
            if mean_entropy < self._entropy_low:
                warn_parts.append(f"entropy too low ({mean_entropy:.4f} < {self._entropy_low:.4f})")
            if mean_entropy > self._entropy_high:
                warn_parts.append(f"entropy too high ({mean_entropy:.4f} > {self._entropy_high:.4f})")
            if warn_parts:
                print(f"[KVAugmentAlert] {'; '.join(warn_parts)}", flush=True)

        return log_dict


class MinAudioAttentionLoss(nn.Module):
    """
    Regularization loss that ensures answer-decision tokens attend to audio.

    This prevents the model from learning to ignore audio entirely.
    Uses hinge loss: penalizes if attention to audio is below threshold.

    Key improvements:
    - Only applies to last-k tokens (answer-decision positions) by default
    - Supports curriculum: strong early, decay later
    """

    def __init__(
        self,
        min_attention: float = 0.1,
        loss_weight: float = 1.0,
        answer_tokens_k: int = 8,  # Only apply to last k tokens (answer-decision)
        min_entropy: float = 0.20,
        max_entropy: float = 0.98,
        max_token_attention: float = 0.95,
        entropy_weight: float = 1.0,
        dominance_weight: float = 0.5,
    ):
        super().__init__()
        self.min_attention = min_attention
        self.loss_weight = loss_weight
        self.answer_tokens_k = answer_tokens_k
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy
        self.max_token_attention = max_token_attention
        self.entropy_weight = entropy_weight
        self.dominance_weight = dominance_weight

        # For curriculum scheduling
        self._initial_min_attention = min_attention
        self._initial_loss_weight = loss_weight

    def update_curriculum(self, step: int, warmup_steps: int = 1000, decay_steps: int = 2000):
        """
        Update min_attention and loss_weight based on training step.

        Curriculum:
        - steps 0 to warmup_steps: full strength (0.1 / 1.0)
        - steps warmup_steps to decay_steps: linear decay to (0.01 / 0.1)
        - steps > decay_steps: stay at (0.01 / 0.1)
        """
        if step < warmup_steps:
            # Full strength during warm-up
            self.min_attention = self._initial_min_attention
            self.loss_weight = self._initial_loss_weight
        elif step < decay_steps:
            # Linear decay
            progress = (step - warmup_steps) / (decay_steps - warmup_steps)
            self.min_attention = self._initial_min_attention * (1 - 0.9 * progress)  # 0.1 -> 0.01
            self.loss_weight = self._initial_loss_weight * (1 - 0.9 * progress)  # 1.0 -> 0.1
        else:
            # Final values
            self.min_attention = 0.01
            self.loss_weight = 0.1

    def forward(
        self,
        attention_weights: Dict[int, torch.Tensor],
        n_audio: int,
        supervised_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute minimum attention regularization loss.

        Args:
            attention_weights: Dict mapping layer_idx to attention weights
                              Each tensor is (bsz, heads, q_len, k_len)
            n_audio: Number of audio tokens (last n_audio positions in K)
            supervised_mask: (bsz, q_len) mask where 1 = supervised token (answer)
                           If None, uses last answer_tokens_k positions

        Returns:
            Scalar loss tensor
        """
        if not attention_weights:
            return torch.tensor(0.0)

        total_loss = 0.0
        count = 0

        for layer_idx, attn in attention_weights.items():
            bsz, heads, q_len, k_len = attn.shape

            # In KV separate-branch mode, `attn` is already attention over audio tokens.
            audio_attn = attn  # (bsz, heads, q_len, n_audio)

            # Create mask for answer-decision tokens (last k positions)
            if supervised_mask is not None:
                mask = supervised_mask.unsqueeze(1).float()  # (bsz, 1, q_len)
            else:
                # Default: last answer_tokens_k positions
                mask = torch.zeros(bsz, 1, q_len, device=attn.device, dtype=attn.dtype)
                k = min(self.answer_tokens_k, q_len)
                mask[:, :, -k:] = 1.0

            # Per-position valid count (answer tokens only)
            denom = (mask.sum() * heads).clamp(min=1.0)

            # 1) Entropy-band regularization to avoid collapse or full-uniform attention.
            # Normalize by log(K) so thresholds are stable across token counts.
            p = audio_attn.clamp(min=1e-8)
            entropy = -(p * p.log()).sum(dim=-1)  # (bsz, heads, q_len)
            norm_denom = math.log(float(max(k_len, 2)))
            normalized_entropy = entropy / norm_denom
            entropy_low = F.relu(self.min_entropy - normalized_entropy)
            entropy_high = F.relu(normalized_entropy - self.max_entropy)
            entropy_loss = ((entropy_low + entropy_high) * mask).sum() / denom

            # 2) Dominance regularization: discourage single-token takeover.
            max_token_mass = audio_attn.max(dim=-1).values  # (bsz, heads, q_len)
            dominance_deficit = F.relu(max_token_mass - self.max_token_attention)
            dominance_loss = (dominance_deficit * mask).sum() / denom

            # 3) Legacy minimum-attention term (kept for backward compatibility).
            # In separate-branch mode sum(attn)=1, so this contributes only when
            # users set min_attention > 1 or custom wrappers change behavior.
            sum_audio_attn = audio_attn.sum(dim=-1)
            min_attn_deficit = F.relu(self.min_attention - sum_audio_attn)
            min_attn_loss = (min_attn_deficit * mask).sum() / denom

            layer_loss = (
                min_attn_loss
                + self.entropy_weight * entropy_loss
                + self.dominance_weight * dominance_loss
            )

            total_loss = total_loss + layer_loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=next(iter(attention_weights.values())).device)

        return self.loss_weight * (total_loss / count)

    def get_current_params(self) -> Dict[str, float]:
        """Get current curriculum parameters for logging."""
        return {
            "min_audio_attention": self.min_attention,
            "min_audio_attention_weight": self.loss_weight,
            "min_audio_entropy": self.min_entropy,
            "max_audio_entropy": self.max_entropy,
            "max_audio_token_attention": self.max_token_attention,
            "entropy_weight": self.entropy_weight,
            "dominance_weight": self.dominance_weight,
        }


# =============================================================================
# Multi-Modal KV Augmentation
# =============================================================================


class MultiModalKVAugmentedAttention(nn.Module):
    """
    Extension of KVAugmentedAttention that handles multiple modalities.

    Each modality has its own KVAugmentationAdapter, and all modality outputs
    are summed together:

        output = text_output + gate_audio * audio_output + gate_pc * pc_output + ...

    This allows each modality to:
    - Have its own learned K,V projections
    - Have its own query adapter (ΔQ)
    - Be independently gated

    Architecture:
        1. TEXT ATTENTION: Frozen, identical to original LlamaAttention
        2. For each modality:
           - Compute Q + ΔQ_modality
           - Attend to modality K,V
           - Gate the output
        3. COMBINE: text_output + sum(gated_modality_outputs)
    """

    def __init__(
        self,
        original_attention: nn.Module,
        layer_idx: int,
    ):
        super().__init__()
        self.original_attention = original_attention
        self.layer_idx = layer_idx

        # Modality adapters will be registered dynamically
        self._modality_adapters: Dict[str, KVAugmentationAdapter] = {}

        # Storage for current forward pass modality data
        self._modality_tokens: Dict[str, torch.Tensor] = {}
        self._modality_masks: Dict[str, Optional[torch.Tensor]] = {}
        self._modality_gates: Dict[str, float] = {}

        # Copy attributes from original attention
        q_proj = getattr(original_attention, 'q_proj', None)
        k_proj = getattr(original_attention, 'k_proj', None)

        self.head_dim = getattr(original_attention, 'head_dim', 128)

        num_heads = getattr(original_attention, 'num_heads', None)
        if num_heads is None and q_proj is not None and hasattr(q_proj, 'out_features'):
            num_heads = q_proj.out_features // self.head_dim
        self.num_heads = num_heads if num_heads is not None else 32

        num_kv_heads = getattr(original_attention, 'num_key_value_heads', None)
        if num_kv_heads is None and k_proj is not None and hasattr(k_proj, 'out_features'):
            num_kv_heads = k_proj.out_features // self.head_dim
        self.num_key_value_heads = num_kv_heads if num_kv_heads is not None else self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Diagnostics and attention weights storage (per modality)
        self._last_diagnostics: Dict[str, Dict[str, float]] = {}
        self._return_format: Optional[int] = None

    def register_modality_adapter(self, modality: str, adapter: KVAugmentationAdapter):
        """Register an adapter for a specific modality."""
        self._modality_adapters[modality] = adapter

    def set_modality_tokens(
        self,
        modality: str,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        gate: float = 1.0,
    ):
        """Set tokens for a specific modality for current forward pass."""
        self._modality_tokens[modality] = tokens
        self._modality_masks[modality] = mask
        self._modality_gates[modality] = gate

    def clear_modality_tokens(self):
        """Clear all modality tokens after forward pass."""
        self._modality_tokens.clear()
        self._modality_masks.clear()
        self._modality_gates.clear()

    def get_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """Get diagnostics per modality."""
        return self._last_diagnostics

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass with optional multi-modal KV augmentation."""
        # If no modality tokens, pass through to original
        if not self._modality_tokens:
            return self.original_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        return self._forward_with_multimodal_kv(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    def _forward_with_multimodal_kv(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor]],
        output_attentions: bool,
        use_cache: bool,
        cache_position: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute attention with multiple modality K,V branches.

        Architecture:
        1. TEXT ATTENTION: Frozen, identical to original
        2. For each modality: Q + ΔQ_mod -> attend to mod K,V -> gate
        3. COMBINE: text_output + sum(gated_modality_outputs)
        """
        bsz, q_len, _ = hidden_states.size()
        orig_attn = self.original_attention

        # Determine return format lazily
        if self._return_format is None:
            try:
                with torch.no_grad():
                    probe = orig_attn(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        **kwargs,
                    )
                self._return_format = len(probe) if isinstance(probe, tuple) else 1
            except Exception:
                self._return_format = 2

        # ============================================================
        # 1. TEXT ATTENTION (frozen)
        # ============================================================
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        query_states = orig_attn.q_proj(hidden_states)
        key_states = orig_attn.k_proj(hidden_states)
        value_states = orig_attn.v_proj(hidden_states)

        # Apply QK norms if present (Qwen3 uses RMSNorm on Q,K heads)
        if hasattr(orig_attn, 'q_norm'):
            query_states = orig_attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if hasattr(orig_attn, 'k_norm'):
            key_states = orig_attn.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        else:
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = None, None
        if hasattr(orig_attn, 'rotary_emb'):
            cos, sin = orig_attn.rotary_emb(value_states, position_ids)
            query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update KV cache for generation (concatenates past K,V with current)
        if past_key_value is not None:
            cache_kwargs = {}
            if sin is not None:
                cache_kwargs["sin"] = sin
                cache_kwargs["cos"] = cos
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            layer_idx = getattr(orig_attn, 'layer_idx', self.layer_idx)
            key_states, value_states = past_key_value.update(
                key_states, value_states, layer_idx, cache_kwargs
            )

        # GQA expansion
        key_states_expanded = key_states
        value_states_expanded = value_states
        if self.num_key_value_groups > 1:
            key_states_expanded = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states_expanded = self._repeat_kv(value_states, self.num_key_value_groups)

        # Text attention
        text_attn_weights = torch.matmul(query_states, key_states_expanded.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            text_attn_weights = text_attn_weights + attention_mask
        text_attn_weights = torch.clamp(text_attn_weights, min=-50.0, max=50.0)
        text_attn_weights = F.softmax(text_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        text_output = torch.matmul(text_attn_weights, value_states_expanded)

        # ============================================================
        # 2. MODALITY ATTENTION (for each modality)
        # ============================================================
        combined_modality_output = torch.zeros_like(text_output)
        self._last_diagnostics = {}

        for modality, mod_tokens in self._modality_tokens.items():
            if modality not in self._modality_adapters:
                continue

            adapter = self._modality_adapters[modality]
            gate = self._modality_gates.get(modality, 1.0)
            mod_mask = self._modality_masks.get(modality)
            n_mod = mod_tokens.size(1)

            # Adapted query: Q + ΔQ_modality
            delta_q = adapter.audio_query_adapter(hidden_states)
            delta_q = delta_q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Apply RoPE to delta_q
            if cos is not None and sin is not None:
                delta_q = self._apply_rotary_pos_emb_single(delta_q, cos, sin)

            query_for_mod = query_states + delta_q

            # Modality K, V
            mod_keys, mod_values = adapter(mod_tokens)
            adapter_num_kv_heads = adapter.num_key_value_heads

            # Handle batch size mismatch
            mod_bsz = mod_keys.size(0)
            if bsz != mod_bsz:
                if bsz > mod_bsz and bsz % mod_bsz == 0:
                    num_beams = bsz // mod_bsz
                    mod_keys = mod_keys.unsqueeze(1).expand(-1, num_beams, -1, -1).reshape(bsz, n_mod, -1)
                    mod_values = mod_values.unsqueeze(1).expand(-1, num_beams, -1, -1).reshape(bsz, n_mod, -1)
                elif mod_bsz > bsz:
                    mod_keys = mod_keys[:bsz]
                    mod_values = mod_values[:bsz]

            mod_keys = mod_keys.view(bsz, n_mod, adapter_num_kv_heads, self.head_dim).transpose(1, 2)
            mod_values = mod_values.view(bsz, n_mod, adapter_num_kv_heads, self.head_dim).transpose(1, 2)

            mod_keys = mod_keys.to(query_for_mod.dtype)
            mod_values = mod_values.to(query_for_mod.dtype)

            # GQA expansion
            adapter_kv_groups = self.num_heads // adapter_num_kv_heads
            if adapter_kv_groups > 1:
                mod_keys = self._repeat_kv(mod_keys, adapter_kv_groups)
                mod_values = self._repeat_kv(mod_values, adapter_kv_groups)

            # Modality attention
            mod_attn_weights = torch.matmul(query_for_mod, mod_keys.transpose(2, 3)) / math.sqrt(self.head_dim)

            # Apply modality mask if provided
            if mod_mask is not None:
                mod_mask_dev = mod_mask.to(device=mod_attn_weights.device)
                if mod_mask_dev.dim() == 2:
                    mask_bsz = mod_mask_dev.size(0)
                    if mask_bsz != bsz:
                        if bsz > mask_bsz and bsz % mask_bsz == 0:
                            num_beams = bsz // mask_bsz
                            mod_mask_dev = mod_mask_dev.unsqueeze(1).expand(-1, num_beams, -1).reshape(bsz, -1)
                        elif mask_bsz > bsz:
                            mod_mask_dev = mod_mask_dev[:bsz]
                    mod_mask_expanded = mod_mask_dev.unsqueeze(1).unsqueeze(2)
                    mod_attn_weights = mod_attn_weights.masked_fill(mod_mask_expanded <= 0.5, float("-inf"))

            mod_attn_weights = torch.clamp(mod_attn_weights, min=-50.0, max=50.0)
            mod_attn_weights = F.softmax(mod_attn_weights, dim=-1, dtype=torch.float32).to(query_for_mod.dtype)
            mod_output = torch.matmul(mod_attn_weights, mod_values)

            # Gate and accumulate
            gated_mod_output = gate * mod_output
            combined_modality_output = combined_modality_output + gated_mod_output

            # Store diagnostics for this modality
            with torch.no_grad():
                mod_rms = torch.sqrt(torch.mean(gated_mod_output.float() ** 2)).item()
                text_rms = torch.sqrt(torch.mean(text_output.float() ** 2)).item()
                delta_q_rms = torch.sqrt(torch.mean(delta_q.float() ** 2)).item()
                q_rms = torch.sqrt(torch.mean(query_states.float() ** 2)).item()

                self._last_diagnostics[modality] = {
                    "text_rms": text_rms,
                    "modality_rms": mod_rms,
                    "rms_ratio": mod_rms / (text_rms + 1e-8),
                    "delta_q_rms": delta_q_rms,
                    "q_rms": q_rms,
                    "delta_q_ratio": delta_q_rms / (q_rms + 1e-8),
                    "gate": gate,
                }

        # ============================================================
        # 3. COMBINE
        # ============================================================
        combined_output = text_output + combined_modality_output

        # Reshape and output projection
        combined_output = combined_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = orig_attn.o_proj(combined_output)

        # Return in expected format
        if self._return_format == 1:
            return attn_output
        if self._return_format == 2:
            return (attn_output, None)
        return (attn_output, None, None)

    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _apply_rotary_pos_emb_single(self, x, cos, sin):
        return (x * cos) + (self._rotate_half(x) * sin)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _repeat_kv(hidden_states, n_rep):
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class MultiModalKVAugmentationHookManager:
    """
    Manages attention module replacement for multi-modal KV augmentation.

    Handles adapters keyed by "modality:layer_idx" (e.g., "audio:5", "pointcloud:5").
    Each layer gets a MultiModalKVAugmentedAttention that can handle N modalities.
    """

    def __init__(
        self,
        model: nn.Module,
        kv_adapters: nn.ModuleDict,  # Keys like "audio:5", "pointcloud:5"
    ):
        self.model = model
        self.kv_adapters = kv_adapters

        # Parse adapters to get layer indices and modalities
        self.layer_modalities: Dict[int, List[str]] = {}  # layer_idx -> list of modalities
        for key in kv_adapters.keys():
            modality, layer_str = key.split(":")
            layer_idx = int(layer_str)
            if layer_idx not in self.layer_modalities:
                self.layer_modalities[layer_idx] = []
            self.layer_modalities[layer_idx].append(modality)

        self.fusion_layers = sorted(self.layer_modalities.keys())
        self.layer_modules = self._discover_layer_modules(model)
        self.original_attentions: Dict[int, nn.Module] = {}
        self.wrapped_attentions: Dict[int, MultiModalKVAugmentedAttention] = {}
        self._is_wrapped = False
        self._alerts_enabled: bool = True
        self._alert_log_every: int = 100
        self._rms_ratio_low: float = 0.01
        self._rms_ratio_high: float = 0.40
        self._entropy_low: float = 0.15
        self._entropy_high: float = 0.98

        # Log configuration
        print(f"[MultiModalKVAug] Initialized with {len(self.fusion_layers)} layers", flush=True)
        for layer_idx in self.fusion_layers:
            mods = self.layer_modalities[layer_idx]
            print(f"[MultiModalKVAug]   Layer {layer_idx}: {mods}", flush=True)

    def configure_alerts(
        self,
        *,
        enabled: bool = True,
        log_every: int = 100,
        rms_ratio_low: float = 0.01,
        rms_ratio_high: float = 0.40,
        entropy_low: float = 0.15,
        entropy_high: float = 0.98,
    ) -> None:
        """Maintain API parity with the single-modality KV hook manager."""
        self._alerts_enabled = bool(enabled)
        self._alert_log_every = max(1, int(log_every))
        self._rms_ratio_low = float(rms_ratio_low)
        self._rms_ratio_high = float(rms_ratio_high)
        self._entropy_low = float(entropy_low)
        self._entropy_high = float(entropy_high)

    def _discover_layer_modules(self, model: nn.Module) -> Dict[int, nn.Module]:
        """Locate decoder layer ModuleList."""
        def _is_module_list(obj):
            return isinstance(obj, (list, nn.ModuleList, tuple))

        def _try_extract(candidate):
            if candidate is None:
                return None
            if hasattr(candidate, "layers") and _is_module_list(getattr(candidate, "layers")):
                return {i: layer for i, layer in enumerate(getattr(candidate, "layers"))}
            if hasattr(candidate, "h") and _is_module_list(getattr(candidate, "h")):
                return {i: layer for i, layer in enumerate(getattr(candidate, "h"))}
            return None

        seeds = [
            model,
            getattr(model, "model", None),
            getattr(model, "language_model", None),
            getattr(model, "decoder", None),
            getattr(model, "transformer", None),
        ]

        seen = set()
        queue = [s for s in seeds if s is not None]
        expand_attrs = ("model", "language_model", "decoder", "transformer")
        max_visits = 50
        visits = 0

        while queue and visits < max_visits:
            candidate = queue.pop(0)
            visits += 1
            key = id(candidate)
            if key in seen:
                continue
            seen.add(key)

            extracted = _try_extract(candidate)
            if extracted is not None:
                return extracted

            for attr in expand_attrs:
                child = getattr(candidate, attr, None)
                if child is not None and id(child) not in seen:
                    queue.append(child)

        raise ValueError(f"Unable to locate decoder layers. Tried {visits} candidates.")

    def wrap_attention_modules(self):
        """Replace attention modules with MultiModalKVAugmentedAttention."""
        if self._is_wrapped:
            return

        for layer_idx in self.fusion_layers:
            if layer_idx not in self.layer_modules:
                print(f"[MultiModalKVAug] Warning: layer {layer_idx} not found", flush=True)
                continue

            layer = self.layer_modules[layer_idx]

            # Find attention module
            if hasattr(layer, 'self_attn'):
                original_attn = layer.self_attn
                attn_attr = 'self_attn'
            elif hasattr(layer, 'attention'):
                original_attn = layer.attention
                attn_attr = 'attention'
            else:
                print(f"[MultiModalKVAug] Warning: no attention in layer {layer_idx}", flush=True)
                continue

            # Store original
            self.original_attentions[layer_idx] = original_attn

            # Create multi-modal wrapped attention
            wrapped = MultiModalKVAugmentedAttention(original_attn, layer_idx)

            # Register all modality adapters for this layer
            for modality in self.layer_modalities[layer_idx]:
                adapter_key = f"{modality}:{layer_idx}"
                adapter = self.kv_adapters[adapter_key]
                wrapped.register_modality_adapter(modality, adapter)

            self.wrapped_attentions[layer_idx] = wrapped

            # Replace in layer
            setattr(layer, attn_attr, wrapped)

        self._is_wrapped = True
        print(f"[MultiModalKVAug] Wrapped {len(self.wrapped_attentions)} attention modules", flush=True)

    def unwrap_attention_modules(self):
        """Restore original attention modules."""
        if not self._is_wrapped:
            return

        for layer_idx, original in self.original_attentions.items():
            layer = self.layer_modules[layer_idx]
            if hasattr(layer, 'self_attn'):
                layer.self_attn = original
            else:
                layer.attention = original

        self.original_attentions.clear()
        self.wrapped_attentions.clear()
        self._is_wrapped = False

    def inject_modality_tokens(
        self,
        modality_tokens: Dict[str, torch.Tensor],  # modality -> tokens
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
        modality_gates: Optional[Dict[str, float]] = None,
    ):
        """
        Set modality tokens for current forward pass.

        Args:
            modality_tokens: Dict mapping modality name to tokens (batch, n_tokens, hidden)
            modality_masks: Optional dict of attention masks per modality
            modality_gates: Optional dict of gate values per modality
        """
        modality_masks = modality_masks or {}
        modality_gates = modality_gates or {}

        for wrapped in self.wrapped_attentions.values():
            for modality, tokens in modality_tokens.items():
                if tokens is not None:
                    wrapped.set_modality_tokens(
                        modality=modality,
                        tokens=tokens,
                        mask=modality_masks.get(modality),
                        gate=modality_gates.get(modality, 1.0),
                    )

    def clear_modality_tokens(self):
        """Clear all modality tokens after forward pass."""
        for wrapped in self.wrapped_attentions.values():
            wrapped.clear_modality_tokens()

    def get_diagnostics(self) -> Dict[int, Dict[str, Dict[str, float]]]:
        """Get diagnostics per layer per modality."""
        diagnostics = {}
        for layer_idx, wrapped in self.wrapped_attentions.items():
            d = wrapped.get_diagnostics()
            if d:
                diagnostics[layer_idx] = d
        return diagnostics
