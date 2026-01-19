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


class KVAugmentationAdapter(nn.Module):
    """
    Trainable projections for audio tokens → K,V space.

    Projects audio tokens to keys and values compatible with LLM's self-attention.
    Uses bottleneck architecture to reduce parameters while maintaining expressivity.
    """

    def __init__(
        self,
        hidden_size: int = 5120,
        num_heads: int = 40,
        head_dim: int = 128,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        use_bottleneck: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_head_size = num_heads * head_dim
        self.use_bottleneck = use_bottleneck

        if use_bottleneck:
            # Bottleneck projection: hidden_size → bottleneck → total_head_size
            self.audio_k_proj = nn.Sequential(
                nn.Linear(hidden_size, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, self.total_head_size),
            )
            self.audio_v_proj = nn.Sequential(
                nn.Linear(hidden_size, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, self.total_head_size),
            )
        else:
            # Direct projection (more parameters)
            self.audio_k_proj = nn.Linear(hidden_size, self.total_head_size)
            self.audio_v_proj = nn.Linear(hidden_size, self.total_head_size)

        # Learnable scaling factor for audio contribution
        # Start at 0.5 for meaningful gradients (same as residual_scale in CrossAttentionBlock)
        self.audio_scale = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("scale_min", torch.tensor(0.1))
        self.register_buffer("scale_max", torch.tensor(2.0))

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
        gate: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project audio tokens to K,V space.

        Args:
            audio_tokens: (batch_size, num_audio_tokens, hidden_size)
            gate: Scalar gate to modulate audio contribution

        Returns:
            audio_keys: (batch_size, num_audio_tokens, total_head_size)
            audio_values: (batch_size, num_audio_tokens, total_head_size)
        """
        # Cast audio tokens to match projection weights dtype
        input_dtype = audio_tokens.dtype
        audio_tokens = audio_tokens.to(self.audio_k_proj[0].weight.dtype if self.use_bottleneck else self.audio_k_proj.weight.dtype)

        # Clamp scale to reasonable range
        scale = torch.clamp(self.audio_scale, self.scale_min, self.scale_max)
        effective_scale = scale * gate

        # Project to K,V space
        audio_keys = self.audio_k_proj(audio_tokens) * effective_scale
        audio_values = self.audio_v_proj(audio_tokens) * effective_scale

        # Cast back to input dtype for compatibility with attention
        audio_keys = audio_keys.to(input_dtype)
        audio_values = audio_values.to(input_dtype)

        return audio_keys, audio_values

    def reshape_for_attention(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reshape projected tensor for multi-head attention.

        Args:
            tensor: (batch_size, seq_len, total_head_size)

        Returns:
            reshaped: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
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
        self.num_heads = getattr(original_attention, 'num_heads', 40)
        self.head_dim = getattr(original_attention, 'head_dim', 128)
        self.num_key_value_heads = getattr(original_attention, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Storage for current forward pass audio data
        self._audio_tokens: Optional[torch.Tensor] = None
        self._audio_mask: Optional[torch.Tensor] = None
        self._gate: float = 1.0

        # For returning attention weights (used in regularization)
        self._last_attention_weights: Optional[torch.Tensor] = None
        self._return_attention_weights: bool = False

        # Detect return format of original attention (for compatibility across HF versions)
        # Probe the original attention's forward signature to determine expected return format
        self._return_format = self._detect_return_format()

    def _detect_return_format(self) -> int:
        """
        Detect the return format of the original attention module.

        Different transformers versions return different tuple lengths:
        - transformers 4.36-4.40: (attn_output, attn_weights, past_key_value) = 3
        - transformers 4.45+: sometimes (attn_output, past_key_value) = 2
        - transformers 4.50+: just attn_output (tensor) = 1

        We check the class name and module to determine the expected format.
        """
        orig_class = type(self.original_attention).__name__

        # SDPA and Flash attention variants often return fewer values
        if 'Sdpa' in orig_class or 'Flash' in orig_class:
            # These typically return just attn_output or (attn_output, None, None)
            # but the decoder layer still unpacks 3 values
            return 3

        # Check if this is an eager attention implementation
        if 'Attention' in orig_class:
            # Standard LlamaAttention returns 3 values
            return 3

        # Default to 3 for safety (most common in LLaVA models)
        return 3

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

    def clear_audio(self):
        """Clear audio tokens after forward pass."""
        self._audio_tokens = None
        self._audio_mask = None
        self._gate = 1.0
        self._last_attention_weights = None

    def set_return_attention_weights(self, return_weights: bool):
        """Enable/disable attention weight capture for regularization."""
        self._return_attention_weights = return_weights

    def get_last_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights from last forward pass."""
        return self._last_attention_weights

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
        Compute attention with audio K,V augmentation.

        Key differences from standard LlamaAttention:
        1. Audio K,V are concatenated to text K,V
        2. Audio K,V do NOT get RoPE (position-agnostic)
        3. Attention mask is extended to include audio positions
        """
        bsz, q_len, _ = hidden_states.size()
        n_audio = self._audio_tokens.size(1)

        # Get original attention's projections
        orig_attn = self.original_attention

        # Project hidden states to Q, K, V
        query_states = orig_attn.q_proj(hidden_states)
        key_states = orig_attn.k_proj(hidden_states)
        value_states = orig_attn.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings to text Q, K (standard LLaMA)
        if hasattr(orig_attn, 'rotary_emb'):
            cos, sin = orig_attn.rotary_emb(value_states, position_ids)
            query_states, key_states = self._apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Project audio tokens to K, V (NO RoPE - position agnostic)
        audio_keys, audio_values = self.kv_adapter(self._audio_tokens, self._gate)

        # Reshape audio K, V for attention
        audio_keys = audio_keys.view(bsz, n_audio, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        audio_values = audio_values.view(bsz, n_audio, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Concatenate audio K,V to text K,V
        # Audio tokens become additional "memory" that all queries can attend to
        key_states = torch.cat([key_states, audio_keys], dim=2)  # (bsz, heads, q_len + n_audio, head_dim)
        value_states = torch.cat([value_states, audio_values], dim=2)

        # Handle grouped-query attention (GQA) if needed
        if self.num_key_value_groups > 1:
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Extend attention mask to include audio positions
        extended_mask = self._extend_attention_mask(
            attention_mask, n_audio, bsz, q_len,
            hidden_states.device, hidden_states.dtype
        )

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if extended_mask is not None:
            attn_weights = attn_weights + extended_mask

        # Numerical stability
        attn_weights = torch.clamp(attn_weights, min=-50.0, max=50.0)

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Store attention weights for regularization if needed
        if self._return_attention_weights:
            self._last_attention_weights = attn_weights.detach()

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # Output projection
        attn_output = orig_attn.o_proj(attn_output)

        # Prepare outputs - MUST match LlamaAttention return signature
        # Different transformers versions have different return formats.
        # We match whatever format we detected from the original attention.
        if output_attentions:
            attn_weights_out = attn_weights
        else:
            attn_weights_out = None

        # Note: KV cache handling with audio augmentation is complex
        # For now, we return None for past_key_value
        past_key_value_out = None

        # LlamaDecoderLayer unpacks: hidden_states, _ = self.self_attn(...)
        # So we must return exactly 2 values
        return (attn_output, attn_weights_out)

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
        """Set audio tokens for current forward pass in all wrapped attentions."""
        for wrapped in self.wrapped_attentions.values():
            wrapped.set_audio(audio_tokens, audio_mask, gate)

    def clear_audio(self):
        """Clear audio tokens after forward pass."""
        for wrapped in self.wrapped_attentions.values():
            wrapped.clear_audio()

    def set_return_attention_weights(self, return_weights: bool):
        """Enable/disable attention weight capture."""
        for wrapped in self.wrapped_attentions.values():
            wrapped.set_return_attention_weights(return_weights)

    def get_attention_weights(self) -> Dict[int, torch.Tensor]:
        """Get attention weights from all layers for regularization."""
        weights = {}
        for idx, wrapped in self.wrapped_attentions.items():
            w = wrapped.get_last_attention_weights()
            if w is not None:
                weights[idx] = w
        return weights


class MinAudioAttentionLoss(nn.Module):
    """
    Regularization loss that ensures answer tokens attend to audio.

    This prevents the model from learning to ignore audio entirely.
    Uses hinge loss: penalizes if attention to audio is below threshold.
    """

    def __init__(
        self,
        min_attention: float = 0.01,
        loss_weight: float = 0.1,
    ):
        super().__init__()
        self.min_attention = min_attention
        self.loss_weight = loss_weight

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

        Returns:
            Scalar loss tensor
        """
        if not attention_weights:
            return torch.tensor(0.0)

        total_loss = 0.0
        count = 0

        for layer_idx, attn in attention_weights.items():
            # Extract attention to audio positions (last n_audio columns)
            audio_attn = attn[:, :, :, -n_audio:]  # (bsz, heads, q_len, n_audio)

            # Average attention to audio per query position
            avg_audio_attn = audio_attn.mean(dim=-1)  # (bsz, heads, q_len)

            # Only penalize on supervised (answer) tokens if mask provided
            if supervised_mask is not None:
                # supervised_mask: (bsz, q_len)
                mask = supervised_mask.unsqueeze(1).float()  # (bsz, 1, q_len)
                masked_attn = avg_audio_attn * mask
                denom = mask.sum() * attn.size(1)  # num supervised positions * heads
            else:
                masked_attn = avg_audio_attn
                denom = avg_audio_attn.numel()

            # Hinge loss: penalize if below minimum
            deficit = F.relu(self.min_attention - masked_attn)
            layer_loss = deficit.sum() / max(denom, 1)

            total_loss = total_loss + layer_loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=next(iter(attention_weights.values())).device)

        return self.loss_weight * (total_loss / count)
