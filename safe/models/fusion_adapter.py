import torch
import math
from typing import Optional, Tuple

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
            attention_mask: Optional mask for audio tokens
            
        Returns:
            output: (batch_size, seq_len, hidden_size) - Fused representations
        """
        # Clean inputs to avoid propagating NaNs/Infs from upstream modules
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)
        audio_tokens = torch.nan_to_num(audio_tokens, nan=0.0, posinf=1e4, neginf=-1e4)

        input_dtype = hidden_states.dtype

        # Compute query, key, value
        query_layer = self.transpose_for_scores(self.query(hidden_states))  # (B, H, seq_len, d)
        key_layer = self.transpose_for_scores(self.key(audio_tokens))       # (B, H, audio_len, d)
        value_layer = self.transpose_for_scores(self.value(audio_tokens))   # (B, H, audio_len, d)

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
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Reshape mask for broadcasting: (batch_size, 1, 1, num_audio_tokens)
            attention_mask = attention_mask.to(attention_scores.dtype).unsqueeze(1).unsqueeze(1)
            # Use safer masking value that won't cause overflow
            attention_scores = attention_scores + (attention_mask * -1e9)

        # Stable softmax computation
        attention_probs = F.softmax(attention_scores, dim=-1)
        # Clean any NaN/Inf that might have appeared and ensure valid probability range
        attention_probs = torch.nan_to_num(attention_probs, nan=0.0, posinf=1.0, neginf=0.0)
        # Renormalize to ensure probabilities sum to 1 after cleaning
        attention_probs = attention_probs / (attention_probs.sum(dim=-1, keepdim=True) + 1e-8)
        attention_probs = self.attention_dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.to(input_dtype)
        
        # Reshape and transpose back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        output = self.output_dense(context_layer)
        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
        output = self.output_dropout(output)
        output = output.to(input_dtype)

        # Apply residual connection BEFORE layer norm and gating
        hidden_states = hidden_states.to(output.dtype)
        output = output + hidden_states

        # Optional layer norm on the full fused output
        if self.layer_norm is not None:
            output = self.layer_norm(output)

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

        return output


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
        lora_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        target_modules: list = None
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.debug_logging = False
        self.last_attention_summary: Optional[dict] = None
        self._attention_log_limit = 5
        self._attention_logs_emitted = 0
        
        # Base cross-attention block
        self.cross_attention = CrossAttentionBlock(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout
        )
        
        # LoRA configuration
        if target_modules is None:
            target_modules = ["query", "value"]  # Apply LoRA to Q and V projections
            
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
        weight = self.cross_attention.base_model.query.weight
        target_dtype = weight.dtype
        target_device = weight.device
        if hidden_states.dtype != target_dtype or hidden_states.device != target_device:
            hidden_states = hidden_states.to(device=target_device, dtype=target_dtype)
        if audio_tokens.dtype != target_dtype or audio_tokens.device != target_device:
            audio_tokens = audio_tokens.to(device=target_device, dtype=target_dtype)

        # Apply cross-attention with LoRA (returns hidden_states + attention_output)
        fused_states = self.cross_attention(
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

        # Apply gating: interpolate between original and fused states
        # gate=0 -> return hidden_states (no audio), gate=1 -> return fused_states (full audio)
        if isinstance(gate, torch.Tensor):
            gate = gate.to(device=hidden_states.device, dtype=hidden_states.dtype)
            while gate.dim() < fused_states.dim():
                gate = gate.unsqueeze(-1)
            output = gate * fused_states + (1.0 - gate) * hidden_states
        else:
            gate = float(gate)
            output = gate * fused_states + (1.0 - gate) * hidden_states

        return output


class MultiLayerFusionAdapter(nn.Module):
    """
    Multi-layer fusion adapter that can insert audio fusion at multiple LLM layers.
    Supports selection of fusion layers during training/inference.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        fusion_layer_indices: list = None,
        num_attention_heads: int = 8,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if fusion_layer_indices is None:
            # Default to mid-layers
            fusion_layer_indices = [num_layers // 3, 2 * num_layers // 3]
            
        self.fusion_layer_indices = fusion_layer_indices
        
        # Create fusion adapters for each target layer
        self.fusion_adapters = nn.ModuleDict()
        for layer_idx in fusion_layer_indices:
            self.fusion_adapters[str(layer_idx)] = LoRAFusionAdapter(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
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
        """
        Apply fusion at specified layer.
        
        Args:
            hidden_states: LLM hidden states at current layer
            audio_tokens: Audio tokens
            layer_idx: Current LLM layer index
            attention_mask: Optional attention mask
            gate: Gating factor
            active_fusion_layer: Specific layer to apply fusion (if None, use all configured layers)
            
        Returns:
            Fused hidden states
        """
        # Check if this layer should apply fusion
        if active_fusion_layer is not None:
            should_fuse = (layer_idx == active_fusion_layer)
        else:
            should_fuse = (layer_idx in self.fusion_layer_indices)
            
        if not should_fuse or str(layer_idx) not in self.fusion_adapters:
            return hidden_states
            
        # Apply fusion
        fusion_adapter = self.fusion_adapters[str(layer_idx)]
        fused_states = fusion_adapter(
            hidden_states=hidden_states,
            audio_tokens=audio_tokens,
            attention_mask=attention_mask,
            gate=gate,
            supervised_mask=supervised_mask,
        )

        return fused_states

    def set_debug_logging(self, enabled: bool, log_limit: int = 5) -> None:
        for adapter in self.fusion_adapters.values():
            adapter.set_debug_logging(enabled, log_limit)

    def configure_attention_probe(self, enabled: bool, log_limit: int = 5) -> None:
        self.set_debug_logging(enabled, log_limit)


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
        gate_init_bias: float = -2.5  # Initialize gate toward OFF
    ):
        super().__init__()
        
        # Core fusion adapter
        self.fusion_adapter = LoRAFusionAdapter(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            lora_rank=lora_rank
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
