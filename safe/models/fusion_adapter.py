import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, LoraModel
from typing import Optional, Tuple
import math


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
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query projection (from LLM hidden states)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        
        # Key and Value projections (from audio tokens)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = nn.Dropout(output_dropout)
        
        # Layer normalization
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
        **kwargs  # Accept and ignore extra kwargs to prevent PEFT errors
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
        # Compute query, key, value
        query_layer = self.transpose_for_scores(self.query(hidden_states))  # (B, H, seq_len, d)
        key_layer = self.transpose_for_scores(self.key(audio_tokens))       # (B, H, audio_len, d)
        value_layer = self.transpose_for_scores(self.value(audio_tokens))   # (B, H, audio_len, d)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Reshape mask for broadcasting: (batch_size, 1, 1, num_audio_tokens)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_scores = attention_scores + (attention_mask * -10000.0)
        
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape and transpose back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        output = self.output_dense(context_layer)
        output = self.output_dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + hidden_states)
        
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
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        gate: float = 1.0
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
        if gate == 0.0:
            # Skip audio fusion entirely
            return hidden_states
            
        weight = self.cross_attention.base_model.query.weight
        target_dtype = weight.dtype
        target_device = weight.device
        if hidden_states.dtype != target_dtype or hidden_states.device != target_device:
            hidden_states = hidden_states.to(device=target_device, dtype=target_dtype)
        if audio_tokens.dtype != target_dtype or audio_tokens.device != target_device:
            audio_tokens = audio_tokens.to(device=target_device, dtype=target_dtype)

        # Apply cross-attention with LoRA
        fused_states = self.cross_attention(
            hidden_states=hidden_states,
            audio_tokens=audio_tokens,
            attention_mask=attention_mask
        )
        
        # Apply gating: interpolate between original and fused states
        if gate != 1.0:
            fused_states = gate * fused_states + (1.0 - gate) * hidden_states
            
        return fused_states


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
        active_fusion_layer: Optional[int] = None
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
            gate=gate
        )
        
        return fused_states


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
        force_gate: Optional[float] = None
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
                sample_fused = self.fusion_adapter(
                    hidden_states=hidden_states[i:i+1],
                    audio_tokens=audio_tokens[i:i+1],
                    attention_mask=attention_mask[i:i+1] if attention_mask is not None else None,
                    gate=gate[i].item()
                )
                fused_states.append(sample_fused)
            fused_states = torch.cat(fused_states, dim=0)
        else:
            # Scalar gating
            fused_states = self.fusion_adapter(
                hidden_states=hidden_states,
                audio_tokens=audio_tokens,
                attention_mask=attention_mask,
                gate=gate.item() if isinstance(gate, torch.Tensor) else gate
            )
        
        return fused_states, gate
