"""
Tests for the SAFE fusion adapter modules.
"""
import pytest
import torch
import torch.nn as nn

from safe.models.fusion_adapter import (
    CrossAttentionBlock,
    LoRAFusionAdapter,
    MultiLayerFusionAdapter,
    GatedFusionAdapter,
)


class TestCrossAttentionBlock:
    """Tests for CrossAttentionBlock class."""

    @pytest.fixture
    def cross_attention(self):
        """Create a CrossAttentionBlock instance."""
        return CrossAttentionBlock(
            hidden_size=64,
            num_attention_heads=8,
            attention_dropout=0.1,
            output_dropout=0.1,
        )

    def test_initialization(self, cross_attention):
        """Test CrossAttentionBlock initializes correctly."""
        assert cross_attention.hidden_size == 64
        assert cross_attention.num_attention_heads == 8
        assert cross_attention.attention_head_size == 8  # 64 // 8
        assert cross_attention.all_head_size == 64

    def test_initialization_prevents_division_by_zero(self):
        """Test initialization with zero heads is handled."""
        block = CrossAttentionBlock(hidden_size=64, num_attention_heads=0)
        assert block.num_attention_heads >= 1

    def test_transpose_for_scores(self, cross_attention):
        """Test transpose_for_scores reshapes correctly."""
        x = torch.randn(2, 10, 64)
        transposed = cross_attention.transpose_for_scores(x)

        assert transposed.shape == (2, 8, 10, 8)  # (batch, heads, seq, head_size)

    def test_forward_basic(self, cross_attention):
        """Test basic forward pass."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)

        output = cross_attention(hidden_states, audio_tokens)

        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()

    def test_forward_with_attention_mask(self, cross_attention):
        """Test forward with attention mask."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)
        attention_mask = torch.ones(2, 4)
        attention_mask[0, 2:] = 0  # Mask some audio tokens

        output = cross_attention(hidden_states, audio_tokens, attention_mask)

        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()

    def test_forward_with_boolean_mask(self, cross_attention):
        """Test forward with boolean attention mask."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)
        attention_mask = torch.tensor([[True, True, False, False],
                                       [True, True, True, False]])

        output = cross_attention(hidden_states, audio_tokens, attention_mask)

        assert torch.isfinite(output).all()

    def test_forward_handles_nan_inputs(self, cross_attention):
        """Test forward handles NaN inputs gracefully."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)

        # Inject NaN values
        hidden_states[0, 0, 0] = float("nan")
        audio_tokens[0, 0, 0] = float("nan")

        output = cross_attention(hidden_states, audio_tokens)

        assert torch.isfinite(output).all()

    def test_forward_handles_extreme_values(self, cross_attention):
        """Test forward handles extreme values."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)

        # Inject extreme values
        hidden_states[0, 0, 0] = 1e6
        audio_tokens[0, 0, 0] = -1e6

        output = cross_attention(hidden_states, audio_tokens)

        assert torch.isfinite(output).all()

    def test_forward_gradient_flow(self, cross_attention):
        """Test gradients flow through the block."""
        hidden_states = torch.randn(2, 12, 64, requires_grad=True)
        audio_tokens = torch.randn(2, 4, 64, requires_grad=True)

        output = cross_attention(hidden_states, audio_tokens)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert audio_tokens.grad is not None
        assert torch.isfinite(hidden_states.grad).all()
        assert torch.isfinite(audio_tokens.grad).all()

    def test_residual_scaling(self, cross_attention):
        """Test residual scaling parameter."""
        assert hasattr(cross_attention, "residual_scale")
        assert cross_attention.residual_scale.requires_grad is True

        # Scale should be clamped during forward
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)

        cross_attention.residual_scale.data.fill_(1.0)  # Set above max
        output = cross_attention(hidden_states, audio_tokens)

        assert torch.isfinite(output).all()

    def test_debug_logging_disabled_by_default(self, cross_attention):
        """Test debug logging is disabled by default."""
        assert cross_attention.debug_logging is False

    def test_different_hidden_sizes(self):
        """Test with different hidden sizes."""
        for hidden_size in [32, 128, 256]:
            block = CrossAttentionBlock(hidden_size=hidden_size, num_attention_heads=4)
            hidden_states = torch.randn(2, 10, hidden_size)
            audio_tokens = torch.randn(2, 4, hidden_size)

            output = block(hidden_states, audio_tokens)
            assert output.shape == hidden_states.shape


class TestLoRAFusionAdapter:
    """Tests for LoRAFusionAdapter class."""

    @pytest.fixture
    def lora_adapter(self):
        """Create a LoRAFusionAdapter instance."""
        return LoRAFusionAdapter(
            hidden_size=64,
            num_attention_heads=8,
            lora_rank=4,
            lora_alpha=8.0,
        )

    def test_initialization(self, lora_adapter):
        """Test LoRAFusionAdapter initializes correctly."""
        assert lora_adapter.hidden_size == 64
        assert lora_adapter.lora_rank == 4
        assert lora_adapter.lora_alpha == 8.0

    def test_forward_basic(self, lora_adapter):
        """Test basic forward pass."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)

        output = lora_adapter(hidden_states, audio_tokens)

        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()

    def test_forward_with_gating(self, lora_adapter):
        """Test forward with different gate values."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)

        output_full = lora_adapter(hidden_states, audio_tokens, gate=1.0)
        output_half = lora_adapter(hidden_states, audio_tokens, gate=0.5)
        output_zero = lora_adapter(hidden_states, audio_tokens, gate=0.0)

        # Gate=0 should return input unchanged
        assert torch.allclose(output_zero, hidden_states.to(output_zero.dtype), atol=1e-5)

        # Different gates should give different outputs
        assert not torch.allclose(output_full, output_half)

    def test_forward_with_tensor_gate(self, lora_adapter):
        """Test forward with tensor gate (batch-wise gating)."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)
        gate = torch.tensor([1.0, 0.0])  # Full gate for first, zero for second

        output = lora_adapter(hidden_states, audio_tokens, gate=gate)

        assert output.shape == hidden_states.shape

    def test_forward_with_attention_mask(self, lora_adapter):
        """Test forward with attention mask."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)
        attention_mask = torch.ones(2, 4)

        output = lora_adapter(hidden_states, audio_tokens, attention_mask)

        assert torch.isfinite(output).all()

    def test_set_debug_logging(self, lora_adapter):
        """Test debug logging configuration."""
        lora_adapter.set_debug_logging(True, log_limit=10)
        assert lora_adapter.debug_logging is True

        lora_adapter.set_debug_logging(False)
        assert lora_adapter.debug_logging is False

    def test_gradient_flow(self, lora_adapter):
        """Test gradients flow through LoRA adapter."""
        hidden_states = torch.randn(2, 12, 64, requires_grad=True)
        audio_tokens = torch.randn(2, 4, 64, requires_grad=True)

        output = lora_adapter(hidden_states, audio_tokens)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert audio_tokens.grad is not None


class TestMultiLayerFusionAdapter:
    """Tests for MultiLayerFusionAdapter class."""

    @pytest.fixture
    def multi_layer_adapter(self):
        """Create a MultiLayerFusionAdapter instance."""
        return MultiLayerFusionAdapter(
            hidden_size=64,
            num_layers=24,
            fusion_layer_indices=[8, 16],
            num_attention_heads=8,
            lora_rank=4,
        )

    def test_initialization(self, multi_layer_adapter):
        """Test MultiLayerFusionAdapter initializes correctly."""
        assert multi_layer_adapter.hidden_size == 64
        assert multi_layer_adapter.num_layers == 24
        assert 8 in multi_layer_adapter.fusion_layer_indices
        assert 16 in multi_layer_adapter.fusion_layer_indices

    def test_initialization_with_default_layers(self):
        """Test initialization with default fusion layers."""
        adapter = MultiLayerFusionAdapter(hidden_size=64, num_layers=24)

        # Should have default layers at 1/3 and 2/3 of depth
        assert len(adapter.fusion_layer_indices) > 0

    def test_initialization_with_modality_mapping(self):
        """Test initialization with modality-specific layer mapping."""
        modalities = {
            "audio": {"layer_indices": [8, 12]},
            "video": {"layer_indices": [12, 16]},
        }
        adapter = MultiLayerFusionAdapter(
            hidden_size=64,
            num_layers=24,
            modalities=modalities,
        )

        assert "audio" in adapter.fusion_layers
        assert "video" in adapter.fusion_layers

    def test_forward_at_fusion_layer(self, multi_layer_adapter):
        """Test forward at a designated fusion layer."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)

        output = multi_layer_adapter(
            hidden_states, audio_tokens, layer_idx=8
        )

        # Should apply fusion
        assert output.shape == hidden_states.shape
        assert not torch.allclose(output, hidden_states.to(output.dtype))

    def test_forward_at_non_fusion_layer(self, multi_layer_adapter):
        """Test forward at a non-fusion layer returns input unchanged."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)

        output = multi_layer_adapter(
            hidden_states, audio_tokens, layer_idx=5  # Not a fusion layer
        )

        assert torch.allclose(output, hidden_states)

    def test_forward_with_active_fusion_layer(self, multi_layer_adapter):
        """Test forward with specific active fusion layer."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)

        # Only apply fusion at layer 8, skip layer 16
        output = multi_layer_adapter(
            hidden_states, audio_tokens, layer_idx=8, active_fusion_layer=8
        )
        assert not torch.allclose(output, hidden_states.to(output.dtype))

        output = multi_layer_adapter(
            hidden_states, audio_tokens, layer_idx=16, active_fusion_layer=8
        )
        assert torch.allclose(output, hidden_states)

    def test_apply_fusion_at_layer(self, multi_layer_adapter):
        """Test apply_fusion_at_layer method."""
        hidden_states = torch.randn(2, 12, 64)
        modality_tokens = {"audio": torch.randn(2, 4, 64)}

        output = multi_layer_adapter.apply_fusion_at_layer(
            layer_idx=8,
            hidden_states=hidden_states,
            modality_tokens=modality_tokens,
        )

        assert output.shape == hidden_states.shape

    def test_set_debug_logging(self, multi_layer_adapter):
        """Test debug logging propagates to all adapters."""
        multi_layer_adapter.set_debug_logging(True, log_limit=5)

        for adapter in multi_layer_adapter.fusion_adapters.values():
            assert adapter.debug_logging is True

    def test_layer_modalities_inversion(self, multi_layer_adapter):
        """Test layer to modalities mapping is correct."""
        # Layer 8 should have audio modality
        assert 8 in multi_layer_adapter.layer_modalities
        assert "audio" in multi_layer_adapter.layer_modalities[8]


class TestGatedFusionAdapter:
    """Tests for GatedFusionAdapter class."""

    @pytest.fixture
    def gated_adapter(self):
        """Create a GatedFusionAdapter instance."""
        return GatedFusionAdapter(
            hidden_size=64,
            num_attention_heads=8,
            lora_rank=4,
            gate_hidden_size=32,
        )

    def test_initialization(self, gated_adapter):
        """Test GatedFusionAdapter initializes correctly."""
        assert hasattr(gated_adapter, "fusion_adapter")
        assert hasattr(gated_adapter, "gate_network")

    def test_forward_returns_output_and_gate(self, gated_adapter):
        """Test forward returns both fused output and gate values."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)

        output, gate = gated_adapter(hidden_states, audio_tokens)

        assert output.shape == hidden_states.shape
        assert gate.shape == (2,)  # One gate value per batch item
        assert (gate >= 0).all() and (gate <= 1).all()  # Sigmoid output

    def test_forward_with_forced_gate(self, gated_adapter):
        """Test forward with forced gate value."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)

        output_forced, gate = gated_adapter(
            hidden_states, audio_tokens, force_gate=0.5
        )

        assert gate.item() == 0.5

    def test_forward_with_attention_mask(self, gated_adapter):
        """Test forward with attention mask."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)
        attention_mask = torch.ones(2, 4)

        output, gate = gated_adapter(hidden_states, audio_tokens, attention_mask)

        assert torch.isfinite(output).all()

    def test_gate_initialized_toward_off(self, gated_adapter):
        """Test gate is initialized to favor OFF state."""
        hidden_states = torch.randn(2, 12, 64)
        audio_tokens = torch.randn(2, 4, 64)

        # With random inputs and initialized bias, gate should be low
        _, gate = gated_adapter(hidden_states, audio_tokens)

        # Gate should be biased toward 0 (OFF) initially
        assert gate.mean() < 0.5

    def test_gradient_flow(self, gated_adapter):
        """Test gradients flow through gated adapter."""
        hidden_states = torch.randn(2, 12, 64, requires_grad=True)
        audio_tokens = torch.randn(2, 4, 64, requires_grad=True)

        output, gate = gated_adapter(hidden_states, audio_tokens)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert audio_tokens.grad is not None

    def test_set_debug_logging(self, gated_adapter):
        """Test debug logging configuration."""
        gated_adapter.set_debug_logging(True)
        assert gated_adapter.fusion_adapter.debug_logging is True

    def test_configure_attention_probe(self, gated_adapter):
        """Test configure_attention_probe method."""
        gated_adapter.configure_attention_probe(True, log_limit=3)
        # Should propagate to fusion adapter
        assert gated_adapter.fusion_adapter.debug_logging is True
