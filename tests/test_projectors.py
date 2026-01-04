"""
Tests for the SAFE audio and vision projector modules.
"""
import pytest
import torch
import torch.nn as nn

from safe.models.projectors import (
    AudioProjector,
    AdaptiveAudioProjector,
    VisionProjector,
)


class TestAudioProjector:
    """Tests for AudioProjector class."""

    @pytest.fixture
    def projector(self):
        """Create an AudioProjector instance."""
        return AudioProjector(
            audio_embed_dim=512,
            llm_hidden_size=256,
            num_audio_tokens=8,
            dropout=0.1,
        )

    def test_initialization(self, projector):
        """Test AudioProjector initializes correctly."""
        assert projector.audio_embed_dim == 512
        assert projector.llm_hidden_size == 256
        assert projector.num_audio_tokens == 8
        assert projector.bottleneck_dim > 0

    def test_initialization_custom_bottleneck(self):
        """Test AudioProjector with custom bottleneck dimension."""
        projector = AudioProjector(
            audio_embed_dim=512,
            llm_hidden_size=256,
            bottleneck_dim=128,
        )
        assert projector.bottleneck_dim == 128

    def test_initialization_activations(self):
        """Test AudioProjector with different activation functions."""
        for activation in ["gelu", "relu", "silu"]:
            projector = AudioProjector(
                audio_embed_dim=512,
                llm_hidden_size=256,
                activation=activation,
            )
            assert projector.activation is not None

    def test_initialization_invalid_activation(self):
        """Test AudioProjector raises for invalid activation."""
        with pytest.raises(ValueError, match="Unsupported activation"):
            AudioProjector(
                audio_embed_dim=512,
                llm_hidden_size=256,
                activation="invalid",
            )

    def test_forward_output_shape(self, projector):
        """Test forward pass produces correct output shape."""
        audio_features = torch.randn(4, 512)  # batch_size=4, audio_embed_dim=512

        output = projector(audio_features)

        # Expected: (batch_size, num_audio_tokens, llm_hidden_size)
        assert output.shape == (4, 8, 256)

    def test_forward_single_sample(self, projector):
        """Test forward with single sample."""
        audio_features = torch.randn(1, 512)

        output = projector(audio_features)

        assert output.shape == (1, 8, 256)

    def test_forward_handles_nan(self, projector):
        """Test forward handles NaN inputs."""
        audio_features = torch.randn(2, 512)
        audio_features[0, 0] = float("nan")

        output = projector(audio_features)

        assert torch.isfinite(output).all()

    def test_forward_handles_extreme_values(self, projector):
        """Test forward handles extreme input values."""
        audio_features = torch.randn(2, 512)
        audio_features[0, 0] = 1e10
        audio_features[1, 0] = -1e10

        output = projector(audio_features)

        assert torch.isfinite(output).all()

    def test_forward_with_output_dtype(self, projector):
        """Test forward with specified output dtype."""
        audio_features = torch.randn(2, 512)

        output = projector(audio_features, out_dtype=torch.float16)

        assert output.dtype == torch.float16

    def test_forward_gradient_flow(self, projector):
        """Test gradients flow through the projector."""
        audio_features = torch.randn(2, 512, requires_grad=True)

        output = projector(audio_features)
        loss = output.sum()
        loss.backward()

        assert audio_features.grad is not None
        assert torch.isfinite(audio_features.grad).all()

    def test_weight_initialization(self, projector):
        """Test weights are properly initialized."""
        # Last layer should have small weights (tiny init)
        last_layer = None
        for module in projector.projector.modules():
            if isinstance(module, nn.Linear):
                last_layer = module

        if last_layer is not None:
            # Weights should be small
            assert last_layer.weight.abs().mean() < 0.01

    def test_set_debug_logging(self, projector):
        """Test debug logging configuration."""
        projector.set_debug_logging(True, log_limit=3)
        assert projector.debug_logging is True
        assert projector._projector_log_limit == 3

    def test_output_is_normalized(self, projector):
        """Test output tokens are normalized."""
        audio_features = torch.randn(2, 512)

        output = projector(audio_features)

        # Output should pass through LayerNorm, so values should be reasonable
        assert output.abs().mean() < 10.0


class TestAdaptiveAudioProjector:
    """Tests for AdaptiveAudioProjector class."""

    @pytest.fixture
    def adaptive_projector(self):
        """Create an AdaptiveAudioProjector instance."""
        return AdaptiveAudioProjector(
            audio_embed_dim=512,
            llm_hidden_size=256,
            max_audio_tokens=12,
            min_audio_tokens=4,
            dropout=0.1,
        )

    def test_initialization(self, adaptive_projector):
        """Test AdaptiveAudioProjector initializes correctly."""
        assert adaptive_projector.audio_embed_dim == 512
        assert adaptive_projector.llm_hidden_size == 256
        assert adaptive_projector.max_audio_tokens == 12
        assert adaptive_projector.min_audio_tokens == 4

    def test_token_generators_created(self, adaptive_projector):
        """Test token generators are created for each token count."""
        # Should have generators for 4, 5, 6, 7, 8, 9, 10, 11, 12 tokens
        for k in range(4, 13):
            assert str(k) in adaptive_projector.token_generators

    def test_forward_fixed_tokens(self, adaptive_projector):
        """Test forward with fixed token count."""
        audio_features = torch.randn(2, 512)

        output = adaptive_projector(audio_features, num_tokens=8)

        assert output.shape == (2, 8, 256)

    def test_forward_predicted_tokens(self, adaptive_projector):
        """Test forward with predicted token count."""
        audio_features = torch.randn(2, 512)

        output = adaptive_projector(audio_features, num_tokens=None)

        # Output should be within min/max range
        assert output.shape[1] >= 4
        assert output.shape[1] <= 12
        assert output.shape == (2, output.shape[1], 256)

    def test_forward_min_tokens(self, adaptive_projector):
        """Test forward with minimum token count."""
        audio_features = torch.randn(2, 512)

        output = adaptive_projector(audio_features, num_tokens=4)

        assert output.shape == (2, 4, 256)

    def test_forward_max_tokens(self, adaptive_projector):
        """Test forward with maximum token count."""
        audio_features = torch.randn(2, 512)

        output = adaptive_projector(audio_features, num_tokens=12)

        assert output.shape == (2, 12, 256)

    def test_forward_handles_nan(self, adaptive_projector):
        """Test forward handles NaN inputs."""
        audio_features = torch.randn(2, 512)
        audio_features[0, 0] = float("nan")

        output = adaptive_projector(audio_features, num_tokens=8)

        assert torch.isfinite(output).all()

    def test_forward_with_output_dtype(self, adaptive_projector):
        """Test forward with specified output dtype."""
        audio_features = torch.randn(2, 512)

        output = adaptive_projector(audio_features, num_tokens=8, out_dtype=torch.float16)

        assert output.dtype == torch.float16

    def test_forward_gradient_flow(self, adaptive_projector):
        """Test gradients flow through the adaptive projector."""
        audio_features = torch.randn(2, 512, requires_grad=True)

        output = adaptive_projector(audio_features, num_tokens=8)
        loss = output.sum()
        loss.backward()

        assert audio_features.grad is not None
        assert torch.isfinite(audio_features.grad).all()

    def test_token_predictor_output(self, adaptive_projector):
        """Test token predictor produces valid logits."""
        audio_features = torch.randn(2, 512)
        normalized = adaptive_projector.input_norm(audio_features)
        features = adaptive_projector.feature_extractor(normalized)
        token_logits = adaptive_projector.token_predictor(features)

        # Should have logits for each possible token count
        num_options = adaptive_projector.max_audio_tokens - adaptive_projector.min_audio_tokens + 1
        assert token_logits.shape == (2, num_options)

    def test_set_debug_logging(self, adaptive_projector):
        """Test debug logging configuration."""
        adaptive_projector.set_debug_logging(True, log_limit=5)
        assert adaptive_projector.debug_logging is True

    def test_different_batch_sizes(self, adaptive_projector):
        """Test with various batch sizes."""
        for batch_size in [1, 4, 16]:
            audio_features = torch.randn(batch_size, 512)
            output = adaptive_projector(audio_features, num_tokens=8)
            assert output.shape[0] == batch_size


class TestVisionProjector:
    """Tests for VisionProjector class."""

    @pytest.fixture
    def vision_projector(self):
        """Create a VisionProjector instance."""
        return VisionProjector(
            vision_embed_dim=768,
            llm_hidden_size=256,
            dropout=0.1,
        )

    def test_initialization(self, vision_projector):
        """Test VisionProjector initializes correctly."""
        assert hasattr(vision_projector, "projector")

    def test_forward_output_shape(self, vision_projector):
        """Test forward produces correct output shape."""
        vision_features = torch.randn(2, 196, 768)  # batch, seq_len, embed_dim

        output = vision_projector(vision_features)

        assert output.shape == (2, 196, 256)

    def test_forward_single_token(self, vision_projector):
        """Test forward with single vision token."""
        vision_features = torch.randn(2, 1, 768)

        output = vision_projector(vision_features)

        assert output.shape == (2, 1, 256)

    def test_forward_gradient_flow(self, vision_projector):
        """Test gradients flow through vision projector."""
        vision_features = torch.randn(2, 196, 768, requires_grad=True)

        output = vision_projector(vision_features)
        loss = output.sum()
        loss.backward()

        assert vision_features.grad is not None
        assert torch.isfinite(vision_features.grad).all()

    def test_weight_initialization(self, vision_projector):
        """Test weights are xavier initialized."""
        for module in vision_projector.projector.modules():
            if isinstance(module, nn.Linear):
                # Xavier init should give reasonable weight magnitudes
                assert module.weight.abs().mean() < 1.0


class TestProjectorInteroperability:
    """Tests for projector interoperability and integration."""

    def test_audio_projector_output_compatible_with_llm(self):
        """Test audio projector output is compatible with LLM input."""
        projector = AudioProjector(
            audio_embed_dim=512,
            llm_hidden_size=4096,  # Typical LLM hidden size
            num_audio_tokens=8,
        )
        audio_features = torch.randn(2, 512)

        output = projector(audio_features)

        # Should be compatible with LLM hidden states
        assert output.shape[-1] == 4096

    def test_different_embedding_dimensions(self):
        """Test projectors work with various embedding dimensions."""
        for audio_dim in [256, 512, 1024]:
            for llm_dim in [768, 2048, 4096]:
                projector = AudioProjector(
                    audio_embed_dim=audio_dim,
                    llm_hidden_size=llm_dim,
                    num_audio_tokens=4,
                )
                audio_features = torch.randn(2, audio_dim)
                output = projector(audio_features)

                assert output.shape == (2, 4, llm_dim)

    def test_projector_trainable_parameters(self):
        """Test projectors have trainable parameters."""
        projector = AudioProjector(
            audio_embed_dim=512,
            llm_hidden_size=256,
        )

        trainable_params = sum(
            p.numel() for p in projector.parameters() if p.requires_grad
        )

        assert trainable_params > 0

    def test_projector_can_be_frozen(self):
        """Test projector parameters can be frozen."""
        projector = AudioProjector(
            audio_embed_dim=512,
            llm_hidden_size=256,
        )

        # Freeze all parameters
        for param in projector.parameters():
            param.requires_grad = False

        trainable_params = sum(
            p.numel() for p in projector.parameters() if p.requires_grad
        )

        assert trainable_params == 0
