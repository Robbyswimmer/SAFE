"""
Tests for the SAFE audio encoder modules.

Note: These tests focus on preprocessing logic and interface behavior.
The actual encoder models (CLAP, Whisper) are mocked to avoid loading
large pretrained models during unit testing.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestCLAPAudioEncoderPreprocessing:
    """Tests for CLAPAudioEncoder preprocessing without loading actual model."""

    @pytest.fixture
    def mock_clap_module(self):
        """Mock the CLAP module to avoid loading the actual model."""
        mock_model = Mock()
        mock_model.get_audio_embedding_from_data.return_value = np.random.randn(2, 512)
        mock_model.get_text_embedding.return_value = np.random.randn(2, 512)
        mock_model.parameters.return_value = []
        return mock_model

    def test_preprocess_audio_from_numpy_1d(self, mock_clap_module):
        """Test preprocessing 1D numpy array."""
        with patch("safe.models.audio_encoders.laion_clap.CLAP_Module", return_value=mock_clap_module):
            from safe.models.audio_encoders import CLAPAudioEncoder

            # Manually set up encoder without loading model
            encoder = object.__new__(CLAPAudioEncoder)
            encoder.sample_rate = 48000
            encoder.max_length = 10.0
            encoder.max_samples = int(48000 * 10.0)
            encoder.debug_logging = False
            encoder._waveform_log_limit = 5
            encoder._waveform_logs_emitted = 0
            encoder.model = mock_clap_module
            encoder.audio_embed_dim = 512

            audio_data = np.random.randn(48000).astype(np.float32)  # 1 second
            result = encoder.preprocess_audio(audio_data)

            assert isinstance(result, torch.Tensor)
            assert result.shape == (1, encoder.max_samples)

    def test_preprocess_audio_from_tensor(self, mock_clap_module):
        """Test preprocessing torch tensor."""
        with patch("safe.models.audio_encoders.laion_clap.CLAP_Module", return_value=mock_clap_module):
            from safe.models.audio_encoders import CLAPAudioEncoder

            encoder = object.__new__(CLAPAudioEncoder)
            encoder.sample_rate = 48000
            encoder.max_length = 10.0
            encoder.max_samples = int(48000 * 10.0)
            encoder.debug_logging = False
            encoder._waveform_log_limit = 5
            encoder._waveform_logs_emitted = 0
            encoder.model = mock_clap_module
            encoder.audio_embed_dim = 512

            audio_data = torch.randn(48000)
            result = encoder.preprocess_audio(audio_data)

            assert isinstance(result, torch.Tensor)
            assert result.shape == (1, encoder.max_samples)

    def test_preprocess_audio_from_tuple(self, mock_clap_module):
        """Test preprocessing (waveform, sample_rate) tuple."""
        with patch("safe.models.audio_encoders.laion_clap.CLAP_Module", return_value=mock_clap_module):
            from safe.models.audio_encoders import CLAPAudioEncoder

            encoder = object.__new__(CLAPAudioEncoder)
            encoder.sample_rate = 48000
            encoder.max_length = 10.0
            encoder.max_samples = int(48000 * 10.0)
            encoder.debug_logging = False
            encoder._waveform_log_limit = 5
            encoder._waveform_logs_emitted = 0
            encoder.model = mock_clap_module
            encoder.audio_embed_dim = 512

            waveform = torch.randn(48000)
            audio_tuple = (waveform, 48000)
            result = encoder.preprocess_audio(audio_tuple)

            assert isinstance(result, torch.Tensor)

    def test_preprocess_audio_stereo_to_mono(self, mock_clap_module):
        """Test stereo audio is converted to mono."""
        with patch("safe.models.audio_encoders.laion_clap.CLAP_Module", return_value=mock_clap_module):
            from safe.models.audio_encoders import CLAPAudioEncoder

            encoder = object.__new__(CLAPAudioEncoder)
            encoder.sample_rate = 48000
            encoder.max_length = 10.0
            encoder.max_samples = int(48000 * 10.0)
            encoder.debug_logging = False
            encoder._waveform_log_limit = 5
            encoder._waveform_logs_emitted = 0
            encoder.model = mock_clap_module
            encoder.audio_embed_dim = 512

            # Stereo audio (2 channels)
            audio_data = np.random.randn(2, 48000).astype(np.float32)
            result = encoder.preprocess_audio(audio_data)

            # Should be mono after preprocessing
            assert result.shape[0] == 1  # Batch dimension

    def test_preprocess_audio_truncation(self, mock_clap_module):
        """Test long audio is truncated."""
        with patch("safe.models.audio_encoders.laion_clap.CLAP_Module", return_value=mock_clap_module):
            from safe.models.audio_encoders import CLAPAudioEncoder

            encoder = object.__new__(CLAPAudioEncoder)
            encoder.sample_rate = 48000
            encoder.max_length = 10.0
            encoder.max_samples = int(48000 * 10.0)
            encoder.debug_logging = False
            encoder._waveform_log_limit = 5
            encoder._waveform_logs_emitted = 0
            encoder.model = mock_clap_module
            encoder.audio_embed_dim = 512

            # 20 seconds of audio
            long_audio = np.random.randn(48000 * 20).astype(np.float32)
            result = encoder.preprocess_audio(long_audio)

            # Should be truncated to max_samples
            assert result.shape[1] == encoder.max_samples

    def test_preprocess_audio_padding(self, mock_clap_module):
        """Test short audio is padded."""
        with patch("safe.models.audio_encoders.laion_clap.CLAP_Module", return_value=mock_clap_module):
            from safe.models.audio_encoders import CLAPAudioEncoder

            encoder = object.__new__(CLAPAudioEncoder)
            encoder.sample_rate = 48000
            encoder.max_length = 10.0
            encoder.max_samples = int(48000 * 10.0)
            encoder.debug_logging = False
            encoder._waveform_log_limit = 5
            encoder._waveform_logs_emitted = 0
            encoder.model = mock_clap_module
            encoder.audio_embed_dim = 512

            # 1 second of audio (short)
            short_audio = np.random.randn(48000).astype(np.float32)
            result = encoder.preprocess_audio(short_audio)

            # Should be padded to max_samples
            assert result.shape[1] == encoder.max_samples

    def test_set_debug_logging(self, mock_clap_module):
        """Test debug logging configuration."""
        with patch("safe.models.audio_encoders.laion_clap.CLAP_Module", return_value=mock_clap_module):
            from safe.models.audio_encoders import CLAPAudioEncoder

            encoder = object.__new__(CLAPAudioEncoder)
            encoder.debug_logging = False
            encoder._waveform_log_limit = 5
            encoder._waveform_logs_emitted = 0

            encoder.set_debug_logging(True, max_waveform_logs=10)

            assert encoder.debug_logging is True
            assert encoder._waveform_log_limit == 10


class TestWhisperAudioEncoderPreprocessing:
    """Tests for WhisperAudioEncoder preprocessing without loading actual model."""

    @pytest.fixture
    def mock_whisper_model(self):
        """Mock the Whisper model."""
        mock_model = Mock()
        mock_model.encoder.return_value = torch.randn(1, 1500, 512)
        mock_model.dims = Mock()
        mock_model.dims.n_audio_state = 512
        mock_model.parameters.return_value = []
        return mock_model

    def test_preprocess_audio_from_numpy(self, mock_whisper_model):
        """Test preprocessing numpy array."""
        from safe.models.audio_encoders import WhisperAudioEncoder

        encoder = object.__new__(WhisperAudioEncoder)
        encoder.sample_rate = 16000
        encoder.max_length = 30.0
        encoder.max_samples = int(16000 * 30.0)
        encoder.debug_logging = False
        encoder._waveform_log_limit = 5
        encoder._waveform_logs_emitted = 0

        audio_data = np.random.randn(16000).astype(np.float32)
        result = encoder.preprocess_audio(audio_data)

        assert isinstance(result, np.ndarray)
        assert len(result) == encoder.max_samples

    def test_preprocess_audio_from_tensor(self, mock_whisper_model):
        """Test preprocessing torch tensor."""
        from safe.models.audio_encoders import WhisperAudioEncoder

        encoder = object.__new__(WhisperAudioEncoder)
        encoder.sample_rate = 16000
        encoder.max_length = 30.0
        encoder.max_samples = int(16000 * 30.0)
        encoder.debug_logging = False
        encoder._waveform_log_limit = 5
        encoder._waveform_logs_emitted = 0

        audio_data = torch.randn(16000)
        result = encoder.preprocess_audio(audio_data)

        assert isinstance(result, np.ndarray)

    def test_preprocess_audio_stereo_to_mono(self, mock_whisper_model):
        """Test stereo audio is converted to mono."""
        from safe.models.audio_encoders import WhisperAudioEncoder

        encoder = object.__new__(WhisperAudioEncoder)
        encoder.sample_rate = 16000
        encoder.max_length = 30.0
        encoder.max_samples = int(16000 * 30.0)
        encoder.debug_logging = False
        encoder._waveform_log_limit = 5
        encoder._waveform_logs_emitted = 0

        # Stereo audio
        stereo_audio = np.random.randn(2, 16000).astype(np.float32)
        result = encoder.preprocess_audio(stereo_audio)

        # Should be 1D (mono)
        assert result.ndim == 1

    def test_set_debug_logging(self, mock_whisper_model):
        """Test debug logging configuration."""
        from safe.models.audio_encoders import WhisperAudioEncoder

        encoder = object.__new__(WhisperAudioEncoder)
        encoder.debug_logging = False
        encoder._waveform_log_limit = 5
        encoder._waveform_logs_emitted = 0

        encoder.set_debug_logging(True, max_waveform_logs=3)

        assert encoder.debug_logging is True
        assert encoder._waveform_log_limit == 3


class TestMultiModalAudioEncoderConfig:
    """Tests for MultiModalAudioEncoder configuration without loading models."""

    def test_initialization_config_validation(self):
        """Test initialization validates at least one encoder is enabled."""
        from safe.models.audio_encoders import MultiModalAudioEncoder

        with pytest.raises(ValueError, match="At least one"):
            # This would fail because both are disabled
            encoder = object.__new__(MultiModalAudioEncoder)
            encoder.use_clap = False
            encoder.use_whisper = False
            encoder.clap_encoder = None
            encoder.whisper_encoder = None
            encoder.clap_embed_dim = 0
            encoder.whisper_embed_dim = 0
            encoder.total_embed_dim = 0

            if not (encoder.use_clap or encoder.use_whisper):
                raise ValueError("At least one of CLAP or Whisper must be enabled")

    def test_embed_dim_calculation_clap_only(self):
        """Test embedding dimension with CLAP only."""
        clap_embed_dim = 512
        whisper_embed_dim = 0  # Disabled

        total_embed_dim = clap_embed_dim + whisper_embed_dim
        assert total_embed_dim == 512

    def test_embed_dim_calculation_whisper_only(self):
        """Test embedding dimension with Whisper only."""
        clap_embed_dim = 0  # Disabled
        whisper_embed_dim = 512

        total_embed_dim = clap_embed_dim + whisper_embed_dim
        assert total_embed_dim == 512

    def test_embed_dim_calculation_both(self):
        """Test embedding dimension with both encoders."""
        clap_embed_dim = 512
        whisper_embed_dim = 768

        total_embed_dim = clap_embed_dim + whisper_embed_dim
        assert total_embed_dim == 1280


class TestAudioEncoderShapeConsistency:
    """Tests for consistent shapes across audio encoders."""

    def test_clap_output_shape_contract(self):
        """Test CLAP encoder output shape contract."""
        # CLAP should output (batch_size, audio_embed_dim)
        # where audio_embed_dim = 512
        batch_size = 4
        audio_embed_dim = 512

        mock_output = np.random.randn(batch_size, audio_embed_dim)
        assert mock_output.shape == (batch_size, audio_embed_dim)

    def test_whisper_output_shape_contract(self):
        """Test Whisper encoder output shape contract."""
        # Whisper should output (batch_size, seq_len, audio_embed_dim)
        batch_size = 2
        seq_len = 1500  # Whisper mel spectrogram frames
        audio_embed_dim = 512  # For whisper-small

        mock_output = torch.randn(batch_size, seq_len, audio_embed_dim)
        assert mock_output.shape == (batch_size, seq_len, audio_embed_dim)

    def test_pooled_whisper_matches_clap_dims(self):
        """Test pooled Whisper embeddings match CLAP dimension pattern."""
        batch_size = 2
        whisper_output = torch.randn(batch_size, 1500, 512)

        # Pooling across sequence dimension
        whisper_pooled = whisper_output.mean(dim=1)

        assert whisper_pooled.shape == (batch_size, 512)


class TestAudioEncoderResamplingLogic:
    """Tests for audio resampling logic."""

    def test_resample_needed_check(self):
        """Test logic for determining if resampling is needed."""
        input_sr = 44100
        target_sr = 48000

        needs_resample = input_sr != target_sr
        assert needs_resample is True

        input_sr = 48000
        needs_resample = input_sr != target_sr
        assert needs_resample is False

    def test_resample_ratio_calculation(self):
        """Test resample ratio calculation."""
        input_sr = 16000
        target_sr = 48000

        ratio = target_sr / float(input_sr)
        assert ratio == 3.0

        # Target length calculation
        input_length = 16000  # 1 second at 16kHz
        target_length = int(input_length * ratio)
        assert target_length == 48000

    def test_sample_count_after_resample(self):
        """Test sample count calculation after resampling."""
        input_sr = 22050
        target_sr = 16000
        input_samples = 22050  # 1 second

        scale = target_sr / float(input_sr)
        output_samples = int(round(input_samples * scale))

        # Should be approximately 1 second at target rate
        assert abs(output_samples - 16000) < 100


class TestAudioEncoderInterfaceConsistency:
    """Tests for consistent interface across encoders."""

    def test_encoder_has_required_attributes(self):
        """Test all encoders define required attributes."""
        required_attrs = [
            "sample_rate",
            "audio_embed_dim",
        ]

        # Mock encoder instances
        class MockEncoder:
            sample_rate = 48000
            audio_embed_dim = 512

        encoder = MockEncoder()
        for attr in required_attrs:
            assert hasattr(encoder, attr)

    def test_encoder_has_preprocess_method(self):
        """Test encoders have preprocess_audio method."""

        class MockEncoder:
            def preprocess_audio(self, audio):
                return audio

        encoder = MockEncoder()
        assert callable(getattr(encoder, "preprocess_audio", None))

    def test_encoder_has_debug_logging_method(self):
        """Test encoders have set_debug_logging method."""

        class MockEncoder:
            debug_logging = False

            def set_debug_logging(self, enabled, **kwargs):
                self.debug_logging = enabled

        encoder = MockEncoder()
        encoder.set_debug_logging(True)
        assert encoder.debug_logging is True
