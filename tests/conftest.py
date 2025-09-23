"""
Pytest configuration and shared fixtures for SAFE testing.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock
import sys
import types
import numpy as np


def _install_test_audio_stubs():
    """Install lightweight stand-ins for heavy audio dependencies during tests."""

    if "laion_clap" not in sys.modules:
        dummy_clap = types.ModuleType("laion_clap")

        class _DummyClapModule:
            def __init__(self, *args, **kwargs):
                pass

            def load_ckpt(self, *args, **kwargs):
                return None

            def get_audio_embedding_from_data(self, x, use_tensor=False):
                import numpy as np

                batch = len(x) if isinstance(x, (list, tuple)) else 1
                return np.zeros((batch, 512), dtype=np.float32)

        dummy_clap.CLAP_Module = lambda *args, **kwargs: _DummyClapModule()
        sys.modules["laion_clap"] = dummy_clap

    if "whisper" not in sys.modules:
        dummy_whisper = types.ModuleType("whisper")

        class _DummyWhisperModel:
            def __init__(self):
                self.dims = types.SimpleNamespace(n_audio_state=512)

            def parameters(self):
                return []

            def eval(self):
                return self

        def _load_model(*args, **kwargs):
            return _DummyWhisperModel()

        dummy_whisper.load_model = _load_model
        sys.modules["whisper"] = dummy_whisper


_install_test_audio_stubs()

from safe.models.safe_model import SAFEModel
from configs.model_configs import get_config


@pytest.fixture
def device():
    """Get the best available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def demo_config():
    """Get demo configuration for testing."""
    return get_config("demo")


@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def mock_audio_data():
    """Generate mock audio data for testing."""
    return {
        "waveform": torch.randn(2, 160000),  # 2 samples, 10 seconds at 16kHz
        "sample_rate": 16000,
        "duration": 10.0
    }


@pytest.fixture
def mock_image_data():
    """Generate mock image data for testing."""
    return {
        "images": torch.randn(2, 3, 224, 224),  # 2 samples, RGB 224x224
        "size": (224, 224)
    }


@pytest.fixture
def mock_text_data():
    """Generate mock text data for testing."""
    return {
        "questions": ["What sound do you hear?", "What do you see?"],
        "answers": ["Bird chirping", "A red car"],
        "input_ids": torch.randint(0, 1000, (2, 50)),  # 2 samples, 50 tokens
        "attention_mask": torch.ones(2, 50)
    }


@pytest.fixture
def mock_batch():
    """Generate a complete mock batch for testing."""
    return {
        "audio": torch.randn(2, 160000),
        "images": torch.randn(2, 3, 224, 224),
        "questions": ["What sound do you hear?", "What do you see?"],
        "answers": ["Bird chirping", "A red car"],
        "has_audio": torch.tensor([True, False]),
        "has_images": torch.tensor([False, True]),
        "question_types": ["audio_dependent", "visual_only"]
    }


@pytest.fixture
def lightweight_model(demo_config, device):
    """Create a lightweight SAFE model for testing."""
    # Override config for minimal resource usage
    config = demo_config.copy()
    config.update({
        "num_audio_tokens": 4,
        "lora_rank": 4,
        "fusion_layer_indices": [8]  # Single fusion layer
    })
    
    # Mock the model creation to avoid loading large pretrained models
    with pytest.MonkeyPatch().context() as m:
        # Mock heavy model components
        m.setattr("transformers.AutoModel.from_pretrained", 
                 lambda *args, **kwargs: MagicMock())
        m.setattr("transformers.AutoTokenizer.from_pretrained", 
                 lambda *args, **kwargs: MagicMock())
        m.setattr("laion_clap.CLAP_Module", lambda *args, **kwargs: MagicMock())
        
        model = SAFEModel(**config)
        model.to(device)
        return model


@pytest.fixture
def curriculum_config():
    """Sample curriculum configuration for testing."""
    return {
        "name": "test_curriculum",
        "stages": [
            {
                "name": "easy",
                "duration_epochs": 2,
                "audio_ratio": 0.3,
                "criteria": {
                    "min_audio_accuracy": 0.6,
                    "max_vl_degradation": 0.05
                }
            },
            {
                "name": "medium", 
                "duration_epochs": 3,
                "audio_ratio": 0.5,
                "criteria": {
                    "min_audio_accuracy": 0.7,
                    "max_vl_degradation": 0.03
                }
            },
            {
                "name": "hard",
                "duration_epochs": 5,
                "audio_ratio": 0.7,
                "criteria": {
                    "min_audio_accuracy": 0.8,
                    "max_vl_degradation": 0.02
                }
            }
        ]
    }


@pytest.fixture
def dataset_registry():
    """Mock dataset registry for testing."""
    return {
        "avqa_test": {
            "type": "AVQA",
            "path": "test_data/avqa",
            "split": "test",
            "difficulty": "medium",
            "modalities": ["audio", "visual"],
            "size": 100
        },
        "audiocaps_test": {
            "type": "AudioCaps",
            "path": "test_data/audiocaps", 
            "split": "test",
            "difficulty": "easy",
            "modalities": ["audio"],
            "size": 200
        },
        "vqa_test": {
            "type": "VQA",
            "path": "test_data/vqa",
            "split": "test", 
            "difficulty": "easy",
            "modalities": ["visual"],
            "size": 150
        }
    }


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for validation."""
    return {
        "audio_tasks": {
            "min_accuracy": 0.65,
            "target_accuracy": 0.80
        },
        "vl_retention": {
            "max_degradation": 0.005,  # 0.5%
            "warning_degradation": 0.003  # 0.3%
        },
        "efficiency": {
            "min_skip_rate": 0.4,  # 40%
            "target_skip_rate": 0.7  # 70%
        }
    }


class MockDataset:
    """Mock dataset class for testing."""
    
    def __init__(self, size: int = 100, modalities: List[str] = None):
        self.size = size
        self.modalities = modalities or ["audio", "visual"]
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        item = {
            "question": f"Test question {idx}",
            "answer": f"Test answer {idx}",
            "question_type": "test"
        }
        
        if "audio" in self.modalities:
            item["audio"] = torch.randn(160000)
        else:
            item["audio"] = None
            
        if "visual" in self.modalities:
            item["images"] = torch.randn(3, 224, 224)
        else:
            item["images"] = None
            
        return item


@pytest.fixture
def mock_dataset():
    """Create mock dataset for testing."""
    return MockDataset


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration  
pytest.mark.curriculum = pytest.mark.curriculum
pytest.mark.dataset = pytest.mark.dataset
pytest.mark.slow = pytest.mark.slow
pytest.mark.gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


# Helper functions
def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple, name: str = "tensor"):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"{name} shape {tensor.shape} != expected {expected_shape}"


def assert_tensor_range(tensor: torch.Tensor, min_val: float = None, max_val: float = None, name: str = "tensor"):
    """Assert tensor values are in expected range."""
    if min_val is not None:
        assert tensor.min() >= min_val, f"{name} min value {tensor.min()} < {min_val}"
    if max_val is not None:
        assert tensor.max() <= max_val, f"{name} max value {tensor.max()} > {max_val}"


def assert_no_nan_inf(tensor: torch.Tensor, name: str = "tensor"):
    """Assert tensor contains no NaN or infinite values."""
    assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
    assert not torch.isinf(tensor).any(), f"{name} contains infinite values"