import sys
import types

import numpy as np
import torch


def _install_dummy_audio_backends():
    """Install lightweight stubs for heavy audio dependencies used in SAFEModel."""

    if "laion_clap" not in sys.modules:
        dummy_clap = types.ModuleType("laion_clap")

        class _DummyClapModule:
            def __init__(self, *args, **kwargs):
                pass

            def load_ckpt(self, *args, **kwargs):
                return None

            def get_audio_embedding_from_data(self, x, use_tensor=False):
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


_install_dummy_audio_backends()

from safe.models.fusion_adapter import CrossAttentionBlock
from safe.models.safe_model import SAFEModel


def test_cross_attention_handles_extreme_inputs():
    block = CrossAttentionBlock(hidden_size=64, num_attention_heads=8)
    block.eval()

    hidden_states = torch.randn(2, 12, 64)
    audio_tokens = torch.randn(2, 4, 64)

    # Inject extreme values that previously produced NaNs during training
    hidden_states[0, 0, 0] = float("nan")
    hidden_states[1, 1, 1] = 1e6
    audio_tokens[0, 0, 0] = float("nan")
    audio_tokens[1, 2, 3] = -1e6

    output = block(hidden_states, audio_tokens)
    assert torch.isfinite(output).all()


def test_scatter_audio_tokens_retains_gradients():
    model = SAFEModel.__new__(SAFEModel)
    audio_tokens = torch.randn(2, 3, 5, requires_grad=True)
    indices = [0, 3]
    scattered = SAFEModel._scatter_audio_tokens(model, audio_tokens, indices, batch_size=4)

    loss = scattered.sum()
    loss.backward()

    assert audio_tokens.grad is not None
    # Gradients should flow back to the original tokens
    assert torch.isfinite(audio_tokens.grad).all()
