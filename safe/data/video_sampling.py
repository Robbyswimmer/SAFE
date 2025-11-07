"""Utility helpers for loading and sampling video clips for SAFE."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover - optional dependency
    from torchvision.io import read_video  # type: ignore
    import torchvision.transforms.functional as TF  # type: ignore
except Exception:  # pragma: no cover - keep optional
    read_video = None  # type: ignore
    TF = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - keep optional
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - keep optional
    Image = None  # type: ignore


@dataclass
class VideoClip:
    """Container for a sampled video clip."""

    frames: torch.Tensor  # (num_frames, channels, height, width)
    fps: Optional[float] = None

    def to_device(self, device: torch.device | str) -> "VideoClip":
        self.frames = self.frames.to(device)
        return self


def _ensure_tensor(frames: torch.Tensor) -> torch.Tensor:
    if frames.dtype != torch.float32:
        frames = frames.float()
    if frames.max() > 1.5:  # assume 0-255 range
        frames = frames / 255.0
    return frames


def _sample_frame_indices(total: int, target: int) -> torch.Tensor:
    if total <= 0:
        raise ValueError("Video has no frames")
    if target >= total:
        return torch.arange(total)
    return torch.linspace(0, total - 1, target).round().long()


def load_video_clip(
    path: str | Path,
    num_frames: int = 8,
    resize: Optional[Tuple[int, int]] = (224, 224),
    device: Optional[torch.device] = None,
) -> Optional[VideoClip]:
    """Load a video from disk and sample a fixed number of frames.

    This helper keeps dependencies optional.  If :mod:`torchvision` is
    unavailable, the function returns ``None`` so call sites can fall back to a
    different pipeline or skip video inputs gracefully.
    """

    if read_video is None:
        return None

    video_path = Path(path)
    if not video_path.exists():
        return None

    try:
        frames, _, info = read_video(str(video_path), pts_unit="sec")
    except Exception:
        return None

    if frames.numel() == 0:
        return None

    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
    frames = _ensure_tensor(frames)

    indices = _sample_frame_indices(frames.shape[0], num_frames)
    clip = frames[indices]

    if resize is not None and TF is not None:
        clip = TF.resize(clip, resize)

    if device is not None:
        clip = clip.to(device)

    fps = None
    if isinstance(info, dict):
        fps = info.get("video_fps") or info.get("audio_fps")

    return VideoClip(frames=clip, fps=fps)


def stack_clips(clips: Sequence[Optional[VideoClip]]) -> Optional[torch.Tensor]:
    """Stack a sequence of clips into a batch tensor.

    Clips that are ``None`` are ignored.  If no valid clips remain, returns
    ``None`` so downstream code can skip video fusion for that batch.
    """

    valid = [clip.frames for clip in clips if clip is not None]
    if not valid:
        return None
    return torch.stack(valid, dim=0)


@dataclass
class DepthMap:
    """Container for depth sensor outputs."""

    values: torch.Tensor  # (1, H, W) or (H, W)
    scale: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_device(self, device: torch.device | str) -> "DepthMap":
        self.values = self.values.to(device)
        return self


def load_depth_map(
    path: str | Path,
    *,
    normalize: bool = True,
    device: Optional[torch.device] = None,
) -> Optional[DepthMap]:
    """Load a depth map stored as ``.npy``, ``.pt`` or image."""

    depth_path = Path(path)
    if not depth_path.exists():
        return None

    tensor: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = {}

    suffix = depth_path.suffix.lower()
    try:
        if suffix in {".npy", ".npz"} and np is not None:
            array = np.load(str(depth_path))
            if isinstance(array, np.lib.npyio.NpzFile):
                # Use first array found
                key = array.files[0]
                metadata["array_key"] = key
                array = array[key]
            tensor = torch.from_numpy(array)
        elif suffix == ".pt":
            tensor = torch.load(str(depth_path))
        elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
            if Image is None or np is None:
                return None
            img = Image.open(depth_path)
            metadata["mode"] = img.mode
            tensor = torch.from_numpy(np.array(img))
        else:
            return None
    except Exception:
        return None

    if tensor is None:
        return None

    tensor = tensor.float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.size(0) != 1:
        # Depth maps are expected to be single channel; average if necessary.
        tensor = tensor.mean(dim=0, keepdim=True)

    scale = None
    if normalize and tensor.numel() > 0:
        max_val = tensor.max().item()
        if max_val > 0:
            scale = max_val
            tensor = tensor / max_val

    if device is not None:
        tensor = tensor.to(device)

    return DepthMap(values=tensor, scale=scale, metadata=metadata)


@dataclass
class TactileSequence:
    """Container for tactile sensor readings."""

    signals: torch.Tensor  # (timesteps, features)
    sampling_rate_hz: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_device(self, device: torch.device | str) -> "TactileSequence":
        self.signals = self.signals.to(device)
        return self


def load_tactile_sequence(
    path: str | Path,
    *,
    normalize: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    sampling_rate_hz: Optional[float] = None,
) -> Optional[TactileSequence]:
    """Load tactile sensor readings from ``.npy``, ``.pt`` or ``.csv``."""

    tactile_path = Path(path)
    if not tactile_path.exists():
        return None

    suffix = tactile_path.suffix.lower()
    tensor: Optional[torch.Tensor] = None

    try:
        if suffix in {".npy", ".npz"} and np is not None:
            array = np.load(str(tactile_path))
            if isinstance(array, np.lib.npyio.NpzFile):
                array = array[array.files[0]]
            tensor = torch.from_numpy(array)
        elif suffix == ".pt":
            tensor = torch.load(str(tactile_path))
        elif suffix == ".csv" and np is not None:
            array = np.loadtxt(str(tactile_path), delimiter=",")
            tensor = torch.from_numpy(array)
        else:
            return None
    except Exception:
        return None

    if tensor is None:
        return None

    tensor = tensor.to(dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)

    if normalize and tensor.numel() > 0:
        std = tensor.std()
        if torch.isfinite(std) and std > 0:
            tensor = (tensor - tensor.mean()) / std

    if device is not None:
        tensor = tensor.to(device)

    return TactileSequence(
        signals=tensor,
        sampling_rate_hz=sampling_rate_hz,
        metadata={"source": str(tactile_path)}
    )


def pad_tactile_sequences(
    sequences: Sequence[Optional[TactileSequence]],
    *,
    padding_value: float = 0.0,
    device: Optional[torch.device] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Pad tactile sequences to a uniform length and return mask."""

    valid = [seq for seq in sequences if seq is not None]
    if not valid:
        return None

    lengths = [seq.signals.size(0) for seq in valid]
    max_len = max(lengths)
    feature_dim = valid[0].signals.size(-1)

    batch = torch.full((len(valid), max_len, feature_dim), padding_value, dtype=valid[0].signals.dtype)
    mask = torch.zeros(len(valid), max_len, dtype=torch.bool)

    for idx, seq in enumerate(valid):
        length = seq.signals.size(0)
        batch[idx, :length] = seq.signals
        mask[idx, :length] = True

    if device is not None:
        batch = batch.to(device)
        mask = mask.to(device)

    return batch, mask
