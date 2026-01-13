"""
Audio augmentation utilities for SAFE training.

Implements SpecAugment-style augmentations adapted for waveform inputs,
plus additional audio-specific augmentations for improved generalization.
"""

import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """
    SpecAugment-style augmentation for audio waveforms.

    Converts waveform to spectrogram, applies time/frequency masking,
    then converts back to waveform for compatibility with CLAP encoder.

    Based on: "SpecAugment: A Simple Data Augmentation Method for ASR"
    https://arxiv.org/abs/1904.08779

    Args:
        freq_mask_param: Maximum width of frequency mask (F in paper)
        time_mask_param: Maximum width of time mask (T in paper)
        num_freq_masks: Number of frequency masks to apply
        num_time_masks: Number of time masks to apply
        p: Probability of applying augmentation
        sample_rate: Audio sample rate (for spectrogram computation)
        n_fft: FFT size for spectrogram
        hop_length: Hop length for spectrogram
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        p: float = 0.5,
        sample_rate: int = 48000,
        n_fft: int = 1024,
        hop_length: int = 512,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freqs = n_fft // 2 + 1

    def _apply_freq_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking to spectrogram."""
        # spec shape: (freq_bins, time_steps)
        n_freqs = spec.shape[0]

        for _ in range(self.num_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, n_freqs - 1))
            f0 = random.randint(0, n_freqs - f)
            spec[f0:f0 + f, :] = 0

        return spec

    def _apply_time_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply time masking to spectrogram."""
        # spec shape: (freq_bins, time_steps)
        n_time = spec.shape[1]

        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, n_time - 1))
            t0 = random.randint(0, n_time - t)
            spec[:, t0:t0 + t] = 0

        return spec

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to waveform.

        Args:
            waveform: Audio waveform tensor of shape (samples,) or (batch, samples)

        Returns:
            Augmented waveform of same shape
        """
        if random.random() > self.p:
            return waveform

        # Handle batched input
        if waveform.dim() == 1:
            return self._augment_single(waveform)
        elif waveform.dim() == 2:
            return torch.stack([self._augment_single(w) for w in waveform])
        else:
            return waveform

    def _augment_single(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a single waveform."""
        # Compute STFT
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            window=torch.hann_window(self.n_fft, device=waveform.device),
        )

        # Get magnitude and phase
        magnitude = torch.abs(spec)
        phase = torch.angle(spec)

        # Apply frequency and time masking to magnitude
        magnitude = self._apply_freq_mask(magnitude)
        magnitude = self._apply_time_mask(magnitude)

        # Reconstruct complex spectrogram
        spec_augmented = magnitude * torch.exp(1j * phase)

        # Inverse STFT
        waveform_augmented = torch.istft(
            spec_augmented,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=waveform.device),
            length=waveform.shape[-1],
        )

        return waveform_augmented


class WaveformAugment(nn.Module):
    """
    Waveform-level augmentations for audio.

    Applies various augmentations directly to the waveform:
    - Time stretching (simulated via resampling)
    - Pitch shifting (simulated)
    - Noise injection
    - Volume perturbation
    - Time shifting

    Args:
        noise_level: Maximum noise amplitude (relative to signal)
        volume_range: Range for volume perturbation (min, max)
        time_shift_range: Maximum time shift in samples
        p: Probability of applying each augmentation
    """

    def __init__(
        self,
        noise_level: float = 0.005,
        volume_range: Tuple[float, float] = (0.8, 1.2),
        time_shift_range: int = 4800,  # 0.1 seconds at 48kHz
        p: float = 0.5,
    ):
        super().__init__()
        self.noise_level = noise_level
        self.volume_range = volume_range
        self.time_shift_range = time_shift_range
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply waveform augmentations.

        Args:
            waveform: Audio waveform tensor of shape (samples,) or (batch, samples)

        Returns:
            Augmented waveform of same shape
        """
        if waveform.dim() == 1:
            return self._augment_single(waveform)
        elif waveform.dim() == 2:
            return torch.stack([self._augment_single(w) for w in waveform])
        else:
            return waveform

    def _augment_single(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to a single waveform."""
        # Volume perturbation
        if random.random() < self.p:
            volume_factor = random.uniform(*self.volume_range)
            waveform = waveform * volume_factor

        # Noise injection
        if random.random() < self.p:
            noise = torch.randn_like(waveform) * self.noise_level
            waveform = waveform + noise

        # Time shifting (circular shift)
        if random.random() < self.p:
            shift = random.randint(-self.time_shift_range, self.time_shift_range)
            waveform = torch.roll(waveform, shifts=shift)

        return waveform


class AudioAugmentPipeline(nn.Module):
    """
    Combined audio augmentation pipeline.

    Applies both SpecAugment and waveform augmentations during training.

    Args:
        use_spec_augment: Whether to use SpecAugment
        use_waveform_augment: Whether to use waveform augmentations
        spec_augment_config: Config dict for SpecAugment
        waveform_augment_config: Config dict for WaveformAugment
    """

    def __init__(
        self,
        use_spec_augment: bool = True,
        use_waveform_augment: bool = True,
        spec_augment_config: Optional[dict] = None,
        waveform_augment_config: Optional[dict] = None,
    ):
        super().__init__()

        self.use_spec_augment = use_spec_augment
        self.use_waveform_augment = use_waveform_augment

        if use_spec_augment:
            config = spec_augment_config or {}
            self.spec_augment = SpecAugment(**config)
        else:
            self.spec_augment = None

        if use_waveform_augment:
            config = waveform_augment_config or {}
            self.waveform_augment = WaveformAugment(**config)
        else:
            self.waveform_augment = None

    def forward(
        self,
        waveform: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Apply augmentation pipeline.

        Args:
            waveform: Audio waveform tensor
            training: Whether in training mode (augmentations only applied if True)

        Returns:
            Augmented waveform (or original if not training)
        """
        if not training:
            return waveform

        # Apply waveform augmentations first
        if self.waveform_augment is not None:
            waveform = self.waveform_augment(waveform)

        # Then apply SpecAugment
        if self.spec_augment is not None:
            waveform = self.spec_augment(waveform)

        return waveform


def create_augment_pipeline(
    enabled: bool = True,
    spec_augment: bool = True,
    waveform_augment: bool = True,
    **kwargs,
) -> Optional[AudioAugmentPipeline]:
    """
    Factory function to create augmentation pipeline.

    Args:
        enabled: Whether to create pipeline (returns None if False)
        spec_augment: Enable SpecAugment
        waveform_augment: Enable waveform augmentations
        **kwargs: Additional config passed to pipeline

    Returns:
        AudioAugmentPipeline instance or None
    """
    if not enabled:
        return None

    return AudioAugmentPipeline(
        use_spec_augment=spec_augment,
        use_waveform_augment=waveform_augment,
        spec_augment_config=kwargs.get("spec_augment_config"),
        waveform_augment_config=kwargs.get("waveform_augment_config"),
    )
