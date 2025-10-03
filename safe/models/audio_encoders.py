import torch
import numpy as np
import torch.nn as nn
import librosa
from typing import Any, Optional, Union, List, Tuple
import laion_clap
import whisper
from transformers import WhisperFeatureExtractor, WhisperModel


class CLAPAudioEncoder(nn.Module):
    """
    Frozen CLAP audio encoder for general audio semantics.
    """
    
    def __init__(
        self,
        model_name: str = 'laion/larger_clap_music_and_speech',
        freeze: bool = True,
        sample_rate: int = 48000,
        max_length: float = 10.0  # seconds
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_samples = int(sample_rate * max_length)
        self.debug_logging = False
        self._waveform_log_limit = 5
        self._waveform_logs_emitted = 0

        # Load CLAP model (suppress verbose output)
        import logging
        import sys
        import os
        from contextlib import redirect_stdout, redirect_stderr
        
        # Suppress all output during model loading
        old_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        try:
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    self.model = laion_clap.CLAP_Module(enable_fusion=False)
                    self.model.load_ckpt()  # Load default checkpoint
        finally:
            logging.getLogger().setLevel(old_level)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            
        # Get audio embedding dimension
        self.audio_embed_dim = 512  # CLAP audio embedding dimension

    def set_debug_logging(self, enabled: bool, max_waveform_logs: int = 5) -> None:
        """Enable or disable verbose waveform statistics logging."""

        self.debug_logging = bool(enabled)
        self._waveform_log_limit = int(max(0, max_waveform_logs))
        self._waveform_logs_emitted = 0

    # ------------------------------------------------------------------
    def _log_waveform_stats(self, audio_data: np.ndarray, source: str) -> None:
        if not self.debug_logging:
            return
        if self._waveform_logs_emitted >= self._waveform_log_limit:
            return

        flattened = audio_data.astype(np.float32).ravel()
        if flattened.size == 0:
            print(f"[AudioDebug] {source}: empty waveform", flush=True)
            self._waveform_logs_emitted += 1
            return

        max_val = float(np.max(flattened))
        min_val = float(np.min(flattened))
        mean_val = float(np.mean(flattened))
        mean_abs = float(np.mean(np.abs(flattened)))
        zero_fraction = float(np.mean(np.isclose(flattened, 0.0)))

        print(
            f"[AudioDebug] {source}: max={max_val:.6f} min={min_val:.6f} "
            f"mean={mean_val:.6f} mean|x|={mean_abs:.6f} zero_frac={zero_fraction:.3f}",
            flush=True,
        )
        self._waveform_logs_emitted += 1

    def preprocess_audio(self, audio: Union[torch.Tensor, np.ndarray, str, Tuple[Any, Any]]) -> torch.Tensor:
        """
        Preprocess audio to the format expected by CLAP.
        
        Args:
            audio: Can be file path (str), numpy array, or torch tensor
            
        Returns:
            Preprocessed audio tensor (1, max_samples)
        """
        input_sr = self.sample_rate

        if isinstance(audio, str):
            audio_data, input_sr = librosa.load(audio, sr=self.sample_rate)
        elif isinstance(audio, tuple) and len(audio) >= 1:
            candidate = audio[0]
            if isinstance(candidate, torch.Tensor):
                audio_data = candidate.detach().cpu().numpy()
            elif isinstance(candidate, np.ndarray):
                audio_data = candidate
            else:
                raise ValueError(f"Unsupported audio tuple payload: {type(candidate)}")
            if len(audio) > 1:
                try:
                    input_sr = int(audio[1])
                except Exception:
                    input_sr = self.sample_rate
        elif isinstance(audio, np.ndarray):
            audio_data = audio
        elif isinstance(audio, torch.Tensor):
            audio_data = audio.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

        # Ensure correct sample rate
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=0)  # Convert to mono

        if input_sr != self.sample_rate:
            try:
                audio_data = librosa.resample(audio_data, orig_sr=input_sr, target_sr=self.sample_rate)
            except Exception:
                # Fall back to naive slicing when resampling fails
                scale = self.sample_rate / float(max(input_sr, 1))
                target_len = int(round(len(audio_data) * scale))
                if target_len <= 0:
                    target_len = len(audio_data)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data) - 1, num=target_len, dtype=np.float32),
                    np.arange(len(audio_data), dtype=np.float32),
                    audio_data.astype(np.float32),
                )

        # Truncate or pad to max_length
        if len(audio_data) > self.max_samples:
            audio_data = audio_data[:self.max_samples]
        else:
            audio_data = np.pad(audio_data, (0, self.max_samples - len(audio_data)))

        self._log_waveform_stats(audio_data, "CLAP.preprocess_audio")

        return torch.from_numpy(audio_data).float().unsqueeze(0)  # (1, max_samples)
    
    def forward(self, audio: Union[torch.Tensor, List[Any]]) -> torch.Tensor:
        """
        Extract audio embeddings using CLAP.
        
        Args:
            audio: Batch of audio data (various formats supported)
            
        Returns:
            Audio embeddings (batch_size, audio_embed_dim)
        """
        if isinstance(audio, list):
            # Process batch of audio files/arrays
            processed_audio = []
            for idx, a in enumerate(audio):
                processed = self.preprocess_audio(a)
                processed_audio.append(processed)
            audio_batch = torch.stack(processed_audio)  # (batch_size, 1, max_samples)
            # Squeeze out the channel dimension for CLAP compatibility
            if audio_batch.dim() == 3 and audio_batch.shape[1] == 1:
                audio_batch = audio_batch.squeeze(1)  # (batch_size, max_samples)
        else:
            # Handle different audio tensor shapes
            if len(audio.shape) == 1:
                # Single 1D audio: (samples,) -> (1, samples)
                audio_batch = audio.unsqueeze(0)
            elif len(audio.shape) == 2:
                # Batch of 1D audio: (batch_size, samples) 
                audio_batch = audio
            elif len(audio.shape) == 3:
                # Already in correct format: (batch_size, channels, samples)
                audio_batch = audio.squeeze(1) if audio.shape[1] == 1 else audio
            else:
                raise ValueError(f"Unexpected audio tensor shape: {audio.shape}")
                
        # Extract embeddings - CLAP expects numpy arrays
        with torch.no_grad():
            # Convert to numpy for CLAP
            if isinstance(audio_batch, torch.Tensor):
                audio_numpy = audio_batch.detach().cpu().numpy()
            else:
                audio_numpy = audio_batch

            audio_embeddings = self.model.get_audio_embedding_from_data(
                x=audio_numpy,
                use_tensor=False
            )
            
        # Ensure output is tensor on correct device
        if not isinstance(audio_embeddings, torch.Tensor):
            audio_embeddings = torch.from_numpy(audio_embeddings)
        
        # Move to same device as input
        if isinstance(audio, torch.Tensor):
            audio_embeddings = audio_embeddings.to(audio.device)
            
        return audio_embeddings  # (batch_size, audio_embed_dim)


class WhisperAudioEncoder(nn.Module):
    """
    Whisper-based audio encoder for speech-heavy tasks.
    Can extract both embeddings and transcripts.
    """
    
    def __init__(
        self,
        model_name: str = "whisper-small",
        freeze: bool = True,
        extract_transcript: bool = True,
        sample_rate: int = 16000,
        max_length: float = 30.0  # seconds
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.max_length = max_length
        self.extract_transcript = extract_transcript
        self.max_samples = int(sample_rate * max_length)
        self.debug_logging = False
        self._waveform_log_limit = 5
        self._waveform_logs_emitted = 0
        
        # Load Whisper model
        self.model = whisper.load_model(model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/{model_name}")
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            
        # Get embedding dimension from Whisper encoder
        self.audio_embed_dim = self.model.dims.n_audio_state  # 512 for small, 768 for base, etc.

    def set_debug_logging(self, enabled: bool, max_waveform_logs: int = 5) -> None:
        self.debug_logging = bool(enabled)
        self._waveform_log_limit = int(max(0, max_waveform_logs))
        self._waveform_logs_emitted = 0

    def _log_waveform_stats(self, audio_data: np.ndarray, source: str) -> None:
        if not self.debug_logging:
            return
        if self._waveform_logs_emitted >= self._waveform_log_limit:
            return

        flattened = audio_data.astype(np.float32).ravel()
        if flattened.size == 0:
            print(f"[AudioDebug] {source}: empty waveform", flush=True)
            self._waveform_logs_emitted += 1
            return

        max_val = float(np.max(flattened))
        min_val = float(np.min(flattened))
        mean_val = float(np.mean(flattened))
        mean_abs = float(np.mean(np.abs(flattened)))
        zero_fraction = float(np.mean(np.isclose(flattened, 0.0)))

        print(
            f"[AudioDebug] {source}: max={max_val:.6f} min={min_val:.6f} "
            f"mean={mean_val:.6f} mean|x|={mean_abs:.6f} zero_frac={zero_fraction:.3f}",
            flush=True,
        )
        self._waveform_logs_emitted += 1
        
    def preprocess_audio(self, audio: Union[torch.Tensor, np.ndarray, str]) -> np.ndarray:
        """
        Preprocess audio for Whisper.
        
        Args:
            audio: Audio data in various formats
            
        Returns:
            Preprocessed audio array
        """
        if isinstance(audio, str):
            audio_data, sr = librosa.load(audio, sr=self.sample_rate)
        elif isinstance(audio, torch.Tensor):
            audio_data = audio.detach().cpu().numpy()
        else:
            audio_data = audio
            
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=0)
            
        # Whisper expects 30-second chunks
        if len(audio_data) > self.max_samples:
            audio_data = audio_data[:self.max_samples]
        else:
            audio_data = np.pad(audio_data, (0, self.max_samples - len(audio_data)))

        self._log_waveform_stats(audio_data, "Whisper.preprocess_audio")

        return audio_data
    
    def forward(
        self, 
        audio: Union[torch.Tensor, List[str], List[np.ndarray]]
    ) -> Tuple[torch.Tensor, Optional[List[str]]]:
        """
        Extract audio embeddings and optionally transcripts using Whisper.
        
        Args:
            audio: Batch of audio data
            
        Returns:
            Tuple of (embeddings, transcripts)
            embeddings: (batch_size, seq_len, audio_embed_dim)
            transcripts: List of transcript strings (if extract_transcript=True)
        """
        if isinstance(audio, list):
            processed_audio = [self.preprocess_audio(a) for a in audio]
            audio_batch = np.stack(processed_audio)
        else:
            if isinstance(audio, torch.Tensor):
                audio_batch = audio.cpu().numpy()
            else:
                audio_batch = audio
                
        embeddings_list = []
        transcripts_list = [] if self.extract_transcript else None
        
        for i in range(audio_batch.shape[0]):
            audio_sample = audio_batch[i]
            
            # Convert to mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_sample).unsqueeze(0)
            
            with torch.no_grad() if self.training == False else torch.enable_grad():
                # Extract embeddings from encoder
                encoder_output = self.model.encoder(mel)
                embeddings_list.append(encoder_output)
                
                # Extract transcript if requested
                if self.extract_transcript:
                    result = whisper.transcribe(self.model, audio_sample)
                    transcripts_list.append(result["text"])
        
        # Stack embeddings
        embeddings = torch.stack(embeddings_list)  # (batch_size, seq_len, embed_dim)
        
        return embeddings, transcripts_list


class MultiModalAudioEncoder(nn.Module):
    """
    Combined audio encoder that can use both CLAP and Whisper.
    """
    
    def __init__(
        self,
        use_clap: bool = True,
        use_whisper: bool = False,
        clap_config: dict = None,
        whisper_config: dict = None
    ):
        super().__init__()
        
        self.use_clap = use_clap
        self.use_whisper = use_whisper
        
        if use_clap:
            clap_config = clap_config or {}
            self.clap_encoder = CLAPAudioEncoder(**clap_config)
            self.clap_embed_dim = self.clap_encoder.audio_embed_dim
        else:
            self.clap_encoder = None
            self.clap_embed_dim = 0
            
        if use_whisper:
            whisper_config = whisper_config or {}
            self.whisper_encoder = WhisperAudioEncoder(**whisper_config)
            self.whisper_embed_dim = self.whisper_encoder.audio_embed_dim
        else:
            self.whisper_encoder = None
            self.whisper_embed_dim = 0
            
        # Total embedding dimension
        self.total_embed_dim = self.clap_embed_dim + (
            self.whisper_embed_dim if use_whisper else 0
        )
        
        if not (use_clap or use_whisper):
            raise ValueError("At least one of CLAP or Whisper must be enabled")
    
    def forward(
        self, 
        audio: Union[torch.Tensor, List[str], List[np.ndarray]]
    ) -> Tuple[torch.Tensor, Optional[List[str]]]:
        """
        Extract audio features using enabled encoders.
        
        Args:
            audio: Audio input
            
        Returns:
            Tuple of (combined_embeddings, transcripts)
        """
        embeddings_list = []
        transcripts = None
        
        if self.use_clap:
            clap_embeddings = self.clap_encoder(audio)  # (batch_size, clap_embed_dim)
            embeddings_list.append(clap_embeddings)
            
        if self.use_whisper:
            whisper_embeddings, transcripts = self.whisper_encoder(audio)
            # Pool whisper embeddings to match CLAP format
            whisper_pooled = whisper_embeddings.mean(dim=1)  # (batch_size, whisper_embed_dim)
            embeddings_list.append(whisper_pooled)
        
        # Concatenate embeddings
        if len(embeddings_list) > 1:
            combined_embeddings = torch.cat(embeddings_list, dim=-1)
        else:
            combined_embeddings = embeddings_list[0]
            
        return combined_embeddings, transcripts
