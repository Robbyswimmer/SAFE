from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

try:
    from transformers import CLIPImageProcessor, CLIPVisionModel
except ImportError:  # pragma: no cover - transformers is an optional runtime dep
    CLIPImageProcessor = None  # type: ignore
    CLIPVisionModel = None  # type: ignore


VideoInput = Union[torch.Tensor, Sequence[torch.Tensor]]


@dataclass
class VideoEncoderOutput:
    """Container that mirrors audio encoder outputs for parity."""

    embeddings: torch.Tensor
    frame_embeddings: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]


class CLIPVideoEncoder(nn.Module):
    """Wrap CLIP's vision backbone so we can encode video frame batches."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        num_frames: int = 8,
        temporal_pooling: str = "mean",
        freeze: bool = True,
        image_processor: Optional[CLIPImageProcessor] = None,
        vision_model: Optional[CLIPVisionModel] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if CLIPImageProcessor is None or CLIPVisionModel is None:
            raise ImportError(
                "transformers must be installed to use CLIPVideoEncoder"
            )

        self.model_name = model_name
        self.temporal_pooling = temporal_pooling
        self.num_frames = num_frames
        self.device = device

        self.processor = image_processor or CLIPImageProcessor.from_pretrained(model_name)
        self.vision_model = vision_model or CLIPVisionModel.from_pretrained(model_name)
        self.video_embed_dim = self.vision_model.config.hidden_size

        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            self.vision_model.eval()

        self.debug_logging = False
        self._log_limit = 5
        self._logs_emitted = 0

    # ------------------------------------------------------------------
    def set_debug_logging(self, enabled: bool, log_limit: int = 5) -> None:
        self.debug_logging = bool(enabled)
        self._log_limit = int(max(0, log_limit))
        self._logs_emitted = 0

    def _maybe_log(self, message: str) -> None:
        if self.debug_logging and self._logs_emitted < self._log_limit:
            print(message, flush=True)
            self._logs_emitted += 1

    # ------------------------------------------------------------------
    def preprocess_frames(self, video: VideoInput) -> Tuple[torch.Tensor, int, int]:
        """Convert a video tensor/list into CLIP-ready pixel values."""

        if isinstance(video, torch.Tensor):
            if video.dim() != 5:
                raise ValueError(
                    "Video tensor must be shaped (batch, frames, channels, height, width)"
                )
            batch, frames, channels, height, width = video.shape
            flattened = video.reshape(batch * frames, channels, height, width)
        else:
            frames_list: List[torch.Tensor] = []
            for frame in video:
                if frame.dim() != 4:
                    raise ValueError("Each frame tensor must be (batch, channels, height, width)")
                frames_list.append(frame)

            if not frames_list:
                raise ValueError("Video sequence must contain at least one frame")

            batch = frames_list[0].size(0)
            channels, height, width = frames_list[0].shape[1:]
            frames = len(frames_list)
            stacked = torch.stack(frames_list, dim=1)  # (batch, frames, C, H, W)
            flattened = stacked.reshape(batch * frames, channels, height, width)

        flattened = flattened.detach().cpu()
        pixel_values = self.processor(images=flattened, return_tensors="pt").pixel_values
        return pixel_values, batch, frames

    def forward(
        self,
        video: VideoInput,
        attention_mask: Optional[torch.Tensor] = None,
        return_frame_embeddings: bool = False,
    ) -> VideoEncoderOutput:
        pixel_values, batch, frames = self.preprocess_frames(video)

        if self.device is not None:
            pixel_values = pixel_values.to(self.device)
            attention_mask = attention_mask.to(self.device) if attention_mask is not None else None

        outputs = self.vision_model(pixel_values=pixel_values)
        frame_embeddings = outputs.pooler_output

        frame_embeddings = frame_embeddings.view(batch, frames, -1)

        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask for video must be shaped (batch, frames)")
            mask = attention_mask.to(frame_embeddings.device).unsqueeze(-1)
            masked = frame_embeddings * mask
            pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            if self.temporal_pooling == "mean":
                pooled = frame_embeddings.mean(dim=1)
            elif self.temporal_pooling == "max":
                pooled = frame_embeddings.max(dim=1).values
            else:  # pragma: no cover - defensive branch
                raise ValueError(f"Unsupported temporal pooling: {self.temporal_pooling}")

        if self.debug_logging:
            norm = pooled.norm(dim=-1).mean().item()
            self._maybe_log(f"[CLIPVideo] batch={batch} frames={frames} pooled_norm={norm:.3f}")

        return VideoEncoderOutput(
            embeddings=pooled,
            frame_embeddings=frame_embeddings if return_frame_embeddings else None,
            attention_mask=attention_mask,
        )
