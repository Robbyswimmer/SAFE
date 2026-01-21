#!/usr/bin/env python3
"""
train_safe.py - Clean, optimized training script for SAFE audio captioning

Focuses on essentials:
- Audio captioning task (cross-entropy loss)
- CIDEr/BLEU evaluation metrics
- Memory-efficient training (gradient accumulation, mixed precision)
- Clear result tracking

Removed:
- Retention loss mechanisms (no longer needed)
- Curriculum learning
- Null-space projection
- SCST fine-tuning
- Complex dataset mixing
"""

import argparse
import json
import os
import platform
import random
import socket
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Optional: Weights & Biases
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

# SAFE imports
from configs.model_configs import get_config
from safe.data.datasets import AudioCapsDataset, WavCapsDataset, ClothoDataset, MACSDataset, create_safe_dataloader
from safe.data.audio_augment import create_augment_pipeline, AudioAugmentPipeline
from safe.models.safe_model import SAFEModel


# ============================================================================
# SECTION 1: UTILITIES
# ============================================================================

def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

        return {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "is_main": rank == 0,
            "distributed": True,
        }
    else:
        return {
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "is_main": True,
            "distributed": False,
        }


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(dist_info: Dict) -> bool:
    """Check if this is the main process (for logging/saving)."""
    return dist_info["is_main"]


def set_seed(seed: int, rank: int = 0):
    """Set random seeds for reproducibility (offset by rank for distributed)"""
    seed = seed + rank  # Different seed per process for data shuffling
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def cast_trainable_params_to_fp32(model: nn.Module) -> int:
    """
    Ensure all *trainable* parameters are fp32.

    AMP GradScaler cannot unscale fp16 gradients. If trainable params (e.g., LoRA
    weights created under a fp16 base model) are fp16, GradScaler will error with:
      ValueError: Attempting to unscale FP16 gradients.
    """
    converted = 0
    for p in model.parameters():
        if getattr(p, "requires_grad", False) and p.dtype == torch.float16:
            p.data = p.data.float()
            converted += 1
    return converted


def summarize_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Return a breakdown of trainable parameters by component and sub-type.
    Keys are stable prefixes suitable for logging / reporting.
    """
    totals: Dict[str, int] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = int(param.numel())

        # Coarse component split
        if name.startswith("audio_projector."):
            bucket = "audio_projector"
        elif name.startswith("fusion_adapter."):
            bucket = "fusion_adapter"
        elif name.startswith("audio_token_embeddings."):
            bucket = "audio_token_embeddings"
        elif "kv_adapter" in name or "kv_augmentation" in name:
            bucket = "kv_adapter"
        else:
            bucket = "other_trainable"
        totals[bucket] = totals.get(bucket, 0) + n

        # KV adapter detailed breakdown
        if bucket == "kv_adapter":
            lower = name.lower()
            if "audio_query_adapter" in lower or "query_adapter" in lower:
                sub = "kv_adapter/query_adapter"
            elif "key_proj" in lower or "k_proj" in lower:
                sub = "kv_adapter/key_proj"
            elif "value_proj" in lower or "v_proj" in lower:
                sub = "kv_adapter/value_proj"
            elif "scale" in lower:
                sub = "kv_adapter/scale"
            else:
                sub = "kv_adapter/other"
            totals[sub] = totals.get(sub, 0) + n

        # More detailed fusion breakdown (helps explain big parameter drops)
        if bucket == "fusion_adapter":
            lower = name.lower()
            if "lora_" in lower or ".lora_" in lower:
                sub = "fusion_adapter/lora"
            elif lower.endswith("residual_scale"):
                sub = "fusion_adapter/residual_scale"
            elif "token_gate" in lower:
                sub = "fusion_adapter/token_gate"
            else:
                sub = "fusion_adapter/base_or_other"
            totals[sub] = totals.get(sub, 0) + n

        # Projector detail (dominant term is usually the output projection)
        if bucket == "audio_projector":
            lower = name.lower()
            if "projector.3" in lower or "projector.2" in lower:
                # Heuristic: final Linear in nn.Sequential is near the end
                sub = "audio_projector/final_linear"
            elif "projector." in lower:
                sub = "audio_projector/mlp"
            elif lower.endswith("output_scale"):
                sub = "audio_projector/output_scale"
            elif "layernorm" in lower or "layer_norm" in lower or ".norm" in lower:
                sub = "audio_projector/norms"
            else:
                sub = "audio_projector/other"
            totals[sub] = totals.get(sub, 0) + n

    return totals


def _format_param_count(n: int) -> str:
    return f"{n/1e6:.2f}M"


def _sanitize_filename(text: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(text))
    return safe.strip("_")[:80] or "sample"


def _export_eval_samples(
    output_dir: Path,
    *,
    split: str,
    epoch: int,
    samples: List[Dict[str, Any]],
    export_audio: bool = False,
) -> Optional[Path]:
    """
    Export qualitative samples for advisor-facing sanity checks.
    Writes JSON always; optionally writes WAV files for a small sample set.
    """
    if not samples:
        return None

    out_root = Path(output_dir) / "eval_samples" / split / f"epoch_{int(epoch)}"
    out_root.mkdir(parents=True, exist_ok=True)
    audio_dir = out_root / "audio"
    if export_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    exported: List[Dict[str, Any]] = []
    for idx, row in enumerate(samples):
        sample_id = row.get("sample_id") or row.get("id") or idx
        audio_path = row.get("audio_path")
        subset = row.get("subset")
        question = row.get("question")
        prediction = row.get("prediction")
        references = row.get("references")

        wav_out = None
        if export_audio:
            audio = row.get("audio")
            if (
                isinstance(audio, tuple)
                and len(audio) == 2
                and torch.is_tensor(audio[0])
                and isinstance(audio[1], (int, float))
            ):
                waveform_tensor = audio[0].detach().cpu().float().flatten()
                sample_rate = int(audio[1])
                # Clamp to [-1,1] then write 16-bit PCM WAV.
                waveform_tensor = torch.clamp(waveform_tensor, -1.0, 1.0)
                pcm16 = (waveform_tensor.numpy() * 32767.0).astype("int16")

                fname = f"{idx:04d}_{_sanitize_filename(sample_id)}.wav"
                wav_path = audio_dir / fname
                with wave.open(str(wav_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(pcm16.tobytes())
                wav_out = str(wav_path)

        exported.append(
            {
                "sample_id": sample_id,
                "subset": subset,
                "audio_path": audio_path,
                "exported_wav": wav_out,
                "question": question,
                "prediction": prediction,
                "references": references,
            }
        )

    json_path = out_root / "samples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(exported, f, indent=2, ensure_ascii=False)

    latest = Path(output_dir) / "eval_samples" / split / "samples_latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(latest, "w", encoding="utf-8") as f:
            json.dump(exported, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return json_path


# ----------------------------------------------------------------------------
# Weights & Biases helpers (cluster-friendly)
# ----------------------------------------------------------------------------

def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _get_rank() -> int:
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        value = _env_int(key)
        if value is not None:
            return value
    return 0


def _is_main_process() -> bool:
    return _get_rank() == 0


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _wandb_log(wandb_run: Any, data: Dict[str, Any], *, step: Optional[int] = None):
    if wandb_run is None:
        return
    try:
        if step is None:
            wandb_run.log(data)
        else:
            wandb_run.log(data, step=step)
    except Exception:
        pass


def _wandb_log_artifact(
    wandb_run: Any,
    *,
    artifact_name: str,
    artifact_type: str,
    files: List[Path],
    aliases: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    if wandb_run is None or wandb is None:
        return
    try:
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=_jsonify(metadata or {}),
        )
        for file_path in files:
            artifact.add_file(str(file_path))
        wandb_run.log_artifact(artifact, aliases=aliases or [])
    except Exception:
        pass


def _maybe_init_wandb(
    args: argparse.Namespace,
    *,
    train_config: Dict[str, Any],
    model_config: Dict[str, Any],
    output_dir: Path,
    model: Optional[nn.Module] = None,
    total_params: Optional[int] = None,
    trainable_params: Optional[int] = None,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
) -> Any:
    if not getattr(args, "wandb", False):
        return None
    if not _is_main_process():
        return None
    if wandb is None:
        print("⚠️  wandb failed to import; continuing without W&B logging.", flush=True)
        return None

    wandb_mode = getattr(args, "wandb_mode", None) or os.environ.get("WANDB_MODE")
    if wandb_mode is None:
        # Default to offline if no key is provided (prevents interactive login prompts on clusters).
        wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    if wandb_mode == "disabled":
        return None

    wandb_dir = getattr(args, "wandb_dir", None) or os.environ.get("WANDB_DIR") or str(output_dir / "wandb")
    tags: List[str] = []
    raw_tags = getattr(args, "wandb_tags", None)
    if raw_tags:
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    run_name = getattr(args, "wandb_name", None) or (f"{output_dir.name}-{slurm_job_id}" if slurm_job_id else output_dir.name)
    run_group = getattr(args, "wandb_group", None) or slurm_job_id

    try:
        settings = wandb.Settings(start_method="thread")
    except Exception:
        settings = None

    wandb_config = {
        "args": _jsonify(vars(args)),
        "train_config": _jsonify(train_config),
        "model_config": _jsonify(model_config),
        "data": _jsonify({"train_size": train_size, "val_size": val_size}),
        "params": _jsonify({"total": total_params, "trainable": trainable_params}),
        "system": _jsonify(
            {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "slurm_job_id": slurm_job_id,
            }
        ),
    }

    init_kwargs: Dict[str, Any] = dict(
        project=getattr(args, "wandb_project", None) or os.environ.get("WANDB_PROJECT") or "SAFE",
        entity=getattr(args, "wandb_entity", None) or os.environ.get("WANDB_ENTITY"),
        name=run_name,
        group=run_group,
        tags=tags if tags else None,
        dir=wandb_dir,
        mode=wandb_mode,
        config=wandb_config,
        save_code=bool(getattr(args, "wandb_log_code", False)),
        notes=getattr(args, "wandb_notes", None),
    )
    if settings is not None:
        init_kwargs["settings"] = settings

    try:
        wandb_run = wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"⚠️  wandb.init failed ({exc}); continuing without W&B logging.", flush=True)
        return None

    # Make charts use our optimizer step when provided.
    try:
        wandb.define_metric("train/optimizer_step")
        for prefix in ("train/*", "val/*", "train_subset/*", "eval/*"):
            wandb.define_metric(prefix, step_metric="train/optimizer_step")
    except Exception:
        pass

    watch_mode = getattr(args, "wandb_watch", "false")
    if model is not None and watch_mode and watch_mode.lower() != "false":
        try:
            wandb_run.watch(
                model,
                log=watch_mode,
                log_freq=int(getattr(args, "wandb_watch_log_freq", 500)),
            )
        except Exception:
            pass

    return wandb_run


def _is_valid_caption_reference(answer: Any) -> bool:
    """
    Return True if `answer` looks like a usable caption reference.

    We treat None/empty strings/empty lists and common sentinel strings like
    "None"/"nan"/"null" as invalid. Lists/dicts are considered valid if they
    contain at least one valid caption-like value.
    """
    if answer is None:
        return False

    if isinstance(answer, str):
        candidate = answer.strip().lower()
        return bool(candidate) and candidate not in {"none", "nan", "null"}

    if isinstance(answer, dict):
        candidate = (
            answer.get("answer")
            or answer.get("text")
            or answer.get("caption")
            or answer.get("captions")
            or answer.get("answers")
        )
        return _is_valid_caption_reference(candidate)

    if isinstance(answer, (list, tuple)):
        if not answer:
            return False
        return any(_is_valid_caption_reference(item) for item in answer)

    candidate = str(answer).strip().lower()
    return bool(candidate) and candidate not in {"none", "nan", "null"}


# Global cache for HuggingFace evaluate metrics (to avoid re-loading)
_EVALUATE_METRIC_CACHE: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Any] = {}


def _normalize_audio_caption(text: Any) -> str:
    """
    Normalization logic copied from StageATrainer._normalize_audio_caption
    so that metrics match the main training pipeline.
    """
    if text is None:
        return ""

    if isinstance(text, (list, tuple)):
        text = " ".join(str(t) for t in text if t)
    elif isinstance(text, dict):
        value = text.get("answer") or text.get("text")
        text = value if value is not None else ""

    import re
    import unicodedata

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\u2019", "'")  # Normalize curly apostrophes
    normalized = normalized.lower()

    # Collapse possessives before stripping punctuation so "dog's" -> "dogs"
    normalized = re.sub(r"'s\b", "s", normalized)

    # Remove residual apostrophes and punctuation (keep alphanumerics + whitespace)
    normalized = re.sub(r"'", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)

    tokens = [tok for tok in normalized.split() if tok]
    if not tokens:
        return ""

    number_map = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
    }

    cleaned_tokens: List[str] = []
    for tok in tokens:
        cleaned_tokens.append(number_map.get(tok, tok))

    if not cleaned_tokens:
        return ""

    return " ".join(cleaned_tokens)


def _embed_texts_for_contrastive(
    model: SAFEModel,
    texts: List[str],
    device: torch.device,
    max_length: int = 48,
) -> torch.Tensor:
    """
    Embed texts using the base LLM embedding layer, pooled over tokens.
    Mirrors StageATrainer._embed_texts behavior.
    """
    # Handle DDP wrapper - get underlying model for attribute access
    base_model = model.module if hasattr(model, 'module') else model
    tokenizer = base_model.base_vl.tokenizer
    embedding_layer = base_model.base_vl.llm.get_input_embeddings()
    hidden_size = embedding_layer.weight.size(1)

    if not texts:
        return torch.empty(0, hidden_size, device=device)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        text_embeds = embedding_layer(input_ids)

    mask = attention_mask.unsqueeze(-1)
    pooled = (text_embeds * mask).sum(dim=1).float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    return pooled / denom

class MixedAudioCaptionDataset(Dataset):
    """
    Simple wrapper to mix AudioCaps and WavCaps with a given ratio.

    - Always uses all AudioCaps samples.
    - Uses wavcaps_ratio * len(WavCaps) samples (clipped to [0, len]).
    - Optionally re-samples the WavCaps subset each epoch to expose more of a large
      dataset without making epochs enormous.

    Memory-optimized: Uses numpy arrays instead of Python lists for index mapping.
    This reduces memory from ~40MB to ~4MB for 500K samples.
    """

    def __init__(
        self,
        audiocaps_dataset: Dataset,
        wavcaps_dataset: Optional[Dataset] = None,
        wavcaps_ratio: float = 0.8,
        shuffle: bool = True,
        seed: int = 42,
        resample_wavcaps_each_epoch: bool = False,
    ) -> None:
        if audiocaps_dataset is None and wavcaps_dataset is None:
            raise ValueError("MixedAudioCaptionDataset requires at least one dataset")

        self.audiocaps_dataset = audiocaps_dataset
        self.wavcaps_dataset = wavcaps_dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.resample_wavcaps_each_epoch = bool(resample_wavcaps_each_epoch)

        self._wavcaps_ratio = max(0.0, min(1.0, float(wavcaps_ratio)))
        self._dataset_ids = np.zeros(0, dtype=np.uint8)
        self._local_indices = np.zeros(0, dtype=np.int32)
        self._build_index(epoch=0)

    def _build_index(self, *, epoch: int) -> None:
        a_count = len(self.audiocaps_dataset) if self.audiocaps_dataset is not None else 0
        if self.wavcaps_dataset is None:
            # Only AudioCaps - use simple range
            self._dataset_ids = np.zeros(a_count, dtype=np.uint8)  # 0 = audiocaps
            self._local_indices = np.arange(a_count, dtype=np.int32)
            return

        w_count = len(self.wavcaps_dataset)
        w_samples = int(w_count * self._wavcaps_ratio)
        w_samples = max(0, min(w_samples, w_count))

        rng = np.random.Generator(np.random.PCG64(self.seed + int(epoch)))
        if w_samples <= 0:
            wavcaps_indices = np.zeros(0, dtype=np.int32)
        elif w_samples >= w_count:
            wavcaps_indices = np.arange(w_count, dtype=np.int32)
        else:
            wavcaps_indices = rng.choice(w_count, size=w_samples, replace=False).astype(np.int32)

        dataset_ids = np.concatenate([
            np.zeros(a_count, dtype=np.uint8),
            np.ones(w_samples, dtype=np.uint8),
        ])
        local_indices = np.concatenate([
            np.arange(a_count, dtype=np.int32),
            wavcaps_indices,
        ])

        if self.shuffle:
            perm = rng.permutation(len(dataset_ids))
            dataset_ids = dataset_ids[perm]
            local_indices = local_indices[perm]

        self._dataset_ids = dataset_ids
        self._local_indices = local_indices

    def set_epoch(self, epoch: int) -> None:
        """Optionally reshuffle/re-sample the WavCaps subset each epoch."""
        if not self.resample_wavcaps_each_epoch:
            return
        self._build_index(epoch=int(epoch))

    def __len__(self) -> int:
        return len(self._dataset_ids)

    def __getitem__(self, idx: int):
        dataset_id = self._dataset_ids[idx]
        local_idx = int(self._local_indices[idx])

        if dataset_id == 0:
            return self.audiocaps_dataset[local_idx]
        else:
            return self.wavcaps_dataset[local_idx]


def _extract_answer_from_generation(generated_text: str) -> str:
    """
    Extract answer text from a chat-style generation.

    Mirrors StageATrainer._extract_answer(..., mode="audio"):
    - Prefer content after an "ASSISTANT:" marker.
    - Strip assistant-style prefixes.
    - Keep full caption (no sentence truncation).
    """
    if not generated_text:
        return ""

    import re

    answer = str(generated_text).strip()
    answer = re.sub(r"[\r\n]+", " ", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    if not answer:
        return ""

    lower_answer = answer.lower()

    # Extract text after ASSISTANT: marker if present
    assistant_match = re.search(r"(?:assistant|ssistant)\s*[:\-]\s*(.+)", lower_answer, re.IGNORECASE)
    if assistant_match:
        match_in_original = re.search(r"(?:assistant|ssistant)\s*[:\-]\s*(.+)", answer, re.IGNORECASE)
        if match_in_original:
            answer = match_in_original.group(1).strip()
            lower_answer = answer.lower()

    # Remove obvious assistant-style prefixes
    prefix_patterns = [
        r"^assistant\s*[:\-]\s*",
        r"^ssistant\s*[:\-]\s*",
        r"^ans(?:wer)?\s*[:\-]\s*",
        r"^ant\s*[:\-]\s*",
        r"^response\s*[:\-]\s*",
        r"^reply\s*[:\-]\s*",
    ]
    for pattern in prefix_patterns:
        if re.match(pattern, lower_answer):
            answer = re.sub(pattern, "", answer, count=1, flags=re.IGNORECASE).strip()
            lower_answer = answer.lower()
            break

    # Generic "prefix: value" handling
    if ":" in answer:
        prefix, remainder = answer.split(":", 1)
        prefix_clean = prefix.strip().lower()
        if (
            prefix_clean
            and len(prefix_clean.split()) <= 3
            and all(ch.isalpha() for ch in prefix_clean.replace(" ", ""))
        ):
            answer = remainder.strip()
            lower_answer = answer.lower()

    # Remove leading bullets / numbering
    answer = re.sub(r"^(?:[\-\*\u2022]+|\d+\.)\s*", "", answer)

    return answer.strip()


def _strip_generation_artifacts(text: str) -> str:
    """
    Remove common boilerplate artifacts that occasionally appear in generations and
    poison caption metrics (e.g., translation templates like '번역결과').
    """
    if not text:
        return text

    cleaned = str(text).strip()

    # Korean "번역결과" ("translation result") template sometimes appears after a newline.
    for marker in ("번역결과", "번역 결과", "Translation result", "translation result"):
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx].strip()
            break

    # Collapse whitespace/newlines after stripping.
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def _normalize_references(answer: Any) -> List[str]:
    """
    Normalize dataset answer payloads into a list of non-empty reference strings.

    Datasets may store captions as:
    - str
    - list[str]
    - list[dict] with keys like "answer"/"caption"
    - dict with "answers"/"captions"
    """
    refs: List[str] = []
    if answer is None:
        return refs

    if isinstance(answer, str):
        s = answer.strip()
        return [s] if s else []

    if isinstance(answer, dict):
        candidate = answer.get("answers") or answer.get("captions") or answer.get("answer") or answer.get("caption")
        return _normalize_references(candidate)

    if isinstance(answer, (list, tuple)):
        for item in answer:
            if item is None:
                continue
            if isinstance(item, str):
                s = item.strip()
                if s:
                    refs.append(s)
                continue
            if isinstance(item, dict):
                candidate = item.get("answer") or item.get("caption") or item.get("text")
                if candidate is None:
                    # fall back to stringifying dict if it has nothing useful
                    continue
                refs.extend(_normalize_references(candidate))
                continue
            s = str(item).strip()
            if s:
                refs.append(s)
        # de-dup preserve order
        seen = set()
        refs = [r for r in refs if not (r in seen or seen.add(r))]
        return refs

    s = str(answer).strip()
    return [s] if s else []

# ============================================================================
# SECTION 2: METRICS
# ============================================================================

def compute_cider(predictions: List[str], references: List[List[str]]) -> float:
    """
    Compute CIDEr score using pycocoevalcap

    Args:
        predictions: List of predicted captions
        references: List of reference caption lists (multiple refs per sample)

    Returns:
        CIDEr score (0-100 scale)
    """
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        print("⚠️  pycocoevalcap not available, CIDEr score will be 0.0")
        return 0.0

    # Convert to pycocoevalcap format: {id: [captions]}
    gts = {i: refs for i, refs in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}

    try:
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts, res)
        return float(score) * 100  # Scale to 0-100
    except Exception as e:
        print(f"⚠️  CIDEr computation failed: {e}")
        return 0.0


def compute_bleu(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Compute BLEU scores using pycocoevalcap

    Returns:
        Dict with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
    except ImportError:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    gts = {i: refs for i, refs in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}

    try:
        bleu_scorer = Bleu(4)
        scores, _ = bleu_scorer.compute_score(gts, res)
        return {
            "bleu1": float(scores[0]),
            "bleu2": float(scores[1]),
            "bleu3": float(scores[2]),
            "bleu4": float(scores[3]),
        }
    except Exception:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}


def compute_meteor(predictions: List[str], references: List[List[str]]) -> float:
    """Compute METEOR score"""
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        meteor_scorer = Meteor()
        score, _ = meteor_scorer.compute_score(gts, res)
        return float(score)
    except Exception:
        return 0.0


def compute_rouge(predictions: List[str], references: List[List[str]]) -> float:
    """Compute ROUGE-L score"""
    try:
        from pycocoevalcap.rouge.rouge import Rouge
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        rouge_scorer = Rouge()
        score, _ = rouge_scorer.compute_score(gts, res)
        return float(score)
    except Exception:
        return 0.0


def compute_caption_metrics(
    predictions: List[str],
    references: List[List[str]],
    compute_bertscore: bool = False,
    light_metrics: bool = False,
    quiet: bool = False,
) -> Dict[str, float]:
    """
    Compute caption metrics, mirroring StageATrainer's metric stack as closely
    as possible while keeping training-time evaluation lightweight when
    `light_metrics=True`.
    """
    # Default structure so callers can always rely on these keys
    metrics: Dict[str, float] = {
        "cider": 0.0,
        "spice": 0.0,
        "spider": 0.0,
        "bleu1": 0.0,
        "bleu2": 0.0,
        "bleu3": 0.0,
        "bleu4": 0.0,
        "meteor": 0.0,
        "rouge_l": 0.0,
        "bertscore_f1": 0.0,
    }

    if not predictions or not references:
        return metrics

    # Normalize and filter empty pairs (match StageATrainer behaviour)
    paired: List[Tuple[str, List[str]]] = []
    for pred, refs in zip(predictions, references):
        pred_clean = _normalize_audio_caption(str(pred).strip())
        refs_clean = [
            _normalize_audio_caption(str(ref).strip())
            for ref in refs
            if str(ref).strip()
        ]
        pred_clean = pred_clean.strip()
        refs_clean = [r for r in refs_clean if r.strip()]
        if pred_clean and refs_clean:
            paired.append((pred_clean, refs_clean))

    if not paired:
        return metrics

    preds_list, refs_list = zip(*paired)
    preds_list = list(preds_list)
    refs_list = [list(r) for r in refs_list]

    # Reference statistics (useful sanity check; mirrors StageATrainer logs)
    ref_counts = [len(r) for r in refs_list]
    if ref_counts and not quiet:
        avg_refs = sum(ref_counts) / len(ref_counts)
        min_refs = min(ref_counts)
        max_refs = max(ref_counts)
        print(
            f"[RefValidation] References per sample: avg={avg_refs:.1f}, "
            f"min={min_refs}, max={max_refs}, total_samples={len(refs_list)}",
            flush=True,
        )
        if avg_refs < 2.0:
            print(
                f"⚠️  WARNING: Low reference count (avg={avg_refs:.1f}). "
                f"AudioCaps-style CIDEr expects ~5 refs/sample.",
                flush=True,
            )

    # HuggingFace evaluate metrics (BLEU/METEOR/ROUGE and optional BERTScore)
    try:
        import evaluate

        def _metric(name: str, **load_kwargs):
            key = (name, tuple(sorted(load_kwargs.items())))
            if key not in _EVALUATE_METRIC_CACHE:
                _EVALUATE_METRIC_CACHE[key] = evaluate.load(name, **load_kwargs)
            return _EVALUATE_METRIC_CACHE[key]

        # BLEU - use pycocoevalcap for consistency with CIDEr (standard for captioning)
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            gts_bleu = {str(i): refs for i, refs in enumerate(refs_list)}
            res_bleu = {str(i): [pred] for i, pred in enumerate(preds_list)}
            bleu_scorer = Bleu(4)
            bleu_scores, _ = bleu_scorer.compute_score(gts_bleu, res_bleu)
            metrics["bleu1"] = float(bleu_scores[0])
            metrics["bleu2"] = float(bleu_scores[1])
            metrics["bleu3"] = float(bleu_scores[2])
            metrics["bleu4"] = float(bleu_scores[3])
        except Exception as exc:
            print(f"⚠️  BLEU metric failed: {exc}", flush=True)

        # METEOR - use pycocoevalcap for consistency with CIDEr/BLEU (standard for captioning)
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            gts_meteor = {str(i): refs for i, refs in enumerate(refs_list)}
            res_meteor = {str(i): [pred] for i, pred in enumerate(preds_list)}
            meteor_scorer = Meteor()
            meteor_score, _ = meteor_scorer.compute_score(gts_meteor, res_meteor)
            metrics["meteor"] = float(meteor_score)
        except Exception as exc:
            print(f"⚠️  METEOR metric failed: {exc}", flush=True)

        # ROUGE-L (best over references per sample, always computed if available)
        try:
            rouge_metric = _metric("rouge")
            rouge_scores: List[float] = []
            for pred, refs in zip(preds_list, refs_list):
                best = 0.0
                for ref in refs:
                    try:
                        result = rouge_metric.compute(predictions=[pred], references=[ref])
                        best = max(best, float(result.get("rougeL", 0.0)))
                    except Exception as rouge_exc:
                        print(f"⚠️  ROUGE-L metric failed on sample: {rouge_exc}", flush=True)
                rouge_scores.append(best)
            if rouge_scores:
                metrics["rouge_l"] = float(sum(rouge_scores) / len(rouge_scores))
        except Exception as exc:
            print(f"⚠️  ROUGE-L metric failed: {exc}", flush=True)

        # Optional BERTScore via evaluate (only in heavy eval mode)
        if compute_bertscore and not light_metrics:
            try:
                bert_metric = _metric("bertscore")
                bert_result = bert_metric.compute(
                    predictions=preds_list,
                    references=[r[0] for r in refs_list],
                    lang="en",
                )
                if "f1" in bert_result:
                    f1_scores = bert_result["f1"]
                    if isinstance(f1_scores, (list, tuple)) and len(f1_scores) > 0:
                        metrics["bertscore_f1"] = float(sum(f1_scores) / len(f1_scores))
            except Exception as exc:
                print(f"⚠️  BERTScore metric failed: {exc}", flush=True)

    except Exception as exc:
        # If evaluate is unavailable, we still compute CIDEr/SPICE below
        print(
            f"⚠️  evaluate library unavailable for BLEU/METEOR/ROUGE/BERTScore: {exc}",
            flush=True,
        )

    # CIDEr + SPICE (pycocoevalcap) – match StageATrainer behaviour
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice

        gts = {str(i): refs for i, refs in enumerate(refs_list)}
        res = {str(i): [pred] for i, pred in enumerate(preds_list)}

        try:
            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(gts, res)
            metrics["cider"] = float(cider_score) * 100.0
        except Exception as exc:
            print(f"⚠️  CIDEr metric failed: {exc}", flush=True)

        # SPICE is heavy (Java CoreNLP); only compute when light_metrics=False
        if not light_metrics:
            try:
                spice_scorer = Spice()
                spice_score, _ = spice_scorer.compute_score(gts, res)
                metrics["spice"] = float(spice_score) * 100.0
            except Exception as exc:
                print(f"⚠️  SPICE metric failed: {exc}", flush=True)

        if metrics["cider"] > 0.0 and metrics["spice"] > 0.0:
            metrics["spider"] = (metrics["cider"] + metrics["spice"]) / 2.0

    except ImportError as exc:
        print(f"⚠️  pycocoevalcap unavailable for CIDEr/SPICE: {exc}", flush=True)

    return metrics


# ============================================================================
# SECTION 3: EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate(
    model: SAFEModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    max_new_tokens: int = 20,
    num_beams: int = 1,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3,
    compute_bertscore: bool = False,
    light_metrics: bool = False,
    suppress_eos_for_audio: bool = True,
    eval_prompt: Optional[str] = None,
    ablate_audio: bool = False,
    sample_output: Optional[List[Dict[str, Any]]] = None,
    sample_limit: int = 0,
) -> Dict[str, float]:
    """
    Evaluate model on audio captioning task

    Args:
        model: SAFE model
        dataloader: Validation dataloader
        device: Device to run on
        max_batches: Maximum batches to evaluate (None = all)
        max_new_tokens: Max tokens to generate
        num_beams: Beam search size
        compute_bertscore: Whether to compute BERTScore

    Returns:
        Dict with metrics (loss, cider, bleu4, etc.)
    """
    model.eval()

    # Handle DDP wrapper - get underlying model for attribute access
    base_model = model.module if hasattr(model, 'module') else model
    is_main = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

    # Ensure audio fusion is fully enabled during evaluation
    if hasattr(base_model, "set_gate"):
        try:
            base_model.set_gate(1.0)
        except Exception:
            pass

    # CRITICAL: Configure generation parameters to prevent hanging
    tokenizer = base_model.base_vl.tokenizer

    # Ensure pad_token exists
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    # Set generation config on the LLM to prevent conflicts
    if hasattr(base_model.base_vl.llm, 'config'):
        base_model.base_vl.llm.config.pad_token_id = tokenizer.pad_token_id
        base_model.base_vl.llm.config.eos_token_id = tokenizer.eos_token_id

    if hasattr(base_model.base_vl.llm, 'generation_config'):
        base_model.base_vl.llm.generation_config.pad_token_id = tokenizer.pad_token_id
        base_model.base_vl.llm.generation_config.eos_token_id = tokenizer.eos_token_id
        # Override max_length to respect max_new_tokens limit
        base_model.base_vl.llm.generation_config.max_length = None
        # KV augmentation does not support KV caching; force cache off during evaluation generation.
        if getattr(base_model, "enable_kv_augmentation", False):
            base_model.base_vl.llm.generation_config.use_cache = False

    total_loss = 0.0
    num_batches = 0
    token_correct = 0
    token_total = 0

    all_predictions = []
    all_references = []

    # One-time tokenizer verification (helps diagnose decoding bugs)
    print(f"[TokenizerCheck] name={type(tokenizer).__name__}, vocab_size={len(tokenizer)}, "
          f"pad_id={tokenizer.pad_token_id}, eos_id={tokenizer.eos_token_id}, "
          f"bos_id={getattr(tokenizer, 'bos_token_id', 'N/A')}", flush=True)

    print(f"Running evaluation (max_batches={max_batches})...", flush=True)
    start_time = time.time()
    skipped_batches = 0
    last_skip_log_time = start_time
    printed_eval_config = False

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Move batch to device
        questions = batch["questions"]
        answers = batch["answers"]
        audio = batch["audio"]
        has_audio_flags = batch.get("has_audio", None)
        sample_ids = batch.get("sample_ids", None)
        audio_paths = batch.get("audio_paths", None)
        subsets = batch.get("subsets", None)
        if (
            is_main
            and not printed_eval_config
            and batch_idx == 0
        ):
            # Print a single eval-config line to catch "step X flips a switch" issues.
            min_attn = None
            min_attn_w = None
            try:
                loss_obj = getattr(base_model, "min_audio_attention_loss", None)
                if loss_obj is not None and hasattr(loss_obj, "get_current_params"):
                    params = loss_obj.get_current_params()
                    min_attn = params.get("min_audio_attention")
                    min_attn_w = params.get("min_audio_attention_weight")
            except Exception:
                pass
            eval_prompt_preview = (eval_prompt.strip() if isinstance(eval_prompt, str) else None)
            eos_suppression_will_apply = bool(suppress_eos_for_audio and tokenizer.eos_token_id is not None)
            print(
                "[EvalConfig] "
                f"ablate_audio={bool(ablate_audio)} "
                f"max_new_tokens={max_new_tokens} num_beams={num_beams} "
                f"suppress_eos_for_audio={bool(suppress_eos_for_audio)} "
                f"eos_suppression_will_apply={bool(eos_suppression_will_apply)} "
                f"repetition_penalty={float(repetition_penalty)} no_repeat_ngram_size={int(no_repeat_ngram_size)} "
                f"eval_prompt={repr(eval_prompt_preview) if eval_prompt_preview else '(dataset)'} "
                f"min_audio_attn={min_attn} min_audio_attn_weight={min_attn_w}",
                flush=True,
            )
            if isinstance(sample_ids, list) and sample_ids:
                print(f"[EvalSet] sample_ids[:5]={sample_ids[:5]}", flush=True)
            printed_eval_config = True

        # Filter out samples with missing audio OR missing caption references.
        # - Missing audio can cause generation to hang (zero-filled audio tokens).
        # - Missing captions produce meaningless loss/metrics and can hide training failure.
        valid_indices = [
            i
            for i, (a, ans) in enumerate(zip(audio, answers))
            if a is not None and _is_valid_caption_reference(ans)
        ]

        if not valid_indices:
            skipped_batches += 1
            missing_audio = sum(1 for a in audio if a is None)
            missing_caps = sum(1 for ans in answers if not _is_valid_caption_reference(ans))
            now = time.time()
            if batch_idx < 5 or (now - last_skip_log_time) > 60:
                print(
                    f"  ⚠️  Skipping batch {batch_idx} - "
                    f"missing_audio={missing_audio}/{len(audio)} missing_captions={missing_caps}/{len(answers)} "
                    f"(skipped_batches={skipped_batches})",
                    flush=True,
                )
                last_skip_log_time = now
            continue

        if len(valid_indices) < len(audio):
            if batch_idx < 5:
                missing_audio = sum(1 for a in audio if a is None)
                missing_caps = sum(1 for ans in answers if not _is_valid_caption_reference(ans))
                print(
                    f"  ⚠️  Filtering batch {batch_idx} - "
                    f"kept={len(valid_indices)}/{len(audio)} "
                    f"missing_audio={missing_audio} missing_captions={missing_caps}",
                    flush=True,
                )
            questions = [questions[i] for i in valid_indices]
            answers = [answers[i] for i in valid_indices]
            audio = [audio[i] for i in valid_indices]
            if isinstance(sample_ids, list):
                sample_ids = [sample_ids[i] for i in valid_indices]
            if isinstance(audio_paths, list):
                audio_paths = [audio_paths[i] for i in valid_indices]
            if isinstance(subsets, list):
                subsets = [subsets[i] for i in valid_indices]

        audio_for_model = None if ablate_audio else audio

        # Prepare inputs (ensure correct device)
        inputs = base_model.prepare_multimodal_inputs(
            text=questions,
            audio=audio_for_model,
            answers=answers,
            device=device,
            training_mode=True,  # For loss computation
        )

        # Move inputs to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs["labels"].to(device)
        audio_tokens = inputs.get("audio_tokens")
        if audio_tokens is not None:
            audio_tokens = audio_tokens.to(device)
        audio_attention_mask = inputs.get("audio_attention_mask")
        if audio_attention_mask is not None:
            audio_attention_mask = audio_attention_mask.to(device)

        # Compute loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            audio_tokens=audio_tokens,
            audio_attention_mask=audio_attention_mask,
        )

        loss = outputs.get("loss")
        if loss is not None:
            total_loss += loss.item()
            num_batches += 1

        logits = outputs.get("logits")
        if isinstance(logits, torch.Tensor) and logits.ndim >= 3 and isinstance(labels, torch.Tensor) and labels.ndim >= 2:
            try:
                with torch.no_grad():
                    shift_logits = logits[..., :-1, :]
                    shift_labels = labels[..., 1:]
                    preds = shift_logits.argmax(dim=-1)
                    mask = shift_labels != -100
                    if mask.any():
                        token_correct += int((preds[mask] == shift_labels[mask]).sum().item())
                        token_total += int(mask.sum().item())
            except Exception:
                pass

        # Generate predictions (reuse same device).
        # Use a non-chatty evaluation prompt (if provided) to reduce refusal-template prior.
        gen_questions = questions
        if isinstance(eval_prompt, str) and eval_prompt.strip():
            gen_questions = [eval_prompt.strip()] * len(questions)
        generation_inputs = base_model.prepare_multimodal_inputs(
            text=gen_questions,
            audio=audio_for_model,
            answers=None,  # No answers for generation
            device=device,
            training_mode=False,
            llava_audio_prompt_style=("plain" if (isinstance(eval_prompt, str) and eval_prompt.strip()) else "question"),
        )

        gen_input_ids = generation_inputs["input_ids"].to(device)
        gen_attention_mask = generation_inputs["attention_mask"].to(device)
        gen_audio_tokens = generation_inputs.get("audio_tokens")
        if gen_audio_tokens is not None:
            gen_audio_tokens = gen_audio_tokens.to(device)
        gen_audio_attention_mask = generation_inputs.get("audio_attention_mask")
        if gen_audio_attention_mask is not None:
            gen_audio_attention_mask = gen_audio_attention_mask.to(device)

        # Build generation kwargs.
        # NOTE: Do NOT suppress EOS during evaluation; it can force rambling generations
        # and makes metrics/qualitative samples much harder to interpret.
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 1,
            "num_beams": num_beams,
            "repetition_penalty": float(repetition_penalty),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        # Optional EOS suppression (disabled by default; only enable explicitly).
        if suppress_eos_for_audio and tokenizer.eos_token_id is not None:
            suppress_tokens = [tokenizer.eos_token_id]
            if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
                suppress_tokens.append(tokenizer.pad_token_id)
            generation_kwargs["suppress_tokens"] = suppress_tokens

        # Generate captions (use base_model for generate method)
        generated_ids = base_model.generate(
            input_ids=gen_input_ids,
            attention_mask=gen_attention_mask,
            audio_tokens=gen_audio_tokens,
            audio_attention_mask=gen_audio_attention_mask,
            **generation_kwargs,
        )

        # Decode predictions.
        # HF generate may return either full sequence (prompt + new) or only new tokens.
        prompt_len = int(gen_input_ids.shape[1])  # includes left padding
        if generated_ids.dim() == 2 and generated_ids.size(1) > prompt_len:
            decoded_ids = generated_ids[:, prompt_len:]
        else:
            decoded_ids = generated_ids

        # Debug: Print token info for first batch to diagnose decoding issues
        if batch_idx == 0:
            print(
                f"[DecodeDebug] prompt_len={prompt_len} input_ids shape: {gen_input_ids.shape}, "
                f"generated_ids shape: {generated_ids.shape}, decoded_ids shape: {decoded_ids.shape}",
                flush=True,
            )
            if isinstance(eval_prompt, str) and eval_prompt.strip():
                print(f"[EvalPrompt] Using eval_prompt={repr(eval_prompt.strip())} (llava_audio_prompt_style=plain)", flush=True)
            preview_ids = decoded_ids[0, :30].tolist() if decoded_ids.dim() == 2 else []
            print(f"[DecodeDebug] First 30 decoded token IDs: {preview_ids}", flush=True)
            raw_decode = tokenizer.decode(decoded_ids[0, :30], skip_special_tokens=False)
            print(f"[DecodeDebug] Raw decode (first 30 decoded tokens): {repr(raw_decode)}", flush=True)
        batch_predictions = tokenizer.batch_decode(
            decoded_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Clean predictions (remove question prompt and extract answer content) + collect references
        for i, pred in enumerate(batch_predictions):
            question = gen_questions[i]
            # Debug: show cleaning steps for first sample of first batch
            if batch_idx == 0 and i == 0:
                print(f"[CleanDebug] Raw batch_decode: {repr(pred[:200])}", flush=True)
            if question and question in pred:
                pred = pred.replace(question, "").strip()
                if batch_idx == 0 and i == 0:
                    print(f"[CleanDebug] After question removal: {repr(pred[:200])}", flush=True)
            pred_answer = _extract_answer_from_generation(pred)
            cleaned_pred = pred_answer if pred_answer else pred.strip()
            cleaned_pred = _strip_generation_artifacts(cleaned_pred)
            if batch_idx == 0 and i == 0:
                print(f"[CleanDebug] After _extract_answer: {repr(pred_answer[:200] if pred_answer else 'EMPTY')}", flush=True)
                print(f"[CleanDebug] Final cleaned_pred: {repr(cleaned_pred[:200])}", flush=True)
            all_predictions.append(cleaned_pred)

            refs = _normalize_references(answers[i])
            all_references.append(refs)

            if sample_output is not None and sample_limit > 0 and len(sample_output) < sample_limit:
                audio_item = None
                audio_path_item = None
                subset_item = None
                sample_id_item = None
                try:
                    audio_item = audio[i]
                except Exception:
                    audio_item = None
                if isinstance(audio_paths, list):
                    audio_path_item = audio_paths[i]
                if isinstance(subsets, list):
                    subset_item = subsets[i]
                if isinstance(sample_ids, list):
                    sample_id_item = sample_ids[i]
                sample_output.append(
                    {
                        "sample_id": sample_id_item,
                        "question": str(question),
                        "prediction": str(cleaned_pred),
                        "references": [str(r) for r in refs],
                        "audio_path": audio_path_item,
                        "subset": subset_item,
                        "audio": audio_item,
                    }
                )

        if (batch_idx + 1) % 10 == 0:
            print(f"  Evaluated {batch_idx + 1} batches...", flush=True)

    elapsed = time.time() - start_time

    # Compute metrics
    print(f"[Metrics] Computing caption metrics on {len(all_predictions)} predictions...", flush=True)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    caption_metrics = compute_caption_metrics(
        all_predictions,
        all_references,
        compute_bertscore=compute_bertscore,
        light_metrics=light_metrics,
    )

    metrics = {
        "loss": avg_loss,
        **caption_metrics,
        "num_samples": len(all_predictions),
        "eval_time": elapsed,
    }
    if token_total > 0:
        metrics["token_accuracy"] = float(token_correct / max(token_total, 1))

    print(f"[Metrics] Caption metrics computed.", flush=True)
    print(f"✓ Evaluation complete ({format_time(elapsed)})", flush=True)
    print(f"  Loss: {avg_loss:.4f}", flush=True)
    # Print all key caption metrics in a consistent order
    metric_keys = [
        "bleu1",
        "bleu2",
        "bleu3",
        "bleu4",
        "meteor",
        "rouge_l",
        "cider",
        "spice",
        "spider",
        "bertscore_f1",
    ]
    for key in metric_keys:
        if key in metrics:
            value = metrics[key]
            if key in ["cider", "spice", "spider"]:
                print(f"  {key.upper()}: {value:.2f}", flush=True)
            elif key == "bertscore_f1":
                print(f"  BERTSCORE_F1: {value:.4f}", flush=True)
            else:
                print(f"  {key.upper()}: {value:.4f}", flush=True)

    # Log sample predictions
    print(f"\n📝 Sample predictions:", flush=True)
    for i in range(min(3, len(all_predictions))):
        print(f"  [{i+1}] Pred: {all_predictions[i]}", flush=True)
        print(f"      Refs: {all_references[i]}", flush=True)

    # Explicit memory cleanup to prevent OOM during long training runs
    del all_predictions, all_references
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return metrics


# ============================================================================
# SECTION 3.5: MODEL CREATION HELPER
# ============================================================================

# Whitelist of valid SAFEModel constructor arguments
SAFE_MODEL_CONSTRUCTOR_KEYS = {
    "llm_model_name",
    "vision_model_name",
    "audio_encoder_type",
    "audio_encoder_config",
    "projector_type",
    "num_audio_tokens",
    "projector_config",
    "fusion_type",
    "fusion_layer_indices",
    "lora_rank",
    "fusion_config",
    "freeze_base_vl",
    "freeze_audio_encoder",
    "label_smoothing",
    "llm_hidden_size",
    "audio_embed_dim",
}


def create_model(config: Dict[str, Any]) -> SAFEModel:
    """
    Create a SAFEModel from a config dictionary.

    This is the canonical way to create SAFE models for training and evaluation.
    It filters config keys to only those accepted by SAFEModel constructor.

    Args:
        config: Model configuration dictionary (e.g., from get_config("phase1"))

    Returns:
        Initialized SAFEModel (on CPU, not moved to device)
    """
    # Filter config to only include valid constructor arguments
    constructor_config = {k: v for k, v in config.items() if k in SAFE_MODEL_CONSTRUCTOR_KEYS}

    print(f"[create_model] Initializing SAFE model...")
    print(f"  LLM: {constructor_config.get('llm_model_name', 'N/A')}")
    print(f"  Vision: {constructor_config.get('vision_model_name', 'N/A')}")
    print(f"  Audio: {constructor_config.get('audio_encoder_type', 'N/A')}")
    print(f"  Fusion type: {constructor_config.get('fusion_type', 'N/A')}")
    print(f"  Fusion layers: {constructor_config.get('fusion_layer_indices', 'N/A')}")
    print(f"  LoRA rank: {constructor_config.get('lora_rank', 'N/A')}")

    # Log KV augmentation specific config if present
    fusion_config = constructor_config.get('fusion_config', {})
    if fusion_config.get('fusion_mode') == 'kv_augment':
        print(f"  [KV Augment Mode]")
        print(f"    Query adapter rank: {fusion_config.get('query_adapter_rank', 'N/A')}")
        print(f"    Bottleneck dim: {fusion_config.get('bottleneck_dim', 'N/A')}")
        print(f"    Min audio attention: {fusion_config.get('min_audio_attention', 0.0)}")

    model = SAFEModel(**constructor_config)
    return model


# ============================================================================
# SECTION 4: TRAINING
# ============================================================================

def _extract_audio_projector_output_scale(model: Any) -> Optional[float]:
    projector = getattr(model, "audio_projector", None)
    if projector is None:
        return None
    scale = getattr(projector, "output_scale", None)
    if scale is None:
        return None
    try:
        if torch.is_tensor(scale):
            return float(scale.detach().cpu().float().item())
        return float(scale)
    except Exception:
        return None


def _extract_fusion_residual_scales(model: Any) -> Dict[str, float]:
    """
    Return a dict of residual_scale values for fusion cross-attention blocks.
    Supports MultiLayerFusionAdapter (ModuleDict) and single adapters.
    """
    fusion_adapter = getattr(model, "fusion_adapter", None)
    if fusion_adapter is None:
        return {}

    # Unwrap gated adapter -> inner LoRAFusionAdapter
    if hasattr(fusion_adapter, "fusion_adapter") and not hasattr(fusion_adapter, "fusion_adapters"):
        inner = getattr(fusion_adapter, "fusion_adapter", None)
        if inner is not None:
            fusion_adapter = inner

    adapters: List[Tuple[str, Any]] = []
    if hasattr(fusion_adapter, "fusion_adapters"):
        try:
            items = list(getattr(fusion_adapter, "fusion_adapters").items())
            adapters.extend([(str(k), v) for k, v in items])
        except Exception:
            pass
    else:
        adapters.append(("fusion", fusion_adapter))

    residuals: Dict[str, float] = {}
    for key, adapter in adapters:
        cross_attention = getattr(adapter, "cross_attention", None)
        if cross_attention is not None:
            candidate = getattr(cross_attention, "base_model", None) or cross_attention
        else:
            candidate = adapter

        residual_param = getattr(candidate, "residual_scale", None)
        if residual_param is None:
            continue

        cap = getattr(candidate, "residual_scale_max", None)
        try:
            if cap is not None:
                value = float(torch.clamp(residual_param, 0.0, float(cap)).detach().cpu().float().item())
            else:
                value = float(torch.clamp(residual_param, 0.0, 5.0).detach().cpu().float().item())
        except Exception:
            continue

        safe_key = str(key).replace("/", "_").replace(":", "_")
        residuals[safe_key] = value

    return residuals


def _extract_kv_adapter_metrics(model: Any) -> Dict[str, float]:
    """
    Extract KV adapter metrics (scales, ΔQ/Q ratio) for logging.
    Returns empty dict if model doesn't use KV augmentation.
    """
    kv_adapters = getattr(model, "kv_adapters", None)
    if kv_adapters is None:
        return {}

    metrics: Dict[str, float] = {}

    for name, adapter in kv_adapters.named_modules():
        # Extract audio scale
        if hasattr(adapter, "audio_scale"):
            try:
                scale = adapter.audio_scale
                if torch.is_tensor(scale):
                    val = float(scale.detach().cpu().float().item())
                else:
                    val = float(scale)
                safe_name = name.replace(".", "_") if name else "root"
                metrics[f"audio_scale/{safe_name}"] = val
            except Exception:
                pass

        # Extract query adapter scale (ΔQ scale)
        if hasattr(adapter, "audio_query_adapter"):
            query_adapter = adapter.audio_query_adapter
            if hasattr(query_adapter, "scale"):
                try:
                    scale = query_adapter.scale
                    if torch.is_tensor(scale):
                        val = float(scale.detach().cpu().float().item())
                    else:
                        val = float(scale)
                    safe_name = name.replace(".", "_") if name else "root"
                    metrics[f"delta_q_scale/{safe_name}"] = val
                except Exception:
                    pass

    return metrics


def _log_comprehensive_diagnostics(
    model: Any,
    base_model: Any,
    optimizer_step: int,
    epoch: int,
    loss: float,
    grad_norm: Optional[float],
    gate_value: Optional[float] = None,
    wandb_run: Any = None,
    console_log: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive training diagnostics for debugging gradient flow and learning.

    Logs:
    - Gradient norms per component (projector, fusion, KV adapters)
    - Audio projector output scale
    - Fusion residual scales (per layer)
    - KV adapter metrics (audio_scale, delta_q_scale)
    - KV hook manager diagnostics (rms_ratio, entropy, attention patterns)
    - Gate value (if warmup active)
    - Parameter statistics (detect collapse)

    Returns dict of all metrics for optional W&B logging.
    """
    diagnostics: Dict[str, Any] = {
        "diag/optimizer_step": optimizer_step,
        "diag/epoch": epoch,
        "diag/loss": loss,
    }

    if grad_norm is not None:
        diagnostics["diag/grad_norm_clipped"] = float(grad_norm)

    if gate_value is not None:
        diagnostics["diag/gate_value"] = float(gate_value)

    # 1. Gradient norms per component
    grad_norms = {"projector": 0.0, "fusion": 0.0, "kv_adapter": 0.0, "other": 0.0}
    grad_counts = {"projector": 0, "fusion": 0, "kv_adapter": 0, "other": 0}
    param_stats = {"projector": [], "fusion": [], "kv_adapter": []}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Gradient norms
        if param.grad is not None:
            g_norm = param.grad.norm().item()
            if "audio_projector" in name:
                grad_norms["projector"] += g_norm
                grad_counts["projector"] += 1
            elif "fusion_adapter" in name:
                grad_norms["fusion"] += g_norm
                grad_counts["fusion"] += 1
            elif "kv_adapter" in name or "kv_augmentation" in name:
                grad_norms["kv_adapter"] += g_norm
                grad_counts["kv_adapter"] += 1
            else:
                grad_norms["other"] += g_norm
                grad_counts["other"] += 1

        # Parameter value stats (detect collapse)
        with torch.no_grad():
            p_val = param.abs().mean().item()
            if "audio_projector" in name:
                param_stats["projector"].append(p_val)
            elif "fusion_adapter" in name:
                param_stats["fusion"].append(p_val)
            elif "kv_adapter" in name:
                param_stats["kv_adapter"].append(p_val)

    for key in grad_norms:
        diagnostics[f"diag/grad_norm/{key}"] = grad_norms[key]
        diagnostics[f"diag/grad_count/{key}"] = grad_counts[key]

    for key in param_stats:
        if param_stats[key]:
            diagnostics[f"diag/param_mean/{key}"] = sum(param_stats[key]) / len(param_stats[key])

    # 2. Audio projector output scale
    proj_scale = _extract_audio_projector_output_scale(base_model)
    if proj_scale is not None:
        diagnostics["diag/projector_output_scale"] = proj_scale

    # 3. Fusion residual scales
    residual_scales = _extract_fusion_residual_scales(base_model)
    if residual_scales:
        values = list(residual_scales.values())
        diagnostics["diag/fusion_res_scale_mean"] = sum(values) / len(values)
        diagnostics["diag/fusion_res_scale_min"] = min(values)
        diagnostics["diag/fusion_res_scale_max"] = max(values)
        for k, v in residual_scales.items():
            diagnostics[f"diag/fusion_res_scale/{k}"] = v

    # 4. KV adapter metrics (scales)
    kv_metrics = _extract_kv_adapter_metrics(base_model)
    for k, v in kv_metrics.items():
        diagnostics[f"diag/kv/{k}"] = v

    # 5. KV hook manager diagnostics (attention patterns, RMS ratios)
    kv_hook_manager = getattr(base_model, "kv_hook_manager", None)
    if kv_hook_manager is not None and hasattr(kv_hook_manager, "log_diagnostics"):
        try:
            kv_diag = kv_hook_manager.log_diagnostics(prefix="diag/kv_attn/")
            diagnostics.update(kv_diag)
        except Exception:
            pass

    # 6. Console output (periodic summary)
    if console_log:
        lines = [
            f"\n{'='*80}",
            f"[DiagCheck] Step {optimizer_step} | Epoch {epoch} | Loss {loss:.4f}",
            f"{'='*80}",
            f"  Gradient norms:",
            f"    projector: {grad_norms['projector']:.2f} ({grad_counts['projector']} params)",
            f"    fusion:    {grad_norms['fusion']:.2f} ({grad_counts['fusion']} params)",
            f"    kv_adapter: {grad_norms['kv_adapter']:.2f} ({grad_counts['kv_adapter']} params)",
        ]

        if grad_norm is not None:
            lines.append(f"    total (clipped): {grad_norm:.2f}")

        if gate_value is not None:
            lines.append(f"  Gate value: {gate_value:.3f}")

        if proj_scale is not None:
            lines.append(f"  Projector output scale: {proj_scale:.4f}")

        if residual_scales:
            lines.append(f"  Fusion residual scales: {residual_scales}")

        if kv_metrics:
            lines.append(f"  KV adapter scales: {kv_metrics}")

        # KV attention diagnostics summary
        if kv_hook_manager is not None:
            try:
                all_kv_diag = kv_hook_manager.get_diagnostics()
                if all_kv_diag:
                    lines.append(f"  KV attention diagnostics:")
                    for layer_idx, diag in sorted(all_kv_diag.items()):
                        rms_ratio = diag.get("rms_ratio", 0)
                        entropy = diag.get("normalized_entropy", 0)
                        audio_rms = diag.get("audio_rms", 0)
                        text_rms = diag.get("text_rms", 0)
                        lines.append(
                            f"    Layer {layer_idx}: rms_ratio={rms_ratio:.4f} "
                            f"entropy={entropy:.4f} audio_rms={audio_rms:.4f} text_rms={text_rms:.4f}"
                        )
            except Exception:
                pass

        # Health indicators
        lines.append(f"  Health indicators:")

        # Check for gradient collapse
        total_grad = grad_norms["projector"] + grad_norms["fusion"] + grad_norms["kv_adapter"]
        if total_grad < 1.0:
            lines.append(f"    ⚠️  LOW GRADIENTS: total audio component grad norm = {total_grad:.4f}")
        elif total_grad < 100.0:
            lines.append(f"    ⚡ Moderate gradients: {total_grad:.2f}")
        else:
            lines.append(f"    ✓ Healthy gradients: {total_grad:.2f}")

        # Check for scale collapse
        if proj_scale is not None and proj_scale < 0.1:
            lines.append(f"    ⚠️  PROJECTOR SCALE COLLAPSE: {proj_scale:.4f}")

        if residual_scales:
            min_res = min(residual_scales.values())
            if min_res <= 0.5:
                lines.append(f"    ⚠️  RESIDUAL AT MINIMUM: {min_res:.3f} (model fighting audio)")

        lines.append(f"{'='*80}\n")
        print("\n".join(lines), flush=True)

    # 7. W&B logging
    if wandb_run is not None:
        try:
            wandb_run.log(diagnostics, step=optimizer_step)
        except Exception:
            pass

    return diagnostics


def _apply_audio_augmentation(
    audio: List[Any],
    pipeline: AudioAugmentPipeline,
    device: torch.device,
) -> List[Any]:
    """
    Apply audio augmentation to a batch of audio samples.

    Args:
        audio: List of audio samples (waveform tensors or tuples of (waveform, sr))
        pipeline: AudioAugmentPipeline instance
        device: Device to use for augmentation

    Returns:
        List of augmented audio samples in the same format as input
    """
    augmented = []
    for item in audio:
        if item is None:
            augmented.append(None)
            continue

        # Extract waveform and sample rate
        if isinstance(item, tuple) and len(item) >= 2:
            waveform, sr = item[0], item[1]
        elif isinstance(item, torch.Tensor):
            waveform = item
            sr = 48000  # Default sample rate
        elif isinstance(item, np.ndarray):
            waveform = torch.from_numpy(item).float()
            sr = 48000
        else:
            # Unknown format, pass through
            augmented.append(item)
            continue

        # Ensure tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(np.array(waveform)).float()

        # Apply augmentation
        try:
            aug_waveform = pipeline(waveform.to(device), training=True)
            aug_waveform = aug_waveform.cpu()

            # Return in same format as input
            if isinstance(item, tuple):
                augmented.append((aug_waveform, sr))
            else:
                augmented.append(aug_waveform)
        except Exception:
            # On error, use original
            augmented.append(item)

    return augmented


def train_epoch(
    model: SAFEModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    scaler: Optional[GradScaler] = None,
    wandb_run: Any = None,
    optimizer_step: int = 0,
    world_size: int = 1,
    train_eval_loader: Optional[DataLoader] = None,
    train_eval_steps: int = 0,
    audio_augment_pipeline: Optional[AudioAugmentPipeline] = None,
) -> Tuple[Dict[str, float], int]:
    """
    Train for one epoch

    Args:
        model: SAFE model
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        config: Training configuration
        scaler: GradScaler for mixed precision (optional)
        world_size: Number of GPUs for accurate throughput calculation

    Returns:
        Dict with training metrics
    """
    # Handle DDP wrapper - get underlying model for attribute access
    base_model = model.module if hasattr(model, 'module') else model
    is_main = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

    # Ensure audio components are in training mode while keeping base VL frozen
    if hasattr(base_model, "enable_audio_training"):
        base_model.enable_audio_training()
    else:
        model.train()  # Use model (not base_model) to ensure DDP hooks work

    total_loss = 0.0
    num_batches = 0
    num_samples = 0

    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    use_amp = config.get("fp16", False) and scaler is not None

    optimizer.zero_grad()

    start_time = time.time()
    last_log_time = start_time
    step_start_time = start_time
    step_loss_sum = 0.0
    step_min_attn_loss_sum = 0.0  # Track min audio attention loss for logging
    step_ablation_hinge_sum = 0.0
    step_ablation_delta_sum = 0.0
    step_samples = 0
    step_micro_batches = 0
    last_proj_grad_norm: Optional[float] = None
    last_fuse_grad_norm: Optional[float] = None
    skipped_batches = 0
    skipped_samples = 0
    filtered_samples = 0
    missing_audio_samples = 0
    missing_caption_samples = 0
    last_skip_log_time = start_time
    token_correct = 0
    token_total = 0

    _min_attn_last_logged = {"step": None}

    def _update_min_audio_attention_schedule(step: int) -> None:
        """
        Step-based curriculum for KV-augment min-audio-attention regularization.
        Disables the regularizer early to avoid dominating learning dynamics.
        """
        if not (hasattr(base_model, "min_audio_attention_loss") and base_model.min_audio_attention_loss is not None):
            return

        fusion_cfg = config.get("fusion_config", {})
        warmup = int(fusion_cfg.get("min_audio_attention_warmup_steps", 0) or 0)
        ramp = int(fusion_cfg.get("min_audio_attention_ramp_steps", 0) or 0)

        start_attn = float(fusion_cfg.get("min_audio_attention_start", 0.0) or 0.0)
        start_weight = float(fusion_cfg.get("min_audio_attention_weight_start", 0.0) or 0.0)
        target_attn = float(fusion_cfg.get("min_audio_attention", 0.0) or 0.0)
        target_weight = float(fusion_cfg.get("min_audio_attention_weight", 0.0) or 0.0)

        if warmup <= 0 and ramp <= 0:
            current_attn = target_attn
            current_weight = target_weight
        elif step < warmup:
            current_attn = start_attn
            current_weight = start_weight
        elif ramp <= 0:
            current_attn = target_attn
            current_weight = target_weight
        else:
            progress = float(step - warmup) / float(max(ramp, 1))
            progress = max(0.0, min(1.0, progress))
            current_attn = start_attn + (target_attn - start_attn) * progress
            current_weight = start_weight + (target_weight - start_weight) * progress

        base_model.min_audio_attention_loss.min_attention = current_attn
        base_model.min_audio_attention_loss.loss_weight = current_weight

        # Log phase boundaries for debugging (only once per optimizer step).
        if (
            is_main
            and step in {0, warmup, warmup + ramp}
            and _min_attn_last_logged["step"] != step
        ):
            print(
                f"[MinAudioAttnSchedule] step={step} min_attention={current_attn:.4f} weight={current_weight:.4f} "
                f"(warmup={warmup}, ramp={ramp})",
                flush=True,
            )
            _min_attn_last_logged["step"] = step

    def _ddp_noop_loss() -> torch.Tensor:
        """
        Create a zero-valued loss that still touches every trainable parameter.

        In DDP, *skipping* a backward pass on one rank (e.g., due to a bad batch)
        can cause other ranks to hang in NCCL all-reduce. This helper ensures we
        always run a backward pass and trigger DDP gradient hooks with zero grads.
        """

        # Use fp32 scalar for stability; gradients will still flow to the params' dtype.
        loss0 = torch.zeros((), device=device, dtype=torch.float32)
        for p in model.parameters():
            if getattr(p, "requires_grad", False):
                # Touch a single element per tensor to keep this O(#tensors), not O(#elements).
                loss0 = loss0 + p.view(-1)[0].float() * 0.0
        return loss0

    ddp_enabled = bool(world_size and int(world_size) > 1 and dist.is_available() and dist.is_initialized())

    for batch_idx, batch in enumerate(dataloader):
        _update_min_audio_attention_schedule(optimizer_step)
        # Move batch to device
        questions = batch["questions"]
        answers = batch["answers"]
        audio = batch["audio"]

        dummy_batch = False
        audio_tokens = None
        audio_attention_mask = None

        # Skip samples with missing audio OR missing captions.
        if isinstance(audio, list):
            valid_indices = [
                i
                for i, (a, ans) in enumerate(zip(audio, answers))
                if a is not None and _is_valid_caption_reference(ans)
            ]
            if not valid_indices:
                skipped_batches += 1
                skipped_samples += len(audio)
                missing_audio = sum(1 for a in audio if a is None)
                missing_caps = sum(1 for ans in answers if not _is_valid_caption_reference(ans))
                missing_audio_samples += missing_audio
                missing_caption_samples += missing_caps

                now = time.time()
                if batch_idx < 5 or (now - last_skip_log_time) > 60:
                    print(
                        f"  ⚠️  Skipping batch {batch_idx} - "
                        f"missing_audio={missing_audio}/{len(audio)} missing_captions={missing_caps}/{len(answers)} "
                        f"(skipped_batches={skipped_batches})",
                        flush=True,
                    )
                    last_skip_log_time = now
                # In DDP, do NOT early-continue: ranks must execute the same number
                # of backward passes to avoid NCCL hangs. Run a no-op backward.
                if ddp_enabled:
                    dummy_batch = True
                    questions = []
                else:
                    continue
            if len(valid_indices) < len(audio):
                filtered_samples += len(audio) - len(valid_indices)
                missing_audio = sum(1 for a in audio if a is None)
                missing_caps = sum(1 for ans in answers if not _is_valid_caption_reference(ans))
                missing_audio_samples += missing_audio
                missing_caption_samples += missing_caps

                if batch_idx < 5:
                    print(
                        f"  ⚠️  Filtering batch {batch_idx} - "
                        f"kept={len(valid_indices)}/{len(audio)} "
                        f"missing_audio={missing_audio} missing_captions={missing_caps}",
                        flush=True,
                    )
                questions = [questions[i] for i in valid_indices]
                answers = [answers[i] for i in valid_indices]
                audio = [audio[i] for i in valid_indices]

        if dummy_batch:
            outputs = {"loss": _ddp_noop_loss(), "logits": None}
            loss = outputs["loss"]
            input_ids = None
            attention_mask = None
            labels = None
        else:
            # Apply audio augmentation if enabled
            if audio_augment_pipeline is not None and audio:
                audio = _apply_audio_augmentation(audio, audio_augment_pipeline, device)

            # Prepare inputs (use base_model for helper method)
            inputs = base_model.prepare_multimodal_inputs(
                text=questions,
                audio=audio,
                answers=answers,
                device=device,
                training_mode=True
            )

            # Move to device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = inputs["labels"].to(device)
            audio_tokens = inputs.get("audio_tokens")
            if audio_tokens is not None:
                audio_tokens = audio_tokens.to(device)
            audio_attention_mask = inputs.get("audio_attention_mask")
            if audio_attention_mask is not None:
                audio_attention_mask = audio_attention_mask.to(device)

            # Enable attention weight capture for min_audio_attention loss (KV augment mode)
            if (hasattr(base_model, 'min_audio_attention_loss') and
                base_model.min_audio_attention_loss is not None and
                hasattr(base_model, 'kv_hook_manager') and
                base_model.kv_hook_manager is not None):
                base_model.kv_hook_manager.set_return_attention_weights(True)

            # Forward pass with optional mixed precision
            if use_amp:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        audio_tokens=audio_tokens,
                        audio_attention_mask=audio_attention_mask,
                    )
                    loss = outputs["loss"]
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    audio_tokens=audio_tokens,
                    audio_attention_mask=audio_attention_mask,
                )
                loss = outputs["loss"]

        # Optional forcing term: encourage audio to reduce loss vs ablated-audio baseline.
        # This is a "with-audio must be better" hinge, but does NOT push the no-audio loss up
        # (no_audio is computed under no_grad and detached).
        ablation_loss_weight = float(config.get("ablation_loss_weight", 0.0) or 0.0)
        ablation_loss_margin = float(config.get("ablation_loss_margin", 0.0) or 0.0)
        ablation_every = int(config.get("ablation_loss_every_steps", 1) or 1)
        ablation_hinge_value = 0.0
        ablation_delta_value = 0.0
        if (
            (not dummy_batch)
            and ablation_loss_weight > 0.0
            and ablation_every > 0
            and (optimizer_step % ablation_every == 0)
            and (batch_idx % int(gradient_accumulation_steps) == 0)
        ):
            try:
                with torch.no_grad():
                    if use_amp:
                        with autocast():
                            outputs_no_audio = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                audio_tokens=None,
                                audio_attention_mask=None,
                            )
                            loss_no_audio = outputs_no_audio["loss"]
                    else:
                        outputs_no_audio = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            audio_tokens=None,
                            audio_attention_mask=None,
                        )
                        loss_no_audio = outputs_no_audio["loss"]

                # Encourage loss_audio <= loss_no_audio - margin.
                ablation_delta = loss_no_audio.detach() - loss
                hinge = torch.nn.functional.relu(ablation_loss_margin - ablation_delta)
                loss = loss + ablation_loss_weight * hinge

                ablation_hinge_value = float(hinge.detach().item())
                ablation_delta_value = float(ablation_delta.detach().item())
            except Exception:
                pass

        # Token-level training accuracy on supervised positions.
        # IMPORTANT: match the shifted next-token objective used in SAFEModel.forward().
        logits = outputs.get("logits") if isinstance(outputs, dict) else None
        if isinstance(logits, torch.Tensor) and logits.ndim >= 3 and isinstance(labels, torch.Tensor) and labels.ndim >= 2:
            try:
                with torch.no_grad():
                    shift_logits = logits[..., :-1, :]
                    shift_labels = labels[..., 1:]
                    preds = shift_logits.argmax(dim=-1)
                    mask = shift_labels != -100
                    if mask.any():
                        token_correct += int((preds[mask] == shift_labels[mask]).sum().item())
                        token_total += int(mask.sum().item())
            except Exception:
                pass

        # Optional audio-text contrastive loss (InfoNCE-style) on this batch
        contrastive_weight = float(config.get("audio_contrastive_weight", 0.0) or 0.0)
        contrastive_temp = float(config.get("audio_contrastive_temperature", 0.07) or 0.07)
        if contrastive_weight > 0.0 and audio_tokens is not None:
            try:
                # Pool audio tokens to a single vector per sample
                # audio_tokens: (B, T, H) -> (B, H)
                audio_vecs = audio_tokens.mean(dim=1)

                # Resolve textual answers for each sample using SAFEModel helper
                resolved_answers: List[str] = []
                for ans in answers:
                    if hasattr(base_model, "_select_training_answer"):
                        text = base_model._select_training_answer(ans)
                    else:
                        text = ans if isinstance(ans, str) else (ans[0] if isinstance(ans, list) and ans else "")
                    resolved_answers.append(str(text or "").strip())

                # Filter out empty pairs
                paired_indices = [i for i, t in enumerate(resolved_answers) if t]
                if paired_indices:
                    audio_vecs = audio_vecs[paired_indices]
                    text_list = [resolved_answers[i] for i in paired_indices]

                    if audio_vecs.size(0) > 1:
                        text_vecs = _embed_texts_for_contrastive(
                            model,
                            text_list,
                            device=device,
                            max_length=int(config.get("audio_contrastive_max_length", 48) or 48),
                        )
                        if text_vecs.numel() > 0:
                            # Normalize
                            audio_norm = torch.nn.functional.normalize(audio_vecs, dim=-1)
                            text_norm = torch.nn.functional.normalize(text_vecs, dim=-1)
                            # Similarity matrix: (B, B)
                            sim = audio_norm @ text_norm.t() / max(contrastive_temp, 1e-5)
                            targets = torch.arange(sim.size(0), device=device)
                            contrastive_loss = torch.nn.functional.cross_entropy(sim, targets)
                            loss = loss + contrastive_weight * contrastive_loss
            except Exception:
                # Fail-safe: ignore contrastive errors to keep training running
                pass

        # Min audio attention regularization loss (KV augmentation mode)
        # Forces the model to attend to audio tokens, preventing language-prior shortcuts
        min_attn_loss_value = 0.0
        if (hasattr(base_model, 'min_audio_attention_loss') and
            base_model.min_audio_attention_loss is not None and
            hasattr(base_model, 'kv_hook_manager') and
            base_model.kv_hook_manager is not None):
            try:
                # Get LIVE attention weights (with gradients) for reg loss computation
                # This gets and clears the live weights to prevent graph retention
                attn_weights = base_model.kv_hook_manager.get_live_attention_weights()
                if attn_weights:
                    n_audio = base_model.num_audio_tokens
                    min_attn_loss = base_model.min_audio_attention_loss(
                        attention_weights=attn_weights,
                        n_audio=n_audio,
                        supervised_mask=None,  # Uses last-k tokens by default
                    )
                    if min_attn_loss.requires_grad:
                        loss = loss + min_attn_loss
                        min_attn_loss_value = min_attn_loss.detach().item()

                    # DEBUG: Log min_attn_loss gradient info (first 3 batches per epoch)
                    if batch_idx < 3 and epoch <= 2:
                        print(f"[MinAttnDebug] batch={batch_idx} "
                              f"requires_grad={min_attn_loss.requires_grad} "
                              f"grad_fn={type(min_attn_loss.grad_fn).__name__ if min_attn_loss.grad_fn else None} "
                              f"value={min_attn_loss_value:.6f}", flush=True)
            except Exception as e:
                # Fail-safe: ignore min attention errors to keep training running
                if batch_idx < 3:
                    print(f"[MinAttnDebug] ERROR: {e}", flush=True)

        # If loss has no gradient path (e.g., SAFE gate effectively off),
        # skip this batch to avoid autograd errors.
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            # In DDP, do not skip backward on only some ranks (can hang).
            if ddp_enabled:
                loss = _ddp_noop_loss()
            else:
                continue

        # Normalize by gradient accumulation steps
        loss = loss / gradient_accumulation_steps

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Accumulate per-optimizer-step stats for W&B
        with torch.no_grad():
            try:
                loss_unscaled_value = float(loss.detach().item() * gradient_accumulation_steps)
            except Exception:
                loss_unscaled_value = 0.0
        step_loss_sum += loss_unscaled_value
        step_min_attn_loss_sum += min_attn_loss_value
        step_ablation_hinge_sum += ablation_hinge_value
        step_ablation_delta_sum += ablation_delta_value
        step_samples += len(questions)
        step_micro_batches += 1

        # Gradient health check (first 3 epochs, every 50 batches)
        # Detects learning failures early by monitoring audio component gradients
        if epoch <= 3 and batch_idx % 50 == 0:
            proj_grad_norm = 0.0
            fuse_grad_norm = 0.0
            kv_grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if "audio_projector" in name:
                        proj_grad_norm += grad_norm
                    if "fusion_adapter" in name:
                        fuse_grad_norm += grad_norm
                    if "kv_adapter" in name or "kv_augmentation" in name:
                        kv_grad_norm += grad_norm
            print(f"[GradCheck] epoch={epoch} batch={batch_idx} proj={proj_grad_norm:.6f} fuse={fuse_grad_norm:.6f} kv={kv_grad_norm:.6f}", flush=True)
            last_proj_grad_norm = float(proj_grad_norm)
            last_fuse_grad_norm = float(fuse_grad_norm)

        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if use_amp:
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            optimizer_step += 1

            # Comprehensive diagnostics (every 10 steps during warmup, every 50 steps after)
            diag_frequency = 10 if optimizer_step <= 500 else 50
            if optimizer_step % diag_frequency == 0:
                # Get current gate value for logging
                current_gate = getattr(base_model, "_default_gate", None)
                avg_loss = step_loss_sum / max(step_micro_batches, 1)
                _log_comprehensive_diagnostics(
                    model=model,
                    base_model=base_model,
                    optimizer_step=optimizer_step,
                    epoch=epoch,
                    loss=avg_loss,
                    grad_norm=float(grad_norm) if grad_norm is not None else None,
                    gate_value=current_gate,
                    wandb_run=wandb_run,
                    console_log=True,
                )

            # W&B logging at optimizer-step granularity
            if wandb_run is not None:
                step_time = max(time.time() - step_start_time, 1e-6)
                lrs = scheduler.get_last_lr()
                log_dict: Dict[str, Any] = {
                    "train/optimizer_step": optimizer_step,
                    "epoch": epoch,
                    "train/batch_idx": batch_idx,
                    "train/micro_batches": step_micro_batches,
                    "train/loss_step": (step_loss_sum / max(step_micro_batches, 1)),
                    "train/min_attn_loss_step": (step_min_attn_loss_sum / max(step_micro_batches, 1)),
                    "train/ablation_hinge_step": (step_ablation_hinge_sum / max(step_micro_batches, 1)),
                    "train/ablation_delta_step": (step_ablation_delta_sum / max(step_micro_batches, 1)),
                    "train/samples_per_sec_step": (step_samples * world_size) / step_time,
                    "train/grad_norm": float(grad_norm) if grad_norm is not None else None,
                }
                for group_idx, (group, lr) in enumerate(zip(optimizer.param_groups, lrs)):
                    group_name = str(group.get("name") or f"group{group_idx}")
                    log_dict[f"train/lr/{group_name}"] = float(lr)

                if use_amp and scaler is not None:
                    try:
                        log_dict["train/amp_scale"] = float(scaler.get_scale())
                    except Exception:
                        pass

                if last_proj_grad_norm is not None:
                    log_dict["train/gradcheck/audio_projector_sum_norm"] = last_proj_grad_norm
                if last_fuse_grad_norm is not None:
                    log_dict["train/gradcheck/fusion_adapter_sum_norm"] = last_fuse_grad_norm

                # Token-level accuracy (training accuracy proxy)
                if token_total > 0:
                    log_dict["train/token_accuracy"] = float(token_correct / token_total)

                if audio_tokens is not None:
                    try:
                        with torch.no_grad():
                            log_dict["train/audio_token_norm"] = float(audio_tokens.norm(dim=-1).mean().item())
                    except Exception:
                        pass

                # Track common "fusion collapse" indicators (scales drifting to ~0).
                proj_scale = _extract_audio_projector_output_scale(base_model)
                if proj_scale is not None:
                    log_dict["train/audio_projector_output_scale"] = float(proj_scale)

                residual_scales = _extract_fusion_residual_scales(base_model)
                if residual_scales:
                    values = list(residual_scales.values())
                    log_dict["train/fusion_residual_scale_mean"] = float(sum(values) / len(values))
                    log_dict["train/fusion_residual_scale_min"] = float(min(values))
                    log_dict["train/fusion_residual_scale_max"] = float(max(values))
                    log_dict["train/fusion_residual_scale_count"] = float(len(values))
                    for k, v in residual_scales.items():
                        log_dict[f"train/fusion_residual_scale/{k}"] = float(v)

                # KV adapter metrics (ΔQ scale, audio scale)
                kv_metrics = _extract_kv_adapter_metrics(base_model)
                if kv_metrics:
                    for k, v in kv_metrics.items():
                        log_dict[f"train/kv_adapter/{k}"] = float(v)

                # KV hook manager diagnostics (attention patterns, RMS ratios)
                kv_hook_manager = getattr(base_model, "kv_hook_manager", None)
                if kv_hook_manager is not None and hasattr(kv_hook_manager, "log_diagnostics"):
                    try:
                        kv_diag = kv_hook_manager.log_diagnostics(prefix="train/kv_attn/")
                        log_dict.update(kv_diag)
                    except Exception:
                        pass

                try:
                    if hasattr(base_model, "get_last_attention_summary"):
                        summary = base_model.get_last_attention_summary()
                        if isinstance(summary, dict):
                            if summary.get("overall_mean", None) is not None:
                                log_dict["train/attn_mean"] = float(summary["overall_mean"])
                            if summary.get("overall_max", None) is not None:
                                log_dict["train/attn_max"] = float(summary["overall_max"])
                except Exception:
                    pass

                if torch.cuda.is_available():
                    try:
                        log_dict["train/gpu_mem_allocated_mb"] = float(torch.cuda.memory_allocated() / (1024**2))
                        log_dict["train/gpu_mem_reserved_mb"] = float(torch.cuda.memory_reserved() / (1024**2))
                        log_dict["train/gpu_max_mem_allocated_mb"] = float(
                            torch.cuda.max_memory_allocated() / (1024**2)
                        )
                    except Exception:
                        pass

                _wandb_log(wandb_run, log_dict, step=optimizer_step)

            # Periodic training accuracy eval (CIDEr/METEOR on training subset)
            if (train_eval_loader is not None and
                train_eval_steps > 0 and
                optimizer_step % train_eval_steps == 0 and
                wandb_run is not None):
                print(f"[TrainEval] Computing training accuracy at step {optimizer_step}...", flush=True)
                try:
                    train_eval_metrics = evaluate(
                        model=model,
                        dataloader=train_eval_loader,
                        device=device,
                        max_batches=None,  # Use all samples in the small loader
                        max_new_tokens=config.get("max_new_tokens", 20),
                        num_beams=1,  # Greedy for speed
                        repetition_penalty=float(config.get("eval_repetition_penalty", 1.1) or 1.1),
                        no_repeat_ngram_size=int(config.get("eval_no_repeat_ngram_size", 3) or 3),
                        light_metrics=True,
                        eval_prompt=config.get("eval_prompt"),
                        suppress_eos_for_audio=False,
                    )
                    train_eval_log = {
                        "train_acc/cider": train_eval_metrics.get("cider", 0.0),
                        "train_acc/meteor": train_eval_metrics.get("meteor", 0.0),
                        "train_acc/bleu4": train_eval_metrics.get("bleu4", 0.0),
                        "train_acc/rouge_l": train_eval_metrics.get("rouge_l", 0.0),
                    }
                    # Optional A/B diagnostic: compute metrics with audio disabled.
                    if bool(config.get("train_eval_ablate_audio", False)):
                        ablate_max_batches = config.get("train_eval_ablate_max_batches", 10)
                        if ablate_max_batches is not None and int(ablate_max_batches) <= 0:
                            ablate_max_batches = None
                        train_eval_metrics_no_audio = evaluate(
                            model=model,
                            dataloader=train_eval_loader,
                            device=device,
                            max_batches=ablate_max_batches,
                            max_new_tokens=config.get("max_new_tokens", 20),
                            num_beams=1,
                            repetition_penalty=float(config.get("eval_repetition_penalty", 1.1) or 1.1),
                            no_repeat_ngram_size=int(config.get("eval_no_repeat_ngram_size", 3) or 3),
                            light_metrics=True,
                            eval_prompt=config.get("eval_prompt"),
                            ablate_audio=True,
                            suppress_eos_for_audio=False,
                        )
                        train_eval_log.update(
                            {
                                "train_acc_no_audio/cider": train_eval_metrics_no_audio.get("cider", 0.0),
                                "train_acc_no_audio/meteor": train_eval_metrics_no_audio.get("meteor", 0.0),
                                "train_acc_delta/cider": train_eval_metrics.get("cider", 0.0)
                                - train_eval_metrics_no_audio.get("cider", 0.0),
                                "train_acc_delta/meteor": train_eval_metrics.get("meteor", 0.0)
                                - train_eval_metrics_no_audio.get("meteor", 0.0),
                            }
                        )
                    _wandb_log(wandb_run, train_eval_log, step=optimizer_step)
                    print(f"[TrainEval] CIDEr={train_eval_metrics.get('cider', 0.0):.2f} "
                          f"METEOR={train_eval_metrics.get('meteor', 0.0):.4f}", flush=True)

                    # Re-enable training mode after eval
                    if hasattr(base_model, "enable_audio_training"):
                        base_model.enable_audio_training()
                    else:
                        model.train()
                except Exception as e:
                    print(f"[TrainEval] Error: {e}", flush=True)

            # Reset step accumulators
            step_start_time = time.time()
            step_loss_sum = 0.0
            step_min_attn_loss_sum = 0.0
            step_samples = 0
            step_micro_batches = 0

        # Logging
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        num_samples += len(questions)

        # Periodic logging
        current_time = time.time()
        if current_time - last_log_time > 60:  # Log every minute
            avg_loss = total_loss / num_batches
            elapsed = current_time - start_time
            samples_per_sec = (num_samples * world_size) / elapsed
            lr = scheduler.get_last_lr()[0]

            # Optional diagnostics for audio fusion strength
            audio_token_norm = None
            if audio_tokens is not None:
                try:
                    with torch.no_grad():
                        audio_token_norm = float(audio_tokens.norm(dim=-1).mean().item())
                except Exception:
                    audio_token_norm = None

            proj_scale = _extract_audio_projector_output_scale(base_model)
            residual_scale_mean = None
            residual_scales = _extract_fusion_residual_scales(base_model)
            if residual_scales:
                values = list(residual_scales.values())
                residual_scale_mean = float(sum(values) / len(values))

            # KV adapter metrics for console logging
            kv_metrics = _extract_kv_adapter_metrics(base_model)
            delta_q_scale_mean = None
            if kv_metrics:
                dq_scales = [v for k, v in kv_metrics.items() if "delta_q_scale" in k]
                if dq_scales:
                    delta_q_scale_mean = sum(dq_scales) / len(dq_scales)

            attn_mean = None
            attn_max = None
            try:
                if hasattr(base_model, "get_last_attention_summary"):
                    summary = base_model.get_last_attention_summary()
                    if isinstance(summary, dict):
                        attn_mean = summary.get("overall_mean", None)
                        attn_max = summary.get("overall_max", None)
            except Exception:
                attn_mean = attn_max = None

            log_msg = (
                f"[Epoch {epoch}] Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                f"Speed: {samples_per_sec:.1f} samples/s"
            )
            extras = []
            if audio_token_norm is not None:
                extras.append(f"audio_norm={audio_token_norm:.2f}")
            if proj_scale is not None:
                extras.append(f"proj_scale={proj_scale:.3f}")
            if residual_scale_mean is not None:
                extras.append(f"res_scale={residual_scale_mean:.3f}")
            if delta_q_scale_mean is not None:
                extras.append(f"dq_scale={delta_q_scale_mean:.3f}")
            if attn_mean is not None and attn_max is not None:
                extras.append(f"attn_mean={attn_mean:.4f} attn_max={attn_max:.4f}")
            if extras:
                log_msg = f"{log_msg} | " + " ".join(extras)

            print(log_msg, flush=True)
            last_log_time = current_time

    # Flush any remaining accumulated gradients so the last partial micro-batch
    # group still contributes an optimizer step.
    if step_micro_batches > 0:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()
        optimizer_step += 1

        if wandb_run is not None:
            step_time = max(time.time() - step_start_time, 1e-6)
            lrs = scheduler.get_last_lr()
            log_dict: Dict[str, Any] = {
                "train/optimizer_step": optimizer_step,
                "epoch": epoch,
                "train/batch_idx": batch_idx,
                "train/micro_batches": step_micro_batches,
                "train/loss_step": (step_loss_sum / max(step_micro_batches, 1)),
                "train/min_attn_loss_step": (step_min_attn_loss_sum / max(step_micro_batches, 1)),
                "train/samples_per_sec_step": (step_samples * world_size) / step_time,
                "train/grad_norm": float(grad_norm) if grad_norm is not None else None,
            }
            for group_idx, (group, lr) in enumerate(zip(optimizer.param_groups, lrs)):
                group_name = str(group.get("name") or f"group{group_idx}")
                log_dict[f"train/lr/{group_name}"] = float(lr)

            if use_amp and scaler is not None:
                try:
                    log_dict["train/amp_scale"] = float(scaler.get_scale())
                except Exception:
                    pass

            if last_proj_grad_norm is not None:
                log_dict["train/gradcheck/audio_projector_sum_norm"] = last_proj_grad_norm
            if last_fuse_grad_norm is not None:
                log_dict["train/gradcheck/fusion_adapter_sum_norm"] = last_fuse_grad_norm

            if audio_tokens is not None:
                try:
                    with torch.no_grad():
                        log_dict["train/audio_token_norm"] = float(audio_tokens.norm(dim=-1).mean().item())
                except Exception:
                    pass

            proj_scale = _extract_audio_projector_output_scale(base_model)
            if proj_scale is not None:
                log_dict["train/audio_projector_output_scale"] = float(proj_scale)

            residual_scales = _extract_fusion_residual_scales(base_model)
            if residual_scales:
                values = list(residual_scales.values())
                log_dict["train/fusion_residual_scale_mean"] = float(sum(values) / len(values))
                log_dict["train/fusion_residual_scale_min"] = float(min(values))
                log_dict["train/fusion_residual_scale_max"] = float(max(values))
                log_dict["train/fusion_residual_scale_count"] = float(len(values))
                for k, v in residual_scales.items():
                    log_dict[f"train/fusion_residual_scale/{k}"] = float(v)

            # KV adapter metrics (ΔQ scale, audio scale)
            kv_metrics = _extract_kv_adapter_metrics(base_model)
            if kv_metrics:
                for k, v in kv_metrics.items():
                    log_dict[f"train/kv_adapter/{k}"] = float(v)

            try:
                if hasattr(base_model, "get_last_attention_summary"):
                    summary = base_model.get_last_attention_summary()
                    if isinstance(summary, dict):
                        if summary.get("overall_mean", None) is not None:
                            log_dict["train/attn_mean"] = float(summary["overall_mean"])
                        if summary.get("overall_max", None) is not None:
                            log_dict["train/attn_max"] = float(summary["overall_max"])
            except Exception:
                pass

            if torch.cuda.is_available():
                try:
                    log_dict["train/gpu_mem_allocated_mb"] = float(torch.cuda.memory_allocated() / (1024**2))
                    log_dict["train/gpu_mem_reserved_mb"] = float(torch.cuda.memory_reserved() / (1024**2))
                    log_dict["train/gpu_max_mem_allocated_mb"] = float(
                        torch.cuda.max_memory_allocated() / (1024**2)
                    )
                except Exception:
                    pass

            _wandb_log(wandb_run, log_dict, step=optimizer_step)

        # Reset step accumulators
        step_start_time = time.time()
        step_loss_sum = 0.0
        step_min_attn_loss_sum = 0.0
        step_samples = 0
        step_micro_batches = 0

    # Final statistics
    if num_batches == 0:
        raise RuntimeError(
            f"No valid training batches were processed in epoch {epoch}. "
            f"Skipped_batches={skipped_batches}, dropped_samples={skipped_samples + filtered_samples}. "
            "This usually means your dataset has missing/empty captions and/or missing audio paths."
        )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    elapsed = time.time() - start_time

    # Total samples across all GPUs
    total_samples_all_gpus = num_samples * world_size

    metrics = {
        "loss": avg_loss,
        "num_samples": total_samples_all_gpus,
        "train_time": elapsed,
        "samples_per_sec": total_samples_all_gpus / elapsed,
        "skipped_batches": skipped_batches,
        "dropped_samples": skipped_samples + filtered_samples,
        "missing_audio_samples": missing_audio_samples,
        "missing_caption_samples": missing_caption_samples,
    }
    if token_total > 0:
        metrics["token_accuracy"] = float(token_correct / max(token_total, 1))

    # Memory cleanup after training epoch to prevent OOM during long runs
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return metrics, optimizer_step


def train(
    model: SAFEModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    wandb_run: Any = None,
    wandb_log_checkpoints: bool = False,
    wandb_sample_count: int = 0,
    dist_info: Optional[Dict[str, Any]] = None,
    train_sampler: Optional[DistributedSampler] = None,
    train_eval_loader: Optional[DataLoader] = None,
    train_eval_steps: int = 0,
) -> Dict[str, Any]:
    """
    Main training loop

    Args:
        model: SAFE model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration
        output_dir: Output directory for checkpoints
        device: Device to train on
        dist_info: Distributed training info (rank, world_size, etc.)
        train_sampler: DistributedSampler for training (if distributed)
        train_eval_loader: DataLoader for periodic training accuracy eval
        train_eval_steps: Evaluate training accuracy every N optimizer steps

    Returns:
        Training history dict
    """
    # Handle distributed training info
    if dist_info is None:
        dist_info = {"rank": 0, "world_size": 1, "is_main": True, "distributed": False}
    is_main = dist_info["is_main"]

    # For DDP, get the underlying model for parameter grouping
    base_model = model.module if dist_info["distributed"] else model

    # Setup optimizer with different learning rates (Stage-A style defaults)
    lr_projector = config.get("learning_rate_projector", 2e-4)
    lr_adapter = config.get("learning_rate_adapter", 1e-4)
    weight_decay = config.get("weight_decay", 0.01)

    def _use_weight_decay(param_name: str, param: torch.nn.Parameter) -> bool:
        """Return True if this parameter should receive weight decay."""
        # Standard practice: do not decay biases / LayerNorm / scalar scales.
        if param.ndim <= 1:
            return False
        name = param_name.lower()
        if name.endswith(".bias") or name.endswith("bias"):
            return False
        if "layernorm" in name or "layer_norm" in name or ".norm" in name or "norm." in name:
            return False
        if name.endswith(("output_scale", "residual_scale")):
            return False
        return True

    # Group parameters by component + (decay/no_decay)
    projector_decay = []
    projector_no_decay = []
    adapter_decay = []
    adapter_no_decay = []
    other_decay = []
    other_no_decay = []

    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue

        if "audio_projector" in name:
            if _use_weight_decay(name, param):
                projector_decay.append(param)
            else:
                projector_no_decay.append(param)
        elif (
            "fusion_adapter" in name
            or "kv_adapter" in name
            or "audio_query_adapter" in name
            or "kv_augmentation" in name
            or "lora" in name.lower()
        ):
            if _use_weight_decay(name, param):
                adapter_decay.append(param)
            else:
                adapter_no_decay.append(param)
        else:
            if _use_weight_decay(name, param):
                other_decay.append(param)
            else:
                other_no_decay.append(param)

    param_groups = []
    if projector_decay:
        param_groups.append(
            {
                "params": projector_decay,
                "lr": lr_projector,
                "weight_decay": weight_decay,
                "name": "projector_decay",
            }
        )
    if projector_no_decay:
        param_groups.append(
            {
                "params": projector_no_decay,
                "lr": lr_projector,
                "weight_decay": 0.0,
                "name": "projector_no_decay",
            }
        )
    if adapter_decay:
        param_groups.append(
            {
                "params": adapter_decay,
                "lr": lr_adapter,
                "weight_decay": weight_decay,
                "name": "adapter_decay",
            }
        )
    if adapter_no_decay:
        param_groups.append(
            {
                "params": adapter_no_decay,
                "lr": lr_adapter,
                "weight_decay": 0.0,
                "name": "adapter_no_decay",
            }
        )
    if other_decay:
        param_groups.append(
            {
                "params": other_decay,
                "lr": lr_adapter,
                "weight_decay": weight_decay,
                "name": "other_decay",
            }
        )
    if other_no_decay:
        param_groups.append(
            {
                "params": other_no_decay,
                "lr": lr_adapter,
                "weight_decay": 0.0,
                "name": "other_no_decay",
            }
        )

    if not param_groups:
        raise RuntimeError("No trainable parameters found to optimize.")

    # DEBUG: Check if kv_adapter params are in optimizer
    if is_main:
        print("\n=== DEBUG: Parameter groups in optimizer ===")
        total_in_optimizer = 0
        kv_count = 0
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                total_in_optimizer += 1
                if "kv_adapter" in name or "kv_adapters" in name:
                    kv_count += 1
                    print(f"  KV param: {name} requires_grad={param.requires_grad}")
        print(f"Total trainable params: {total_in_optimizer}, KV adapter params: {kv_count}")
        if kv_count == 0 and hasattr(base_model, 'kv_adapters') and base_model.kv_adapters is not None:
            print("  ⚠️ WARNING: kv_adapters exists but 0 params have requires_grad=True!")
            print("  Checking kv_adapters state:")
            for name, param in base_model.kv_adapters.named_parameters():
                print(f"    {name}: requires_grad={param.requires_grad}")
        print("=" * 50 + "\n")

    optimizer = AdamW(param_groups, weight_decay=0.0)

    # Learning rate scheduler
    num_epochs = int(config.get("num_epochs", 20) or 0)
    warmup_steps = int(config.get("warmup_steps", 1000) or 0)
    grad_accum = max(1, int(config.get("gradient_accumulation_steps", 1) or 1))
    steps_per_epoch = (len(train_loader) + grad_accum - 1) // grad_accum
    total_steps = max(1, steps_per_epoch * max(1, num_epochs))
    warmup_steps = min(max(0, warmup_steps), total_steps)
    min_lr_ratio = float(config.get("min_lr_ratio", 0.1) or 0.1)  # Floor at 10% of base LR

    # Cosine schedule with warmup and minimum LR floor.
    # NOTE: scheduler.step() is called per optimizer step (i.e., per grad-accum update),
    # so total_steps must be in optimizer-step units, not micro-batch units.
    def lr_lambda(step: int) -> float:
        step = int(step)
        if warmup_steps > 0 and step < warmup_steps:
            return step / float(warmup_steps)

        denom = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / float(denom)
        progress = float(min(1.0, max(0.0, progress)))

        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = GradScaler() if config.get("fp16", False) else None

    # Training loop
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_cider": [],
        "val_bleu4": [],
        "epochs": [],
    }

    best_cider = 0.0
    patience = config.get("early_stopping_patience", 5)
    patience_counter = 0

    optimizer_step = 0

    print(f"\n{'='*80}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"  Projector LR: {lr_projector}")
    print(f"  Adapter LR: {lr_adapter}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Mixed precision: {config.get('fp16', False)}")
    print(f"{'='*80}\n")

    # Optional initial evaluation before training (sanity check)
    initial_max_eval = config.get("max_eval_batches", None)
    if initial_max_eval is not None and initial_max_eval <= 0:
        initial_max_eval = None
    suppress_eos_steps = int(config.get("suppress_eos_for_audio_early_steps", 0) or 0)
    suppress_eos_for_audio = bool(suppress_eos_steps > 0 and optimizer_step < suppress_eos_steps)
    print(
        f"[InitEval] Running initial evaluation on validation set "
        f"(max_batches={initial_max_eval}, suppress_eos_for_audio={suppress_eos_for_audio})",
        flush=True,
    )
    init_samples: Optional[List[Dict[str, Any]]] = [] if (wandb_run is not None and wandb_sample_count > 0) else None
    init_metrics = evaluate(
        model,
        val_loader,
        device,
        max_batches=initial_max_eval,
        max_new_tokens=config.get("max_new_tokens", 20),
        num_beams=config.get("num_beams", 1),
        repetition_penalty=float(config.get("eval_repetition_penalty", 1.1) or 1.1),
        no_repeat_ngram_size=int(config.get("eval_no_repeat_ngram_size", 3) or 3),
        # Training-time eval: use light metrics (BLEU, METEOR, ROUGE, CIDEr), skip SPICE/BERTScore
        compute_bertscore=False,
        light_metrics=True,
        suppress_eos_for_audio=suppress_eos_for_audio,
        eval_prompt=config.get("eval_prompt"),
        sample_output=init_samples,
        sample_limit=wandb_sample_count,
    )
    print(f"[InitEval] CIDEr={init_metrics.get('cider', 0.0):.2f} BLEU-4={init_metrics.get('bleu4', 0.0):.4f}", flush=True)
    if bool(config.get("eval_ablate_audio", False)):
        init_metrics_no_audio = evaluate(
            model,
            val_loader,
            device,
            max_batches=initial_max_eval,
            max_new_tokens=config.get("max_new_tokens", 20),
            num_beams=config.get("num_beams", 1),
            repetition_penalty=float(config.get("eval_repetition_penalty", 1.1) or 1.1),
            no_repeat_ngram_size=int(config.get("eval_no_repeat_ngram_size", 3) or 3),
            compute_bertscore=False,
            light_metrics=True,
            suppress_eos_for_audio=suppress_eos_for_audio,
            eval_prompt=config.get("eval_prompt"),
            ablate_audio=True,
            sample_output=None,
            sample_limit=0,
        )
        print(
            f"[InitEval A/B] CIDEr audio={init_metrics.get('cider', 0.0):.2f} "
            f"no_audio={init_metrics_no_audio.get('cider', 0.0):.2f} "
            f"delta={init_metrics.get('cider', 0.0) - init_metrics_no_audio.get('cider', 0.0):.2f}",
            flush=True,
        )
        if wandb_run is not None:
            ablate_log = {"train/optimizer_step": optimizer_step, "epoch": 0}
            ablate_log.update({f"val_no_audio/{k}": v for k, v in init_metrics_no_audio.items() if isinstance(v, (int, float))})
            ablate_log["val_delta/cider"] = float(init_metrics.get("cider", 0.0) - init_metrics_no_audio.get("cider", 0.0))
            ablate_log["val_delta/meteor"] = float(init_metrics.get("meteor", 0.0) - init_metrics_no_audio.get("meteor", 0.0))
            _wandb_log(wandb_run, ablate_log, step=optimizer_step)
    if wandb_run is not None:
        init_log = {"train/optimizer_step": optimizer_step, "epoch": 0}
        init_log.update({f"val/{k}": v for k, v in init_metrics.items() if isinstance(v, (int, float))})
        _wandb_log(wandb_run, init_log, step=optimizer_step)
        if init_samples and wandb is not None:
            try:
                table = wandb.Table(columns=["sample_id", "audio", "question", "prediction", "references", "audio_path", "subset"])
                audio_success_count = 0
                audio_fail_count = 0
                for row_idx, row in enumerate(init_samples):
                    audio_cell = None
                    audio_value = row.get("audio")
                    audio_path_value = row.get("audio_path")

                    # Debug: log audio data format for first few samples
                    if row_idx < 3:
                        print(f"[W&B Audio Debug] Sample {row_idx}:", flush=True)
                        print(f"  audio type: {type(audio_value)}", flush=True)
                        if isinstance(audio_value, tuple):
                            print(f"  audio tuple len: {len(audio_value)}", flush=True)
                            if len(audio_value) >= 1:
                                print(f"  audio[0] type: {type(audio_value[0])}, is_tensor: {torch.is_tensor(audio_value[0]) if audio_value[0] is not None else 'N/A'}", flush=True)
                            if len(audio_value) >= 2:
                                print(f"  audio[1] (sample_rate): {audio_value[1]}", flush=True)
                        print(f"  audio_path: {audio_path_value}", flush=True)

                    if isinstance(audio_value, tuple) and len(audio_value) == 2 and torch.is_tensor(audio_value[0]):
                        try:
                            audio_cell = wandb.Audio(
                                audio_value[0].detach().cpu().numpy(),
                                sample_rate=int(audio_value[1]),
                            )
                            audio_success_count += 1
                        except Exception as e:
                            if row_idx < 3:
                                print(f"  [W&B Audio] Failed to create from tensor: {e}", flush=True)
                            audio_cell = None
                    elif audio_path_value:
                        try:
                            audio_cell = wandb.Audio(str(audio_path_value))
                            audio_success_count += 1
                        except Exception as e:
                            if row_idx < 3:
                                print(f"  [W&B Audio] Failed to create from path: {e}", flush=True)
                            audio_cell = None

                    if audio_cell is None:
                        audio_fail_count += 1

                    table.add_data(
                        row.get("sample_id"),
                        audio_cell,
                        row.get("question"),
                        row.get("prediction"),
                        "\n".join(row.get("references") or []),
                        row.get("audio_path"),
                        row.get("subset"),
                    )
                print(f"[W&B Audio] Created {audio_success_count}/{len(init_samples)} audio cells, {audio_fail_count} failed", flush=True)
                _wandb_log(wandb_run, {"train/optimizer_step": optimizer_step, "val/samples": table}, step=optimizer_step)
            except Exception as e:
                print(f"[W&B Audio] Failed to create/log table: {e}", flush=True)

    if init_samples and bool(config.get("export_eval_samples", False)):
        _export_eval_samples(
            output_dir,
            split="val",
            epoch=0,
            samples=init_samples,
            export_audio=bool(config.get("export_eval_samples_audio", False)),
        )

    # Create audio augmentation pipeline if enabled (off by default)
    audio_augment_pipeline = None
    if config.get("audio_augment", False):
        augment_prob = config.get("audio_augment_prob", 0.5)
        audio_augment_pipeline = create_augment_pipeline(
            enabled=True,
            spec_augment=True,
            waveform_augment=True,
            spec_augment_config={"p": augment_prob},
            waveform_augment_config={"p": augment_prob},
        )
        if is_main:
            print(f"[AudioAugment] Enabled with probability {augment_prob}")

    for epoch in range(1, num_epochs + 1):
        if is_main:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*80}\n")

        # Apply scale minimum warmup to prevent early collapse while forcing
        # strong audio signal later in training
        if hasattr(model, 'set_scale_minimum_warmup'):
            model.set_scale_minimum_warmup(
                epoch=epoch - 1,  # 0-indexed
                warmup_epochs=5,
                start_min=0.5,
                end_min=1.0,
            )

        # min_audio_attention curriculum is step-based (handled inside train_epoch).

        # Set epoch on DistributedSampler for proper shuffling across epochs
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Allow datasets to reshuffle/resample between epochs (e.g., WavCaps subset sampling).
        try:
            dataset = getattr(train_loader, "dataset", None)
            if dataset is not None and hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
        except Exception:
            pass

        # Train
        train_metrics, optimizer_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            config,
            scaler,
            wandb_run=wandb_run,
            optimizer_step=optimizer_step,
            world_size=dist_info["world_size"],
            train_eval_loader=train_eval_loader,
            train_eval_steps=train_eval_steps,
            audio_augment_pipeline=audio_augment_pipeline,
        )
        if wandb_run is not None:
            train_log = {"train/optimizer_step": optimizer_step, "epoch": epoch}
            train_log.update({f"train/epoch_{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))})
            _wandb_log(wandb_run, train_log, step=optimizer_step)

        # Gate warmup: ramp SAFE gate from 0 → 1 over a configured number of optimizer steps
        gate_warmup_steps = int(config.get("gate_warmup_steps", 0) or 0)
        if gate_warmup_steps > 0 and hasattr(base_model, "set_gate_warmup"):
            base_model.set_gate_warmup(optimizer_step, warmup_steps=gate_warmup_steps)

        # Residual scale warmup: ramp residual scale from start to end over optimizer steps
        # This prevents the model from learning to "fight" audio early in training
        res_warmup_steps = int(config.get("residual_scale_warmup_steps", 0) or 0)
        if res_warmup_steps > 0 and hasattr(base_model, "set_residual_scale_step_warmup"):
            res_start = float(config.get("residual_scale_warmup_start", 0.2))
            res_end = float(config.get("residual_scale_warmup_end", 1.0))
            base_model.set_residual_scale_step_warmup(
                optimizer_step,
                warmup_steps=res_warmup_steps,
                start_scale=res_start,
                end_scale=res_end,
            )

        print(f"\n✓ Training complete:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Time: {format_time(train_metrics['train_time'])}")
        print(f"  Speed: {train_metrics['samples_per_sec']:.1f} samples/sec")

        # Evaluate
        eval_frequency = config.get("eval_frequency", 1)
        if epoch % eval_frequency == 0:
            print(f"\n{'='*80}")
            print("Running validation...")
            print(f"{'='*80}\n")

            suppress_eos_steps = int(config.get("suppress_eos_for_audio_early_steps", 0) or 0)
            suppress_eos_for_audio = bool(suppress_eos_steps > 0 and optimizer_step < suppress_eos_steps)

            val_samples: Optional[List[Dict[str, Any]]] = (
                [] if (wandb_run is not None and wandb_sample_count > 0) else None
            )
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                max_batches=config.get("max_eval_batches"),
                max_new_tokens=config.get("max_new_tokens", 20),
                num_beams=config.get("num_beams", 1),
                repetition_penalty=float(config.get("eval_repetition_penalty", 1.1) or 1.1),
                no_repeat_ngram_size=int(config.get("eval_no_repeat_ngram_size", 3) or 3),
                # Training-time eval: light metrics (no SPICE/BERTScore).
                compute_bertscore=False,
                light_metrics=True,
                suppress_eos_for_audio=suppress_eos_for_audio,
                eval_prompt=config.get("eval_prompt"),
                sample_output=val_samples,
                sample_limit=wandb_sample_count,
            )
            if bool(config.get("eval_ablate_audio", False)):
                val_metrics_no_audio = evaluate(
                    model,
                    val_loader,
                    device,
                    max_batches=config.get("max_eval_batches"),
                    max_new_tokens=config.get("max_new_tokens", 20),
                    num_beams=config.get("num_beams", 1),
                    repetition_penalty=float(config.get("eval_repetition_penalty", 1.1) or 1.1),
                    no_repeat_ngram_size=int(config.get("eval_no_repeat_ngram_size", 3) or 3),
                    compute_bertscore=False,
                    light_metrics=True,
                    suppress_eos_for_audio=suppress_eos_for_audio,
                    eval_prompt=config.get("eval_prompt"),
                    ablate_audio=True,
                    sample_output=None,
                    sample_limit=0,
                )
                if is_main:
                    print(
                        f"[Val A/B] CIDEr audio={val_metrics.get('cider', 0.0):.2f} "
                        f"no_audio={val_metrics_no_audio.get('cider', 0.0):.2f} "
                        f"delta={val_metrics.get('cider', 0.0) - val_metrics_no_audio.get('cider', 0.0):.2f}",
                        flush=True,
                    )
                if wandb_run is not None:
                    ablate_log = {"train/optimizer_step": optimizer_step, "epoch": epoch}
                    ablate_log.update({f"val_no_audio/{k}": v for k, v in val_metrics_no_audio.items() if isinstance(v, (int, float))})
                    ablate_log["val_delta/cider"] = float(val_metrics.get("cider", 0.0) - val_metrics_no_audio.get("cider", 0.0))
                    ablate_log["val_delta/meteor"] = float(val_metrics.get("meteor", 0.0) - val_metrics_no_audio.get("meteor", 0.0))
                    _wandb_log(wandb_run, ablate_log, step=optimizer_step)
            if val_samples and bool(config.get("export_eval_samples", False)):
                _export_eval_samples(
                    output_dir,
                    split="val",
                    epoch=epoch,
                    samples=val_samples,
                    export_audio=bool(config.get("export_eval_samples_audio", False)),
                )

            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_cider"].append(val_metrics["cider"])
            history["val_bleu4"].append(val_metrics["bleu4"])
            history["epochs"].append(epoch)

            # Save checkpoint (only on main process)
            is_best = val_metrics["cider"] > best_cider
            if is_best:
                best_cider = val_metrics["cider"]
                patience_counter = 0
                if is_main:
                    print(f"\n🎉 New best CIDEr: {best_cider:.2f}")
            else:
                patience_counter += 1

            # Only save checkpoints from main process
            if is_main:
                # For DDP, save the underlying model, not the wrapper
                model_to_save = base_model
                ckpt_paths = save_checkpoint(
                    model_to_save,
                    optimizer,
                    scheduler,
                    {**train_metrics, **val_metrics, "epoch": epoch},
                    output_dir,
                    is_best=is_best,
                    save_full_checkpoint=bool(config.get("save_full_checkpoint", False)),
                )

                # Save history
                with open(output_dir / "history.json", "w") as f:
                    json.dump(history, f, indent=2)

            # Sync processes after checkpoint save
            if dist_info["distributed"]:
                dist.barrier()

            if wandb_run is not None:
                val_log = {
                    "train/optimizer_step": optimizer_step,
                    "epoch": epoch,
                    "val/best_cider": float(best_cider),
                    "val/is_best": int(bool(is_best)),
                    "early_stopping/patience_counter": int(patience_counter),
                }
                val_log.update({f"val/{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})
                _wandb_log(wandb_run, val_log, step=optimizer_step)
                if val_samples and wandb is not None:
                    try:
                        table = wandb.Table(
                            columns=["sample_id", "audio", "question", "prediction", "references", "audio_path", "subset"]
                        )
                        audio_success_count = 0
                        audio_fail_count = 0
                        for row_idx, row in enumerate(val_samples):
                            audio_cell = None
                            audio_value = row.get("audio")
                            audio_path_value = row.get("audio_path")

                            if (
                                isinstance(audio_value, tuple)
                                and len(audio_value) == 2
                                and torch.is_tensor(audio_value[0])
                            ):
                                try:
                                    audio_cell = wandb.Audio(
                                        audio_value[0].detach().cpu().numpy(),
                                        sample_rate=int(audio_value[1]),
                                    )
                                    audio_success_count += 1
                                except Exception as e:
                                    if row_idx < 3:
                                        print(f"[W&B Audio Epoch] Failed to create from tensor: {e}", flush=True)
                                    audio_cell = None
                            elif audio_path_value:
                                try:
                                    audio_cell = wandb.Audio(str(audio_path_value))
                                    audio_success_count += 1
                                except Exception as e:
                                    if row_idx < 3:
                                        print(f"[W&B Audio Epoch] Failed to create from path: {e}", flush=True)
                                    audio_cell = None

                            if audio_cell is None:
                                audio_fail_count += 1

                            table.add_data(
                                row.get("sample_id"),
                                audio_cell,
                                row.get("question"),
                                row.get("prediction"),
                                "\n".join(row.get("references") or []),
                                row.get("audio_path"),
                                row.get("subset"),
                            )
                        print(f"[W&B Audio Epoch {epoch}] Created {audio_success_count}/{len(val_samples)} audio cells, {audio_fail_count} failed", flush=True)
                        _wandb_log(
                            wandb_run,
                            {"train/optimizer_step": optimizer_step, "val/samples": table},
                            step=optimizer_step,
                        )
                    except Exception as e:
                        print(f"[W&B Audio Epoch] Failed to create/log table: {e}", flush=True)
                if (
                    wandb_log_checkpoints
                    and is_best
                    and isinstance(ckpt_paths, dict)
                    and ckpt_paths.get("best") is not None
                ):
                    try:
                        run_id = getattr(wandb_run, "id", None) or "run"
                        _wandb_log_artifact(
                            wandb_run,
                            artifact_name=f"checkpoint-{run_id}",
                            artifact_type="model",
                            files=[ckpt_paths["best"]],
                            aliases=["best", f"epoch{epoch}"],
                            metadata={
                                "epoch": epoch,
                                "optimizer_step": optimizer_step,
                                "metrics": {**train_metrics, **val_metrics},
                                "output_dir": str(output_dir),
                            },
                        )
                    except Exception:
                        pass

            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered (patience={patience})")
                break

    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"  Best CIDEr: {best_cider:.2f}")
    print(f"  Checkpoints saved to: {output_dir}")
    print(f"{'='*80}\n")

    return history


# ============================================================================
# SECTION 5: CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(
    model: SAFEModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, float],
    output_dir: Path,
    is_best: bool = False,
    save_full_checkpoint: bool = False,
) -> Dict[str, Path]:
    """Save model checkpoint"""
    output_dir.mkdir(parents=True, exist_ok=True)

    def _trainable_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for name, param in module.named_parameters():
            if param.requires_grad:
                state[name] = param.detach().cpu()
        return state

    # Default to saving only trainable weights (SAFE Stage-A style). This avoids
    # multi-GB checkpoint writes when the base VL model is frozen.
    model_state_dict = model.state_dict() if save_full_checkpoint else _trainable_state_dict(model)

    checkpoint = {
        "format": "full" if save_full_checkpoint else "trainable_only",
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }

    # Save last checkpoint
    checkpoint_path = output_dir / "checkpoint_last.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Saved checkpoint: {checkpoint_path}")
    paths: Dict[str, Path] = {"last": checkpoint_path}

    # Save best checkpoint
    if is_best:
        best_path = output_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"💾 Saved BEST checkpoint: {best_path}")
        paths["best"] = best_path

    return paths


def load_checkpoint(
    model: SAFEModel,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    checkpoint_path: Path,
    device: torch.device,
    debug_keys: bool = False,
) -> Dict[str, float]:
    """Load model checkpoint"""
    print(f"📂 Loading checkpoint: {checkpoint_path}")

    # Always load checkpoints onto CPU first to avoid GPU memory spikes during deserialization.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Support both full checkpoints and adapter-only checkpoints.
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if state_dict is None and isinstance(checkpoint, dict):
        state_dict = checkpoint

    # Debug: print checkpoint keys vs model expected keys
    if debug_keys:
        print("\n[DEBUG] === CHECKPOINT STATE DICT KEYS ===")
        ckpt_safe_keys = [k for k in sorted(state_dict.keys())
                         if k.startswith(("audio_projector.", "fusion_adapter.", "audio_token_embeddings."))]
        for k in ckpt_safe_keys[:20]:
            print(f"  CKPT: {k}")
        if len(ckpt_safe_keys) > 20:
            print(f"  ... and {len(ckpt_safe_keys) - 20} more")

        print("\n[DEBUG] === MODEL EXPECTED KEYS (trainable) ===")
        model_safe_keys = [n for n, p in model.named_parameters() if p.requires_grad
                          and n.startswith(("audio_projector.", "fusion_adapter.", "audio_token_embeddings."))]
        for k in model_safe_keys[:20]:
            print(f"  MODEL: {k}")
        if len(model_safe_keys) > 20:
            print(f"  ... and {len(model_safe_keys) - 20} more")
        print()

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        relevant_missing = [
            key
            for key in missing_keys
            if key.startswith(("audio_projector.", "fusion_adapter.", "audio_token_embeddings."))
        ]
        if relevant_missing:
            print(
                f"⚠️  Missing {len(relevant_missing)} SAFE trainable keys in checkpoint "
                f"(showing up to 10): {relevant_missing[:10]}",
                flush=True,
            )
    if unexpected_keys:
        relevant_unexpected = [
            key
            for key in unexpected_keys
            if key.startswith(("audio_projector.", "fusion_adapter.", "audio_token_embeddings."))
        ]
        if relevant_unexpected:
            print(
                f"⚠️  Unexpected {len(relevant_unexpected)} SAFE keys in checkpoint "
                f"(showing up to 10): {relevant_unexpected[:10]}",
                flush=True,
            )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    metrics = checkpoint.get("metrics", {})
    print(f"✓ Checkpoint loaded (epoch={metrics.get('epoch', 'unknown')})")

    return metrics


# ============================================================================
# SECTION 6: MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SAFE model on audio captioning")

    # Model configuration
    parser.add_argument("--model-config", type=str, default="phase1",
                        choices=["demo", "full", "multimodal", "phase1", "kv_augment"],
                        help="Model configuration name")
    parser.add_argument("--fusion-layer-indices", type=str, default=None,
                        help="Comma-separated layer indices for fusion injection (e.g., '8,16,24'). "
                             "Overrides the config default.")
    parser.add_argument("--lora-rank", type=int, default=None,
                        help="LoRA rank for fusion adapter (e.g., 4, 8, 16). "
                             "Overrides the config default.")
    parser.add_argument("--label-smoothing", type=float, default=None,
                        help="Label smoothing factor for cross-entropy loss (e.g., 0.1). "
                             "Overrides the config default.")

    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--train-split", type=str, default="train",
                        help="Training split name")
    parser.add_argument("--val-split", type=str, default="val",
                        help="Validation split name")
    parser.add_argument("--use-wavcaps", action="store_true",
                        help="Include WavCaps in training mix")
    parser.add_argument("--wavcaps-ratio", type=float, default=0.8,
                        help="Fraction of WavCaps train samples to include (0.0-1.0)")
    parser.add_argument("--wavcaps-split", type=str, default="train",
                        help="WavCaps split to use for training")
    parser.add_argument(
        "--resample-wavcaps-each-epoch",
        action="store_true",
        help="Re-sample the WavCaps subset each epoch (useful when wavcaps_ratio < 1.0).",
    )
    parser.add_argument("--use-clotho", action="store_true",
                        help="Include Clotho dataset in training mix")
    parser.add_argument("--use-macs", action="store_true",
                        help="Include MACS dataset in training mix")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit number of training samples for smoke testing")

    # Training
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--num-epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=8,
                        help="Validation batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32,
                        help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")

    # Optimization
    # Match Phase 1 defaults from README: 1e-3 / 5e-4
    parser.add_argument("--learning-rate-projector", type=float, default=1e-3,
                        help="Learning rate for audio projector")
    parser.add_argument("--learning-rate-adapter", type=float, default=5e-4,
                        help="Learning rate for fusion adapter")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=2000,
                        help="Warmup steps for learning rate")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                        help="Minimum LR as ratio of base LR (prevents plateau, default 0.1 = 10%%)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")

    # Optional audio contrastive loss (Stage-A style)
    parser.add_argument("--audio-contrastive-weight", type=float, default=0.0,
                        help="Weight for optional audio-text contrastive loss (0.0 = disabled)")
    parser.add_argument("--audio-contrastive-temperature", type=float, default=0.07,
                        help="Temperature for audio-text contrastive loss")
    parser.add_argument("--audio-contrastive-max-length", type=int, default=48,
                        help="Max caption length (tokens) for contrastive text embeddings")
    parser.add_argument("--gate-warmup-steps", type=int, default=0,
                        help="If >0, ramp SAFE gate 0→1 over this many optimizer steps")
    parser.add_argument("--residual-scale-warmup-steps", type=int, default=0,
                        help="If >0, ramp residual scale from start to end over this many optimizer steps")
    parser.add_argument("--residual-scale-warmup-start", type=float, default=0.2,
                        help="Starting residual scale for warmup (default 0.2 = 20%% audio)")
    parser.add_argument("--residual-scale-warmup-end", type=float, default=1.0,
                        help="Ending residual scale for warmup (default 1.0 = 100%% audio)")
    parser.add_argument(
        "--ablation-loss-weight",
        type=float,
        default=0.0,
        help="Optional forcing term: weight for hinge loss encouraging audio to reduce LM loss vs ablated-audio baseline.",
    )
    parser.add_argument(
        "--ablation-loss-margin",
        type=float,
        default=0.0,
        help="Margin for ablation hinge: target improvement (loss_no_audio - loss_audio) >= margin.",
    )
    parser.add_argument(
        "--ablation-loss-every-steps",
        type=int,
        default=1,
        help="Apply ablation forcing loss every N optimizer steps (1 = every step).",
    )

    # Audio augmentation (off by default)
    parser.add_argument("--audio-augment", action="store_true",
                        help="Enable audio augmentation (SpecAugment + waveform augment) during training")
    parser.add_argument("--audio-augment-prob", type=float, default=0.5,
                        help="Probability of applying audio augmentation per sample")

    # Evaluation
    parser.add_argument("--eval-frequency", type=int, default=1,
                        help="Evaluate every N epochs")
    parser.add_argument("--train-eval-steps", type=int, default=500,
                        help="Compute CIDEr/METEOR on training subset every N optimizer steps (0=disabled)")
    parser.add_argument("--train-eval-samples", type=int, default=50,
                        help="Number of training samples to use for training accuracy eval")
    parser.add_argument(
        "--train-eval-split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to use for periodic train-eval metrics (val is multi-ref for AudioCaps).",
    )
    parser.add_argument(
        "--eval-ablate-audio",
        action="store_true",
        help="Also run evaluation with audio disabled (A/B diagnostic).",
    )
    parser.add_argument(
        "--train-eval-ablate-audio",
        action="store_true",
        help="Also run periodic train-eval with audio disabled (A/B diagnostic; can be expensive).",
    )
    parser.add_argument(
        "--train-eval-ablate-max-batches",
        type=int,
        default=10,
        help="Max batches for ablated-audio train-eval (keeps A/B diagnostic cheap).",
    )
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="Max batches for validation (None = all)")
    parser.add_argument("--max-new-tokens", type=int, default=20,
                        help="Max new tokens for generation")
    parser.add_argument("--num-beams", type=int, default=1,
                        help="Beam search size")
    parser.add_argument(
        "--eval-repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty used during evaluation generation.",
    )
    parser.add_argument(
        "--eval-no-repeat-ngram-size",
        type=int,
        default=3,
        help="No-repeat ngram size used during evaluation generation.",
    )
    parser.add_argument(
        "--suppress-eos-for-audio-early-steps",
        type=int,
        default=0,
        help=(
            "Suppress EOS/PAD during audio generation for the first N optimizer steps "
            "(helps avoid empty captions early; 0 disables)."
        ),
    )

    # Checkpointing
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only (requires --resume)")
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Early stopping patience (epochs)")
    parser.add_argument(
        "--save-full-checkpoint",
        action="store_true",
        help=(
            "Save full model weights in checkpoints (very large for LLaVA/BLIP2). "
            "Default saves only trainable SAFE components."
        ),
    )

    # Memory optimization
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory (trades compute for memory)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")

    # Weights & Biases logging
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (defaults to $WANDB_PROJECT or 'SAFE')")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity (user/team).")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="W&B run name (defaults to output dir name).")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="W&B group (defaults to $SLURM_JOB_ID when set).")
    parser.add_argument("--wandb-tags", type=str, default=None,
                        help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-mode", type=str, default=None,
                        choices=["online", "offline", "disabled"],
                        help="W&B mode override. If unset, defaults to offline when WANDB_API_KEY is missing.")
    parser.add_argument("--wandb-dir", type=str, default=None,
                        help="Directory for W&B run files (recommended on clusters, e.g. $SCRATCH/wandb).")
    parser.add_argument("--wandb-notes", type=str, default=None,
                        help="Optional W&B notes.")
    parser.add_argument("--wandb-log-code", action="store_true",
                        help="Log a code snapshot to W&B (can be slow/large).")
    parser.add_argument("--wandb-watch", type=str, default="false",
                        choices=["false", "gradients", "parameters", "all"],
                        help="Enable wandb.watch (expensive).")
    parser.add_argument("--wandb-watch-log-freq", type=int, default=500,
                        help="wandb.watch log frequency.")
    parser.add_argument("--wandb-log-checkpoints", action="store_true",
                        help="Log best checkpoints as W&B artifacts.")
    parser.add_argument("--wandb-sample-count", type=int, default=30,
                        help="How many sample predictions to log per evaluation.")
    parser.add_argument(
        "--export-eval-samples",
        action="store_true",
        help="Export qualitative eval samples to output_dir/eval_samples (JSON, and optional WAV).",
    )
    parser.add_argument(
        "--export-eval-samples-audio",
        action="store_true",
        help="When exporting eval samples, also write WAV files (for small sample_count).",
    )

    args = parser.parse_args()

    # Setup distributed training
    dist_info = setup_distributed()
    is_main = is_main_process(dist_info)

    # Set seed (offset by rank for different data ordering per GPU)
    set_seed(args.seed, rank=dist_info["rank"])

    # Device - use local_rank for multi-GPU
    if dist_info["distributed"]:
        device = torch.device(f"cuda:{dist_info['local_rank']}")
        if is_main:
            print(f"🚀 Distributed training: {dist_info['world_size']} GPUs")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if is_main:
        print(f"Using device: {device}")

    # Create output directory (only on main process)
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Sync all processes before continuing
    if dist_info["distributed"]:
        dist.barrier()

    # Save args (only on main process)
    if is_main:
        with open(output_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # Load model config
    if is_main:
        print(f"\nLoading model config: {args.model_config}")
    model_config = get_config(args.model_config)

    # Override fusion layer indices if specified via CLI
    if args.fusion_layer_indices:
        layer_indices = [int(x.strip()) for x in args.fusion_layer_indices.split(",")]
        model_config["fusion_layer_indices"] = layer_indices
        # Also update nested fusion_adapter config if present
        if "fusion_adapter" in model_config and isinstance(model_config["fusion_adapter"], dict):
            model_config["fusion_adapter"]["layer_indices"] = layer_indices
        # CRITICAL: Also update fusion_config.modalities.audio.layer_indices
        # This is what MultiLayerFusionAdapter actually uses when modalities is provided
        if "fusion_config" in model_config and isinstance(model_config["fusion_config"], dict):
            fusion_cfg = model_config["fusion_config"]
            if "modalities" in fusion_cfg and isinstance(fusion_cfg["modalities"], dict):
                for modality_name, modality_cfg in fusion_cfg["modalities"].items():
                    if isinstance(modality_cfg, dict):
                        modality_cfg["layer_indices"] = layer_indices
                if is_main:
                    print(f"  ✓ Updated modalities config with layer indices: {layer_indices}")
        if is_main:
            print(f"  ✓ Fusion layer indices overridden: {layer_indices}")

    # Override LoRA rank if specified via CLI
    if args.lora_rank is not None:
        model_config["lora_rank"] = args.lora_rank
        if is_main:
            print(f"  ✓ LoRA rank overridden: {args.lora_rank}")

    # Override label smoothing if specified via CLI
    if args.label_smoothing is not None:
        model_config["label_smoothing"] = args.label_smoothing
        if is_main:
            print(f"  ✓ Label smoothing overridden: {args.label_smoothing}")

    # Initialize model using the canonical create_model helper
    model = create_model(model_config) if is_main else create_model(model_config)
    model = model.to(device)

    # If the base model is fp16 (common), LoRA/adapters may also be fp16.
    # GradScaler cannot unscale fp16 gradients, so keep trainable params fp32.
    converted = cast_trainable_params_to_fp32(model)
    if is_main and converted:
        print(f"  ✓ Cast {converted} trainable parameter tensors to fp32 (AMP-safe)", flush=True)

    # Wrap model with DDP for distributed training
    if dist_info["distributed"]:
        # DDP configuration for SAFE:
        # - find_unused_parameters=False for performance (we know our param usage is consistent)
        # - static_graph=True enables optimizations since our computation graph doesn't change
        # - gradient_as_bucket_view=True reduces memory copies
        model = DDP(
            model,
            device_ids=[dist_info["local_rank"]],
            output_device=dist_info["local_rank"],
            find_unused_parameters=False,
            static_graph=True,
            gradient_as_bucket_view=True,
        )
        if is_main:
            print(f"  ✓ Model wrapped with DistributedDataParallel (static_graph=True)")

    # Enable gradient checkpointing if requested (saves ~10-15GB memory)
    # Access the underlying model if wrapped in DDP
    base_model = model.module if dist_info["distributed"] else model
    if args.gradient_checkpointing:
        if is_main:
            print(f"  Enabling gradient checkpointing for memory optimization...")
        if hasattr(base_model.base_vl, 'llm') and hasattr(base_model.base_vl.llm, 'gradient_checkpointing_enable'):
            base_model.base_vl.llm.gradient_checkpointing_enable()
            if is_main:
                print(f"  ✓ Gradient checkpointing enabled on LLM")
        else:
            if is_main:
                print(f"  ⚠️  LLM does not support gradient checkpointing")

    # Count parameters (use base_model for accurate count)
    total_params, trainable_params = count_parameters(base_model)
    if is_main:
        print(f"\n📊 Model parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        breakdown = summarize_trainable_parameters(base_model)
        if breakdown:
            print("  Trainable breakdown:")
            for key in sorted(breakdown.keys()):
                print(f"    - {key}: {_format_param_count(breakdown[key])}")

        # Check for LoRA or KV adapter parameters and warn if missing
        lora_params = breakdown.get("fusion_adapter/lora", 0)
        kv_adapter_params = breakdown.get("kv_adapter", 0)
        query_adapter_params = breakdown.get("kv_adapter/query_adapter", 0)

        if kv_adapter_params > 0:
            print(f"\n  ✓ KV Adapter parameters detected: {_format_param_count(kv_adapter_params)} (training enabled)")
            if query_adapter_params > 0:
                print(f"    - Query adapter (ΔQ): {_format_param_count(query_adapter_params)}")
        elif lora_params > 0:
            print(f"\n  ✓ LoRA parameters detected: {_format_param_count(lora_params)} (training enabled)")
        else:
            print(f"\n  ⚠️  WARNING: No LoRA or KV adapter parameters found in trainable params!")
            print(f"     This may indicate adapter weights are frozen")
            print(f"     Expected: ~10-15M LoRA params (cross-attention) or ~1-5M KV adapter params")

    # Load datasets
    if is_main:
        print(f"\n📂 Loading datasets from: {args.data_path}")
    audiocaps_train = AudioCapsDataset(args.data_path, split=args.train_split)
    if args.max_train_samples is not None:
        if is_main:
            print(f"  ⚠️  Limiting training samples to {args.max_train_samples} for smoke testing")
        indices = list(range(min(len(audiocaps_train), args.max_train_samples)))
        audiocaps_train = torch.utils.data.Subset(audiocaps_train, indices)

    val_dataset = AudioCapsDataset(args.data_path, split=args.val_split)

    # Collect additional datasets
    additional_datasets = []

    # WavCaps
    wavcaps_train = None
    if args.use_wavcaps:
        try:
            wavcaps_train = WavCapsDataset(args.data_path, split=args.wavcaps_split)
            if is_main:
                print(f"  WavCaps train: {len(wavcaps_train)} samples (split='{args.wavcaps_split}')")
                print(f"  WavCaps ratio: {args.wavcaps_ratio:.2f}")
        except Exception as e:
            if is_main:
                print(f"⚠️  Failed to load WavCaps dataset: {e}. Continuing without WavCaps.", flush=True)
            wavcaps_train = None

    # Clotho
    clotho_train = None
    if args.use_clotho:
        try:
            clotho_train = ClothoDataset(args.data_path, split="train")
            additional_datasets.append(clotho_train)
            if is_main:
                print(f"  Clotho train: {len(clotho_train)} samples")
        except Exception as e:
            if is_main:
                print(f"⚠️  Failed to load Clotho dataset: {e}. Continuing without Clotho.", flush=True)
            clotho_train = None

    # MACS
    macs_train = None
    if args.use_macs:
        try:
            macs_train = MACSDataset(args.data_path, split="train")
            additional_datasets.append(macs_train)
            if is_main:
                print(f"  MACS train: {len(macs_train)} samples")
        except Exception as e:
            if is_main:
                print(f"⚠️  Failed to load MACS dataset: {e}. Continuing without MACS.", flush=True)
            macs_train = None

    # Combine all datasets
    if wavcaps_train is not None:
        train_dataset = MixedAudioCaptionDataset(
            audiocaps_dataset=audiocaps_train,
            wavcaps_dataset=wavcaps_train,
            wavcaps_ratio=args.wavcaps_ratio,
            shuffle=True,
            seed=args.seed,
            resample_wavcaps_each_epoch=bool(args.resample_wavcaps_each_epoch),
        )
        if is_main:
            print(f"  AudioCaps train: {len(audiocaps_train)} samples")
            print(f"  Mixed train samples: {len(train_dataset)} (AudioCaps + WavCaps subset)")
    else:
        train_dataset = audiocaps_train
        if is_main:
            print(f"  Train (AudioCaps only): {len(train_dataset)} samples")

    # Add Clotho and MACS via ConcatDataset if available
    if additional_datasets:
        all_datasets = [train_dataset] + additional_datasets
        train_dataset = torch.utils.data.ConcatDataset(all_datasets)
        if is_main:
            total_samples = sum(len(d) for d in all_datasets)
            dataset_names = ["AudioCaps/WavCaps" if wavcaps_train else "AudioCaps"]
            if clotho_train:
                dataset_names.append("Clotho")
            if macs_train:
                dataset_names.append("MACS")
            print(f"  Combined train: {total_samples} samples ({' + '.join(dataset_names)})")

    if is_main:
        print(f"  Val (AudioCaps): {len(val_dataset)} samples")

    # Create dataloaders with DistributedSampler for multi-GPU
    train_sampler = None
    val_sampler = None
    if dist_info["distributed"]:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist_info["world_size"],
            rank=dist_info["rank"],
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist_info["world_size"],
            rank=dist_info["rank"],
            shuffle=False,
        )

    train_loader = create_safe_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # Don't shuffle if using DistributedSampler
        num_workers=args.num_workers,
        sampler=train_sampler,
    )

    val_loader = create_safe_dataloader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=val_sampler,
    )

    # Create training accuracy eval loader (small fixed subset of training data)
    train_eval_loader = None
    if args.train_eval_steps > 0 and args.train_eval_samples > 0:
        # Use a fixed subset for consistent measurement.
        # AudioCaps train samples typically have 1 caption each; AudioCaps val is multi-ref and yields
        # much more meaningful CIDEr trends, so default to val.
        train_eval_source = val_dataset if args.train_eval_split == "val" else audiocaps_train
        train_eval_indices = list(range(min(len(train_eval_source), args.train_eval_samples)))
        train_eval_subset = torch.utils.data.Subset(train_eval_source, train_eval_indices)
        train_eval_loader = create_safe_dataloader(
            train_eval_subset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=0,  # Keep it lightweight
        )
        if is_main:
            print(
                f"  Train eval ({args.train_eval_split}): {len(train_eval_subset)} samples "
                f"(every {args.train_eval_steps} steps)"
            )
            # Helpful for diagnosing "metrics jump then die" due to eval-set changes.
            try:
                preview_indices = train_eval_indices[:5]
                print(f"  Train eval indices[:5]: {preview_indices}", flush=True)
            except Exception:
                pass

    # Training config
    resolved_max_eval_batches = args.max_eval_batches
    if resolved_max_eval_batches is None:
        resolved_max_eval_batches = 50  # Keep default eval fast unless explicitly overridden
    elif int(resolved_max_eval_batches) <= 0:
        resolved_max_eval_batches = None  # Evaluate full validation set

    config = {
        "num_epochs": args.num_epochs,
        "learning_rate_projector": args.learning_rate_projector,
        "learning_rate_adapter": args.learning_rate_adapter,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "max_grad_norm": args.max_grad_norm,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "fp16": args.fp16,
        "eval_frequency": args.eval_frequency,
        "max_eval_batches": resolved_max_eval_batches,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "eval_repetition_penalty": args.eval_repetition_penalty,
        "eval_no_repeat_ngram_size": args.eval_no_repeat_ngram_size,
        "suppress_eos_for_audio_early_steps": args.suppress_eos_for_audio_early_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "save_full_checkpoint": args.save_full_checkpoint,
        "audio_contrastive_weight": args.audio_contrastive_weight,
        "audio_contrastive_temperature": args.audio_contrastive_temperature,
        "audio_contrastive_max_length": args.audio_contrastive_max_length,
        "gate_warmup_steps": args.gate_warmup_steps,
        "residual_scale_warmup_steps": args.residual_scale_warmup_steps,
        "residual_scale_warmup_start": args.residual_scale_warmup_start,
        "residual_scale_warmup_end": args.residual_scale_warmup_end,
        "ablation_loss_weight": args.ablation_loss_weight,
        "ablation_loss_margin": args.ablation_loss_margin,
        "ablation_loss_every_steps": args.ablation_loss_every_steps,
        "audio_augment": args.audio_augment,
        "audio_augment_prob": args.audio_augment_prob,
        "export_eval_samples": bool(args.export_eval_samples),
        "export_eval_samples_audio": bool(args.export_eval_samples_audio),
    }

    # Include model-side settings needed by training/eval helpers.
    # (train_safe uses `config` in several places for evaluation prompt override and
    # min-audio-attention scheduling, while the model constructor uses `model_config`.)
    config["eval_prompt"] = model_config.get("eval_prompt")
    config["fusion_config"] = model_config.get("fusion_config", {})
    config["eval_ablate_audio"] = bool(args.eval_ablate_audio)
    config["train_eval_ablate_audio"] = bool(args.train_eval_ablate_audio)
    config["train_eval_ablate_max_batches"] = args.train_eval_ablate_max_batches

    # Optional W&B init (after model + data are available so config is complete)
    wandb_run = _maybe_init_wandb(
        args,
        train_config=config,
        model_config=model_config,
        output_dir=output_dir,
        model=model,
        total_params=total_params,
        trainable_params=trainable_params,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
    )

    # Evaluation only mode
    if args.eval_only:
        if args.resume is None:
            raise ValueError("--eval-only requires --resume")

        load_checkpoint(model, None, None, Path(args.resume), device)

        print(f"\n{'='*80}")
        print(f"Running evaluation only")
        print(f"{'='*80}\n")

        eval_samples: Optional[List[Dict[str, Any]]] = (
            [] if (wandb_run is not None and args.wandb_sample_count > 0) else None
        )
        metrics = evaluate(
            model,
            val_loader,
            device,
            max_batches=args.max_eval_batches,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            compute_bertscore=True,   # Full metrics in eval-only mode
            light_metrics=False,
            suppress_eos_for_audio=False,
            eval_prompt=config.get("eval_prompt"),
            sample_output=eval_samples,
            sample_limit=args.wandb_sample_count,
        )

        # Save results
        results_path = output_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

        if eval_samples and bool(config.get("export_eval_samples", False)):
            _export_eval_samples(
                output_dir,
                split="eval",
                epoch=0,
                samples=eval_samples,
                export_audio=bool(config.get("export_eval_samples_audio", False)),
            )

        if wandb_run is not None:
            eval_log = {"train/optimizer_step": 0, "epoch": 0}
            eval_log.update({f"eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
            _wandb_log(wandb_run, eval_log, step=0)
            if eval_samples and wandb is not None:
                try:
                    table = wandb.Table(
                        columns=["sample_id", "audio", "question", "prediction", "references", "audio_path", "subset"]
                    )
                    for row in eval_samples:
                        audio_cell = None
                        audio_value = row.get("audio")
                        if (
                            isinstance(audio_value, tuple)
                            and len(audio_value) == 2
                            and torch.is_tensor(audio_value[0])
                        ):
                            try:
                                audio_cell = wandb.Audio(
                                    audio_value[0].detach().cpu().numpy(),
                                    sample_rate=int(audio_value[1]),
                                )
                            except Exception:
                                audio_cell = None
                        elif row.get("audio_path"):
                            try:
                                audio_cell = wandb.Audio(str(row.get("audio_path")))
                            except Exception:
                                audio_cell = None

                        table.add_data(
                            row.get("sample_id"),
                            audio_cell,
                            row.get("question"),
                            row.get("prediction"),
                            "\n".join(row.get("references") or []),
                            row.get("audio_path"),
                            row.get("subset"),
                        )
                    _wandb_log(wandb_run, {"train/optimizer_step": 0, "eval/samples": table}, step=0)
                except Exception:
                    pass
            try:
                wandb_run.finish()
            except Exception:
                pass

        print(f"\n💾 Results saved to: {results_path}")
        return

    # Resume from checkpoint if specified
    if args.resume:
        load_checkpoint(model, None, None, Path(args.resume), device)

    # Train
    train(
        model,
        train_loader,
        val_loader,
        config,
        output_dir,
        device,
        wandb_run=wandb_run,
        wandb_log_checkpoints=bool(args.wandb_log_checkpoints),
        wandb_sample_count=int(args.wandb_sample_count),
        dist_info=dist_info,
        train_sampler=train_sampler,
        train_eval_loader=train_eval_loader,
        train_eval_steps=args.train_eval_steps,
    )

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    # Cleanup distributed training
    cleanup_distributed()

    if is_main:
        print(f"\n✅ Training complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
