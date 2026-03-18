#!/usr/bin/env python3
"""
Train/evaluate SAFE on AVQA/MUSIC-AVQA composition QA.

This is a Pre-FFN-first experiment runner for ECCV composition work.
It expects train/val manifests in JSONL where each row has:
  - question
  - answer
  - audio_path (optional for image-only ablations)
  - image_path (optional for audio-only ablations)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset
# tqdm removed — use explicit print logging for clean stdout/stderr separation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.model_configs import get_config
from safe.models.safe_model import SAFEModel

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    # Strip common LLM preamble patterns
    for prefix in ("the answer is", "answer:", "a:", "it is", "this is"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.strip().rstrip(".")
    return " ".join(text.split())


DEFAULT_TEXT_PROMPT_PREFIX = (
    "Use only the question text and general world knowledge. "
    "Do not assume access to audio or visual evidence."
)
DEFAULT_AUDIO_PROMPT_PREFIX = (
    "Use only auditory evidence from the provided audio. "
    "Ignore visual priors and answer from what you hear, such as timbre, pitch, rhythm, or source sound."
)
DEFAULT_IMAGE_PROMPT_PREFIX = (
    "Use only visual evidence from the provided image. "
    "Ignore audio priors and answer from what you see, such as object appearance, motion, count, or scene cues."
)
DEFAULT_BOTH_PROMPT_PREFIX = (
    "Combine auditory and visual evidence. "
    "Use sound for audio properties, use vision for visual properties, and resolve conflicts by requiring consistency across both modalities."
)


def build_modality_aware_questions(
    questions: Union[str, Sequence[str]],
    modality: str,
    args: argparse.Namespace,
) -> Union[str, List[str]]:
    if not getattr(args, "modality_aware_prompts", False):
        return questions

    if isinstance(questions, str):
        q_list = [questions]
        single = True
    else:
        q_list = list(questions)
        single = False

    modality_key = str(modality).strip().lower()
    if modality_key == "audio":
        prefix = str(getattr(args, "audio_prompt_prefix", DEFAULT_AUDIO_PROMPT_PREFIX))
    elif modality_key == "image":
        prefix = str(getattr(args, "image_prompt_prefix", DEFAULT_IMAGE_PROMPT_PREFIX))
    elif modality_key == "both":
        prefix = str(getattr(args, "both_prompt_prefix", DEFAULT_BOTH_PROMPT_PREFIX))
    else:
        prefix = str(getattr(args, "text_prompt_prefix", DEFAULT_TEXT_PROMPT_PREFIX))

    prompted = [f"{prefix}\n{q}" for q in q_list]
    if single:
        return prompted[0]
    return prompted


# Known MUSIC-AVQA answer vocabulary (from official dataset)
AVQA_ANSWER_VOCAB = [
    "yes", "no",
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "left", "right",
    "accordion", "acoustic_guitar", "bagpipe", "banjo", "bassoon",
    "cello", "clarinet", "congas", "drum", "electric_bass", "erhu",
    "flute", "guzheng", "piano", "pipa", "saxophone", "trumpet",
    "tuba", "ukulele", "violin", "xylophone",
]

# Aliases: common LLM outputs that map to canonical answers
_ANSWER_ALIASES = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven",
    "guitar": "acoustic_guitar", "electric guitar": "electric_bass",
    "drums": "drum", "conga": "congas",
    "acoustic guitar": "acoustic_guitar",
    "electric bass": "electric_bass",
    "sax": "saxophone",
    "ukelele": "ukulele",
}


def extract_answer(raw_pred: str, vocab: List[str] = AVQA_ANSWER_VOCAB) -> str:
    """Extract the best matching answer from generated text against known vocabulary."""
    norm = normalize_answer(raw_pred)
    # Exact match first
    if norm in vocab:
        return norm
    # Check aliases
    if norm in _ANSWER_ALIASES:
        return _ANSWER_ALIASES[norm]
    # Check if any vocab answer appears as substring in the prediction
    # Prefer longer matches (e.g., "acoustic_guitar" over "guitar")
    found = []
    for ans in vocab:
        # Match with underscores replaced by spaces too
        ans_space = ans.replace("_", " ")
        if ans in norm or ans_space in norm:
            found.append(ans)
    if found:
        return max(found, key=len)
    # Check aliases as substrings
    for alias, canonical in _ANSWER_ALIASES.items():
        if alias in norm:
            return canonical
    # No match — return normalized prediction as-is
    return norm


def token_f1(pred: str, ref: str) -> float:
    p = normalize_answer(pred).split()
    r = normalize_answer(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    common = 0
    r_counts: Dict[str, int] = defaultdict(int)
    for t in r:
        r_counts[t] += 1
    for t in p:
        if r_counts[t] > 0:
            common += 1
            r_counts[t] -= 1
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(r)
    return 2 * precision * recall / (precision + recall)


def categorical_f1(pred: str, ref: str) -> float:
    """
    F1 over canonicalized categorical answers (useful for AVQA-style vocab answers).
    """
    p = extract_answer(pred)
    r = extract_answer(ref)
    return 1.0 if p == r else 0.0


class ManifestAVQADataset(Dataset):
    def __init__(self, manifest_path: Path, media_root: Path):
        self.manifest_path = manifest_path
        self.media_root = media_root
        self.rows: List[Dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        self.media_stats = self._compute_media_stats(max_samples=512)

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_media_path(self, path_value: str) -> Optional[Path]:
        if not path_value:
            return None
        p = Path(path_value).expanduser()
        # Try absolute path as-is
        if p.is_absolute():
            if p.exists():
                return p
            return None
        # Try relative to media_root
        candidate = self.media_root / p
        if candidate.exists():
            return candidate
        # Try relative to repo root (common when manifests store repo-relative paths)
        repo_candidate = REPO_ROOT / p
        if repo_candidate.exists():
            return repo_candidate
        # Try just the filename under common AVQA/MUSIC-AVQA subdirs.
        stem = p.stem
        for subdir in ("audio", "audio_old", "frames", "image", "image_31", "images"):
            for ext in (".wav", ".mp3", ".m4a", ".flac", ".jpg", ".jpeg", ".png"):
                c = self.media_root / subdir / f"{stem}{ext}"
                if c.exists():
                    return c
        # Try recursive stem search as last resort for heterogeneous layouts.
        for subdir in ("audio", "audio_old", "frames", "image", "image_31", "images"):
            base = self.media_root / subdir
            if not base.exists():
                continue
            for ext in (".wav", ".mp3", ".m4a", ".flac", ".jpg", ".jpeg", ".png"):
                matches = list(base.rglob(f"{stem}{ext}"))
                if matches:
                    return matches[0]
        return None

    def _compute_media_stats(self, max_samples: int = 512) -> Dict[str, int]:
        """
        Lightweight sanity stats to verify manifest/media alignment before training.
        Uses path resolution only (no audio/image decoding).
        """
        n = min(len(self.rows), max_samples)
        audio_ok = 0
        image_ok = 0
        both_ok = 0
        for i in range(n):
            row = self.rows[i]
            has_audio = self._resolve_media_path(row.get("audio_path", "")) is not None
            has_image = self._resolve_media_path(row.get("image_path", "")) is not None
            if has_audio:
                audio_ok += 1
            if has_image:
                image_ok += 1
            if has_audio and has_image:
                both_ok += 1
        return {
            "checked_samples": n,
            "audio_paths_found": audio_ok,
            "image_paths_found": image_ok,
            "both_paths_found": both_ok,
        }

    def _load_image(self, path_value: str) -> Optional[Image.Image]:
        image_path = self._resolve_media_path(path_value)
        if image_path is None:
            return None
        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            return None

    def _load_audio(self, path_value: str) -> Optional[Sequence[Any]]:
        audio_path = self._resolve_media_path(path_value)
        if audio_path is None:
            return None
        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(audio_path))
            if waveform.dim() == 2 and waveform.size(0) > 1:
                waveform = waveform.mean(dim=0)
            elif waveform.dim() == 2:
                waveform = waveform.squeeze(0)

            target_sr = 48000
            if sr != target_sr:
                waveform = torchaudio.functional.resample(
                    waveform.unsqueeze(0), sr, target_sr
                ).squeeze(0)
                sr = target_sr
            return (waveform, sr)
        except Exception:
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        return {
            "sample_id": row.get("sample_id", f"sample_{idx}"),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "question_type": row.get("question_type", "unknown"),
            "audio": self._load_audio(row.get("audio_path", "")),
            "image": self._load_image(row.get("image_path", "")),
        }


def collate_avqa(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_ids": [],
        "questions": [],
        "answers": [],
        "question_types": [],
        "audio": [],
        "images": [],
    }
    for item in batch:
        out["sample_ids"].append(item["sample_id"])
        out["questions"].append(item["question"])
        out["answers"].append(item["answer"])
        out["question_types"].append(item["question_type"])
        out["audio"].append(item["audio"])
        out["images"].append(item["image"])
    return out


def build_model_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = get_config(args.model_config)
    if args.llm_model:
        cfg["llm_model_name"] = args.llm_model

    fusion_cfg = dict(cfg.get("fusion_config", {}))

    cfg["num_audio_tokens"] = args.num_audio_tokens
    if cfg.get("fusion_type") != "concat":
        cfg["fusion_type"] = "multilayer"  # Required for mid-layer hook-based Pre-FFN fusion
    if args.fusion_layers is not None:
        layer_indices = [int(x) for x in args.fusion_layers.split(",") if x.strip()]
        cfg["fusion_layer_indices"] = layer_indices
        # Keep modality-specific fusion layers in sync when config provides them.
        # MultiLayerFusionAdapter prioritizes fusion_config.modalities over
        # top-level fusion_layer_indices.
        modalities = fusion_cfg.get("modalities")
        if isinstance(modalities, dict):
            for modality, mcfg in modalities.items():
                if isinstance(mcfg, dict):
                    mcfg["layer_indices"] = list(layer_indices)
                elif isinstance(mcfg, (list, tuple)):
                    modalities[modality] = {"layer_indices": list(layer_indices)}
    # Optional modality-specific fusion layer overrides (for layer pruning).
    # These take precedence over --fusion-layers when provided.
    modalities = fusion_cfg.get("modalities")
    if (getattr(args, "audio_fusion_layers", None) or getattr(args, "vision_fusion_layers", None)) and not isinstance(modalities, dict):
        modalities = {}
        fusion_cfg["modalities"] = modalities
    if isinstance(modalities, dict):
        if getattr(args, "audio_fusion_layers", None):
            audio_layers = [int(x) for x in args.audio_fusion_layers.split(",") if x.strip()]
            if "audio" not in modalities or not isinstance(modalities.get("audio"), dict):
                modalities["audio"] = {}
            modalities["audio"]["layer_indices"] = list(audio_layers)
        if getattr(args, "vision_fusion_layers", None):
            vision_layers = [int(x) for x in args.vision_fusion_layers.split(",") if x.strip()]
            if "vision" not in modalities or not isinstance(modalities.get("vision"), dict):
                modalities["vision"] = {}
            modalities["vision"]["layer_indices"] = list(vision_layers)
    cfg["freeze_base_vl"] = True
    cfg["freeze_audio_encoder"] = args.freeze_audio_encoder
    if args.label_smoothing is not None:
        cfg["label_smoothing"] = args.label_smoothing

    if cfg.get("fusion_type") != "concat":
        fusion_mode = str(fusion_cfg.get("fusion_mode", "residual")).strip().lower()
        if fusion_mode not in {"residual", "film", "affine", "fixed_point", "kv_augment"}:
            fusion_mode = "residual"
        fusion_cfg["fusion_mode"] = fusion_mode
        if fusion_mode != "kv_augment":
            fusion_cfg["injection_point"] = "pre_ffn"
            fusion_cfg.setdefault("use_bottleneck", True)
            if getattr(args, "bottleneck_dim", None) is not None:
                fusion_cfg["bottleneck_dim"] = args.bottleneck_dim
            else:
                fusion_cfg.setdefault("bottleneck_dim", 256)
        # Per-layer learned gating
        if getattr(args, "learned_gate", False):
            fusion_cfg["use_learned_gate"] = True
            fusion_cfg["learned_gate_init"] = getattr(args, "learned_gate_init", 0.0)
        # Cross-layer transport mitigation controls.
        fusion_cfg["delta_norm_cap_ratio"] = float(getattr(args, "delta_norm_cap_ratio", 0.0))
        fusion_cfg["delta_norm_cap_eps"] = float(getattr(args, "delta_norm_cap_eps", 1e-6))
        fusion_cfg["gate_depth_decay"] = float(getattr(args, "gate_depth_decay", 1.0))
        fusion_cfg["audio_gate_depth_decay"] = float(getattr(args, "audio_gate_depth_decay", 1.0))
        fusion_cfg["vision_gate_depth_decay"] = float(getattr(args, "vision_gate_depth_decay", 1.0))
        fusion_cfg["interaction_mixer_enable"] = bool(getattr(args, "icm_enable", False))
        fusion_cfg["interaction_mixer_dim"] = int(getattr(args, "icm_dim", 512))
        fusion_cfg["interaction_mixer_heads"] = int(getattr(args, "icm_heads", 8))
        fusion_cfg["interaction_mixer_layers"] = int(getattr(args, "icm_layers", 1))
        fusion_cfg["interaction_mixer_dropout"] = float(getattr(args, "icm_dropout", 0.1))
        fusion_cfg["interaction_mixer_gate_init"] = float(getattr(args, "icm_gate_init", -2.0))
        fusion_cfg["interaction_mixer_min_modalities"] = int(getattr(args, "icm_min_modalities", 2))
        fusion_cfg["interaction_mixer_util_target"] = float(getattr(args, "icm_util_target", 0.7))
    # Slim projector (default ON): output at bottleneck_dim instead of llm_hidden_size
    # Saves ~80% of trainable params. Disable with --no-slim-projector.
    # For concat mode, projectors MUST output at full llm_hidden_size.
    if cfg.get("fusion_type") == "concat":
        print("[RKCA] Projector outputs at full llm_hidden_size (concat mode)", flush=True)
    elif getattr(args, "slim_projector", True):
        slim_dim = fusion_cfg.get("bottleneck_dim", 256)
        cfg.setdefault("projector_config", {})["output_dim"] = slim_dim
        print(f"[SlimProjector] Projector output_dim set to {slim_dim} (bottleneck_dim)", flush=True)
    else:
        print(f"[SlimProjector] Disabled — projector outputs at full llm_hidden_size", flush=True)

    cfg["fusion_config"] = fusion_cfg
    # Only pass keys that SAFEModel.__init__ accepts
    valid_keys = {
        "llm_model_name", "vision_model_name",
        "audio_encoder_type", "audio_encoder_config",
        "projector_type", "num_audio_tokens", "projector_config",
        "fusion_type", "fusion_layer_indices", "lora_rank", "fusion_config",
        "freeze_base_vl", "freeze_audio_encoder", "label_smoothing",
        "llm_hidden_size", "audio_embed_dim",
        "vision_embed_dim", "num_vision_tokens", "vision_projector_config",
        "enable_gradient_checkpointing",
    }
    if "gradient_checkpointing" in cfg and "enable_gradient_checkpointing" not in cfg:
        cfg["enable_gradient_checkpointing"] = cfg["gradient_checkpointing"]
    return {k: v for k, v in cfg.items() if k in valid_keys}


def resolve_modality_batch(batch: Dict[str, Any], modality: str) -> Dict[str, Any]:
    if modality == "text":
        return {"audio": None, "images": None}
    if modality == "audio":
        return {"audio": batch["audio"], "images": None}
    if modality == "image":
        return {"audio": None, "images": batch["images"]}
    return {"audio": batch["audio"], "images": batch["images"]}


def build_optimizer(model: SAFEModel, args: argparse.Namespace) -> AdamW:
    """
    Use parameter-grouped weight decay:
    - no decay on bias/norm parameters
    - decay on matrix weights
    """
    no_decay_terms = ("bias", "norm.weight", "layer_norm.weight", "LayerNorm.weight")
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(term in name for term in no_decay_terms):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)


def configure_composition_calibration_trainables(
    model: SAFEModel,
    mode: str,
) -> Dict[str, int]:
    mode = str(mode).strip().lower()

    for _, param in model.named_parameters():
        param.requires_grad = False

    counts: Dict[str, int] = {
        "audio_projector": 0,
        "vision_projector": 0,
        "fusion_adapter": 0,
        "layer_gates": 0,
        "interaction_mixer": 0,
        "kv_adapters": 0,
        "audio_token_embeddings": 0,
    }

    def _enable_module(module: Optional[torch.nn.Module], key: str) -> None:
        if module is None:
            return
        n = 0
        for param in module.parameters():
            param.requires_grad = True
            n += int(param.numel())
        counts[key] += n

    if mode == "all":
        _enable_module(getattr(model, "audio_projector", None), "audio_projector")
        _enable_module(getattr(model, "vision_projector", None), "vision_projector")
        _enable_module(getattr(model, "fusion_adapter", None), "fusion_adapter")
        _enable_module(getattr(model, "interaction_mixer", None), "interaction_mixer")
        _enable_module(getattr(model, "kv_adapters", None), "kv_adapters")
        _enable_module(getattr(model, "audio_token_embeddings", None), "audio_token_embeddings")
    elif mode == "fusion":
        _enable_module(getattr(model, "fusion_adapter", None), "fusion_adapter")
    elif mode == "projectors":
        _enable_module(getattr(model, "audio_projector", None), "audio_projector")
        _enable_module(getattr(model, "vision_projector", None), "vision_projector")
    elif mode == "projectors_fusion":
        _enable_module(getattr(model, "audio_projector", None), "audio_projector")
        _enable_module(getattr(model, "vision_projector", None), "vision_projector")
        _enable_module(getattr(model, "fusion_adapter", None), "fusion_adapter")
    elif mode == "gates":
        fusion_adapter = getattr(model, "fusion_adapter", None)
        layer_gates = getattr(fusion_adapter, "layer_gates", None) if fusion_adapter is not None else None
        if layer_gates is not None:
            for _, gate_param in layer_gates.items():
                gate_param.requires_grad = True
                counts["layer_gates"] += int(gate_param.numel())
    elif mode == "interaction_mixer":
        _enable_module(getattr(model, "interaction_mixer", None), "interaction_mixer")
    else:
        raise ValueError(
            f"Unsupported compose_calibration_trainable mode: {mode}. "
            "Expected one of: all, fusion, projectors, projectors_fusion, gates, interaction_mixer"
        )

    return counts


def configure_trainable_modalities(
    model: SAFEModel,
    mode: str,
) -> Dict[str, int]:
    """Restrict training to a single modality's adapter parameters."""
    mode = str(mode).strip().lower()
    if mode == "all":
        return {"audio": 0, "vision": 0, "other": 0}

    prefix_map = {
        "audio": (
            "audio_projector.",
            "audio_token_embeddings.",
            "fusion_adapter.fusion_adapters.audio:",
            "fusion_adapter.layer_gates.audio:",
            "kv_adapters.audio:",
        ),
        "vision": (
            "vision_projector.",
            "fusion_adapter.fusion_adapters.vision:",
            "fusion_adapter.layer_gates.vision:",
            "kv_adapters.vision:",
        ),
    }
    if mode not in prefix_map:
        raise ValueError(
            f"Unsupported trainable modality mode: {mode}. "
            "Expected one of: all, audio, vision"
        )

    allowed_prefixes = prefix_map[mode]
    for name, param in model.named_parameters():
        if param.requires_grad and not name.startswith(allowed_prefixes):
            param.requires_grad = False

    counts: Dict[str, int] = {"audio": 0, "vision": 0, "other": 0}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(prefix_map["audio"]):
            counts["audio"] += int(param.numel())
        elif name.startswith(prefix_map["vision"]):
            counts["vision"] += int(param.numel())
        else:
            counts["other"] += int(param.numel())

    return counts


def build_lr_scheduler(
    optimizer: AdamW,
    total_update_steps: int,
    args: argparse.Namespace,
) -> tuple[Optional[LambdaLR], int]:
    if args.lr_scheduler == "none" or total_update_steps <= 0:
        return None, 0

    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int(total_update_steps * args.warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, max(0, total_update_steps - 1)))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, total_update_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)

        if args.lr_scheduler == "linear":
            return max(args.min_lr_ratio, 1.0 - progress)
        if args.lr_scheduler == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
        return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda), warmup_steps


def log_gradient_attribution(
    model: SAFEModel,
    step: int,
    wandb_run: Any = None,
) -> Dict[str, float]:
    """
    Log per-layer gradient norms for fusion adapter parameters.

    This enables a principled study of which decoder layers benefit most
    from audio fusion by comparing gradient flow across fusion layers.
    Layers with higher gradient norms are contributing more to loss reduction,
    suggesting they are more valuable injection points.

    Returns dict of {layer_key: grad_norm} for external logging.
    """
    grad_norms: Dict[str, float] = {}
    param_norms: Dict[str, float] = {}
    gate_values: Dict[str, float] = {}

    # Collect per-adapter gradient norms
    if hasattr(model, "fusion_adapter") and model.fusion_adapter is not None:
        adapter = model.fusion_adapter
        if hasattr(adapter, "fusion_adapters"):
            for key, sub_adapter in adapter.fusion_adapters.items():
                total_grad_norm = 0.0
                total_param_norm = 0.0
                param_count = 0
                for name, param in sub_adapter.named_parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                    total_param_norm += param.data.norm(2).item() ** 2
                    param_count += param.numel()
                grad_norms[key] = total_grad_norm ** 0.5
                param_norms[key] = total_param_norm ** 0.5

        # Collect learned gate values
        if hasattr(adapter, "get_learned_gate_values"):
            gate_values = adapter.get_learned_gate_values()

    # Collect projector gradient norm
    if hasattr(model, "audio_projector"):
        proj_grad = 0.0
        for param in model.audio_projector.parameters():
            if param.grad is not None:
                proj_grad += param.grad.data.norm(2).item() ** 2
        grad_norms["projector"] = proj_grad ** 0.5

    # Log to stdout
    if grad_norms:
        parts = []
        for key in sorted(grad_norms.keys()):
            g = grad_norms[key]
            entry = f"{key}={g:.3e}"
            if key in gate_values:
                entry += f"(gate={gate_values[key]:.3f})"
            parts.append(entry)
        print(f"  [grad_attribution] step={step} " + " | ".join(parts), flush=True)

    # Log to wandb
    if wandb_run is not None:
        log_payload = {}
        for key, norm in grad_norms.items():
            safe_key = key.replace(":", "_")
            log_payload[f"grad_norm/{safe_key}"] = norm
        for key, val in gate_values.items():
            safe_key = key.replace(":", "_")
            log_payload[f"learned_gate/{safe_key}"] = val
        for key, norm in param_norms.items():
            safe_key = key.replace(":", "_")
            log_payload[f"param_norm/{safe_key}"] = norm
        wandb_run.log(log_payload, step=step)

    return grad_norms


def _recover_loss_from_logits(outputs: Any, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Recover CE loss when model-provided loss is detached but logits still carry grad.
    """
    if labels is None:
        return None

    logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
    if logits is None or not torch.is_tensor(logits) or not logits.requires_grad:
        return None

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    valid = flat_labels != -100
    if not valid.any():
        return None

    return F.cross_entropy(flat_logits[valid], flat_labels[valid])


def _sequence_nll_from_output(
    outputs: Any,
    labels: Optional[torch.Tensor],
    require_grad: bool = False,
) -> Optional[torch.Tensor]:
    """
    Return token-level CE/NLL scalar from model outputs.
    Falls back to CE(logits, labels) when model loss is detached.
    """
    raw_loss = outputs.get("loss") if isinstance(outputs, dict) else getattr(outputs, "loss", None)
    if torch.is_tensor(raw_loss):
        if (not require_grad) or raw_loss.requires_grad:
            return raw_loss

    recovered = _recover_loss_from_logits(outputs, labels)
    if recovered is not None:
        if (not require_grad) or recovered.requires_grad:
            return recovered
    return None


def _confidence_from_logits(
    logits: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
) -> float:
    """
    Confidence proxy in [0,1] using mean top-1 probability on supervised tokens.
    This avoids full entropy computation while still tracking certainty.
    """
    if logits is None or labels is None or not torch.is_tensor(logits) or not torch.is_tensor(labels):
        return 0.5

    if logits.size(1) < 2:
        return 0.5

    shift_logits = logits[..., :-1, :].contiguous()  # (B, T-1, V)
    shift_labels = labels[..., 1:].contiguous()      # (B, T-1)
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    valid = flat_labels != -100
    if not valid.any():
        return 0.5

    with torch.no_grad():
        valid_logits = flat_logits[valid].float()
        top1 = valid_logits.max(dim=-1).values
        lse = torch.logsumexp(valid_logits, dim=-1)
        top1_prob = torch.exp(top1 - lse)
        conf = float(top1_prob.mean().item())
    if not math.isfinite(conf):
        return 0.5
    return float(max(0.0, min(1.0, conf)))


def _flatten_valid_logits(
    logits: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Return flattened logits over supervised next-token positions: (N_valid, V).
    """
    if logits is None or labels is None or not torch.is_tensor(logits) or not torch.is_tensor(labels):
        return None
    if logits.size(1) < 2:
        return None

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    valid = flat_labels != -100
    if not valid.any():
        return None
    return flat_logits[valid]


def _extract_next_answer_logits(
    logits: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Return logits used to predict the first answer token.
    With left-padded prompts, the final input position predicts the first generated token.
    """
    if logits is None or not torch.is_tensor(logits):
        return None
    if logits.ndim != 3 or logits.size(1) == 0:
        return None

    if attention_mask is None or not torch.is_tensor(attention_mask):
        return logits[:, -1, :]

    last_pos = attention_mask.sum(dim=1).long().clamp(min=1) - 1
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    return logits[batch_idx, last_pos, :]


def _logit_entropy(logits_2d: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if logits_2d is None or not torch.is_tensor(logits_2d):
        return None
    probs = F.softmax(logits_2d.float(), dim=-1).clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return entropy.mean()


def _kl_divergence_from_logits(
    logits_p: Optional[torch.Tensor],
    logits_q: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if logits_p is None or logits_q is None:
        return None
    p_log = F.log_softmax(logits_p.float(), dim=-1)
    q_log = F.log_softmax(logits_q.float(), dim=-1)
    p = p_log.exp()
    return (p * (p_log - q_log)).sum(dim=-1).mean()


def _parse_float_csv(csv_value: str) -> List[float]:
    out: List[float] = []
    for raw in str(csv_value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.append(float(raw))
    return out


def _slice_eval_batch(batch: Dict[str, Any], idx: int) -> Dict[str, Any]:
    sliced: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, list):
            sliced[key] = [value[idx]]
        else:
            sliced[key] = value
    return sliced


def _get_ttc_gate_keys(
    model: SAFEModel,
    active_modalities: Sequence[str],
    active_fusion_layers: Optional[Sequence[int]] = None,
) -> List[str]:
    fusion_adapter = getattr(model, "fusion_adapter", None)
    if fusion_adapter is None:
        return []

    active_set = {str(m) for m in active_modalities}
    layer_filter = None
    if active_fusion_layers is not None:
        layer_filter = {int(x) for x in active_fusion_layers}

    keys: List[str] = []
    for modality in sorted(active_set):
        for layer_idx in _get_modality_fusion_layers(model, modality):
            if layer_filter is not None and int(layer_idx) not in layer_filter:
                continue
            key = f"{modality}:{int(layer_idx)}"
            if hasattr(fusion_adapter, "fusion_adapters") and key in fusion_adapter.fusion_adapters:
                keys.append(key)
    return keys


def _set_runtime_gate_overrides(
    model: SAFEModel,
    overrides: Optional[Dict[str, torch.Tensor]],
) -> None:
    fusion_adapter = getattr(model, "fusion_adapter", None)
    if fusion_adapter is None or not hasattr(fusion_adapter, "set_runtime_gate_overrides"):
        return
    fusion_adapter.set_runtime_gate_overrides(overrides)


def _clear_runtime_gate_overrides(model: SAFEModel) -> None:
    fusion_adapter = getattr(model, "fusion_adapter", None)
    if fusion_adapter is None or not hasattr(fusion_adapter, "clear_runtime_gate_overrides"):
        return
    fusion_adapter.clear_runtime_gate_overrides()


def _get_ttc_interaction_layers(
    model: SAFEModel,
    active_fusion_layers: Optional[Sequence[int]] = None,
) -> List[int]:
    audio_layers = set(_get_modality_fusion_layers(model, "audio"))
    vision_layers = set(_get_modality_fusion_layers(model, "vision"))
    shared = sorted(audio_layers.intersection(vision_layers))
    if active_fusion_layers is None:
        return shared
    active_set = {int(x) for x in active_fusion_layers}
    return [int(x) for x in shared if int(x) in active_set]


def _set_runtime_interaction_overrides(
    model: SAFEModel,
    overrides: Optional[Dict[int, Any]],
) -> None:
    fusion_adapter = getattr(model, "fusion_adapter", None)
    if fusion_adapter is None or not hasattr(fusion_adapter, "set_runtime_interaction_overrides"):
        return
    fusion_adapter.set_runtime_interaction_overrides(overrides)


def _clear_runtime_interaction_overrides(model: SAFEModel) -> None:
    fusion_adapter = getattr(model, "fusion_adapter", None)
    if fusion_adapter is None or not hasattr(fusion_adapter, "clear_runtime_interaction_overrides"):
        return
    fusion_adapter.clear_runtime_interaction_overrides()


def _make_ttc_frozen_gate_tensors(
    gate_params: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach()
        for key, value in gate_params.items()
    }


def _build_ttc_interaction_params(
    interaction_layers: Sequence[int],
    hidden_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[int, Dict[str, torch.nn.Parameter]]:
    module_type = str(getattr(args, "ttc_interaction_module", "diag")).strip().lower()
    params: Dict[int, Dict[str, torch.nn.Parameter]] = {}
    for layer in interaction_layers:
        if module_type == "scalar":
            params[int(layer)] = {
                "scale": torch.nn.Parameter(
                    torch.tensor(float(args.ttc_interaction_init), device=device, dtype=torch.float32)
                )
            }
        elif module_type == "lowrank":
            rank = int(getattr(args, "ttc_interaction_rank", 8))
            init_std = float(getattr(args, "ttc_interaction_matrix_init", 0.01))
            down = torch.empty((int(hidden_size), rank), device=device, dtype=torch.float32)
            up = torch.empty((rank, int(hidden_size)), device=device, dtype=torch.float32)
            torch.nn.init.normal_(down, mean=0.0, std=init_std)
            torch.nn.init.normal_(up, mean=0.0, std=init_std)
            params[int(layer)] = {
                "scale": torch.nn.Parameter(
                    torch.tensor(1.0, device=device, dtype=torch.float32)
                ),
                "down": torch.nn.Parameter(down),
                "up": torch.nn.Parameter(up),
            }
        else:
            params[int(layer)] = {
                "scale": torch.nn.Parameter(
                    torch.tensor(1.0, device=device, dtype=torch.float32)
                ),
                "diag": torch.nn.Parameter(
                    torch.full(
                        (int(hidden_size),),
                        float(getattr(args, "ttc_interaction_diag_init", 0.0)),
                        device=device,
                        dtype=torch.float32,
                    )
                ),
            }
    return params


def _flatten_ttc_interaction_params(
    interaction_params: Dict[int, Dict[str, torch.nn.Parameter]],
) -> List[torch.nn.Parameter]:
    flat: List[torch.nn.Parameter] = []
    for per_layer in interaction_params.values():
        flat.extend(list(per_layer.values()))
    return flat


def _runtime_ttc_interaction_overrides(
    interaction_params: Dict[int, Dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> Dict[int, Dict[str, torch.Tensor]]:
    module_type = str(getattr(args, "ttc_interaction_module", "diag")).strip().lower()
    overrides: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer, per_layer in interaction_params.items():
        row: Dict[str, torch.Tensor] = {"mode": module_type}  # type: ignore[assignment]
        if "scale" in per_layer:
            row["scale"] = per_layer["scale"]
        if module_type == "diag" and "diag" in per_layer:
            row["diag"] = per_layer["diag"]
        if module_type == "lowrank":
            if "down" in per_layer:
                row["down"] = per_layer["down"]
            if "up" in per_layer:
                row["up"] = per_layer["up"]
        overrides[int(layer)] = row
    return overrides


def _interaction_layer_grad_norms(
    interaction_params: Dict[int, Dict[str, torch.nn.Parameter]],
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for layer, per_layer in interaction_params.items():
        norms: List[float] = []
        for param in per_layer.values():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            norms.append(float(grad.norm().item()) / math.sqrt(float(max(1, grad.numel()))))
        if norms:
            out[int(layer)] = sum(norms) / float(len(norms))
    return out


def _summarize_interaction_params(
    interaction_params: Dict[int, Dict[str, torch.nn.Parameter]],
    args: argparse.Namespace,
) -> Dict[int, Dict[str, float]]:
    module_type = str(getattr(args, "ttc_interaction_module", "diag")).strip().lower()
    summary: Dict[int, Dict[str, float]] = {}
    for layer, per_layer in interaction_params.items():
        row: Dict[str, float] = {}
        if "scale" in per_layer:
            row["scale"] = float(per_layer["scale"].detach().item())
        if module_type == "diag" and "diag" in per_layer:
            diag = torch.tanh(per_layer["diag"].detach())
            row["diag_mean_abs"] = float(diag.abs().mean().item())
            row["diag_max_abs"] = float(diag.abs().max().item())
        if module_type == "lowrank":
            if "down" in per_layer:
                down = torch.tanh(per_layer["down"].detach())
                row["down_mean_abs"] = float(down.abs().mean().item())
                row["down_max_abs"] = float(down.abs().max().item())
            if "up" in per_layer:
                up = torch.tanh(per_layer["up"].detach())
                row["up_mean_abs"] = float(up.abs().mean().item())
                row["up_max_abs"] = float(up.abs().max().item())
        summary[int(layer)] = row
    return summary


def _clone_param_dict_state(
    params: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in params.items()}


def _restore_param_dict_state(
    params: Dict[str, torch.Tensor],
    state: Optional[Dict[str, torch.Tensor]],
) -> None:
    if not state:
        return
    with torch.no_grad():
        for key, value in state.items():
            if key in params:
                params[key].copy_(value)


def _clone_nested_param_state(
    params: Dict[int, Dict[str, torch.Tensor]],
) -> Dict[int, Dict[str, torch.Tensor]]:
    return {
        int(layer): {name: tensor.detach().clone() for name, tensor in per_layer.items()}
        for layer, per_layer in params.items()
    }


def _restore_nested_param_state(
    params: Dict[int, Dict[str, torch.Tensor]],
    state: Optional[Dict[int, Dict[str, torch.Tensor]]],
) -> None:
    if not state:
        return
    with torch.no_grad():
        for layer, per_layer in state.items():
            if int(layer) not in params:
                continue
            for name, value in per_layer.items():
                if name in params[int(layer)]:
                    params[int(layer)][name].copy_(value)


@torch.no_grad()
def _compute_modality_entropy(
    model: SAFEModel,
    batch: Dict[str, Any],
    device: torch.device,
    modality: str,
    args: argparse.Namespace,
    active_fusion_layers: Optional[Sequence[int]] = None,
) -> Optional[float]:
    mm = resolve_modality_batch(batch, modality)
    inputs = model.prepare_multimodal_inputs(
        text=build_modality_aware_questions(batch["questions"], modality, args),
        images=mm["images"],
        audio=mm["audio"],
        answers=None,
        device=str(device),
        training_mode=False,
    )
    audio_tokens = inputs.pop("audio_tokens", None)
    audio_mask = inputs.pop("audio_attention_mask", None)
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        labels=None,
        pixel_values=inputs.get("pixel_values"),
        audio_tokens=audio_tokens,
        audio_attention_mask=audio_mask,
        gate=args.fusion_gate,
        active_fusion_layers=active_fusion_layers,
    )
    logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
    next_logits = _extract_next_answer_logits(logits, inputs.get("attention_mask"))
    entropy = _logit_entropy(next_logits)
    if entropy is None:
        return None
    return float(entropy.detach().item())


def _compute_ttc_loss(
    entropy: torch.Tensor,
    args: argparse.Namespace,
    gate_params: Dict[str, torch.Tensor],
    best_single_entropy: Optional[float],
    interaction_params: Optional[Any] = None,
) -> torch.Tensor:
    gate_reg = torch.stack([(param - 1.0).pow(2) for param in gate_params.values()]).mean()
    loss = entropy + float(args.ttc_gate_reg_lambda) * gate_reg

    if args.ttc_objective == "entropy_noharm" and best_single_entropy is not None:
        best_single_t = torch.tensor(
            float(best_single_entropy),
            device=entropy.device,
            dtype=entropy.dtype,
        )
        loss = loss + float(args.ttc_noharm_lambda) * F.relu(
            entropy - best_single_t + float(args.ttc_noharm_margin)
        )

    if interaction_params:
        interaction_tensors: List[torch.Tensor] = []
        if isinstance(interaction_params, dict):
            for value in interaction_params.values():
                if torch.is_tensor(value):
                    interaction_tensors.append(value)
                elif isinstance(value, dict):
                    for inner in value.values():
                        if torch.is_tensor(inner):
                            interaction_tensors.append(inner)
        elif isinstance(interaction_params, (list, tuple)):
            for value in interaction_params:
                if torch.is_tensor(value):
                    interaction_tensors.append(value)
        if interaction_tensors:
            interaction_reg = torch.stack([param.pow(2).mean() for param in interaction_tensors]).mean()
        else:
            interaction_reg = entropy.new_zeros(())
        loss = loss + float(args.ttc_interaction_reg_lambda) * interaction_reg

    return loss


def _forward_joint_next_logits(
    model: SAFEModel,
    both_inputs: Dict[str, Any],
    audio_tokens: Optional[torch.Tensor],
    audio_mask: Optional[torch.Tensor],
    args: argparse.Namespace,
    active_fusion_layers: Optional[Sequence[int]] = None,
) -> Optional[torch.Tensor]:
    outputs = model(
        input_ids=both_inputs["input_ids"],
        attention_mask=both_inputs.get("attention_mask"),
        labels=None,
        pixel_values=both_inputs.get("pixel_values"),
        audio_tokens=audio_tokens,
        audio_attention_mask=audio_mask,
        gate=args.fusion_gate,
        active_fusion_layers=active_fusion_layers,
    )
    logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
    return _extract_next_answer_logits(logits, both_inputs.get("attention_mask"))


def _forward_joint_entropy(
    model: SAFEModel,
    both_inputs: Dict[str, Any],
    audio_tokens: Optional[torch.Tensor],
    audio_mask: Optional[torch.Tensor],
    args: argparse.Namespace,
    active_fusion_layers: Optional[Sequence[int]] = None,
) -> Optional[torch.Tensor]:
    next_logits = _forward_joint_next_logits(
        model=model,
        both_inputs=both_inputs,
        audio_tokens=audio_tokens,
        audio_mask=audio_mask,
        args=args,
        active_fusion_layers=active_fusion_layers,
    )
    return _logit_entropy(next_logits)


def _build_ttc_candidate_gate_configs(
    gate_keys: Sequence[str],
    args: argparse.Namespace,
) -> List[Tuple[str, Dict[str, float]]]:
    scales = _parse_float_csv(getattr(args, "ttc_candidate_grid", "0.75,1.0,1.25"))
    if not scales:
        scales = [1.0]

    configs: List[Tuple[str, Dict[str, float]]] = []
    seen = set()
    for audio_scale in scales:
        for vision_scale in scales:
            name = f"a{audio_scale:.2f}_v{vision_scale:.2f}"
            if name in seen:
                continue
            seen.add(name)
            overrides: Dict[str, float] = {}
            for key in gate_keys:
                if key.startswith("audio:"):
                    overrides[key] = float(audio_scale)
                elif key.startswith("vision:"):
                    overrides[key] = float(vision_scale)
                else:
                    overrides[key] = 1.0
            configs.append((name, overrides))
    return configs


@torch.no_grad()
def _evaluate_ttc_candidate(
    model: SAFEModel,
    both_inputs: Dict[str, Any],
    audio_tokens: Optional[torch.Tensor],
    audio_mask: Optional[torch.Tensor],
    args: argparse.Namespace,
    candidate_overrides: Dict[str, float],
    best_single_entropy: Optional[float],
    active_fusion_layers: Optional[Sequence[int]] = None,
) -> Optional[Dict[str, float]]:
    gate_tensors = {
        key: torch.tensor(float(val), device=both_inputs["input_ids"].device, dtype=torch.float32)
        for key, val in candidate_overrides.items()
    }
    _set_runtime_gate_overrides(model, gate_tensors)
    next_logits = _forward_joint_next_logits(
        model=model,
        both_inputs=both_inputs,
        audio_tokens=audio_tokens,
        audio_mask=audio_mask,
        args=args,
        active_fusion_layers=active_fusion_layers,
    )
    entropy = _logit_entropy(next_logits)
    if entropy is None or next_logits is None:
        _clear_runtime_gate_overrides(model)
        return None

    gate_reg = sum((float(v) - 1.0) ** 2 for v in candidate_overrides.values()) / float(max(1, len(candidate_overrides)))
    score = float(entropy.detach().item()) + float(args.ttc_gate_reg_lambda) * gate_reg
    noharm_penalty = 0.0
    if args.ttc_objective == "entropy_noharm" and best_single_entropy is not None:
        noharm_penalty = max(
            0.0,
            float(entropy.detach().item()) - float(best_single_entropy) + float(args.ttc_noharm_margin),
        )
        score += float(args.ttc_noharm_lambda) * noharm_penalty

    instability = 0.0
    stable = True
    if getattr(args, "ttc_stability_enable", False):
        perturbed = {}
        perturb = float(args.ttc_stability_perturb)
        for idx, (key, val) in enumerate(candidate_overrides.items()):
            sign = -1.0 if (idx % 2 == 0) else 1.0
            perturbed[key] = min(
                float(args.ttc_gate_max),
                max(float(args.ttc_gate_min), float(val) * (1.0 + sign * perturb)),
            )
        perturbed_tensors = {
            key: torch.tensor(float(val), device=both_inputs["input_ids"].device, dtype=torch.float32)
            for key, val in perturbed.items()
        }
        _set_runtime_gate_overrides(model, perturbed_tensors)
        perturbed_logits = _forward_joint_next_logits(
            model=model,
            both_inputs=both_inputs,
            audio_tokens=audio_tokens,
            audio_mask=audio_mask,
            args=args,
            active_fusion_layers=active_fusion_layers,
        )
        stability_kl = _kl_divergence_from_logits(next_logits, perturbed_logits)
        instability = float(stability_kl.detach().item()) if stability_kl is not None else float("inf")
        stable = instability <= float(args.ttc_stability_threshold)

    _clear_runtime_gate_overrides(model)
    return {
        "entropy": float(entropy.detach().item()),
        "score": float(score),
        "instability": float(instability),
        "stable": float(stable),
        "noharm_penalty": float(noharm_penalty),
    }


def _optimize_ttc_gate_overrides(
    model: SAFEModel,
    batch: Dict[str, Any],
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
    active_fusion_layers: Optional[Sequence[int]] = None,
) -> Tuple[str, Dict[str, Any]]:
    gate_keys = _get_ttc_gate_keys(
        model,
        active_modalities=("audio", "vision"),
        active_fusion_layers=active_fusion_layers,
    )
    if not gate_keys:
        raise RuntimeError("TTC requested but no audio/vision fusion gates are available.")

    both_inputs = model.prepare_multimodal_inputs(
        text=build_modality_aware_questions(batch["questions"], "both", args),
        images=batch["images"],
        audio=batch["audio"],
        answers=None,
        device=str(device),
        training_mode=False,
    )
    audio_tokens = both_inputs.pop("audio_tokens", None)
    audio_mask = both_inputs.pop("audio_attention_mask", None)
    if torch.is_tensor(audio_tokens):
        audio_tokens = audio_tokens.detach()
    if torch.is_tensor(audio_mask):
        audio_mask = audio_mask.detach()
    for key in ("input_ids", "attention_mask", "pixel_values"):
        value = both_inputs.get(key)
        if torch.is_tensor(value):
            both_inputs[key] = value.detach()

    base_entropy = _compute_modality_entropy(
        model, batch, device, modality="both", args=args, active_fusion_layers=active_fusion_layers
    )
    ref_audio_entropy = None
    ref_image_entropy = None
    best_single_entropy = None
    if args.ttc_objective == "entropy_noharm":
        ref_audio_entropy = _compute_modality_entropy(
            model, batch, device, modality="audio", args=args, active_fusion_layers=active_fusion_layers
        )
        ref_image_entropy = _compute_modality_entropy(
            model, batch, device, modality="image", args=args, active_fusion_layers=active_fusion_layers
        )
        refs = [x for x in (ref_audio_entropy, ref_image_entropy) if x is not None]
        if refs:
            best_single_entropy = min(refs)

    gate_params = {
        key: torch.nn.Parameter(
            torch.tensor(float(args.ttc_init_gate), device=device, dtype=torch.float32)
        )
        for key in gate_keys
    }
    optimizer = AdamW(gate_params.values(), lr=args.ttc_lr, weight_decay=0.0)
    interaction_layers = _get_ttc_interaction_layers(
        model,
        active_fusion_layers=active_fusion_layers,
    )
    search_mode = str(getattr(args, "ttc_search_mode", "gradient")).lower().strip()

    stats: Dict[str, Any] = {
        "enabled": True,
        "objective": str(args.ttc_objective),
        "steps": int(args.ttc_steps),
        "gate_keys": list(gate_keys),
        "base_entropy": base_entropy,
        "audio_entropy": ref_audio_entropy,
        "image_entropy": ref_image_entropy,
        "stage_c_enabled": bool(getattr(args, "ttc_interaction_enable", False)),
        "compute_budget_forward_equiv": float(2 + max(0, int(args.ttc_steps))),
        "search_mode": search_mode,
    }

    try:
        stage_b_entropy = None
        if search_mode in {"candidate", "hybrid"}:
            candidates = _build_ttc_candidate_gate_configs(gate_keys, args)
            candidate_results: List[Dict[str, Any]] = []
            for name, overrides in candidates:
                result = _evaluate_ttc_candidate(
                    model=model,
                    both_inputs=both_inputs,
                    audio_tokens=audio_tokens,
                    audio_mask=audio_mask,
                    args=args,
                    candidate_overrides=overrides,
                    best_single_entropy=best_single_entropy,
                    active_fusion_layers=active_fusion_layers,
                )
                if result is None:
                    continue
                candidate_results.append({"name": name, "overrides": overrides, **result})

            stats["candidate_count"] = int(len(candidate_results))
            stable_candidates = [row for row in candidate_results if bool(row.get("stable", 0.0))]
            stats["stable_candidate_count"] = int(len(stable_candidates))
            if candidate_results:
                stats["candidate_score_range"] = float(
                    max(row["score"] for row in candidate_results) - min(row["score"] for row in candidate_results)
                )
            else:
                stats["candidate_score_range"] = None

            best_pool = stable_candidates if stable_candidates else candidate_results
            if best_pool:
                best_candidate = min(best_pool, key=lambda row: (row["score"], row["entropy"]))
                for key, param in gate_params.items():
                    param.data.fill_(float(best_candidate["overrides"][key]))
                stage_b_entropy = float(best_candidate["entropy"])
                stats["candidate_selected"] = best_candidate["name"]
                stats["candidate_selected_stable"] = bool(best_candidate.get("stable", 0.0))
                stats["stage_b_method"] = "candidate"
            else:
                stats["candidate_selected"] = None
                stats["candidate_selected_stable"] = False
                stats["stage_b_failure"] = "flat_or_invalid_candidate_landscape"
                stats["stage_b_method"] = "default"

        if search_mode in {"gradient", "hybrid"}:
            best_stage_b_loss = float("inf")
            best_gate_state = _clone_param_dict_state(gate_params)
            for _ in range(max(0, int(args.ttc_steps))):
                optimizer.zero_grad(set_to_none=True)
                _set_runtime_gate_overrides(model, gate_params)
                entropy = _forward_joint_entropy(
                    model=model,
                    both_inputs=both_inputs,
                    audio_tokens=audio_tokens,
                    audio_mask=audio_mask,
                    args=args,
                    active_fusion_layers=active_fusion_layers,
                )
                if entropy is None:
                    break

                loss = _compute_ttc_loss(
                    entropy=entropy,
                    args=args,
                    gate_params=gate_params,
                    best_single_entropy=best_single_entropy,
                )
                current_loss = float(loss.detach().item())
                if math.isfinite(current_loss) and current_loss < best_stage_b_loss:
                    best_stage_b_loss = current_loss
                    best_gate_state = _clone_param_dict_state(gate_params)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(gate_params.values()), float(args.max_grad_norm))
                optimizer.step()
                with torch.no_grad():
                    for param in gate_params.values():
                        param.clamp_(float(args.ttc_gate_min), float(args.ttc_gate_max))
            _restore_param_dict_state(gate_params, best_gate_state)
            with torch.no_grad():
                _set_runtime_gate_overrides(model, gate_params)
                final_stage_b = _forward_joint_entropy(
                    model=model,
                    both_inputs=both_inputs,
                    audio_tokens=audio_tokens,
                    audio_mask=audio_mask,
                    args=args,
                    active_fusion_layers=active_fusion_layers,
                )
                if final_stage_b is not None:
                    stage_b_entropy = float(final_stage_b.detach().item())
                _clear_runtime_gate_overrides(model)
            stats["stage_b_method"] = "gradient" if search_mode == "gradient" else "hybrid"

        gate_grad_ratios: Dict[int, float] = {}
        interaction_params: Dict[int, Dict[str, torch.nn.Parameter]] = {}
        interaction_optimizer = None
        if getattr(args, "ttc_interaction_enable", False) and interaction_layers:
            frozen_gate_params = _make_ttc_frozen_gate_tensors(gate_params)
            _set_runtime_gate_overrides(model, frozen_gate_params)
            _clear_runtime_interaction_overrides(model)
            probe_entropy = _forward_joint_entropy(
                model=model,
                both_inputs=both_inputs,
                audio_tokens=audio_tokens,
                audio_mask=audio_mask,
                args=args,
                active_fusion_layers=active_fusion_layers,
            )
            if probe_entropy is not None:
                stage_b_entropy = float(probe_entropy.detach().item())
                probe_loss = _compute_ttc_loss(
                    entropy=probe_entropy,
                    args=args,
                    gate_params=frozen_gate_params,
                    best_single_entropy=best_single_entropy,
                )
                probe_loss.backward()
                gate_grad_by_layer: Dict[int, float] = {}
                for layer in interaction_layers:
                    layer_norms: List[float] = []
                    for key in (f"audio:{int(layer)}", f"vision:{int(layer)}"):
                        param = gate_params.get(key)
                        if param is not None and param.grad is not None:
                            layer_norms.append(float(param.grad.detach().norm().item()))
                    if layer_norms:
                        gate_grad_by_layer[int(layer)] = sum(layer_norms) / float(len(layer_norms))
                for param in gate_params.values():
                    param.grad = None

                interaction_params = _build_ttc_interaction_params(
                    interaction_layers=interaction_layers,
                    hidden_size=int(getattr(model, "llm_hidden_size", 4096)),
                    device=device,
                    args=args,
                )
                _set_runtime_interaction_overrides(
                    model,
                    _runtime_ttc_interaction_overrides(interaction_params, args),
                )
                interaction_probe_entropy = _forward_joint_entropy(
                    model=model,
                    both_inputs=both_inputs,
                    audio_tokens=audio_tokens,
                    audio_mask=audio_mask,
                    args=args,
                    active_fusion_layers=active_fusion_layers,
                )
                if interaction_probe_entropy is not None:
                    interaction_probe_loss = _compute_ttc_loss(
                        entropy=interaction_probe_entropy,
                        args=args,
                        gate_params=frozen_gate_params,
                        best_single_entropy=best_single_entropy,
                        interaction_params=interaction_params,
                    )
                    interaction_probe_loss.backward()
                    interaction_grad_norms = _interaction_layer_grad_norms(interaction_params)
                    for layer, grad_norm in interaction_grad_norms.items():
                        ratio = grad_norm / max(gate_grad_by_layer.get(int(layer), 0.0), 1e-8)
                        gate_grad_ratios[int(layer)] = ratio
                    for per_layer in interaction_params.values():
                        for param in per_layer.values():
                            param.grad = None
                    alive_count = sum(1 for ratio in gate_grad_ratios.values() if ratio > 0.1)
                    weak_count = sum(1 for ratio in gate_grad_ratios.values() if ratio >= 0.01)
                    stage_c_allowed = (
                        alive_count >= max(1, math.ceil(len(interaction_layers) / 2.0))
                        or weak_count == len(interaction_layers)
                    )
                    stats["interaction_grad_ratios"] = gate_grad_ratios
                    stats["interaction_stage_allowed"] = bool(stage_c_allowed)
                    stats["interaction_stage_threshold"] = {
                        "alive_ratio": 0.1,
                        "weak_ratio": 0.01,
                    }
                    if not stage_c_allowed:
                        stats["interaction_failure_mode"] = "dead_interaction_gradients"
                    if stage_c_allowed:
                        interaction_optimizer = AdamW(
                            _flatten_ttc_interaction_params(interaction_params),
                            lr=float(args.ttc_interaction_lr),
                            weight_decay=0.0,
                        )
                    else:
                        interaction_params = {}
                _clear_runtime_interaction_overrides(model)

        if interaction_optimizer is not None and interaction_params:
            frozen_gate_params = _make_ttc_frozen_gate_tensors(gate_params)
            _set_runtime_gate_overrides(model, frozen_gate_params)
            best_stage_c_loss = float("inf")
            best_interaction_state = _clone_nested_param_state(interaction_params)
            for _ in range(max(0, int(args.ttc_interaction_steps))):
                interaction_optimizer.zero_grad(set_to_none=True)
                _set_runtime_interaction_overrides(
                    model,
                    _runtime_ttc_interaction_overrides(interaction_params, args),
                )
                entropy = _forward_joint_entropy(
                    model=model,
                    both_inputs=both_inputs,
                    audio_tokens=audio_tokens,
                    audio_mask=audio_mask,
                    args=args,
                    active_fusion_layers=active_fusion_layers,
                )
                if entropy is None:
                    break
                loss = _compute_ttc_loss(
                    entropy=entropy,
                    args=args,
                    gate_params=frozen_gate_params,
                    best_single_entropy=best_single_entropy,
                    interaction_params=interaction_params,
                )
                current_loss = float(loss.detach().item())
                if math.isfinite(current_loss) and current_loss < best_stage_c_loss:
                    best_stage_c_loss = current_loss
                    best_interaction_state = _clone_nested_param_state(interaction_params)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    _flatten_ttc_interaction_params(interaction_params),
                    float(args.max_grad_norm),
                )
                interaction_optimizer.step()
                with torch.no_grad():
                    for per_layer in interaction_params.values():
                        if "scale" in per_layer:
                            per_layer["scale"].clamp_(
                                float(args.ttc_interaction_min),
                                float(args.ttc_interaction_max),
                            )
                        if "diag" in per_layer:
                            per_layer["diag"].clamp_(
                                float(args.ttc_interaction_diag_min),
                                float(args.ttc_interaction_diag_max),
                            )
                        if "down" in per_layer:
                            per_layer["down"].clamp_(
                                float(args.ttc_interaction_matrix_min),
                                float(args.ttc_interaction_matrix_max),
                            )
                        if "up" in per_layer:
                            per_layer["up"].clamp_(
                                float(args.ttc_interaction_matrix_min),
                                float(args.ttc_interaction_matrix_max),
                            )
            _restore_nested_param_state(interaction_params, best_interaction_state)
            stats["compute_budget_forward_equiv"] = float(
                2 + max(0, int(args.ttc_steps)) + max(0, int(args.ttc_interaction_steps))
            )

        with torch.no_grad():
            _set_runtime_gate_overrides(model, gate_params)
            if interaction_params:
                _set_runtime_interaction_overrides(
                    model,
                    _runtime_ttc_interaction_overrides(interaction_params, args),
                )
            final_outputs = model.generate(
                text=None,
                input_ids=both_inputs["input_ids"],
                attention_mask=both_inputs.get("attention_mask"),
                pixel_values=both_inputs.get("pixel_values"),
                audio_tokens=audio_tokens,
                audio_attention_mask=audio_mask,
                active_fusion_layers=active_fusion_layers,
                gate=args.fusion_gate,
                max_new_tokens=args.max_answer_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )

            prompt_width = int(both_inputs["input_ids"].size(1))
            seq = final_outputs[0]
            gen = seq[prompt_width:] if seq.size(0) > prompt_width else seq
            pred = tokenizer.decode(gen, skip_special_tokens=True).strip()

            final_entropy = _forward_joint_entropy(
                model=model,
                both_inputs=both_inputs,
                audio_tokens=audio_tokens,
                audio_mask=audio_mask,
                args=args,
                active_fusion_layers=active_fusion_layers,
            )

            stats["final_entropy"] = float(final_entropy.detach().item()) if final_entropy is not None else None
            stats["gate_values"] = {
                key: float(param.detach().item()) for key, param in gate_params.items()
            }
            stats["stage_b_entropy"] = stage_b_entropy
            if stage_b_entropy is not None and stats["final_entropy"] is not None:
                stats["interaction_gain_over_stage_b"] = float(stage_b_entropy - stats["final_entropy"])
            if interaction_params:
                stats["interaction_values"] = _summarize_interaction_params(interaction_params, args)
                stats["stage_c_lower_bound"] = True

        return pred, stats
    finally:
        _clear_runtime_gate_overrides(model)
        _clear_runtime_interaction_overrides(model)


@torch.no_grad()
def _generate_eval_prediction(
    model: SAFEModel,
    batch: Dict[str, Any],
    tokenizer,
    device: torch.device,
    modality: str,
    args: argparse.Namespace,
    active_fusion_layers: Optional[Sequence[int]] = None,
) -> str:
    mm = resolve_modality_batch(batch, modality)
    inputs = model.prepare_multimodal_inputs(
        text=build_modality_aware_questions(batch["questions"], modality, args),
        images=mm["images"],
        audio=mm["audio"],
        answers=None,
        device=str(device),
        training_mode=False,
    )
    audio_tokens = inputs.pop("audio_tokens", None)
    audio_mask = inputs.pop("audio_attention_mask", None)
    output_ids = model.generate(
        text=None,
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        pixel_values=inputs.get("pixel_values"),
        audio_tokens=audio_tokens,
        audio_attention_mask=audio_mask,
        active_fusion_layers=active_fusion_layers,
        gate=args.fusion_gate,
        max_new_tokens=args.max_answer_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_width = int(inputs["input_ids"].size(1))
    seq = output_ids[0]
    gen = seq[prompt_width:] if seq.size(0) > prompt_width else seq
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def _parse_layer_list(layer_csv: str) -> List[int]:
    if not layer_csv:
        return []
    out: List[int] = []
    for raw in layer_csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.append(int(raw))
    return sorted(set(out))


def _get_modality_fusion_layers(model: SAFEModel, modality: str) -> List[int]:
    adapter = getattr(model, "fusion_adapter", None)
    if adapter is None:
        return []

    layers: List[int] = []
    mapping = getattr(adapter, "fusion_layers", None)
    if isinstance(mapping, dict):
        vals = mapping.get(modality, [])
        if isinstance(vals, (list, tuple)):
            layers = [int(v) for v in vals]

    if not layers:
        fallback = getattr(adapter, "fusion_layer_indices", None)
        if isinstance(fallback, (list, tuple)):
            layers = [int(v) for v in fallback]

    return sorted(set(layers))


def _pool_token_states(
    hidden_states: torch.Tensor,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    # Prefer supervised target positions; fall back to attention mask.
    if labels is not None and torch.is_tensor(labels):
        token_mask = labels != -100
    elif attention_mask is not None and torch.is_tensor(attention_mask):
        token_mask = attention_mask > 0
    else:
        token_mask = torch.ones(
            hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool
        )

    mask = token_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / denom


def _extract_pooled_layer_states(
    all_hidden_states: Any,
    layer_indices: Sequence[int],
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> Dict[int, torch.Tensor]:
    if not isinstance(all_hidden_states, (list, tuple)) or len(all_hidden_states) == 0:
        return {}

    pooled: Dict[int, torch.Tensor] = {}
    n_states = len(all_hidden_states)
    for layer_idx in layer_indices:
        # HuggingFace convention: hidden_states[0] is embedding output,
        # hidden_states[l+1] is output of decoder layer l.
        candidates = [layer_idx + 1, layer_idx]
        state = None
        for idx in candidates:
            if 0 <= idx < n_states and torch.is_tensor(all_hidden_states[idx]):
                state = all_hidden_states[idx]
                break
        if state is None:
            continue
        pooled[layer_idx] = _pool_token_states(state, labels=labels, attention_mask=attention_mask)
    return pooled


def _fit_shift_basis(samples: torch.Tensor, rank: int) -> Optional[torch.Tensor]:
    if samples.ndim != 2 or samples.size(0) < 2:
        return None
    centered = samples - samples.mean(dim=0, keepdim=True)
    max_rank = min(int(rank), int(centered.size(0) - 1), int(centered.size(1)))
    if max_rank <= 0:
        return None

    try:
        # V: (d, q) principal directions in feature space
        _, _, v = torch.pca_lowrank(centered, q=max_rank, center=False, niter=2)
        basis = v[:, :max_rank].contiguous()
    except Exception:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:max_rank].transpose(0, 1).contiguous()

    return F.normalize(basis, dim=0)


def _compute_layer_weights_from_stats(
    stats_by_layer: Dict[int, Dict[str, float]],
    normalize: bool = True,
    eps: float = 1e-6,
) -> Dict[int, float]:
    """
    Inverse-energy weighting so large-shift late layers do not dominate the regularizer.
    """
    weights: Dict[int, float] = {}
    for layer, st in stats_by_layer.items():
        mean_norm = float(st.get("mean_shift_norm", 0.0))
        weights[int(layer)] = 1.0 / (mean_norm * mean_norm + eps)

    if not weights:
        return weights

    if normalize:
        avg = sum(weights.values()) / float(max(1, len(weights)))
        if avg > 0:
            for layer in list(weights.keys()):
                weights[layer] = weights[layer] / avg

    return weights


@torch.no_grad()
def collect_audio_shift_subspaces(
    model: SAFEModel,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    layer_indices: Sequence[int],
    use_bf16_amp: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Collect Δh_{a->l} = pooled_hidden(audio_on) - pooled_hidden(audio_off)
    at target layers and fit low-rank PCA bases per layer.
    """
    if not layer_indices:
        return None

    max_samples = int(getattr(args, "compat_reg_audio_samples", 0))
    min_samples = int(getattr(args, "compat_reg_min_samples", 0))
    rank = int(getattr(args, "compat_reg_rank", 0))
    if max_samples <= 0 or rank <= 0:
        return None

    def _amp_context():
        if args.fp16 and torch.cuda.is_available():
            return autocast(enabled=True, dtype=torch.float16)
        if use_bf16_amp and torch.cuda.is_available():
            return autocast(enabled=True, dtype=torch.bfloat16)
        return nullcontext()

    was_training = model.training
    model.eval()

    started = time.time()
    delta_rows: Dict[int, List[torch.Tensor]] = {int(l): [] for l in layer_indices}
    token_bank: List[torch.Tensor] = []
    mask_bank: List[Optional[torch.Tensor]] = []
    qtype_bank: List[str] = []
    bank_size = int(getattr(args, "compat_add_bank_size", 64))
    add_bank_enabled = bool(
        (
            getattr(args, "compat_add_reg_enable", False)
            or getattr(args, "compat_noharm_enable", False)
            or getattr(args, "compat_logit_fusion_enable", False)
            or getattr(args, "compat_icm_cancel_enable", False)
            or getattr(args, "compat_poe_enable", False)
            or getattr(args, "compat_gate_add_enable", False)
            or float(getattr(args, "compat_icm_identity_lambda", 0.0)) > 0.0
            or float(getattr(args, "compat_icm_small_lambda", 0.0)) > 0.0
            or float(getattr(args, "compat_icm_util_lambda", 0.0)) > 0.0
        )
        and bank_size > 0
    )
    collected = 0

    for batch in dataloader:
        mm = resolve_modality_batch(batch, "audio")
        inputs = model.prepare_multimodal_inputs(
            text=build_modality_aware_questions(batch["questions"], "audio", args),
            images=mm["images"],
            audio=mm["audio"],
            answers=batch["answers"],
            device=str(device),
            training_mode=True,
        )
        audio_tokens = inputs.pop("audio_tokens", None)
        audio_mask = inputs.pop("audio_attention_mask", None)
        if audio_tokens is None:
            continue

        if add_bank_enabled and len(token_bank) < bank_size:
            tok_cpu = audio_tokens.detach().to(device="cpu")
            mask_cpu = audio_mask.detach().to(device="cpu") if torch.is_tensor(audio_mask) else None
            qtypes = batch.get("question_types", None)
            for i in range(tok_cpu.size(0)):
                token_bank.append(tok_cpu[i:i + 1].clone())
                if mask_cpu is not None:
                    mask_bank.append(mask_cpu[i:i + 1].clone())
                else:
                    mask_bank.append(None)
                if isinstance(qtypes, (list, tuple)) and i < len(qtypes):
                    qtype_bank.append(str(qtypes[i]))
                else:
                    qtype_bank.append("unknown")
                if len(token_bank) >= bank_size:
                    break

        labels = inputs.get("labels")
        attn = inputs.get("attention_mask")

        with _amp_context():
            off_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=attn,
                labels=labels,
                pixel_values=inputs.get("pixel_values"),
                audio_tokens=audio_tokens,
                audio_attention_mask=audio_mask,
                gate=0.0,
                output_hidden_states=True,
            )
            on_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=attn,
                labels=labels,
                pixel_values=inputs.get("pixel_values"),
                audio_tokens=audio_tokens,
                audio_attention_mask=audio_mask,
                gate=args.fusion_gate,
                output_hidden_states=True,
            )

        off_hs = off_outputs.get("all_hidden_states") if isinstance(off_outputs, dict) else None
        on_hs = on_outputs.get("all_hidden_states") if isinstance(on_outputs, dict) else None
        if off_hs is None or on_hs is None:
            continue

        off_pooled = _extract_pooled_layer_states(off_hs, layer_indices, labels=labels, attention_mask=attn)
        on_pooled = _extract_pooled_layer_states(on_hs, layer_indices, labels=labels, attention_mask=attn)
        if not off_pooled or not on_pooled:
            continue

        for layer in layer_indices:
            if layer not in off_pooled or layer not in on_pooled:
                continue
            delta = (on_pooled[layer] - off_pooled[layer]).detach().float().cpu()
            if delta.numel() == 0:
                continue
            delta_rows[layer].append(delta)

        collected += int(inputs["input_ids"].size(0))
        if collected >= max_samples:
            break

    if was_training:
        model.train()

    basis_by_layer: Dict[int, torch.Tensor] = {}
    stats_by_layer: Dict[int, Dict[str, float]] = {}

    for layer in layer_indices:
        rows = delta_rows.get(layer, [])
        if not rows:
            continue
        mat = torch.cat(rows, dim=0)
        if mat.size(0) > max_samples:
            mat = mat[:max_samples]
        if mat.size(0) < max(2, min_samples):
            continue

        basis = _fit_shift_basis(mat, rank=rank)
        if basis is None:
            continue

        basis_by_layer[int(layer)] = basis
        stats_by_layer[int(layer)] = {
            "num_samples": float(mat.size(0)),
            "rank": float(basis.size(1)),
            "mean_shift_norm": float(mat.norm(dim=1).mean().item()),
            "std_shift_norm": float(mat.norm(dim=1).std(unbiased=False).item()),
        }

    elapsed = time.time() - started
    if not basis_by_layer:
        if add_bank_enabled and token_bank:
            print(
                f"[compat] No shift basis fit (layers={list(layer_indices)}; "
                f"collected={collected}); returning token-bank-only state "
                f"(bank={len(token_bank)}; elapsed={elapsed/60.0:.1f}m)",
                flush=True,
            )
            return {
                "layers": [int(l) for l in sorted(set(int(x) for x in layer_indices))],
                "basis_by_layer": {},
                "stats_by_layer": {},
                "layer_weights": {},
                "audio_token_bank": token_bank,
                "audio_mask_bank": mask_bank,
                "audio_qtype_bank": qtype_bank,
                "num_samples_collected": int(collected),
            }
        print(
            f"[compat] Failed to build audio shift bases (layers={list(layer_indices)}; "
            f"collected={collected}; elapsed={elapsed/60.0:.1f}m)",
            flush=True,
        )
        return None

    layer_weights = _compute_layer_weights_from_stats(
        stats_by_layer,
        normalize=bool(getattr(args, "compat_reg_weight_by_shift_norm", True)),
    )

    print(
        f"[compat] Collected audio shift subspaces on {len(basis_by_layer)}/{len(layer_indices)} layers "
        f"(samples={collected}, elapsed={elapsed/60.0:.1f}m)",
        flush=True,
    )
    for layer in sorted(basis_by_layer.keys()):
        st = stats_by_layer[layer]
        print(
            f"[compat] layer={layer} rank={int(st['rank'])} n={int(st['num_samples'])} "
            f"mean_shift_norm={st['mean_shift_norm']:.4f} "
            f"weight={layer_weights.get(layer, 1.0):.4f}",
            flush=True,
        )
    if add_bank_enabled:
        print(f"[compat] additivity audio token bank size={len(token_bank)}", flush=True)

    return {
        "layers": [int(l) for l in sorted(basis_by_layer.keys())],
        "basis_by_layer": basis_by_layer,
        "stats_by_layer": stats_by_layer,
        "layer_weights": layer_weights,
        "audio_token_bank": token_bank,
        "audio_mask_bank": mask_bank,
        "audio_qtype_bank": qtype_bank,
        "num_samples_collected": int(collected),
    }


@torch.no_grad()
def _collect_rkca_token_bank(
    model: SAFEModel,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    modality: str,
    bank_size: int = 128,
    use_bf16_amp: bool = False,
) -> Optional[torch.Tensor]:
    """Collect projector token outputs for one modality.

    Returns: (bank_size, num_tokens, hidden_dim) tensor on CPU, detached.
    Returns None if no tokens could be collected.
    """
    model.eval()
    collected: List[torch.Tensor] = []
    token_key = "audio_projector_tokens" if modality == "audio" else "vision_projector_tokens"
    resolve_mod = "audio" if modality == "audio" else "image"

    def _amp_ctx():
        if use_bf16_amp and torch.cuda.is_available():
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    for batch in dataloader:
        if len(collected) >= bank_size:
            break
        mm = resolve_modality_batch(batch, resolve_mod)
        try:
            inputs = model.prepare_multimodal_inputs(
                text=build_modality_aware_questions(batch["questions"], resolve_mod, args),
                images=mm["images"],
                audio=mm["audio"],
                device=device,
            )
        except Exception:
            continue

        audio_tokens_in = inputs.pop("audio_tokens", None)
        audio_mask_in = inputs.pop("audio_attention_mask", None)

        try:
            with _amp_ctx():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    labels=inputs.get("labels"),
                    pixel_values=inputs.get("pixel_values"),
                    audio_tokens=audio_tokens_in,
                    audio_attention_mask=audio_mask_in,
                    gate=1.0,
                )
        except Exception:
            continue

        tokens = outputs.get(token_key) if isinstance(outputs, dict) else None
        if tokens is not None:
            collected.append(tokens.detach().cpu().float())

    model.train()
    if not collected:
        return None
    # Stack to (bank_size, num_tokens, hidden_dim)
    bank = torch.cat(collected, dim=0)[:bank_size]
    return bank


def compute_rkca_subspace_loss(
    current_tokens: torch.Tensor,      # (B, N_tokens, H) — has grad
    banked_basis: torch.Tensor,         # (H, rank) — no grad, on device
) -> torch.Tensor:
    """Penalize current tokens' projection onto the other modality's subspace."""
    # Flatten tokens: (B*N, H)
    flat = current_tokens.reshape(-1, current_tokens.size(-1))
    # Project onto basis: (B*N, rank)
    proj = flat.float().matmul(banked_basis.float())
    # Loss = mean squared projection magnitude
    return proj.pow(2).mean()


@torch.no_grad()
def collect_vision_shift_subspaces(
    model: SAFEModel,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    layer_indices: Sequence[int],
    use_bf16_amp: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Collect Δh_{v->l} = pooled_hidden(vision_on) - pooled_hidden(vision_off)
    at *audio* fusion layers and fit low-rank PCA bases per layer.

    This is the **reverse** of collect_audio_shift_subspaces(): we measure
    InternVL's built-in vision activation patterns so the audio adapter can
    be penalized for projecting onto them.

    Both passes use gate=0 (audio OFF) so only vision presence varies:
      Pass 1: pixel_values=pv, gate=0  → vision ON,  audio OFF
      Pass 2: pixel_values=None, gate=0 → vision OFF, audio OFF
    """
    if not layer_indices:
        return None

    max_samples = int(getattr(args, "compat_reg_audio_samples", 0))
    min_samples = int(getattr(args, "compat_reg_min_samples", 0))
    rank = int(getattr(args, "compat_reg_rank", 0))
    if max_samples <= 0 or rank <= 0:
        return None

    def _amp_context():
        if args.fp16 and torch.cuda.is_available():
            return autocast(enabled=True, dtype=torch.float16)
        if use_bf16_amp and torch.cuda.is_available():
            return autocast(enabled=True, dtype=torch.bfloat16)
        return nullcontext()

    was_training = model.training
    model.eval()

    started = time.time()
    delta_rows: Dict[int, List[torch.Tensor]] = {int(l): [] for l in layer_indices}
    collected = 0

    print(
        f"[compat-reverse] Collecting vision shift subspaces at audio layers={list(layer_indices)}",
        flush=True,
    )

    for batch in dataloader:
        mm = resolve_modality_batch(batch, "both")
        inputs = model.prepare_multimodal_inputs(
            text=build_modality_aware_questions(batch["questions"], "both", args),
            images=mm["images"],
            audio=mm["audio"],
            answers=batch["answers"],
            device=str(device),
            training_mode=True,
        )
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            continue

        # Pop audio tokens — we don't inject audio in either pass
        audio_tokens = inputs.pop("audio_tokens", None)
        audio_mask = inputs.pop("audio_attention_mask", None)

        labels = inputs.get("labels")
        attn = inputs.get("attention_mask")

        with _amp_context():
            # Pass 1: vision ON, audio OFF (gate=0)
            on_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=attn,
                labels=labels,
                pixel_values=pixel_values,
                audio_tokens=None,
                audio_attention_mask=None,
                gate=0.0,
                output_hidden_states=True,
            )
            # Pass 2: vision OFF, audio OFF (gate=0)
            off_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=attn,
                labels=labels,
                pixel_values=None,
                audio_tokens=None,
                audio_attention_mask=None,
                gate=0.0,
                output_hidden_states=True,
            )

        on_hs = on_outputs.get("all_hidden_states") if isinstance(on_outputs, dict) else None
        off_hs = off_outputs.get("all_hidden_states") if isinstance(off_outputs, dict) else None
        if on_hs is None or off_hs is None:
            continue

        on_pooled = _extract_pooled_layer_states(on_hs, layer_indices, labels=labels, attention_mask=attn)
        off_pooled = _extract_pooled_layer_states(off_hs, layer_indices, labels=labels, attention_mask=attn)
        if not on_pooled or not off_pooled:
            continue

        for layer in layer_indices:
            if layer not in on_pooled or layer not in off_pooled:
                continue
            delta = (on_pooled[layer] - off_pooled[layer]).detach().float().cpu()
            if delta.numel() == 0:
                continue
            delta_rows[layer].append(delta)

        collected += int(inputs["input_ids"].size(0))
        if collected >= max_samples:
            break

    if was_training:
        model.train()

    basis_by_layer: Dict[int, torch.Tensor] = {}
    stats_by_layer: Dict[int, Dict[str, float]] = {}

    for layer in layer_indices:
        rows = delta_rows.get(layer, [])
        if not rows:
            continue
        mat = torch.cat(rows, dim=0)
        if mat.size(0) > max_samples:
            mat = mat[:max_samples]
        if mat.size(0) < max(2, min_samples):
            continue

        basis = _fit_shift_basis(mat, rank=rank)
        if basis is None:
            continue

        basis_by_layer[int(layer)] = basis
        stats_by_layer[int(layer)] = {
            "num_samples": float(mat.size(0)),
            "rank": float(basis.size(1)),
            "mean_shift_norm": float(mat.norm(dim=1).mean().item()),
            "std_shift_norm": float(mat.norm(dim=1).std(unbiased=False).item()),
        }

    elapsed = time.time() - started
    if not basis_by_layer:
        print(
            f"[compat-reverse] Failed to build vision shift bases (layers={list(layer_indices)}; "
            f"collected={collected}; elapsed={elapsed/60.0:.1f}m)",
            flush=True,
        )
        return None

    layer_weights = _compute_layer_weights_from_stats(
        stats_by_layer,
        normalize=bool(getattr(args, "compat_reg_weight_by_shift_norm", True)),
    )

    print(
        f"[compat-reverse] Collected vision shift subspaces on {len(basis_by_layer)}/{len(layer_indices)} layers "
        f"(samples={collected}, elapsed={elapsed/60.0:.1f}m)",
        flush=True,
    )
    for layer in sorted(basis_by_layer.keys()):
        st = stats_by_layer[layer]
        print(
            f"[compat-reverse] layer={layer} rank={int(st['rank'])} n={int(st['num_samples'])} "
            f"mean_shift_norm={st['mean_shift_norm']:.4f} "
            f"weight={layer_weights.get(layer, 1.0):.4f}",
            flush=True,
        )

    return {
        "layers": [int(l) for l in sorted(basis_by_layer.keys())],
        "basis_by_layer": basis_by_layer,
        "stats_by_layer": stats_by_layer,
        "layer_weights": layer_weights,
        "audio_token_bank": [],
        "audio_mask_bank": [],
        "audio_qtype_bank": [],
        "num_samples_collected": int(collected),
    }


def compute_vision_subspace_regularizer(
    all_hidden_states: Any,
    baseline_pooled: Dict[int, torch.Tensor],
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    basis_by_layer: Dict[int, torch.Tensor],
    layers: Sequence[int],
    layer_weights: Optional[Dict[int, float]] = None,
) -> tuple[Optional[torch.Tensor], Dict[int, float]]:
    if not basis_by_layer:
        return None, {}

    current_pooled = _extract_pooled_layer_states(
        all_hidden_states,
        layer_indices=layers,
        labels=labels,
        attention_mask=attention_mask,
    )
    if not current_pooled:
        return None, {}

    reg_loss: Optional[torch.Tensor] = None
    per_layer: Dict[int, float] = {}
    for layer in layers:
        if layer not in basis_by_layer:
            continue
        if layer not in current_pooled or layer not in baseline_pooled:
            continue

        cur = current_pooled[layer]
        base = baseline_pooled[layer]
        if base.device != cur.device:
            base = base.to(cur.device)
        delta = cur - base

        basis = basis_by_layer[layer]
        if basis.device != cur.device:
            basis = basis.to(cur.device)

        proj = delta.float().matmul(basis.float())  # (B, rank)
        layer_loss = (proj.pow(2).sum(dim=1)).mean()
        weight = 1.0
        if layer_weights is not None:
            weight = float(layer_weights.get(int(layer), 1.0))
        weighted = layer_loss * weight
        per_layer[int(layer)] = float(weighted.detach().item())
        reg_loss = weighted if reg_loss is None else (reg_loss + weighted)

    return reg_loss, per_layer


def _sample_audio_bank_entry(
    compatibility_state: Dict[str, Any],
    preferred_types: Optional[Sequence[str]] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    token_bank = compatibility_state.get("audio_token_bank", [])
    mask_bank = compatibility_state.get("audio_mask_bank", [])
    qtype_bank = compatibility_state.get("audio_qtype_bank", [])
    if not token_bank:
        return None, None

    idx: Optional[int] = None
    if preferred_types and qtype_bank:
        preferred = {str(x) for x in preferred_types if x is not None}
        if preferred:
            candidates = [
                i for i, qt in enumerate(qtype_bank[: len(token_bank)])
                if str(qt) in preferred
            ]
            if candidates:
                idx = random.choice(candidates)

    if idx is None:
        idx = random.randrange(len(token_bank))

    tok = token_bank[idx]
    msk = mask_bank[idx] if idx < len(mask_bank) else None
    return tok, msk


def _parse_modality_layers(
    model: SAFEModel,
) -> tuple[List[int], List[int]]:
    fusion_adapter = getattr(model, "fusion_adapter", None)
    if fusion_adapter is None:
        return [], []

    fusion_layers = getattr(fusion_adapter, "fusion_layers", None)
    if isinstance(fusion_layers, dict):
        audio_layers = sorted(int(x) for x in fusion_layers.get("audio", []))
        vision_layers = sorted(int(x) for x in fusion_layers.get("vision", []))
        return audio_layers, vision_layers

    return [], []


def _build_gate_pairs(
    audio_layers: Sequence[int],
    vision_layers: Sequence[int],
    pairing: str = "zip",
) -> List[tuple[str, str, int]]:
    pairing = str(pairing).lower().strip()
    a_sorted = sorted(int(x) for x in audio_layers)
    v_sorted = sorted(int(x) for x in vision_layers)
    pairs: List[tuple[str, str, int]] = []

    shared = sorted(set(a_sorted).intersection(v_sorted))
    if pairing == "shared" and shared:
        for layer in shared:
            pairs.append((f"audio:{layer}", f"vision:{layer}", int(layer)))
        return pairs

    if pairing == "all":
        for a in a_sorted:
            for v in v_sorted:
                pairs.append((f"audio:{a}", f"vision:{v}", int(v)))
        return pairs

    # Default: zip by depth order (audio_i -> vision_i). If shared exists, use it first.
    if shared:
        for layer in shared:
            pairs.append((f"audio:{layer}", f"vision:{layer}", int(layer)))
        return pairs

    for a, v in zip(a_sorted, v_sorted):
        pairs.append((f"audio:{a}", f"vision:{v}", int(v)))
    return pairs


def compute_gate_additivity_objective(
    model: SAFEModel,
    gate_value: float,
    rho_proxy_by_layer: Optional[Dict[int, float]] = None,
    fallback_layer_weights: Optional[Dict[int, float]] = None,
    pairing: str = "zip",
    target_mode: str = "inverse_rho",
    target_product: float = -1.0,
    rho_beta: float = 2.0,
    min_effective: float = 0.0,
    floor_lambda: float = 0.0,
) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
    """
    Additivity-first gate objective:
      sum_{pairs} rho_l * (g_a * g_v - p*_l)^2
    where p*_l is fixed or inverse-rho target.
    """
    stats: Dict[str, float] = {}
    fusion_adapter = getattr(model, "fusion_adapter", None)
    if fusion_adapter is None:
        return None, stats

    layer_gates = getattr(fusion_adapter, "layer_gates", None)
    if layer_gates is None or len(layer_gates) == 0:
        return None, stats

    audio_layers, vision_layers = _parse_modality_layers(model)
    pairs = _build_gate_pairs(audio_layers, vision_layers, pairing=pairing)
    if not pairs:
        return None, stats

    # Effective modality gate magnitudes (include global scalar gate).
    gate_mag: Dict[str, torch.Tensor] = {}
    for key, param in layer_gates.items():
        g = torch.abs(torch.tanh(param)) * float(gate_value)
        gate_mag[str(key)] = g

    base_target = float(target_product)
    if base_target < 0.0:
        base_target = float(gate_value) * float(gate_value)

    rho_source = rho_proxy_by_layer or {}
    if not rho_source and fallback_layer_weights:
        # Fallback proxy when online residual unavailable.
        rho_source = {int(k): float(v) for k, v in fallback_layer_weights.items()}
    rho_default = float(sum(rho_source.values()) / max(1, len(rho_source))) if rho_source else 1.0
    rho_beta = max(0.0, float(rho_beta))
    min_effective = max(0.0, float(min_effective))
    floor_lambda = max(0.0, float(floor_lambda))

    pair_terms: List[torch.Tensor] = []
    prod_vals: List[float] = []
    target_vals: List[float] = []
    rho_vals: List[float] = []
    used_gate_keys: List[str] = []

    mode = str(target_mode).lower().strip()
    for a_key, v_key, ref_layer in pairs:
        if a_key not in gate_mag or v_key not in gate_mag:
            continue
        ga = gate_mag[a_key]
        gv = gate_mag[v_key]
        prod = ga * gv
        rho_l = float(rho_source.get(int(ref_layer), rho_default))
        rho_l = max(0.0, rho_l)
        if mode == "inverse_rho":
            tgt = base_target / (1.0 + rho_beta * rho_l)
        else:
            tgt = base_target
        rho_t = torch.tensor(rho_l, device=prod.device, dtype=prod.dtype)
        tgt_t = torch.tensor(float(tgt), device=prod.device, dtype=prod.dtype)
        pair_terms.append(rho_t * (prod - tgt_t).pow(2))
        prod_vals.append(float(prod.detach().item()))
        target_vals.append(float(tgt))
        rho_vals.append(float(rho_l))
        used_gate_keys.extend([a_key, v_key])

    if not pair_terms:
        return None, stats

    loss = torch.stack(pair_terms).mean()

    if floor_lambda > 0.0 and min_effective > 0.0:
        floor_terms: List[torch.Tensor] = []
        gmin_t = None
        for key in set(used_gate_keys):
            g = gate_mag[key]
            if gmin_t is None:
                gmin_t = torch.tensor(min_effective, device=g.device, dtype=g.dtype)
            floor_terms.append(F.relu(gmin_t - g).pow(2))
        if floor_terms:
            floor_loss = torch.stack(floor_terms).mean()
            loss = loss + floor_lambda * floor_loss
            stats["gate_floor_loss"] = float(floor_loss.detach().item())

    stats["gate_pair_count"] = float(len(pair_terms))
    stats["gate_avg_product"] = float(sum(prod_vals) / max(1, len(prod_vals)))
    stats["gate_avg_target"] = float(sum(target_vals) / max(1, len(target_vals)))
    stats["gate_avg_rho"] = float(sum(rho_vals) / max(1, len(rho_vals)))
    return loss, stats


def compute_unpaired_additivity_regularizer(
    model: SAFEModel,
    inputs: Dict[str, Any],
    compatibility_state: Dict[str, Any],
    gate_value: float,
    add_layers: Sequence[int],
    current_question_types: Optional[Sequence[str]] = None,
    transport_layers: Optional[Sequence[int]] = None,
    layer_weights: Optional[Dict[int, float]] = None,
    normalize: bool = True,
    no_harm_enable: bool = False,
    no_harm_margin: float = 0.0,
    no_harm_use_best_single: bool = False,
    logit_fusion_enable: bool = False,
    logit_fusion_conf_temp: float = 0.5,
    transport_enable: bool = False,
    transport_cap: float = 0.0,
    transport_normalize: bool = True,
    icm_cancel_enable: bool = False,
    icm_cancel_loss_type: str = "mse",
    icm_cancel_logit_temp: float = 1.0,
    poe_enable: bool = False,
    poe_weight_temp: float = 0.5,
    poe_loss_type: str = "kl",
    poe_logit_temp: float = 1.0,
    routing_enable: bool = False,
    routing_min: float = 0.25,
    routing_max: float = 1.0,
) -> tuple[Optional[torch.Tensor], Dict[int, float], Dict[int, float], Dict[str, Optional[torch.Tensor]], Dict[str, float]]:
    """
    Unpaired additivity loss (no joint AV supervision):
      ||Δ_av - Δ_a - Δ_v||^2 at selected layers.
    Audio is sampled from the compatibility token bank, preferring
    matching question types when available (stratified synthetic pairs).
    """
    aux_losses: Dict[str, Optional[torch.Tensor]] = {
        "no_harm": None,
        "logit_fusion": None,
        "transport": None,
        "icm_cancel": None,
        "icm_identity": None,
        "icm_small": None,
        "icm_util": None,
        "poe_consistency": None,
    }
    aux_stats: Dict[str, float] = {}

    add_layers = [int(layer) for layer in add_layers]
    if transport_layers is None:
        transport_layers = list(add_layers)
    transport_layers = [int(layer) for layer in transport_layers]

    need_hidden_states = bool(add_layers or transport_layers)
    if not need_hidden_states and not (
        no_harm_enable
        or logit_fusion_enable
        or transport_enable
        or icm_cancel_enable
        or poe_enable
    ):
        return None, {}, {}, aux_losses, aux_stats

    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        return None, {}, {}, aux_losses, aux_stats

    sampled_audio, sampled_mask = _sample_audio_bank_entry(
        compatibility_state,
        preferred_types=current_question_types,
    )
    if sampled_audio is None:
        return None, {}, {}, aux_losses, aux_stats

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    labels = inputs.get("labels")
    sampled_audio = sampled_audio.to(device=input_ids.device)
    sampled_mask = sampled_mask.to(device=input_ids.device) if torch.is_tensor(sampled_mask) else None

    no_harm_enable = bool(no_harm_enable)
    no_harm_margin = float(no_harm_margin)
    no_harm_use_best_single = bool(no_harm_use_best_single)
    logit_fusion_enable = bool(logit_fusion_enable)
    logit_fusion_conf_temp = float(logit_fusion_conf_temp)
    transport_enable = bool(transport_enable)
    transport_cap = float(transport_cap)
    transport_normalize = bool(transport_normalize)
    icm_cancel_enable = bool(icm_cancel_enable)
    icm_cancel_loss_type = str(icm_cancel_loss_type).lower().strip()
    icm_cancel_logit_temp = float(icm_cancel_logit_temp)
    poe_enable = bool(poe_enable)
    poe_weight_temp = float(poe_weight_temp)
    poe_loss_type = str(poe_loss_type).lower().strip()
    poe_logit_temp = float(poe_logit_temp)
    routing_enable = bool(routing_enable)
    routing_min = float(routing_min)
    routing_max = float(routing_max)
    routing_max = max(routing_min, routing_max)
    routed_gate = float(gate_value)

    # Text-only and audio-only terms are constants for this regularizer.
    # IMPORTANT: avoid the pure passthrough branch (no modalities), because some
    # model paths only return `hidden_states` there and omit `all_hidden_states`.
    # We keep audio tokens present and set gate=0.0 to force hook-capable path
    # while remaining semantically text-only.
    with torch.no_grad():
        out_text = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=None,
            audio_tokens=sampled_audio,
            audio_attention_mask=sampled_mask,
            gate=0.0,
            output_hidden_states=need_hidden_states,
        )
        out_audio = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=None,
            audio_tokens=sampled_audio,
            audio_attention_mask=sampled_mask,
            gate=gate_value,
            output_hidden_states=need_hidden_states,
        )

    conf_audio = _confidence_from_logits(
        out_audio.get("logits") if isinstance(out_audio, dict) else getattr(out_audio, "logits", None),
        labels=labels,
    )
    aux_stats["conf_audio"] = float(conf_audio)
    if routing_enable:
        route_scale = routing_min + (routing_max - routing_min) * conf_audio
        route_scale = float(max(routing_min, min(routing_max, route_scale)))
        routed_gate = float(gate_value) * route_scale
        aux_stats["route_audio_scale"] = route_scale
    else:
        aux_stats["route_audio_scale"] = 1.0

    # Vision-only and both carry gradients to vision adapters.
    out_vision = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        pixel_values=pixel_values,
        audio_tokens=None,
        audio_attention_mask=None,
        gate=gate_value,
        output_hidden_states=need_hidden_states,
    )
    out_both = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        pixel_values=pixel_values,
        audio_tokens=sampled_audio,
        audio_attention_mask=sampled_mask,
        gate=routed_gate,
        output_hidden_states=need_hidden_states,
    )

    conf_vision = _confidence_from_logits(
        out_vision.get("logits") if isinstance(out_vision, dict) else getattr(out_vision, "logits", None),
        labels=labels,
    )
    aux_stats["conf_vision"] = float(conf_vision)
    aux_stats["routed_gate"] = float(routed_gate)

    def _get_out_tensor(out_obj: Any, key: str) -> Optional[torch.Tensor]:
        if isinstance(out_obj, dict):
            t = out_obj.get(key)
            if torch.is_tensor(t):
                return t
        return None

    # ICM auxiliary terms emitted by SAFEModel forward.
    id_terms: List[torch.Tensor] = []
    for src in (out_audio, out_vision):
        t = _get_out_tensor(src, "icm_identity_loss")
        if t is not None:
            id_terms.append(t)
    if id_terms:
        aux_losses["icm_identity"] = torch.stack([t.float() for t in id_terms]).mean()

    t_small = _get_out_tensor(out_both, "icm_small_loss")
    if t_small is not None:
        aux_losses["icm_small"] = t_small.float()

    t_util = _get_out_tensor(out_both, "icm_util_loss")
    if t_util is not None:
        aux_losses["icm_util"] = t_util.float()

    t_entropy = _get_out_tensor(out_both, "icm_entropy")
    if t_entropy is not None:
        aux_stats["icm_entropy"] = float(t_entropy.detach().item())
    t_gate = _get_out_tensor(out_both, "icm_gate_mean")
    if t_gate is not None:
        aux_stats["icm_gate_mean"] = float(t_gate.detach().item())
    t_corr = _get_out_tensor(out_both, "icm_correction_norm")
    if t_corr is not None:
        aux_stats["icm_corr_norm"] = float(t_corr.detach().item())

    reg_loss: Optional[torch.Tensor] = None
    per_layer: Dict[int, float] = {}
    rho_proxy_by_layer: Dict[int, float] = {}
    if need_hidden_states:
        hs_text = out_text.get("all_hidden_states") if isinstance(out_text, dict) else None
        hs_audio = out_audio.get("all_hidden_states") if isinstance(out_audio, dict) else None
        hs_vision = out_vision.get("all_hidden_states") if isinstance(out_vision, dict) else None
        hs_both = out_both.get("all_hidden_states") if isinstance(out_both, dict) else None
        if any(x is None for x in (hs_text, hs_audio, hs_vision, hs_both)):
            return None, {}, {}, aux_losses, aux_stats

        pooled_layers = sorted(set(add_layers) | set(transport_layers))
        pooled_t = _extract_pooled_layer_states(hs_text, pooled_layers, labels=labels, attention_mask=attention_mask)
        pooled_a = _extract_pooled_layer_states(hs_audio, pooled_layers, labels=labels, attention_mask=attention_mask)
        pooled_v = _extract_pooled_layer_states(hs_vision, pooled_layers, labels=labels, attention_mask=attention_mask)
        pooled_av = _extract_pooled_layer_states(hs_both, pooled_layers, labels=labels, attention_mask=attention_mask)

        for layer in add_layers:
            if layer not in pooled_t or layer not in pooled_a or layer not in pooled_v or layer not in pooled_av:
                continue

            dt = pooled_t[layer]
            da = (pooled_a[layer] - dt).detach()
            dv = pooled_v[layer] - dt
            dav = pooled_av[layer] - dt
            residual = dav - da - dv

            layer_loss = residual.pow(2).mean()
            if normalize:
                denom = da.pow(2).mean().detach() + dv.pow(2).mean().detach() + 1e-6
                layer_loss = layer_loss / denom
            rho_proxy_by_layer[int(layer)] = float(layer_loss.detach().item())

            weight = 1.0
            if layer_weights is not None:
                weight = float(layer_weights.get(int(layer), 1.0))
            weighted = layer_loss * weight
            per_layer[int(layer)] = float(weighted.detach().item())
            reg_loss = weighted if reg_loss is None else (reg_loss + weighted)

        if transport_enable:
            transport_loss: Optional[torch.Tensor] = None
            transport_vals: List[float] = []
            for layer in transport_layers:
                if layer not in pooled_v or layer not in pooled_av:
                    continue
                # Transported shift at destination layer:
                # how much enabling audio perturbs vision hidden states.
                shift = pooled_av[layer] - pooled_v[layer]
                layer_t = shift.pow(2).mean()
                if transport_normalize:
                    denom_t = pooled_v[layer].pow(2).mean().detach() + 1e-6
                    layer_t = layer_t / denom_t
                if transport_cap > 0.0:
                    layer_t = F.relu(layer_t - float(transport_cap)).pow(2)
                weight = 1.0
                if layer_weights is not None:
                    weight = float(layer_weights.get(int(layer), 1.0))
                layer_t = layer_t * weight
                transport_vals.append(float(layer_t.detach().item()))
                transport_loss = layer_t if transport_loss is None else (transport_loss + layer_t)
            if transport_loss is not None:
                aux_losses["transport"] = transport_loss
                aux_stats["transport_layer_mean"] = float(sum(transport_vals) / max(1, len(transport_vals)))

    nll_text = _sequence_nll_from_output(out_text, labels=labels, require_grad=False)
    nll_audio = _sequence_nll_from_output(out_audio, labels=labels, require_grad=False)
    nll_vision = _sequence_nll_from_output(out_vision, labels=labels, require_grad=False)
    nll_both = _sequence_nll_from_output(out_both, labels=labels, require_grad=True)

    if no_harm_enable and nll_both is not None and nll_vision is not None:
        ref = nll_vision.detach()
        if no_harm_use_best_single and nll_audio is not None:
            ref = torch.minimum(ref, nll_audio.detach())
        aux_losses["no_harm"] = F.relu(nll_both - ref + no_harm_margin)
        aux_stats["nll_ref"] = float(ref.item())
        aux_stats["nll_both"] = float(nll_both.detach().item())
        aux_stats["nll_gap_both_minus_ref"] = float((nll_both.detach() - ref).item())

    if logit_fusion_enable and nll_both is not None and nll_audio is not None and nll_vision is not None:
        # Confidence-calibrated single-modality mixture in NLL space.
        # Lower confidence => lower contribution in fusion target.
        temp = max(1e-3, logit_fusion_conf_temp)
        wa = math.exp(conf_audio / temp)
        wv = math.exp(conf_vision / temp)
        wsum = max(1e-6, wa + wv)
        wa /= wsum
        wv /= wsum

        nll_t = nll_text.detach() if nll_text is not None else nll_vision.detach()
        nll_target = nll_t + wa * (nll_audio.detach() - nll_t) + wv * (nll_vision.detach() - nll_t)
        aux_losses["logit_fusion"] = (nll_both - nll_target).pow(2)
        aux_stats["fusion_w_audio"] = float(wa)
        aux_stats["fusion_w_vision"] = float(wv)
        aux_stats["nll_fusion_target"] = float(nll_target.item())

    if icm_cancel_enable or poe_enable:
        logits_text = out_text.get("logits") if isinstance(out_text, dict) else getattr(out_text, "logits", None)
        logits_audio = out_audio.get("logits") if isinstance(out_audio, dict) else getattr(out_audio, "logits", None)
        logits_vision = out_vision.get("logits") if isinstance(out_vision, dict) else getattr(out_vision, "logits", None)
        logits_both = out_both.get("logits") if isinstance(out_both, dict) else getattr(out_both, "logits", None)

        flat_t = _flatten_valid_logits(logits_text, labels=labels)
        flat_a = _flatten_valid_logits(logits_audio, labels=labels)
        flat_v = _flatten_valid_logits(logits_vision, labels=labels)
        flat_av = _flatten_valid_logits(logits_both, labels=labels)

        if all(torch.is_tensor(x) for x in (flat_t, flat_a, flat_v, flat_av)):
            # ICM cancel objective: match corrected both logits to additive ideal.
            # z* = z_a + z_v - z_0 (unpaired synthetic composition target).
            if icm_cancel_enable:
                z0 = flat_t.detach().float()
                za = flat_a.detach().float()
                zv = flat_v.detach().float()
                zav = flat_av.float()
                z_star = za + zv - z0
                if icm_cancel_loss_type == "kl":
                    t = max(1e-3, icm_cancel_logit_temp)
                    target = F.softmax(z_star / t, dim=-1)
                    pred_log = F.log_softmax(zav / t, dim=-1)
                    aux_losses["icm_cancel"] = F.kl_div(pred_log, target, reduction="batchmean")
                else:
                    aux_losses["icm_cancel"] = (zav - z_star).pow(2).mean()

            if poe_enable:
                # Confidence-weighted residual PoE:
                # z_poe = z0 + alpha*(za-z0) + beta*(zv-z0)
                # alpha,beta from confidence softmax.
                wtemp = max(1e-3, poe_weight_temp)
                alpha = math.exp(conf_audio / wtemp)
                beta = math.exp(conf_vision / wtemp)
                wsum = max(1e-6, alpha + beta)
                alpha /= wsum
                beta /= wsum

                z0 = flat_t.detach().float()
                za = flat_a.detach().float()
                zv = flat_v.detach().float()
                zav = flat_av.float()
                z_poe = z0 + alpha * (za - z0) + beta * (zv - z0)

                if poe_loss_type == "mse":
                    poe_loss = (zav - z_poe).pow(2).mean()
                else:
                    # KL( p_poe || p_av ): composed logits should match PoE target.
                    t = max(1e-3, poe_logit_temp)
                    tgt = F.softmax(z_poe / t, dim=-1)
                    pred_log = F.log_softmax(zav / t, dim=-1)
                    poe_loss = F.kl_div(pred_log, tgt, reduction="batchmean")

                aux_losses["poe_consistency"] = poe_loss
                aux_stats["poe_alpha"] = float(alpha)
                aux_stats["poe_beta"] = float(beta)

    return reg_loss, per_layer, rho_proxy_by_layer, aux_losses, aux_stats


def train_epoch(
    model: SAFEModel,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: Optional[LambdaLR],
    scaler: GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    use_bf16_amp: bool = False,
    wandb_run: Any = None,
    global_step: int = 0,
    compatibility_state: Optional[Dict[str, Any]] = None,
    rkca_subspace_state: Optional[Dict[str, Any]] = None,
) -> tuple:
    model.train()
    total_loss = 0.0
    total_compat_reg = 0.0
    compat_reg_batches = 0
    total_add_reg = 0.0
    add_reg_batches = 0
    total_noharm_reg = 0.0
    noharm_reg_batches = 0
    total_logit_reg = 0.0
    logit_reg_batches = 0
    total_transport_reg = 0.0
    transport_reg_batches = 0
    total_poe_reg = 0.0
    poe_reg_batches = 0
    total_icm_cancel_reg = 0.0
    icm_cancel_reg_batches = 0
    total_icm_identity_reg = 0.0
    icm_identity_reg_batches = 0
    total_icm_small_reg = 0.0
    icm_small_reg_batches = 0
    total_icm_util_reg = 0.0
    icm_util_reg_batches = 0
    total_icm_entropy = 0.0
    icm_entropy_batches = 0
    total_gateadd_reg = 0.0
    gateadd_reg_batches = 0
    total_gateadd_prod = 0.0
    gateadd_prod_batches = 0
    total_route_scale = 0.0
    route_scale_batches = 0
    total_subspace_reg = 0.0
    subspace_reg_batches = 0
    total_batches = 0
    num_batches = len(dataloader)
    log_every = max(1, min(100, num_batches // 20))  # Log at least every 100 steps
    optimizer.zero_grad()
    trainable_for_clip = [p for p in model.parameters() if p.requires_grad]

    compat_reg_reverse = bool(getattr(args, "compat_reg_reverse", False))
    compat_state_available = bool(
        compatibility_state is not None
        and (
            (args.train_modality == "image" and not compat_reg_reverse)
            or (args.train_modality == "audio" and compat_reg_reverse)
        )
    )
    compat_layers = compatibility_state.get("layers", []) if compat_state_available else []
    compat_basis = compatibility_state.get("basis_by_layer", {}) if compat_state_available else {}
    compat_layer_weights = compatibility_state.get("layer_weights", {}) if compat_state_available else {}
    compat_reg_active = bool(
        compat_state_available
        and getattr(args, "compat_reg_enable", False)
        and bool(compat_basis)
    )
    compat_warned_hidden = False
    unpaired_aux_enabled = bool(
        compat_state_available
        and (
            getattr(args, "compat_add_reg_enable", False)
            or getattr(args, "compat_noharm_enable", False)
            or getattr(args, "compat_logit_fusion_enable", False)
            or getattr(args, "compat_transport_enable", False)
            or getattr(args, "compat_icm_cancel_enable", False)
            or getattr(args, "compat_poe_enable", False)
            or getattr(args, "compat_gate_add_enable", False)
            or float(getattr(args, "compat_icm_identity_lambda", 0.0)) > 0.0
            or float(getattr(args, "compat_icm_small_lambda", 0.0)) > 0.0
            or float(getattr(args, "compat_icm_util_lambda", 0.0)) > 0.0
        )
        and compatibility_state.get("audio_token_bank")
    )
    add_reg_enabled = bool(unpaired_aux_enabled and getattr(args, "compat_add_reg_enable", False))
    noharm_enabled = bool(unpaired_aux_enabled and getattr(args, "compat_noharm_enable", False))
    logit_fusion_enabled = bool(unpaired_aux_enabled and getattr(args, "compat_logit_fusion_enable", False))
    transport_enabled = bool(unpaired_aux_enabled and getattr(args, "compat_transport_enable", False))
    gateadd_enabled = bool(unpaired_aux_enabled and getattr(args, "compat_gate_add_enable", False))

    add_layers_hint = str(getattr(args, "compat_add_reg_layers", ""))
    add_layers = _parse_layer_list(add_layers_hint) if (add_reg_enabled or gateadd_enabled) else []
    if (add_reg_enabled or gateadd_enabled) and not add_layers:
        add_layers = list(compat_layers)
    transport_layers_hint = str(getattr(args, "compat_transport_layers", ""))
    transport_layers = _parse_layer_list(transport_layers_hint) if transport_enabled else []
    if transport_enabled and not transport_layers:
        transport_layers = list(add_layers) if add_layers else list(compat_layers)
    add_every = max(1, int(getattr(args, "compat_add_reg_every", 200)))
    add_lambda = float(getattr(args, "compat_add_reg_lambda", 0.0))
    add_norm = bool(getattr(args, "compat_add_reg_normalize", True))
    noharm_lambda = float(getattr(args, "compat_noharm_lambda", 0.0))
    noharm_margin = float(getattr(args, "compat_noharm_margin", 0.0))
    noharm_use_best_single = bool(getattr(args, "compat_noharm_use_best_single", False))
    logit_fusion_lambda = float(getattr(args, "compat_logit_fusion_lambda", 0.0))
    logit_fusion_conf_temp = float(getattr(args, "compat_logit_fusion_conf_temp", 0.5))
    transport_lambda = float(getattr(args, "compat_transport_lambda", 0.0))
    transport_cap = float(getattr(args, "compat_transport_cap", 0.0))
    transport_norm = bool(getattr(args, "compat_transport_normalize", True))
    icm_cancel_enabled = bool(unpaired_aux_enabled and getattr(args, "compat_icm_cancel_enable", False))
    icm_cancel_lambda = float(getattr(args, "compat_icm_cancel_lambda", 0.0))
    icm_cancel_loss_type = str(getattr(args, "compat_icm_cancel_loss_type", "mse"))
    icm_cancel_logit_temp = float(getattr(args, "compat_icm_cancel_logit_temp", 1.0))
    icm_cancel_start_step = int(getattr(args, "compat_icm_cancel_start_step", 0))
    icm_noharm_start_step = int(getattr(args, "compat_icm_noharm_start_step", 0))
    icm_util_lambda = float(getattr(args, "compat_icm_util_lambda", 0.0))
    icm_util_start_step = int(getattr(args, "compat_icm_util_start_step", 0))
    icm_identity_lambda = float(getattr(args, "compat_icm_identity_lambda", 0.0))
    icm_small_lambda = float(getattr(args, "compat_icm_small_lambda", 0.0))
    poe_enabled = bool(unpaired_aux_enabled and getattr(args, "compat_poe_enable", False))
    poe_lambda = float(getattr(args, "compat_poe_lambda", 0.0))
    poe_weight_temp = float(getattr(args, "compat_poe_weight_temp", 0.5))
    poe_loss_type = str(getattr(args, "compat_poe_loss_type", "kl"))
    poe_logit_temp = float(getattr(args, "compat_poe_logit_temp", 1.0))
    routing_enabled = bool(getattr(args, "compat_routing_enable", False))
    routing_min_scale = float(getattr(args, "compat_routing_min_scale", 0.25))
    routing_max_scale = float(getattr(args, "compat_routing_max_scale", 1.0))
    gateadd_lambda = float(getattr(args, "compat_gate_add_lambda", 0.0))
    gateadd_pairing = str(getattr(args, "compat_gate_pairing", "zip"))
    gateadd_target_mode = str(getattr(args, "compat_gate_target_mode", "inverse_rho"))
    gateadd_target_product = float(getattr(args, "compat_gate_product_target", -1.0))
    gateadd_rho_beta = float(getattr(args, "compat_gate_rho_beta", 2.0))
    gateadd_min_effective = float(getattr(args, "compat_gate_min_effective", 0.0))
    gateadd_floor_lambda = float(getattr(args, "compat_gate_floor_lambda", 0.0))
    add_warned = False

    def _amp_context():
        if args.fp16 and torch.cuda.is_available():
            return autocast(enabled=True, dtype=torch.float16)
        if use_bf16_amp and torch.cuda.is_available():
            return autocast(enabled=True, dtype=torch.bfloat16)
        return nullcontext()

    pending_accum_steps = 0
    for step, batch in enumerate(dataloader):
        mm = resolve_modality_batch(batch, args.train_modality)

        # Debug: log first batch modality presence
        if step == 0:
            if mm["audio"] is None:
                print("  [debug] batch 0: audio input disabled by train_modality", flush=True)
            else:
                audio_ok = sum(1 for a in mm["audio"] if a is not None)
                print(f"  [debug] batch 0: {audio_ok}/{len(mm['audio'])} samples have audio", flush=True)
            if mm["images"] is None:
                print("  [debug] batch 0: image input disabled by train_modality", flush=True)
            else:
                image_ok = sum(1 for img in mm["images"] if img is not None)
                print(f"  [debug] batch 0: {image_ok}/{len(mm['images'])} samples have images", flush=True)

        inputs = model.prepare_multimodal_inputs(
            text=build_modality_aware_questions(batch["questions"], args.train_modality, args),
            images=mm["images"],
            audio=mm["audio"],
            answers=batch["answers"],
            device=str(device),
            training_mode=True,
        )

        audio_tokens = inputs.pop("audio_tokens", None)
        audio_mask = inputs.pop("audio_attention_mask", None)
        if step == 0:
            pv = inputs.get("pixel_values")
            if pv is None:
                print("  [debug] batch 0: pixel_values=None (vision not entering model)", flush=True)
            else:
                print(f"  [debug] batch 0: pixel_values shape={tuple(pv.shape)}", flush=True)

        gate_value = args.fusion_gate
        if args.gate_warmup_steps > 0:
            progress = min(1.0, float(global_step + 1) / float(max(1, args.gate_warmup_steps)))
            gate_value = args.fusion_gate * progress

        baseline_pooled: Dict[int, torch.Tensor] = {}
        if compat_reg_active:
            with torch.no_grad():
                with _amp_context():
                    baseline_outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        labels=inputs.get("labels"),
                        pixel_values=inputs.get("pixel_values"),
                        audio_tokens=audio_tokens,
                        audio_attention_mask=audio_mask,
                        gate=0.0,
                        output_hidden_states=True,
                    )
            baseline_hs = baseline_outputs.get("all_hidden_states") if isinstance(baseline_outputs, dict) else None
            baseline_pooled = _extract_pooled_layer_states(
                baseline_hs,
                layer_indices=compat_layers,
                labels=inputs.get("labels"),
                attention_mask=inputs.get("attention_mask"),
            )

        with _amp_context():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=inputs.get("labels"),
                pixel_values=inputs.get("pixel_values"),
                audio_tokens=audio_tokens,
                audio_attention_mask=audio_mask,
                gate=gate_value,
                output_hidden_states=bool(compat_reg_active),
            )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

            # Skip batch if loss has no gradient (e.g., all audio failed to load)
            if loss is None or not loss.requires_grad:
                recovered_loss = _recover_loss_from_logits(outputs, inputs.get("labels"))
                if recovered_loss is not None and recovered_loss.requires_grad:
                    if step == 0:
                        print(
                            "  [warn] first batch model loss detached; using CE(logits, labels) fallback",
                            flush=True,
                        )
                    loss = recovered_loss
                else:
                    if step == 0:
                        logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
                        logits_req = bool(torch.is_tensor(logits) and logits.requires_grad)
                        n_valid = -1
                        lbl = inputs.get("labels")
                        if torch.is_tensor(lbl):
                            n_valid = int((lbl != -100).sum().item())
                        print(
                            f"  [warn] first batch loss has no grad — "
                            f"logits_requires_grad={logits_req} valid_label_tokens={n_valid}",
                            flush=True,
                        )
                    continue

            if compat_reg_active:
                all_hs = outputs.get("all_hidden_states") if isinstance(outputs, dict) else None
                compat_reg, _ = compute_vision_subspace_regularizer(
                    all_hidden_states=all_hs,
                    baseline_pooled=baseline_pooled,
                    labels=inputs.get("labels"),
                    attention_mask=inputs.get("attention_mask"),
                    basis_by_layer=compat_basis,
                    layers=compat_layers,
                    layer_weights=compat_layer_weights,
                )
                if compat_reg is not None:
                    total_compat_reg += float(compat_reg.detach().item())
                    compat_reg_batches += 1
                    loss = loss + float(args.compat_reg_lambda) * compat_reg
                elif step == 0 and not compat_warned_hidden:
                    print(
                        "  [compat] warning: hidden-state regularizer inactive on first batch "
                        "(missing all_hidden_states at requested layers)",
                        flush=True,
                    )
                    compat_warned_hidden = True

            # Optional curriculum scheduling for ICM-related auxiliary terms.
            curr_step = int(global_step + step)
            icm_cancel_lambda_eff = icm_cancel_lambda if curr_step >= icm_cancel_start_step else 0.0
            icm_util_lambda_eff = icm_util_lambda if curr_step >= icm_util_start_step else 0.0
            noharm_lambda_eff = noharm_lambda if curr_step >= icm_noharm_start_step else 0.0

            # Optional unpaired composition objectives on sparse steps
            run_unpaired_aux = (
                unpaired_aux_enabled
                and (
                    (add_reg_enabled and add_lambda > 0.0)
                    or (noharm_enabled and noharm_lambda_eff > 0.0)
                    or (logit_fusion_enabled and logit_fusion_lambda > 0.0)
                    or (transport_enabled and transport_lambda > 0.0)
                    or (icm_cancel_enabled and icm_cancel_lambda_eff > 0.0)
                    or (poe_enabled and poe_lambda > 0.0)
                    or (icm_identity_lambda > 0.0)
                    or (icm_small_lambda > 0.0)
                    or (icm_util_lambda_eff > 0.0)
                    or (gateadd_enabled and gateadd_lambda > 0.0)
                )
                and (step % add_every == 0)
            )
            if run_unpaired_aux:
                add_reg, _, add_rho_proxy, aux_losses, aux_stats = compute_unpaired_additivity_regularizer(
                    model=model,
                    inputs=inputs,
                    compatibility_state=compatibility_state,
                    gate_value=gate_value,
                    add_layers=add_layers,
                    current_question_types=batch.get("question_types"),
                    transport_layers=transport_layers,
                    layer_weights=compat_layer_weights,
                    normalize=add_norm,
                    no_harm_enable=noharm_enabled,
                    no_harm_margin=noharm_margin,
                    no_harm_use_best_single=noharm_use_best_single,
                    logit_fusion_enable=logit_fusion_enabled,
                    logit_fusion_conf_temp=logit_fusion_conf_temp,
                    transport_enable=transport_enabled,
                    transport_cap=transport_cap,
                    transport_normalize=transport_norm,
                    icm_cancel_enable=icm_cancel_enabled,
                    icm_cancel_loss_type=icm_cancel_loss_type,
                    icm_cancel_logit_temp=icm_cancel_logit_temp,
                    poe_enable=poe_enabled,
                    poe_weight_temp=poe_weight_temp,
                    poe_loss_type=poe_loss_type,
                    poe_logit_temp=poe_logit_temp,
                    routing_enable=routing_enabled,
                    routing_min=routing_min_scale,
                    routing_max=routing_max_scale,
                )

                if add_reg_enabled and add_reg is not None and add_lambda > 0.0:
                    total_add_reg += float(add_reg.detach().item())
                    add_reg_batches += 1
                    loss = loss + add_lambda * add_reg

                noharm_loss = aux_losses.get("no_harm")
                if noharm_enabled and noharm_loss is not None and noharm_lambda_eff > 0.0:
                    total_noharm_reg += float(noharm_loss.detach().item())
                    noharm_reg_batches += 1
                    loss = loss + noharm_lambda_eff * noharm_loss

                logit_loss = aux_losses.get("logit_fusion")
                if logit_fusion_enabled and logit_loss is not None and logit_fusion_lambda > 0.0:
                    total_logit_reg += float(logit_loss.detach().item())
                    logit_reg_batches += 1
                    loss = loss + logit_fusion_lambda * logit_loss

                transport_loss = aux_losses.get("transport")
                if transport_enabled and transport_loss is not None and transport_lambda > 0.0:
                    total_transport_reg += float(transport_loss.detach().item())
                    transport_reg_batches += 1
                    loss = loss + transport_lambda * transport_loss

                icm_cancel_loss = aux_losses.get("icm_cancel")
                if icm_cancel_enabled and icm_cancel_loss is not None and icm_cancel_lambda_eff > 0.0:
                    total_icm_cancel_reg += float(icm_cancel_loss.detach().item())
                    icm_cancel_reg_batches += 1
                    loss = loss + icm_cancel_lambda_eff * icm_cancel_loss

                icm_id_loss = aux_losses.get("icm_identity")
                if icm_id_loss is not None and icm_identity_lambda > 0.0:
                    total_icm_identity_reg += float(icm_id_loss.detach().item())
                    icm_identity_reg_batches += 1
                    loss = loss + icm_identity_lambda * icm_id_loss

                icm_small_loss = aux_losses.get("icm_small")
                if icm_small_loss is not None and icm_small_lambda > 0.0:
                    total_icm_small_reg += float(icm_small_loss.detach().item())
                    icm_small_reg_batches += 1
                    loss = loss + icm_small_lambda * icm_small_loss

                icm_util_loss = aux_losses.get("icm_util")
                if icm_util_loss is not None and icm_util_lambda_eff > 0.0:
                    total_icm_util_reg += float(icm_util_loss.detach().item())
                    icm_util_reg_batches += 1
                    loss = loss + icm_util_lambda_eff * icm_util_loss

                if "icm_entropy" in aux_stats:
                    total_icm_entropy += float(aux_stats["icm_entropy"])
                    icm_entropy_batches += 1

                poe_loss = aux_losses.get("poe_consistency")
                if poe_enabled and poe_loss is not None and poe_lambda > 0.0:
                    total_poe_reg += float(poe_loss.detach().item())
                    poe_reg_batches += 1
                    loss = loss + poe_lambda * poe_loss

                gateadd_loss = None
                gateadd_stats: Dict[str, float] = {}
                if gateadd_enabled and gateadd_lambda > 0.0:
                    gateadd_loss, gateadd_stats = compute_gate_additivity_objective(
                        model=model,
                        gate_value=gate_value,
                        rho_proxy_by_layer=add_rho_proxy,
                        fallback_layer_weights=compat_layer_weights,
                        pairing=gateadd_pairing,
                        target_mode=gateadd_target_mode,
                        target_product=gateadd_target_product,
                        rho_beta=gateadd_rho_beta,
                        min_effective=gateadd_min_effective,
                        floor_lambda=gateadd_floor_lambda,
                    )
                    if gateadd_loss is not None:
                        total_gateadd_reg += float(gateadd_loss.detach().item())
                        gateadd_reg_batches += 1
                        loss = loss + gateadd_lambda * gateadd_loss
                        if "gate_avg_product" in gateadd_stats:
                            total_gateadd_prod += float(gateadd_stats["gate_avg_product"])
                            gateadd_prod_batches += 1

                if "route_audio_scale" in aux_stats:
                    total_route_scale += float(aux_stats["route_audio_scale"])
                    route_scale_batches += 1

                if (
                    add_reg is None
                    and noharm_loss is None
                    and logit_loss is None
                    and transport_loss is None
                    and icm_cancel_loss is None
                    and icm_id_loss is None
                    and icm_small_loss is None
                    and icm_util_loss is None
                    and poe_loss is None
                    and gateadd_loss is None
                    and not add_warned
                ):
                    print(
                        "  [compat] warning: unpaired composition objective inactive "
                        "(missing hidden states or audio token bank)",
                        flush=True,
                    )
                    add_warned = True

            # RKCA subspace avoidance
            if rkca_subspace_state is not None:
                _sub_basis = rkca_subspace_state["basis"]
                _sub_lambda = rkca_subspace_state["lambda"]

                if args.train_modality == "audio":
                    _sub_tokens = outputs.get("audio_projector_tokens") if isinstance(outputs, dict) else None
                else:
                    _sub_tokens = outputs.get("vision_projector_tokens") if isinstance(outputs, dict) else None

                if _sub_tokens is not None and _sub_tokens.requires_grad:
                    sub_loss = compute_rkca_subspace_loss(
                        _sub_tokens, _sub_basis.to(_sub_tokens.device),
                    )
                    loss = loss + _sub_lambda * sub_loss
                    total_subspace_reg += float(sub_loss.detach())
                    subspace_reg_batches += 1

            loss = loss / args.gradient_accumulation_steps

        scaler.scale(loss).backward()
        pending_accum_steps += 1

        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Log gradient attribution BEFORE clipping (raw gradient signal)
            if getattr(args, "grad_attribution", False):
                grad_log_every = getattr(args, "grad_log_every", 200)
                if global_step % grad_log_every == 0:
                    log_gradient_attribution(model, global_step, wandb_run)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_for_clip, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            pending_accum_steps = 0

        total_loss += loss.item() * args.gradient_accumulation_steps
        total_batches += 1

        if (step + 1) % log_every == 0 or step == 0:
            avg_loss = total_loss / max(total_batches, 1)
            current_lr = optimizer.param_groups[0]["lr"]
            compat_suffix = ""
            if compat_reg_batches > 0:
                compat_avg = total_compat_reg / float(max(1, compat_reg_batches))
                compat_suffix = f" compat_reg={compat_avg:.4f}"
            if add_reg_batches > 0:
                add_avg = total_add_reg / float(max(1, add_reg_batches))
                compat_suffix += f" add_reg={add_avg:.4f}"
            if noharm_reg_batches > 0:
                noharm_avg = total_noharm_reg / float(max(1, noharm_reg_batches))
                compat_suffix += f" noharm_reg={noharm_avg:.4f}"
            if logit_reg_batches > 0:
                logit_avg = total_logit_reg / float(max(1, logit_reg_batches))
                compat_suffix += f" logit_reg={logit_avg:.4f}"
            if transport_reg_batches > 0:
                transport_avg = total_transport_reg / float(max(1, transport_reg_batches))
                compat_suffix += f" transport_reg={transport_avg:.4f}"
            if icm_cancel_reg_batches > 0:
                icm_cancel_avg = total_icm_cancel_reg / float(max(1, icm_cancel_reg_batches))
                compat_suffix += f" icm_cancel={icm_cancel_avg:.4f}"
            if icm_identity_reg_batches > 0:
                icm_id_avg = total_icm_identity_reg / float(max(1, icm_identity_reg_batches))
                compat_suffix += f" icm_id={icm_id_avg:.4f}"
            if icm_small_reg_batches > 0:
                icm_small_avg = total_icm_small_reg / float(max(1, icm_small_reg_batches))
                compat_suffix += f" icm_small={icm_small_avg:.4f}"
            if icm_util_reg_batches > 0:
                icm_util_avg = total_icm_util_reg / float(max(1, icm_util_reg_batches))
                compat_suffix += f" icm_util={icm_util_avg:.4f}"
            if icm_entropy_batches > 0:
                icm_entropy_avg = total_icm_entropy / float(max(1, icm_entropy_batches))
                compat_suffix += f" icm_H={icm_entropy_avg:.3f}"
            if route_scale_batches > 0:
                route_avg = total_route_scale / float(max(1, route_scale_batches))
                compat_suffix += f" route_scale={route_avg:.3f}"
            if poe_reg_batches > 0:
                poe_avg = total_poe_reg / float(max(1, poe_reg_batches))
                compat_suffix += f" poe_reg={poe_avg:.4f}"
            if gateadd_reg_batches > 0:
                gateadd_avg = total_gateadd_reg / float(max(1, gateadd_reg_batches))
                compat_suffix += f" gateadd_reg={gateadd_avg:.4f}"
            if gateadd_prod_batches > 0:
                gateprod_avg = total_gateadd_prod / float(max(1, gateadd_prod_batches))
                compat_suffix += f" gate_prod={gateprod_avg:.4f}"
            if subspace_reg_batches > 0:
                subspace_avg = total_subspace_reg / float(max(1, subspace_reg_batches))
                compat_suffix += f" subspace_reg={subspace_avg:.4f}"
            print(
                f"  [train] step {step + 1}/{num_batches} "
                f"loss={avg_loss:.4f} lr={current_lr:.2e} gate={gate_value:.3f}{compat_suffix}",
                flush=True,
            )

    # Flush remainder gradients if epoch length is not divisible by grad accumulation.
    if pending_accum_steps > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_for_clip, args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    return total_loss / max(total_batches, 1), global_step


@torch.no_grad()
def evaluate(
    model: SAFEModel,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    modality: str,
    args: argparse.Namespace,
    active_fusion_layers: Optional[Sequence[int]] = None,
    silent: bool = False,
) -> Dict[str, Any]:
    model.eval()
    eval_start = time.time()
    ttc_enabled = bool(getattr(args, "ttc_enable", False)) and modality == "both"

    exact_total = 0.0
    extracted_total = 0.0
    f1_total = 0.0
    categorical_f1_total = 0.0
    count = 0
    ttc_count = 0
    ttc_entropy_before = 0.0
    ttc_entropy_after = 0.0
    ttc_gate_means: List[float] = []
    fixed_point_delta_norms: List[float] = []
    fixed_point_self_norms: List[List[float]] = []
    fixed_point_state_step_norms: List[List[float]] = []
    by_type: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"exact": 0.0, "extracted": 0.0, "f1": 0.0, "categorical_f1": 0.0, "n": 0.0}
    )
    debug_print_budget = max(0, int(getattr(args, "eval_debug_samples", 0)))

    eval_batches = len(dataloader)
    eval_log_every = max(1, eval_batches // 5)  # Log ~5 times per eval
    for eval_step, batch in enumerate(dataloader):
        if ttc_enabled:
            for sample_idx in range(len(batch["questions"])):
                sample_batch = _slice_eval_batch(batch, sample_idx)
                try:
                    with torch.enable_grad():
                        pred, ttc_stats = _optimize_ttc_gate_overrides(
                            model=model,
                            batch=sample_batch,
                            tokenizer=tokenizer,
                            device=device,
                            args=args,
                            active_fusion_layers=active_fusion_layers,
                        )
                except Exception as exc:
                    pred = _generate_eval_prediction(
                        model=model,
                        batch=sample_batch,
                        tokenizer=tokenizer,
                        device=device,
                        modality=modality,
                        args=args,
                        active_fusion_layers=active_fusion_layers,
                    )
                    ttc_stats = {
                        "enabled": False,
                        "objective": str(args.ttc_objective),
                        "error": str(exc),
                    }
                    if not silent:
                        print(f"  [eval:{modality}:ttc-warning] fallback_to_standard_generation error={exc}", flush=True)
                ref = sample_batch["answers"][0]
                qtype = sample_batch["question_types"][0] if "question_types" in sample_batch else "unknown"

                norm_ref = normalize_answer(ref)
                norm_pred = normalize_answer(pred)
                exact = float(norm_pred == norm_ref)
                extracted_pred = extract_answer(pred)
                extracted_ref = extract_answer(ref)
                extracted = float(extracted_pred == extracted_ref)
                f1 = token_f1(pred, ref)
                cat_f1 = categorical_f1(pred, ref)
                exact_total += exact
                extracted_total += extracted
                f1_total += f1
                categorical_f1_total += cat_f1
                count += 1

                by_type[qtype]["exact"] += exact
                by_type[qtype]["extracted"] += extracted
                by_type[qtype]["f1"] += f1
                by_type[qtype]["categorical_f1"] += cat_f1
                by_type[qtype]["n"] += 1.0

                if ttc_stats.get("base_entropy") is not None:
                    ttc_entropy_before += float(ttc_stats["base_entropy"])
                if ttc_stats.get("final_entropy") is not None:
                    ttc_entropy_after += float(ttc_stats["final_entropy"])
                gate_values = ttc_stats.get("gate_values", {})
                if gate_values:
                    ttc_gate_means.append(
                        sum(float(v) for v in gate_values.values()) / float(len(gate_values))
                    )
                ttc_count += 1

                if debug_print_budget > 0 and not silent:
                    print(
                        f"  [eval:{modality}:ttc] q={sample_batch['questions'][0]!r} "
                        f"pred={pred!r} extracted={extracted_pred!r} ref={ref!r} "
                        f"objective={ttc_stats.get('objective')}",
                        flush=True,
                    )
                    debug_print_budget -= 1

            if (eval_step + 1) % eval_log_every == 0 and not silent:
                running_em = 100.0 * exact_total / max(1, count)
                running_ext = 100.0 * extracted_total / max(1, count)
                elapsed = max(1e-6, time.time() - eval_start)
                avg_batch_sec = elapsed / float(eval_step + 1)
                eta_sec = avg_batch_sec * float(eval_batches - (eval_step + 1))
                print(
                    f"  [eval:{modality}] step {eval_step + 1}/{eval_batches} "
                    f"raw_em={running_em:.2f}% extracted_em={running_ext:.2f}% "
                    f"elapsed={elapsed/60.0:.1f}m eta={eta_sec/60.0:.1f}m "
                    f"ttc={ttc_count}",
                    flush=True,
                )
            continue

        mm = resolve_modality_batch(batch, modality)
        inputs = model.prepare_multimodal_inputs(
            text=build_modality_aware_questions(batch["questions"], modality, args),
            images=mm["images"],
            audio=mm["audio"],
            answers=None,
            device=str(device),
            training_mode=False,
        )

        audio_tokens = inputs.pop("audio_tokens", None)
        audio_mask = inputs.pop("audio_attention_mask", None)
        output_ids = model.generate(
            text=None,
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            audio_tokens=audio_tokens,
            audio_attention_mask=audio_mask,
            active_fusion_layers=active_fusion_layers,
            gate=args.fusion_gate,
            max_new_tokens=args.max_answer_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        fusion_adapter = getattr(model, "fusion_adapter", None)
        comp_summary = getattr(fusion_adapter, "last_composition_summary", None) if fusion_adapter is not None else None
        if modality == "both" and isinstance(comp_summary, dict) and comp_summary.get("mode") == "fixed_point":
            final_update_norm = comp_summary.get("final_update_norm")
            if final_update_norm is not None:
                fixed_point_delta_norms.append(float(final_update_norm))
            self_norms = comp_summary.get("self_norms")
            if isinstance(self_norms, list) and self_norms:
                fixed_point_self_norms.append([float(x) for x in self_norms])
            state_delta_norms = comp_summary.get("state_delta_norms")
            if isinstance(state_delta_norms, list) and state_delta_norms:
                fixed_point_state_step_norms.append([float(x) for x in state_delta_norms])

        prompt_mask = inputs.get("attention_mask")
        prompt_width = int(inputs["input_ids"].size(1))
        for i in range(output_ids.size(0)):
            seq = output_ids[i]
            # Some model paths return full sequence (prompt + generation),
            # while others return only newly generated tokens.
            # InternVL vision path can differ from audio-only path here.
            # IMPORTANT: with left-padding, prompt length for slicing full-sequence
            # outputs should use the padded input width, not attention sum.
            if seq.size(0) > prompt_width:
                gen = seq[prompt_width:]
            else:
                # Fallback: treat returned tokens as generated tokens directly.
                gen = seq
                if eval_step == 0 and i == 0:
                    attn_prompt = int(prompt_mask[i].sum().item()) if prompt_mask is not None else prompt_width
                    if not silent:
                        print(
                            f"  [eval:{modality}] decode fallback active "
                            f"(seq_len={int(seq.size(0))} prompt_width={prompt_width} attn_prompt={attn_prompt})",
                            flush=True,
                        )
            pred = tokenizer.decode(gen, skip_special_tokens=True).strip()
            ref = batch["answers"][i]
            qtype = batch["question_types"][i] if "question_types" in batch else "unknown"

            norm_ref = normalize_answer(ref)
            norm_pred = normalize_answer(pred)
            exact = float(norm_pred == norm_ref)
            # Extracted match: map generated text to closest known answer
            extracted_pred = extract_answer(pred)
            extracted_ref = extract_answer(ref)
            extracted = float(extracted_pred == extracted_ref)
            f1 = token_f1(pred, ref)
            cat_f1 = categorical_f1(pred, ref)
            exact_total += exact
            extracted_total += extracted
            f1_total += f1
            categorical_f1_total += cat_f1
            count += 1

            by_type[qtype]["exact"] += exact
            by_type[qtype]["extracted"] += extracted
            by_type[qtype]["f1"] += f1
            by_type[qtype]["categorical_f1"] += cat_f1
            by_type[qtype]["n"] += 1.0

            if debug_print_budget > 0 and not silent:
                print(
                    f"  [eval:{modality}:sample] q={batch['questions'][i]!r} "
                    f"pred={pred!r} extracted={extracted_pred!r} ref={ref!r}",
                    flush=True,
                )
                debug_print_budget -= 1

        if (eval_step + 1) % eval_log_every == 0 and not silent:
            running_em = 100.0 * exact_total / max(1, count)
            running_ext = 100.0 * extracted_total / max(1, count)
            elapsed = max(1e-6, time.time() - eval_start)
            avg_batch_sec = elapsed / float(eval_step + 1)
            eta_sec = avg_batch_sec * float(eval_batches - (eval_step + 1))
            print(
                f"  [eval:{modality}] step {eval_step + 1}/{eval_batches} "
                f"raw_em={running_em:.2f}% extracted_em={running_ext:.2f}% "
                f"elapsed={elapsed/60.0:.1f}m eta={eta_sec/60.0:.1f}m",
                flush=True,
            )

    result = {
        "modality": modality,
        "exact_match": 100.0 * exact_total / max(1, count),
        "extracted_match": 100.0 * extracted_total / max(1, count),
        "token_f1": 100.0 * f1_total / max(1, count),
        "categorical_f1": 100.0 * categorical_f1_total / max(1, count),
        "num_samples": count,
        "by_question_type": {},
    }
    if ttc_enabled:
        result["ttc"] = {
            "enabled": True,
            "objective": str(args.ttc_objective),
            "num_samples": int(ttc_count),
            "mean_base_entropy": ttc_entropy_before / max(1, ttc_count),
            "mean_final_entropy": ttc_entropy_after / max(1, ttc_count),
            "mean_gate_value": (
                sum(ttc_gate_means) / float(len(ttc_gate_means)) if ttc_gate_means else None
            ),
        }
    if fixed_point_delta_norms:
        def _mean_step(values: List[List[float]]) -> List[float]:
            max_len = max(len(v) for v in values)
            out: List[float] = []
            for idx in range(max_len):
                elems = [v[idx] for v in values if idx < len(v)]
                out.append(sum(elems) / float(len(elems)))
            return out

        result["fixed_point"] = {
            "mean_final_update_norm": sum(fixed_point_delta_norms) / float(len(fixed_point_delta_norms)),
            "mean_self_norms_by_step": _mean_step(fixed_point_self_norms) if fixed_point_self_norms else [],
            "mean_state_delta_norms_by_step": _mean_step(fixed_point_state_step_norms) if fixed_point_state_step_norms else [],
        }
    for k, v in by_type.items():
        n = max(1.0, v["n"])
        result["by_question_type"][k] = {
            "exact_match": 100.0 * v["exact"] / n,
            "extracted_match": 100.0 * v["extracted"] / n,
            "token_f1": 100.0 * v["f1"] / n,
            "categorical_f1": 100.0 * v["categorical_f1"] / n,
            "num_samples": int(v["n"]),
        }
    if not silent:
        total_elapsed = time.time() - eval_start
        print(
            f"  [eval:{modality}] complete in {total_elapsed/60.0:.1f}m "
            f"({total_elapsed/max(1, count):.3f}s/sample)",
            flush=True,
        )
    return result


@torch.no_grad()
def run_layer_additivity_probe(
    model: SAFEModel,
    dataset: Dataset,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """
    Automated per-layer additivity probe for composition analysis.

    For each fusion layer l, computes extracted-match-based epsilon:
      eps_l = |y0 - ya - yv + yav| / (|ya-y0| + |yv-y0| + eps)
    where y0=text, ya=audio, yv=image, yav=both.
    """
    if not getattr(args, "layer_additivity_probe", False):
        return None
    if not hasattr(model, "fusion_adapter") or model.fusion_adapter is None:
        return None

    layers = sorted(getattr(model.fusion_adapter, "fusion_layer_indices", []))
    if not layers:
        return None

    probe_start = time.time()
    probe_n = int(getattr(args, "layer_probe_samples", 0) or 0)
    if probe_n > 0 and probe_n < len(dataset):
        probe_dataset: Dataset = Subset(dataset, list(range(probe_n)))
    else:
        probe_dataset = dataset

    probe_loader = DataLoader(
        probe_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_avqa,
        pin_memory=torch.cuda.is_available(),
    )

    text_metrics = evaluate(
        model, probe_loader, tokenizer, device, "text", args, silent=True
    )
    y0 = text_metrics["extracted_match"] / 100.0

    rows: List[Dict[str, Any]] = []
    for layer_idx, layer in enumerate(layers):
        layer_start = time.time()
        active = [int(layer)]
        audio_metrics = evaluate(
            model, probe_loader, tokenizer, device, "audio", args,
            active_fusion_layers=active, silent=True,
        )
        image_metrics = evaluate(
            model, probe_loader, tokenizer, device, "image", args,
            active_fusion_layers=active, silent=True,
        )
        both_metrics = evaluate(
            model, probe_loader, tokenizer, device, "both", args,
            active_fusion_layers=active, silent=True,
        )

        ya = audio_metrics["extracted_match"] / 100.0
        yv = image_metrics["extracted_match"] / 100.0
        yav = both_metrics["extracted_match"] / 100.0

        denom = abs(ya - y0) + abs(yv - y0) + 1e-6
        eps_l = abs(y0 - ya - yv + yav) / denom
        synergy = yav - (ya + yv - y0)
        gain_vs_best_single = yav - max(ya, yv)

        rows.append(
            {
                "layer": int(layer),
                "text_extracted_match": 100.0 * y0,
                "audio_extracted_match": 100.0 * ya,
                "image_extracted_match": 100.0 * yv,
                "both_extracted_match": 100.0 * yav,
                "epsilon_additivity": float(eps_l),
                "synergy": float(synergy),
                "gain_vs_best_single": float(gain_vs_best_single),
            }
        )
        elapsed = time.time() - layer_start
        print(
            f"  [additivity_probe] layer {layer_idx + 1}/{len(layers)}={layer} "
            f"done in {elapsed/60.0:.1f}m",
            flush=True,
        )

    # Lower epsilon is better additivity; use both score as tiebreaker.
    ranking = sorted(
        rows,
        key=lambda r: (r["epsilon_additivity"], -r["both_extracted_match"]),
    )

    print(
        f"  [additivity_probe] samples={len(probe_dataset)} text_extracted={100.0 * y0:.2f}",
        flush=True,
    )
    for r in ranking:
        print(
            "  [additivity_probe] "
            f"layer={r['layer']} eps={r['epsilon_additivity']:.4f} "
            f"both={r['both_extracted_match']:.2f} "
            f"audio={r['audio_extracted_match']:.2f} "
            f"image={r['image_extracted_match']:.2f} "
            f"gain_vs_best_single={100.0 * r['gain_vs_best_single']:+.2f}",
            flush=True,
        )

    total_probe_elapsed = time.time() - probe_start
    print(
        f"  [additivity_probe] complete in {total_probe_elapsed/60.0:.1f}m",
        flush=True,
    )

    return {
        "num_samples": len(probe_dataset),
        "rows": rows,
        "ranking": ranking,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AVQA/MUSIC-AVQA composition training")
    p.add_argument("--dataset", type=str, default="music_avqa", choices=["avqa", "music_avqa"])
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--model-config", type=str, default="phase1",
                   help="Base model config name (e.g. phase1, internvl, qwen)")
    p.add_argument("--llm-model", type=str, default=None)
    p.add_argument("--fusion-layers", type=str, default=None)
    p.add_argument("--audio-fusion-layers", type=str, default=None,
                   help="Optional comma-separated audio fusion layers (overrides config/modalities)")
    p.add_argument("--vision-fusion-layers", type=str, default=None,
                   help="Optional comma-separated vision fusion layers (overrides config/modalities)")
    p.add_argument("--num-audio-tokens", type=int, default=8)

    p.add_argument("--train-modality", type=str, default="both", choices=["audio", "image", "both", "interleaved", "interleaved_vision_first"])
    p.add_argument("--trainable-modalities", type=str, default="all", choices=["all", "audio", "vision"],
                   help="Restrict trainable adapter params to a subset while keeping the selected input modality active")
    p.add_argument("--eval-modalities", type=str, default="both,audio,image")
    p.add_argument("--fusion-gate", type=float, default=0.2)
    p.add_argument("--modality-aware-prompts", action="store_true",
                   help="Prepend modality-specific instructions so audio/image/both train against different prompt contexts")
    p.add_argument("--text-prompt-prefix", type=str, default=DEFAULT_TEXT_PROMPT_PREFIX)
    p.add_argument("--audio-prompt-prefix", type=str, default=DEFAULT_AUDIO_PROMPT_PREFIX)
    p.add_argument("--image-prompt-prefix", type=str, default=DEFAULT_IMAGE_PROMPT_PREFIX)
    p.add_argument("--both-prompt-prefix", type=str, default=DEFAULT_BOTH_PROMPT_PREFIX)
    p.add_argument("--gate-warmup-steps", type=int, default=0,
                   help="Linearly warm fusion gate from 0 to --fusion-gate over N optimizer steps")
    p.add_argument("--bottleneck-dim", type=int, default=None,
                   help="Override fusion bottleneck dimension (default: 256, try 410 for 8B to match 4B's 10%% ratio)")
    p.add_argument("--no-slim-projector", dest="slim_projector", action="store_false",
                   help="Disable slim projector (output at full llm_hidden_size instead of bottleneck_dim)")
    p.set_defaults(slim_projector=True)

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lr-scheduler", type=str, default="cosine", choices=["none", "cosine", "linear"])
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Override warmup ratio with explicit warmup steps")
    p.add_argument("--min-lr-ratio", type=float, default=0.1,
                   help="Final LR = min_lr_ratio * base_lr for cosine/linear schedulers")
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-answer-tokens", type=int, default=8)
    p.add_argument("--label-smoothing", type=float, default=None,
                   help="Override model config label smoothing")

    p.add_argument("--freeze-audio-encoder", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # Learned gating
    p.add_argument("--learned-gate", action="store_true",
                   help="Enable per-layer learned gating (Flamingo-style tanh gates)")
    p.add_argument("--learned-gate-init", type=float, default=0.0,
                   help="Initial value for learned gate params (tanh-squashed, 0.0=gate off)")
    p.add_argument("--train-gates-only", action="store_true",
                   help="Freeze all trainable adapter params except per-layer learned gates")
    p.add_argument("--icm-enable", action="store_true",
                   help="Enable Interaction Correction Mixer (set-style sidecar)")
    p.add_argument("--icm-dim", type=int, default=512,
                   help="Hidden width of interaction mixer")
    p.add_argument("--icm-heads", type=int, default=8,
                   help="Attention heads in interaction mixer")
    p.add_argument("--icm-layers", type=int, default=1,
                   help="Number of set-mixer blocks")
    p.add_argument("--icm-dropout", type=float, default=0.1,
                   help="Dropout used in interaction mixer")
    p.add_argument("--icm-gate-init", type=float, default=-2.0,
                   help="Initial gate bias for interaction mixer (negative keeps it near-off)")
    p.add_argument("--icm-min-modalities", type=int, default=2,
                   help="Minimum active modalities required before applying mixer correction")
    p.add_argument("--icm-util-target", type=float, default=0.7,
                   help="Target normalized entropy floor for modality utilization")
    p.add_argument("--delta-norm-cap-ratio", type=float, default=0.0,
                   help="Cap per-token fusion residual norm: ||delta|| <= ratio * ||hidden|| (0=disabled)")
    p.add_argument("--delta-norm-cap-eps", type=float, default=1e-6,
                   help="Numerical epsilon for delta norm capping")
    p.add_argument("--gate-depth-decay", type=float, default=1.0,
                   help="Global depth decay for per-layer gates (<1 suppresses early layers, 1=off)")
    p.add_argument("--audio-gate-depth-decay", type=float, default=1.0,
                   help="Audio-specific depth decay for per-layer gates (<1 suppresses early audio layers)")
    p.add_argument("--vision-gate-depth-decay", type=float, default=1.0,
                   help="Vision-specific depth decay for per-layer gates (<1 suppresses early vision layers)")

    # Gradient attribution study
    p.add_argument("--grad-attribution", dest="grad_attribution", action="store_true",
                   help="Log per-layer gradient norms for fusion layer selection study")
    p.add_argument("--no-grad-attribution", dest="grad_attribution", action="store_false",
                   help="Disable per-layer gradient attribution logging")
    p.add_argument("--grad-log-every", type=int, default=100,
                   help="Log gradient attribution every N steps")
    p.set_defaults(grad_attribution=True)

    # Composition experiment: load separate modality checkpoints for zero-shot composition eval
    p.add_argument("--compose-audio-ckpt", type=Path, default=None,
                   help="Path to audio-only adapter checkpoint for composition eval")
    p.add_argument("--compose-vision-ckpt", type=Path, default=None,
                   help="Path to vision-only adapter checkpoint for composition eval")
    p.add_argument("--compose-calibration-enable", action="store_true",
                   help="Load composed unimodal checkpoints and continue training on paired data instead of eval-only")
    p.add_argument("--compose-calibration-trainable", type=str, default="fusion",
                   choices=["all", "fusion", "projectors", "projectors_fusion", "gates", "interaction_mixer"],
                   help="Parameter group to calibrate after loading composed unimodal checkpoints")
    p.add_argument("--init-audio-ckpt", type=Path, default=None,
                   help="Optional audio adapter checkpoint to initialize model before training")
    p.add_argument("--init-vision-ckpt", type=Path, default=None,
                   help="Optional vision adapter checkpoint to initialize model before training")

    # Sequential composition objective (Audio -> Vision, no joint AV pairs)
    p.add_argument("--compat-reg-enable", action="store_true",
                   help="Enable vision subspace compatibility regularizer using audio-induced shift bases")
    p.add_argument("--compat-reg-lambda", type=float, default=0.05,
                   help="Weight for compatibility regularizer added during vision phase")
    p.add_argument("--compat-reg-rank", type=int, default=8,
                   help="PCA rank for per-layer audio shift subspaces")
    p.add_argument("--compat-reg-audio-samples", type=int, default=256,
                   help="Number of audio-text samples used to fit shift subspaces per refresh")
    p.add_argument("--compat-reg-min-samples", type=int, default=64,
                   help="Minimum samples required to fit a layer subspace")
    p.add_argument("--compat-reg-refresh-every", type=int, default=1,
                   help="Recompute audio shift subspaces every N epochs")
    p.add_argument("--compat-reg-layers", type=str, default="",
                   help="Optional comma-separated layer indices for compatibility regularizer "
                        "(default: vision fusion layers)")
    p.add_argument("--compat-reg-weight-by-shift-norm", action="store_true",
                   help="Use inverse shift-energy layer weighting for compat/additivity losses")
    p.add_argument("--no-compat-reg-weight-by-shift-norm", dest="compat_reg_weight_by_shift_norm",
                   action="store_false",
                   help="Disable inverse shift-energy weighting")
    p.add_argument("--compat-reg-reverse", action="store_true",
                   help="Reverse subspace avoidance: collect vision shifts, penalize audio.")

    # Additivity-positive objective (unpaired AV; no joint labels required)
    p.add_argument("--compat-add-reg-enable", action="store_true",
                   help="Enable unpaired additivity regularizer ||Δ_av - Δ_a - Δ_v||^2 on sparse steps")
    p.add_argument("--compat-add-reg-lambda", type=float, default=0.01,
                   help="Weight for unpaired additivity regularizer")
    p.add_argument("--compat-add-reg-every", type=int, default=200,
                   help="Compute additivity regularizer every N vision steps")
    p.add_argument("--compat-add-reg-layers", type=str, default="",
                   help="Optional comma-separated layers for additivity regularizer "
                        "(default: compatibility layers)")
    p.add_argument("--compat-add-reg-normalize", action="store_true",
                   help="Normalize additivity residual loss by shift energy")
    p.add_argument("--no-compat-add-reg-normalize", dest="compat_add_reg_normalize",
                   action="store_false",
                   help="Disable additivity loss normalization")
    p.add_argument("--compat-add-bank-size", type=int, default=64,
                   help="Max number of audio token samples to cache for unpaired additivity regularizer")
    p.add_argument("--compat-transport-enable", action="store_true",
                   help="Enable cross-layer transport penalty ||h_both - h_vision||^2 at vision layers")
    p.add_argument("--compat-transport-lambda", type=float, default=0.02,
                   help="Weight for transport penalty")
    p.add_argument("--compat-transport-cap", type=float, default=0.0,
                   help="Optional hinge cap on normalized transport energy (0=quadratic without cap)")
    p.add_argument("--compat-transport-normalize", action="store_true",
                   help="Normalize transport penalty by vision hidden energy")
    p.add_argument("--no-compat-transport-normalize", dest="compat_transport_normalize",
                   action="store_false",
                   help="Disable transport loss normalization")
    p.add_argument("--compat-transport-layers", type=str, default="",
                   help="Optional comma-separated layers for transport penalty (default: compatibility layers)")
    p.add_argument("--compat-icm-cancel-enable", action="store_true",
                   help="Enable ICM cancel loss: match both logits to additive ideal (za + zv - z0)")
    p.add_argument("--compat-icm-cancel-lambda", type=float, default=0.02,
                   help="Weight for ICM cancel loss")
    p.add_argument("--compat-icm-cancel-loss-type", type=str, default="mse", choices=["mse", "kl"],
                   help="Loss type for ICM cancel objective")
    p.add_argument("--compat-icm-cancel-logit-temp", type=float, default=1.0,
                   help="Logit temperature for KL-based ICM cancel loss")
    p.add_argument("--compat-icm-cancel-start-step", type=int, default=0,
                   help="Start step for ICM cancel loss (curriculum)")
    p.add_argument("--compat-icm-noharm-start-step", type=int, default=0,
                   help="Start step for no-harm loss when used with ICM")
    p.add_argument("--compat-icm-util-lambda", type=float, default=0.0,
                   help="Weight for ICM utilization entropy floor loss")
    p.add_argument("--compat-icm-util-start-step", type=int, default=0,
                   help="Start step for ICM utilization loss (curriculum)")
    p.add_argument("--compat-icm-identity-lambda", type=float, default=0.01,
                   help="Weight for ICM identity loss (single-modality bypass)")
    p.add_argument("--compat-icm-small-lambda", type=float, default=0.001,
                   help="Weight for ICM correction magnitude loss")

    # Do-no-harm objective: prevent composed loss from exceeding best/single modality loss
    p.add_argument("--compat-noharm-enable", action="store_true",
                   help="Enable unpaired do-no-harm penalty: ReLU(NLL_both - NLL_ref + margin)")
    p.add_argument("--compat-noharm-lambda", type=float, default=0.02,
                   help="Weight for do-no-harm penalty")
    p.add_argument("--compat-noharm-margin", type=float, default=0.0,
                   help="Slack margin for do-no-harm penalty")
    p.add_argument("--compat-noharm-use-best-single", action="store_true",
                   help="Use min(NLL_audio, NLL_vision) as no-harm reference (else vision-only)")

    # Confidence-calibrated score fusion objective (NLL-space)
    p.add_argument("--compat-logit-fusion-enable", action="store_true",
                   help="Enable confidence-calibrated NLL fusion consistency loss")
    p.add_argument("--compat-logit-fusion-lambda", type=float, default=0.02,
                   help="Weight for NLL fusion consistency penalty")
    p.add_argument("--compat-logit-fusion-conf-temp", type=float, default=0.5,
                   help="Temperature for confidence -> fusion weights softmax")

    # Residual-PoE consistency objective (logit space)
    p.add_argument("--compat-poe-enable", action="store_true",
                   help="Enable residual-PoE logit consistency: z_av ~ z0 + a*(za-z0) + b*(zv-z0)")
    p.add_argument("--compat-poe-lambda", type=float, default=0.02,
                   help="Weight for residual-PoE consistency loss")
    p.add_argument("--compat-poe-weight-temp", type=float, default=0.5,
                   help="Temperature for confidence->(alpha,beta) PoE weights")
    p.add_argument("--compat-poe-loss-type", type=str, default="kl", choices=["kl", "mse"],
                   help="Residual-PoE loss form")
    p.add_argument("--compat-poe-logit-temp", type=float, default=1.0,
                   help="Logit temperature for KL-based residual-PoE loss")

    # Confidence-routed gate for unpaired auxiliary composition passes
    p.add_argument("--compat-routing-enable", action="store_true",
                   help="Scale audio gate in unpaired composition passes using audio confidence")
    p.add_argument("--compat-routing-min-scale", type=float, default=0.25,
                   help="Minimum routing gate scale")
    p.add_argument("--compat-routing-max-scale", type=float, default=1.0,
                   help="Maximum routing gate scale")

    # Additivity-first gate-product objective (trainable learned gates).
    p.add_argument("--compat-gate-add-enable", action="store_true",
                   help="Enable gate-product additivity objective weighted by measured non-additivity")
    p.add_argument("--compat-gate-add-lambda", type=float, default=0.02,
                   help="Weight for gate-product additivity objective")
    p.add_argument("--compat-gate-pairing", type=str, default="zip", choices=["zip", "shared", "all"],
                   help="How to pair audio and vision layers for gate-product objective")
    p.add_argument("--compat-gate-target-mode", type=str, default="inverse_rho", choices=["fixed", "inverse_rho"],
                   help="Gate-product target schedule mode")
    p.add_argument("--compat-gate-product-target", type=float, default=-1.0,
                   help="Target effective gate product; <0 uses gate^2 as automatic base target")
    p.add_argument("--compat-gate-rho-beta", type=float, default=2.0,
                   help="Inverse-rho target sharpness: target = base/(1+beta*rho)")
    p.add_argument("--compat-gate-min-effective", type=float, default=0.0,
                   help="Optional minimum effective gate magnitude for floor penalty")
    p.add_argument("--compat-gate-floor-lambda", type=float, default=0.0,
                   help="Weight for gate floor penalty")

    p.set_defaults(
        compat_reg_weight_by_shift_norm=True,
        compat_add_reg_normalize=True,
        compat_transport_normalize=True,
    )

    # RKCA subspace avoidance
    p.add_argument("--rkca-subspace-enable", action="store_true",
                   help="Activate subspace avoidance between audio/vision projectors")
    p.add_argument("--rkca-subspace-lambda", type=float, default=0.05,
                   help="Weight for subspace avoidance loss")
    p.add_argument("--rkca-subspace-bank-size", type=int, default=128,
                   help="Number of token sets to bank per phase")
    p.add_argument("--rkca-subspace-rank", type=int, default=8,
                   help="PCA rank for subspace basis")
    p.add_argument("--rkca-subspace-refit-every", type=int, default=1,
                   help="Refit basis every N epochs (default: every epoch)")

    p.add_argument("--max-samples", type=int, default=0,
                   help="Limit train/val to N samples for quick sanity runs (0=unlimited)")
    p.add_argument("--train-max-samples", type=int, default=0,
                   help="Limit only the training set to N samples (0=unlimited)")
    p.add_argument("--val-max-samples", type=int, default=0,
                   help="Limit only the validation set to N samples (0=unlimited)")
    p.add_argument("--eval-debug-samples", type=int, default=0,
                   help="Print first N eval predictions per run for decode/debug checks")
    p.add_argument("--layer-additivity-probe", action="store_true",
                   help="Run per-layer additivity probe (epsilon_l) during eval")
    p.add_argument("--layer-probe-samples", type=int, default=256,
                   help="Max validation samples for layer additivity probe (0=full val)")
    p.add_argument("--layer-probe-every", type=int, default=1,
                   help="Run layer additivity probe every N epochs")
    p.add_argument("--ttc-enable", action="store_true",
                   help="Enable test-time composition by optimizing runtime gate overrides on 'both' eval")
    p.add_argument("--ttc-objective", type=str, default="simple", choices=["simple", "entropy_noharm"],
                   help="TTC objective: simple entropy minimization or entropy + no-harm vs best single modality")
    p.add_argument("--ttc-search-mode", type=str, default="gradient", choices=["gradient", "candidate", "hybrid"],
                   help="Stage-B TTC search policy: gradient descent, black-box candidate search, or candidate+gradient refinement")
    p.add_argument("--ttc-candidate-grid", type=str, default="0.75,1.0,1.25",
                   help="Comma-separated gate scales used for candidate/hybrid TTC search")
    p.add_argument("--ttc-steps", type=int, default=5,
                   help="Number of per-sample TTC optimization steps")
    p.add_argument("--ttc-lr", type=float, default=5e-2,
                   help="Learning rate for TTC runtime gate optimization")
    p.add_argument("--ttc-init-gate", type=float, default=1.0,
                   help="Initial runtime TTC gate multiplier")
    p.add_argument("--ttc-gate-min", type=float, default=0.0,
                   help="Clamp minimum for TTC runtime gate multipliers")
    p.add_argument("--ttc-gate-max", type=float, default=2.5,
                   help="Clamp maximum for TTC runtime gate multipliers")
    p.add_argument("--ttc-gate-reg-lambda", type=float, default=0.02,
                   help="Quadratic penalty weight for deviating TTC gates from 1.0")
    p.add_argument("--ttc-noharm-lambda", type=float, default=0.1,
                   help="Weight on TTC no-harm penalty for the complex objective")
    p.add_argument("--ttc-noharm-margin", type=float, default=0.0,
                   help="Margin used in TTC no-harm hinge")
    p.add_argument("--ttc-stability-enable", action="store_true",
                   help="Enable a hard TTC stability filter based on prediction KL under small gate perturbations")
    p.add_argument("--ttc-stability-threshold", type=float, default=0.15,
                   help="Maximum allowed TTC instability KL before rejecting a candidate")
    p.add_argument("--ttc-stability-perturb", type=float, default=0.1,
                   help="Relative gate perturbation used for the TTC stability check")
    p.add_argument("--ttc-interaction-enable", action="store_true",
                   help="Enable stage-C TTC: freeze gate solution and optimize a joint-only interaction module per shared layer")
    p.add_argument("--ttc-interaction-module", type=str, default="diag", choices=["scalar", "diag", "lowrank"],
                   help="Stage-C interaction module type: scalar or diagonal joint-only sidecar")
    p.add_argument("--ttc-interaction-steps", type=int, default=5,
                   help="Number of stage-C interaction optimization steps")
    p.add_argument("--ttc-interaction-lr", type=float, default=5e-2,
                   help="Learning rate for stage-C interaction parameters")
    p.add_argument("--ttc-interaction-init", type=float, default=0.0,
                   help="Initial value for stage-C interaction scalars")
    p.add_argument("--ttc-interaction-min", type=float, default=-0.5,
                   help="Minimum clamp for stage-C interaction scalar scales")
    p.add_argument("--ttc-interaction-max", type=float, default=0.5,
                   help="Maximum clamp for stage-C interaction scalar scales")
    p.add_argument("--ttc-interaction-diag-init", type=float, default=0.0,
                   help="Initial value for diagonal stage-C interaction parameters")
    p.add_argument("--ttc-interaction-diag-min", type=float, default=-0.5,
                   help="Minimum clamp for diagonal stage-C interaction parameters")
    p.add_argument("--ttc-interaction-diag-max", type=float, default=0.5,
                   help="Maximum clamp for diagonal stage-C interaction parameters")
    p.add_argument("--ttc-interaction-rank", type=int, default=8,
                   help="Rank for low-rank stage-C interaction module")
    p.add_argument("--ttc-interaction-matrix-init", type=float, default=0.01,
                   help="Initialization std for low-rank interaction matrices")
    p.add_argument("--ttc-interaction-matrix-min", type=float, default=-0.25,
                   help="Minimum clamp for low-rank interaction matrices")
    p.add_argument("--ttc-interaction-matrix-max", type=float, default=0.25,
                   help="Maximum clamp for low-rank interaction matrices")
    p.add_argument("--ttc-interaction-reg-lambda", type=float, default=0.05,
                   help="Quadratic penalty on stage-C interaction parameters")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="SAFE-AVQA-Composition")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] dataset={args.dataset} device={device}")

    train_ds = ManifestAVQADataset(args.train_manifest, args.media_root)
    val_ds = ManifestAVQADataset(args.val_manifest, args.media_root)
    if args.max_samples > 0:
        train_ds.rows = train_ds.rows[:args.max_samples]
        val_ds.rows = val_ds.rows[:args.max_samples]
        print(f"[info] --max-samples={args.max_samples}: truncated datasets")
    if args.train_max_samples > 0:
        train_ds.rows = train_ds.rows[:args.train_max_samples]
        print(f"[info] --train-max-samples={args.train_max_samples}: truncated train set", flush=True)
    if args.val_max_samples > 0:
        val_ds.rows = val_ds.rows[:args.val_max_samples]
        print(f"[info] --val-max-samples={args.val_max_samples}: truncated val set", flush=True)
    print(f"[info] train_samples={len(train_ds)} val_samples={len(val_ds)}")
    print(f"[info] train_media_stats={train_ds.media_stats}")
    print(f"[info] val_media_stats={val_ds.media_stats}")

    wandb_run = None
    if args.wandb:
        if wandb is None:
            print("[warn] --wandb requested but wandb is not installed. Continuing without wandb.")
        else:
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                tags=tags if tags else None,
                config={
                    "dataset": args.dataset,
                    "architecture": "pre_ffn",
                    "train_modality": args.train_modality,
                    "trainable_modalities": args.trainable_modalities,
                    "eval_modalities": args.eval_modalities,
                    "fusion_layers": args.fusion_layers,
                    "audio_fusion_layers": args.audio_fusion_layers,
                    "vision_fusion_layers": args.vision_fusion_layers,
                    "num_audio_tokens": args.num_audio_tokens,
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "lr_scheduler": args.lr_scheduler,
                    "warmup_ratio": args.warmup_ratio,
                    "warmup_steps": args.warmup_steps,
                    "min_lr_ratio": args.min_lr_ratio,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "seed": args.seed,
                    "train_manifest": str(args.train_manifest),
                    "val_manifest": str(args.val_manifest),
                    "fusion_gate": args.fusion_gate,
                    "gate_warmup_steps": args.gate_warmup_steps,
                    "learned_gate": args.learned_gate,
                    "learned_gate_init": args.learned_gate_init,
                    "train_gates_only": args.train_gates_only,
                    "icm_enable": args.icm_enable,
                    "icm_dim": args.icm_dim,
                    "icm_heads": args.icm_heads,
                    "icm_layers": args.icm_layers,
                    "icm_dropout": args.icm_dropout,
                    "icm_gate_init": args.icm_gate_init,
                    "icm_min_modalities": args.icm_min_modalities,
                    "icm_util_target": args.icm_util_target,
                    "delta_norm_cap_ratio": args.delta_norm_cap_ratio,
                    "gate_depth_decay": args.gate_depth_decay,
                    "audio_gate_depth_decay": args.audio_gate_depth_decay,
                    "vision_gate_depth_decay": args.vision_gate_depth_decay,
                    "label_smoothing": args.label_smoothing,
                    "layer_additivity_probe": args.layer_additivity_probe,
                    "layer_probe_samples": args.layer_probe_samples,
                    "layer_probe_every": args.layer_probe_every,
                    "ttc_enable": args.ttc_enable,
                    "ttc_objective": args.ttc_objective,
                    "ttc_search_mode": args.ttc_search_mode,
                    "ttc_candidate_grid": args.ttc_candidate_grid,
                    "ttc_steps": args.ttc_steps,
                    "ttc_lr": args.ttc_lr,
                    "ttc_init_gate": args.ttc_init_gate,
                    "ttc_gate_min": args.ttc_gate_min,
                    "ttc_gate_max": args.ttc_gate_max,
                    "ttc_gate_reg_lambda": args.ttc_gate_reg_lambda,
                    "ttc_noharm_lambda": args.ttc_noharm_lambda,
                    "ttc_noharm_margin": args.ttc_noharm_margin,
                    "ttc_stability_enable": args.ttc_stability_enable,
                    "ttc_stability_threshold": args.ttc_stability_threshold,
                    "ttc_stability_perturb": args.ttc_stability_perturb,
                    "ttc_interaction_enable": args.ttc_interaction_enable,
                    "ttc_interaction_module": args.ttc_interaction_module,
                    "ttc_interaction_steps": args.ttc_interaction_steps,
                    "ttc_interaction_lr": args.ttc_interaction_lr,
                    "ttc_interaction_init": args.ttc_interaction_init,
                    "ttc_interaction_min": args.ttc_interaction_min,
                    "ttc_interaction_max": args.ttc_interaction_max,
                    "ttc_interaction_diag_init": args.ttc_interaction_diag_init,
                    "ttc_interaction_diag_min": args.ttc_interaction_diag_min,
                    "ttc_interaction_diag_max": args.ttc_interaction_diag_max,
                    "ttc_interaction_rank": args.ttc_interaction_rank,
                    "ttc_interaction_matrix_init": args.ttc_interaction_matrix_init,
                    "ttc_interaction_matrix_min": args.ttc_interaction_matrix_min,
                    "ttc_interaction_matrix_max": args.ttc_interaction_matrix_max,
                    "ttc_interaction_reg_lambda": args.ttc_interaction_reg_lambda,
                    "compat_reg_enable": args.compat_reg_enable,
                    "compat_reg_lambda": args.compat_reg_lambda,
                    "compat_reg_rank": args.compat_reg_rank,
                    "compat_reg_audio_samples": args.compat_reg_audio_samples,
                    "compat_reg_min_samples": args.compat_reg_min_samples,
                    "compat_reg_refresh_every": args.compat_reg_refresh_every,
                    "compat_reg_layers": args.compat_reg_layers,
                    "compat_reg_weight_by_shift_norm": args.compat_reg_weight_by_shift_norm,
                    "compat_add_reg_enable": args.compat_add_reg_enable,
                    "compat_add_reg_lambda": args.compat_add_reg_lambda,
                    "compat_add_reg_every": args.compat_add_reg_every,
                    "compat_add_reg_layers": args.compat_add_reg_layers,
                    "compat_add_reg_normalize": args.compat_add_reg_normalize,
                    "compat_add_bank_size": args.compat_add_bank_size,
                    "compat_transport_enable": args.compat_transport_enable,
                    "compat_transport_lambda": args.compat_transport_lambda,
                    "compat_transport_cap": args.compat_transport_cap,
                    "compat_transport_normalize": args.compat_transport_normalize,
                    "compat_transport_layers": args.compat_transport_layers,
                    "compat_icm_cancel_enable": args.compat_icm_cancel_enable,
                    "compat_icm_cancel_lambda": args.compat_icm_cancel_lambda,
                    "compat_icm_cancel_loss_type": args.compat_icm_cancel_loss_type,
                    "compat_icm_cancel_logit_temp": args.compat_icm_cancel_logit_temp,
                    "compat_icm_cancel_start_step": args.compat_icm_cancel_start_step,
                    "compat_icm_noharm_start_step": args.compat_icm_noharm_start_step,
                    "compat_icm_util_lambda": args.compat_icm_util_lambda,
                    "compat_icm_util_start_step": args.compat_icm_util_start_step,
                    "compat_icm_identity_lambda": args.compat_icm_identity_lambda,
                    "compat_icm_small_lambda": args.compat_icm_small_lambda,
                    "compat_noharm_enable": args.compat_noharm_enable,
                    "compat_noharm_lambda": args.compat_noharm_lambda,
                    "compat_noharm_margin": args.compat_noharm_margin,
                    "compat_noharm_use_best_single": args.compat_noharm_use_best_single,
                    "compat_logit_fusion_enable": args.compat_logit_fusion_enable,
                    "compat_logit_fusion_lambda": args.compat_logit_fusion_lambda,
                    "compat_logit_fusion_conf_temp": args.compat_logit_fusion_conf_temp,
                    "compat_poe_enable": args.compat_poe_enable,
                    "compat_poe_lambda": args.compat_poe_lambda,
                    "compat_poe_weight_temp": args.compat_poe_weight_temp,
                    "compat_poe_loss_type": args.compat_poe_loss_type,
                    "compat_poe_logit_temp": args.compat_poe_logit_temp,
                    "compat_routing_enable": args.compat_routing_enable,
                    "compat_routing_min_scale": args.compat_routing_min_scale,
                    "compat_routing_max_scale": args.compat_routing_max_scale,
                    "compat_gate_add_enable": args.compat_gate_add_enable,
                    "compat_gate_add_lambda": args.compat_gate_add_lambda,
                    "compat_gate_pairing": args.compat_gate_pairing,
                    "compat_gate_target_mode": args.compat_gate_target_mode,
                    "compat_gate_product_target": args.compat_gate_product_target,
                    "compat_gate_rho_beta": args.compat_gate_rho_beta,
                    "compat_gate_min_effective": args.compat_gate_min_effective,
                    "compat_gate_floor_lambda": args.compat_gate_floor_lambda,
                    "init_audio_ckpt": str(args.init_audio_ckpt) if args.init_audio_ckpt else None,
                    "init_vision_ckpt": str(args.init_vision_ckpt) if args.init_vision_ckpt else None,
                    "compose_audio_ckpt": str(args.compose_audio_ckpt) if args.compose_audio_ckpt else None,
                    "compose_vision_ckpt": str(args.compose_vision_ckpt) if args.compose_vision_ckpt else None,
                    "compose_calibration_enable": args.compose_calibration_enable,
                    "compose_calibration_trainable": args.compose_calibration_trainable,
                    "train_max_samples": args.train_max_samples,
                    "val_max_samples": args.val_max_samples,
                },
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_avqa,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_avqa,
        pin_memory=torch.cuda.is_available(),
    )

    model_cfg = build_model_config(args)
    model = SAFEModel(**model_cfg)
    model.enable_audio_training()
    llm_gc_enabled = bool(
        getattr(model.base_vl.llm, "is_gradient_checkpointing", False)
        or getattr(model.base_vl.llm, "gradient_checkpointing", False)
    )
    print(f"[info] llm_gradient_checkpointing={llm_gc_enabled}", flush=True)
    if llm_gc_enabled and bool(getattr(model, "enable_midlayer_fusion", False)):
        raise RuntimeError(
            "Gradient checkpointing is enabled while using hook-based multilayer fusion. "
            "This configuration is known to break adapter gradients. "
            "Set SAFE_GRAD_CKPT=0 or pass gradient_checkpointing=False in the model config."
        )
    model.to_device(device)
    if hasattr(model, "get_runtime_device"):
        device = model.get_runtime_device()
        print(f"[info] runtime_device={device}", flush=True)
    if hasattr(model, "fusion_adapter") and model.fusion_adapter is not None:
        layers = getattr(model.fusion_adapter, "fusion_layer_indices", None)
        if layers is not None:
            print(f"[info] effective_fusion_layers={list(layers)}", flush=True)
    if args.delta_norm_cap_ratio > 0.0:
        print(
            f"[transport] delta_norm_cap enabled ratio={args.delta_norm_cap_ratio} "
            f"eps={args.delta_norm_cap_eps}",
            flush=True,
        )
    if (
        abs(args.gate_depth_decay - 1.0) > 1e-8
        or abs(args.audio_gate_depth_decay - 1.0) > 1e-8
        or abs(args.vision_gate_depth_decay - 1.0) > 1e-8
    ):
        print(
            f"[transport] depth-decay gates global={args.gate_depth_decay} "
            f"audio={args.audio_gate_depth_decay} vision={args.vision_gate_depth_decay}",
            flush=True,
        )
    if args.icm_enable:
        print(
            f"[icm] enabled dim={args.icm_dim} heads={args.icm_heads} "
            f"layers={args.icm_layers} gate_init={args.icm_gate_init} "
            f"min_modalities={args.icm_min_modalities} util_target={args.icm_util_target}",
            flush=True,
        )
    tokenizer = model.base_vl.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optional initialization from modality checkpoints before training.
    # This does NOT force eval-only mode.
    if args.init_audio_ckpt:
        model.load_modality_adapters(str(args.init_audio_ckpt), "audio")
        print(f"[init] loaded audio adapters from {args.init_audio_ckpt}", flush=True)
    if args.init_vision_ckpt:
        model.load_modality_adapters(str(args.init_vision_ckpt), "vision")
        print(f"[init] loaded vision adapters from {args.init_vision_ckpt}", flush=True)

    # Composition experiment: load separate modality checkpoints for zero-shot eval
    composition_eval_only = False
    if args.compose_audio_ckpt and args.compose_vision_ckpt:
        model.load_modality_adapters(str(args.compose_audio_ckpt), "audio")
        model.load_modality_adapters(str(args.compose_vision_ckpt), "vision")
        composition_eval_only = not bool(getattr(args, "compose_calibration_enable", False))
        if composition_eval_only:
            print("[composition] Loaded both modality checkpoints — running eval-only", flush=True)
        else:
            print(
                "[composition] Loaded both modality checkpoints — running paired calibration "
                f"with trainable={args.compose_calibration_trainable}",
                flush=True,
            )
            calib_counts = configure_composition_calibration_trainables(
                model,
                mode=args.compose_calibration_trainable,
            )
            calib_total = sum(calib_counts.values())
            print(
                f"[composition-calibration] trainable_total={calib_total:,} "
                + " ".join(f"{k}={v:,}" for k, v in calib_counts.items() if v > 0),
                flush=True,
            )

    if args.train_gates_only:
        gate_param_count = 0
        # Freeze all currently trainable params.
        for _, param in model.named_parameters():
            param.requires_grad = False
        # Re-enable only learned per-layer gates.
        fusion_adapter = getattr(model, "fusion_adapter", None)
        layer_gates = getattr(fusion_adapter, "layer_gates", None) if fusion_adapter is not None else None
        if layer_gates is not None:
            for _, gate_param in layer_gates.items():
                gate_param.requires_grad = True
                gate_param_count += int(gate_param.numel())
        if gate_param_count <= 0:
            raise RuntimeError(
                "--train-gates-only requested, but no learned gates found. "
                "Enable --learned-gate (or config fusion_config.use_learned_gate=True)."
            )
        print(f"[calib] train_gates_only=True gate_params={gate_param_count}", flush=True)

    if args.trainable_modalities != "all":
        modality_counts = configure_trainable_modalities(model, args.trainable_modalities)
        print(
            "[trainable-modalities] "
            f"mode={args.trainable_modalities} "
            f"audio={modality_counts['audio']:,} "
            f"vision={modality_counts['vision']:,} "
            f"other={modality_counts['other']:,}",
            flush=True,
        )

    trainable_params = list(model.get_trainable_parameters())
    print(f"[info] trainable_parameters={sum(p.numel() for p in trainable_params if p.requires_grad):,}")
    optimizer = build_optimizer(model, args)

    # Disable fp16 GradScaler for bf16 models (InternVL, Qwen) — GradScaler is incompatible with bf16
    use_fp16 = args.fp16
    use_bf16_amp = False
    if use_fp16:
        base_dtype = next(model.base_vl.llm.parameters()).dtype
        if base_dtype == torch.bfloat16:
            print("[info] Model uses bfloat16 — disabling fp16 GradScaler (incompatible)", flush=True)
            use_fp16 = False
            use_bf16_amp = True
    else:
        if next(model.base_vl.llm.parameters()).dtype == torch.bfloat16:
            use_bf16_amp = True
    args.fp16 = use_fp16  # Update so train_epoch sees corrected value
    scaler = GradScaler(enabled=use_fp16)
    if use_bf16_amp:
        print("[info] bfloat16 autocast enabled", flush=True)

    best_score = -1.0
    history: List[Dict[str, Any]] = []
    eval_modalities = [m.strip() for m in args.eval_modalities.split(",") if m.strip()]
    if args.ttc_enable and "both" in eval_modalities:
        eval_modalities = ["both"] + [m for m in eval_modalities if m != "both"]
    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
    # Interleaved mode does 2 passes per epoch (audio + vision), so double the step count
    passes_per_epoch = 2 if args.train_modality in ("interleaved", "interleaved_vision_first") else 1
    total_update_steps = updates_per_epoch * passes_per_epoch * max(0, args.num_epochs)
    scheduler, warmup_steps = build_lr_scheduler(optimizer, total_update_steps, args)
    if scheduler is not None:
        print(
            f"[info] lr_scheduler={args.lr_scheduler} total_update_steps={total_update_steps} "
            f"warmup_steps={warmup_steps} min_lr_ratio={args.min_lr_ratio}",
            flush=True,
        )

    # Eval-only modes:
    # 1. Composition eval: both modality checkpoints loaded, skip training
    # 2. Image-only baseline: no trainable vision projector (frozen LLaVA), eval only
    eval_only = composition_eval_only or (
        args.train_modality == "image" and model.vision_projector is None
    )
    if eval_only:
        reason = "composition" if composition_eval_only else "image-only baseline (no trainable vision projector)"
        print(f"\n[eval-only] {reason} — running eval-only")
        model.eval()
        epoch_result: Dict[str, Any] = {"epoch": 0, "train_loss": 0.0, "eval": {}}
        for modality in eval_modalities:
            metrics = evaluate(model, val_loader, tokenizer, device, modality, args)
            epoch_result["eval"][modality] = metrics
            print(f"  [eval:{modality}] raw_em={metrics['exact_match']:.2f} extracted_em={metrics['extracted_match']:.2f} f1={metrics['token_f1']:.2f} n={metrics['num_samples']}")
        if args.layer_additivity_probe:
            probe = run_layer_additivity_probe(model, val_ds, tokenizer, device, args)
            if probe is not None:
                epoch_result["layer_additivity_probe"] = probe
        history.append(epoch_result)
        best_score = epoch_result["eval"].get(args.train_modality, epoch_result["eval"].get("image", {})).get("extracted_match", -1.0)
        if wandb_run is not None:
            log_payload: Dict[str, Any] = {"epoch": 0}
            for modality, metrics in epoch_result["eval"].items():
                log_payload[f"val/{modality}/exact_match"] = metrics["exact_match"]
                log_payload[f"val/{modality}/extracted_match"] = metrics["extracted_match"]
                log_payload[f"val/{modality}/token_f1"] = metrics["token_f1"]
            probe = epoch_result.get("layer_additivity_probe")
            if probe is not None and probe.get("ranking"):
                top = probe["ranking"][0]
                log_payload["probe/best_layer"] = top["layer"]
                log_payload["probe/best_layer_epsilon"] = top["epsilon_additivity"]
                log_payload["probe/best_layer_both_extracted"] = top["both_extracted_match"]
            wandb_run.log(log_payload, step=0)
    elif args.train_modality in ("interleaved", "interleaved_vision_first"):
        # ── Interleaved composition training ──
        # Each epoch: train modality-A adapters → train modality-B adapters → evaluate all 4:
        #   text (baseline), audio+text, vision+text, audio+vision+text (composition)
        # Default order: audio→vision. Vision-first: vision→audio.
        # This trains both modalities independently within the same model,
        # then evaluates composition (both) to track emergence over time.
        vision_first = (args.train_modality == "interleaved_vision_first")
        interleaved_eval_modalities = ["text", "audio", "image", "both"]
        global_step = 0
        best_composed_score = -1.0
        compat_layers: List[int] = []
        compatibility_state: Optional[Dict[str, Any]] = None
        compat_objective_enabled = bool(
            args.compat_reg_enable
            or args.compat_add_reg_enable
            or args.compat_transport_enable
            or args.compat_noharm_enable
            or args.compat_logit_fusion_enable
            or args.compat_icm_cancel_enable
            or args.compat_poe_enable
            or args.compat_gate_add_enable
            or args.compat_icm_identity_lambda > 0.0
            or args.compat_icm_small_lambda > 0.0
            or args.compat_icm_util_lambda > 0.0
        )

        if compat_objective_enabled:
            compat_layers = _parse_layer_list(args.compat_reg_layers)
            if not compat_layers:
                compat_layers = _get_modality_fusion_layers(model, "vision")
            if not compat_layers:
                print("[compat] No vision fusion layers found; disabling compatibility objectives.", flush=True)
                args.compat_reg_enable = False
                args.compat_add_reg_enable = False
                args.compat_transport_enable = False
                args.compat_noharm_enable = False
                args.compat_logit_fusion_enable = False
                args.compat_icm_cancel_enable = False
                args.compat_poe_enable = False
                args.compat_routing_enable = False
                compat_objective_enabled = False
            else:
                print(
                    f"[compat] enabled lambda={args.compat_reg_lambda} rank={args.compat_reg_rank} "
                    f"audio_samples={args.compat_reg_audio_samples} refresh_every={args.compat_reg_refresh_every} "
                    f"layers={compat_layers} weight_by_shift_norm={args.compat_reg_weight_by_shift_norm}",
                    flush=True,
                )
                if args.compat_add_reg_enable:
                    print(
                        f"[compat] additivity objective enabled lambda={args.compat_add_reg_lambda} "
                        f"every={args.compat_add_reg_every} layers={args.compat_add_reg_layers or compat_layers} "
                        f"normalize={args.compat_add_reg_normalize}",
                        flush=True,
                    )
                if args.compat_transport_enable:
                    transport_layers = args.compat_transport_layers or args.compat_add_reg_layers or compat_layers
                    print(
                        f"[compat] transport objective enabled lambda={args.compat_transport_lambda} "
                        f"cap={args.compat_transport_cap} layers={transport_layers} "
                        f"normalize={args.compat_transport_normalize}",
                        flush=True,
                    )
                if args.compat_noharm_enable:
                    print(
                        f"[compat] no-harm objective enabled lambda={args.compat_noharm_lambda} "
                        f"margin={args.compat_noharm_margin} "
                        f"use_best_single={args.compat_noharm_use_best_single} "
                        f"start_step={args.compat_icm_noharm_start_step}",
                        flush=True,
                    )
                if args.compat_icm_cancel_enable:
                    print(
                        f"[compat] icm-cancel objective enabled lambda={args.compat_icm_cancel_lambda} "
                        f"loss={args.compat_icm_cancel_loss_type} "
                        f"logit_temp={args.compat_icm_cancel_logit_temp} "
                        f"start_step={args.compat_icm_cancel_start_step}",
                        flush=True,
                    )
                if args.compat_icm_identity_lambda > 0.0:
                    print(
                        f"[compat] icm-identity objective enabled lambda={args.compat_icm_identity_lambda}",
                        flush=True,
                    )
                if args.compat_icm_small_lambda > 0.0:
                    print(
                        f"[compat] icm-small objective enabled lambda={args.compat_icm_small_lambda}",
                        flush=True,
                    )
                if args.compat_icm_util_lambda > 0.0:
                    print(
                        f"[compat] icm-util objective enabled lambda={args.compat_icm_util_lambda} "
                        f"start_step={args.compat_icm_util_start_step}",
                        flush=True,
                    )
                if args.compat_logit_fusion_enable:
                    print(
                        f"[compat] logit-fusion objective enabled lambda={args.compat_logit_fusion_lambda} "
                        f"conf_temp={args.compat_logit_fusion_conf_temp}",
                        flush=True,
                    )
                if args.compat_poe_enable:
                    print(
                        f"[compat] residual-PoE objective enabled lambda={args.compat_poe_lambda} "
                        f"weight_temp={args.compat_poe_weight_temp} "
                        f"loss={args.compat_poe_loss_type} logit_temp={args.compat_poe_logit_temp}",
                        flush=True,
                    )
                if args.compat_routing_enable:
                    print(
                        f"[compat] confidence routing enabled min_scale={args.compat_routing_min_scale} "
                        f"max_scale={args.compat_routing_max_scale}",
                        flush=True,
                    )
                if args.compat_gate_add_enable:
                    print(
                        f"[compat] gate-add objective enabled lambda={args.compat_gate_add_lambda} "
                        f"pairing={args.compat_gate_pairing} target_mode={args.compat_gate_target_mode} "
                        f"target={args.compat_gate_product_target} rho_beta={args.compat_gate_rho_beta} "
                        f"gmin={args.compat_gate_min_effective} floor_lambda={args.compat_gate_floor_lambda}",
                        flush=True,
                    )

        # RKCA subspace avoidance state
        rkca_subspace_enabled = bool(getattr(args, "rkca_subspace_enable", False))
        rkca_subspace_state_first: Optional[Dict[str, Any]] = None
        rkca_subspace_state_second: Optional[Dict[str, Any]] = None
        if rkca_subspace_enabled:
            print(
                f"[rkca-subspace] enabled lambda={args.rkca_subspace_lambda} "
                f"bank_size={args.rkca_subspace_bank_size} rank={args.rkca_subspace_rank} "
                f"refit_every={args.rkca_subspace_refit_every}",
                flush=True,
            )

        # Epoch 0: text-only baseline before any adapter training
        print("\n[epoch 0/{}] (text-only baseline)".format(args.num_epochs))
        epoch_result = {"epoch": 0, "audio_train_loss": 0.0, "vision_train_loss": 0.0, "eval": {}}
        for modality in interleaved_eval_modalities:
            metrics = evaluate(model, val_loader, tokenizer, device, modality, args)
            epoch_result["eval"][modality] = metrics
            print(
                f"  [eval:{modality}] raw_em={metrics['exact_match']:.2f} "
                f"extracted_em={metrics['extracted_match']:.2f} "
                f"cat_f1={metrics['categorical_f1']:.2f} "
                f"f1={metrics['token_f1']:.2f} n={metrics['num_samples']}"
            )
        if args.layer_additivity_probe:
            probe = run_layer_additivity_probe(model, val_ds, tokenizer, device, args)
            if probe is not None:
                epoch_result["layer_additivity_probe"] = probe
        text_em = epoch_result["eval"]["text"]["extracted_match"]
        print(f"  [baseline] text-only={text_em:.2f}")
        history.append(epoch_result)
        if wandb_run is not None:
            log_payload = {"epoch": 0}
            for modality, metrics in epoch_result["eval"].items():
                log_payload[f"val/{modality}/exact_match"] = metrics["exact_match"]
                log_payload[f"val/{modality}/extracted_match"] = metrics["extracted_match"]
                log_payload[f"val/{modality}/token_f1"] = metrics["token_f1"]
                log_payload[f"val/{modality}/categorical_f1"] = metrics["categorical_f1"]
            probe = epoch_result.get("layer_additivity_probe")
            if probe is not None and probe.get("ranking"):
                top = probe["ranking"][0]
                log_payload["probe/best_layer"] = top["layer"]
                log_payload["probe/best_layer_epsilon"] = top["epsilon_additivity"]
                log_payload["probe/best_layer_both_extracted"] = top["both_extracted_match"]
            wandb_run.log(log_payload, step=0)
        with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        for epoch in range(args.num_epochs):
            order_label = "vision→audio" if vision_first else "audio→vision"
            print(f"\n[epoch {epoch + 1}/{args.num_epochs}] (interleaved, {order_label})")

            # Determine phase order
            first_modality = "image" if vision_first else "audio"
            second_modality = "audio" if vision_first else "image"
            first_label = "vision" if vision_first else "audio"
            second_label = "audio" if vision_first else "vision"

            # RKCA subspace: bank second modality tokens from prior epoch
            if rkca_subspace_enabled and epoch > 0:
                refit_every = max(1, int(args.rkca_subspace_refit_every))
                if epoch % refit_every == 0:
                    # Bank second modality tokens (trained last epoch) → basis for first modality to avoid
                    second_mod_name = "audio" if second_modality == "audio" else "vision"
                    print(f"  [rkca-subspace] banking {second_mod_name} tokens for {first_label} phase...", flush=True)
                    bank = _collect_rkca_token_bank(
                        model, train_loader, device, args,
                        modality=second_mod_name,
                        bank_size=args.rkca_subspace_bank_size,
                        use_bf16_amp=use_bf16_amp,
                    )
                    if bank is not None:
                        flat_bank = bank.reshape(-1, bank.size(-1))  # (bank_size*N_tokens, H)
                        basis = _fit_shift_basis(flat_bank, rank=args.rkca_subspace_rank)
                        if basis is not None:
                            rkca_subspace_state_first = {"basis": basis, "lambda": args.rkca_subspace_lambda}
                            print(f"  [rkca-subspace] fitted basis for {first_label} phase: {tuple(basis.shape)}", flush=True)
                        else:
                            rkca_subspace_state_first = None
                            print(f"  [rkca-subspace] PCA fit failed for {first_label} phase", flush=True)
                    else:
                        rkca_subspace_state_first = None
                        print(f"  [rkca-subspace] no {second_mod_name} tokens collected", flush=True)

            # Phase A: First modality training pass
            print(f"  [phase:{first_label}] training {first_label} adapters...")
            args_first = argparse.Namespace(**vars(args))
            args_first.train_modality = first_modality
            first_loss, global_step = train_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, args_first,
                use_bf16_amp=use_bf16_amp,
                wandb_run=wandb_run, global_step=global_step,
                rkca_subspace_state=rkca_subspace_state_first,
            )
            print(f"  [phase:{first_label}] loss={first_loss:.4f}")

            if compat_objective_enabled:
                refresh_every = max(1, int(args.compat_reg_refresh_every))
                refresh_now = compatibility_state is None or (epoch % refresh_every == 0)
                if refresh_now:
                    compatibility_state = collect_audio_shift_subspaces(
                        model=model,
                        dataloader=train_loader,
                        device=device,
                        args=args,
                        layer_indices=compat_layers,
                        use_bf16_amp=use_bf16_amp,
                    )
                if compatibility_state is None:
                    print("[compat] Warning: compatibility state unavailable; second phase will run without unpaired composition objectives.", flush=True)

            # RKCA subspace: bank first modality tokens → basis for second modality to avoid
            if rkca_subspace_enabled:
                refit_every = max(1, int(args.rkca_subspace_refit_every))
                if rkca_subspace_state_second is None or (epoch % refit_every == 0):
                    first_mod_name = "audio" if first_modality == "audio" else "vision"
                    print(f"  [rkca-subspace] banking {first_mod_name} tokens for {second_label} phase...", flush=True)
                    bank = _collect_rkca_token_bank(
                        model, train_loader, device, args,
                        modality=first_mod_name,
                        bank_size=args.rkca_subspace_bank_size,
                        use_bf16_amp=use_bf16_amp,
                    )
                    if bank is not None:
                        flat_bank = bank.reshape(-1, bank.size(-1))
                        basis = _fit_shift_basis(flat_bank, rank=args.rkca_subspace_rank)
                        if basis is not None:
                            rkca_subspace_state_second = {"basis": basis, "lambda": args.rkca_subspace_lambda}
                            print(f"  [rkca-subspace] fitted basis for {second_label} phase: {tuple(basis.shape)}", flush=True)
                        else:
                            rkca_subspace_state_second = None
                            print(f"  [rkca-subspace] PCA fit failed for {second_label} phase", flush=True)
                    else:
                        rkca_subspace_state_second = None
                        print(f"  [rkca-subspace] no {first_mod_name} tokens collected", flush=True)

            # Phase B: Second modality training pass
            print(f"  [phase:{second_label}] training {second_label} adapters...")
            args_second = argparse.Namespace(**vars(args))
            args_second.train_modality = second_modality
            second_loss, global_step = train_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, args_second,
                use_bf16_amp=use_bf16_amp,
                wandb_run=wandb_run, global_step=global_step,
                compatibility_state=compatibility_state,
                rkca_subspace_state=rkca_subspace_state_second,
            )
            print(f"  [phase:{second_label}] loss={second_loss:.4f}")

            # Map back to audio/vision for logging
            audio_loss = second_loss if vision_first else first_loss
            vision_loss = first_loss if vision_first else second_loss

            # Phase C: 3-way evaluation
            epoch_result = {
                "epoch": epoch + 1,
                "audio_train_loss": float(audio_loss),
                "vision_train_loss": float(vision_loss),
                "eval": {},
            }
            if compat_objective_enabled and compatibility_state is not None:
                epoch_result["compat_reg"] = {
                    "lambda": float(args.compat_reg_lambda),
                    "layers": list(compatibility_state.get("layers", [])),
                    "stats_by_layer": compatibility_state.get("stats_by_layer", {}),
                    "layer_weights": compatibility_state.get("layer_weights", {}),
                    "num_samples_collected": int(compatibility_state.get("num_samples_collected", 0)),
                    "additivity": {
                        "enabled": bool(args.compat_add_reg_enable),
                        "lambda": float(args.compat_add_reg_lambda),
                        "every": int(args.compat_add_reg_every),
                        "layers": _parse_layer_list(args.compat_add_reg_layers)
                        if args.compat_add_reg_layers
                        else list(compatibility_state.get("layers", [])),
                    },
                    "transport": {
                        "enabled": bool(args.compat_transport_enable),
                        "lambda": float(args.compat_transport_lambda),
                        "cap": float(args.compat_transport_cap),
                        "normalize": bool(args.compat_transport_normalize),
                        "layers": _parse_layer_list(args.compat_transport_layers)
                        if args.compat_transport_layers
                        else list(compatibility_state.get("layers", [])),
                    },
                    "icm": {
                        "enabled": bool(args.icm_enable),
                        "dim": int(args.icm_dim),
                        "heads": int(args.icm_heads),
                        "layers": int(args.icm_layers),
                        "gate_init": float(args.icm_gate_init),
                        "min_modalities": int(args.icm_min_modalities),
                        "util_target": float(args.icm_util_target),
                        "cancel_enabled": bool(args.compat_icm_cancel_enable),
                        "cancel_lambda": float(args.compat_icm_cancel_lambda),
                        "cancel_loss_type": str(args.compat_icm_cancel_loss_type),
                        "cancel_logit_temp": float(args.compat_icm_cancel_logit_temp),
                        "cancel_start_step": int(args.compat_icm_cancel_start_step),
                        "identity_lambda": float(args.compat_icm_identity_lambda),
                        "small_lambda": float(args.compat_icm_small_lambda),
                        "util_lambda": float(args.compat_icm_util_lambda),
                        "util_start_step": int(args.compat_icm_util_start_step),
                        "noharm_start_step": int(args.compat_icm_noharm_start_step),
                    },
                    "noharm": {
                        "enabled": bool(args.compat_noharm_enable),
                        "lambda": float(args.compat_noharm_lambda),
                        "margin": float(args.compat_noharm_margin),
                        "use_best_single": bool(args.compat_noharm_use_best_single),
                    },
                    "logit_fusion": {
                        "enabled": bool(args.compat_logit_fusion_enable),
                        "lambda": float(args.compat_logit_fusion_lambda),
                        "conf_temp": float(args.compat_logit_fusion_conf_temp),
                    },
                    "residual_poe": {
                        "enabled": bool(args.compat_poe_enable),
                        "lambda": float(args.compat_poe_lambda),
                        "weight_temp": float(args.compat_poe_weight_temp),
                        "loss_type": str(args.compat_poe_loss_type),
                        "logit_temp": float(args.compat_poe_logit_temp),
                    },
                    "routing": {
                        "enabled": bool(args.compat_routing_enable),
                        "min_scale": float(args.compat_routing_min_scale),
                        "max_scale": float(args.compat_routing_max_scale),
                    },
                }
            for modality in interleaved_eval_modalities:
                metrics = evaluate(model, val_loader, tokenizer, device, modality, args)
                epoch_result["eval"][modality] = metrics
                print(
                    f"  [eval:{modality}] raw_em={metrics['exact_match']:.2f} "
                    f"extracted_em={metrics['extracted_match']:.2f} "
                    f"cat_f1={metrics['categorical_f1']:.2f} "
                    f"f1={metrics['token_f1']:.2f} n={metrics['num_samples']}"
                )

            if args.layer_additivity_probe and ((epoch + 1) % max(1, args.layer_probe_every) == 0):
                probe = run_layer_additivity_probe(model, val_ds, tokenizer, device, args)
                if probe is not None:
                    epoch_result["layer_additivity_probe"] = probe

            # Composition summary line
            text_em_now = epoch_result["eval"]["text"]["extracted_match"]
            audio_em = epoch_result["eval"]["audio"]["extracted_match"]
            image_em = epoch_result["eval"]["image"]["extracted_match"]
            both_em = epoch_result["eval"]["both"]["extracted_match"]
            composition_gain = both_em - max(audio_em, image_em)
            gain_over_text = both_em - text_em_now
            print(
                f"  [composition] text={text_em_now:.2f} audio={audio_em:.2f} "
                f"vision={image_em:.2f} both={both_em:.2f} "
                f"gain_vs_best_single={composition_gain:+.2f} "
                f"gain_vs_text={gain_over_text:+.2f}"
            )

            history.append(epoch_result)
            if wandb_run is not None:
                log_payload = {
                    "epoch": epoch + 1,
                    "train/audio_loss": float(audio_loss),
                    "train/vision_loss": float(vision_loss),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "composition/gain_vs_best_single": composition_gain,
                    "composition/gain_vs_text": gain_over_text,
                }
                if compat_objective_enabled and compatibility_state is not None:
                    log_payload["compat/lambda"] = float(args.compat_reg_lambda)
                    log_payload["compat/num_layers"] = float(len(compatibility_state.get("layers", [])))
                    log_payload["compat/num_samples_collected"] = float(
                        compatibility_state.get("num_samples_collected", 0)
                    )
                    log_payload["compat/add_enabled"] = float(bool(args.compat_add_reg_enable))
                    if args.compat_add_reg_enable:
                        log_payload["compat/add_lambda"] = float(args.compat_add_reg_lambda)
                        log_payload["compat/add_every"] = float(args.compat_add_reg_every)
                    log_payload["compat/transport_enabled"] = float(bool(args.compat_transport_enable))
                    if args.compat_transport_enable:
                        log_payload["compat/transport_lambda"] = float(args.compat_transport_lambda)
                        log_payload["compat/transport_cap"] = float(args.compat_transport_cap)
                        log_payload["compat/transport_normalize"] = float(bool(args.compat_transport_normalize))
                    log_payload["icm/enabled"] = float(bool(args.icm_enable))
                    if args.icm_enable:
                        log_payload["icm/dim"] = float(args.icm_dim)
                        log_payload["icm/heads"] = float(args.icm_heads)
                        log_payload["icm/layers"] = float(args.icm_layers)
                        log_payload["icm/gate_init"] = float(args.icm_gate_init)
                        log_payload["icm/min_modalities"] = float(args.icm_min_modalities)
                        log_payload["icm/util_target"] = float(args.icm_util_target)
                    log_payload["compat/icm_cancel_enabled"] = float(bool(args.compat_icm_cancel_enable))
                    if args.compat_icm_cancel_enable:
                        log_payload["compat/icm_cancel_lambda"] = float(args.compat_icm_cancel_lambda)
                        log_payload["compat/icm_cancel_loss_is_kl"] = float(str(args.compat_icm_cancel_loss_type).lower() == "kl")
                        log_payload["compat/icm_cancel_temp"] = float(args.compat_icm_cancel_logit_temp)
                        log_payload["compat/icm_cancel_start_step"] = float(args.compat_icm_cancel_start_step)
                    log_payload["compat/icm_identity_lambda"] = float(args.compat_icm_identity_lambda)
                    log_payload["compat/icm_small_lambda"] = float(args.compat_icm_small_lambda)
                    log_payload["compat/icm_util_lambda"] = float(args.compat_icm_util_lambda)
                    log_payload["compat/icm_util_start_step"] = float(args.compat_icm_util_start_step)
                    log_payload["compat/icm_noharm_start_step"] = float(args.compat_icm_noharm_start_step)
                    log_payload["compat/noharm_enabled"] = float(bool(args.compat_noharm_enable))
                    if args.compat_noharm_enable:
                        log_payload["compat/noharm_lambda"] = float(args.compat_noharm_lambda)
                        log_payload["compat/noharm_margin"] = float(args.compat_noharm_margin)
                        log_payload["compat/noharm_use_best_single"] = float(bool(args.compat_noharm_use_best_single))
                    log_payload["compat/logit_fusion_enabled"] = float(bool(args.compat_logit_fusion_enable))
                    if args.compat_logit_fusion_enable:
                        log_payload["compat/logit_fusion_lambda"] = float(args.compat_logit_fusion_lambda)
                        log_payload["compat/logit_fusion_conf_temp"] = float(args.compat_logit_fusion_conf_temp)
                    log_payload["compat/poe_enabled"] = float(bool(args.compat_poe_enable))
                    if args.compat_poe_enable:
                        log_payload["compat/poe_lambda"] = float(args.compat_poe_lambda)
                        log_payload["compat/poe_weight_temp"] = float(args.compat_poe_weight_temp)
                        log_payload["compat/poe_loss_type_is_kl"] = float(str(args.compat_poe_loss_type).lower() == "kl")
                        log_payload["compat/poe_logit_temp"] = float(args.compat_poe_logit_temp)
                    log_payload["compat/routing_enabled"] = float(bool(args.compat_routing_enable))
                    if args.compat_routing_enable:
                        log_payload["compat/routing_min_scale"] = float(args.compat_routing_min_scale)
                        log_payload["compat/routing_max_scale"] = float(args.compat_routing_max_scale)
                for modality, metrics in epoch_result["eval"].items():
                    log_payload[f"val/{modality}/exact_match"] = metrics["exact_match"]
                    log_payload[f"val/{modality}/extracted_match"] = metrics["extracted_match"]
                    log_payload[f"val/{modality}/token_f1"] = metrics["token_f1"]
                    log_payload[f"val/{modality}/categorical_f1"] = metrics["categorical_f1"]
                probe = epoch_result.get("layer_additivity_probe")
                if probe is not None and probe.get("ranking"):
                    top = probe["ranking"][0]
                    log_payload["probe/best_layer"] = top["layer"]
                    log_payload["probe/best_layer_epsilon"] = top["epsilon_additivity"]
                    log_payload["probe/best_layer_both_extracted"] = top["both_extracted_match"]
                wandb_run.log(log_payload, step=epoch + 1)

            # Save per-epoch checkpoint (for post-hoc composition analysis)
            epoch_ckpt_path = args.output_dir / f"epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), epoch_ckpt_path)
            print(f"  [save] epoch checkpoint -> {epoch_ckpt_path}")

            # Track best composed score
            if both_em > best_composed_score:
                best_composed_score = both_em
                best_score = both_em
                best_ckpt_path = args.output_dir / "best_model.pt"
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  [save] best composition checkpoint -> {best_ckpt_path}")

            with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        final_path = args.output_dir / "final_model.pt"
        torch.save(model.state_dict(), final_path)
    else:
        global_step = 0
        reverse_compat_state = None
        reverse_compat_enabled = bool(
            getattr(args, "compat_reg_reverse", False)
            and getattr(args, "compat_reg_enable", False)
        )
        if reverse_compat_enabled:
            reverse_layers = _parse_layer_list(args.compat_reg_layers)
            if not reverse_layers:
                reverse_layers = _get_modality_fusion_layers(model, "audio")
            if not reverse_layers:
                print("[compat-reverse] No audio fusion layers found; disabling reverse compat.", flush=True)
                reverse_compat_enabled = False

        for epoch in range(args.num_epochs):
            if reverse_compat_enabled:
                refresh_every = max(1, int(args.compat_reg_refresh_every))
                if reverse_compat_state is None or (epoch % refresh_every == 0):
                    reverse_compat_state = collect_vision_shift_subspaces(
                        model=model, dataloader=train_loader, device=device,
                        args=args, layer_indices=reverse_layers,
                        use_bf16_amp=use_bf16_amp,
                    )

            print(f"\n[epoch {epoch + 1}/{args.num_epochs}]")
            train_loss, global_step = train_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, args,
                use_bf16_amp=use_bf16_amp,
                wandb_run=wandb_run, global_step=global_step,
                compatibility_state=reverse_compat_state,
            )
            epoch_result = {"epoch": epoch + 1, "train_loss": float(train_loss), "eval": {}}

            for modality in eval_modalities:
                metrics = evaluate(model, val_loader, tokenizer, device, modality, args)
                epoch_result["eval"][modality] = metrics
                print(
                    f"  [eval:{modality}] raw_em={metrics['exact_match']:.2f} "
                    f"extracted_em={metrics['extracted_match']:.2f} "
                    f"cat_f1={metrics['categorical_f1']:.2f} "
                    f"f1={metrics['token_f1']:.2f} n={metrics['num_samples']}"
                )

            if args.layer_additivity_probe and ((epoch + 1) % max(1, args.layer_probe_every) == 0):
                probe = run_layer_additivity_probe(model, val_ds, tokenizer, device, args)
                if probe is not None:
                    epoch_result["layer_additivity_probe"] = probe

            history.append(epoch_result)
            if wandb_run is not None:
                log_payload = {
                    "epoch": epoch + 1,
                    "train/loss": float(train_loss),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                for modality, metrics in epoch_result["eval"].items():
                    log_payload[f"val/{modality}/exact_match"] = metrics["exact_match"]
                    log_payload[f"val/{modality}/extracted_match"] = metrics["extracted_match"]
                    log_payload[f"val/{modality}/token_f1"] = metrics["token_f1"]
                    log_payload[f"val/{modality}/categorical_f1"] = metrics["categorical_f1"]
                probe = epoch_result.get("layer_additivity_probe")
                if probe is not None and probe.get("ranking"):
                    top = probe["ranking"][0]
                    log_payload["probe/best_layer"] = top["layer"]
                    log_payload["probe/best_layer_epsilon"] = top["epsilon_additivity"]
                    log_payload["probe/best_layer_both_extracted"] = top["both_extracted_match"]
                wandb_run.log(log_payload, step=epoch + 1)

            # Track best score using extracted_match (more fair for generative models)
            score_key = args.train_modality  # "both" or "audio"
            score = epoch_result["eval"].get(score_key, {}).get("extracted_match", -1.0)
            if score > best_score:
                best_score = score
                ckpt_path = args.output_dir / "best_model.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"  [save] best checkpoint -> {ckpt_path}")

            with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        final_path = args.output_dir / "final_model.pt"
        torch.save(model.state_dict(), final_path)

    results = {
        "dataset": args.dataset,
        "architecture": "pre_ffn",
        "train_modality": args.train_modality,
        "eval_modalities": eval_modalities,
        "best_exact_match": best_score,
        "best_modality": args.train_modality,
        "ttc_enabled": bool(args.ttc_enable),
        "ttc_objective": str(args.ttc_objective),
        "compose_calibration_enabled": bool(args.compose_calibration_enable),
        "compose_calibration_trainable": str(args.compose_calibration_trainable),
        "train_max_samples": int(args.train_max_samples),
        "val_max_samples": int(args.val_max_samples),
        "history_path": str(args.output_dir / "history.json"),
    }
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("[summary] " + json.dumps(results, indent=2))

    if wandb_run is not None:
        wandb_run.summary["best_exact_match"] = best_score
        wandb_run.summary["results_path"] = str(args.output_dir / "results.json")
        wandb_run.finish()


if __name__ == "__main__":
    main()
