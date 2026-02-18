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
from typing import Any, Dict, List, Optional, Sequence

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
    cfg["freeze_base_vl"] = True
    cfg["freeze_audio_encoder"] = args.freeze_audio_encoder
    if args.label_smoothing is not None:
        cfg["label_smoothing"] = args.label_smoothing

    fusion_cfg["fusion_mode"] = "residual"
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
    # Slim projector (default ON): output at bottleneck_dim instead of llm_hidden_size
    # Saves ~80% of trainable params. Disable with --no-slim-projector.
    if getattr(args, "slim_projector", True):
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
    }
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
    bank_size = int(getattr(args, "compat_add_bank_size", 64))
    add_bank_enabled = bool(
        (
            getattr(args, "compat_add_reg_enable", False)
            or getattr(args, "compat_noharm_enable", False)
            or getattr(args, "compat_logit_fusion_enable", False)
            or getattr(args, "compat_poe_enable", False)
        )
        and bank_size > 0
    )
    collected = 0

    for batch in dataloader:
        mm = resolve_modality_batch(batch, "audio")
        inputs = model.prepare_multimodal_inputs(
            text=batch["questions"],
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
            for i in range(tok_cpu.size(0)):
                token_bank.append(tok_cpu[i:i + 1].clone())
                if mask_cpu is not None:
                    mask_bank.append(mask_cpu[i:i + 1].clone())
                else:
                    mask_bank.append(None)
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
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    token_bank = compatibility_state.get("audio_token_bank", [])
    mask_bank = compatibility_state.get("audio_mask_bank", [])
    if not token_bank:
        return None, None
    idx = random.randrange(len(token_bank))
    tok = token_bank[idx]
    msk = mask_bank[idx] if idx < len(mask_bank) else None
    return tok, msk


def compute_unpaired_additivity_regularizer(
    model: SAFEModel,
    inputs: Dict[str, Any],
    compatibility_state: Dict[str, Any],
    gate_value: float,
    add_layers: Sequence[int],
    layer_weights: Optional[Dict[int, float]] = None,
    normalize: bool = True,
    no_harm_enable: bool = False,
    no_harm_margin: float = 0.0,
    no_harm_use_best_single: bool = False,
    logit_fusion_enable: bool = False,
    logit_fusion_conf_temp: float = 0.5,
    poe_enable: bool = False,
    poe_weight_temp: float = 0.5,
    poe_loss_type: str = "kl",
    poe_logit_temp: float = 1.0,
    routing_enable: bool = False,
    routing_min: float = 0.25,
    routing_max: float = 1.0,
) -> tuple[Optional[torch.Tensor], Dict[int, float], Dict[str, Optional[torch.Tensor]], Dict[str, float]]:
    """
    Unpaired additivity loss (no joint AV supervision):
      ||Δ_av - Δ_a - Δ_v||^2 at selected layers.
    Audio is sampled from the compatibility token bank.
    """
    aux_losses: Dict[str, Optional[torch.Tensor]] = {
        "no_harm": None,
        "logit_fusion": None,
        "poe_consistency": None,
    }
    aux_stats: Dict[str, float] = {}

    need_hidden_states = bool(add_layers)
    if not need_hidden_states and not (no_harm_enable or logit_fusion_enable or poe_enable):
        return None, {}, aux_losses, aux_stats

    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        return None, {}, aux_losses, aux_stats

    sampled_audio, sampled_mask = _sample_audio_bank_entry(compatibility_state)
    if sampled_audio is None:
        return None, {}, aux_losses, aux_stats

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

    reg_loss: Optional[torch.Tensor] = None
    per_layer: Dict[int, float] = {}
    if need_hidden_states:
        hs_text = out_text.get("all_hidden_states") if isinstance(out_text, dict) else None
        hs_audio = out_audio.get("all_hidden_states") if isinstance(out_audio, dict) else None
        hs_vision = out_vision.get("all_hidden_states") if isinstance(out_vision, dict) else None
        hs_both = out_both.get("all_hidden_states") if isinstance(out_both, dict) else None
        if any(x is None for x in (hs_text, hs_audio, hs_vision, hs_both)):
            return None, {}, aux_losses, aux_stats

        pooled_t = _extract_pooled_layer_states(hs_text, add_layers, labels=labels, attention_mask=attention_mask)
        pooled_a = _extract_pooled_layer_states(hs_audio, add_layers, labels=labels, attention_mask=attention_mask)
        pooled_v = _extract_pooled_layer_states(hs_vision, add_layers, labels=labels, attention_mask=attention_mask)
        pooled_av = _extract_pooled_layer_states(hs_both, add_layers, labels=labels, attention_mask=attention_mask)

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

            weight = 1.0
            if layer_weights is not None:
                weight = float(layer_weights.get(int(layer), 1.0))
            weighted = layer_loss * weight
            per_layer[int(layer)] = float(weighted.detach().item())
            reg_loss = weighted if reg_loss is None else (reg_loss + weighted)

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

    if poe_enable:
        logits_text = out_text.get("logits") if isinstance(out_text, dict) else getattr(out_text, "logits", None)
        logits_audio = out_audio.get("logits") if isinstance(out_audio, dict) else getattr(out_audio, "logits", None)
        logits_vision = out_vision.get("logits") if isinstance(out_vision, dict) else getattr(out_vision, "logits", None)
        logits_both = out_both.get("logits") if isinstance(out_both, dict) else getattr(out_both, "logits", None)

        flat_t = _flatten_valid_logits(logits_text, labels=labels)
        flat_a = _flatten_valid_logits(logits_audio, labels=labels)
        flat_v = _flatten_valid_logits(logits_vision, labels=labels)
        flat_av = _flatten_valid_logits(logits_both, labels=labels)

        if all(torch.is_tensor(x) for x in (flat_t, flat_a, flat_v, flat_av)):
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

    return reg_loss, per_layer, aux_losses, aux_stats


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
    total_poe_reg = 0.0
    poe_reg_batches = 0
    total_route_scale = 0.0
    route_scale_batches = 0
    total_batches = 0
    num_batches = len(dataloader)
    log_every = max(1, min(100, num_batches // 20))  # Log at least every 100 steps
    optimizer.zero_grad()
    trainable_for_clip = [p for p in model.parameters() if p.requires_grad]

    compat_state_available = bool(
        args.train_modality == "image"
        and compatibility_state is not None
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
            or getattr(args, "compat_poe_enable", False)
        )
        and compatibility_state.get("audio_token_bank")
    )
    add_reg_enabled = bool(unpaired_aux_enabled and getattr(args, "compat_add_reg_enable", False))
    noharm_enabled = bool(unpaired_aux_enabled and getattr(args, "compat_noharm_enable", False))
    logit_fusion_enabled = bool(unpaired_aux_enabled and getattr(args, "compat_logit_fusion_enable", False))

    add_layers = _parse_layer_list(getattr(args, "compat_add_reg_layers", "")) if add_reg_enabled else []
    if add_reg_enabled and not add_layers:
        add_layers = list(compat_layers)
    add_every = max(1, int(getattr(args, "compat_add_reg_every", 200)))
    add_lambda = float(getattr(args, "compat_add_reg_lambda", 0.0))
    add_norm = bool(getattr(args, "compat_add_reg_normalize", True))
    noharm_lambda = float(getattr(args, "compat_noharm_lambda", 0.0))
    noharm_margin = float(getattr(args, "compat_noharm_margin", 0.0))
    noharm_use_best_single = bool(getattr(args, "compat_noharm_use_best_single", False))
    logit_fusion_lambda = float(getattr(args, "compat_logit_fusion_lambda", 0.0))
    logit_fusion_conf_temp = float(getattr(args, "compat_logit_fusion_conf_temp", 0.5))
    poe_enabled = bool(unpaired_aux_enabled and getattr(args, "compat_poe_enable", False))
    poe_lambda = float(getattr(args, "compat_poe_lambda", 0.0))
    poe_weight_temp = float(getattr(args, "compat_poe_weight_temp", 0.5))
    poe_loss_type = str(getattr(args, "compat_poe_loss_type", "kl"))
    poe_logit_temp = float(getattr(args, "compat_poe_logit_temp", 1.0))
    routing_enabled = bool(getattr(args, "compat_routing_enable", False))
    routing_min_scale = float(getattr(args, "compat_routing_min_scale", 0.25))
    routing_max_scale = float(getattr(args, "compat_routing_max_scale", 1.0))
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
            text=batch["questions"],
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

            # Optional unpaired composition objectives on sparse steps
            run_unpaired_aux = (
                unpaired_aux_enabled
                and (
                    (add_reg_enabled and add_lambda > 0.0)
                    or (noharm_enabled and noharm_lambda > 0.0)
                    or (logit_fusion_enabled and logit_fusion_lambda > 0.0)
                    or (poe_enabled and poe_lambda > 0.0)
                )
                and (step % add_every == 0)
            )
            if run_unpaired_aux:
                add_reg, _, aux_losses, aux_stats = compute_unpaired_additivity_regularizer(
                    model=model,
                    inputs=inputs,
                    compatibility_state=compatibility_state,
                    gate_value=gate_value,
                    add_layers=add_layers,
                    layer_weights=compat_layer_weights,
                    normalize=add_norm,
                    no_harm_enable=noharm_enabled,
                    no_harm_margin=noharm_margin,
                    no_harm_use_best_single=noharm_use_best_single,
                    logit_fusion_enable=logit_fusion_enabled,
                    logit_fusion_conf_temp=logit_fusion_conf_temp,
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
                if noharm_enabled and noharm_loss is not None and noharm_lambda > 0.0:
                    total_noharm_reg += float(noharm_loss.detach().item())
                    noharm_reg_batches += 1
                    loss = loss + noharm_lambda * noharm_loss

                logit_loss = aux_losses.get("logit_fusion")
                if logit_fusion_enabled and logit_loss is not None and logit_fusion_lambda > 0.0:
                    total_logit_reg += float(logit_loss.detach().item())
                    logit_reg_batches += 1
                    loss = loss + logit_fusion_lambda * logit_loss

                poe_loss = aux_losses.get("poe_consistency")
                if poe_enabled and poe_loss is not None and poe_lambda > 0.0:
                    total_poe_reg += float(poe_loss.detach().item())
                    poe_reg_batches += 1
                    loss = loss + poe_lambda * poe_loss

                if "route_audio_scale" in aux_stats:
                    total_route_scale += float(aux_stats["route_audio_scale"])
                    route_scale_batches += 1

                if (
                    add_reg is None
                    and noharm_loss is None
                    and logit_loss is None
                    and poe_loss is None
                    and not add_warned
                ):
                    print(
                        "  [compat] warning: unpaired composition objective inactive "
                        "(missing hidden states or audio token bank)",
                        flush=True,
                    )
                    add_warned = True

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
            if route_scale_batches > 0:
                route_avg = total_route_scale / float(max(1, route_scale_batches))
                compat_suffix += f" route_scale={route_avg:.3f}"
            if poe_reg_batches > 0:
                poe_avg = total_poe_reg / float(max(1, poe_reg_batches))
                compat_suffix += f" poe_reg={poe_avg:.4f}"
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

    exact_total = 0.0
    extracted_total = 0.0
    f1_total = 0.0
    categorical_f1_total = 0.0
    count = 0
    by_type: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"exact": 0.0, "extracted": 0.0, "f1": 0.0, "categorical_f1": 0.0, "n": 0.0}
    )
    debug_print_budget = max(0, int(getattr(args, "eval_debug_samples", 0)))

    eval_batches = len(dataloader)
    eval_log_every = max(1, eval_batches // 5)  # Log ~5 times per eval
    for eval_step, batch in enumerate(dataloader):
        mm = resolve_modality_batch(batch, modality)
        inputs = model.prepare_multimodal_inputs(
            text=batch["questions"],
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
    p.add_argument("--num-audio-tokens", type=int, default=8)

    p.add_argument("--train-modality", type=str, default="both", choices=["audio", "image", "both", "interleaved"])
    p.add_argument("--eval-modalities", type=str, default="both,audio,image")
    p.add_argument("--fusion-gate", type=float, default=0.2)
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

    p.set_defaults(
        compat_reg_weight_by_shift_norm=True,
        compat_add_reg_normalize=True,
    )

    p.add_argument("--max-samples", type=int, default=0,
                   help="Limit train/val to N samples for quick sanity runs (0=unlimited)")
    p.add_argument("--eval-debug-samples", type=int, default=0,
                   help="Print first N eval predictions per run for decode/debug checks")
    p.add_argument("--layer-additivity-probe", action="store_true",
                   help="Run per-layer additivity probe (epsilon_l) during eval")
    p.add_argument("--layer-probe-samples", type=int, default=256,
                   help="Max validation samples for layer additivity probe (0=full val)")
    p.add_argument("--layer-probe-every", type=int, default=1,
                   help="Run layer additivity probe every N epochs")
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
                    "eval_modalities": args.eval_modalities,
                    "fusion_layers": args.fusion_layers,
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
                    "label_smoothing": args.label_smoothing,
                    "layer_additivity_probe": args.layer_additivity_probe,
                    "layer_probe_samples": args.layer_probe_samples,
                    "layer_probe_every": args.layer_probe_every,
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
    model.to_device(device)
    if hasattr(model, "get_runtime_device"):
        device = model.get_runtime_device()
        print(f"[info] runtime_device={device}", flush=True)
    if hasattr(model, "fusion_adapter") and model.fusion_adapter is not None:
        layers = getattr(model.fusion_adapter, "fusion_layer_indices", None)
        if layers is not None:
            print(f"[info] effective_fusion_layers={list(layers)}", flush=True)
    tokenizer = model.base_vl.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Composition experiment: load separate modality checkpoints for zero-shot eval
    composition_eval_only = False
    if args.compose_audio_ckpt and args.compose_vision_ckpt:
        model.load_modality_adapters(str(args.compose_audio_ckpt), "audio")
        model.load_modality_adapters(str(args.compose_vision_ckpt), "vision")
        composition_eval_only = True
        print("[composition] Loaded both modality checkpoints — running eval-only", flush=True)

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
    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
    # Interleaved mode does 2 passes per epoch (audio + vision), so double the step count
    passes_per_epoch = 2 if args.train_modality == "interleaved" else 1
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
    elif args.train_modality == "interleaved":
        # ── Interleaved composition training ──
        # Each epoch: train audio adapters → train vision adapters → evaluate all 4:
        #   text (baseline), audio+text, vision+text, audio+vision+text (composition)
        # This trains both modalities independently within the same model,
        # then evaluates composition (both) to track emergence over time.
        interleaved_eval_modalities = ["text", "audio", "image", "both"]
        global_step = 0
        best_composed_score = -1.0
        compat_layers: List[int] = []
        compatibility_state: Optional[Dict[str, Any]] = None
        compat_objective_enabled = bool(
            args.compat_reg_enable
            or args.compat_add_reg_enable
            or args.compat_noharm_enable
            or args.compat_logit_fusion_enable
            or args.compat_poe_enable
        )

        if compat_objective_enabled:
            compat_layers = _parse_layer_list(args.compat_reg_layers)
            if not compat_layers:
                compat_layers = _get_modality_fusion_layers(model, "vision")
            if not compat_layers:
                print("[compat] No vision fusion layers found; disabling compatibility objectives.", flush=True)
                args.compat_reg_enable = False
                args.compat_add_reg_enable = False
                args.compat_noharm_enable = False
                args.compat_logit_fusion_enable = False
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
                if args.compat_noharm_enable:
                    print(
                        f"[compat] no-harm objective enabled lambda={args.compat_noharm_lambda} "
                        f"margin={args.compat_noharm_margin} "
                        f"use_best_single={args.compat_noharm_use_best_single}",
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
            print(f"\n[epoch {epoch + 1}/{args.num_epochs}] (interleaved)")

            # Phase A: Audio training pass
            print(f"  [phase:audio] training audio adapters...")
            args_audio = argparse.Namespace(**vars(args))
            args_audio.train_modality = "audio"
            audio_loss, global_step = train_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, args_audio,
                use_bf16_amp=use_bf16_amp,
                wandb_run=wandb_run, global_step=global_step,
            )
            print(f"  [phase:audio] loss={audio_loss:.4f}")

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
                    print("[compat] Warning: compatibility state unavailable; vision phase will run without unpaired composition objectives.", flush=True)

            # Phase B: Vision training pass
            print(f"  [phase:vision] training vision adapters...")
            args_vision = argparse.Namespace(**vars(args))
            args_vision.train_modality = "image"
            vision_loss, global_step = train_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, args_vision,
                use_bf16_amp=use_bf16_amp,
                wandb_run=wandb_run, global_step=global_step,
                compatibility_state=compatibility_state,
            )
            print(f"  [phase:vision] loss={vision_loss:.4f}")

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
        for epoch in range(args.num_epochs):
            print(f"\n[epoch {epoch + 1}/{args.num_epochs}]")
            train_loss, global_step = train_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, args,
                use_bf16_amp=use_bf16_amp,
                wandb_run=wandb_run, global_step=global_step,
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
