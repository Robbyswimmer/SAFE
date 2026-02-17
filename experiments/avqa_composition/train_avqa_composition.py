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
    # Slim projector: output at bottleneck_dim instead of llm_hidden_size
    if getattr(args, "slim_projector", False):
        slim_dim = fusion_cfg.get("bottleneck_dim", 256)
        cfg.setdefault("projector_config", {})["output_dim"] = slim_dim
        print(f"[SlimProjector] Projector output_dim set to {slim_dim} (bottleneck_dim)", flush=True)

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
) -> tuple:
    model.train()
    total_loss = 0.0
    total_batches = 0
    num_batches = len(dataloader)
    log_every = max(1, min(100, num_batches // 20))  # Log at least every 100 steps
    optimizer.zero_grad()
    trainable_for_clip = [p for p in model.parameters() if p.requires_grad]

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

        if args.fp16 and torch.cuda.is_available():
            amp_ctx = autocast(enabled=True, dtype=torch.float16)
        elif use_bf16_amp and torch.cuda.is_available():
            amp_ctx = autocast(enabled=True, dtype=torch.bfloat16)
        else:
            amp_ctx = nullcontext()

        with amp_ctx:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=inputs.get("labels"),
                pixel_values=inputs.get("pixel_values"),
                audio_tokens=audio_tokens,
                audio_attention_mask=audio_mask,
                gate=gate_value,
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
            print(
                f"  [train] step {step + 1}/{num_batches} "
                f"loss={avg_loss:.4f} lr={current_lr:.2e} gate={gate_value:.3f}",
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
    p.add_argument("--slim-projector", action="store_true",
                   help="Output projector at bottleneck_dim instead of llm_hidden_size, "
                        "saving ~80%% of trainable params by eliminating expand-then-compress path")

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

            # Phase B: Vision training pass
            print(f"  [phase:vision] training vision adapters...")
            args_vision = argparse.Namespace(**vars(args))
            args_vision.train_modality = "image"
            vision_loss, global_step = train_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, args_vision,
                use_bf16_amp=use_bf16_amp,
                wandb_run=wandb_run, global_step=global_step,
            )
            print(f"  [phase:vision] loss={vision_loss:.4f}")

            # Phase C: 3-way evaluation
            epoch_result = {
                "epoch": epoch + 1,
                "audio_train_loss": float(audio_loss),
                "vision_train_loss": float(vision_loss),
                "eval": {},
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
