#!/usr/bin/env python3
"""
train_audio_llm_probe.py

Step-1 "LLM readout" for AVE classification:
  audio -> CLAP (frozen) -> projector (trainable) -> SAFE fusion (trainable) -> frozen LLM
  pooled LLM hidden state -> linear head (trainable) -> 28-way classification

This avoids free-form generation and directly measures whether audio fusion changes
LLM representations in a way that supports classification.
"""

import argparse
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:
    wandb = None

from configs.model_configs import get_config
from safe.models.safe_model import SAFEModel


AVE_CATEGORIES = [
    "Church bell",
    "Male speech, man speaking",
    "Bark",
    "Fixed-wing aircraft, airplane",
    "Race car, auto racing",
    "Female speech, woman speaking",
    "Helicopter",
    "Violin, fiddle",
    "Flute",
    "Ukulele",
    "Frying (food)",
    "Truck",
    "Shofar",
    "Motorcycle",
    "Acoustic guitar",
    "Train horn",
    "Clock",
    "Banjo",
    "Goat",
    "Baby cry, infant cry",
    "Bus",
    "Chainsaw",
    "Cat",
    "Horse",
    "Toilet flush",
    "Rodents, rats, mice",
    "Accordion",
    "Mandolin",
]

AVE_LABEL_TO_IDX = {label: idx for idx, label in enumerate(AVE_CATEGORIES)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AVEDataset(Dataset):
    """
    AVE loader for audio classification.
    Expects annotations in the standard AVE format:
      category&video_id&quality&start&end
    Audio files are typically named:
      {video_id}_{start}_{end}.wav
    And stored under:
      train/audio/ and test/audio/
    """

    def __init__(
        self,
        data_path: str,
        split: str,
        sample_rate: int = 48000,
        max_length: float = 10.0,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.sample_rate = int(sample_rate)
        self.max_length = float(max_length)

        split_files = {"train": "trainSet.txt", "val": "valSet.txt", "test": "testSet.txt"}
        data_file = self.data_path / split_files.get(split, f"{split}Set.txt")
        if not data_file.exists():
            data_file = self.data_path / "ave" / split_files.get(split, f"{split}Set.txt")
        if not data_file.exists():
            raise FileNotFoundError(f"Could not find split file for '{split}': {data_file}")

        self.examples = self._load_data(data_file)

        # Quick path sanity check
        check_n = min(50, len(self.examples))
        found = 0
        for i in range(check_n):
            if self._resolve_audio_path(self.examples[i]["audio_name"]) is not None:
                found += 1
        print(f"[AVEDataset] {split}: {len(self.examples)} samples", flush=True)
        if check_n > 0:
            print(f"[AVEDataset] {split}: audio found for {found}/{check_n} sample check", flush=True)

    def _load_data(self, data_file: Path) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("&")
                if len(parts) < 2:
                    continue
                category = parts[0].strip()
                video_id = parts[1].strip()
                if category not in AVE_LABEL_TO_IDX:
                    continue
                start = parts[3].strip() if len(parts) > 3 else "0"
                end = parts[4].strip() if len(parts) > 4 else "10"
                audio_name = f"{video_id}_{start}_{end}.wav"
                examples.append(
                    {
                        "category": category,
                        "label": AVE_LABEL_TO_IDX[category],
                        "video_id": video_id,
                        "start": start,
                        "end": end,
                        "audio_name": audio_name,
                    }
                )
        return examples

    def _resolve_audio_path(self, audio_name: str) -> Optional[Path]:
        candidates = [
            self.data_path / "train" / "audio" / audio_name,
            self.data_path / "test" / "audio" / audio_name,
            self.data_path / "val" / "audio" / audio_name,
            self.data_path / "audio" / audio_name,
            self.data_path / audio_name,
            self.data_path / "AVE" / audio_name,
            self.data_path / "ave" / audio_name,
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        audio_path = self._resolve_audio_path(ex["audio_name"])
        return {
            "audio": str(audio_path) if audio_path is not None else None,
            "label": int(ex["label"]),
            "category": ex["category"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    audio: List[str] = []
    labels: List[int] = []
    categories: List[str] = []

    for item in batch:
        if not item.get("audio"):
            continue
        # SAFE/CLAP audio encoder supports raw file paths directly; do not wrap in tuples.
        audio.append(str(item["audio"]))
        labels.append(int(item["label"]))
        categories.append(str(item["category"]))

    if not audio:
        return {"audio": None, "labels": None, "categories": []}

    return {
        "audio": audio,
        "labels": torch.tensor(labels, dtype=torch.long),
        "categories": categories,
    }


class SAFELLMProbe(nn.Module):
    PROMPT = "What is happening in the audio?"

    def __init__(self, config: Dict[str, Any], num_classes: int):
        super().__init__()

        constructor_keys = {
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
        constructor_config = {k: v for k, v in config.items() if k in constructor_keys}
        self.safe_model = SAFEModel(**constructor_config)

        # Ensure only audio components train
        self.safe_model.enable_audio_training()
        hidden_size = int(config.get("llm_hidden_size", 5120))
        self.head = nn.Linear(hidden_size, num_classes)

        # Ensure the underlying HF model actually returns hidden states when requested.
        # Some configurations ignore call-time flags unless config is set.
        try:
            llm = self.safe_model.base_vl.llm
            for candidate in [llm, getattr(llm, "language_model", None), getattr(llm, "model", None)]:
                if candidate is None or not hasattr(candidate, "config"):
                    continue
                try:
                    candidate.config.output_hidden_states = True
                except Exception:
                    pass
                try:
                    candidate.config.return_dict = True
                except Exception:
                    pass
        except Exception:
            pass

    def freeze_safe(self, freeze: bool = True) -> None:
        """Freeze/unfreeze SAFE trainable components (projector + fusion)."""
        for p in self.safe_model.get_trainable_parameters():
            p.requires_grad = not freeze

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = list(self.safe_model.get_trainable_parameters())
        params.extend(list(self.head.parameters()))
        return params

    def get_safe_params(self) -> List[nn.Parameter]:
        return list(self.safe_model.get_trainable_parameters())

    def get_head_params(self) -> List[nn.Parameter]:
        return list(self.head.parameters())

    def forward(
        self,
        audio: List[str],
        device: torch.device,
        pooling: str = "last",
    ) -> torch.Tensor:
        batch_size = len(audio)
        inputs = self.safe_model.prepare_multimodal_inputs(
            text=[self.PROMPT] * batch_size,
            images=None,
            audio=audio,
            answers=None,
            device=device,
            training_mode=False,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        audio_tokens = inputs.get("audio_tokens")
        if audio_tokens is not None:
            audio_tokens = audio_tokens.to(device)
        audio_attention_mask = inputs.get("audio_attention_mask")
        if audio_attention_mask is not None:
            audio_attention_mask = audio_attention_mask.to(device)

        # Fallback: capture the final hidden states via a pre-hook on the output embedding
        # module (often `lm_head`). This is robust when HF wrappers ignore hidden-state flags.
        hidden_capture: Dict[str, Optional[torch.Tensor]] = {"last": None}
        hook_handle = None
        try:
            llm = self.safe_model.base_vl.llm
            head_module = None
            for candidate_owner in [
                llm,
                getattr(llm, "language_model", None),
                getattr(llm, "model", None),
            ]:
                if candidate_owner is None:
                    continue
                try:
                    get_out = getattr(candidate_owner, "get_output_embeddings", None)
                    if callable(get_out):
                        head_module = get_out()
                except Exception:
                    head_module = None
                if head_module is not None:
                    break
                head_module = getattr(candidate_owner, "lm_head", None)
                if head_module is not None:
                    break

            if head_module is not None:
                def _capture_head_input(_module, inputs):
                    try:
                        if inputs and torch.is_tensor(inputs[0]):
                            hidden_capture["last"] = inputs[0]
                    except Exception:
                        hidden_capture["last"] = None

                hook_handle = head_module.register_forward_pre_hook(_capture_head_input)
        except Exception:
            hook_handle = None

        outputs = self.safe_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_tokens=audio_tokens,
            audio_attention_mask=audio_attention_mask,
            labels=None,
            output_hidden_states=True,
            return_dict=True,
        )
        if hook_handle is not None:
            try:
                hook_handle.remove()
            except Exception:
                pass

        hidden = outputs.get("hidden_states")
        if hidden is None:
            hidden = hidden_capture.get("last")
        if hidden is None:
            if not hasattr(self, "_hidden_debug_logged"):
                self._hidden_debug_logged = True
                try:
                    llm = self.safe_model.base_vl.llm
                    cfg = getattr(llm, "config", None)
                    cfg_hs = getattr(cfg, "output_hidden_states", None) if cfg is not None else None
                    cfg_rd = getattr(cfg, "return_dict", None) if cfg is not None else None
                except Exception:
                    cfg_hs = None
                    cfg_rd = None
                print(
                    "[LLMProbeDebug] hidden_states missing. "
                    f"input_ids={tuple(input_ids.shape)} "
                    f"attn_sum={int(attention_mask.sum().item()) if attention_mask is not None else 'None'} "
                    f"audio_tokens={tuple(audio_tokens.shape) if audio_tokens is not None else None} "
                    f"audio_numel={int(audio_tokens.numel()) if audio_tokens is not None else 0} "
                    f"enable_midlayer_fusion={getattr(self.safe_model, 'enable_midlayer_fusion', None)} "
                    f"llm.config.output_hidden_states={cfg_hs} "
                    f"llm.config.return_dict={cfg_rd}",
                    flush=True,
                )
            raise RuntimeError("SAFEModel did not return hidden_states; expected last hidden state tensor.")

        if pooling == "mean":
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.to(hidden.dtype).unsqueeze(-1)  # (B, T, 1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            if attention_mask is None:
                pooled = hidden[:, -1, :]
            else:
                lengths = attention_mask.long().sum(dim=1).clamp_min(1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]

        # Lightweight sanity checks (printed once)
        if not hasattr(self, "_probe_stats_logged"):
            self._probe_stats_logged = True
            with torch.no_grad():
                pooled_var = float(pooled.float().var(dim=0).mean().item())
                pooled_norm = float(pooled.float().norm(dim=-1).mean().item())
            attn_summary = None
            try:
                attn_summary = self.safe_model.get_last_attention_summary()
            except Exception:
                attn_summary = None
            print(
                f"[LLMProbe] pooled_norm={pooled_norm:.3f} pooled_var={pooled_var:.6e} "
                f"attn_summary={'yes' if attn_summary is not None else 'no'}",
                flush=True,
            )

        logits = self.head(pooled.float())
        return logits


def _split_decay(params: List[nn.Parameter]) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for p in params:
        if not p.requires_grad:
            continue
        if p.ndim == 1:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def build_optimizer(
    model: SAFELLMProbe,
    safe_lr: float,
    head_lr: float,
    safe_weight_decay: float,
    head_weight_decay: float,
) -> torch.optim.Optimizer:
    safe_decay, safe_no_decay = _split_decay(model.get_safe_params())
    head_decay, head_no_decay = _split_decay(model.get_head_params())

    param_groups = [
        {"name": "safe_decay", "params": safe_decay, "lr": safe_lr, "weight_decay": safe_weight_decay},
        {"name": "safe_no_decay", "params": safe_no_decay, "lr": safe_lr, "weight_decay": 0.0},
        {"name": "head_decay", "params": head_decay, "lr": head_lr, "weight_decay": head_weight_decay},
        {"name": "head_no_decay", "params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
    ]
    param_groups = [g for g in param_groups if g["params"]]

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def load_safe_checkpoint_into(model: SAFELLMProbe, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Try loading directly into SAFEModel inside the probe.
    adapted: Dict[str, torch.Tensor] = {}
    skipped = 0
    safe_sd = model.safe_model.state_dict()
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[len("module.") :]
        if key in safe_sd:
            adapted[key] = v
        else:
            skipped += 1

    missing, unexpected = model.safe_model.load_state_dict(adapted, strict=False)
    print(f"[Checkpoint] Loaded {len(adapted)} tensors into SAFEModel", flush=True)
    if skipped:
        print(f"[Checkpoint] Skipped {skipped} tensors (not in SAFEModel)", flush=True)
    if missing:
        print(f"[Checkpoint] Missing {len(missing)} keys (first 5): {missing[:5]}", flush=True)
    if unexpected:
        print(f"[Checkpoint] Unexpected {len(unexpected)} keys (first 5): {unexpected[:5]}", flush=True)


def train_epoch(
    model: SAFELLMProbe,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> Tuple[Dict[str, float], int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    num_batches = 0
    start = time.time()

    for batch_idx, batch in enumerate(loader):
        audio = batch["audio"]
        labels = batch["labels"]
        if audio is None or labels is None or labels.numel() == 0:
            continue

        # Warmups
        if args.force_gate is not None:
            try:
                model.safe_model.set_gate(float(args.force_gate))
            except Exception:
                pass
        elif args.gate_warmup_steps > 0:
            warmup_steps = max(1, int(args.gate_warmup_steps))
            progress = min(1.0, float(global_step) / float(warmup_steps))
            gate = float(args.gate_warmup_start) + (1.0 - float(args.gate_warmup_start)) * progress
            try:
                model.safe_model.set_gate(gate)
            except Exception:
                pass

        if args.scale_min_warmup_epochs > 0:
            try:
                model.safe_model.set_scale_minimum_warmup(
                    epoch=epoch - 1,
                    warmup_epochs=int(args.scale_min_warmup_epochs),
                    start_min=float(args.scale_min_start),
                    end_min=float(args.scale_min_end),
                )
            except Exception:
                pass

        if args.residual_warmup_epochs > 0 and (epoch - 1) < args.residual_warmup_epochs:
            try:
                model.safe_model.set_residual_scale_warmup(
                    epoch=epoch - 1,
                    warmup_epochs=int(args.residual_warmup_epochs),
                    start_scale=float(args.residual_warmup_start),
                    end_scale=1.0,
                )
            except Exception:
                pass

        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Optional head-only warmup: freeze SAFE LR for early steps, then restore.
        if getattr(args, "head_warmup_steps", 0):
            warmup_steps = int(args.head_warmup_steps)
            safe_lr = float(args.safe_learning_rate)
            effective_safe_lr = 0.0 if global_step < warmup_steps else safe_lr
            for group in optimizer.param_groups:
                if str(group.get("name", "")).startswith("safe_"):
                    group["lr"] = effective_safe_lr
            # Log once when SAFE LR turns on (crosses warmup boundary)
            if (
                effective_safe_lr > 0.0
                and not hasattr(model, "_safe_lr_on_logged")
            ):
                model._safe_lr_on_logged = True
                lr_by_group = {g.get("name", f"g{idx}"): g.get("lr") for idx, g in enumerate(optimizer.param_groups)}
                print(f"  [LR] SAFE warmup complete at step={global_step}. lrs={lr_by_group}", flush=True)

        with autocast(enabled=args.fp16):
            logits = model(audio=audio, device=device, pooling=args.pooling)
            loss = F.cross_entropy(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), args.max_grad_norm)
            optimizer.step()

        preds = logits.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total_seen += int(labels.numel())
        total_loss += float(loss.item()) * int(labels.numel())
        num_batches += 1
        global_step += 1

        if (batch_idx + 1) % args.log_interval == 0:
            elapsed = max(1e-6, time.time() - start)
            sps = (total_seen / elapsed)
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {total_loss / max(total_seen, 1):.4f} | "
                f"Acc: {total_correct / max(total_seen, 1):.4f} | {sps:.1f} samples/s",
                flush=True,
            )

            # Log gradients/LRs on first log interval, and once right after SAFE LR turns on.
            should_log_grads = not hasattr(model, "_grad_logged")
            if getattr(args, "head_warmup_steps", 0) and hasattr(model, "_safe_lr_on_logged"):
                should_log_grads = should_log_grads or not hasattr(model, "_grad_after_warmup_logged")
            if should_log_grads:
                if hasattr(model, "_grad_logged"):
                    model._grad_after_warmup_logged = True
                else:
                    model._grad_logged = True
                with torch.no_grad():
                    head_g = 0.0
                    head_n = 0
                    for p in model.get_head_params():
                        if p.grad is not None:
                            head_g += float(p.grad.float().norm().item()) ** 2
                            head_n += 1
                    head_g = head_g ** 0.5 if head_n else 0.0

                    safe_g = 0.0
                    safe_n = 0
                    for p in model.get_safe_params():
                        if p.grad is not None:
                            safe_g += float(p.grad.float().norm().item()) ** 2
                            safe_n += 1
                    safe_g = safe_g ** 0.5 if safe_n else 0.0

                lr_by_group = {g.get("name", f"g{idx}"): g.get("lr") for idx, g in enumerate(optimizer.param_groups)}
                print(f"  [Gradients] head={head_g:.4f} ({head_n}) safe={safe_g:.4f} ({safe_n}) lrs={lr_by_group}", flush=True)

            if wandb is not None and args.wandb:
                wandb.log(
                    {
                        "train/loss": total_loss / max(total_seen, 1),
                        "train/acc": total_correct / max(total_seen, 1),
                        "train/gate": getattr(model.safe_model, "_default_gate", 1.0),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

    return {"loss": total_loss / max(total_seen, 1), "acc": total_correct / max(total_seen, 1)}, global_step


@torch.no_grad()
def evaluate(
    model: SAFELLMProbe,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    if args.force_gate is not None:
        model.safe_model.set_gate(float(args.force_gate))
    else:
        model.safe_model.set_gate(1.0)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch in loader:
        audio = batch["audio"]
        labels = batch["labels"]
        if audio is None or labels is None or labels.numel() == 0:
            continue
        labels = labels.to(device)
        with autocast(enabled=args.fp16):
            logits = model(audio=audio, device=device, pooling=args.pooling)
            loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total_seen += int(labels.numel())
        total_loss += float(loss.item()) * int(labels.numel())

    return {"loss": total_loss / max(total_seen, 1), "acc": total_correct / max(total_seen, 1)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAFE LLM-probe audio classification (AVE)")
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs/ave_llm_probe")
    p.add_argument("--model-config", type=str, default="phase1")
    p.add_argument("--fusion-layer-indices", type=str, default="1", help="Comma-separated fusion layers (default: 1)")
    p.add_argument(
        "--fusion-injection-point",
        type=str,
        default=None,
        choices=["pre_ffn", "post_layer"],
        help="Override fusion injection point for hooks (debugging).",
    )
    p.add_argument(
        "--fusion-mode",
        type=str,
        default=None,
        choices=["residual", "film"],
        help="Fusion update type (bottleneck only).",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--safe-learning-rate", type=float, default=6e-5, help="LR for projector+fusion")
    p.add_argument("--head-learning-rate", type=float, default=1e-3, help="LR for linear probe head")
    p.add_argument("--safe-weight-decay", type=float, default=0.01)
    p.add_argument("--head-weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument(
        "--force-gate",
        type=float,
        default=None,
        help="If set, override SAFE fusion gate to this constant (disables gate warmup).",
    )

    # Warmups (enabled by default)
    p.add_argument("--gate-warmup-steps", type=int, default=500)
    p.add_argument("--gate-warmup-start", type=float, default=0.1)
    p.add_argument("--scale-min-warmup-epochs", type=int, default=5)
    p.add_argument("--scale-min-start", type=float, default=0.5)
    p.add_argument("--scale-min-end", type=float, default=1.0)
    p.add_argument("--residual-warmup-epochs", type=int, default=5)
    p.add_argument("--residual-warmup-start", type=float, default=0.1)

    # Logging
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="SAFE")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--load-checkpoint", type=str, default=None, help="Load a SAFE checkpoint (e.g., from train_safe.py)")
    p.add_argument(
        "--head-only",
        action="store_true",
        help="Freeze projector+fusion and train only the linear head.",
    )
    p.add_argument(
        "--head-warmup-steps",
        type=int,
        default=0,
        help="If >0, train head-only for this many optimizer steps (SAFE LR=0), then unfreeze SAFE LR.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_config(args.model_config)
    if args.fusion_layer_indices:
        layers = [int(x.strip()) for x in str(args.fusion_layer_indices).split(",") if x.strip()]
        config["fusion_layer_indices"] = layers
    if args.fusion_injection_point is not None:
        config.setdefault("fusion_config", {})
        config["fusion_config"]["injection_point"] = str(args.fusion_injection_point)
        print(f"[Config] fusion_injection_point={args.fusion_injection_point}", flush=True)
    if args.fusion_mode is not None:
        config.setdefault("fusion_config", {})
        config["fusion_config"]["fusion_mode"] = str(args.fusion_mode)
        print(f"[Config] fusion_mode={args.fusion_mode}", flush=True)

    train_ds = AVEDataset(args.data_path, split="train")
    test_ds = AVEDataset(args.data_path, split="test")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = SAFELLMProbe(config=config, num_classes=len(AVE_CATEGORIES)).to(device)

    if args.load_checkpoint:
        print(f"[Checkpoint] Loading: {args.load_checkpoint}", flush=True)
        load_safe_checkpoint_into(model, args.load_checkpoint, device=device)

    if args.head_only:
        print("[Mode] head-only: freezing SAFE trainables (projector+fusion)", flush=True)
        model.freeze_safe(True)

    optimizer = build_optimizer(
        model=model,
        safe_lr=float(args.safe_learning_rate),
        head_lr=float(args.head_learning_rate),
        safe_weight_decay=float(args.safe_weight_decay),
        head_weight_decay=float(args.head_weight_decay),
    )
    scaler = GradScaler() if args.fp16 else None

    if args.wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"llm-probe-ave-layer{args.fusion_layer_indices}",
            config=vars(args),
        )

    best = -1.0
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}\n" + "-" * 40, flush=True)
        train_metrics, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            epoch=epoch,
            global_step=global_step,
        )
        val_metrics = evaluate(model=model, loader=test_loader, device=device, args=args)
        print(f"Train | loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f}", flush=True)
        print(f"Test  | loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f}", flush=True)

        if args.wandb and wandb is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "val/loss": val_metrics["loss"],
                    "val/acc": val_metrics["acc"],
                },
                step=global_step,
            )

        if val_metrics["acc"] > best:
            best = val_metrics["acc"]
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best,
                "config": config,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> New best acc: {best:.4f}", flush=True)

    if args.wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
