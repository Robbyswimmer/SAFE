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
    audio: List[Tuple[str, int]] = []
    labels: List[int] = []
    categories: List[str] = []

    for item in batch:
        if not item.get("audio"):
            continue
        audio.append((item["audio"], 48000))
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

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = list(self.safe_model.get_trainable_parameters())
        params.extend(list(self.head.parameters()))
        return params

    def forward(
        self,
        audio: List[Tuple[str, int]],
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

        outputs = self.safe_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_tokens=audio_tokens,
            audio_attention_mask=audio_attention_mask,
            labels=None,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.get("hidden_states")
        if hidden is None:
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

        logits = self.head(pooled.float())
        return logits


def build_optimizer(params: List[nn.Parameter], lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for p in params:
        if not p.requires_grad:
            continue
        if p.ndim == 1:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.999),
    )


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
        if args.gate_warmup_steps > 0:
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
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=6e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=10)

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
    optimizer = build_optimizer(model.get_trainable_params(), lr=args.learning_rate, weight_decay=args.weight_decay)
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

