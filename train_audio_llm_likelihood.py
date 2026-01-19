#!/usr/bin/env python3
"""
train_audio_llm_likelihood.py

Closed-set likelihood classification for AVE with a frozen LLM:
  audio -> CLAP (frozen) -> projector (trainable) -> fusion (trainable) -> frozen LLM

For each class c, define a fixed answer string (template + label). Score each class by
mean negative log-likelihood of the answer tokens under the frozen LLM conditioned on audio.

Training uses sampled negatives for efficiency; evaluation can score all classes.
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
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
    def __init__(self, data_path: str, split: str):
        self.data_path = Path(data_path)
        self.split = split

        split_files = {"train": "trainSet.txt", "val": "valSet.txt", "test": "testSet.txt"}
        data_file = self.data_path / split_files.get(split, f"{split}Set.txt")
        if not data_file.exists():
            data_file = self.data_path / "ave" / split_files.get(split, f"{split}Set.txt")
        if not data_file.exists():
            raise FileNotFoundError(f"Could not find split file: {data_file}")

        self.examples = self._load_data(data_file)

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
                        "audio_name": audio_name,
                        "label": AVE_LABEL_TO_IDX[category],
                        "category": category,
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
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    audio: List[str] = []
    labels: List[int] = []
    for item in batch:
        if not item.get("audio"):
            continue
        audio.append(str(item["audio"]))
        labels.append(int(item["label"]))
    if not audio:
        return {"audio": None, "labels": None}
    return {"audio": audio, "labels": torch.tensor(labels, dtype=torch.long)}


class SAFEClosedSetLikelihood(torch.nn.Module):
    PROMPT = "What is happening in the audio?"

    def __init__(self, config: Dict[str, Any], template: str):
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
        self.safe_model.enable_audio_training()
        self.template = template

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        return list(self.safe_model.get_trainable_parameters())

    def _answers_from_class_indices(self, class_indices: Sequence[int]) -> List[str]:
        answers: List[str] = []
        for idx in class_indices:
            label = AVE_CATEGORIES[int(idx)]
            answers.append(self.template.format(label=label))
        return answers

    def score_candidates(
        self,
        audio: List[str],
        candidate_class_indices: torch.Tensor,  # (B, K)
        device: torch.device,
        force_gate: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Return score logits for candidates: higher is better.
        """
        batch_size, num_cand = candidate_class_indices.shape

        # Encode prompt + audio once per batch (avoid re-encoding audio per candidate).
        base = self.safe_model.prepare_multimodal_inputs(
            text=[self.PROMPT] * batch_size,
            images=None,
            audio=audio,
            answers=None,
            device=device,
            training_mode=False,
        )

        audio_tokens = base.get("audio_tokens")
        audio_attention_mask = base.get("audio_attention_mask")
        if audio_tokens is not None:
            audio_tokens = audio_tokens.to(device)
        if audio_attention_mask is not None:
            audio_attention_mask = audio_attention_mask.to(device)

        # Candidate scores
        scores = torch.empty((batch_size, num_cand), device=device, dtype=torch.float32)

        for j in range(num_cand):
            cand = candidate_class_indices[:, j].tolist()
            answers = self._answers_from_class_indices(cand)

            # Clone prompt-only tokenization and apply answers (tokenization only).
            inputs = {
                "input_ids": base["input_ids"].clone(),
                "attention_mask": base["attention_mask"].clone(),
            }
            self.safe_model._apply_answers_to_inputs(inputs, answers=answers, device=device)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = inputs["labels"].to(device)

            outputs = self.safe_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_tokens=audio_tokens,
                audio_attention_mask=audio_attention_mask,
                labels=labels,
                gate=force_gate,
                return_dict=True,
            )
            logits = outputs.get("logits")
            if logits is None:
                raise RuntimeError("SAFEModel did not return logits")

            # Per-example mean NLL over supervised (answer) tokens.
            shift_logits = logits[..., :-1, :].float()
            shift_labels = labels[..., 1:]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_lp = log_probs.gather(dim=-1, index=shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            mask = shift_labels != -100
            token_lp = token_lp * mask.to(token_lp.dtype)
            denom = mask.sum(dim=-1).clamp_min(1)
            mean_nll = -(token_lp.sum(dim=-1) / denom.to(token_lp.dtype))
            scores[:, j] = -mean_nll

        return scores


def _sample_negatives(labels: torch.Tensor, num_classes: int, num_neg: int) -> torch.Tensor:
    bsz = labels.size(0)
    out = torch.empty((bsz, 1 + num_neg), dtype=torch.long)
    out[:, 0] = labels.cpu()
    for i in range(bsz):
        true = int(labels[i].item())
        choices = list(range(num_classes))
        choices.remove(true)
        negs = random.sample(choices, k=min(num_neg, len(choices)))
        if len(negs) < num_neg:
            negs = (negs * (num_neg // max(len(negs), 1) + 1))[:num_neg]
        out[i, 1:] = torch.tensor(negs[:num_neg], dtype=torch.long)
    return out


def _build_optimizer(params: List[torch.nn.Parameter], lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for p in params:
        if not p.requires_grad:
            continue
        (no_decay if p.ndim == 1 else decay).append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))


def train_epoch(
    model: SAFEClosedSetLikelihood,
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
    start = time.time()

    for batch_idx, batch in enumerate(loader):
        audio = batch["audio"]
        labels = batch["labels"]
        if audio is None or labels is None or labels.numel() == 0:
            continue
        labels = labels.to(device)

        # Gate schedule
        force_gate = None
        if args.force_gate is not None:
            force_gate = float(args.force_gate)
        elif args.gate_warmup_steps > 0:
            warm = max(1, int(args.gate_warmup_steps))
            progress = min(1.0, float(global_step) / float(warm))
            force_gate = float(args.gate_warmup_start) + (1.0 - float(args.gate_warmup_start)) * progress

        candidates = _sample_negatives(labels.detach().cpu(), num_classes=len(AVE_CATEGORIES), num_neg=args.num_negatives)
        candidates = candidates.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.fp16):
            scores = model.score_candidates(audio=audio, candidate_class_indices=candidates, device=device, force_gate=force_gate)
            target = torch.zeros(scores.size(0), dtype=torch.long, device=device)
            loss = F.cross_entropy(scores, target)

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

        preds = scores.argmax(dim=-1)
        total_correct += int((preds == 0).sum().item())
        total_seen += int(labels.numel())
        total_loss += float(loss.item()) * int(labels.numel())
        global_step += 1

        if (batch_idx + 1) % args.log_interval == 0:
            elapsed = max(1e-6, time.time() - start)
            sps = total_seen / elapsed
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {total_loss / max(total_seen, 1):.4f} | "
                f"Acc@K: {total_correct / max(total_seen, 1):.4f} | {sps:.1f} samples/s",
                flush=True,
            )
            if wandb is not None and args.wandb:
                wandb.log(
                    {
                        "train/loss": total_loss / max(total_seen, 1),
                        "train/acc_at_k": total_correct / max(total_seen, 1),
                        "epoch": epoch,
                        "train/gate": float(force_gate) if force_gate is not None else None,
                    },
                    step=global_step,
                )

    return {"loss": total_loss / max(total_seen, 1), "acc_at_k": total_correct / max(total_seen, 1)}, global_step


@torch.no_grad()
def evaluate(
    model: SAFEClosedSetLikelihood,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    total_correct = 0
    total_seen = 0
    total_loss = 0.0

    for batch in loader:
        audio = batch["audio"]
        labels = batch["labels"]
        if audio is None or labels is None or labels.numel() == 0:
            continue
        labels = labels.to(device)

        # Score all classes
        bsz = labels.size(0)
        all_candidates = torch.arange(len(AVE_CATEGORIES), device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

        with autocast(enabled=args.fp16):
            scores = model.score_candidates(
                audio=audio,
                candidate_class_indices=all_candidates,
                device=device,
                force_gate=float(args.force_gate) if args.force_gate is not None else 1.0,
            )
            preds = scores.argmax(dim=-1)
            total_correct += int((preds == labels).sum().item())
            total_seen += int(labels.numel())

            # Also compute a proper CE loss over the 28-way scores for monitoring
            total_loss += float(F.cross_entropy(scores, labels, reduction="sum").item())

    return {"loss": total_loss / max(total_seen, 1), "acc": total_correct / max(total_seen, 1)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAFE closed-set likelihood classification (AVE)")
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs/ave_llm_likelihood")
    p.add_argument("--model-config", type=str, default="phase1")
    p.add_argument("--fusion-layer-indices", type=str, default="12", help="Comma-separated fusion layers")
    p.add_argument("--fusion-injection-point", type=str, default=None, choices=["pre_ffn", "post_layer"])
    p.add_argument("--use-bottleneck", dest="use_bottleneck", action="store_true", help="Force bottleneck fusion adapters")
    p.add_argument("--no-bottleneck", dest="use_bottleneck", action="store_false", help="Force non-bottleneck (LoRA) fusion adapters")
    p.set_defaults(use_bottleneck=None)
    p.add_argument("--bottleneck-dim", type=int, default=None, help="Override fusion bottleneck_dim (when using bottleneck)")
    p.add_argument("--lora-rank", type=int, default=None, help="Override fusion LoRA rank (when not bottleneck)")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size (lower is safer; scoring is heavier).")
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--template", type=str, default="The sound is: {label}.")
    p.add_argument("--num-negatives", type=int, default=7, help="Negatives per sample during training.")

    p.add_argument("--force-gate", type=float, default=None, help="Constant gate for training/eval (None = warmup/train, 1.0 eval).")
    p.add_argument("--gate-warmup-steps", type=int, default=500)
    p.add_argument("--gate-warmup-start", type=float, default=0.1)

    p.add_argument("--load-checkpoint", type=str, default=None, help="Load a SAFE checkpoint (e.g., captioning).")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="SAFE")
    p.add_argument("--wandb-run-name", type=str, default=None)
    return p.parse_args()


def _load_safe_checkpoint_into(model: SAFEClosedSetLikelihood, checkpoint_path: str, device: torch.device) -> None:
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

    def _normalize_key(raw_key: str) -> List[str]:
        """
        Generate candidate keys by stripping common wrapper prefixes.
        We try multiple variants because different trainers save different roots.
        """
        key = str(raw_key)
        candidates = [key]
        prefixes = [
            "module.",
            "model.",
            "base_model.",
            "safe_model.",
            "model.safe_model.",
            "module.safe_model.",
            "module.model.",
            "module.base_model.",
        ]
        for _ in range(3):
            expanded: List[str] = []
            for cand in candidates:
                expanded.append(cand)
                for p in prefixes:
                    if cand.startswith(p):
                        expanded.append(cand[len(p) :])
            # de-dup but keep order
            seen = set()
            candidates = [c for c in expanded if not (c in seen or seen.add(c))]
        return candidates

    adapted: Dict[str, torch.Tensor] = {}
    safe_sd = model.safe_model.state_dict()
    shape_mismatch = 0
    matched = 0
    tried = 0
    for k, v in state_dict.items():
        tried += 1
        for cand in _normalize_key(k):
            target = safe_sd.get(cand)
            if target is None:
                continue
            if hasattr(target, "shape") and hasattr(v, "shape") and tuple(target.shape) != tuple(v.shape):
                shape_mismatch += 1
                continue
            adapted[cand] = v
            matched += 1
            break

    missing, unexpected = model.safe_model.load_state_dict(adapted, strict=False)

    # Summarize loaded components
    counts: Dict[str, int] = {"audio_projector": 0, "fusion_adapter": 0, "audio_token_embeddings": 0, "other": 0}
    fusion_layer_counts: Dict[str, int] = {}
    for key in adapted.keys():
        root = key.split(".", 1)[0]
        if root in counts:
            counts[root] += 1
        else:
            counts["other"] += 1
        if key.startswith("fusion_adapter.fusion_adapters."):
            # Example key: fusion_adapter.fusion_adapters.audio:24.cross_attention....
            parts = key.split(".")
            if len(parts) > 2:
                adapter_key = parts[2]
                fusion_layer_counts[adapter_key] = fusion_layer_counts.get(adapter_key, 0) + 1

    print(
        f"[Checkpoint] Loaded {len(adapted)} tensors into SAFEModel "
        f"(matched {matched}/{tried}, skipped {shape_mismatch} shape mismatches).",
        flush=True,
    )
    print(f"[Checkpoint] Loaded counts: {counts}", flush=True)
    if fusion_layer_counts:
        top = sorted(fusion_layer_counts.items(), key=lambda x: (-x[1], x[0]))
        preview = ", ".join([f\"{k}={v}\" for k, v in top[:8]])
        print(f\"[Checkpoint] Loaded fusion adapters: {preview}\", flush=True)
    if missing:
        print(f"[Checkpoint] Missing {len(missing)} keys (first 5): {missing[:5]}", flush=True)
    if unexpected:
        print(f"[Checkpoint] Unexpected {len(unexpected)} keys (first 5): {unexpected[:5]}", flush=True)


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
    if args.use_bottleneck is not None:
        config.setdefault("fusion_config", {})
        config["fusion_config"]["use_bottleneck"] = bool(args.use_bottleneck)
        print(f"[Config] use_bottleneck={args.use_bottleneck}", flush=True)
    if args.bottleneck_dim is not None:
        config.setdefault("fusion_config", {})
        config["fusion_config"]["bottleneck_dim"] = int(args.bottleneck_dim)
        print(f"[Config] bottleneck_dim={args.bottleneck_dim}", flush=True)
    if args.lora_rank is not None:
        config["lora_rank"] = int(args.lora_rank)
        print(f"[Config] lora_rank={args.lora_rank}", flush=True)

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

    model = SAFEClosedSetLikelihood(config=config, template=args.template).to(device)
    if args.load_checkpoint:
        print(f"[Checkpoint] Loading: {args.load_checkpoint}", flush=True)
        _load_safe_checkpoint_into(model, args.load_checkpoint, device=device)

    optimizer = _build_optimizer(model.get_trainable_params(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler() if args.fp16 else None

    if args.wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"ave-llm-likelihood-{args.fusion_layer_indices}",
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
        print(f"Train | loss={train_metrics['loss']:.4f} acc@K={train_metrics['acc_at_k']:.4f}", flush=True)
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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.safe_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best,
                    "config": config,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, "best_model.pt"),
            )
            print(f"  -> New best acc: {best:.4f}", flush=True)

    if args.wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
