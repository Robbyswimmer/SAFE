#!/usr/bin/env python3
"""
Train/evaluate SAFE on SQA3D with AVQA-style modality ablations.

This mirrors the AVQA experiment structure at a practical level:
- optional interleaved unimodal training phases per epoch
- per-epoch evaluation on text / image / pointcloud / both
- frozen VLM + trainable point-cloud projector/fusion path
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import time

import numpy as np
import torch
from contextlib import nullcontext
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.pointcloud_configs import get_pointcloud_config
from safe.data.sqa3d_dataset import SQA3DDataset, collate_sqa3d_batch
from train_scanqa_composition import (
    ScanQACompositionModel,
    compute_answer_ce_loss,
    prepare_qa_inputs,
    select_training_answers,
)

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
    for prefix in ("the answer is", "answer:", "a:", "it is", "this is"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.strip().rstrip(".")
    return " ".join(text.split())


def token_f1(pred: str, ref: str) -> float:
    p = normalize_answer(pred).split()
    r = normalize_answer(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    counts: Dict[str, int] = defaultdict(int)
    common = 0
    for tok in r:
        counts[tok] += 1
    for tok in p:
        if counts[tok] > 0:
            counts[tok] -= 1
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(r)
    return 2.0 * precision * recall / max(precision + recall, 1e-8)


def compute_sqa3d_metrics(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    question_types: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    exact_total = 0.0
    norm_total = 0.0
    f1_total = 0.0
    by_type: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"exact_match": 0.0, "accuracy": 0.0, "token_f1": 0.0, "n": 0.0}
    )

    for idx, (pred, refs) in enumerate(zip(predictions, references)):
        refs = [str(r) for r in refs if str(r).strip()]
        pred_raw = (pred or "").strip()
        pred_norm = normalize_answer(pred_raw)

        exact = any(pred_raw == ref for ref in refs)
        accuracy = any(pred_norm == normalize_answer(ref) for ref in refs)
        best_f1 = max((token_f1(pred_raw, ref) for ref in refs), default=0.0)

        exact_total += float(exact)
        norm_total += float(accuracy)
        f1_total += best_f1

        qtype = str(question_types[idx]) if question_types is not None and idx < len(question_types) else "unknown"
        row = by_type[qtype]
        row["exact_match"] += float(exact)
        row["accuracy"] += float(accuracy)
        row["token_f1"] += best_f1
        row["n"] += 1.0

    n = max(1, len(predictions))
    by_type_out = {}
    for qtype, row in by_type.items():
        denom = max(1.0, row["n"])
        by_type_out[qtype] = {
            "exact_match": 100.0 * row["exact_match"] / denom,
            "accuracy": 100.0 * row["accuracy"] / denom,
            "token_f1": 100.0 * row["token_f1"] / denom,
            "n": int(row["n"]),
        }

    return {
        "exact_match": 100.0 * exact_total / n,
        "accuracy": 100.0 * norm_total / n,
        "token_f1": 100.0 * f1_total / n,
        "num_samples": len(predictions),
        "by_type": by_type_out,
    }


def build_model_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = get_pointcloud_config(args.model_config)
    if args.llm_model:
        config["llm_model_name"] = args.llm_model
    if args.num_points is not None:
        config["num_points"] = int(args.num_points)
        pc_cfg = dict(config.get("pointcloud_encoder_config", {}))
        pc_cfg["num_points"] = int(args.num_points)
        config["pointcloud_encoder_config"] = pc_cfg
    if args.num_pointcloud_tokens is not None:
        config["num_tokens"] = int(args.num_pointcloud_tokens)
    if args.fusion_layer_indices:
        config["fusion_layer_indices"] = [int(x) for x in args.fusion_layer_indices.split(",") if x.strip()]
    if args.label_smoothing is not None:
        config["label_smoothing"] = float(args.label_smoothing)
    return config


def resolve_eval_modalities(args: argparse.Namespace, model: ScanQACompositionModel) -> List[str]:
    supported = model.supported_eval_modalities()
    if str(args.eval_modalities).strip().lower() == "auto":
        return supported
    selected: List[str] = []
    for item in str(args.eval_modalities).split(","):
        key = item.strip().lower()
        if key and key in supported and key not in selected:
            selected.append(key)
    return selected or supported


def resolve_modality_batch(batch: Dict[str, Any], modality: str) -> Dict[str, Any]:
    modality = str(modality).lower()
    if modality == "text":
        return {"pointclouds": None, "images": None}
    if modality == "image":
        return {"pointclouds": None, "images": batch.get("images")}
    if modality == "pointcloud":
        return {"pointclouds": batch.get("pointclouds"), "images": None}
    return {"pointclouds": batch.get("pointclouds"), "images": batch.get("images")}


def build_optimizer(model: ScanQACompositionModel, args: argparse.Namespace) -> AdamW:
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() <= 1 or "norm" in name.lower() or "bias" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.safe_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def build_scheduler(
    optimizer: AdamW,
    total_optimizer_steps: int,
    args: argparse.Namespace,
) -> Optional[LambdaLR]:
    if args.lr_scheduler == "none":
        return None

    warmup_steps = int(args.warmup_steps)
    if warmup_steps <= 0:
        warmup_steps = int(args.warmup_ratio * total_optimizer_steps)
    warmup_steps = max(0, min(warmup_steps, total_optimizer_steps))
    min_ratio = float(args.min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if total_optimizer_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_optimizer_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if args.lr_scheduler == "linear":
            return max(min_ratio, 1.0 - (1.0 - min_ratio) * progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_sqa3d_batch,
        pin_memory=True,
    )


def trim_dataset(dataset, max_samples: int):
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, list(range(max_samples)))


def train_epoch(
    model: ScanQACompositionModel,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: Optional[LambdaLR],
    scaler: GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    tokenizer,
    train_modality: str,
    global_step: int,
    wandb_run: Any = None,
) -> Tuple[float, int]:
    model.train()
    optimizer.zero_grad()

    def _amp_context():
        if args.fp16 and torch.cuda.is_available():
            return autocast(enabled=True, dtype=torch.float16)
        return nullcontext()

    total_loss = 0.0
    total_batches = 0
    pending_accum_steps = 0
    trainable_for_clip = model.get_trainable_params()
    num_batches = len(dataloader)
    log_every = max(1, int(args.log_every))

    for step, batch in enumerate(dataloader):
        if not batch:
            continue

        train_answers = select_training_answers(batch, strategy=args.train_answer_mode)
        train_batch = dict(batch)
        train_batch["answers"] = train_answers
        qa_inputs = prepare_qa_inputs(train_batch, tokenizer, device, max_length=args.max_seq_length)

        gate_value = float(args.fusion_gate)
        if args.gate_warmup_steps > 0:
            progress = min(1.0, float(global_step + 1) / float(max(1, args.gate_warmup_steps)))
            gate_value = float(args.fusion_gate) * progress

        mm = resolve_modality_batch(batch, train_modality)
        kwargs = {
            "input_ids": qa_inputs["input_ids"],
            "attention_mask": qa_inputs["attention_mask"],
            "labels": qa_inputs["labels"],
            "gate": gate_value,
        }
        if mm["pointclouds"] is not None:
            kwargs["pointclouds"] = mm["pointclouds"].to(device)
        if mm["images"] is not None:
            kwargs["images"] = mm["images"]

        with _amp_context():
            outputs = model(**kwargs)
            loss = outputs.get("loss")
            logits = outputs.get("logits")
            manual_loss = compute_answer_ce_loss(
                logits,
                kwargs["labels"],
                label_smoothing=float(args.label_smoothing),
            )
            if manual_loss is not None:
                loss = manual_loss
            if loss is None:
                continue

            loss = loss / int(args.gradient_accumulation_steps)
        scaler.scale(loss).backward()
        pending_accum_steps += 1

        if pending_accum_steps % int(args.gradient_accumulation_steps) == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_for_clip, float(args.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        step_loss = loss.item() * int(args.gradient_accumulation_steps)
        total_loss += step_loss
        total_batches += 1

        if (step + 1) % log_every == 0 or step == 0:
            avg_loss = total_loss / max(1, total_batches)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [train:{train_modality}] step {step + 1}/{num_batches} "
                f"loss={avg_loss:.4f} lr={current_lr:.2e} gate={gate_value:.3f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"train/{train_modality}/loss": avg_loss,
                        f"train/{train_modality}/lr": current_lr,
                        f"train/{train_modality}/gate": gate_value,
                    },
                    step=global_step,
                )

    if pending_accum_steps > 0 and pending_accum_steps % int(args.gradient_accumulation_steps) != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_for_clip, float(args.max_grad_norm))
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    return total_loss / max(1, total_batches), global_step


@torch.no_grad()
def evaluate(
    model: ScanQACompositionModel,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    modality: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model.eval()
    eval_start = time.time()
    predictions: List[str] = []
    references: List[List[str]] = []
    question_types: List[str] = []
    eval_batches = len(dataloader)
    eval_log_every = max(1, eval_batches // 5)

    for eval_step, batch in enumerate(dataloader):
        if not batch:
            continue

        prompts = [f"Question: {q}\nAnswer:" for q in batch["questions"]]
        prompt_encodings = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors="pt",
        )

        kwargs = {
            "input_ids": prompt_encodings["input_ids"].to(device),
            "attention_mask": prompt_encodings["attention_mask"].to(device),
            "max_new_tokens": args.max_answer_tokens,
            "do_sample": False,
            "num_beams": args.eval_num_beams,
            "repetition_penalty": args.eval_repetition_penalty,
            "no_repeat_ngram_size": args.eval_no_repeat_ngram_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        mm = resolve_modality_batch(batch, modality)
        if mm["pointclouds"] is not None:
            kwargs["pointclouds"] = mm["pointclouds"].to(device)
        if mm["images"] is not None:
            kwargs["images"] = mm["images"]

        output_ids = model.generate_for_eval(eval_modality=modality, **kwargs)
        extra_prefix_len = int(model.eval_prompt_prefix_length(modality))
        prompt_width = int(prompt_encodings["input_ids"].size(1))

        for i, seq in enumerate(output_ids):
            if seq.size(0) > prompt_width + extra_prefix_len:
                gen = seq[prompt_width + extra_prefix_len:]
            else:
                gen = seq
            predictions.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
            references.append(list(batch["all_answers"][i]))
            question_types.append(str(batch["question_types"][i]))

        if (eval_step + 1) % eval_log_every == 0:
            elapsed = time.time() - eval_start
            avg_batch_sec = elapsed / float(eval_step + 1)
            eta_sec = avg_batch_sec * float(eval_batches - (eval_step + 1))
            print(
                f"  [eval:{modality}] step {eval_step + 1}/{eval_batches} "
                f"n={len(predictions)} elapsed={elapsed/60.0:.1f}m eta={eta_sec/60.0:.1f}m",
                flush=True,
            )

    total_elapsed = time.time() - eval_start
    print(
        f"  [eval:{modality}] complete in {total_elapsed/60.0:.1f}m "
        f"({total_elapsed/max(1, len(predictions)):.3f}s/sample)",
        flush=True,
    )
    metrics = compute_sqa3d_metrics(predictions, references, question_types)
    metrics["predictions"] = predictions
    metrics["references"] = references
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SQA3D composition training")
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-config", type=str, default="sqa3d_internvl_1b")
    p.add_argument("--llm-model", type=str, default=None)
    p.add_argument(
        "--train-modality",
        type=str,
        default="interleaved",
        choices=["pointcloud", "image", "both", "interleaved", "interleaved_vision_first"],
    )
    p.add_argument("--eval-modalities", type=str, default="both,pointcloud,image,text")
    p.add_argument("--fusion-gate", type=float, default=1.0)
    p.add_argument("--gate-warmup-steps", type=int, default=0)
    p.add_argument("--fusion-layer-indices", type=str, default=None)
    p.add_argument("--num-pointcloud-tokens", type=int, default=None)
    p.add_argument("--num-points", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--safe-lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lr-scheduler", type=str, default="cosine", choices=["none", "cosine", "linear"])
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=None)
    p.add_argument("--train-answer-mode", type=str, default="random", choices=["random", "first", "shortest"])
    p.add_argument("--max-answer-tokens", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--eval-num-beams", type=int, default=1)
    p.add_argument("--eval-repetition-penalty", type=float, default=1.0)
    p.add_argument("--eval-no-repeat-ngram-size", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--include-situation", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--freeze-llm", action="store_true")
    p.add_argument("--unfreeze-encoder-last-n", type=int, default=0)
    p.add_argument("--encoder-checkpoint", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--train-max-samples", type=int, default=0)
    p.add_argument("--val-max-samples", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="SAFE-SQA3D-Composition")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = build_model_config(args)
    if args.batch_size is None:
        args.batch_size = int(config.get("recommended_batch_size", 2))
    if args.safe_lr is None:
        args.safe_lr = float(config.get("safe_lr", 5e-5))
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    if args.num_points is None:
        args.num_points = int(config.get("num_points", 8192))
    if args.label_smoothing is None:
        args.label_smoothing = float(config.get("label_smoothing", 0.0))

    device = torch.device(args.device)
    model = ScanQACompositionModel(
        modality="both",
        llm_model_name=config.get("llm_model_name", ""),
        pointcloud_encoder_checkpoint=args.encoder_checkpoint,
        num_tokens=int(config.get("num_tokens", 8)),
        fusion_layer_indices=list(config.get("fusion_layer_indices", [1])),
        freeze_llm=bool(args.freeze_llm),
        freeze_encoder=bool(config.get("freeze_pointcloud_encoder", True)),
        unfreeze_encoder_last_n=int(args.unfreeze_encoder_last_n),
        config=config,
    ).to(device)

    eval_modalities = resolve_eval_modalities(args, model)
    train_dataset_modality = "both" if args.train_modality in {"both", "interleaved", "interleaved_vision_first"} else args.train_modality
    eval_dataset_modality = "both" if any(m in {"both", "pointcloud"} for m in eval_modalities) else "image"

    train_ds = SQA3DDataset(
        args.data_path,
        split="train",
        modality=train_dataset_modality,
        num_points=args.num_points,
        augment=True,
        include_situation=bool(args.include_situation),
    )
    val_ds = SQA3DDataset(
        args.data_path,
        split="val",
        modality=eval_dataset_modality,
        num_points=args.num_points,
        augment=False,
        include_situation=bool(args.include_situation),
    )

    train_limit = int(args.train_max_samples or args.max_samples)
    val_limit = int(args.val_max_samples or args.max_samples)
    train_ds = trim_dataset(train_ds, train_limit)
    val_ds = trim_dataset(val_ds, val_limit)

    train_loader = build_dataloader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = build_dataloader(val_ds, args.batch_size, False, args.num_workers)

    tokenizer = model.safe_model.base_vl.tokenizer if hasattr(model, "safe_model") else model.processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    passes_per_epoch = 2 if args.train_modality in {"interleaved", "interleaved_vision_first"} else 1
    total_optimizer_steps = math.ceil(len(train_loader) * int(args.num_epochs) * passes_per_epoch / max(1, int(args.gradient_accumulation_steps)))
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, total_optimizer_steps, args)
    scaler = GradScaler(enabled=bool(args.fp16))

    print(f"Train samples:     {len(train_ds)}")
    print(f"Val samples:       {len(val_ds)}")
    print(f"Train modality:    {args.train_modality}")
    print(f"Eval modalities:   {eval_modalities}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Grad accum:        {args.gradient_accumulation_steps}")
    print(f"Effective batch:   {args.batch_size * args.gradient_accumulation_steps}")
    print(f"FP16:              {args.fp16}")
    print(f"SAFE LR:           {args.safe_lr}")
    print(f"Fusion gate:       {args.fusion_gate}")
    print(f"Include situation: {args.include_situation}")

    wandb_run = None
    if args.wandb and wandb is not None:
        tags = args.wandb_tags.split(",") if args.wandb_tags else None
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=tags,
            config=vars(args),
        )

    history: List[Dict[str, Any]] = []
    best_accuracy = -1.0
    global_step = 0

    def run_eval(epoch_idx: int) -> Dict[str, Any]:
        epoch_result: Dict[str, Any] = {"epoch": epoch_idx, "eval": {}}
        for modality in eval_modalities:
            metrics = evaluate(model, val_loader, tokenizer, device, modality, args)
            epoch_result["eval"][modality] = metrics
            print(
                f"  [eval:{modality}] accuracy={metrics['accuracy']:.2f} "
                f"exact={metrics['exact_match']:.2f} f1={metrics['token_f1']:.2f} "
                f"n={metrics['num_samples']}",
                flush=True,
            )
        if {"pointcloud", "image", "both"}.issubset(set(epoch_result["eval"].keys())):
            pc_acc = epoch_result["eval"]["pointcloud"]["accuracy"]
            img_acc = epoch_result["eval"]["image"]["accuracy"]
            both_acc = epoch_result["eval"]["both"]["accuracy"]
            text_acc = epoch_result["eval"].get("text", {}).get("accuracy", 0.0)
            print(
                f"  [composition] text={text_acc:.2f} pointcloud={pc_acc:.2f} "
                f"image={img_acc:.2f} both={both_acc:.2f} "
                f"gain_vs_best_single={both_acc - max(pc_acc, img_acc):+.2f}",
                flush=True,
            )
        return epoch_result

    print(f"\n[epoch 0/{args.num_epochs}] (baseline eval)")
    baseline = run_eval(0)
    history.append(baseline)

    if wandb_run is not None:
        payload = {"epoch": 0}
        for modality, metrics in baseline["eval"].items():
            payload[f"val/{modality}/accuracy"] = metrics["accuracy"]
            payload[f"val/{modality}/exact_match"] = metrics["exact_match"]
            payload[f"val/{modality}/token_f1"] = metrics["token_f1"]
        wandb_run.log(payload, step=0)

    for epoch in range(args.num_epochs):
        print(f"\n[epoch {epoch + 1}/{args.num_epochs}]")

        epoch_train: Dict[str, float] = {}
        if args.train_modality in {"interleaved", "interleaved_vision_first"}:
            order = ["image", "pointcloud"] if args.train_modality == "interleaved_vision_first" else ["pointcloud", "image"]
            for phase_modality in order:
                print(f"  [phase:{phase_modality}]")
                loss, global_step = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    scaler,
                    device,
                    args,
                    tokenizer,
                    train_modality=phase_modality,
                    global_step=global_step,
                    wandb_run=wandb_run,
                )
                epoch_train[f"{phase_modality}_train_loss"] = float(loss)
                print(f"  [phase:{phase_modality}] loss={loss:.4f}", flush=True)
        else:
            loss, global_step = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                args,
                tokenizer,
                train_modality=args.train_modality,
                global_step=global_step,
                wandb_run=wandb_run,
            )
            epoch_train[f"{args.train_modality}_train_loss"] = float(loss)
            print(f"  [phase:{args.train_modality}] loss={loss:.4f}", flush=True)

        epoch_result = {"epoch": epoch + 1, **epoch_train, "eval": {}}
        if (epoch + 1) % int(args.eval_every) == 0:
            epoch_result = {"epoch": epoch + 1, **epoch_train, **run_eval(epoch + 1)}

        history.append(epoch_result)

        score_modality = "both" if "both" in epoch_result["eval"] else eval_modalities[0]
        score = float(epoch_result["eval"].get(score_modality, {}).get("accuracy", -1.0))
        if score > best_accuracy:
            best_accuracy = score
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "args": vars(args),
                "config": config,
                "best_accuracy": best_accuracy,
            }
            torch.save(ckpt, args.output_dir / "best.pt")
            print(f"  [checkpoint] saved best.pt ({score_modality} accuracy={best_accuracy:.2f})", flush=True)

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "args": vars(args),
                "config": config,
            },
            args.output_dir / "last.pt",
        )

        with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if wandb_run is not None and epoch_result["eval"]:
            payload: Dict[str, Any] = {"epoch": epoch + 1}
            for key, value in epoch_train.items():
                payload[f"train/{key}"] = value
            for modality, metrics in epoch_result["eval"].items():
                payload[f"val/{modality}/accuracy"] = metrics["accuracy"]
                payload[f"val/{modality}/exact_match"] = metrics["exact_match"]
                payload[f"val/{modality}/token_f1"] = metrics["token_f1"]
            if {"pointcloud", "image", "both"}.issubset(set(epoch_result["eval"].keys())):
                payload["val/composition_gain_accuracy"] = (
                    epoch_result["eval"]["both"]["accuracy"]
                    - max(epoch_result["eval"]["pointcloud"]["accuracy"], epoch_result["eval"]["image"]["accuracy"])
                )
            wandb_run.log(payload, step=global_step)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
