#!/usr/bin/env python3
"""
Point Cloud Training Script for SAFE Architecture.

Trains the SAFE architecture on point cloud data to validate that the
architecture generalizes beyond audio modality.

Usage:
    # Classification on ModelNet40
    python train_pointcloud.py --config modelnet40 --phase classification

    # Captioning on Cap3D
    python train_pointcloud.py --config cap3d --phase captioning
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from safe.models.safe_pointcloud_model import SAFEPointCloudModel
from safe.models.pointcloud_classifier import PointCloudClassifier
from safe.models.pointcloud_llm_probe import SAFEPointCloudLLMProbe
from safe.data.pointcloud_datasets import (
    ModelNet40Dataset,
    Cap3DDataset,
    ShapeNetPartDataset,
    collate_pointcloud_batch,
    create_pointcloud_dataloader,
)
from configs.pointcloud_configs import get_pointcloud_config, list_pointcloud_configs


def _grad_summary(params: List[torch.nn.Parameter]) -> Dict[str, float]:
    total = 0
    with_grad = 0
    none_grad = 0
    nan_grad = 0
    inf_grad = 0
    l2_sum = 0.0
    max_abs = 0.0

    for p in params:
        if not getattr(p, "requires_grad", False):
            continue
        total += 1
        g = getattr(p, "grad", None)
        if g is None:
            none_grad += 1
            continue
        with_grad += 1
        g_detached = g.detach()
        if not torch.isfinite(g_detached).all():
            nan_grad += int(torch.isnan(g_detached).any().item())
            inf_grad += int(torch.isinf(g_detached).any().item())
        l2_sum += float(g_detached.float().pow(2).sum().item())
        try:
            max_abs = max(max_abs, float(g_detached.abs().max().item()))
        except Exception:
            pass

    l2_norm = math.sqrt(l2_sum) if l2_sum > 0 else 0.0
    return {
        "params_total": float(total),
        "params_with_grad": float(with_grad),
        "params_none_grad": float(none_grad),
        "nan_grad_any": float(nan_grad),
        "inf_grad_any": float(inf_grad),
        "grad_l2": float(l2_norm),
        "grad_max_abs": float(max_abs),
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SAFE architecture on point cloud data"
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="modelnet40",
        choices=list_pointcloud_configs(),
        help="Configuration name",
    )

    # Task
    parser.add_argument(
        "--phase",
        type=str,
        default="classification",
        choices=["classification", "captioning"],
        help="Training phase/task",
    )

    # Data
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Root data directory",
    )

    # Training
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--safe-lr", type=float, default=None, help="LLM-probe: LR for projector+fusion params")
    parser.add_argument("--head-lr", type=float, default=None, help="LLM-probe: LR for classifier head")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--head-weight-decay", type=float, default=0.0, help="LLM-probe: weight decay for head")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Evaluation
    parser.add_argument("--eval-every", type=int, default=1, help="Eval every N epochs")
    parser.add_argument("--max-eval-batches", type=int, default=50)

    # Output
    parser.add_argument("--output-dir", type=str, default="./checkpoints/pointcloud")
    parser.add_argument("--save-every", type=int, default=5)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)

    # Classification mode
    parser.add_argument(
        "--classification-head",
        action="store_true",
        help="Use a simple encoder+MLP classification head (non-generative baseline)",
    )
    parser.add_argument(
        "--llm-probe-head",
        action="store_true",
        help="Use full SAFE (projector+fusion+LLM) with a classification head over pooled LLM hidden states",
    )
    parser.add_argument(
        "--probe-pooling",
        type=str,
        default="mean",
        choices=["last", "mean"],
        help="Pooling strategy for LLM probe hidden states",
    )
    parser.add_argument(
        "--probe-head-type",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
        help="Head type for LLM probe",
    )
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default=None,
        choices=["residual", "film", "kv_augment"],
        help="Override fusion_config.fusion_mode from the config (useful for KV-augment experiments)",
    )

    # Debug
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--debug-checks",
        action="store_true",
        help="Extra sanity checks (finite activations, grad flow); more verbose",
    )
    parser.add_argument(
        "--debug-check-every",
        type=int,
        default=50,
        help="How often to run debug checks (in steps)",
    )
    parser.add_argument(
        "--debug-fusion",
        action="store_true",
        help="Log fusion strength (||delta||/||hidden||) at injection sites",
    )
    parser.add_argument(
        "--debug-fusion-every",
        type=int,
        default=50,
        help="How often fusion strength logs print (in hook calls)",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log every N steps when tqdm is disabled (e.g., Slurm logs)",
    )

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="SAFE", help="Wandb project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity (optional)")
    parser.add_argument("--wandb-group", type=str, default=None, help="Wandb group (optional)")
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated wandb tags")
    parser.add_argument("--wandb-notes", type=str, default=None, help="Wandb notes (optional)")

    # Encoder checkpoint
    parser.add_argument(
        "--encoder-checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained PointBERT checkpoint",
    )
    parser.add_argument(
        "--unfreeze-encoder-last-n",
        type=int,
        default=0,
        help="Unfreeze last N pointcloud encoder transformer blocks (0 = fully frozen)",
    )

    return parser.parse_args()


def create_dataset(
    args: argparse.Namespace,
    config: Dict[str, Any],
    split: str,
) -> Any:
    """Create dataset based on config and phase."""
    dataset_name = config.get("dataset", "modelnet40")
    num_points = config.get("num_points", 1024)

    if dataset_name in ["modelnet40", "modelnet40_cls"]:
        dataset = ModelNet40Dataset(
            data_path=args.data_path,
            split=split,
            num_points=num_points,
        )
    elif dataset_name in ["cap3d", "cap3d_captioning"]:
        dataset = Cap3DDataset(
            data_path=args.data_path,
            split=split,
            num_points=num_points,
            max_samples=args.max_train_samples,
        )
    elif dataset_name == "shapenet_part":
        dataset = ShapeNetPartDataset(
            data_path=args.data_path,
            split=split,
            num_points=num_points,
            task="classification" if args.phase == "classification" else "captioning",
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


def create_model(
    config: Dict[str, Any],
    device: str,
    encoder_checkpoint: Optional[str] = None,
    unfreeze_encoder_last_n: int = 0,
) -> SAFEPointCloudModel:
    """Create model from config."""
    # Get encoder config and add checkpoint path if provided
    encoder_config = dict(config.get("pointcloud_encoder_config", {}))
    if encoder_checkpoint:
        encoder_config["checkpoint_path"] = encoder_checkpoint
        print(f"Using encoder checkpoint: {encoder_checkpoint}")
    if int(unfreeze_encoder_last_n) > 0:
        encoder_config["unfreeze_last_n_blocks"] = int(unfreeze_encoder_last_n)

    # Extract constructor arguments
    model_kwargs = {
        "llm_model_name": config.get("llm_model_name", "llava-hf/llava-1.5-13b-hf"),
        "vision_model_name": config.get("vision_model_name", "openai/clip-vit-large-patch14"),
        "pointcloud_encoder_type": config.get("pointcloud_encoder_type", "pointbert"),
        "pointcloud_encoder_config": encoder_config,
        "projector_type": config.get("projector_type", "standard"),
        "num_tokens": config.get("num_tokens", 8),
        "projector_config": config.get("projector_config", {}),
        "fusion_type": config.get("fusion_type", "multilayer"),
        "fusion_layer_indices": config.get("fusion_layer_indices", [12, 24, 36]),
        "lora_rank": config.get("lora_rank", 8),
        "fusion_config": config.get("fusion_config", {}),
        "freeze_base_vl": config.get("freeze_base_vl", True),
        "freeze_pointcloud_encoder": config.get("freeze_pointcloud_encoder", True),
        "label_smoothing": config.get("label_smoothing", 0.0),
        "llm_hidden_size": config.get("llm_hidden_size", 5120),
        "pointcloud_embed_dim": config.get("pointcloud_embed_dim", 768),
    }

    model = SAFEPointCloudModel(**model_kwargs)
    model.enable_pointcloud_training()
    model = model.to(device)

    return model


def create_classifier_model(
    config: Dict[str, Any],
    device: str,
    encoder_checkpoint: Optional[str] = None,
    unfreeze_encoder_last_n: int = 0,
) -> PointCloudClassifier:
    """Create a simple point cloud classifier (encoder+MLP)."""
    encoder_config = dict(config.get("pointcloud_encoder_config", {}))
    if encoder_checkpoint:
        encoder_config["checkpoint_path"] = encoder_checkpoint
        print(f"Using encoder checkpoint: {encoder_checkpoint}")
    if int(unfreeze_encoder_last_n) > 0:
        encoder_config["unfreeze_last_n_blocks"] = int(unfreeze_encoder_last_n)

    num_classes = int(config.get("num_classes", 40))
    model = PointCloudClassifier(
        num_classes=num_classes,
        encoder_model_name=encoder_config.get("model_name", "pointbert-base"),
        encoder_num_points=int(config.get("num_points", encoder_config.get("num_points", 1024))),
        encoder_embed_dim=int(config.get("pointcloud_embed_dim", encoder_config.get("embed_dim", 768))),
        encoder_checkpoint_path=encoder_config.get("checkpoint_path"),
        unfreeze_last_n_blocks=int(unfreeze_encoder_last_n),
        hidden_dim=int(config.get("classifier_hidden_dim", 512)),
        dropout=float(config.get("classifier_dropout", 0.3)),
    ).to(device)
    return model


def create_llm_probe_model(
    config: Dict[str, Any],
    device: str,
    encoder_checkpoint: Optional[str] = None,
    pooling: str = "last",
    head_type: str = "linear",
    unfreeze_encoder_last_n: int = 0,
) -> SAFEPointCloudLLMProbe:
    """Create a pointcloud LLM probe model (SAFE + classifier head)."""
    safe_config = dict(config)
    encoder_config = dict(config.get("pointcloud_encoder_config", {}))
    if encoder_checkpoint:
        encoder_config["checkpoint_path"] = encoder_checkpoint
        safe_config["pointcloud_encoder_config"] = encoder_config
        print(f"Using encoder checkpoint: {encoder_checkpoint}")
    if int(unfreeze_encoder_last_n) > 0:
        encoder_config["unfreeze_last_n_blocks"] = int(unfreeze_encoder_last_n)
        safe_config["pointcloud_encoder_config"] = encoder_config

    num_classes = int(config.get("num_classes", 40))
    safe_config.setdefault("freeze_base_vl", True)
    safe_config.setdefault("freeze_pointcloud_encoder", True)
    model = SAFEPointCloudLLMProbe(
        safe_config=safe_config,
        num_classes=num_classes,
        head_type=head_type,
        pooling=pooling,
    ).to(device)
    return model


def train_epoch_classification(
    model: SAFEPointCloudModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: str,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    """Train one epoch for classification task."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    use_tqdm = sys.stdout.isatty()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not use_tqdm)

    for batch_idx, batch in enumerate(pbar):
        step_start = time.time()
        pointclouds = batch["pointclouds"].to(device)
        labels = batch["labels"].to(device)
        questions = batch["questions"]
        answers = batch["answers"]

        # Tokenize question + answer for training
        tokenizer = model.base_vl.tokenizer
        texts = [f"Question: {q} Answer: {a}" for q, a in zip(questions, answers)]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Create labels (shift by 1 for causal LM)
        lm_labels = input_ids.clone()
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
            pointcloud=pointclouds,
        )

        loss = outputs["loss"]
        loss = loss / args.gradient_accumulation

        if args.debug:
            print(f"[DEBUG] Step {batch_idx}: before backward", flush=True)
        loss.backward()
        if args.debug:
            print(f"[DEBUG] Step {batch_idx}: after backward", flush=True)

        if (batch_idx + 1) % args.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_parameters(),
                args.max_grad_norm,
            )
            if args.debug:
                print(f"[DEBUG] Step {batch_idx}: optimizer step", flush=True)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * args.gradient_accumulation
        num_batches += 1

        # Simple accuracy tracking (compare generated to target)
        # For classification, we just track if the model is learning
        total += len(labels)
        correct += len(labels)  # Placeholder - real eval done separately

        avg_loss = total_loss / num_batches
        do_log = (batch_idx % max(args.log_every, 1)) == 0
        if use_tqdm:
            pbar.set_postfix({"loss": avg_loss})
        if do_log:
            step_ms = (time.time() - step_start) * 1000.0
            if not use_tqdm:
                print(
                    f"[TRAIN] epoch={epoch+1} step={batch_idx} loss={avg_loss:.4f} step_ms={step_ms:.0f}",
                    flush=True,
                )
            if getattr(args, "wandb", False) and wandb is not None:
                global_step = epoch * len(dataloader) + batch_idx
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/step_ms": step_ms,
                        "epoch": epoch + 1,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

    return {
        "loss": total_loss / max(num_batches, 1),
    }


def train_epoch_classification_head(
    model: PointCloudClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: str,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    """Train one epoch for classification using a direct classification head."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    use_tqdm = sys.stdout.isatty()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not use_tqdm)
    loss_fct = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(pbar):
        step_start = time.time()
        pointclouds = batch["pointclouds"].to(device)
        labels = batch["labels"].to(device, dtype=torch.long)

        logits = model(pointclouds)
        loss = loss_fct(logits, labels) / args.gradient_accumulation

        if args.debug:
            print(f"[DEBUG] Step {batch_idx}: before backward", flush=True)
        loss.backward()
        if args.debug:
            print(f"[DEBUG] Step {batch_idx}: after backward", flush=True)

        if (batch_idx + 1) % args.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
            if args.debug:
                print(f"[DEBUG] Step {batch_idx}: optimizer step", flush=True)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * args.gradient_accumulation
        num_batches += 1

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

        avg_loss = total_loss / num_batches
        avg_acc = correct / max(total, 1)
        do_log = (batch_idx % max(args.log_every, 1)) == 0
        if use_tqdm:
            pbar.set_postfix({"loss": avg_loss, "acc": avg_acc})
        if do_log:
            step_ms = (time.time() - step_start) * 1000.0
            if not use_tqdm:
                print(
                    f"[TRAIN] epoch={epoch+1} step={batch_idx} loss={avg_loss:.4f} acc={avg_acc:.4f} "
                    f"step_ms={step_ms:.0f}",
                    flush=True,
                )
            if getattr(args, "wandb", False) and wandb is not None:
                global_step = epoch * len(dataloader) + batch_idx
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/accuracy": avg_acc,
                        "train/step_ms": step_ms,
                        "epoch": epoch + 1,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

    return {"loss": total_loss / max(num_batches, 1), "accuracy": correct / max(total, 1)}


def evaluate_classification(
    model: SAFEPointCloudModel,
    dataloader: DataLoader,
    device: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Evaluate classification accuracy."""
    model.eval()

    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    tokenizer = model.base_vl.tokenizer

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if batch_idx >= args.max_eval_batches:
                break

            pointclouds = batch["pointclouds"].to(device)
            true_answers = batch["answers"]
            questions = batch["questions"]

            # Generate predictions
            prompts = [f"Question: {q} Answer:" for q in questions]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
                max_new_tokens=10,
                num_beams=1,  # Greedy for speed
            )

            # Decode predictions
            predictions = tokenizer.batch_decode(
                outputs[:, input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            # Compare
            for pred, true in zip(predictions, true_answers):
                pred_clean = pred.strip().lower()
                true_clean = true.strip().lower()

                total += 1
                if true_clean not in class_total:
                    class_total[true_clean] = 0
                    class_correct[true_clean] = 0
                class_total[true_clean] += 1

                # Avoid counting empty/short generations as correct:
                # in Python, "" in "chair" is True.
                if pred_clean and true_clean and (true_clean in pred_clean):
                    correct += 1
                    class_correct[true_clean] += 1

    accuracy = correct / max(total, 1)

    # Per-class accuracy
    per_class_acc = {
        cls: class_correct[cls] / class_total[cls]
        for cls in class_total
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class_accuracy": per_class_acc,
    }


def evaluate_classification_head(
    model: PointCloudClassifier,
    dataloader: DataLoader,
    device: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Evaluate direct classification head accuracy."""
    model.eval()

    correct = 0
    total = 0
    num_classes = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if batch_idx >= args.max_eval_batches:
                break

            pointclouds = batch["pointclouds"].to(device)
            labels = batch["labels"].to(device, dtype=torch.long)

            logits = model(pointclouds)
            if num_classes is None:
                num_classes = logits.shape[-1]

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "num_classes": num_classes,
    }


def train_epoch_llm_probe_head(
    model: SAFEPointCloudLLMProbe,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: str,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    """Train one epoch for classification using LLM hidden-state probe head."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    use_tqdm = sys.stdout.isatty()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not use_tqdm)
    loss_fct = nn.CrossEntropyLoss()

    tokenizer = model.safe_model.base_vl.tokenizer

    for batch_idx, batch in enumerate(pbar):
        step_start = time.time()
        if args.debug_fusion and batch_idx == 0:
            try:
                model.safe_model.set_fusion_debug(True, log_every=args.debug_fusion_every)
            except Exception:
                pass
        pointclouds = batch["pointclouds"].to(device)
        labels = batch["labels"].to(device, dtype=torch.long)
        questions = batch["questions"]

        if args.debug_checks and epoch == 0 and batch_idx == 0:
            pc_finite = torch.isfinite(pointclouds).all().item()
            pc_min = pointclouds.min().item()
            pc_max = pointclouds.max().item()
            pc_mean = pointclouds.mean().item()
            pc_std = pointclouds.float().std().item()
            print(
                f"[DEBUG] PC input: shape={tuple(pointclouds.shape)} finite={pc_finite} "
                f"min={pc_min:.4f} max={pc_max:.4f} mean={pc_mean:.4f} std={pc_std:.4f}",
                flush=True,
            )
            print(
                f"[DEBUG] Labels: shape={tuple(labels.shape)} min={labels.min().item()} max={labels.max().item()}",
                flush=True,
            )

        encoded = tokenizer(
            list(questions),
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        probe_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pointcloud=pointclouds,
        )
        logits = probe_out.logits
        if args.debug_checks and epoch == 0 and batch_idx == 0:
            pooled = probe_out.pooled
            print(
                f"[DEBUG] Pooled: shape={tuple(pooled.shape)} finite={torch.isfinite(pooled).all().item()} "
                f"min={pooled.min().item():.4f} max={pooled.max().item():.4f}",
                flush=True,
            )
            print(
                f"[DEBUG] Logits: shape={tuple(logits.shape)} finite={torch.isfinite(logits).all().item()} "
                f"min={logits.min().item():.4f} max={logits.max().item():.4f}",
                flush=True,
            )
        loss = loss_fct(logits, labels) / args.gradient_accumulation

        if args.debug:
            print(f"[DEBUG] Step {batch_idx}: before backward", flush=True)
        loss.backward()
        if args.debug:
            print(f"[DEBUG] Step {batch_idx}: after backward", flush=True)

        if (batch_idx + 1) % args.gradient_accumulation == 0:
            total_norm = torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), args.max_grad_norm)
            do_checks = args.debug_checks and ((batch_idx % max(args.debug_check_every, 1)) == 0)
            if do_checks:
                safe_stats = _grad_summary(model.get_safe_params())
                head_stats = _grad_summary(model.get_head_params())
                print(
                    f"[DEBUG] GradClip: total_norm={float(total_norm):.4f} "
                    f"safe_l2={safe_stats['grad_l2']:.4f} safe_max={safe_stats['grad_max_abs']:.4g} "
                    f"safe_none={int(safe_stats['params_none_grad'])}/{int(safe_stats['params_total'])} "
                    f"head_l2={head_stats['grad_l2']:.4f} head_max={head_stats['grad_max_abs']:.4g} "
                    f"head_none={int(head_stats['params_none_grad'])}/{int(head_stats['params_total'])} "
                    f"nan_any={int(safe_stats['nan_grad_any']+head_stats['nan_grad_any'])} "
                    f"inf_any={int(safe_stats['inf_grad_any']+head_stats['inf_grad_any'])}",
                    flush=True,
                )
            if args.debug:
                print(f"[DEBUG] Step {batch_idx}: optimizer step", flush=True)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * args.gradient_accumulation
        num_batches += 1

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

        avg_loss = total_loss / num_batches
        avg_acc = correct / max(total, 1)
        do_log = (batch_idx % max(args.log_every, 1)) == 0
        if use_tqdm:
            pbar.set_postfix({"loss": avg_loss, "acc": avg_acc})
        if do_log:
            step_ms = (time.time() - step_start) * 1000.0
            if not use_tqdm:
                print(
                    f"[TRAIN] epoch={epoch+1} step={batch_idx} loss={avg_loss:.4f} acc={avg_acc:.4f} "
                    f"step_ms={step_ms:.0f}",
                    flush=True,
                )
            if getattr(args, "wandb", False) and wandb is not None:
                global_step = epoch * len(dataloader) + batch_idx
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/accuracy": avg_acc,
                        "train/step_ms": step_ms,
                        "epoch": epoch + 1,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

    return {"loss": total_loss / max(num_batches, 1), "accuracy": correct / max(total, 1)}


def evaluate_llm_probe_head(
    model: SAFEPointCloudLLMProbe,
    dataloader: DataLoader,
    device: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Evaluate LLM probe head accuracy."""
    model.eval()

    correct = 0
    total = 0
    tokenizer = model.safe_model.base_vl.tokenizer

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if batch_idx >= args.max_eval_batches:
                break
            pointclouds = batch["pointclouds"].to(device)
            labels = batch["labels"].to(device, dtype=torch.long)
            questions = batch["questions"]

            encoded = tokenizer(
                list(questions),
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
            ).logits

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def train_epoch_captioning(
    model: SAFEPointCloudModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: str,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    """Train one epoch for captioning task."""
    model.train()

    total_loss = 0.0
    num_batches = 0

    use_tqdm = sys.stdout.isatty()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not use_tqdm)

    for batch_idx, batch in enumerate(pbar):
        step_start = time.time()
        pointclouds = batch["pointclouds"].to(device)
        questions = batch["questions"]
        answers = batch["answers"]

        # Tokenize
        tokenizer = model.base_vl.tokenizer
        texts = [f"Question: {q} Answer: {a}" for q, a in zip(questions, answers)]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Labels
        lm_labels = input_ids.clone()
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
            pointcloud=pointclouds,
        )

        loss = outputs["loss"] / args.gradient_accumulation
        if args.debug:
            print(f"[DEBUG] Step {batch_idx}: before backward", flush=True)
        loss.backward()
        if args.debug:
            print(f"[DEBUG] Step {batch_idx}: after backward", flush=True)

        if (batch_idx + 1) % args.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_parameters(),
                args.max_grad_norm,
            )
            if args.debug:
                print(f"[DEBUG] Step {batch_idx}: optimizer step", flush=True)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * args.gradient_accumulation
        num_batches += 1

        avg_loss = total_loss / num_batches
        do_log = (batch_idx % max(args.log_every, 1)) == 0
        if use_tqdm:
            pbar.set_postfix({"loss": avg_loss})
        if do_log:
            step_ms = (time.time() - step_start) * 1000.0
            if not use_tqdm:
                print(
                    f"[TRAIN] epoch={epoch+1} step={batch_idx} loss={avg_loss:.4f} step_ms={step_ms:.0f}",
                    flush=True,
                )
            if getattr(args, "wandb", False) and wandb is not None:
                global_step = epoch * len(dataloader) + batch_idx
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/step_ms": step_ms,
                        "epoch": epoch + 1,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

    return {"loss": total_loss / max(num_batches, 1)}


def evaluate_captioning(
    model: SAFEPointCloudModel,
    dataloader: DataLoader,
    device: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Evaluate captioning (sample predictions)."""
    model.eval()

    samples = []
    tokenizer = model.base_vl.tokenizer

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:  # Just sample a few
                break

            pointclouds = batch["pointclouds"].to(device)
            questions = batch["questions"]
            true_captions = batch["answers"]

            # Generate
            prompts = [f"Question: {q} Answer:" for q in questions]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
                max_new_tokens=50,
                num_beams=3,
            )

            predictions = tokenizer.batch_decode(
                outputs[:, input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            for pred, true in zip(predictions, true_captions):
                samples.append({
                    "prediction": pred.strip(),
                    "reference": true,
                })

    return {"samples": samples}


def save_checkpoint(
    model: SAFEPointCloudModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    output_dir: Path,
    name: str = "checkpoint",
) -> None:
    """Save model checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trainable weights only
    trainable_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": trainable_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    path = output_dir / f"{name}_epoch{epoch}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")

    # Also save as latest
    latest_path = output_dir / f"{name}_latest.pt"
    torch.save(checkpoint, latest_path)


def main():
    args = parse_args()

    print("=" * 60)
    print("SAFE Point Cloud Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Phase: {args.phase}")
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # Wandb
    if args.wandb and wandb is None:
        print("[WANDB] --wandb set but wandb is not installed; skipping logging.", flush=True)
    if args.wandb and wandb is not None:
        print(f"[WANDB] mode={os.environ.get('WANDB_MODE', '(unset)')}", flush=True)
        print(f"[WANDB] disabled={os.environ.get('WANDB_DISABLED', '(unset)')}", flush=True)
        print(f"[WANDB] api_key={'set' if os.environ.get('WANDB_API_KEY') else 'not set'}", flush=True)
        tags = None
        if args.wandb_tags:
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or f"pointcloud-{args.config}-{args.phase}-{time.strftime('%Y%m%d-%H%M%S')}",
            notes=args.wandb_notes,
            tags=tags,
            config=vars(args),
        )

    # Load config
    config = get_pointcloud_config(args.config)
    print(f"Loaded config: {config['name']}")

    # Optional overrides (must happen BEFORE model creation)
    if args.fusion_mode is not None:
        config.setdefault("fusion_config", {})
        config["fusion_config"]["fusion_mode"] = args.fusion_mode
        print(f"[ConfigOverride] fusion_mode={args.fusion_mode}", flush=True)

        # KV augmentation requires modality tokens in encoder space (input_dim),
        # not LLM hidden space; force projector output_dim accordingly unless user overrides.
        if args.fusion_mode == "kv_augment":
            config.setdefault("projector_config", {})
            config["projector_config"]["output_dim"] = int(config.get("pointcloud_embed_dim", 768))
            print(
                f"[ConfigOverride] kv_augment: projector_config.output_dim={config['projector_config']['output_dim']}",
                flush=True,
            )

    # Override batch size if specified
    if args.batch_size is None:
        args.batch_size = config.get("recommended_batch_size", 8)

    # Create output directory
    output_dir = Path(args.output_dir) / args.config / args.phase
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = create_dataset(args, config, "train")
    val_dataset = create_dataset(args, config, "test")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = create_pointcloud_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = create_pointcloud_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Create model
    print("\nCreating model...")
    if args.phase == "classification" and args.classification_head:
        model = create_classifier_model(
            config,
            args.device,
            encoder_checkpoint=args.encoder_checkpoint,
            unfreeze_encoder_last_n=args.unfreeze_encoder_last_n,
        )
    elif args.phase == "classification" and args.llm_probe_head:
        model = create_llm_probe_model(
            config,
            args.device,
            encoder_checkpoint=args.encoder_checkpoint,
            pooling=args.probe_pooling,
            head_type=args.probe_head_type,
            unfreeze_encoder_last_n=args.unfreeze_encoder_last_n,
        )
    else:
        model = create_model(
            config,
            args.device,
            encoder_checkpoint=args.encoder_checkpoint,
            unfreeze_encoder_last_n=args.unfreeze_encoder_last_n,
        )

    # Optimizer + scheduler (stepped per optimizer update)
    total_updates = max(1, (len(train_loader) * args.num_epochs) // max(args.gradient_accumulation, 1))
    warmup_steps = max(0, int(args.warmup_steps))

    def _lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_updates - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    if args.llm_probe_head and hasattr(model, "get_safe_params") and hasattr(model, "get_head_params"):
        safe_lr = float(args.safe_lr) if args.safe_lr is not None else float(args.lr)
        head_lr = float(args.head_lr) if args.head_lr is not None else max(1e-4, float(args.lr) * 10.0)
        optimizer = AdamW(
            [
                {"params": model.get_safe_params(), "lr": safe_lr, "weight_decay": float(args.weight_decay)},
                {"params": model.get_head_params(), "lr": head_lr, "weight_decay": float(args.head_weight_decay)},
            ]
        )
    else:
        if hasattr(model, "get_trainable_params"):
            trainable_params = model.get_trainable_params()
        elif hasattr(model, "get_trainable_parameters"):
            trainable_params = model.get_trainable_parameters()
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # Training loop
    print("\nStarting training...")
    best_metric = 0.0

    for epoch in range(args.num_epochs):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*40}")

        # Train
        if args.phase == "classification":
            if args.classification_head:
                train_metrics = train_epoch_classification_head(
                    model, train_loader, optimizer, scheduler, args.device, args, epoch
                )
            elif args.llm_probe_head:
                train_metrics = train_epoch_llm_probe_head(
                    model, train_loader, optimizer, scheduler, args.device, args, epoch
                )
            else:
                train_metrics = train_epoch_classification(
                    model, train_loader, optimizer, scheduler, args.device, args, epoch
                )
        else:
            train_metrics = train_epoch_captioning(
                model, train_loader, optimizer, scheduler, args.device, args, epoch
            )

        print(f"Train Loss: {train_metrics['loss']:.4f}")

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            if args.phase == "classification":
                if args.classification_head:
                    eval_metrics = evaluate_classification_head(model, val_loader, args.device, args)
                elif args.llm_probe_head:
                    eval_metrics = evaluate_llm_probe_head(model, val_loader, args.device, args)
                else:
                    eval_metrics = evaluate_classification(model, val_loader, args.device, args)
                print(f"Val Accuracy: {eval_metrics['accuracy']:.4f}")
                if args.wandb and wandb is not None:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train/epoch_loss": train_metrics.get("loss"),
                            "train/epoch_accuracy": train_metrics.get("accuracy"),
                            "val/accuracy": eval_metrics.get("accuracy"),
                            "lr": scheduler.get_last_lr()[0],
                        },
                        step=(epoch + 1) * len(train_loader),
                    )

                # Track best
                if eval_metrics["accuracy"] > best_metric:
                    best_metric = eval_metrics["accuracy"]
                    save_checkpoint(model, optimizer, epoch, eval_metrics, output_dir, "best")
            else:
                eval_metrics = evaluate_captioning(model, val_loader, args.device, args)
                print("Sample predictions:")
                for s in eval_metrics["samples"][:3]:
                    print(f"  Pred: {s['prediction']}")
                    print(f"  Ref:  {s['reference']}")
                    print()
                if args.wandb and wandb is not None:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train/epoch_loss": train_metrics.get("loss"),
                            "val/samples": len(eval_metrics.get("samples", [])),
                            "lr": scheduler.get_last_lr()[0],
                        },
                        step=(epoch + 1) * len(train_loader),
                    )

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch,
                {"train_loss": train_metrics["loss"]},
                output_dir, "checkpoint"
            )

    print("\n" + "=" * 60)
    print("Training complete!")
    if args.phase == "classification":
        print(f"Best accuracy: {best_metric:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
