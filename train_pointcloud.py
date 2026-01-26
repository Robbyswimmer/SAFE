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
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from safe.models.safe_pointcloud_model import SAFEPointCloudModel
from safe.data.pointcloud_datasets import (
    ModelNet40Dataset,
    Cap3DDataset,
    ShapeNetPartDataset,
    collate_pointcloud_batch,
    create_pointcloud_dataloader,
)
from configs.pointcloud_configs import get_pointcloud_config, list_pointcloud_configs


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
    parser.add_argument("--weight-decay", type=float, default=0.01)
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

    # Debug
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log every N steps when tqdm is disabled (e.g., Slurm logs)",
    )

    # Encoder checkpoint
    parser.add_argument(
        "--encoder-checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained PointBERT checkpoint",
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
) -> SAFEPointCloudModel:
    """Create model from config."""
    # Get encoder config and add checkpoint path if provided
    encoder_config = dict(config.get("pointcloud_encoder_config", {}))
    if encoder_checkpoint:
        encoder_config["checkpoint_path"] = encoder_checkpoint
        print(f"Using encoder checkpoint: {encoder_checkpoint}")

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


def train_epoch_classification(
    model: SAFEPointCloudModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
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

        total_loss += loss.item() * args.gradient_accumulation
        num_batches += 1

        # Simple accuracy tracking (compare generated to target)
        # For classification, we just track if the model is learning
        total += len(labels)
        correct += len(labels)  # Placeholder - real eval done separately

        avg_loss = total_loss / num_batches
        if use_tqdm:
            pbar.set_postfix({"loss": avg_loss})
        elif (batch_idx % max(args.log_every, 1)) == 0:
            step_ms = (time.time() - step_start) * 1000.0
            print(
                f"[TRAIN] epoch={epoch+1} step={batch_idx} loss={avg_loss:.4f} step_ms={step_ms:.0f}",
                flush=True,
            )

    return {
        "loss": total_loss / max(num_batches, 1),
    }


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

                if true_clean in pred_clean or pred_clean in true_clean:
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


def train_epoch_captioning(
    model: SAFEPointCloudModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
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

        total_loss += loss.item() * args.gradient_accumulation
        num_batches += 1

        avg_loss = total_loss / num_batches
        if use_tqdm:
            pbar.set_postfix({"loss": avg_loss})
        elif (batch_idx % max(args.log_every, 1)) == 0:
            step_ms = (time.time() - step_start) * 1000.0
            print(
                f"[TRAIN] epoch={epoch+1} step={batch_idx} loss={avg_loss:.4f} step_ms={step_ms:.0f}",
                flush=True,
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

    # Load config
    config = get_pointcloud_config(args.config)
    print(f"Loaded config: {config['name']}")

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
    model = create_model(config, args.device, encoder_checkpoint=args.encoder_checkpoint)

    # Create optimizer
    trainable_params = model.get_trainable_parameters()
    optimizer = AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    print("\nStarting training...")
    best_metric = 0.0

    for epoch in range(args.num_epochs):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*40}")

        # Train
        if args.phase == "classification":
            train_metrics = train_epoch_classification(
                model, train_loader, optimizer, args.device, args, epoch
            )
        else:
            train_metrics = train_epoch_captioning(
                model, train_loader, optimizer, args.device, args, epoch
            )

        scheduler.step()

        print(f"Train Loss: {train_metrics['loss']:.4f}")

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            if args.phase == "classification":
                eval_metrics = evaluate_classification(model, val_loader, args.device, args)
                print(f"Val Accuracy: {eval_metrics['accuracy']:.4f}")

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
