#!/usr/bin/env python3
"""
ScanNet Composition Training Script.

Tests point cloud + image composition for scene classification.

Usage:
    # Point cloud only
    python train_scannet_composition.py --modality pointcloud

    # Image only
    python train_scannet_composition.py --modality image

    # Both (composition)
    python train_scannet_composition.py --modality both
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
import torch.nn.functional as F
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

from safe.data.scannet_dataset import ScanNetDataset, collate_scannet_batch, SCANNET_SCENE_TYPES


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(smooth_target * log_probs).sum(dim=-1)
        return loss.mean()


class ScanNetCompositionModel(nn.Module):
    """
    Model for ScanNet scene classification with optional modality composition.

    Composition approach:
    - Image-only: LLaVA processes image+text (already trained)
    - PC-only: SAFE injects PC tokens into LLM via fusion
    - Both: LLaVA processes image+text natively, SAFE adds PC residuals
            This is TRUE composition - both modalities in single forward pass
    """

    def __init__(
        self,
        modality: str = "both",
        num_classes: int = len(SCANNET_SCENE_TYPES),
        llm_model_name: str = "llava-hf/llava-1.5-13b-hf",
        pointcloud_encoder_checkpoint: Optional[str] = None,
        num_tokens: int = 8,
        fusion_layer_indices: List[int] = [1, 5, 9, 13, 17, 21],
        freeze_llm: bool = True,
        freeze_encoder: bool = True,
        unfreeze_encoder_last_n: int = 0,
    ):
        super().__init__()
        self.modality = modality
        self.num_classes = num_classes

        if modality == "image":
            # Image-only: Use standalone LLaVA
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            print("Loading LLaVA model for image-only mode...")
            self.llava = LlavaForConditionalGeneration.from_pretrained(
                llm_model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            self.processor = AutoProcessor.from_pretrained(llm_model_name)

            # Freeze LLaVA - it's already trained
            for param in self.llava.parameters():
                param.requires_grad = False

            self.llm_hidden_size = self.llava.config.text_config.hidden_size

        else:
            # PC-only or Both: Use SAFE model (which contains LLaVA as base_vl)
            from safe.models.safe_pointcloud_model import SAFEPointCloudModel
            print(f"Loading SAFE point cloud model for {modality} mode...")

            safe_config = {
                "llm_model_name": llm_model_name,
                "pointcloud_encoder_type": "pointbert",
                "pointcloud_encoder_config": {
                    "model_name": "pointbert-base",
                    "checkpoint_path": pointcloud_encoder_checkpoint,
                    "unfreeze_last_n_blocks": unfreeze_encoder_last_n,
                },
                "num_tokens": num_tokens,
                "fusion_layer_indices": fusion_layer_indices,
                "freeze_base_vl": freeze_llm,
                "freeze_pointcloud_encoder": freeze_encoder,
            }

            self.safe_model = SAFEPointCloudModel(**safe_config)
            self.safe_model.enable_pointcloud_training()

            # For "both" mode, we'll use SAFE's processor for images
            self.processor = self.safe_model.base_vl.processor

            self.llm_hidden_size = self.safe_model.base_vl.llm.config.text_config.hidden_size

        # Classification head
        self.classifier = nn.Linear(self.llm_hidden_size, num_classes)

    def _process_images(self, images: List, device) -> torch.Tensor:
        """Process PIL images to pixel_values tensor."""
        processed = self.processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=device, dtype=torch.float16)
        return pixel_values

    def forward(
        self,
        pointclouds: Optional[torch.Tensor] = None,
        images: Optional[List] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""

        if self.modality == "pointcloud":
            # Point cloud only path through SAFE
            outputs = self.safe_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
                output_hidden_states=True,
            )
            # Get last hidden state, pool last token
            hidden = outputs["hidden_states"][:, -1, :]  # [B, hidden_dim]

        elif self.modality == "image":
            # Image only path through LLaVA (frozen, zero-shot)
            inputs = self.processor(
                images=images,
                text=["What type of room is this?"] * len(images),
                return_tensors="pt",
                padding=True,
            ).to(self.llava.device)

            with torch.no_grad():
                outputs = self.llava(
                    **inputs,
                    output_hidden_states=True,
                )
            hidden = outputs.hidden_states[-1][:, -1, :]

        else:  # "both" - TRUE COMPOSITION
            # Process images to pixel_values
            pixel_values = self._process_images(images, input_ids.device)

            # TRUE COMPOSITION:
            # LLaVA processes image+text natively (vision tokens + text tokens)
            # SAFE injects PC tokens as residuals at fusion layers
            # Both modalities contribute to same forward pass
            outputs = self.safe_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
            hidden = outputs["hidden_states"][:, -1, :]

        # Classify
        logits = self.classifier(hidden.float())

        return {"logits": logits, "hidden": hidden}

    def get_trainable_params(self):
        """Get trainable parameters (classifier + SAFE adapter)."""
        params = list(self.classifier.parameters())

        if self.modality in ["pointcloud", "both"]:
            if hasattr(self, 'safe_model') and hasattr(self.safe_model, "get_trainable_params"):
                params.extend(self.safe_model.get_trainable_params())

        return params


def parse_args():
    parser = argparse.ArgumentParser(description="ScanNet Composition Training")

    # Data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--modality", type=str, default="both", choices=["pointcloud", "image", "both"])
    parser.add_argument("--num-points", type=int, default=8192)

    # Model
    parser.add_argument("--fusion-layer-indices", type=str, default="1,5,9,13,17,21")
    parser.add_argument("--num-pointcloud-tokens", type=int, default=8)
    parser.add_argument("--encoder-checkpoint", type=str, default=None)
    parser.add_argument("--unfreeze-encoder-last-n", type=int, default=0)

    # Training
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--safe-lr", type=float, default=6e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lr-scheduler", type=str, default="constant")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Eval
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-eval-batches", type=int, default=999)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ScanNet-Composition")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)

    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, scheduler, device, args, epoch):
    """Train one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    loss_fn = LabelSmoothingCrossEntropy(args.label_smoothing) if args.label_smoothing > 0 else nn.CrossEntropyLoss()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for batch_idx, batch in enumerate(pbar):
        if not batch:
            continue

        labels = batch["labels"].to(device)

        # Prepare inputs based on modality
        kwargs = {}
        if args.modality in ["pointcloud", "both"] and batch.get("pointclouds") is not None:
            kwargs["pointclouds"] = batch["pointclouds"].to(device)
        if args.modality in ["image", "both"] and batch.get("images") is not None:
            kwargs["images"] = batch["images"]

        # Tokenize questions for point cloud path
        if args.modality in ["pointcloud", "both"]:
            tokenizer = model.safe_model.base_vl.tokenizer
            encoded = tokenizer(
                batch["questions"],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            kwargs["input_ids"] = encoded["input_ids"].to(device)
            kwargs["attention_mask"] = encoded["attention_mask"].to(device)

        # Forward
        outputs = model(**kwargs)
        logits = outputs["logits"]

        loss = loss_fn(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Track metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, args, max_batches=None):
    """Evaluate model."""
    model.eval()

    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
        if not batch:
            continue

        labels = batch["labels"].to(device)

        # Prepare inputs
        kwargs = {}
        if args.modality in ["pointcloud", "both"] and batch.get("pointclouds") is not None:
            kwargs["pointclouds"] = batch["pointclouds"].to(device)
        if args.modality in ["image", "both"] and batch.get("images") is not None:
            kwargs["images"] = batch["images"]

        if args.modality in ["pointcloud", "both"]:
            tokenizer = model.safe_model.base_vl.tokenizer
            encoded = tokenizer(
                batch["questions"],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            kwargs["input_ids"] = encoded["input_ids"].to(device)
            kwargs["attention_mask"] = encoded["attention_mask"].to(device)

        outputs = model(**kwargs)
        logits = outputs["logits"]

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"accuracy": correct / max(total, 1)}


def main():
    args = parse_args()

    print("=" * 60)
    print("ScanNet Composition Training")
    print("=" * 60)
    print(f"Modality: {args.modality}")
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # Create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Parse fusion layers
    fusion_layers = [int(x) for x in args.fusion_layer_indices.split(",")]

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ScanNetDataset(
        args.data_path,
        split="train",
        modality=args.modality,
        num_points=args.num_points,
    )
    val_dataset = ScanNetDataset(
        args.data_path,
        split="val",
        modality=args.modality,
        num_points=args.num_points,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_scannet_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_scannet_batch,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print("\nCreating model...")
    model = ScanNetCompositionModel(
        modality=args.modality,
        pointcloud_encoder_checkpoint=args.encoder_checkpoint,
        num_tokens=args.num_pointcloud_tokens,
        fusion_layer_indices=fusion_layers,
        unfreeze_encoder_last_n=args.unfreeze_encoder_last_n,
    )
    model = model.to(args.device)

    if args.fp16:
        model = model.half()

    # Optimizer
    trainable_params = model.get_trainable_params()
    optimizer = AdamW([
        {"params": trainable_params, "lr": args.safe_lr},
    ])

    # Scheduler
    total_steps = len(train_loader) * args.num_epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        if args.lr_scheduler == "constant":
            return 1.0
        else:  # cosine
            progress = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # W&B
    if args.wandb and wandb:
        tags = args.wandb_tags.split(",") if args.wandb_tags else []
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=tags,
            config=vars(args),
        )

    # Training loop
    print("\nStarting training...")
    best_acc = 0.0

    for epoch in range(args.num_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, args.device, args, epoch)

        print(f"\nEpoch {epoch+1} - Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            val_metrics = evaluate(model, val_loader, args.device, args, args.max_eval_batches)
            print(f"Val Acc: {val_metrics['accuracy']:.4f}")

            # Save best
            if val_metrics["accuracy"] > best_acc:
                best_acc = val_metrics["accuracy"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": {k: v for k, v in model.state_dict().items() if "classifier" in k or "safe" in k},
                    "best_acc": best_acc,
                    "args": vars(args),
                }, Path(args.output_dir) / "best_model.pt")
                print(f"New best! Saved checkpoint.")

            if args.wandb and wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_acc": val_metrics["accuracy"],
                    "best_acc": best_acc,
                    "lr": scheduler.get_last_lr()[0],
                })

    print(f"\nTraining complete! Best accuracy: {best_acc:.4f}")

    if args.wandb and wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
