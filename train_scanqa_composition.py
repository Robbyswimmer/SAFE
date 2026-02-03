#!/usr/bin/env python3
"""
ScanQA Composition Training Script.

Tests point cloud + image composition for 3D question answering.

Usage:
    # Point cloud only
    python train_scanqa_composition.py --modality pointcloud

    # Image only
    python train_scanqa_composition.py --modality image

    # Both (composition)
    python train_scanqa_composition.py --modality both
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

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: nltk not available. Install for BLEU/METEOR metrics.")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from safe.data.scanqa_dataset import ScanQADataset, collate_scanqa_batch


def compute_qa_metrics(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Compute QA evaluation metrics.

    Args:
        predictions: List of predicted answer strings
        references: List of lists of reference answer strings

    Returns:
        Dictionary with BLEU-1, BLEU-4, METEOR scores
    """
    if not NLTK_AVAILABLE:
        return {"bleu1": 0.0, "bleu4": 0.0, "meteor": 0.0, "exact_match": 0.0}

    if len(predictions) == 0:
        return {"bleu1": 0.0, "bleu4": 0.0, "meteor": 0.0, "exact_match": 0.0}

    bleu1_scores = []
    bleu4_scores = []
    meteor_scores = []
    exact_matches = []

    smoother = SmoothingFunction()

    for pred, refs in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens_list = [ref.lower().split() for ref in refs if ref]

        # Exact match (against any reference)
        exact = any(pred.lower().strip() == ref.lower().strip() for ref in refs if ref)
        exact_matches.append(float(exact))

        # BLEU scores
        if pred_tokens and ref_tokens_list:
            bleu1 = sentence_bleu(ref_tokens_list, pred_tokens,
                                  weights=(1.0, 0, 0, 0),
                                  smoothing_function=smoother.method1)
            bleu4 = sentence_bleu(ref_tokens_list, pred_tokens,
                                  weights=(0.25, 0.25, 0.25, 0.25),
                                  smoothing_function=smoother.method1)
            bleu1_scores.append(bleu1)
            bleu4_scores.append(bleu4)

            # METEOR (max across references)
            meteor = max(meteor_score([ref.split()], pred_tokens) for ref in refs if ref)
            meteor_scores.append(meteor)
        else:
            bleu1_scores.append(0.0)
            bleu4_scores.append(0.0)
            meteor_scores.append(0.0)

    n = len(bleu1_scores)
    return {
        "bleu1": sum(bleu1_scores) / n * 100 if n > 0 else 0.0,
        "bleu4": sum(bleu4_scores) / n * 100 if n > 0 else 0.0,
        "meteor": sum(meteor_scores) / n * 100 if n > 0 else 0.0,
        "exact_match": sum(exact_matches) / len(exact_matches) * 100 if exact_matches else 0.0,
    }


class ScanQACompositionModel(nn.Module):
    """
    Model for ScanQA with optional modality composition.

    Training approach:
    - Image-only: Use LLaVA as-is (already trained for vision+language)
    - PC-only: Train SAFE adapter (PointBERT -> fusion -> LLM)
    - Both: Use SAFE model with images processed through LLaVA's vision encoder
            PC tokens injected via SAFE fusion, images via LLaVA's native path

    Key insight: SAFE's base_vl IS LLaVA, so "both" mode uses a single model
    that processes images natively AND receives PC tokens via fusion.
    """

    def __init__(
        self,
        modality: str = "both",
        llm_model_name: str = "llava-hf/llava-1.5-7b-hf",
        pointcloud_encoder_checkpoint: Optional[str] = None,
        num_tokens: int = 8,
        fusion_layer_indices: List[int] = [1, 5, 9, 13, 17, 21],
        freeze_llm: bool = True,  # Freeze LLM, only train SAFE adapter
        freeze_encoder: bool = True,
        unfreeze_encoder_last_n: int = 0,
    ):
        super().__init__()
        self.modality = modality
        self.llm_model_name = llm_model_name

        if modality == "image":
            # Image-only: Use standalone LLaVA (already trained)
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

            # For "both" mode, we'll use SAFE's processor for images too
            self.processor = self.safe_model.base_vl.processor

            self.llm_hidden_size = self.safe_model.base_vl.llm.config.hidden_size

    def _process_images(self, images: List, device) -> torch.Tensor:
        """Process PIL images to pixel_values tensor."""
        processed = self.processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=device, dtype=torch.float16)
        return pixel_values

    def forward(
        self,
        pointclouds: Optional[torch.Tensor] = None,
        images: Optional[List] = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training (with labels) or inference."""

        if self.modality == "image":
            # Image-only: Use LLaVA directly (zero-shot / frozen)
            if pixel_values is None and images is not None:
                pixel_values = self._process_images(images, input_ids.device)

            outputs = self.llava(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
            return {"loss": outputs.loss, "logits": outputs.logits}

        elif self.modality == "pointcloud":
            # PC-only: Use SAFE with point cloud fusion (no images)
            outputs = self.safe_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
                labels=labels,
            )
            return {"loss": outputs.loss if hasattr(outputs, 'loss') else None,
                    "logits": outputs.logits if hasattr(outputs, 'logits') else None}

        else:  # "both" - TRUE COMPOSITION
            # Process images if needed
            if pixel_values is None and images is not None:
                pixel_values = self._process_images(images, input_ids.device)

            # TRUE COMPOSITION:
            # 1. LLaVA processes image + text natively (vision tokens + text tokens)
            # 2. SAFE injects PC tokens as residuals at fusion layers
            # Both modalities contribute to the same forward pass
            outputs = self.safe_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
                pixel_values=pixel_values,
                labels=labels,
            )
            return {"loss": outputs.loss if hasattr(outputs, 'loss') else None,
                    "logits": outputs.logits if hasattr(outputs, 'logits') else None}

    @torch.no_grad()
    def generate(
        self,
        pointclouds: Optional[torch.Tensor] = None,
        images: Optional[List] = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate answers."""

        if self.modality == "image":
            if pixel_values is None and images is not None:
                pixel_values = self._process_images(images, input_ids.device)

            outputs = self.llava.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )
            return outputs

        elif self.modality == "pointcloud":
            outputs = self.safe_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )
            return outputs

        else:  # "both" - TRUE COMPOSITION
            if pixel_values is None and images is not None:
                pixel_values = self._process_images(images, input_ids.device)

            # Generate with both image and PC
            outputs = self.safe_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )
            return outputs

    def get_trainable_params(self):
        """Get trainable parameters (only SAFE adapter, LLaVA is frozen)."""
        params = []

        if self.modality == "image":
            # Image-only: LLaVA is frozen, no trainable params
            # (unless we add a small adapter later)
            pass
        else:
            # PC-only or Both: Train SAFE adapter
            if hasattr(self, 'safe_model') and hasattr(self.safe_model, "get_trainable_params"):
                params.extend(self.safe_model.get_trainable_params())

        return params


def parse_args():
    parser = argparse.ArgumentParser(description="ScanQA Composition Training")

    # Data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--modality", type=str, default="both", choices=["pointcloud", "image", "both"])
    parser.add_argument("--num-points", type=int, default=8192)

    # Model
    parser.add_argument("--llm-model", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--fusion-layer-indices", type=str, default="1,5,9,13,17,21")
    parser.add_argument("--num-pointcloud-tokens", type=int, default=8)
    parser.add_argument("--encoder-checkpoint", type=str, default=None)
    parser.add_argument("--unfreeze-encoder-last-n", type=int, default=0)
    parser.add_argument("--freeze-llm", action="store_true", help="Freeze LLM (use linear probe)")

    # Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--safe-lr", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)

    # Generation
    parser.add_argument("--max-answer-tokens", type=int, default=32)

    # Eval
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-eval-samples", type=int, default=500)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ScanQA-Composition")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)

    return parser.parse_args()


def prepare_qa_inputs(batch, tokenizer, device, max_length=256):
    """Prepare inputs for QA training/inference."""
    questions = batch["questions"]
    answers = batch["answers"]

    # Format: "Question: {q}\nAnswer: {a}"
    prompts = [f"Question: {q}\nAnswer:" for q in questions]
    full_texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)]

    # Tokenize full texts (for training)
    full_encodings = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Create labels: -100 for prompt tokens, actual tokens for answer
    # We need to find where each prompt ends in the tokenized full text
    labels = full_encodings["input_ids"].clone()

    for i, (prompt, full_text) in enumerate(zip(prompts, full_texts)):
        # Tokenize prompt without padding to get actual length
        prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        prompt_len = len(prompt_ids)

        # Mask prompt tokens in labels (set to -100 to ignore in loss)
        labels[i, :prompt_len] = -100

        # Also mask padding tokens
        pad_mask = full_encodings["attention_mask"][i] == 0
        labels[i, pad_mask] = -100

    # Tokenize prompts for generation (separate)
    prompt_encodings = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": full_encodings["input_ids"].to(device),
        "attention_mask": full_encodings["attention_mask"].to(device),
        "labels": labels.to(device),
        "prompt_input_ids": prompt_encodings["input_ids"].to(device),
        "prompt_attention_mask": prompt_encodings["attention_mask"].to(device),
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, args, epoch, tokenizer):
    """Train one epoch."""
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        if not batch:
            continue

        # Prepare inputs
        qa_inputs = prepare_qa_inputs(batch, tokenizer, device)

        kwargs = {
            "input_ids": qa_inputs["input_ids"],
            "attention_mask": qa_inputs["attention_mask"],
            "labels": qa_inputs["labels"],
        }

        if args.modality in ["pointcloud", "both"] and batch.get("pointclouds") is not None:
            kwargs["pointclouds"] = batch["pointclouds"].to(device)
        if args.modality in ["image", "both"] and batch.get("images") is not None:
            kwargs["images"] = batch["images"]

        # Forward
        outputs = model(**kwargs)
        loss = outputs["loss"]

        if loss is None:
            continue

        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * args.gradient_accumulation_steps
        num_batches += 1

        pbar.set_postfix({"loss": total_loss / num_batches})

    return {"loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def evaluate(model, dataloader, device, args, tokenizer, max_samples=None):
    """Evaluate model with generation."""
    model.eval()

    all_predictions = []
    all_references = []

    num_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        if not batch:
            continue
        if max_samples and num_samples >= max_samples:
            break

        # Prepare inputs for generation
        questions = batch["questions"]
        prompts = [f"Question: {q}\nAnswer:" for q in questions]

        prompt_encodings = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        kwargs = {
            "input_ids": prompt_encodings["input_ids"].to(device),
            "attention_mask": prompt_encodings["attention_mask"].to(device),
            "max_new_tokens": args.max_answer_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }

        if args.modality in ["pointcloud", "both"] and batch.get("pointclouds") is not None:
            kwargs["pointclouds"] = batch["pointclouds"].to(device)
        if args.modality in ["image", "both"] and batch.get("images") is not None:
            kwargs["images"] = batch["images"]

        # Generate
        output_ids = model.generate(**kwargs)

        # Decode predictions
        for i, ids in enumerate(output_ids):
            # Get only the generated part
            prompt_len = prompt_encodings["attention_mask"][i].sum()
            generated_ids = ids[prompt_len:]
            pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            all_predictions.append(pred)
            all_references.append(batch["all_answers"][i])

        num_samples += len(batch["questions"])

    # Compute metrics
    metrics = compute_qa_metrics(all_predictions, all_references)

    return metrics, all_predictions, all_references


def main():
    args = parse_args()

    print("=" * 60)
    print("ScanQA Composition Training")
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
    train_dataset = ScanQADataset(
        args.data_path,
        split="train",
        modality=args.modality,
        num_points=args.num_points,
    )
    val_dataset = ScanQADataset(
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
        collate_fn=collate_scanqa_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_scanqa_batch,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print("\nCreating model...")
    model = ScanQACompositionModel(
        modality=args.modality,
        llm_model_name=args.llm_model,
        pointcloud_encoder_checkpoint=args.encoder_checkpoint,
        num_tokens=args.num_pointcloud_tokens,
        fusion_layer_indices=fusion_layers,
        freeze_llm=args.freeze_llm,
        unfreeze_encoder_last_n=args.unfreeze_encoder_last_n,
    )
    model = model.to(args.device)

    if args.fp16:
        model = model.half()

    # Get tokenizer
    if args.modality in ["pointcloud", "both"]:
        tokenizer = model.safe_model.base_vl.tokenizer
    else:
        tokenizer = model.processor.tokenizer

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Optimizer
    trainable_params = model.get_trainable_params()
    optimizer = AdamW(trainable_params, lr=args.safe_lr, weight_decay=0.01)

    # Scheduler
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        if args.lr_scheduler == "constant":
            return 1.0
        else:  # cosine
            progress = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

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
    best_bleu4 = 0.0

    for epoch in range(args.num_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            args.device, args, epoch, tokenizer
        )

        print(f"\nEpoch {epoch+1} - Train Loss: {train_metrics['loss']:.4f}")

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            val_metrics, predictions, references = evaluate(
                model, val_loader, args.device, args,
                tokenizer, args.max_eval_samples
            )

            print(f"Val Metrics:")
            print(f"  BLEU-1: {val_metrics['bleu1']:.2f}")
            print(f"  BLEU-4: {val_metrics['bleu4']:.2f}")
            print(f"  METEOR: {val_metrics['meteor']:.2f}")
            print(f"  Exact Match: {val_metrics['exact_match']:.2f}")

            # Show some examples
            print("\nSample predictions:")
            for i in range(min(3, len(predictions))):
                print(f"  Pred: {predictions[i]}")
                print(f"  Refs: {references[i][:2]}")
                print()

            # Save best
            if val_metrics["bleu4"] > best_bleu4:
                best_bleu4 = val_metrics["bleu4"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_bleu4": best_bleu4,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                }, Path(args.output_dir) / "best_model.pt")
                print(f"New best! Saved checkpoint (BLEU-4: {best_bleu4:.2f})")

            if args.wandb and wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "val_bleu1": val_metrics["bleu1"],
                    "val_bleu4": val_metrics["bleu4"],
                    "val_meteor": val_metrics["meteor"],
                    "val_exact_match": val_metrics["exact_match"],
                    "best_bleu4": best_bleu4,
                    "lr": scheduler.get_last_lr()[0],
                })

    print(f"\nTraining complete! Best BLEU-4: {best_bleu4:.2f}")

    if args.wandb and wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
