#!/usr/bin/env python3
"""
train_safe.py - Clean, optimized training script for SAFE audio captioning

Focuses on essentials:
- Audio captioning task (cross-entropy loss)
- CIDEr/BLEU evaluation metrics
- Memory-efficient training (gradient accumulation, mixed precision)
- Clear result tracking

Removed:
- Retention loss mechanisms (no longer needed)
- Curriculum learning
- Null-space projection
- SCST fine-tuning
- Complex dataset mixing
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

# SAFE imports
from configs.model_configs import get_config
from safe.data.datasets import AudioCapsDataset, create_safe_dataloader
from safe.models.safe_model import SAFEModel


# ============================================================================
# SECTION 1: UTILITIES
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# SECTION 2: METRICS
# ============================================================================

def compute_cider(predictions: List[str], references: List[List[str]]) -> float:
    """
    Compute CIDEr score using pycocoevalcap

    Args:
        predictions: List of predicted captions
        references: List of reference caption lists (multiple refs per sample)

    Returns:
        CIDEr score (0-100 scale)
    """
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        print("⚠️  pycocoevalcap not available, CIDEr score will be 0.0")
        return 0.0

    # Convert to pycocoevalcap format: {id: [captions]}
    gts = {i: refs for i, refs in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}

    try:
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts, res)
        return float(score) * 100  # Scale to 0-100
    except Exception as e:
        print(f"⚠️  CIDEr computation failed: {e}")
        return 0.0


def compute_bleu(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Compute BLEU scores using pycocoevalcap

    Returns:
        Dict with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
    except ImportError:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    gts = {i: refs for i, refs in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}

    try:
        bleu_scorer = Bleu(4)
        scores, _ = bleu_scorer.compute_score(gts, res)
        return {
            "bleu1": float(scores[0]),
            "bleu2": float(scores[1]),
            "bleu3": float(scores[2]),
            "bleu4": float(scores[3]),
        }
    except Exception:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}


def compute_meteor(predictions: List[str], references: List[List[str]]) -> float:
    """Compute METEOR score"""
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        meteor_scorer = Meteor()
        score, _ = meteor_scorer.compute_score(gts, res)
        return float(score)
    except Exception:
        return 0.0


def compute_rouge(predictions: List[str], references: List[List[str]]) -> float:
    """Compute ROUGE-L score"""
    try:
        from pycocoevalcap.rouge.rouge import Rouge
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        rouge_scorer = Rouge()
        score, _ = rouge_scorer.compute_score(gts, res)
        return float(score)
    except Exception:
        return 0.0


def compute_caption_metrics(
    predictions: List[str],
    references: List[List[str]],
    compute_bertscore: bool = False
) -> Dict[str, float]:
    """
    Compute all caption metrics

    Args:
        predictions: List of predicted captions
        references: List of reference caption lists
        compute_bertscore: Whether to compute BERTScore (slow)

    Returns:
        Dict with all metrics
    """
    metrics = {}

    # CIDEr (most important for audio captioning)
    metrics["cider"] = compute_cider(predictions, references)

    # BLEU scores
    bleu_scores = compute_bleu(predictions, references)
    metrics.update(bleu_scores)

    # METEOR and ROUGE
    metrics["meteor"] = compute_meteor(predictions, references)
    metrics["rouge_l"] = compute_rouge(predictions, references)

    # BERTScore (optional, slow)
    if compute_bertscore:
        try:
            from bert_score import score as bert_score_fn
            _, _, F1 = bert_score_fn(
                predictions,
                [refs[0] for refs in references],  # Use first reference
                lang="en",
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=32
            )
            metrics["bertscore_f1"] = float(F1.mean())
        except Exception as e:
            print(f"⚠️  BERTScore computation failed: {e}")
            metrics["bertscore_f1"] = 0.0

    return metrics


# ============================================================================
# SECTION 3: EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate(
    model: SAFEModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    max_new_tokens: int = 20,
    num_beams: int = 1,
    compute_bertscore: bool = False,
) -> Dict[str, float]:
    """
    Evaluate model on audio captioning task

    Args:
        model: SAFE model
        dataloader: Validation dataloader
        device: Device to run on
        max_batches: Maximum batches to evaluate (None = all)
        max_new_tokens: Max tokens to generate
        num_beams: Beam search size
        compute_bertscore: Whether to compute BERTScore

    Returns:
        Dict with metrics (loss, cider, bleu4, etc.)
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_references = []

    print(f"Running evaluation (max_batches={max_batches})...", flush=True)
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Move batch to device
        questions = batch["questions"]
        answers = batch["answers"]
        audio = batch["audio"]

        # Prepare inputs
        inputs = model.prepare_multimodal_inputs(
            text=questions,
            audio=audio,
            answers=answers,
            training_mode=True  # For loss computation
        )

        # Move inputs to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs["labels"].to(device)
        audio_tokens = inputs.get("audio_tokens")
        if audio_tokens is not None:
            audio_tokens = audio_tokens.to(device)

        # Compute loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            audio_tokens=audio_tokens,
        )

        loss = outputs.get("loss")
        if loss is not None:
            total_loss += loss.item()
            num_batches += 1

        # Generate predictions
        generation_inputs = model.prepare_multimodal_inputs(
            text=questions,
            audio=audio,
            answers=None,  # No answers for generation
            training_mode=False
        )

        gen_input_ids = generation_inputs["input_ids"].to(device)
        gen_attention_mask = generation_inputs["attention_mask"].to(device)
        gen_audio_tokens = generation_inputs.get("audio_tokens")
        if gen_audio_tokens is not None:
            gen_audio_tokens = gen_audio_tokens.to(device)

        # Generate captions
        generated_ids = model.generate(
            input_ids=gen_input_ids,
            attention_mask=gen_attention_mask,
            audio_tokens=gen_audio_tokens,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=False,
        )

        # Decode predictions
        tokenizer = model.base_vl.tokenizer
        batch_predictions = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Clean predictions (remove question prompt)
        for i, pred in enumerate(batch_predictions):
            # Remove the question from the prediction
            question = questions[i]
            if question and question in pred:
                pred = pred.replace(question, "").strip()
            all_predictions.append(pred)

        # Collect references
        for answer in answers:
            if isinstance(answer, str):
                refs = [answer]
            elif isinstance(answer, list):
                refs = [str(a) for a in answer]
            else:
                refs = [str(answer)]
            all_references.append(refs)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Evaluated {batch_idx + 1} batches...", flush=True)

    elapsed = time.time() - start_time

    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    caption_metrics = compute_caption_metrics(
        all_predictions,
        all_references,
        compute_bertscore=compute_bertscore
    )

    metrics = {
        "loss": avg_loss,
        **caption_metrics,
        "num_samples": len(all_predictions),
        "eval_time": elapsed,
    }

    print(f"✓ Evaluation complete ({format_time(elapsed)})", flush=True)
    print(f"  Loss: {avg_loss:.4f}", flush=True)
    print(f"  CIDEr: {metrics['cider']:.2f}", flush=True)
    print(f"  BLEU-4: {metrics['bleu4']:.4f}", flush=True)

    # Log sample predictions
    print(f"\n📝 Sample predictions:", flush=True)
    for i in range(min(3, len(all_predictions))):
        print(f"  [{i+1}] Pred: {all_predictions[i]}", flush=True)
        print(f"      Refs: {all_references[i]}", flush=True)

    return metrics


# ============================================================================
# SECTION 4: TRAINING
# ============================================================================

def train_epoch(
    model: SAFEModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """
    Train for one epoch

    Args:
        model: SAFE model
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        config: Training configuration
        scaler: GradScaler for mixed precision (optional)

    Returns:
        Dict with training metrics
    """
    model.train()

    total_loss = 0.0
    num_batches = 0
    num_samples = 0

    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    use_amp = config.get("fp16", False) and scaler is not None

    optimizer.zero_grad()

    start_time = time.time()
    last_log_time = start_time

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        questions = batch["questions"]
        answers = batch["answers"]
        audio = batch["audio"]

        # Prepare inputs
        inputs = model.prepare_multimodal_inputs(
            text=questions,
            audio=audio,
            answers=answers,
            training_mode=True
        )

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs["labels"].to(device)
        audio_tokens = inputs.get("audio_tokens")
        if audio_tokens is not None:
            audio_tokens = audio_tokens.to(device)

        # Forward pass with optional mixed precision
        if use_amp:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    audio_tokens=audio_tokens,
                )
                loss = outputs["loss"]
                loss = loss / gradient_accumulation_steps
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                audio_tokens=audio_tokens,
            )
            loss = outputs["loss"]
            loss = loss / gradient_accumulation_steps

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if use_amp:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        # Logging
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        num_samples += len(questions)

        # Periodic logging
        current_time = time.time()
        if current_time - last_log_time > 60:  # Log every minute
            avg_loss = total_loss / num_batches
            elapsed = current_time - start_time
            samples_per_sec = num_samples / elapsed
            lr = scheduler.get_last_lr()[0]

            print(
                f"[Epoch {epoch}] Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                f"Speed: {samples_per_sec:.1f} samples/s",
                flush=True
            )
            last_log_time = current_time

    # Final statistics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    elapsed = time.time() - start_time

    metrics = {
        "loss": avg_loss,
        "num_samples": num_samples,
        "train_time": elapsed,
        "samples_per_sec": num_samples / elapsed,
    }

    return metrics


def train(
    model: SAFEModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Main training loop

    Args:
        model: SAFE model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration
        output_dir: Output directory for checkpoints
        device: Device to train on

    Returns:
        Training history dict
    """
    # Setup optimizer with different learning rates
    lr_projector = config.get("learning_rate_projector", 1e-3)
    lr_adapter = config.get("learning_rate_adapter", 5e-4)
    weight_decay = config.get("weight_decay", 0.01)

    # Group parameters by component
    projector_params = []
    adapter_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "audio_projector" in name:
            projector_params.append(param)
        elif "fusion_adapter" in name or "lora" in name.lower():
            adapter_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": projector_params, "lr": lr_projector, "name": "projector"},
        {"params": adapter_params, "lr": lr_adapter, "name": "adapter"},
    ]

    if other_params:
        param_groups.append({"params": other_params, "lr": lr_adapter, "name": "other"})

    optimizer = AdamW(param_groups, weight_decay=weight_decay)

    # Learning rate scheduler
    num_epochs = config.get("num_epochs", 20)
    warmup_steps = config.get("warmup_steps", 500)
    total_steps = len(train_loader) * num_epochs

    # Cosine schedule with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = GradScaler() if config.get("fp16", False) else None

    # Training loop
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_cider": [],
        "val_bleu4": [],
        "epochs": [],
    }

    best_cider = 0.0
    patience = config.get("early_stopping_patience", 5)
    patience_counter = 0

    print(f"\n{'='*80}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"  Projector LR: {lr_projector}")
    print(f"  Adapter LR: {lr_adapter}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Mixed precision: {config.get('fp16', False)}")
    print(f"{'='*80}\n")

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}\n")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, config, scaler
        )

        print(f"\n✓ Training complete:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Time: {format_time(train_metrics['train_time'])}")
        print(f"  Speed: {train_metrics['samples_per_sec']:.1f} samples/sec")

        # Evaluate
        eval_frequency = config.get("eval_frequency", 1)
        if epoch % eval_frequency == 0:
            print(f"\n{'='*80}")
            print(f"Running validation...")
            print(f"{'='*80}\n")

            val_metrics = evaluate(
                model,
                val_loader,
                device,
                max_batches=config.get("max_eval_batches"),
                max_new_tokens=config.get("max_new_tokens", 20),
                num_beams=config.get("num_beams", 1),
                compute_bertscore=False,
            )

            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_cider"].append(val_metrics["cider"])
            history["val_bleu4"].append(val_metrics["bleu4"])
            history["epochs"].append(epoch)

            # Save checkpoint
            is_best = val_metrics["cider"] > best_cider
            if is_best:
                best_cider = val_metrics["cider"]
                patience_counter = 0
                print(f"\n🎉 New best CIDEr: {best_cider:.2f}")
            else:
                patience_counter += 1

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                {**train_metrics, **val_metrics, "epoch": epoch},
                output_dir,
                is_best=is_best,
            )

            # Save history
            with open(output_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered (patience={patience})")
                break

    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"  Best CIDEr: {best_cider:.2f}")
    print(f"  Checkpoints saved to: {output_dir}")
    print(f"{'='*80}\n")

    return history


# ============================================================================
# SECTION 5: CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(
    model: SAFEModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, float],
    output_dir: Path,
    is_best: bool = False,
):
    """Save model checkpoint"""
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }

    # Save last checkpoint
    checkpoint_path = output_dir / "checkpoint_last.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Saved checkpoint: {checkpoint_path}")

    # Save best checkpoint
    if is_best:
        best_path = output_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"💾 Saved BEST checkpoint: {best_path}")


def load_checkpoint(
    model: SAFEModel,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, float]:
    """Load model checkpoint"""
    print(f"📂 Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    metrics = checkpoint.get("metrics", {})
    print(f"✓ Checkpoint loaded (epoch={metrics.get('epoch', 'unknown')})")

    return metrics


# ============================================================================
# SECTION 6: MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SAFE model on audio captioning")

    # Model configuration
    parser.add_argument("--model-config", type=str, default="phase1",
                        choices=["demo", "full", "multimodal", "phase1"],
                        help="Model configuration name")

    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--train-split", type=str, default="train",
                        help="Training split name")
    parser.add_argument("--val-split", type=str, default="val",
                        help="Validation split name")

    # Training
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--num-epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=8,
                        help="Validation batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32,
                        help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")

    # Optimization
    parser.add_argument("--learning-rate-projector", type=float, default=1e-3,
                        help="Learning rate for audio projector")
    parser.add_argument("--learning-rate-adapter", type=float, default=5e-4,
                        help="Learning rate for fusion adapter")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Warmup steps for learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")

    # Evaluation
    parser.add_argument("--eval-frequency", type=int, default=1,
                        help="Evaluate every N epochs")
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="Max batches for validation (None = all)")
    parser.add_argument("--max-new-tokens", type=int, default=20,
                        help="Max new tokens for generation")
    parser.add_argument("--num-beams", type=int, default=1,
                        help="Beam search size")

    # Checkpointing
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only (requires --resume)")
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Early stopping patience (epochs)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model config
    print(f"\nLoading model config: {args.model_config}")
    model_config = get_config(args.model_config)

    # Initialize model
    print(f"\nInitializing SAFE model...")
    model = SAFEModel(**model_config)
    model = model.to(device)

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\n📊 Model parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Load datasets
    print(f"\n📂 Loading datasets from: {args.data_path}")
    train_dataset = AudioCapsDataset(args.data_path, split=args.train_split)
    val_dataset = AudioCapsDataset(args.data_path, split=args.val_split)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = create_safe_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = create_safe_dataloader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Training config
    config = {
        "num_epochs": args.num_epochs,
        "learning_rate_projector": args.learning_rate_projector,
        "learning_rate_adapter": args.learning_rate_adapter,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "fp16": args.fp16,
        "eval_frequency": args.eval_frequency,
        "max_eval_batches": args.max_eval_batches,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "early_stopping_patience": args.early_stopping_patience,
    }

    # Evaluation only mode
    if args.eval_only:
        if args.resume is None:
            raise ValueError("--eval-only requires --resume")

        load_checkpoint(model, None, None, Path(args.resume), device)

        print(f"\n{'='*80}")
        print(f"Running evaluation only")
        print(f"{'='*80}\n")

        metrics = evaluate(
            model,
            val_loader,
            device,
            max_batches=args.max_eval_batches,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            compute_bertscore=True,  # Compute BERTScore in eval-only mode
        )

        # Save results
        results_path = output_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n💾 Results saved to: {results_path}")
        return

    # Resume from checkpoint if specified
    if args.resume:
        load_checkpoint(model, None, None, Path(args.resume), device)

    # Train
    train(
        model,
        train_loader,
        val_loader,
        config,
        output_dir,
        device,
    )

    print(f"\n✅ Training complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
