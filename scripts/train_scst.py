#!/usr/bin/env python3
"""
SCST (Self-Critical Sequence Training) for Audio Captioning

This script fine-tunes a trained SAFE model using reinforcement learning
with CIDEr as the reward signal. It loads any checkpoint and trains on
AudioCaps data only.

Usage:
    python scripts/train_scst.py \
        --checkpoint checkpoints/baseline/checkpoint_best.pt \
        --fusion-layer-indices "12,24,36" \
        --data-path experiments/full_training/data \
        --output-dir checkpoints/scst

Algorithm:
    For each batch:
    1. Generate sampled caption (do_sample=True)
    2. Generate greedy caption (baseline)
    3. Compute reward: CIDEr(sampled) - CIDEr(greedy)
    4. Compute log_prob of sampled tokens
    5. Loss = -reward * log_prob (REINFORCE policy gradient)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_configs import get_config
from train_safe import (
    create_model,
    load_checkpoint,
    save_checkpoint,
    compute_caption_metrics,
    set_seed,
)
from safe.data.datasets import AudioCapsDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="SCST fine-tuning for audio captioning"
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint to fine-tune",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to data directory containing AudioCaps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save SCST checkpoints",
    )

    # Model configuration
    parser.add_argument(
        "--fusion-layer-indices",
        type=str,
        default=None,
        help="Comma-separated fusion layer indices (must match checkpoint)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="phase1",
        help="Base model config name",
    )
    parser.add_argument(
        "--num-audio-tokens",
        type=int,
        default=None,
        help="Number of audio tokens (must match checkpoint)",
    )
    parser.add_argument(
        "--bottleneck-dim",
        type=int,
        default=None,
        help="Bottleneck dimension for fusion adapter (must match checkpoint)",
    )

    # SCST hyperparameters
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of SCST epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (should be lower than stage 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for caption generation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples per input for variance reduction",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum new tokens to generate",
    )

    # Training configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (keep small, SCST is memory-heavy)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Evaluation
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=1,
        help="Evaluate every N epochs",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=500,
        help="Max samples for evaluation",
    )

    # Wandb logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="SAFE-SCST",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Wandb run name",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Wandb mode",
    )

    return parser.parse_args()


def collate_fn(batch: List[Dict]) -> Dict[str, List]:
    """Collate batch of samples."""
    questions = []
    audio_paths = []
    answers = []

    for idx, sample in enumerate(batch):
        # Build question prompt
        question = sample.get("question", "Describe this audio.")
        questions.append(question)

        # Get audio path - handle both tensor and path formats
        audio = sample.get("audio")
        audio_path = sample.get("audio_path")
        if audio is not None and not isinstance(audio, str):
            # Audio is already loaded as tensor/waveform
            audio_paths.append(audio)
        else:
            audio_paths.append(audio_path or audio)

        # Get reference captions (may be list) - try multiple field names
        # AudioCaps uses "answers" which should be a list of 5 captions
        ans = sample.get("answers")
        if ans is None:
            ans = sample.get("captions")
        if ans is None:
            ans = sample.get("answer")
        if ans is None:
            ans = sample.get("caption")

        # Ensure it's a list
        if ans is None:
            ans = []
        elif isinstance(ans, str):
            ans = [ans]
        elif not isinstance(ans, (list, tuple)):
            ans = [str(ans)]


        answers.append(ans)

    return {
        "questions": questions,
        "audio": audio_paths,
        "answers": answers,
    }


def generate_greedy(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    audio_tokens: torch.Tensor,
    audio_attention_mask: Optional[torch.Tensor],
    max_new_tokens: int = 20,
) -> torch.Tensor:
    """Generate caption using greedy decoding (baseline)."""
    with torch.no_grad():
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_tokens=audio_tokens,
            audio_attention_mask=audio_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=model.base_vl.tokenizer.pad_token_id,
            eos_token_id=model.base_vl.tokenizer.eos_token_id,
        )


def generate_sampled(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    audio_tokens: torch.Tensor,
    audio_attention_mask: Optional[torch.Tensor],
    max_new_tokens: int = 20,
    temperature: float = 0.7,
) -> torch.Tensor:
    """Generate caption using sampling for SCST."""
    with torch.no_grad():
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_tokens=audio_tokens,
            audio_attention_mask=audio_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_beams=1,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=model.base_vl.tokenizer.pad_token_id,
            eos_token_id=model.base_vl.tokenizer.eos_token_id,
        )


def compute_per_sample_cider(
    predictions: List[str],
    references: List[List[str]],
) -> List[float]:
    """Compute CIDEr score for each sample individually."""
    import io
    import sys

    scores = []
    for pred, refs in zip(predictions, references):
        try:
            # Suppress pycocoevalcap verbose output
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                metrics = compute_caption_metrics([pred], [refs], light_metrics=True, quiet=True)
                scores.append(metrics.get("cider", 0.0))
            finally:
                sys.stdout = old_stdout
        except Exception:
            scores.append(0.0)
    return scores


def compute_scst_reward(
    sampled_captions: List[str],
    baseline_captions: List[str],
    references: List[List[str]],
) -> torch.Tensor:
    """
    Compute SCST reward: CIDEr(sampled) - CIDEr(baseline)

    Returns per-sample rewards as a tensor.
    """
    sampled_scores = compute_per_sample_cider(sampled_captions, references)
    baseline_scores = compute_per_sample_cider(baseline_captions, references)

    rewards = [s - b for s, b in zip(sampled_scores, baseline_scores)]
    return torch.tensor(rewards, dtype=torch.float32)


def compute_log_probs(
    model,
    prompt_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    audio_tokens: torch.Tensor,
    audio_attention_mask: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute log probability of generated tokens given prompt + audio.

    Args:
        prompt_ids: Input prompt token IDs [B, prompt_len]
        generated_ids: Full generated sequence [B, total_len]
        audio_tokens: Audio token embeddings
        audio_attention_mask: Audio attention mask

    Returns:
        Log probability (scalar tensor with gradients)
    """
    prompt_len = prompt_ids.size(1)

    # Extract only the generated part (after prompt)
    target_ids = generated_ids[:, prompt_len:]

    if target_ids.size(1) == 0:
        # No tokens generated
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Full sequence for forward pass
    full_ids = generated_ids
    full_mask = torch.ones_like(full_ids)

    # Labels: -100 for prompt (ignored), actual IDs for generated part
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100

    # Forward pass with gradients
    outputs = model(
        input_ids=full_ids,
        attention_mask=full_mask,
        audio_tokens=audio_tokens,
        audio_attention_mask=audio_attention_mask,
        labels=labels,
    )

    # Loss is -log_prob averaged over tokens
    # So -loss gives us the mean log_prob
    return -outputs["loss"]


def scst_train_step(
    model,
    batch: Dict,
    tokenizer,
    device: torch.device,
    temperature: float = 0.7,
    num_samples: int = 5,
    max_new_tokens: int = 20,
    use_amp: bool = False,
) -> Dict[str, float]:
    """
    Single SCST training step.

    Returns dict with loss and reward metrics.
    """
    questions = batch["questions"]
    audio = batch["audio"]
    references = batch["answers"]

    # Prepare multimodal inputs
    inputs = model.prepare_multimodal_inputs(
        text=questions,
        audio=audio,
        device=str(device),
        training_mode=False,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    audio_tokens = inputs["audio_tokens"]
    audio_attention_mask = inputs.get("audio_attention_mask")

    # Generate greedy baseline (no gradients needed)
    greedy_ids = generate_greedy(
        model,
        input_ids,
        attention_mask,
        audio_tokens,
        audio_attention_mask,
        max_new_tokens=max_new_tokens,
    )
    greedy_captions = tokenizer.batch_decode(greedy_ids, skip_special_tokens=True)

    # Clean up greedy captions
    greedy_captions = [cap.strip() for cap in greedy_captions]

    # Sample multiple captions and accumulate policy gradient
    total_loss = torch.tensor(0.0, device=device)
    total_reward = 0.0
    total_sampled_cider = 0.0
    total_baseline_cider = 0.0

    for sample_idx in range(num_samples):
        # Generate sampled caption
        sampled_ids = generate_sampled(
            model,
            input_ids,
            attention_mask,
            audio_tokens,
            audio_attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        sampled_captions = tokenizer.batch_decode(sampled_ids, skip_special_tokens=True)
        sampled_captions = [cap.strip() for cap in sampled_captions]

        # Compute reward
        rewards = compute_scst_reward(sampled_captions, greedy_captions, references)
        rewards = rewards.to(device)

        # Track CIDEr scores
        sampled_scores = compute_per_sample_cider(sampled_captions, references)
        baseline_scores = compute_per_sample_cider(greedy_captions, references)
        total_sampled_cider += sum(sampled_scores) / len(sampled_scores)
        total_baseline_cider += sum(baseline_scores) / len(baseline_scores)

        # Compute log probabilities (with gradients)
        if use_amp:
            with autocast():
                log_probs = compute_log_probs(
                    model,
                    input_ids,
                    sampled_ids,
                    audio_tokens,
                    audio_attention_mask,
                    device,
                )
        else:
            log_probs = compute_log_probs(
                model,
                input_ids,
                sampled_ids,
                audio_tokens,
                audio_attention_mask,
                device,
            )

        # Policy gradient loss: -reward * log_prob
        # We want to maximize reward, so minimize negative reward * log_prob
        batch_reward = rewards.mean()
        policy_loss = -batch_reward * log_probs

        total_loss = total_loss + policy_loss
        total_reward += batch_reward.item()

    # Average over samples
    avg_loss = total_loss / num_samples
    avg_reward = total_reward / num_samples
    avg_sampled_cider = total_sampled_cider / num_samples
    avg_baseline_cider = total_baseline_cider / num_samples

    return {
        "loss": avg_loss,
        "reward": avg_reward,
        "sampled_cider": avg_sampled_cider,
        "baseline_cider": avg_baseline_cider,
    }


def evaluate(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    max_samples: int = 500,
    max_new_tokens: int = 20,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()

    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(all_predictions) >= max_samples:
                break

            questions = batch["questions"]
            audio = batch["audio"]
            references = batch["answers"]

            # Prepare inputs
            inputs = model.prepare_multimodal_inputs(
                text=questions,
                audio=audio,
                device=str(device),
                training_mode=False,
            )

            # Generate with beam search for evaluation
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                audio_tokens=inputs["audio_tokens"],
                audio_attention_mask=inputs.get("audio_attention_mask"),
                max_new_tokens=max_new_tokens,
                num_beams=5,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

            captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            captions = [cap.strip() for cap in captions]

            all_predictions.extend(captions)
            all_references.extend(references)

    # Compute metrics
    all_predictions = all_predictions[:max_samples]
    all_references = all_references[:max_samples]

    metrics = compute_caption_metrics(all_predictions, all_references)

    model.train()
    return metrics


def main():
    args = parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SCST Training for Audio Captioning")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Temperature: {args.temperature}")
    print(f"Num samples: {args.num_samples}")
    print(f"Learning rate: {args.lr}")
    print(f"Wandb: {args.wandb}")
    print("=" * 60)

    # Initialize wandb
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_name = args.wandb_name or f"scst_{Path(args.checkpoint).stem}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            mode=args.wandb_mode,
            config={
                "checkpoint": args.checkpoint,
                "temperature": args.temperature,
                "num_samples": args.num_samples,
                "lr": args.lr,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "max_new_tokens": args.max_new_tokens,
                "fusion_layer_indices": args.fusion_layer_indices,
                "num_audio_tokens": args.num_audio_tokens,
                "bottleneck_dim": args.bottleneck_dim,
            },
        )
        print(f"  Wandb run: {wandb_run.url}")

    # Load model config
    print("\n[1/5] Loading model configuration...")
    model_config = get_config(args.config)

    # Override fusion layer indices if specified
    if args.fusion_layer_indices:
        layer_indices = [int(x.strip()) for x in args.fusion_layer_indices.split(",")]
        model_config["fusion_layer_indices"] = layer_indices

        # Update nested configs
        if "fusion_config" in model_config:
            fusion_cfg = model_config["fusion_config"]
            if "modalities" in fusion_cfg:
                for mod_cfg in fusion_cfg["modalities"].values():
                    if isinstance(mod_cfg, dict):
                        mod_cfg["layer_indices"] = layer_indices

        print(f"  Fusion layer indices: {layer_indices}")

    # Override num_audio_tokens if specified
    if args.num_audio_tokens is not None:
        model_config["num_audio_tokens"] = args.num_audio_tokens
        if "fusion_config" in model_config:
            fusion_cfg = model_config["fusion_config"]
            if "modalities" in fusion_cfg:
                for mod_cfg in fusion_cfg["modalities"].values():
                    if isinstance(mod_cfg, dict):
                        mod_cfg["num_tokens"] = args.num_audio_tokens
        print(f"  Num audio tokens: {args.num_audio_tokens}")

    # Override bottleneck_dim if specified
    if args.bottleneck_dim is not None:
        if "fusion_config" in model_config:
            model_config["fusion_config"]["bottleneck_dim"] = args.bottleneck_dim
        print(f"  Bottleneck dim: {args.bottleneck_dim}")

    # Create model
    print("\n[2/5] Creating model...")
    model = create_model(model_config)
    model = model.to(device)

    # Load checkpoint
    print(f"\n[3/5] Loading checkpoint: {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_checkpoint(model, None, None, checkpoint_path, device)
    print("  Checkpoint loaded successfully")

    tokenizer = model.base_vl.tokenizer

    # Load datasets
    print("\n[4/5] Loading AudioCaps dataset...")
    train_dataset = AudioCapsDataset(args.data_path, split="train")
    val_dataset = AudioCapsDataset(args.data_path, split="val")

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Check reference count in first few samples
    sample = train_dataset[0]
    ans = sample.get("answers") or sample.get("captions") or sample.get("answer")
    if isinstance(ans, str):
        ref_count = 1
    elif isinstance(ans, (list, tuple)):
        ref_count = len(ans)
    else:
        ref_count = 1
    print(f"  References per sample (sample 0): {ref_count}")
    if ref_count < 5:
        print(f"  ⚠️  WARNING: Expected ~5 refs/sample for AudioCaps. Check data format.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Setup optimizer and scheduler
    print("\n[5/5] Setting up optimizer...")
    trainable_params = list(model.get_trainable_parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = (len(train_loader) // args.gradient_accumulation) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )

    scaler = GradScaler() if args.fp16 else None

    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"  Total training steps: {total_steps}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting SCST Training")
    print("=" * 60)

    best_cider = 0.0
    optimizer_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_reward = 0.0
        epoch_sampled_cider = 0.0
        epoch_baseline_cider = 0.0
        num_batches = 0

        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # SCST training step
            metrics = scst_train_step(
                model,
                batch,
                tokenizer,
                device,
                temperature=args.temperature,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                use_amp=args.fp16,
            )

            loss = metrics["loss"]

            # Normalize by gradient accumulation
            loss = loss / args.gradient_accumulation

            # Backward
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += metrics["loss"].item() if torch.is_tensor(metrics["loss"]) else metrics["loss"]
            epoch_reward += metrics["reward"]
            epoch_sampled_cider += metrics["sampled_cider"]
            epoch_baseline_cider += metrics["baseline_cider"]
            num_batches += 1

            # Progress logging every batch
            if batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx}/{len(train_loader)} ({elapsed:.1f}s) - "
                      f"reward={metrics['reward']:.3f}, sampled_cider={metrics['sampled_cider']:.1f}", flush=True)

            # Optimizer step
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                # Log progress (every step, not just every 10)
                if optimizer_step % 1 == 0:
                    avg_loss = epoch_loss / num_batches
                    avg_reward = epoch_reward / num_batches
                    avg_samp = epoch_sampled_cider / num_batches
                    avg_base = epoch_baseline_cider / num_batches
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"  Step {optimizer_step}: loss={avg_loss:.4f}, "
                        f"reward={avg_reward:.4f}, sampled_cider={avg_samp:.2f}, "
                        f"baseline_cider={avg_base:.2f}, lr={lr:.2e}"
                    )

                    # Wandb logging
                    if wandb_run is not None:
                        wandb_run.log({
                            "train/loss": avg_loss,
                            "train/reward": avg_reward,
                            "train/sampled_cider": avg_samp,
                            "train/baseline_cider": avg_base,
                            "train/lr": lr,
                            "train/step": optimizer_step,
                        })

        # Epoch summary
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / num_batches
        avg_reward = epoch_reward / num_batches
        avg_sampled_cider = epoch_sampled_cider / num_batches
        avg_baseline_cider = epoch_baseline_cider / num_batches

        print(f"\nEpoch {epoch + 1}/{args.num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train loss: {avg_loss:.4f}")
        print(f"  Train reward: {avg_reward:.4f}")
        print(f"  Sampled CIDEr: {avg_sampled_cider:.2f}")
        print(f"  Baseline CIDEr: {avg_baseline_cider:.2f}")

        # Evaluation
        if (epoch + 1) % args.eval_frequency == 0:
            print("\n  Evaluating...")
            val_metrics = evaluate(
                model,
                val_loader,
                tokenizer,
                device,
                max_samples=args.max_eval_samples,
                max_new_tokens=args.max_new_tokens,
            )

            val_cider = val_metrics.get("cider", 0.0)
            val_meteor = val_metrics.get("meteor", 0.0)
            val_bleu4 = val_metrics.get("bleu4", 0.0)

            print(f"  Val CIDEr: {val_cider:.2f}")
            print(f"  Val METEOR: {val_meteor:.4f}")
            print(f"  Val BLEU-4: {val_bleu4:.4f}")

            # Wandb logging for validation
            if wandb_run is not None:
                wandb_run.log({
                    "val/cider": val_cider,
                    "val/meteor": val_meteor,
                    "val/bleu4": val_bleu4,
                    "epoch": epoch + 1,
                })

            # Save checkpoint
            is_best = val_cider > best_cider
            if is_best:
                best_cider = val_cider

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                {
                    "epoch": epoch + 1,
                    "cider": val_cider,
                    "meteor": val_meteor,
                    "bleu4": val_bleu4,
                    "reward": avg_reward,
                },
                output_dir,
                is_best=is_best,
            )

    print("\n" + "=" * 60)
    print("SCST Training Complete!")
    print(f"Best validation CIDEr: {best_cider:.2f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 60)

    # Finish wandb
    if wandb_run is not None:
        wandb_run.log({"best_cider": best_cider})
        wandb_run.finish()


if __name__ == "__main__":
    main()
