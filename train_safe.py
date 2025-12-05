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


# Global cache for HuggingFace evaluate metrics (to avoid re-loading)
_EVALUATE_METRIC_CACHE: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Any] = {}


def _normalize_audio_caption(text: Any) -> str:
    """
    Normalization logic copied from StageATrainer._normalize_audio_caption
    so that metrics match the main training pipeline.
    """
    if text is None:
        return ""

    if isinstance(text, (list, tuple)):
        text = " ".join(str(t) for t in text if t)
    elif isinstance(text, dict):
        value = text.get("answer") or text.get("text")
        text = value if value is not None else ""

    import re
    import unicodedata

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\u2019", "'")  # Normalize curly apostrophes
    normalized = normalized.lower()

    # Collapse possessives before stripping punctuation so "dog's" -> "dogs"
    normalized = re.sub(r"'s\b", "s", normalized)

    # Remove residual apostrophes and punctuation (keep alphanumerics + whitespace)
    normalized = re.sub(r"'", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)

    tokens = [tok for tok in normalized.split() if tok]
    if not tokens:
        return ""

    number_map = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
    }

    cleaned_tokens: List[str] = []
    for tok in tokens:
        cleaned_tokens.append(number_map.get(tok, tok))

    if not cleaned_tokens:
        return ""

    return " ".join(cleaned_tokens)


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
    compute_bertscore: bool = False,
    light_metrics: bool = False,
) -> Dict[str, float]:
    """
    Compute caption metrics, mirroring StageATrainer's metric stack as closely
    as possible while keeping training-time evaluation lightweight when
    `light_metrics=True`.
    """
    # Default structure so callers can always rely on these keys
    metrics: Dict[str, float] = {
        "cider": 0.0,
        "spice": 0.0,
        "spider": 0.0,
        "bleu1": 0.0,
        "bleu2": 0.0,
        "bleu3": 0.0,
        "bleu4": 0.0,
        "meteor": 0.0,
        "rouge_l": 0.0,
        "bertscore_f1": 0.0,
    }

    if not predictions or not references:
        return metrics

    # Normalize and filter empty pairs (match StageATrainer behaviour)
    paired: List[Tuple[str, List[str]]] = []
    for pred, refs in zip(predictions, references):
        pred_clean = _normalize_audio_caption(str(pred).strip())
        refs_clean = [
            _normalize_audio_caption(str(ref).strip())
            for ref in refs
            if str(ref).strip()
        ]
        pred_clean = pred_clean.strip()
        refs_clean = [r for r in refs_clean if r.strip()]
        if pred_clean and refs_clean:
            paired.append((pred_clean, refs_clean))

    if not paired:
        return metrics

    preds_list, refs_list = zip(*paired)
    preds_list = list(preds_list)
    refs_list = [list(r) for r in refs_list]

    # Reference statistics (useful sanity check; mirrors StageATrainer logs)
    ref_counts = [len(r) for r in refs_list]
    if ref_counts:
        avg_refs = sum(ref_counts) / len(ref_counts)
        min_refs = min(ref_counts)
        max_refs = max(ref_counts)
        print(
            f"[RefValidation] References per sample: avg={avg_refs:.1f}, "
            f"min={min_refs}, max={max_refs}, total_samples={len(refs_list)}",
            flush=True,
        )
        if avg_refs < 2.0:
            print(
                f"⚠️  WARNING: Low reference count (avg={avg_refs:.1f}). "
                f"AudioCaps-style CIDEr expects ~5 refs/sample.",
                flush=True,
            )

    # HuggingFace evaluate metrics (BLEU/METEOR/ROUGE and optional BERTScore)
    try:
        import evaluate

        def _metric(name: str, **load_kwargs):
            key = (name, tuple(sorted(load_kwargs.items())))
            if key not in _EVALUATE_METRIC_CACHE:
                _EVALUATE_METRIC_CACHE[key] = evaluate.load(name, **load_kwargs)
            return _EVALUATE_METRIC_CACHE[key]

        # BLEU
        try:
            bleu_metric = _metric("bleu")
            bleu_result = bleu_metric.compute(predictions=preds_list, references=refs_list)
            if bleu_result:
                precisions = bleu_result.get("precisions", [])
                for n in range(min(4, len(precisions))):
                    metrics[f"bleu{n + 1}"] = float(precisions[n])
        except Exception as exc:
            print(f"⚠️  BLEU metric failed: {exc}", flush=True)

        if not light_metrics:
            # METEOR
            try:
                meteor_metric = _metric("meteor")
                meteor_result = meteor_metric.compute(predictions=preds_list, references=refs_list)
                if meteor_result and "meteor" in meteor_result:
                    metrics["meteor"] = float(meteor_result["meteor"])
            except Exception as exc:
                print(f"⚠️  METEOR metric failed: {exc}", flush=True)

            # ROUGE-L (best over references per sample)
            try:
                rouge_metric = _metric("rouge")
                rouge_scores: List[float] = []
                for pred, refs in zip(preds_list, refs_list):
                    best = 0.0
                    for ref in refs:
                        try:
                            result = rouge_metric.compute(predictions=[pred], references=[ref])
                            best = max(best, float(result.get("rougeL", 0.0)))
                        except Exception as rouge_exc:
                            print(f"⚠️  ROUGE-L metric failed on sample: {rouge_exc}", flush=True)
                    rouge_scores.append(best)
                if rouge_scores:
                    metrics["rouge_l"] = float(sum(rouge_scores) / len(rouge_scores))
            except Exception as exc:
                print(f"⚠️  ROUGE-L metric failed: {exc}", flush=True)

        # Optional BERTScore via evaluate (only in heavy eval mode)
        if compute_bertscore and not light_metrics:
            try:
                bert_metric = _metric("bertscore")
                bert_result = bert_metric.compute(
                    predictions=preds_list,
                    references=[r[0] for r in refs_list],
                    lang="en",
                )
                if "f1" in bert_result:
                    f1_scores = bert_result["f1"]
                    if isinstance(f1_scores, (list, tuple)) and len(f1_scores) > 0:
                        metrics["bertscore_f1"] = float(sum(f1_scores) / len(f1_scores))
            except Exception as exc:
                print(f"⚠️  BERTScore metric failed: {exc}", flush=True)

    except Exception as exc:
        # If evaluate is unavailable, we still compute CIDEr/SPICE below
        print(
            f"⚠️  evaluate library unavailable for BLEU/METEOR/ROUGE/BERTScore: {exc}",
            flush=True,
        )

    # CIDEr + SPICE (pycocoevalcap) – match StageATrainer behaviour
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice

        gts = {str(i): refs for i, refs in enumerate(refs_list)}
        res = {str(i): [pred] for i, pred in enumerate(preds_list)}

        try:
            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(gts, res)
            metrics["cider"] = float(cider_score) * 100.0
        except Exception as exc:
            print(f"⚠️  CIDEr metric failed: {exc}", flush=True)

        if not light_metrics:
            try:
                spice_scorer = Spice()
                spice_score, _ = spice_scorer.compute_score(gts, res)
                metrics["spice"] = float(spice_score) * 100.0
            except Exception as exc:
                print(f"⚠️  SPICE metric failed: {exc}", flush=True)

        if metrics["cider"] > 0.0 and metrics["spice"] > 0.0:
            metrics["spider"] = (metrics["cider"] + metrics["spice"]) / 2.0

    except ImportError as exc:
        print(f"⚠️  pycocoevalcap unavailable for CIDEr/SPICE: {exc}", flush=True)

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
    light_metrics: bool = False,
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

    # CRITICAL: Configure generation parameters to prevent hanging
    tokenizer = model.base_vl.tokenizer

    # Ensure pad_token exists
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    # Set generation config on the LLM to prevent conflicts
    if hasattr(model.base_vl.llm, 'config'):
        model.base_vl.llm.config.pad_token_id = tokenizer.pad_token_id
        model.base_vl.llm.config.eos_token_id = tokenizer.eos_token_id

    if hasattr(model.base_vl.llm, 'generation_config'):
        model.base_vl.llm.generation_config.pad_token_id = tokenizer.pad_token_id
        model.base_vl.llm.generation_config.eos_token_id = tokenizer.eos_token_id
        # Override max_length to respect max_new_tokens limit
        model.base_vl.llm.generation_config.max_length = None

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

        # CRITICAL FIX: Filter out samples with missing audio files
        # When audio files are missing, the model creates zero-filled audio_tokens
        # which causes generation to hang. Skip these samples entirely.
        valid_indices = [i for i, a in enumerate(audio) if a is not None]

        if not valid_indices:
            # Skip batch entirely if all audio is missing
            if batch_idx < 5:  # Only log first few skipped batches
                print(f"  ⚠️  Skipping batch {batch_idx} - all audio files missing", flush=True)
            continue

        if len(valid_indices) < len(audio):
            # Partial batch - filter to only valid samples
            if batch_idx < 5:
                print(f"  ⚠️  Filtering batch {batch_idx} - {len(audio) - len(valid_indices)}/{len(audio)} audio files missing", flush=True)
            questions = [questions[i] for i in valid_indices]
            answers = [answers[i] for i in valid_indices]
            audio = [audio[i] for i in valid_indices]

        # Prepare inputs (ensure correct device)
        inputs = model.prepare_multimodal_inputs(
            text=questions,
            audio=audio,
            answers=answers,
            device=device,
            training_mode=True,  # For loss computation
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

        # Generate predictions (reuse same device)
        generation_inputs = model.prepare_multimodal_inputs(
            text=questions,
            audio=audio,
            answers=None,  # No answers for generation
            device=device,
            training_mode=False,
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
            min_new_tokens=1,  # CRITICAL: Force at least 1 token to prevent empty generation
            num_beams=num_beams,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
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
    print(f"[Metrics] Computing caption metrics on {len(all_predictions)} predictions...", flush=True)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    caption_metrics = compute_caption_metrics(
        all_predictions,
        all_references,
        compute_bertscore=compute_bertscore,
        light_metrics=light_metrics,
    )

    metrics = {
        "loss": avg_loss,
        **caption_metrics,
        "num_samples": len(all_predictions),
        "eval_time": elapsed,
    }

    print(f"[Metrics] Caption metrics computed.", flush=True)
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
    # Ensure audio components are in training mode while keeping base VL frozen
    if hasattr(model, "enable_audio_training"):
        model.enable_audio_training()
    else:
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

    # Optional initial evaluation before training (sanity check)
    initial_max_eval = config.get("max_eval_batches", None)
    if initial_max_eval is not None and initial_max_eval <= 0:
        initial_max_eval = None
    print(f"[InitEval] Running initial evaluation on validation set (max_batches={initial_max_eval})", flush=True)
    init_metrics = evaluate(
        model,
        val_loader,
        device,
        max_batches=initial_max_eval,
        max_new_tokens=config.get("max_new_tokens", 20),
        num_beams=config.get("num_beams", 1),
        # Full caption metrics by default (BLEU, METEOR, ROUGE, CIDEr, SPICE).
        # BERTScore stays off here to keep this quick.
        compute_bertscore=False,
        light_metrics=False,
    )
    print(f"[InitEval] CIDEr={init_metrics.get('cider', 0.0):.2f} BLEU-4={init_metrics.get('bleu4', 0.0):.4f}", flush=True)

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
                # Full caption metrics during validation; BERTScore still off.
                compute_bertscore=False,
                light_metrics=False,
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

    # Whitelist of valid SAFEModel constructor arguments
    safe_model_keys = {
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
        "llm_hidden_size",
        "audio_embed_dim",
    }

    # Filter config to only include valid constructor arguments
    constructor_config = {k: v for k, v in model_config.items() if k in safe_model_keys}

    # Initialize model
    print(f"\nInitializing SAFE model...")
    print(f"  LLM: {constructor_config.get('llm_model_name', 'N/A')}")
    print(f"  Vision: {constructor_config.get('vision_model_name', 'N/A')}")
    print(f"  Audio: {constructor_config.get('audio_encoder_type', 'N/A')}")

    model = SAFEModel(**constructor_config)
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
        # Default eval cap if not specified to keep metrics manageable
        "max_eval_batches": args.max_eval_batches if args.max_eval_batches is not None else 50,
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
            compute_bertscore=True,   # Full metrics in eval-only mode
            light_metrics=False,
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
