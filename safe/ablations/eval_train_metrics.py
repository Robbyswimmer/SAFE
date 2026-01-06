#!/usr/bin/env python3
"""
eval_train_metrics.py - Evaluate a trained SAFE checkpoint on training data

Computes BLEU, METEOR, CIDEr on training set to measure fit quality.
Use alongside validation metrics to assess overfitting/underfitting.

Usage:
    python safe/ablations/eval_train_metrics.py \
        --checkpoint /path/to/checkpoint.pt \
        --data_path /path/to/data \
        --max_samples 1000

On cluster:
    python safe/ablations/eval_train_metrics.py \
        --checkpoint /data/.../checkpoints/best_model.pt \
        --data_path /data/.../experiments/full_training/data \
        --max_samples 2000
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_configs import get_config
from safe.models.safe_model import SAFEModel
from safe.data.datasets import AudioCapsDataset, create_safe_dataloader, _collate_multimodal_batch


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def load_model_from_checkpoint(
    checkpoint_path: Path,
    config_name: str = "phase1",
    device: str = "cuda"
) -> SAFEModel:
    """Load SAFE model and restore checkpoint weights."""

    print(f"[INFO] Loading config: {config_name}")
    config = get_config(config_name)

    print(f"[INFO] Initializing SAFE model...")
    model = SAFEModel(
        llm_model_name=config["llm_model_name"],
        vision_model_name=config["vision_model_name"],
        audio_encoder_type=config["audio_encoder_type"],
        audio_encoder_config=config["audio_encoder_config"],
        llm_hidden_size=config["llm_hidden_size"],
        audio_embed_dim=config["audio_embed_dim"],
        projector_type=config["projector_type"],
        num_audio_tokens=config["num_audio_tokens"],
        projector_config=config.get("projector_config", {}),
        fusion_type=config["fusion_type"],
        fusion_layer_indices=config["fusion_layer_indices"],
        lora_rank=config["lora_rank"],
        fusion_config=config.get("fusion_config", {}),
        freeze_base_vl=config["freeze_base_vl"],
        freeze_audio_encoder=config["freeze_audio_encoder"],
    )

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Support both full checkpoints and adapter-only checkpoints
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    if state_dict is None:
        state_dict = checkpoint

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Report only SAFE-relevant missing keys
    relevant_missing = [
        k for k in missing_keys
        if k.startswith(("audio_projector.", "fusion_adapter.", "audio_token_embeddings."))
    ]
    if relevant_missing:
        print(f"[WARN] Missing {len(relevant_missing)} SAFE keys: {relevant_missing[:5]}...")

    # Get checkpoint info
    if isinstance(checkpoint, dict):
        metrics = checkpoint.get("metrics", {})
        epoch = metrics.get("epoch", checkpoint.get("epoch", "unknown"))
        print(f"[INFO] Checkpoint from epoch: {epoch}")

    model = model.to(device)
    model.eval()

    return model


def compute_caption_metrics(
    predictions: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """Compute BLEU, METEOR, CIDEr, ROUGE-L metrics."""

    if not predictions:
        return {"bleu1": 0, "bleu4": 0, "meteor": 0, "rouge_l": 0, "cider": 0}

    # Try pycocoevalcap first (preferred)
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider

        # Format for pycocoevalcap: {id: [caption]}
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}

        metrics = {}

        # BLEU
        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(gts, res)
        metrics["bleu1"] = bleu_scores[0]
        metrics["bleu2"] = bleu_scores[1]
        metrics["bleu3"] = bleu_scores[2]
        metrics["bleu4"] = bleu_scores[3]

        # METEOR
        meteor_scorer = Meteor()
        meteor_score, _ = meteor_scorer.compute_score(gts, res)
        metrics["meteor"] = meteor_score

        # ROUGE-L
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(gts, res)
        metrics["rouge_l"] = rouge_score

        # CIDEr
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)
        metrics["cider"] = cider_score

        return metrics

    except ImportError:
        print("[WARN] pycocoevalcap not available, using nltk fallback")

    # Fallback to nltk
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score

        smoothing = SmoothingFunction().method1

        bleu1_scores = []
        bleu4_scores = []
        meteor_scores = []

        for pred, refs in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = [r.lower().split() for r in refs]

            # BLEU
            bleu1 = sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu4 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            bleu1_scores.append(bleu1)
            bleu4_scores.append(bleu4)

            # METEOR (use first reference)
            try:
                m = meteor_score([refs[0].split()], pred.split())
                meteor_scores.append(m)
            except:
                meteor_scores.append(0.0)

        return {
            "bleu1": sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0,
            "bleu4": sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0,
            "meteor": sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0,
            "rouge_l": 0.0,  # Not computed in fallback
            "cider": 0.0,    # Not computed in fallback
        }

    except ImportError:
        print("[ERROR] Neither pycocoevalcap nor nltk available")
        return {"bleu1": 0, "bleu4": 0, "meteor": 0, "rouge_l": 0, "cider": 0}


def evaluate_on_split(
    model: SAFEModel,
    dataloader: DataLoader,
    device: str,
    max_batches: Optional[int] = None,
    max_new_tokens: int = 30,
    num_beams: int = 1,
) -> Dict[str, Any]:
    """Evaluate model on a data split."""

    model.eval()
    tokenizer = model.base_vl.tokenizer

    # Ensure pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    all_predictions = []
    all_references = []
    total_loss = 0.0
    num_batches = 0

    print(f"[INFO] Running evaluation (max_batches={max_batches})...", flush=True)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            questions = batch["questions"]
            answers = batch["answers"]
            audio = batch["audio"]

            # Filter valid samples
            valid_indices = [
                i for i, (a, ans) in enumerate(zip(audio, answers))
                if a is not None and ans is not None
            ]

            if not valid_indices:
                continue

            if len(valid_indices) < len(audio):
                questions = [questions[i] for i in valid_indices]
                answers = [answers[i] for i in valid_indices]
                audio = [audio[i] for i in valid_indices]

            # Compute loss
            inputs = model.prepare_multimodal_inputs(
                text=questions,
                audio=audio,
                answers=answers,
                device=device,
                training_mode=True,
            )

            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                labels=inputs["labels"].to(device),
                audio_tokens=inputs.get("audio_tokens", torch.tensor([])).to(device) if inputs.get("audio_tokens") is not None else None,
                audio_attention_mask=inputs.get("audio_attention_mask", torch.tensor([])).to(device) if inputs.get("audio_attention_mask") is not None else None,
            )

            if outputs.get("loss") is not None:
                total_loss += outputs["loss"].item()
                num_batches += 1

            # Generate predictions
            gen_inputs = model.prepare_multimodal_inputs(
                text=questions,
                audio=audio,
                answers=None,
                device=device,
                training_mode=False,
            )

            # Generate with settings matching train_safe.py
            gen_input_ids = gen_inputs["input_ids"].to(device)
            gen_attention_mask = gen_inputs["attention_mask"].to(device)
            gen_audio_tokens = gen_inputs.get("audio_tokens")
            if gen_audio_tokens is not None:
                gen_audio_tokens = gen_audio_tokens.to(device)
            gen_audio_attention_mask = gen_inputs.get("audio_attention_mask")
            if gen_audio_attention_mask is not None:
                gen_audio_attention_mask = gen_audio_attention_mask.to(device)

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 1,
                "num_beams": num_beams,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            generated_ids = model.generate(
                input_ids=gen_input_ids,
                attention_mask=gen_attention_mask,
                audio_tokens=gen_audio_tokens,
                audio_attention_mask=gen_audio_attention_mask,
                **generation_kwargs,
            )

            # Decode predictions
            batch_predictions = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Clean and store
            for i, pred in enumerate(batch_predictions):
                # Remove question from prediction if present
                question = questions[i]
                if question and question in pred:
                    pred = pred.replace(question, "").strip()

                # Extract answer after ASSISTANT:
                if "ASSISTANT:" in pred:
                    pred = pred.split("ASSISTANT:")[-1].strip()

                all_predictions.append(pred)

                # Handle references
                answer = answers[i]
                if isinstance(answer, str):
                    refs = [answer]
                elif isinstance(answer, list):
                    refs = [str(a) for a in answer]
                else:
                    refs = [str(answer)]
                all_references.append(refs)

            # Progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}, samples: {len(all_predictions)}", flush=True)

    elapsed = time.time() - start_time

    # Compute metrics
    print(f"[INFO] Computing metrics on {len(all_predictions)} samples...")
    metrics = compute_caption_metrics(all_predictions, all_references)

    metrics["loss"] = total_loss / num_batches if num_batches > 0 else 0.0
    metrics["num_samples"] = len(all_predictions)
    metrics["eval_time"] = elapsed

    return {
        "metrics": metrics,
        "predictions": all_predictions[:10],  # Save first 10 for inspection
        "references": all_references[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAFE checkpoint on training data")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--config", type=str, default="phase1",
                        choices=["demo", "full", "multimodal", "phase1"],
                        help="Model config name")
    parser.add_argument("--split", type=str, default="train",
                        help="Data split to evaluate (train or val)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (None = all)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="Max tokens to generate")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Beam search size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: auto-generate)")
    args = parser.parse_args()

    print("=" * 60)
    print("SAFE Training Data Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data path: {args.data_path}")
    print(f"Split: {args.split}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print("=" * 60)

    # Load model
    model = load_model_from_checkpoint(
        Path(args.checkpoint),
        config_name=args.config,
        device=args.device
    )

    # Load dataset
    print(f"\n[INFO] Loading {args.split} dataset...")
    data_path = Path(args.data_path)

    dataset = AudioCapsDataset(
        data_path=data_path,
        split=args.split,
    )

    print(f"[INFO] Dataset size: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_multimodal_batch,
    )

    # Compute max_batches
    max_batches = None
    if args.max_samples:
        max_batches = (args.max_samples + args.batch_size - 1) // args.batch_size

    # Evaluate
    print(f"\n[INFO] Starting evaluation...")
    results = evaluate_on_split(
        model=model,
        dataloader=dataloader,
        device=args.device,
        max_batches=max_batches,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    # Print results
    metrics = results["metrics"]
    print("\n" + "=" * 60)
    print(f"RESULTS ({args.split} split)")
    print("=" * 60)
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"Evaluation time: {format_time(metrics['eval_time'])}")
    print(f"\nLoss: {metrics['loss']:.4f}")
    print(f"BLEU-1: {metrics['bleu1']:.4f}")
    print(f"BLEU-4: {metrics['bleu4']:.4f}")
    print(f"METEOR: {metrics['meteor']:.4f}")
    print(f"ROUGE-L: {metrics['rouge_l']:.4f}")
    print(f"CIDEr: {metrics['cider']:.2f}")
    print("=" * 60)

    # Sample predictions
    print("\nSample predictions:")
    for i, (pred, refs) in enumerate(zip(results["predictions"][:5], results["references"][:5])):
        print(f"  [{i+1}] Pred: {pred}")
        print(f"       Ref:  {refs[0]}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        checkpoint_name = Path(args.checkpoint).stem
        output_path = PROJECT_ROOT / "safe" / "ablations" / f"train_eval_{checkpoint_name}_{args.split}.json"

    output_data = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "config": args.config,
        "metrics": metrics,
        "sample_predictions": results["predictions"],
        "sample_references": results["references"],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[INFO] Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
