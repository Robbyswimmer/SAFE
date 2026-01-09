#!/usr/bin/env python3
"""
benchmark_eval.py - Comprehensive evaluation benchmark for SAFE audio captioning models.

Runs paper-worthy benchmarks on trained SAFE checkpoints across multiple datasets:
- AudioCaps test set (standard benchmark)
- Clotho evaluation set (secondary benchmark)

Computes all standard captioning metrics:
- CIDEr (primary metric for audio captioning)
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L
- SPICE (semantic propositional image caption evaluation)

Usage:
    # Evaluate on AudioCaps test
    python scripts/benchmark_eval.py --checkpoint checkpoints/best.pt --data-path data/

    # Evaluate on both AudioCaps and Clotho
    python scripts/benchmark_eval.py --checkpoint checkpoints/best.pt --data-path data/ --datasets audiocaps,clotho

    # Specify model config explicitly
    python scripts/benchmark_eval.py --checkpoint checkpoints/best.pt --model-config phase1

    # Save results to JSON
    python scripts/benchmark_eval.py --checkpoint checkpoints/best.pt --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_configs import get_config as get_model_config
from safe.data.datasets import AudioCapsDataset, ClothoDataset
from safe.models.safe_model import SAFEModel


# =============================================================================
# Text Normalization (matches training evaluation)
# =============================================================================

def normalize_caption(text: Any) -> str:
    """
    Normalize caption text for metric computation.
    Matches the normalization used during training evaluation.
    """
    if text is None:
        return ""

    if isinstance(text, (list, tuple)):
        text = " ".join(str(t) for t in text if t)
    elif isinstance(text, dict):
        value = text.get("answer") or text.get("text")
        text = value if value is not None else ""

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\u2019", "'")  # Normalize curly apostrophes
    normalized = normalized.lower()

    # Collapse possessives before stripping punctuation
    normalized = re.sub(r"'s\b", "s", normalized)

    # Remove punctuation (keep alphanumerics + whitespace)
    normalized = re.sub(r"'", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)

    tokens = [tok for tok in normalized.split() if tok]
    if not tokens:
        return ""

    # Number word to digit mapping
    number_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20",
    }
    cleaned_tokens = [number_map.get(tok, tok) for tok in tokens]

    return " ".join(cleaned_tokens) if cleaned_tokens else ""


def strip_prompt_prefix(text: str) -> str:
    """Remove chat template prefixes so only the assistant reply remains."""
    if not text:
        return ""

    lowered = text.lower()
    marker = "assistant:"
    idx = lowered.rfind(marker)
    if idx != -1:
        return text[idx + len(marker):].lstrip()

    user_idx = lowered.rfind("user:")
    if user_idx != -1:
        remainder = text[user_idx + len("user:"):].lstrip()
        if remainder:
            return remainder
    return text


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_all_metrics(predictions: Dict[str, List[str]],
                        references: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Compute all standard captioning metrics using pycocoevalcap.

    Args:
        predictions: Dict of {sample_id: [predicted_caption]}
        references: Dict of {sample_id: [ref_caption1, ref_caption2, ...]}

    Returns:
        Dict with all metric scores (0-100 scale)
    """
    metrics = {}

    # CIDEr
    try:
        from pycocoevalcap.cider.cider import Cider
        print("  Computing CIDEr...", flush=True)
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(references, predictions)
        metrics["CIDEr"] = cider_score * 100.0
    except Exception as e:
        print(f"  Warning: CIDEr computation failed: {e}")
        metrics["CIDEr"] = 0.0

    # BLEU
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        print("  Computing BLEU...", flush=True)
        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(references, predictions)
        for i, score in enumerate(bleu_scores):
            metrics[f"BLEU-{i+1}"] = score * 100.0
    except Exception as e:
        print(f"  Warning: BLEU computation failed: {e}")
        for i in range(4):
            metrics[f"BLEU-{i+1}"] = 0.0

    # METEOR
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        print("  Computing METEOR...", flush=True)
        meteor_scorer = Meteor()
        meteor_score, _ = meteor_scorer.compute_score(references, predictions)
        metrics["METEOR"] = meteor_score * 100.0
    except Exception as e:
        print(f"  Warning: METEOR computation failed: {e}")
        metrics["METEOR"] = 0.0

    # ROUGE-L
    try:
        from pycocoevalcap.rouge.rouge import Rouge
        print("  Computing ROUGE-L...", flush=True)
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(references, predictions)
        metrics["ROUGE-L"] = rouge_score * 100.0
    except Exception as e:
        print(f"  Warning: ROUGE computation failed: {e}")
        metrics["ROUGE-L"] = 0.0

    # SPICE
    try:
        from pycocoevalcap.spice.spice import Spice
        print("  Computing SPICE...", flush=True)
        spice_scorer = Spice()
        spice_score, _ = spice_scorer.compute_score(references, predictions)
        metrics["SPICE"] = spice_score * 100.0
    except Exception as e:
        print(f"  Warning: SPICE computation failed: {e}")
        metrics["SPICE"] = 0.0

    # SPIDEr (CIDEr + SPICE) / 2
    metrics["SPIDEr"] = (metrics.get("CIDEr", 0.0) + metrics.get("SPICE", 0.0)) / 2.0

    return metrics


# =============================================================================
# Model Loading
# =============================================================================

SAFE_MODEL_KEYS = {
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


def load_model_from_checkpoint(
    checkpoint_path: Path,
    model_config_name: str = "phase1",
    device: str = "cuda",
    config_overrides: Optional[Dict[str, Any]] = None,
) -> SAFEModel:
    """
    Load SAFE model from checkpoint with proper weight initialization.

    Handles both 'trainable_only' and 'full' checkpoint formats.
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Detect checkpoint format
    ckpt_format = "unknown"
    if isinstance(checkpoint, dict):
        ckpt_format = checkpoint.get("format", "unknown")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        ckpt_metrics = checkpoint.get("metrics", {})
        print(f"  Checkpoint format: {ckpt_format}")
        if ckpt_metrics:
            print(f"  Checkpoint metrics: epoch={ckpt_metrics.get('epoch', '?')}, "
                  f"CIDEr={ckpt_metrics.get('cider', '?'):.2f}")
    else:
        state_dict = checkpoint
        ckpt_metrics = {}

    # Load model config
    try:
        model_config = get_model_config(model_config_name)
    except ValueError as e:
        raise SystemExit(f"Unknown model config '{model_config_name}': {e}")

    # Apply any overrides
    if config_overrides:
        model_config.update(config_overrides)

    # Extract SAFE model kwargs
    model_kwargs = {k: model_config[k] for k in SAFE_MODEL_KEYS if k in model_config}

    print(f"\nInitializing SAFE model:")
    print(f"  Config: {model_config_name}")
    print(f"  LLM: {model_kwargs.get('llm_model_name')}")
    print(f"  Audio tokens: {model_kwargs.get('num_audio_tokens')}")
    print(f"  Fusion layers: {model_kwargs.get('fusion_layer_indices')}")

    # Initialize model
    model = SAFEModel(**model_kwargs)

    # Load weights
    print("\nLoading weights...")

    # Handle prefix 'safe_model.' if present (from Lightning)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("safe_model."):
            new_state_dict[k.replace("safe_model.", "")] = v
        else:
            new_state_dict[k] = v

    # Check for key components
    projector_keys = [k for k in new_state_dict.keys() if "audio_projector" in k]
    fusion_keys = [k for k in new_state_dict.keys() if "fusion_adapter" in k]
    print(f"  Found {len(projector_keys)} audio_projector keys")
    print(f"  Found {len(fusion_keys)} fusion_adapter keys")

    if not projector_keys:
        print("  WARNING: No audio_projector keys found - audio processing will be random!")

    # Load with strict=False for trainable_only checkpoints
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    # Report relevant missing/unexpected keys
    relevant_missing = [k for k in missing_keys
                       if any(x in k for x in ["audio_projector", "fusion_adapter", "audio_token"])]
    if relevant_missing:
        print(f"  WARNING: Missing {len(relevant_missing)} relevant keys: {relevant_missing[:5]}...")

    # Move to device
    if hasattr(model, "to_device"):
        model.to_device(device)
    else:
        model.to(device)

    model.eval()
    print(f"  Model loaded and moved to {device}")

    return model


# =============================================================================
# Evaluation Loop
# =============================================================================

def run_evaluation(
    model: SAFEModel,
    dataloader: DataLoader,
    device: str,
    max_new_tokens: int = 40,
    num_beams: int = 4,
    show_samples: int = 5,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Run evaluation and collect predictions/references.

    Returns:
        Tuple of (predictions_dict, references_dict) in pycocoevalcap format
    """
    model.eval()

    tokenizer = model.base_vl.tokenizer
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    predictions = {}
    references = {}
    sample_idx = 0

    # Set gate to 1.0 for evaluation (full audio contribution)
    gate_controller = getattr(model, "set_gate", None)
    saved_gate = None
    if callable(gate_controller):
        saved_gate = getattr(model, "_default_gate", None)
        try:
            gate_controller(1.0)
        except Exception:
            pass

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": False,
        "pad_token_id": pad_token_id,
        "repetition_penalty": 1.2,
        "length_penalty": 1.0,
    }
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                questions = batch.get("questions") or []
                audio_entries = batch.get("audio")

                if isinstance(audio_entries, list):
                    has_audio = any(a is not None for a in audio_entries)
                else:
                    has_audio = audio_entries is not None

                # Prepare inputs
                inputs = model.prepare_multimodal_inputs(
                    text=questions,
                    audio=audio_entries,
                    device=device,
                    include_audio_tokens=has_audio,
                    training_mode=False,
                )
                inputs.pop("labels", None)

                input_ids = inputs.get("input_ids")
                if input_ids is None:
                    continue

                attention_mask = inputs.get("attention_mask")
                audio_tokens = inputs.get("audio_tokens")
                audio_attention_mask = inputs.get("audio_attention_mask")

                gen_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "audio_tokens": audio_tokens,
                    "audio_attention_mask": audio_attention_mask,
                }

                # Generate
                gen_outputs = model.generate(**gen_inputs, **generation_kwargs)
                if not isinstance(gen_outputs, torch.Tensor):
                    gen_outputs = torch.as_tensor(gen_outputs)

                if gen_outputs.dim() == 3:
                    gen_outputs = gen_outputs[:, 0, :]

                # Get prompt lengths for decoding
                if isinstance(attention_mask, torch.Tensor):
                    prompt_lengths = attention_mask.sum(dim=1)
                else:
                    prompt_lengths = torch.full(
                        (gen_outputs.size(0),),
                        input_ids.size(1),
                        dtype=torch.long,
                        device=gen_outputs.device,
                    )

                # Decode predictions
                decoded_preds = []
                for i in range(gen_outputs.size(0)):
                    tokens = gen_outputs[i]
                    start_idx = int(prompt_lengths[i].item())
                    if start_idx < tokens.size(0):
                        gen_tokens = tokens[start_idx:]
                    else:
                        gen_tokens = tokens[-max_new_tokens:] if max_new_tokens else tokens

                    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    decoded_preds.append(decoded)

                # Store predictions and references
                batch_answers = batch.get("answers")

                for i, pred in enumerate(decoded_preds):
                    sample_id = str(sample_idx)

                    # Normalize prediction
                    stripped_pred = strip_prompt_prefix(pred)
                    pred_norm = normalize_caption(stripped_pred)
                    predictions[sample_id] = [pred_norm]

                    # Normalize references
                    if batch_answers:
                        refs = batch_answers[i]
                        if isinstance(refs, str):
                            refs = [refs]
                        refs_norm = [normalize_caption(r) for r in refs if r]
                        references[sample_id] = refs_norm if refs_norm else [""]
                    else:
                        references[sample_id] = [""]

                    # Show sample predictions
                    if sample_idx < show_samples:
                        print(f"\n[Sample {sample_idx}]")
                        print(f"  Pred: {stripped_pred.strip()[:100]}...")
                        print(f"  Ref:  {references[sample_id][0][:100]}...")

                    sample_idx += 1

    finally:
        if callable(gate_controller) and saved_gate is not None:
            try:
                gate_controller(saved_gate)
            except Exception:
                pass

    return predictions, references


# =============================================================================
# Dataset Loading
# =============================================================================

def smart_collate(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for evaluation batches."""
    audio_data = []
    for x in batch:
        audio_entry = x.get("audio")
        if isinstance(audio_entry, tuple):
            audio_data.append(audio_entry[0])  # Unpack (waveform, sr) -> waveform
        else:
            audio_data.append(audio_entry)

    return {
        "questions": [x.get("question", "Describe the audio.") for x in batch],
        "audio": audio_data,
        "answers": [x.get("answers", x.get("captions", [])) for x in batch],
    }


def load_dataset(dataset_name: str, data_path: Path, split: str):
    """Load evaluation dataset by name."""
    dataset_name = dataset_name.lower()

    if dataset_name == "audiocaps":
        return AudioCapsDataset(data_path=str(data_path), split=split)
    elif dataset_name == "clotho":
        return ClothoDataset(data_path=str(data_path), split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: audiocaps, clotho")


# =============================================================================
# Result Formatting
# =============================================================================

def format_results_table(all_results: Dict[str, Dict[str, float]]) -> str:
    """Format results as a paper-ready table."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("BENCHMARK RESULTS")
    lines.append("=" * 80)

    # Header
    metrics = ["CIDEr", "BLEU-4", "METEOR", "ROUGE-L", "SPICE", "SPIDEr"]
    header = f"{'Dataset':<15} | " + " | ".join(f"{m:>8}" for m in metrics)
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for dataset, results in all_results.items():
        row_values = [f"{results.get(m, 0.0):>8.2f}" for m in metrics]
        lines.append(f"{dataset:<15} | " + " | ".join(row_values))

    lines.append("=" * 80)

    # LaTeX table format
    lines.append("\n% LaTeX table format:")
    lines.append("% \\begin{tabular}{l" + "c" * len(metrics) + "}")
    lines.append("% \\toprule")
    lines.append("% Dataset & " + " & ".join(metrics) + " \\\\")
    lines.append("% \\midrule")
    for dataset, results in all_results.items():
        row_values = [f"{results.get(m, 0.0):.1f}" for m in metrics]
        lines.append(f"% {dataset} & " + " & ".join(row_values) + " \\\\")
    lines.append("% \\bottomrule")
    lines.append("% \\end{tabular}")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmark evaluation on SAFE audio captioning models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate best checkpoint on AudioCaps test
    python scripts/benchmark_eval.py --checkpoint checkpoints/best.pt --data-path data/

    # Evaluate on both AudioCaps and Clotho
    python scripts/benchmark_eval.py --checkpoint checkpoints/best.pt --datasets audiocaps,clotho

    # Save results to JSON
    python scripts/benchmark_eval.py --checkpoint checkpoints/best.pt --output results.json
        """,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--data-path", "-d",
        type=Path,
        default=Path("experiments/full_training/data"),
        help="Path to data directory containing dataset folders",
    )

    # Optional arguments
    parser.add_argument(
        "--datasets",
        type=str,
        default="audiocaps",
        help="Comma-separated list of datasets to evaluate (audiocaps, clotho)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="phase1",
        choices=["demo", "full", "multimodal", "phase1"],
        help="Model configuration to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams for beam search",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=5,
        help="Number of sample predictions to display",
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    # Validate data path exists
    if not args.data_path.exists():
        raise SystemExit(f"Data path not found: {args.data_path}")

    print("=" * 80)
    print("SAFE Benchmark Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data path: {args.data_path}")
    print(f"Datasets: {args.datasets}")
    print(f"Model config: {args.model_config}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Num beams: {args.num_beams}")
    print("=" * 80)

    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_config_name=args.model_config,
        device=args.device,
    )

    # Parse datasets
    datasets_to_eval = [d.strip().lower() for d in args.datasets.split(",")]

    # Dataset split mapping
    split_map = {
        "audiocaps": "test",
        "clotho": "test",  # Clotho uses "eval" which maps to "test" in our dataset class
    }

    all_results = {}

    for dataset_name in datasets_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name.upper()}")
        print("=" * 60)

        split = split_map.get(dataset_name, "test")

        try:
            dataset = load_dataset(dataset_name, args.data_path, split)
            print(f"Loaded {len(dataset)} samples from {dataset_name} {split} split")
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            continue

        if len(dataset) == 0:
            print(f"Warning: {dataset_name} has 0 samples, skipping")
            continue

        # Check reference quality
        probe_count = min(len(dataset), 10)
        ref_counts = []
        for idx in range(probe_count):
            sample = dataset[idx]
            answers = sample.get("answers", sample.get("captions", []))
            if isinstance(answers, list):
                ref_counts.append(len([a for a in answers if str(a).strip()]))
            elif answers:
                ref_counts.append(1)
            else:
                ref_counts.append(0)

        avg_refs = sum(ref_counts) / len(ref_counts) if ref_counts else 0
        print(f"Average references per sample: {avg_refs:.1f}")

        if avg_refs < 1.5 and dataset_name == "audiocaps":
            print("Warning: Low reference count - ensure using multi-reference dataset")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=smart_collate,
            num_workers=args.num_workers,
            shuffle=False,
        )

        # Run evaluation
        start_time = time.time()
        predictions, references = run_evaluation(
            model=model,
            dataloader=dataloader,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            show_samples=args.show_samples,
        )
        eval_time = time.time() - start_time

        print(f"\nEvaluation completed in {eval_time:.1f}s")
        print(f"Collected {len(predictions)} predictions")

        # Compute metrics
        print("\nComputing metrics...")
        metrics = compute_all_metrics(predictions, references)
        all_results[dataset_name] = metrics

        # Print results for this dataset
        print(f"\n{dataset_name.upper()} Results:")
        print("-" * 40)
        for metric, value in sorted(metrics.items()):
            print(f"  {metric}: {value:.2f}")

    # Print combined results table
    print(format_results_table(all_results))

    # Save results if requested
    if args.output:
        output_data = {
            "checkpoint": str(args.checkpoint),
            "model_config": args.model_config,
            "datasets": datasets_to_eval,
            "generation_config": {
                "max_new_tokens": args.max_new_tokens,
                "num_beams": args.num_beams,
            },
            "results": all_results,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\nBenchmark evaluation complete!")


if __name__ == "__main__":
    main()
