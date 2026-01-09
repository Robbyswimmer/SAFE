#!/usr/bin/env python3
"""
eval_train_metrics.py - Evaluate a trained SAFE checkpoint on training data

Uses the same model creation and evaluate() function from train_safe.py to ensure
identical behavior and proper checkpoint loading.

Usage:
    python safe/ablations/eval_train_metrics.py \
        --checkpoint /path/to/checkpoint.pt \
        --data_path /path/to/data \
        --max_samples 1000
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_configs import get_config
from safe.data.datasets import AudioCapsDataset, create_safe_dataloader
# Import model creation and checkpoint loading directly from train_safe.py
# This ensures identical architecture and weight loading behavior
from train_safe import create_model, load_checkpoint, evaluate, format_time


def load_model_from_checkpoint(
    checkpoint_path: Path,
    config_name: str = "phase1",
    device: str = "cuda",
    fusion_layer_indices: list = None,
):
    """Load SAFE model using the same code path as train_safe.py."""

    print(f"[INFO] Loading config: {config_name}")
    config = get_config(config_name)

    # Override fusion layer indices if specified
    if fusion_layer_indices:
        config["fusion_layer_indices"] = fusion_layer_indices
        # Also update nested configs
        if "fusion_config" in config and isinstance(config["fusion_config"], dict):
            fusion_cfg = config["fusion_config"]
            if "modalities" in fusion_cfg and isinstance(fusion_cfg["modalities"], dict):
                for modality_name, modality_cfg in fusion_cfg["modalities"].items():
                    if isinstance(modality_cfg, dict):
                        modality_cfg["layer_indices"] = fusion_layer_indices
        print(f"[INFO] Using custom fusion layer indices: {fusion_layer_indices}")

    print(f"[INFO] Creating model using train_safe.create_model()...")
    # Use the exact same model creation as training
    model = create_model(config)

    # Move to device first (required for load_checkpoint)
    device_obj = torch.device(device)
    model = model.to(device_obj)

    print(f"[INFO] Loading checkpoint using train_safe.load_checkpoint()...")
    # Use the exact same checkpoint loading as training
    # Enable debug_keys=True to see the key format mismatch
    metrics = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=None,  # Not needed for eval
        scheduler=None,  # Not needed for eval
        device=device_obj,
        debug_keys=True,  # Show checkpoint vs model key formats
    )

    print(f"[INFO] Checkpoint loaded from epoch: {metrics.get('epoch', 'unknown')}")
    if 'val_cider' in metrics:
        print(f"[INFO] Checkpoint val CIDEr: {metrics['val_cider']:.2f}")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAFE checkpoint on training data")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--config", type=str, default="phase1",
                        choices=["demo", "full", "multimodal", "phase1"],
                        help="Model config name")
    parser.add_argument("--fusion-layer-indices", type=str, default=None,
                        help="Comma-separated layer indices (e.g., '8' or '8,16'). Must match checkpoint.")
    parser.add_argument("--split", type=str, default="train",
                        help="Data split to evaluate (train or val)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (None = all)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=20,
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

    # Parse fusion layer indices if provided
    fusion_layer_indices = None
    if args.fusion_layer_indices:
        fusion_layer_indices = [int(x.strip()) for x in args.fusion_layer_indices.split(",")]

    # Load model
    device = torch.device(args.device)
    model = load_model_from_checkpoint(
        Path(args.checkpoint),
        config_name=args.config,
        device=args.device,
        fusion_layer_indices=fusion_layer_indices,
    )

    # Load dataset
    print(f"\n[INFO] Loading {args.split} dataset...")
    data_path = Path(args.data_path)

    dataset = AudioCapsDataset(
        data_path=data_path,
        split=args.split,
    )

    print(f"[INFO] Dataset size: {len(dataset)} samples")

    dataloader = create_safe_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Compute max_batches
    max_batches = None
    if args.max_samples:
        max_batches = (args.max_samples + args.batch_size - 1) // args.batch_size

    # Use the SAME evaluate function from train_safe.py
    print(f"\n[INFO] Starting evaluation using train_safe.evaluate()...")
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"RESULTS ({args.split} split)")
    print("=" * 60)
    print(f"Samples evaluated: {metrics.get('num_samples', 'N/A')}")
    print(f"Evaluation time: {format_time(metrics.get('eval_time', 0))}")
    print(f"\nLoss: {metrics.get('loss', 0):.4f}")
    print(f"BLEU-1: {metrics.get('bleu1', 0):.4f}")
    print(f"BLEU-4: {metrics.get('bleu4', 0):.4f}")
    print(f"METEOR: {metrics.get('meteor', 0):.4f}")
    print(f"ROUGE-L: {metrics.get('rouge_l', 0):.4f}")
    print(f"CIDEr: {metrics.get('cider', 0):.2f}")
    print("=" * 60)

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
        "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[INFO] Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
