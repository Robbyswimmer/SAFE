#!/usr/bin/env python3
"""
eval_train_metrics.py - Evaluate a trained SAFE checkpoint on training data

Uses the same evaluate() function from train_safe.py to ensure identical behavior.

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
from safe.models.safe_model import SAFEModel
from safe.data.datasets import AudioCapsDataset, create_safe_dataloader
from train_safe import evaluate, format_time


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

    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    if state_dict is None:
        state_dict = checkpoint

    # Debug: show checkpoint contents
    if isinstance(checkpoint, dict):
        print(f"[DEBUG] Checkpoint format: {checkpoint.get('format', 'unknown')}")
        print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())}")
    print(f"[DEBUG] State dict has {len(state_dict)} keys")
    print(f"[DEBUG] First 10 state dict keys: {list(state_dict.keys())[:10]}")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(f"[DEBUG] Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    relevant_missing = [
        k for k in missing_keys
        if k.startswith(("audio_projector.", "fusion_adapter.", "audio_token_embeddings."))
    ]
    if relevant_missing:
        print(f"[WARN] Missing {len(relevant_missing)} SAFE keys: {relevant_missing[:5]}...")
    else:
        print(f"[INFO] All SAFE adapter keys loaded successfully")

    if isinstance(checkpoint, dict):
        metrics = checkpoint.get("metrics", {})
        epoch = metrics.get("epoch", checkpoint.get("epoch", "unknown"))
        print(f"[INFO] Checkpoint from epoch: {epoch}")

    model = model.to(device)
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

    # Load model
    device = torch.device(args.device)
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
