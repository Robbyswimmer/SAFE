#!/usr/bin/env python3
"""
Average multiple SAFE checkpoints for improved generalization.

Checkpoint averaging (also called model soup) combines weights from multiple
checkpoints trained on the same task to reduce variance and improve performance.

Usage:
    python scripts/average_checkpoints.py \
        --checkpoints checkpoint_1.pt checkpoint_2.pt checkpoint_3.pt \
        --output averaged_checkpoint.pt

    # Or average the last N checkpoints from a directory:
    python scripts/average_checkpoints.py \
        --checkpoint-dir checkpoints/my_run \
        --last-n 5 \
        --output averaged_checkpoint.pt

Reference:
    "Model soups: averaging weights of multiple fine-tuned models improves
    accuracy without increasing inference time" (Wortsman et al., 2022)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch


def load_checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    """Load checkpoint and extract model state dict."""
    checkpoint = torch.load(path, map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        elif "model" in checkpoint:
            return checkpoint["model"]
        else:
            # Assume the checkpoint itself is the state dict
            return checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")


def average_checkpoints(
    checkpoint_paths: List[Path],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Average multiple checkpoint state dicts.

    Args:
        checkpoint_paths: List of paths to checkpoint files
        weights: Optional weights for weighted averaging (must sum to 1.0)

    Returns:
        Averaged state dict
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided")

    n = len(checkpoint_paths)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(f"Number of weights ({len(weights)}) must match checkpoints ({n})")
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            print(f"[Warning] Weights sum to {total}, normalizing to 1.0")
            weights = [w / total for w in weights]

    print(f"Averaging {n} checkpoints with weights: {weights}")

    # Load first checkpoint as base
    print(f"  Loading: {checkpoint_paths[0].name}")
    averaged = load_checkpoint_state(checkpoint_paths[0])

    # Convert to float and apply first weight
    for key in averaged:
        if averaged[key].dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            averaged[key] = averaged[key].float() * weights[0]

    # Add remaining checkpoints
    for i, path in enumerate(checkpoint_paths[1:], 1):
        print(f"  Loading: {path.name}")
        state = load_checkpoint_state(path)
        weight = weights[i]

        for key in averaged:
            if key not in state:
                print(f"  [Warning] Key '{key}' not found in {path.name}, skipping")
                continue
            if averaged[key].dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                averaged[key] = averaged[key] + state[key].float() * weight

    return averaged


def find_checkpoints_in_dir(
    checkpoint_dir: Path,
    pattern: str = "checkpoint_*.pt",
    last_n: Optional[int] = None,
    exclude_best: bool = False,
) -> List[Path]:
    """
    Find checkpoint files in a directory.

    Args:
        checkpoint_dir: Directory to search
        pattern: Glob pattern for checkpoint files
        last_n: If specified, return only the last N checkpoints (by modification time)
        exclude_best: If True, exclude 'checkpoint_best.pt'

    Returns:
        List of checkpoint paths sorted by modification time (oldest first)
    """
    checkpoints = list(checkpoint_dir.glob(pattern))

    if exclude_best:
        checkpoints = [p for p in checkpoints if "best" not in p.name.lower()]

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime)

    if last_n is not None and last_n > 0:
        checkpoints = checkpoints[-last_n:]

    return checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="Average multiple SAFE checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        help="List of checkpoint files to average",
    )
    input_group.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory containing checkpoints to average",
    )

    # Options for directory mode
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Average only the last N checkpoints (by modification time)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="checkpoint_*.pt",
        help="Glob pattern for finding checkpoints in directory",
    )
    parser.add_argument(
        "--exclude-best",
        action="store_true",
        help="Exclude checkpoint_best.pt from averaging",
    )

    # Weighting options
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Weights for each checkpoint (must match number of checkpoints)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for averaged checkpoint",
    )

    # Additional options
    parser.add_argument(
        "--save-full",
        action="store_true",
        help="Save full checkpoint format (with optimizer state placeholder)",
    )

    args = parser.parse_args()

    # Collect checkpoint paths
    if args.checkpoints:
        checkpoint_paths = args.checkpoints
        # Validate all exist
        for p in checkpoint_paths:
            if not p.exists():
                print(f"Error: Checkpoint not found: {p}")
                sys.exit(1)
    else:
        if not args.checkpoint_dir.exists():
            print(f"Error: Directory not found: {args.checkpoint_dir}")
            sys.exit(1)

        checkpoint_paths = find_checkpoints_in_dir(
            args.checkpoint_dir,
            pattern=args.pattern,
            last_n=args.last_n,
            exclude_best=args.exclude_best,
        )

        if not checkpoint_paths:
            print(f"Error: No checkpoints found in {args.checkpoint_dir} matching '{args.pattern}'")
            sys.exit(1)

    print(f"\nFound {len(checkpoint_paths)} checkpoints to average:")
    for p in checkpoint_paths:
        print(f"  - {p}")
    print()

    # Average checkpoints
    averaged_state = average_checkpoints(checkpoint_paths, weights=args.weights)

    # Save result
    print(f"\nSaving averaged checkpoint to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.save_full:
        # Save in full checkpoint format
        checkpoint = {
            "model_state_dict": averaged_state,
            "averaged_from": [str(p) for p in checkpoint_paths],
            "weights": args.weights or [1.0 / len(checkpoint_paths)] * len(checkpoint_paths),
        }
        torch.save(checkpoint, args.output)
    else:
        # Save just the state dict
        torch.save(averaged_state, args.output)

    print("Done!")

    # Print some stats
    total_params = sum(p.numel() for p in averaged_state.values() if isinstance(p, torch.Tensor))
    print(f"\nCheckpoint stats:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Number of keys: {len(averaged_state)}")


if __name__ == "__main__":
    main()
