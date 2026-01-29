#!/usr/bin/env python3
"""
MCUB Composition Ablation Evaluation Script.

Tests modality composition for pre-FFN vs KV augmentation architectures:
1. Audio only: Load both adapters, provide only audio input
2. Point cloud only: Load both adapters, provide only point cloud input
3. Both modalities: Load both adapters, provide audio + point cloud

Usage:
    # Evaluate pre-FFN composition
    python eval_mcub.py \
        --config composition_preffn \
        --audio-adapter outputs/audio_preffn/best_model.pt \
        --pc-adapter outputs/pc_preffn/best_model.pt \
        --data-path /path/to/mcub \
        --output-dir outputs/eval_mcub_preffn

    # Evaluate KV augmentation composition
    python eval_mcub.py \
        --config composition_kvaug \
        --audio-adapter outputs/audio_kvaug/best_model.pt \
        --pc-adapter outputs/pc_kvaug/best_model.pt \
        --data-path /path/to/mcub \
        --output-dir outputs/eval_mcub_kvaug

    # Use synthetic data for testing
    python eval_mcub.py --synthetic --config composition_preffn
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from configs.composition_configs import get_composition_config, COMPOSITION_CONFIGS
from safe.models.safe_multimodal import SAFEMultiModalModel
from safe.data.mcub_dataset import (
    MCUBDataset,
    SyntheticMCUBDataset,
    mcub_collate_fn,
    create_mcub_dataloader,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MCUB Composition Ablation Evaluation")

    # Model configuration
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=list(COMPOSITION_CONFIGS.keys()),
        help="Composition config name",
    )
    parser.add_argument(
        "--audio-adapter",
        type=str,
        default=None,
        help="Path to trained audio adapter checkpoint",
    )
    parser.add_argument(
        "--pc-adapter",
        type=str,
        default=None,
        help="Path to trained point cloud adapter checkpoint",
    )

    # Data configuration
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/mcub",
        help="Path to MCUB dataset",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for testing pipeline",
    )
    parser.add_argument(
        "--num-synthetic-samples",
        type=int,
        default=100,
        help="Number of synthetic samples to generate",
    )

    # Evaluation settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["audio_only", "pc_only", "both"],
        choices=["audio_only", "pc_only", "both"],
        help="Evaluation conditions to run",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval_mcub",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    return parser.parse_args()


def load_model(
    config: Dict[str, Any],
    audio_adapter_path: Optional[str] = None,
    pc_adapter_path: Optional[str] = None,
    device: str = "cuda",
) -> SAFEMultiModalModel:
    """
    Load SAFEMultiModalModel with optional pre-trained adapters.

    Args:
        config: Model configuration dict
        audio_adapter_path: Path to trained audio adapter
        pc_adapter_path: Path to trained point cloud adapter
        device: Device to load model on

    Returns:
        Loaded model with adapters
    """
    print(f"\n{'='*60}")
    print(f"Loading model: {config['name']}")
    print(f"{'='*60}")

    # Create model
    model = SAFEMultiModalModel(
        llm_model_name=config.get("llm_model_name", "llava-hf/llava-1.5-13b-hf"),
        vision_model_name=config.get("vision_model_name", "openai/clip-vit-large-patch14"),
        audio_encoder_type=config.get("audio_encoder_type", "clap"),
        audio_encoder_config=config.get("audio_encoder_config"),
        audio_embed_dim=config.get("audio_embed_dim", 512),
        num_audio_tokens=config.get("num_audio_tokens", 8),
        pointcloud_encoder_type=config.get("pointcloud_encoder_type", "pointbert"),
        pointcloud_encoder_config=config.get("pointcloud_encoder_config"),
        pointcloud_embed_dim=config.get("pointcloud_embed_dim", 768),
        num_pointcloud_tokens=config.get("num_pointcloud_tokens", 8),
        projector_type=config.get("projector_type", "standard"),
        projector_config=config.get("projector_config"),
        fusion_type=config.get("fusion_type", "multilayer"),
        fusion_layer_indices=config.get("fusion_layer_indices"),
        lora_rank=config.get("lora_rank", 16),
        fusion_config=config.get("fusion_config"),
        freeze_base_vl=True,
        freeze_audio_encoder=True,
        freeze_pointcloud_encoder=True,
        llm_hidden_size=config.get("llm_hidden_size", 5120),
    )

    # Load audio adapter if provided
    if audio_adapter_path and Path(audio_adapter_path).exists():
        print(f"\nLoading audio adapter from: {audio_adapter_path}")
        model.load_adapters(audio_adapter_path, modality="audio")

    # Load point cloud adapter if provided
    if pc_adapter_path and Path(pc_adapter_path).exists():
        print(f"\nLoading point cloud adapter from: {pc_adapter_path}")
        model.load_adapters(pc_adapter_path, modality="pointcloud")

    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel loaded: {total_params/1e6:.1f}M params ({trainable_params/1e6:.1f}M trainable)")

    return model


def evaluate_condition(
    model: SAFEMultiModalModel,
    dataloader: DataLoader,
    condition: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluate model on MCUB under a specific condition.

    Args:
        model: The multi-modal model
        dataloader: MCUB dataloader
        condition: One of "audio_only", "pc_only", "both"
        device: Device to use

    Returns:
        Dict with accuracy and per-sample results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating condition: {condition}")
    print(f"{'='*60}")

    # Configure modality masking based on condition
    if condition == "audio_only":
        model.enable_modality("audio", True)
        model.enable_modality("pointcloud", False)
    elif condition == "pc_only":
        model.enable_modality("audio", False)
        model.enable_modality("pointcloud", True)
    elif condition == "both":
        model.enable_modality("audio", True)
        model.enable_modality("pointcloud", True)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Evaluation loop
    results = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval {condition}"):
            # Move to device
            audio = batch["audio"]
            pointcloud = batch["pointcloud"]

            if audio is not None:
                audio = audio.to(device)
            if pointcloud is not None:
                pointcloud = pointcloud.to(device)

            # Forward pass
            outputs = model(
                audio=audio if condition != "pc_only" else None,
                pointcloud=pointcloud if condition != "audio_only" else None,
                return_hidden_states=True,
            )

            # Get predictions
            # For classification, we use the last hidden state pooled representation
            # In a full evaluation, this would be compared against answer choices
            logits = outputs.get("logits")
            last_hidden = outputs.get("last_hidden_state")

            # For each sample in batch
            for i, sample_id in enumerate(batch["sample_ids"]):
                answer = batch["answers"][i]
                choices = batch["choices"][i]

                # Simple accuracy computation
                # In real MCUB evaluation, you would:
                # 1. Generate text from the model
                # 2. Compare against answer choices
                # 3. Use exact match or fuzzy matching

                # For now, we track that the forward pass works
                # and store metadata for analysis
                result = {
                    "sample_id": sample_id,
                    "condition": condition,
                    "answer": answer,
                    "choices": choices,
                    "has_audio": audio is not None,
                    "has_pointcloud": pointcloud is not None,
                }

                # If we have a label (synthetic data), compute accuracy
                if "label" in batch:
                    label = batch["label"][i] if isinstance(batch["label"], list) else batch["label"][i].item()
                    # For synthetic data, we can compute simple classification accuracy
                    # using the hidden state similarity to class embeddings
                    result["label"] = label

                results.append(result)
                total += 1

    # Compute metrics
    metrics = {
        "condition": condition,
        "total_samples": total,
        "results": results,
    }

    # If we have labels (synthetic), compute accuracy
    if results and "label" in results[0]:
        # For synthetic data, accuracy would be computed here
        # For real MCUB, you need text generation and matching
        metrics["note"] = "Full MCUB accuracy requires text generation evaluation"

    return metrics


def run_evaluation(
    model: SAFEMultiModalModel,
    dataloader: DataLoader,
    conditions: List[str],
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run full composition ablation evaluation.

    Args:
        model: The multi-modal model
        dataloader: MCUB dataloader
        conditions: List of conditions to evaluate
        device: Device to use

    Returns:
        Dict with results for all conditions
    """
    all_results = {}

    for condition in conditions:
        metrics = evaluate_condition(
            model=model,
            dataloader=dataloader,
            condition=condition,
            device=device,
        )
        all_results[condition] = metrics

    return all_results


def print_results_summary(results: Dict[str, Any], config_name: str):
    """Print a summary of evaluation results."""
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY: {config_name}")
    print(f"{'='*60}")

    for condition, metrics in results.items():
        print(f"\n{condition}:")
        print(f"  Total samples: {metrics['total_samples']}")
        if "accuracy" in metrics:
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
        if "note" in metrics:
            print(f"  Note: {metrics['note']}")

    # Print composition analysis
    print(f"\n{'='*60}")
    print("COMPOSITION ANALYSIS")
    print(f"{'='*60}")

    if "audio_only" in results and "pc_only" in results and "both" in results:
        print("\nKey questions answered by this ablation:")
        print("1. Modality interference: Do inactive adapters degrade performance?")
        print("   - Compare 'audio_only' vs single-modality audio baseline")
        print("   - Compare 'pc_only' vs single-modality PC baseline")
        print("2. Composition benefit: Does having both modalities improve accuracy?")
        print("   - Compare 'both' vs 'audio_only' and 'pc_only'")
        print("3. Architecture difference: Is this architecture good for composition?")
        print("   - Compare these results vs the other architecture (pre-FFN vs KV-aug)")


def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = get_composition_config(args.config)
    print(f"\nConfig: {config['name']}")
    print(f"Description: {config['description']}")

    # Create dataloader
    if args.synthetic:
        print("\nUsing synthetic data for testing...")
        dataset = SyntheticMCUBDataset(
            num_samples=args.num_synthetic_samples,
            num_classes=10,
            audio_sample_rate=config.get("audio_encoder_config", {}).get("sample_rate", 48000),
            num_points=config.get("pointcloud_encoder_config", {}).get("num_points", 1024),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=mcub_collate_fn,
        )
    else:
        print(f"\nLoading MCUB from: {args.data_path}")
        dataloader = create_mcub_dataloader(
            data_path=args.data_path,
            modalities=["audio", "pointcloud"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # Load model
    model = load_model(
        config=config,
        audio_adapter_path=args.audio_adapter,
        pc_adapter_path=args.pc_adapter,
        device=args.device,
    )

    # Run evaluation
    print(f"\nRunning evaluation with conditions: {args.conditions}")
    start_time = time.time()

    results = run_evaluation(
        model=model,
        dataloader=dataloader,
        conditions=args.conditions,
        device=args.device,
    )

    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.1f}s")

    # Save results
    results_path = output_dir / f"results_{config['name']}.json"
    with open(results_path, "w") as f:
        # Convert results to JSON-serializable format
        json_results = {
            "config": config["name"],
            "conditions": args.conditions,
            "audio_adapter": args.audio_adapter,
            "pc_adapter": args.pc_adapter,
            "synthetic": args.synthetic,
            "elapsed_seconds": elapsed,
            "results": {
                cond: {
                    "total_samples": metrics["total_samples"],
                    "note": metrics.get("note", ""),
                }
                for cond, metrics in results.items()
            },
        }
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Print summary
    print_results_summary(results, config["name"])

    # Save per-sample results for detailed analysis
    detailed_path = output_dir / f"detailed_{config['name']}.json"
    detailed_results = []
    for cond, metrics in results.items():
        for r in metrics.get("results", []):
            detailed_results.append(r)

    with open(detailed_path, "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"Detailed results saved to: {detailed_path}")


if __name__ == "__main__":
    main()
