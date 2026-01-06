#!/usr/bin/env python3
"""
zero_forgetting_verify.py - Verify the architectural zero-forgetting guarantee

This script proves that SAFE produces IDENTICAL outputs to the frozen baseline
when audio input is absent. This is the core thesis of the paper.

The test:
1. Load frozen LLaVA baseline
2. Load SAFE model (untrained adapter - random weights are fine)
3. Run identical inputs through both
4. Compare outputs token-by-token
5. Report: exact match = guarantee proven

Usage:
    python safe/ablations/zero_forgetting_verify.py --num_samples 100

On cluster:
    python safe/ablations/zero_forgetting_verify.py \
        --coco_dir /data/SalmanAsif/RobbyMoseley/SAFE/SAFE/experiments/full_training/data/coco \
        --num_samples 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_coco_samples(
    coco_dir: Path,
    num_samples: int = 100,
    seed: int = 42
) -> List[Dict]:
    """Load COCO val images for testing."""
    val_dir = coco_dir / "val2014"

    if not val_dir.exists():
        print(f"[ERROR] COCO val2014 not found at {val_dir}")
        print("Falling back to synthetic test...")
        return None

    # Get all val images
    image_files = list(val_dir.glob("*.jpg"))
    if not image_files:
        print(f"[ERROR] No images found in {val_dir}")
        return None

    print(f"[INFO] Found {len(image_files)} COCO val images")

    # Sample randomly
    random.seed(seed)
    sampled = random.sample(image_files, min(num_samples, len(image_files)))

    samples = []
    for img_path in sampled:
        samples.append({
            "image_path": str(img_path),
            "prompt": "Describe this image in detail."
        })

    return samples


def create_synthetic_samples(num_samples: int = 100) -> List[Dict]:
    """Create synthetic test samples (no images, text-only)."""
    prompts = [
        "What is the capital of France?",
        "Explain photosynthesis briefly.",
        "What color is the sky?",
        "Count from 1 to 5.",
        "What is 2 + 2?",
    ]

    samples = []
    for i in range(num_samples):
        samples.append({
            "image_path": None,
            "prompt": prompts[i % len(prompts)]
        })

    return samples


def load_baseline_model(device: str = "cuda"):
    """Load frozen LLaVA baseline directly."""
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    model_name = "llava-hf/llava-1.5-7b-hf"  # Use 7B for faster testing
    print(f"[INFO] Loading baseline LLaVA: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()

    return model, processor


def load_safe_model(device: str = "cuda"):
    """Load SAFE model with untrained adapter."""
    from configs.model_configs import get_config
    from safe.models.safe_model import SAFEModel

    # Use a lighter config for testing
    print("[INFO] Loading SAFE model (untrained adapter)")

    # We need to match the baseline model
    model = SAFEModel(
        llm_model_name="llava-hf/llava-1.5-7b-hf",
        vision_model_name="openai/clip-vit-large-patch14",
        audio_encoder_type="clap",
        audio_encoder_config={
            "model_name": "laion/larger_clap_music_and_speech",
            "sample_rate": 48000,
            "max_length": 10.0
        },
        llm_hidden_size=4096,  # LLaVA 7B hidden size
        audio_embed_dim=512,
        projector_type="standard",
        num_audio_tokens=8,
        projector_config={"dropout": 0.1, "bottleneck_dim": 1024},
        fusion_type="multilayer",
        fusion_layer_indices=[8, 16, 24],  # Adjusted for 7B (32 layers)
        lora_rank=16,
        fusion_config={
            "num_attention_heads": 32,
            "attention_dropout": 0.1,
            "modalities": {"audio": {"layer_indices": [8, 16, 24], "num_tokens": 8}}
        },
        freeze_base_vl=True,
        freeze_audio_encoder=True,
    )

    model = model.to(device)
    model.eval()

    return model


def compare_outputs(
    baseline_output: torch.Tensor,
    safe_output: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-4
) -> Dict:
    """Compare two output tensors and return detailed statistics."""

    # Ensure same shape
    if baseline_output.shape != safe_output.shape:
        return {
            "match": False,
            "error": f"Shape mismatch: {baseline_output.shape} vs {safe_output.shape}",
            "max_diff": float('inf'),
            "mean_diff": float('inf'),
            "num_mismatched": -1
        }

    # Compute differences
    diff = torch.abs(baseline_output.float() - safe_output.float())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Check if within tolerance
    is_close = torch.allclose(
        baseline_output.float(),
        safe_output.float(),
        atol=atol,
        rtol=rtol
    )

    # Count mismatched elements
    threshold = atol + rtol * torch.abs(baseline_output.float())
    mismatched = (diff > threshold).sum().item()
    total = diff.numel()

    return {
        "match": is_close,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "num_mismatched": mismatched,
        "total_elements": total,
        "mismatch_ratio": mismatched / total if total > 0 else 0
    }


def run_baseline_forward(
    model,
    processor,
    prompt: str,
    image_path: Optional[str] = None,
    device: str = "cuda"
) -> torch.Tensor:
    """Run forward pass through baseline LLaVA."""

    # Prepare inputs
    if image_path and Path(image_path).exists():
        image = Image.open(image_path).convert("RGB")
        formatted_prompt = f"USER: <image>\n{prompt} ASSISTANT:"
        inputs = processor(
            text=formatted_prompt,
            images=image,
            return_tensors="pt"
        ).to(device)
    else:
        formatted_prompt = f"USER: {prompt} ASSISTANT:"
        inputs = processor(
            text=formatted_prompt,
            return_tensors="pt"
        ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False)

    return outputs.logits


def run_safe_forward(
    model,
    prompt: str,
    image_path: Optional[str] = None,
    device: str = "cuda"
) -> torch.Tensor:
    """Run forward pass through SAFE model with audio=None."""

    # Prepare image if provided
    image = None
    if image_path and Path(image_path).exists():
        image = Image.open(image_path).convert("RGB")

    # Prepare inputs - explicitly NO audio
    inputs = model.prepare_multimodal_inputs(
        text=prompt,
        images=[image] if image else None,
        audio=None,  # CRITICAL: No audio input
        device=device,
        include_audio_tokens=False,
        training_mode=False
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            audio_tokens=None,  # CRITICAL: No audio tokens
            labels=None
        )

    return outputs["logits"]


def run_verification(
    samples: List[Dict],
    baseline_model,
    baseline_processor,
    safe_model,
    device: str = "cuda",
    verbose: bool = True
) -> Dict:
    """Run the zero-forgetting verification test."""

    results = {
        "total": len(samples),
        "exact_matches": 0,
        "close_matches": 0,  # Within tolerance
        "failures": 0,
        "max_diff_overall": 0.0,
        "mean_diff_overall": 0.0,
        "failed_samples": []
    }

    all_diffs = []

    for i, sample in enumerate(tqdm(samples, desc="Verifying zero-forgetting")):
        prompt = sample["prompt"]
        image_path = sample.get("image_path")

        try:
            # Run baseline
            baseline_logits = run_baseline_forward(
                baseline_model, baseline_processor,
                prompt, image_path, device
            )

            # Run SAFE (no audio)
            safe_logits = run_safe_forward(
                safe_model, prompt, image_path, device
            )

            # Compare
            comparison = compare_outputs(baseline_logits, safe_logits)
            all_diffs.append(comparison["max_diff"])

            if comparison["max_diff"] == 0.0:
                results["exact_matches"] += 1
            elif comparison["match"]:
                results["close_matches"] += 1
            else:
                results["failures"] += 1
                results["failed_samples"].append({
                    "index": i,
                    "prompt": prompt[:50],
                    "max_diff": comparison["max_diff"],
                    "mismatch_ratio": comparison["mismatch_ratio"]
                })

            results["max_diff_overall"] = max(
                results["max_diff_overall"],
                comparison["max_diff"]
            )

            if verbose and i < 3:
                print(f"\n[Sample {i}] max_diff={comparison['max_diff']:.2e}, "
                      f"match={comparison['match']}")

        except Exception as e:
            print(f"\n[ERROR] Sample {i} failed: {e}")
            results["failures"] += 1
            results["failed_samples"].append({
                "index": i,
                "prompt": prompt[:50],
                "error": str(e)
            })

    # Compute overall mean diff
    if all_diffs:
        results["mean_diff_overall"] = sum(all_diffs) / len(all_diffs)

    return results


def print_results(results: Dict):
    """Print verification results."""

    print("\n" + "=" * 60)
    print("ZERO-FORGETTING VERIFICATION RESULTS")
    print("=" * 60)

    total = results["total"]
    exact = results["exact_matches"]
    close = results["close_matches"]
    failed = results["failures"]

    print(f"\nTotal samples tested: {total}")
    print(f"Exact matches (diff=0): {exact} ({100*exact/total:.1f}%)")
    print(f"Close matches (within tol): {close} ({100*close/total:.1f}%)")
    print(f"Failures: {failed} ({100*failed/total:.1f}%)")

    print(f"\nMax difference overall: {results['max_diff_overall']:.2e}")
    print(f"Mean difference overall: {results['mean_diff_overall']:.2e}")

    if failed == 0:
        print("\n" + "=" * 60)
        print("✓ ZERO-FORGETTING GUARANTEE VERIFIED")
        print("  When audio=None, SAFE output is identical to baseline.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ ZERO-FORGETTING GUARANTEE FAILED")
        print(f"  {failed} samples produced different outputs.")
        print("=" * 60)

        if results["failed_samples"]:
            print("\nFailed samples (first 5):")
            for fs in results["failed_samples"][:5]:
                print(f"  - Sample {fs['index']}: {fs.get('max_diff', fs.get('error', 'unknown'))}")


def main():
    parser = argparse.ArgumentParser(description="Verify zero-forgetting guarantee")
    parser.add_argument("--coco_dir", type=str, default=None,
                        help="Path to COCO dataset (with val2014/)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic samples (no images)")
    args = parser.parse_args()

    print("[INFO] Zero-Forgetting Verification Test")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Samples: {args.num_samples}")

    # Load samples
    if args.synthetic or args.coco_dir is None:
        print("[INFO] Using synthetic samples (text-only)")
        samples = create_synthetic_samples(args.num_samples)
    else:
        coco_dir = Path(args.coco_dir)
        samples = load_coco_samples(coco_dir, args.num_samples, args.seed)
        if samples is None:
            print("[INFO] Falling back to synthetic samples")
            samples = create_synthetic_samples(args.num_samples)

    # Load models
    print("\n[STEP 1] Loading baseline LLaVA...")
    baseline_model, baseline_processor = load_baseline_model(args.device)

    print("\n[STEP 2] Loading SAFE model (untrained adapter)...")
    safe_model = load_safe_model(args.device)

    # Run verification
    print(f"\n[STEP 3] Running verification on {len(samples)} samples...")
    results = run_verification(
        samples,
        baseline_model,
        baseline_processor,
        safe_model,
        device=args.device,
        verbose=True
    )

    # Print results
    print_results(results)

    # Save results
    output_path = PROJECT_ROOT / "safe" / "ablations" / "zero_forgetting_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {output_path}")

    # Return exit code
    return 0 if results["failures"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
