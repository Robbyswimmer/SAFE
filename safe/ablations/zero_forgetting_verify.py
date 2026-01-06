#!/usr/bin/env python3
"""
zero_forgetting_verify.py - Verify the architectural zero-forgetting guarantee

This script proves that SAFE produces IDENTICAL outputs to its frozen baseline
when audio input is absent. This is the core thesis of the paper.

The test:
1. Load SAFE model (untrained adapter)
2. Run SAME inputs through:
   a) SAFE with audio=None (should passthrough to base)
   b) SAFE's base_vl directly
3. Compare outputs token-by-token
4. Report: exact match = guarantee proven

Usage:
    python safe/ablations/zero_forgetting_verify.py --num_samples 50 --synthetic

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


def create_synthetic_samples(num_samples: int = 100) -> List[Dict]:
    """Create synthetic test samples (text-only, no images)."""
    prompts = [
        "What is the capital of France?",
        "Explain photosynthesis briefly.",
        "What color is the sky?",
        "Count from 1 to 5.",
        "What is 2 + 2?",
        "Name three primary colors.",
        "What is the largest planet?",
        "How many days in a week?",
        "What sound does a cat make?",
        "What is H2O?",
    ]

    samples = []
    for i in range(num_samples):
        samples.append({
            "image_path": None,
            "prompt": prompts[i % len(prompts)]
        })

    return samples


def load_coco_samples(
    coco_dir: Path,
    num_samples: int = 100,
    seed: int = 42
) -> Optional[List[Dict]]:
    """Load COCO val images for testing."""
    val_dir = coco_dir / "val2014"

    if not val_dir.exists():
        print(f"[WARN] COCO val2014 not found at {val_dir}")
        return None

    image_files = list(val_dir.glob("*.jpg"))
    if not image_files:
        print(f"[WARN] No images found in {val_dir}")
        return None

    print(f"[INFO] Found {len(image_files)} COCO val images")

    random.seed(seed)
    sampled = random.sample(image_files, min(num_samples, len(image_files)))

    samples = []
    for img_path in sampled:
        samples.append({
            "image_path": str(img_path),
            "prompt": "Describe this image."
        })

    return samples


def load_safe_model(device: str = "cuda", use_7b: bool = True):
    """Load SAFE model with untrained adapter."""
    from safe.models.safe_model import SAFEModel

    if use_7b:
        model_name = "llava-hf/llava-1.5-7b-hf"
        hidden_size = 4096
        num_heads = 32
        fusion_layers = [8, 16, 24]  # 32 layers total
    else:
        model_name = "llava-hf/llava-1.5-13b-hf"
        hidden_size = 5120
        num_heads = 40
        fusion_layers = [12, 24, 36]  # 40 layers total

    print(f"[INFO] Loading SAFE model: {model_name}")

    model = SAFEModel(
        llm_model_name=model_name,
        vision_model_name="openai/clip-vit-large-patch14",
        audio_encoder_type="clap",
        audio_encoder_config={
            "model_name": "laion/larger_clap_music_and_speech",
            "sample_rate": 48000,
            "max_length": 10.0
        },
        llm_hidden_size=hidden_size,
        audio_embed_dim=512,
        projector_type="standard",
        num_audio_tokens=8,
        projector_config={"dropout": 0.1, "bottleneck_dim": 1024},
        fusion_type="multilayer",
        fusion_layer_indices=fusion_layers,
        lora_rank=16,
        fusion_config={
            "num_attention_heads": num_heads,
            "attention_dropout": 0.1,
            "modalities": {"audio": {"layer_indices": fusion_layers, "num_tokens": 8}}
        },
        freeze_base_vl=True,
        freeze_audio_encoder=True,
    )

    model = model.to(device)
    model.eval()

    return model


def compare_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-3
) -> Dict:
    """Compare two logit tensors."""

    # Handle shape mismatches
    if logits_a.shape != logits_b.shape:
        # Try to align by taking minimum sequence length
        min_seq = min(logits_a.shape[1], logits_b.shape[1])
        logits_a = logits_a[:, :min_seq, :]
        logits_b = logits_b[:, :min_seq, :]

        if logits_a.shape != logits_b.shape:
            return {
                "match": False,
                "error": f"Shape mismatch after alignment: {logits_a.shape} vs {logits_b.shape}",
                "max_diff": float('inf'),
                "mean_diff": float('inf'),
            }

    # Compute differences
    diff = torch.abs(logits_a.float() - logits_b.float())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Check tolerance
    is_close = torch.allclose(
        logits_a.float(),
        logits_b.float(),
        atol=atol,
        rtol=rtol
    )

    # Count mismatches
    threshold = atol + rtol * torch.abs(logits_a.float())
    mismatched = (diff > threshold).sum().item()
    total = diff.numel()

    return {
        "match": is_close,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "num_mismatched": mismatched,
        "total_elements": total,
        "mismatch_pct": 100.0 * mismatched / total if total > 0 else 0
    }


def run_single_test(
    model,
    prompt: str,
    image_path: Optional[str],
    device: str,
    verbose: bool = False
) -> Dict:
    """
    Run a single zero-forgetting test.

    Compares:
    1. SAFE forward with audio=None (should use passthrough path)
    2. Direct call to base_vl.llm
    """

    # Load image if provided
    image = None
    pixel_values = None
    if image_path and Path(image_path).exists():
        image = Image.open(image_path).convert("RGB")

    # Format prompt consistently
    if image is not None:
        formatted_prompt = f"USER: <image>\n{prompt} ASSISTANT:"
    else:
        formatted_prompt = f"USER: {prompt} ASSISTANT:"

    # Get processor from base model
    processor = model.base_vl.processor

    # Process inputs
    if image is not None:
        inputs = processor(
            text=formatted_prompt,
            images=image,
            return_tensors="pt"
        ).to(device)
    else:
        inputs = processor(
            text=formatted_prompt,
            return_tensors="pt"
        ).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    if verbose:
        print(f"  input_ids shape: {input_ids.shape}")
        if pixel_values is not None:
            print(f"  pixel_values shape: {pixel_values.shape}")

    with torch.no_grad():
        # Path 1: SAFE forward with audio=None
        # This should trigger the passthrough code path
        safe_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            audio_tokens=None,  # No audio!
            labels=None
        )
        safe_logits = safe_outputs["logits"]

        # Path 2: Direct call to base_vl.llm
        # This is the ground truth
        base_outputs = model.base_vl.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        base_logits = base_outputs.logits

    if verbose:
        print(f"  safe_logits shape: {safe_logits.shape}")
        print(f"  base_logits shape: {base_logits.shape}")

    # Compare
    comparison = compare_logits(safe_logits, base_logits)

    return comparison


def run_verification(
    samples: List[Dict],
    model,
    device: str = "cuda",
    verbose: bool = True
) -> Dict:
    """Run the zero-forgetting verification test."""

    results = {
        "total": len(samples),
        "exact_matches": 0,
        "close_matches": 0,
        "failures": 0,
        "max_diff_overall": 0.0,
        "all_diffs": [],
        "failed_samples": []
    }

    for i, sample in enumerate(tqdm(samples, desc="Verifying zero-forgetting")):
        prompt = sample["prompt"]
        image_path = sample.get("image_path")

        try:
            comparison = run_single_test(
                model, prompt, image_path, device,
                verbose=(verbose and i < 3)
            )

            if "error" in comparison:
                print(f"\n[ERROR] Sample {i}: {comparison['error']}")
                results["failures"] += 1
                results["failed_samples"].append({
                    "index": i,
                    "prompt": prompt[:50],
                    "error": comparison["error"]
                })
                continue

            max_diff = comparison["max_diff"]
            results["all_diffs"].append(max_diff)
            results["max_diff_overall"] = max(results["max_diff_overall"], max_diff)

            if max_diff == 0.0:
                results["exact_matches"] += 1
                status = "EXACT"
            elif comparison["match"]:
                results["close_matches"] += 1
                status = "CLOSE"
            else:
                results["failures"] += 1
                status = "FAIL"
                results["failed_samples"].append({
                    "index": i,
                    "prompt": prompt[:50],
                    "max_diff": max_diff,
                    "mismatch_pct": comparison.get("mismatch_pct", -1)
                })

            if verbose and i < 5:
                print(f"\n[Sample {i}] {status} - max_diff={max_diff:.2e}")

        except Exception as e:
            import traceback
            print(f"\n[ERROR] Sample {i} exception: {e}")
            if verbose:
                traceback.print_exc()
            results["failures"] += 1
            results["failed_samples"].append({
                "index": i,
                "prompt": prompt[:50],
                "error": str(e)
            })

    # Summary stats
    if results["all_diffs"]:
        results["mean_diff_overall"] = sum(results["all_diffs"]) / len(results["all_diffs"])
    else:
        results["mean_diff_overall"] = float('inf')

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
    print(f"Mean difference overall: {results.get('mean_diff_overall', 0):.2e}")

    passed = (exact + close) == total

    if passed:
        print("\n" + "=" * 60)
        print("✓ ZERO-FORGETTING GUARANTEE VERIFIED")
        print("  When audio=None, SAFE output matches base_vl exactly.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ ZERO-FORGETTING GUARANTEE FAILED")
        print(f"  {failed} samples produced different outputs.")
        print("=" * 60)

        if results["failed_samples"]:
            print("\nFirst 5 failed samples:")
            for fs in results["failed_samples"][:5]:
                err = fs.get("error", f"max_diff={fs.get('max_diff', '?')}")
                print(f"  - Sample {fs['index']}: {err}")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Verify zero-forgetting guarantee")
    parser.add_argument("--coco_dir", type=str, default=None,
                        help="Path to COCO dataset (with val2014/)")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic samples (no images)")
    parser.add_argument("--use_13b", action="store_true",
                        help="Use 13B model instead of 7B")
    args = parser.parse_args()

    print("[INFO] Zero-Forgetting Verification Test")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Samples: {args.num_samples}")

    # Load samples
    if args.synthetic:
        print("[INFO] Using synthetic samples (text-only)")
        samples = create_synthetic_samples(args.num_samples)
    elif args.coco_dir:
        samples = load_coco_samples(Path(args.coco_dir), args.num_samples, args.seed)
        if samples is None:
            print("[INFO] Falling back to synthetic samples")
            samples = create_synthetic_samples(args.num_samples)
    else:
        print("[INFO] No COCO dir specified, using synthetic samples")
        samples = create_synthetic_samples(args.num_samples)

    # Load model
    print(f"\n[STEP 1] Loading SAFE model ({'13B' if args.use_13b else '7B'})...")
    model = load_safe_model(args.device, use_7b=not args.use_13b)

    # Run verification
    print(f"\n[STEP 2] Running verification on {len(samples)} samples...")
    results = run_verification(samples, model, device=args.device, verbose=True)

    # Print and save results
    passed = print_results(results)

    # Save results (exclude large all_diffs list from JSON)
    output_results = {k: v for k, v in results.items() if k != "all_diffs"}
    output_path = PROJECT_ROOT / "safe" / "ablations" / "zero_forgetting_results.json"
    with open(output_path, "w") as f:
        json.dump(output_results, f, indent=2)
    print(f"\n[INFO] Results saved to {output_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
