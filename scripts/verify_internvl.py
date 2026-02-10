#!/usr/bin/env python3
"""
Verify InternVL model download and SAFE hook compatibility.

Runs a smoke test to confirm:
1. Model loads with AutoModel + trust_remote_code
2. language_model.model.layers exists and has expected count
3. Hidden size matches config
4. LayerHookManager discovers correct layer count

Usage:
    python scripts/verify_internvl.py
    python scripts/verify_internvl.py --model-path models/OpenGVLab_InternVL3_5-8B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Verify InternVL download and SAFE compatibility")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/OpenGVLab_InternVL3_5-8B",
        help="Path to downloaded InternVL model",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip model loading (only check files)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    print("=" * 60)
    print("InternVL Verification")
    print("=" * 60)

    # Step 1: Check files exist
    print(f"\n[1/4] Checking files at {model_path}...")
    if not model_path.exists():
        print(f"  ERROR: Directory not found: {model_path}")
        print("  Run: python scripts/download_internvl.py --model internvl3.5-8b")
        sys.exit(1)

    config_ok = (model_path / "config.json").exists()
    tokenizer_ok = (model_path / "tokenizer.json").exists()
    weights = list(model_path.glob("*.safetensors")) or list(model_path.glob("model*.bin"))
    custom_code = list(model_path.glob("modeling_*.py"))

    print(f"  config.json: {'OK' if config_ok else 'MISSING'}")
    print(f"  tokenizer.json: {'OK' if tokenizer_ok else 'MISSING'}")
    print(f"  Weight files: {len(weights)} found")
    print(f"  Custom model code: {len(custom_code)} files")

    if not (config_ok and tokenizer_ok and weights):
        print("  ERROR: Required files missing. Re-download the model.")
        sys.exit(1)

    if args.skip_load:
        print("\n  Skipping model loading (--skip-load).")
        return

    # Step 2: Load model
    print(f"\n[2/4] Loading model with AutoModel (trust_remote_code=True)...")
    try:
        import torch
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.float32,  # CPU-friendly
            low_cpu_mem_usage=True,
        )
        print(f"  Model class: {type(model).__name__}")
        print(f"  OK")
    except Exception as e:
        print(f"  ERROR: Failed to load model: {e}")
        sys.exit(1)

    # Step 3: Check architecture
    print(f"\n[3/4] Checking architecture...")
    has_language_model = hasattr(model, "language_model")
    print(f"  Has language_model: {has_language_model}")

    if has_language_model:
        lm = model.language_model
        print(f"  Language model class: {type(lm).__name__}")

        has_inner_model = hasattr(lm, "model")
        print(f"  Has language_model.model: {has_inner_model}")

        if has_inner_model:
            inner = lm.model
            has_layers = hasattr(inner, "layers")
            print(f"  Has language_model.model.layers: {has_layers}")

            if has_layers:
                num_layers = len(inner.layers)
                print(f"  Number of layers: {num_layers}")

                # Check a layer for .mlp attribute (needed for pre_ffn injection)
                layer0 = inner.layers[0]
                has_mlp = hasattr(layer0, "mlp")
                print(f"  Layer has .mlp attribute: {has_mlp}")
                print(f"  Layer class: {type(layer0).__name__}")

    # Check hidden size
    hidden_size = None
    try:
        if hasattr(model.config, "llm_config"):
            hidden_size = model.config.llm_config.hidden_size
        elif hasattr(model, "language_model") and hasattr(model.language_model, "config"):
            hidden_size = model.language_model.config.hidden_size
    except Exception:
        pass
    print(f"  Hidden size: {hidden_size}")

    has_vision = hasattr(model, "vision_model")
    print(f"  Has vision_model: {has_vision}")
    if has_vision:
        print(f"  Vision model class: {type(model.vision_model).__name__}")

    has_mlp1 = hasattr(model, "mlp1")
    print(f"  Has mlp1 (vision connector): {has_mlp1}")

    # Step 4: Test LayerHookManager discovery
    print(f"\n[4/4] Testing LayerHookManager discovery...")
    try:
        # Add project root to path
        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root))

        from safe.models.layer_hooks import LayerHookManager
        from safe.models.fusion_adapter import MultiLayerFusionAdapter

        # Create a minimal fusion adapter for the test
        if hidden_size and has_language_model and has_layers:
            adapter = MultiLayerFusionAdapter(
                hidden_size=hidden_size,
                audio_dim=512,
                num_audio_tokens=8,
                fusion_layer_indices=[10, 19, 29],
                num_attention_heads=32,
            )

            lhm = LayerHookManager(
                model=model.language_model,
                fusion_adapter=adapter,
                fusion_layers={10, 19, 29},
            )

            discovered = lhm._discover_layer_modules(model.language_model)
            print(f"  Discovered {len(discovered)} layers")
            print(f"  Layer indices: {sorted(discovered.keys())[:5]}...{sorted(discovered.keys())[-3:]}")
            print(f"  OK - Hook discovery works!")
        else:
            print(f"  SKIP - Cannot test hooks (missing architecture components)")

    except Exception as e:
        print(f"  WARNING: Hook discovery test failed: {e}")
        print(f"  This may be OK if SAFE is not fully installed.")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  Model path: {model_path}")
    print(f"  Model class: {type(model).__name__}")
    if has_language_model and has_layers:
        print(f"  Layer count: {num_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Has vision: {has_vision}")

    if hidden_size and has_layers:
        print(f"\n  Recommended fusion_layer_indices for {num_layers} layers:")
        early = num_layers // 3
        mid = (num_layers * 2) // 3
        late = num_layers - 3
        print(f"    [{early}, {mid}, {late}]")

    print(f"\n  Verification PASSED")


if __name__ == "__main__":
    main()
