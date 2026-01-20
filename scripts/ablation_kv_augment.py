#!/usr/bin/env python3
"""
Ablation Study for KV Augmentation with Query Adapter

Tests three conditions:
A: ΔQ enabled (full system)
B: ΔQ forced to 0 (no query adaptation)
C: Null audio (silence/zeros)

Modes:
- verify_text: Verify wrapped attention == original when no audio
- forward_check: Quick single-batch A/B/C comparison (catches wiring issues)
- ablation: Full ablation on multiple samples
- both: verify_text + ablation

Usage:
    python scripts/ablation_kv_augment.py --data-path /path/to/AVE_Dataset --mode verify_text
    python scripts/ablation_kv_augment.py --data-path /path/to/AVE_Dataset --mode forward_check
    python scripts/ablation_kv_augment.py --data-path /path/to/AVE_Dataset --mode ablation
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

from configs.model_configs import get_config
from safe.models.safe_model import SAFEModel

# Import AVEDataset from the probe training script where it's defined
from train_audio_llm_probe import AVEDataset


def compute_normalized_entropy(attn_weights, n_audio_tokens):
    """
    Compute normalized entropy H(π)/log(n_audio).
    ~1.0 = uniform (bad if stays there)
    <1.0 trending down = learning to focus (good)
    """
    # attn_weights: (batch, heads, seq, n_audio)
    # Sum over audio tokens dimension to get per-position distribution
    attn_dist = attn_weights.sum(dim=-1)  # (batch, heads, seq)

    # For audio attention specifically, look at attention TO audio tokens
    # attn_weights[:, :, :, :] is attention from each position to audio
    # We want the distribution over audio tokens for each query position

    # Normalize to get probability distribution over audio tokens
    attn_to_audio = attn_weights  # (batch, heads, seq, n_audio)
    # Clamp for numerical stability
    attn_to_audio = attn_to_audio.clamp(min=1e-10)

    # Compute entropy for each query position's attention over audio
    entropy = -(attn_to_audio * attn_to_audio.log()).sum(dim=-1)  # (batch, heads, seq)

    # Average across positions and heads
    mean_entropy = entropy.mean().item()

    # Normalize by max entropy (uniform distribution)
    max_entropy = math.log(n_audio_tokens)
    normalized_entropy = mean_entropy / max_entropy if max_entropy > 0 else 0.0

    return normalized_entropy, mean_entropy


def verify_text_path_identical(model, tokenizer, device, prompts=None):
    """
    Verify that with audio disabled, our attention matches the original exactly.

    Checks:
    1. Max absolute difference
    2. Relative error
    3. Argmax token match
    """
    print("=" * 60)
    print("VERIFY TEXT PATH IDENTICAL (Audio Disabled)")
    print("=" * 60)

    if prompts is None:
        prompts = [
            "What is happening in the audio?",
            "Describe the sound you hear.",
            "The audio contains",
        ]

    all_passed = True

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        print(f"\nTest prompt: '{prompt}'")
        print(f"  Input shape: {input_ids.shape}")

        with torch.no_grad():
            # Run 1: Original attention (no KV augmentation wrapper)
            if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
                model.kv_hook_manager.unwrap_attention_modules()

            outputs_original = model.base_vl.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            logits_original = outputs_original.logits

            # Re-wrap for augmented path
            if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
                model.kv_hook_manager.wrap_attention_modules()

            # Run 2: Augmented attention but NO audio injected
            outputs_wrapped = model.base_vl.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            logits_wrapped = outputs_wrapped.logits

            # Metric 1: Max absolute difference
            max_abs = (logits_original - logits_wrapped).abs().max().item()

            # Metric 2: Relative error
            ref_max = logits_original.abs().max().clamp(min=1e-6).item()
            rel_error = max_abs / ref_max

            # Metric 3: Argmax token match
            argmax_original = logits_original[:, -1, :].argmax(dim=-1)
            argmax_wrapped = logits_wrapped[:, -1, :].argmax(dim=-1)
            argmax_match = (argmax_original == argmax_wrapped).all().item()

            # Metric 4: Cosine similarity (for reference)
            cosine = F.cosine_similarity(
                logits_original.flatten().unsqueeze(0),
                logits_wrapped.flatten().unsqueeze(0)
            ).item()

            print(f"  Max absolute diff:  {max_abs:.2e}")
            print(f"  Relative error:     {rel_error:.2e}")
            print(f"  Argmax match:       {argmax_match}")
            print(f"  Cosine similarity:  {cosine:.8f}")

            # Check criteria
            # Allow some tolerance for floating point / attention implementation differences
            passed = max_abs < 1e-3 and rel_error < 1e-4 and argmax_match

            if passed:
                print(f"  ✓ PASS")
            else:
                print(f"  ❌ FAIL")
                all_passed = False

    print("\n" + "-" * 40)
    if all_passed:
        print("✓ TEXT PATH VERIFIED IDENTICAL")
        print("  Safe to proceed - no HF attention divergence.")
    else:
        print("❌ TEXT PATH DIVERGES")
        print("  The wrapped attention produces different outputs!")
        print("  Fix: ensure text branch exactly mirrors original attention.")

    return all_passed


def get_attention_diagnostics(model, audio_tokens, input_ids, attention_mask, device, n_audio_tokens):
    """
    Run forward pass and extract attention diagnostics.

    Returns dict with:
    - audio_attn_mass: attention mass on audio tokens
    - normalized_entropy: H(π)/log(n_audio)
    - delta_q_rms: ||ΔQ||_rms
    - q_rms: ||Q||_rms
    - delta_q_ratio: ||ΔQ|| / ||Q||
    - logits: output logits
    """
    # Default values in case kv_hook_manager isn't available
    diagnostics = {
        'audio_attn_mass': 0.0,
        'normalized_entropy': 1.0,
        'delta_q_ratio': 0.0,
        'logits': None,
    }

    # Check if KV augmentation is set up
    has_kv_manager = hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None

    if not has_kv_manager:
        print("⚠️  WARNING: model.kv_hook_manager is not set up!")
        print("   KV augmentation may not be initialized. Check SAFEModel setup.")

    # Inject audio if manager exists
    if has_kv_manager:
        model.kv_hook_manager.inject_audio(audio_tokens, gate=1.0)

    # Forward pass
    outputs = model.base_vl.llm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    diagnostics['logits'] = outputs.logits[:, -1, :].clone()

    # Get diagnostics from hook manager
    if has_kv_manager:
        hook_diag = model.kv_hook_manager.get_diagnostics()

        if not hook_diag:
            print("⚠️  WARNING: kv_hook_manager.get_diagnostics() returned empty!")
            print("   Attention modules may not be wrapped.")
        else:
            # Audio attention mass (average across layers)
            masses = []
            entropies = []
            dq_ratios = []

            for layer_idx, layer_diag in hook_diag.items():
                if 'audio_attn_mass' in layer_diag:
                    masses.append(layer_diag['audio_attn_mass'])

                # Get attention weights for entropy calculation
                if 'audio_attn_weights' in layer_diag:
                    attn_w = layer_diag['audio_attn_weights']
                    norm_ent, _ = compute_normalized_entropy(attn_w, n_audio_tokens)
                    entropies.append(norm_ent)

                # Get ΔQ/Q ratio
                if 'delta_q_rms' in layer_diag and 'q_rms' in layer_diag:
                    q_rms = max(layer_diag['q_rms'], 1e-8)
                    ratio = layer_diag['delta_q_rms'] / q_rms
                    dq_ratios.append(ratio)

            diagnostics['audio_attn_mass'] = sum(masses) / len(masses) if masses else 0.0
            diagnostics['normalized_entropy'] = sum(entropies) / len(entropies) if entropies else 1.0
            diagnostics['delta_q_ratio'] = sum(dq_ratios) / len(dq_ratios) if dq_ratios else 0.0

        model.kv_hook_manager.clear_audio()

    return diagnostics


def forward_check(model, tokenizer, device, dataset, config):
    """
    Quick single-batch A/B/C comparison.
    Catches wiring issues before running full ablation.
    """
    print("=" * 60)
    print("FORWARD CHECK: Quick A/B/C on Single Batch")
    print("=" * 60)

    n_audio_tokens = config["num_audio_tokens"]

    # Find a sample with valid audio path
    sample = None
    for i in range(min(100, len(dataset))):
        s = dataset[i]
        if s["audio"] is not None:
            sample = s
            print(f"Found valid sample at index {i}")
            break

    if sample is None:
        print("❌ ERROR: No valid audio files found in first 100 samples!")
        print("   Check that audio files exist in the dataset directory.")
        return False

    audio = sample["audio"]
    label = sample.get("label", "unknown")

    prompt = "What is happening in the audio?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"\nSample label: {label}")
    print(f"Prompt: '{prompt}'")

    with torch.no_grad():
        # Get audio features
        audio_tensor = model.audio_encoder.preprocess_audio(audio).to(device)
        audio_features = model.audio_encoder(audio_tensor)
        audio_tokens = model.audio_projector(audio_features)
        null_audio = torch.zeros_like(audio_tokens)

        print(f"Audio tokens shape: {audio_tokens.shape}")
        print(f"Audio tokens norm: {audio_tokens.norm().item():.2f}")

        # === CONDITION A: Full system (ΔQ enabled, real audio) ===
        print("\n--- Condition A: ΔQ enabled + real audio ---")
        diag_A = get_attention_diagnostics(
            model, audio_tokens, input_ids, attention_mask, device, n_audio_tokens
        )

        # === CONDITION B: ΔQ forced to 0 ===
        print("\n--- Condition B: ΔQ=0 + real audio ---")
        # Zero out query adapter
        original_scales = {}
        if hasattr(model, 'kv_adapters'):
            for layer_idx, adapter in model.kv_adapters.items():
                if hasattr(adapter, 'audio_query_adapter'):
                    original_scales[layer_idx] = adapter.audio_query_adapter.scale.data.clone()
                    adapter.audio_query_adapter.scale.data.zero_()

        diag_B = get_attention_diagnostics(
            model, audio_tokens, input_ids, attention_mask, device, n_audio_tokens
        )

        # Restore query adapter
        if hasattr(model, 'kv_adapters'):
            for layer_idx, scale in original_scales.items():
                model.kv_adapters[layer_idx].audio_query_adapter.scale.data = scale

        # === CONDITION C: Null audio ===
        print("\n--- Condition C: ΔQ enabled + null audio ---")
        diag_C = get_attention_diagnostics(
            model, null_audio, input_ids, attention_mask, device, n_audio_tokens
        )

    # === RESULTS TABLE ===
    print("\n" + "=" * 60)
    print("FORWARD CHECK RESULTS")
    print("=" * 60)

    print("\n┌─────────────────────┬───────────┬───────────┬───────────┐")
    print("│ Metric              │     A     │     B     │     C     │")
    print("├─────────────────────┼───────────┼───────────┼───────────┤")
    print(f"│ Audio attn mass     │ {diag_A['audio_attn_mass']:9.4f} │ {diag_B['audio_attn_mass']:9.4f} │ {diag_C['audio_attn_mass']:9.4f} │")
    print(f"│ Normalized entropy  │ {diag_A['normalized_entropy']:9.4f} │ {diag_B['normalized_entropy']:9.4f} │ {diag_C['normalized_entropy']:9.4f} │")
    print(f"│ ΔQ/Q ratio          │ {diag_A['delta_q_ratio']:9.6f} │ {diag_B['delta_q_ratio']:9.6f} │ {diag_C['delta_q_ratio']:9.6f} │")
    print("└─────────────────────┴───────────┴───────────┴───────────┘")

    # Logits comparison
    logits_A = diag_A['logits']
    logits_B = diag_B['logits']
    logits_C = diag_C['logits']

    diff_AB = (logits_A - logits_B).abs().mean().item()
    diff_AC = (logits_A - logits_C).abs().mean().item()
    diff_BC = (logits_B - logits_C).abs().mean().item()

    # Attention mass changes
    mass_diff_AB = diag_A['audio_attn_mass'] - diag_B['audio_attn_mass']
    mass_diff_AC = diag_A['audio_attn_mass'] - diag_C['audio_attn_mass']

    print("\n┌─────────────────────────────────────────┬───────────────┐")
    print("│ Comparison                              │     Value     │")
    print("├─────────────────────────────────────────┼───────────────┤")
    print(f"│ |logits_A - logits_B| (ΔQ effect)       │ {diff_AB:13.6f} │")
    print(f"│ |logits_A - logits_C| (audio content)   │ {diff_AC:13.6f} │")
    print(f"│ |logits_B - logits_C| (baseline diff)   │ {diff_BC:13.6f} │")
    print(f"│ mass_A - mass_B (ΔQ → attention)        │ {mass_diff_AB:+13.6f} │")
    print(f"│ mass_A - mass_C (content → attention)   │ {mass_diff_AC:+13.6f} │")
    print("└─────────────────────────────────────────┴───────────────┘")

    # === INTERPRETATION ===
    print("\n" + "-" * 40)
    print("INTERPRETATION:")
    print("-" * 40)

    issues = []
    goods = []

    # Check 1: Does ΔQ change anything?
    if diff_AB < 1e-4:
        issues.append("❌ A ≈ B: ΔQ has NO effect on logits")
    else:
        goods.append(f"✓ A ≠ B: ΔQ changes logits (diff={diff_AB:.4f})")

    # Check 2: Does ΔQ increase attention mass?
    if mass_diff_AB <= 0:
        issues.append(f"❌ mass_A <= mass_B: ΔQ not increasing attention ({mass_diff_AB:+.4f})")
    else:
        goods.append(f"✓ mass_A > mass_B: ΔQ increases attention ({mass_diff_AB:+.4f})")

    # Check 3: Does audio content matter?
    if diff_AC < 1e-4 and diff_BC < 1e-4:
        issues.append("❌ A ≈ B ≈ C: Audio content has no effect")
    elif diff_AC > diff_BC:
        goods.append(f"✓ Real audio differs from null (diff_AC={diff_AC:.4f} > diff_BC={diff_BC:.4f})")

    # Check 4: Is entropy uniform (bad) or focused?
    if diag_A['normalized_entropy'] > 0.95:
        issues.append(f"⚠️  Entropy ≈ 1.0: Attention is uniform (possibly forced by reg)")
    elif diag_A['normalized_entropy'] < 0.8:
        goods.append(f"✓ Entropy < 0.8: Attention is focused ({diag_A['normalized_entropy']:.3f})")

    # Check 5: Is ΔQ/Q ratio reasonable?
    if diag_A['delta_q_ratio'] < 0.001:
        issues.append(f"⚠️  ΔQ/Q < 0.1%: Query adapter may be too weak ({diag_A['delta_q_ratio']*100:.4f}%)")
    elif diag_A['delta_q_ratio'] > 0.01:
        goods.append(f"✓ ΔQ/Q > 1%: Query adapter has meaningful magnitude ({diag_A['delta_q_ratio']*100:.2f}%)")

    for g in goods:
        print(g)
    for i in issues:
        print(i)

    if not issues:
        print("\n✓ FORWARD CHECK PASSED - System appears wired correctly")
        return True
    elif len(issues) >= 3:
        print("\n❌ FORWARD CHECK FAILED - Multiple issues detected")
        return False
    else:
        print("\n⚠️  FORWARD CHECK INCONCLUSIVE - Some issues detected")
        return None


def run_ablation(model, tokenizer, device, dataset, config, num_samples=10):
    """
    Run full ablation: A (ΔQ enabled) vs B (ΔQ=0) vs C (null audio)
    with proper metrics.
    """
    print("=" * 60)
    print("ABLATION: ΔQ enabled vs ΔQ=0 vs Null Audio")
    print("=" * 60)

    n_audio_tokens = config["num_audio_tokens"]
    num_samples = min(num_samples, len(dataset))

    prompt = "What is happening in the audio?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Storage for per-sample results
    results = {
        "A": {"logits": [], "mass": [], "entropy": [], "dq_ratio": []},
        "B": {"logits": [], "mass": [], "entropy": [], "dq_ratio": []},
        "C": {"logits": [], "mass": [], "entropy": [], "dq_ratio": []},
    }

    print(f"\nRunning ablation on {num_samples} samples...")

    processed = 0
    idx = 0
    while processed < num_samples and idx < len(dataset):
        sample = dataset[idx]
        audio = sample["audio"]
        idx += 1

        # Skip samples without valid audio
        if audio is None:
            continue

        with torch.no_grad():
            # Get audio features
            audio_tensor = model.audio_encoder.preprocess_audio(audio).to(device)
            audio_features = model.audio_encoder(audio_tensor)
            audio_tokens = model.audio_projector(audio_features)
            null_audio = torch.zeros_like(audio_tokens)

            # Condition A: Full system
            diag_A = get_attention_diagnostics(
                model, audio_tokens, input_ids, attention_mask, device, n_audio_tokens
            )
            results["A"]["logits"].append(diag_A['logits'])
            results["A"]["mass"].append(diag_A['audio_attn_mass'])
            results["A"]["entropy"].append(diag_A['normalized_entropy'])
            results["A"]["dq_ratio"].append(diag_A['delta_q_ratio'])

            # Condition B: ΔQ=0
            original_scales = {}
            if hasattr(model, 'kv_adapters'):
                for layer_idx, adapter in model.kv_adapters.items():
                    if hasattr(adapter, 'audio_query_adapter'):
                        original_scales[layer_idx] = adapter.audio_query_adapter.scale.data.clone()
                        adapter.audio_query_adapter.scale.data.zero_()

            diag_B = get_attention_diagnostics(
                model, audio_tokens, input_ids, attention_mask, device, n_audio_tokens
            )
            results["B"]["logits"].append(diag_B['logits'])
            results["B"]["mass"].append(diag_B['audio_attn_mass'])
            results["B"]["entropy"].append(diag_B['normalized_entropy'])
            results["B"]["dq_ratio"].append(diag_B['delta_q_ratio'])

            # Restore
            if hasattr(model, 'kv_adapters'):
                for layer_idx, scale in original_scales.items():
                    model.kv_adapters[layer_idx].audio_query_adapter.scale.data = scale

            # Condition C: Null audio
            diag_C = get_attention_diagnostics(
                model, null_audio, input_ids, attention_mask, device, n_audio_tokens
            )
            results["C"]["logits"].append(diag_C['logits'])
            results["C"]["mass"].append(diag_C['audio_attn_mass'])
            results["C"]["entropy"].append(diag_C['normalized_entropy'])
            results["C"]["dq_ratio"].append(diag_C['delta_q_ratio'])

        processed += 1
        if processed % 5 == 0:
            print(f"  Processed {processed}/{num_samples} samples")

    # === AGGREGATE RESULTS ===
    print("\n" + "=" * 60)
    print(f"ABLATION RESULTS (over {processed} samples)")
    print("=" * 60)

    if processed == 0:
        print("❌ ERROR: No valid audio samples were processed!")
        return False

    # Stack logits
    logits_A = torch.cat(results["A"]["logits"], dim=0)
    logits_B = torch.cat(results["B"]["logits"], dim=0)
    logits_C = torch.cat(results["C"]["logits"], dim=0)

    # Average metrics
    avg_mass_A = sum(results["A"]["mass"]) / len(results["A"]["mass"])
    avg_mass_B = sum(results["B"]["mass"]) / len(results["B"]["mass"])
    avg_mass_C = sum(results["C"]["mass"]) / len(results["C"]["mass"])

    avg_ent_A = sum(results["A"]["entropy"]) / len(results["A"]["entropy"])
    avg_ent_B = sum(results["B"]["entropy"]) / len(results["B"]["entropy"])
    avg_ent_C = sum(results["C"]["entropy"]) / len(results["C"]["entropy"])

    avg_dq_A = sum(results["A"]["dq_ratio"]) / len(results["A"]["dq_ratio"])

    print("\n┌─────────────────────┬───────────┬───────────┬───────────┐")
    print("│ Metric (avg)        │     A     │     B     │     C     │")
    print("├─────────────────────┼───────────┼───────────┼───────────┤")
    print(f"│ Audio attn mass     │ {avg_mass_A:9.4f} │ {avg_mass_B:9.4f} │ {avg_mass_C:9.4f} │")
    print(f"│ Normalized entropy  │ {avg_ent_A:9.4f} │ {avg_ent_B:9.4f} │ {avg_ent_C:9.4f} │")
    print(f"│ ΔQ/Q ratio (A only) │ {avg_dq_A:9.6f} │    n/a    │    n/a    │")
    print("└─────────────────────┴───────────┴───────────┴───────────┘")

    # Logits differences
    diff_AB = (logits_A - logits_B).abs().mean().item()
    diff_AC = (logits_A - logits_C).abs().mean().item()
    diff_BC = (logits_B - logits_C).abs().mean().item()

    # Per-sample logits differences (for variance)
    per_sample_diff_AB = [(results["A"]["logits"][i] - results["B"]["logits"][i]).abs().mean().item()
                          for i in range(processed)]
    std_diff_AB = torch.tensor(per_sample_diff_AB).std().item()

    # Mass differences
    mass_diff_AB = avg_mass_A - avg_mass_B
    mass_diff_AC = avg_mass_A - avg_mass_C

    print("\n┌─────────────────────────────────────────┬───────────────┐")
    print("│ Comparison                              │     Value     │")
    print("├─────────────────────────────────────────┼───────────────┤")
    print(f"│ |logits_A - logits_B| mean              │ {diff_AB:13.6f} │")
    print(f"│ |logits_A - logits_B| std               │ {std_diff_AB:13.6f} │")
    print(f"│ |logits_A - logits_C| mean              │ {diff_AC:13.6f} │")
    print(f"│ |logits_B - logits_C| mean              │ {diff_BC:13.6f} │")
    print(f"│ mass_A - mass_B                         │ {mass_diff_AB:+13.6f} │")
    print(f"│ mass_A - mass_C                         │ {mass_diff_AC:+13.6f} │")
    print("└─────────────────────────────────────────┴───────────────┘")

    # === INTERPRETATION ===
    print("\n" + "-" * 40)
    print("INTERPRETATION:")
    print("-" * 40)

    # Scoring
    score = 0
    max_score = 5

    # 1. ΔQ effect on logits
    if diff_AB > 0.01:
        print(f"✓ [1/5] ΔQ changes logits significantly (diff={diff_AB:.4f} > 0.01)")
        score += 1
    elif diff_AB > 0.001:
        print(f"⚠️  [0.5/5] ΔQ has weak effect on logits (diff={diff_AB:.4f})")
        score += 0.5
    else:
        print(f"❌ [0/5] ΔQ has NO effect on logits (diff={diff_AB:.6f})")

    # 2. ΔQ increases attention mass
    if mass_diff_AB > 0.01:
        print(f"✓ [2/5] ΔQ increases attention mass (+{mass_diff_AB:.4f})")
        score += 1
    elif mass_diff_AB > 0:
        print(f"⚠️  [1.5/5] ΔQ slightly increases attention (+{mass_diff_AB:.4f})")
        score += 0.5
    else:
        print(f"❌ [1/5] ΔQ does NOT increase attention ({mass_diff_AB:+.4f})")

    # 3. Audio content matters
    if diff_AC > diff_BC * 1.5:
        print(f"✓ [3/5] Real audio differs from null (ratio={diff_AC/max(diff_BC, 1e-6):.2f}x)")
        score += 1
    else:
        print(f"⚠️  [2.5/5] Audio content effect unclear")
        score += 0.5

    # 4. Entropy not uniform
    if avg_ent_A < 0.85:
        print(f"✓ [4/5] Attention is focused (entropy={avg_ent_A:.3f})")
        score += 1
    elif avg_ent_A < 0.95:
        print(f"⚠️  [3.5/5] Attention somewhat focused (entropy={avg_ent_A:.3f})")
        score += 0.5
    else:
        print(f"❌ [3/5] Attention is uniform/forced (entropy={avg_ent_A:.3f})")

    # 5. ΔQ/Q ratio reasonable
    if avg_dq_A > 0.01:
        print(f"✓ [5/5] ΔQ magnitude healthy ({avg_dq_A*100:.2f}% of Q)")
        score += 1
    elif avg_dq_A > 0.001:
        print(f"⚠️  [4.5/5] ΔQ magnitude weak ({avg_dq_A*100:.3f}% of Q)")
        score += 0.5
    else:
        print(f"❌ [4/5] ΔQ magnitude tiny ({avg_dq_A*100:.4f}% of Q)")

    print(f"\n{'='*40}")
    print(f"ABLATION SCORE: {score}/{max_score}")
    print(f"{'='*40}")

    if score >= 4:
        print("✓ ABLATION PASSED - System is working")
        return True
    elif score >= 2.5:
        print("⚠️  ABLATION INCONCLUSIVE - Partial signal detected")
        return None
    else:
        print("❌ ABLATION FAILED - Audio branch not contributing")
        return False


def main():
    parser = argparse.ArgumentParser(description="Ablation study for KV augmentation")
    parser.add_argument("--data-path", type=str, required=True, help="Path to AVE dataset")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["verify_text", "forward_check", "ablation", "both", "all"],
                        help="Which test to run")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples for ablation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    config = get_config("kv_augment")

    # Load model
    print("\nLoading SAFE model...")
    model = SAFEModel(
        llm_model_name=config["llm_model_name"],
        audio_encoder_type=config["audio_encoder_type"],
        audio_encoder_config=config["audio_encoder_config"],
        llm_hidden_size=config["llm_hidden_size"],
        audio_embed_dim=config["audio_embed_dim"],
        num_audio_tokens=config["num_audio_tokens"],
        fusion_layer_indices=config["fusion_layer_indices"],
        fusion_config=config["fusion_config"],
    )
    model = model.to(device)
    model.eval()

    # Initialize KV hook manager if kv_adapters exist
    if hasattr(model, 'kv_adapters') and model.kv_adapters is not None:
        from safe.models.kv_augmentation import KVAugmentationHookManager
        fusion_layers = config["fusion_layer_indices"]
        print(f"Initializing KV augmentation at layers: {fusion_layers}")

        if model.kv_hook_manager is None:
            model.kv_hook_manager = KVAugmentationHookManager(
                model=model.base_vl.llm,
                kv_adapters=model.kv_adapters,
                fusion_layer_indices=fusion_layers,
            )
        model.kv_hook_manager.wrap_attention_modules()
        print("✓ KV hook manager initialized and attention modules wrapped")
    else:
        print("⚠️  WARNING: model.kv_adapters is None - KV augmentation not set up!")
        print("   Check that fusion_config has 'fusion_mode': 'kv_augment'")

    # Get tokenizer
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config["llm_model_name"])
    tokenizer = processor.tokenizer

    # Load dataset (needed for forward_check and ablation)
    dataset = None
    if args.mode in ["forward_check", "ablation", "both", "all"]:
        print(f"\nLoading dataset from: {args.data_path}")
        dataset = AVEDataset(
            data_path=args.data_path,
            split="val",
            max_length=10.0,
        )

    # Run tests based on mode
    if args.mode in ["verify_text", "both", "all"]:
        text_ok = verify_text_path_identical(model, tokenizer, device)
        if not text_ok and args.mode in ["both", "all"]:
            print("\n⚠️  Skipping further tests - fix text path first!")
            sys.exit(1)

    if args.mode in ["forward_check", "all"]:
        print("\n")
        forward_ok = forward_check(model, tokenizer, device, dataset, config)
        if forward_ok is False and args.mode == "all":
            print("\n⚠️  Forward check failed - fix wiring before full ablation!")
            sys.exit(1)

    if args.mode in ["ablation", "both", "all"]:
        print("\n")
        ablation_ok = run_ablation(model, tokenizer, device, dataset, config, args.num_samples)
        sys.exit(0 if ablation_ok else 1)


if __name__ == "__main__":
    main()
