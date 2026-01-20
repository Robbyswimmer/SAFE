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
    - normalized_entropy: H(π)/log(n_audio) - should decrease with training
    - delta_q_ratio: ||ΔQ|| / ||Q|| - should increase with training
    - audio_token_rms: per-token RMS of audio tokens (should be 0.5-5, not 20+)
    - logits: output logits

    Note: audio_attn_mass removed - meaningless in separate-branch architecture
    (softmax over audio-only branch always sums to 1.0)
    """
    # Default values in case kv_hook_manager isn't available
    diagnostics = {
        'normalized_entropy': 1.0,  # Uniform at init
        'delta_q_ratio': 0.0,       # Zero at init (expected)
        'audio_token_rms': 0.0,
        'logits': None,
    }

    # Check if KV augmentation is set up
    has_kv_manager = hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None

    if not has_kv_manager:
        print("⚠️  WARNING: model.kv_hook_manager is not set up!")
        print("   KV augmentation may not be initialized. Check SAFEModel setup.")

    # Compute per-token RMS for audio tokens
    if audio_tokens is not None and audio_tokens.numel() > 0:
        # audio_tokens shape: (batch, n_audio, hidden_dim)
        per_token_rms = audio_tokens.pow(2).mean(dim=-1).sqrt()  # (batch, n_audio)
        diagnostics['audio_token_rms'] = per_token_rms.mean().item()

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
            entropies = []
            dq_ratios = []

            for layer_idx, layer_diag in hook_diag.items():
                # Get normalized entropy from hook if available
                if 'normalized_entropy' in layer_diag:
                    entropies.append(layer_diag['normalized_entropy'])
                # Fallback: compute from attention weights
                elif 'audio_attn_weights' in layer_diag:
                    attn_w = layer_diag['audio_attn_weights']
                    norm_ent, _ = compute_normalized_entropy(attn_w, n_audio_tokens)
                    entropies.append(norm_ent)

                # Get ΔQ/Q ratio
                if 'delta_q_rms' in layer_diag and 'q_rms' in layer_diag:
                    q_rms = max(layer_diag['q_rms'], 1e-8)
                    ratio = layer_diag['delta_q_rms'] / q_rms
                    dq_ratios.append(ratio)

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

    # Key metrics: normalized_entropy (should decrease with training), ΔQ/Q (should increase)
    # Audio token RMS: should be 0.5-5, not 20+ (indicates scaling issue)
    print("\n┌─────────────────────┬───────────┬───────────┬───────────┐")
    print("│ Metric              │     A     │     B     │     C     │")
    print("├─────────────────────┼───────────┼───────────┼───────────┤")
    print(f"│ Normalized entropy  │ {diag_A['normalized_entropy']:9.4f} │ {diag_B['normalized_entropy']:9.4f} │ {diag_C['normalized_entropy']:9.4f} │")
    print(f"│ ΔQ/Q ratio          │ {diag_A['delta_q_ratio']:9.6f} │ {diag_B['delta_q_ratio']:9.6f} │ {diag_C['delta_q_ratio']:9.6f} │")
    print(f"│ Audio token RMS     │ {diag_A['audio_token_rms']:9.4f} │ {diag_B['audio_token_rms']:9.4f} │ {diag_C['audio_token_rms']:9.4f} │")
    print("└─────────────────────┴───────────┴───────────┴───────────┘")

    # Check audio token RMS (should be 0.5-5, not 20+)
    audio_rms = diag_A['audio_token_rms']
    if audio_rms > 10:
        print(f"⚠️  Audio token RMS = {audio_rms:.2f} (high, expected 0.5-5)")
        print("   Consider adding LayerNorm to audio projector output")
    elif audio_rms < 0.1:
        print(f"⚠️  Audio token RMS = {audio_rms:.4f} (too low, might vanish)")
    else:
        print(f"✓ Audio token RMS = {audio_rms:.2f} (healthy range)")

    # Logits comparison
    logits_A = diag_A['logits']
    logits_B = diag_B['logits']
    logits_C = diag_C['logits']

    diff_AB = (logits_A - logits_B).abs().mean().item()
    diff_AC = (logits_A - logits_C).abs().mean().item()
    diff_BC = (logits_B - logits_C).abs().mean().item()

    print("\n┌─────────────────────────────────────────┬───────────────┐")
    print("│ Comparison                              │     Value     │")
    print("├─────────────────────────────────────────┼───────────────┤")
    print(f"│ |logits_A - logits_B| (ΔQ effect)       │ {diff_AB:13.6f} │")
    print(f"│ |logits_A - logits_C| (audio content)   │ {diff_AC:13.6f} │")
    print(f"│ |logits_B - logits_C| (baseline diff)   │ {diff_BC:13.6f} │")
    print("└─────────────────────────────────────────┴───────────────┘")

    # === INTERPRETATION (adjusted for untrained model expectations) ===
    print("\n" + "-" * 40)
    print("INTERPRETATION (untrained model expectations):")
    print("-" * 40)

    # For UNTRAINED model:
    # - A≈B is EXPECTED (ΔQ is zero-initialized)
    # - A≠C is the KEY test (audio content matters - proves wiring is correct)
    # - entropy ≈ 1.0 is EXPECTED (uniform attention at init)
    # - ΔQ/Q ≈ 0 is EXPECTED (zero-init up_proj)

    critical_pass = False
    notes = []

    # CRITICAL CHECK: Does audio content affect output? (A≠C or B≠C)
    if diff_AC > 0.01 or diff_BC > 0.01:
        print(f"✓ CRITICAL: Audio content affects output (|A-C|={diff_AC:.4f}, |B-C|={diff_BC:.4f})")
        critical_pass = True
    elif diff_AC > 0.001 or diff_BC > 0.001:
        print(f"⚠️  CRITICAL: Weak audio content effect (|A-C|={diff_AC:.6f}, |B-C|={diff_BC:.6f})")
        critical_pass = True  # Still passing, just weak
    else:
        print(f"❌ CRITICAL: Audio content has NO effect! (|A-C|={diff_AC:.6f}, |B-C|={diff_BC:.6f})")
        print("   This means audio is not being injected or KV values are zero.")
        critical_pass = False

    # Expected for untrained: A≈B (ΔQ is zero-init)
    if diff_AB < 1e-4:
        notes.append(f"• A ≈ B: ΔQ=0 (expected at init, will change with training)")
    else:
        notes.append(f"• A ≠ B: ΔQ already has effect (diff={diff_AB:.4f}) - unexpected at init")

    # Expected for untrained: entropy ≈ 1.0
    if diag_A['normalized_entropy'] > 0.95:
        notes.append(f"• Entropy ≈ 1.0: Uniform attention (expected at init)")
    else:
        notes.append(f"• Entropy = {diag_A['normalized_entropy']:.3f}: Attention already focused (unexpected at init)")

    # Expected for untrained: ΔQ/Q ≈ 0
    if diag_A['delta_q_ratio'] < 0.001:
        notes.append(f"• ΔQ/Q ≈ 0: Query adapter dormant (expected at init)")
    else:
        notes.append(f"• ΔQ/Q = {diag_A['delta_q_ratio']*100:.3f}%: Query adapter active (unexpected at init)")

    print("\nStatus notes (for untrained model):")
    for note in notes:
        print(note)

    print("\n" + "-" * 40)
    if critical_pass:
        print("✓ FORWARD CHECK PASSED")
        print("  Audio content affects output - wiring is correct.")
        print("  After training, expect: entropy↓, ΔQ/Q↑, |A-B|↑")
        return True
    else:
        print("❌ FORWARD CHECK FAILED")
        print("  Audio content has no effect on output!")
        print("  Check: audio projector, KV adapters, hook manager wiring.")
        return False


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
    # Note: audio_attn_mass removed - meaningless in separate-branch architecture
    results = {
        "A": {"logits": [], "entropy": [], "dq_ratio": [], "audio_rms": []},
        "B": {"logits": [], "entropy": [], "dq_ratio": [], "audio_rms": []},
        "C": {"logits": [], "entropy": [], "dq_ratio": [], "audio_rms": []},
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
            results["A"]["entropy"].append(diag_A['normalized_entropy'])
            results["A"]["dq_ratio"].append(diag_A['delta_q_ratio'])
            results["A"]["audio_rms"].append(diag_A['audio_token_rms'])

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
            results["B"]["entropy"].append(diag_B['normalized_entropy'])
            results["B"]["dq_ratio"].append(diag_B['delta_q_ratio'])
            results["B"]["audio_rms"].append(diag_B['audio_token_rms'])

            # Restore
            if hasattr(model, 'kv_adapters'):
                for layer_idx, scale in original_scales.items():
                    model.kv_adapters[layer_idx].audio_query_adapter.scale.data = scale

            # Condition C: Null audio
            diag_C = get_attention_diagnostics(
                model, null_audio, input_ids, attention_mask, device, n_audio_tokens
            )
            results["C"]["logits"].append(diag_C['logits'])
            results["C"]["entropy"].append(diag_C['normalized_entropy'])
            results["C"]["dq_ratio"].append(diag_C['delta_q_ratio'])
            results["C"]["audio_rms"].append(diag_C['audio_token_rms'])

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
    avg_ent_A = sum(results["A"]["entropy"]) / len(results["A"]["entropy"])
    avg_ent_B = sum(results["B"]["entropy"]) / len(results["B"]["entropy"])
    avg_ent_C = sum(results["C"]["entropy"]) / len(results["C"]["entropy"])

    avg_dq_A = sum(results["A"]["dq_ratio"]) / len(results["A"]["dq_ratio"])

    avg_rms_A = sum(results["A"]["audio_rms"]) / len(results["A"]["audio_rms"])

    print("\n┌─────────────────────┬───────────┬───────────┬───────────┐")
    print("│ Metric (avg)        │     A     │     B     │     C     │")
    print("├─────────────────────┼───────────┼───────────┼───────────┤")
    print(f"│ Normalized entropy  │ {avg_ent_A:9.4f} │ {avg_ent_B:9.4f} │ {avg_ent_C:9.4f} │")
    print(f"│ ΔQ/Q ratio (A only) │ {avg_dq_A:9.6f} │    n/a    │    n/a    │")
    print(f"│ Audio token RMS     │ {avg_rms_A:9.4f} │    n/a    │    0.0    │")
    print("└─────────────────────┴───────────┴───────────┴───────────┘")

    # Check audio token RMS
    if avg_rms_A > 10:
        print(f"⚠️  Audio token RMS = {avg_rms_A:.2f} (high, expected 0.5-5)")
    elif avg_rms_A < 0.1:
        print(f"⚠️  Audio token RMS = {avg_rms_A:.4f} (too low)")
    else:
        print(f"✓ Audio token RMS = {avg_rms_A:.2f} (healthy range)")

    # Logits differences
    diff_AB = (logits_A - logits_B).abs().mean().item()
    diff_AC = (logits_A - logits_C).abs().mean().item()
    diff_BC = (logits_B - logits_C).abs().mean().item()

    # Per-sample logits differences (for variance)
    per_sample_diff_AB = [(results["A"]["logits"][i] - results["B"]["logits"][i]).abs().mean().item()
                          for i in range(processed)]
    per_sample_diff_AC = [(results["A"]["logits"][i] - results["C"]["logits"][i]).abs().mean().item()
                          for i in range(processed)]
    std_diff_AB = torch.tensor(per_sample_diff_AB).std().item()
    std_diff_AC = torch.tensor(per_sample_diff_AC).std().item()

    print("\n┌─────────────────────────────────────────┬───────────────┐")
    print("│ Comparison                              │     Value     │")
    print("├─────────────────────────────────────────┼───────────────┤")
    print(f"│ |logits_A - logits_B| mean (ΔQ effect)  │ {diff_AB:13.6f} │")
    print(f"│ |logits_A - logits_B| std               │ {std_diff_AB:13.6f} │")
    print(f"│ |logits_A - logits_C| mean (content)    │ {diff_AC:13.6f} │")
    print(f"│ |logits_A - logits_C| std               │ {std_diff_AC:13.6f} │")
    print(f"│ |logits_B - logits_C| mean (baseline)   │ {diff_BC:13.6f} │")
    print("└─────────────────────────────────────────┴───────────────┘")

    # === INTERPRETATION (adjusted for untrained vs trained model) ===
    print("\n" + "-" * 40)
    print("INTERPRETATION:")
    print("-" * 40)

    # Critical test: Audio content must affect output (A≠C or B≠C)
    # This is the MOST important check - proves audio pathway is wired correctly
    audio_wired = False
    if diff_AC > 0.01 or diff_BC > 0.01:
        print(f"✓ CRITICAL: Audio content affects output (|A-C|={diff_AC:.4f}, |B-C|={diff_BC:.4f})")
        audio_wired = True
    elif diff_AC > 0.001 or diff_BC > 0.001:
        print(f"⚠️  CRITICAL: Weak audio content effect (|A-C|={diff_AC:.6f}, |B-C|={diff_BC:.6f})")
        audio_wired = True
    else:
        print(f"❌ CRITICAL: Audio content has NO effect! (|A-C|={diff_AC:.6f}, |B-C|={diff_BC:.6f})")
        print("   Audio KV values may be zero or not injected.")

    # Training indicators (these should improve with training)
    print("\nTraining indicators (expected to improve with training):")

    # 1. ΔQ effect (should grow with training)
    if diff_AB > 0.01:
        print(f"  ✓ ΔQ effect: strong ({diff_AB:.4f}) - trained model behavior")
    elif diff_AB > 0.001:
        print(f"  ⚠️  ΔQ effect: weak ({diff_AB:.4f}) - needs more training")
    else:
        print(f"  • ΔQ effect: ~0 ({diff_AB:.6f}) - expected at init")

    # 2. Entropy (should decrease with training)
    if avg_ent_A < 0.85:
        print(f"  ✓ Entropy: focused ({avg_ent_A:.3f}) - trained model behavior")
    elif avg_ent_A < 0.95:
        print(f"  ⚠️  Entropy: somewhat focused ({avg_ent_A:.3f})")
    else:
        print(f"  • Entropy: uniform ({avg_ent_A:.3f}) - expected at init")

    # 3. ΔQ/Q ratio (should increase with training)
    if avg_dq_A > 0.01:
        print(f"  ✓ ΔQ/Q ratio: {avg_dq_A*100:.2f}% - trained model behavior")
    elif avg_dq_A > 0.001:
        print(f"  ⚠️  ΔQ/Q ratio: {avg_dq_A*100:.4f}% - needs more training")
    else:
        print(f"  • ΔQ/Q ratio: ~0 ({avg_dq_A*100:.6f}%) - expected at init")

    print(f"\n{'='*40}")
    if audio_wired:
        if diff_AB > 0.01 and avg_ent_A < 0.85:
            print("✓ ABLATION PASSED - Trained model working")
        else:
            print("✓ ABLATION PASSED (INIT) - Audio wired correctly")
            print("  ΔQ and entropy at init values - run training to see improvement")
        return True
    else:
        print("❌ ABLATION FAILED - Audio not affecting output")
        print("  Check: audio projector, KV adapters, hook manager")
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
            split="test",
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
