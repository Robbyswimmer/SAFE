#!/usr/bin/env python3
"""
Sanity Check for KV Augmentation with Query Adapter

Run ONE forward pass (no training) and verify:
1. Mean audio attention mass per layer (should be nonzero)
2. RMS ratio: RMS(g * audio_output) / RMS(text_output) - want 1%-10%
3. Delta logits: with audio vs null audio (should differ)

If any fail, don't waste a GPU day.

Usage:
    python scripts/sanity_check_kv_augment.py --data-path /path/to/AVE_Dataset
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.model_configs import get_config
from safe.models.safe_model import SAFEModel

# Import AVEDataset from the probe training script where it's defined
from train_audio_llm_probe import AVEDataset


def compute_rms(tensor: torch.Tensor) -> float:
    """Compute root mean square of tensor."""
    return torch.sqrt(torch.mean(tensor.float() ** 2)).item()


def run_sanity_check(args):
    print("=" * 60)
    print("KV Augmentation Sanity Check")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    config = get_config("kv_augment")
    print(f"\nConfig: kv_augment")
    print(f"  Fusion layers: {config['fusion_layer_indices']}")
    print(f"  Query adapter rank: {config['fusion_config'].get('query_adapter_rank', 16)}")
    print(f"  Min audio attention: {config['fusion_config'].get('min_audio_attention', 0.1)}")

    # Load model
    print("\nLoading SAFE model with KV augmentation...")
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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load one batch
    print(f"\nLoading dataset from: {args.data_path}")
    dataset = AVEDataset(
        data_path=args.data_path,
        split="val",
        max_audio_length=10.0,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))

    audio = batch["audio"].to(device)
    label = batch["label"].to(device)
    class_name = dataset.idx_to_class.get(label.item(), "unknown")
    print(f"Sample class: {class_name} (idx={label.item()})")
    print(f"Audio shape: {audio.shape}")

    # =========================================================================
    # SETUP: Forward pass to collect diagnostics
    # =========================================================================
    print("\n" + "=" * 60)
    print("Running forward pass to collect diagnostics...")
    print("=" * 60)

    # Enable attention weight capture
    if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
        model.kv_hook_manager.set_return_attention_weights(True)

    # Forward pass with audio
    with torch.no_grad():
        # Get audio tokens
        audio_features = model.audio_encoder(audio)
        audio_tokens = model.audio_projector(audio_features)
        print(f"Audio tokens shape: {audio_tokens.shape}")
        print(f"Audio tokens RMS (before norm): {compute_rms(audio_tokens):.4f}")

        # Prepare prompt
        prompt = "What sound is in this audio?"
        tokenizer = model.vl_model.language_model.tokenizer if hasattr(model.vl_model, 'language_model') else None
        if tokenizer is None:
            # Try to get tokenizer from processor
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(config["llm_model_name"])
            tokenizer = processor.tokenizer

        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Inject audio and forward
        model.enable_audio_training()

        if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
            model.kv_hook_manager.inject_audio(audio_tokens, gate=1.0)

        # Get hidden states through the model
        outputs = model.vl_model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    # =========================================================================
    # CHECK 1: Audio attention mass per layer
    # =========================================================================
    print("\n" + "=" * 60)
    print("CHECK 1: Audio Attention Mass Per Layer")
    print("=" * 60)

    if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
        diagnostics = model.kv_hook_manager.get_diagnostics()

        if diagnostics:
            print("\nPer-layer diagnostics:")
            for layer_idx, diag in sorted(diagnostics.items()):
                attn_mass = diag["audio_attn_mass"]
                status = "✓" if attn_mass >= 0.01 else "⚠️ LOW"
                print(f"  Layer {layer_idx}:")
                print(f"    audio_attn_mass: {attn_mass:.4f} ({attn_mass*100:.2f}%) {status}")
        else:
            print("  ⚠️  No diagnostics captured!")

        model.kv_hook_manager.clear_audio()

    # =========================================================================
    # CHECK 2: RMS ratio of audio branch vs text output
    # =========================================================================
    print("\n" + "=" * 60)
    print("CHECK 2: RMS Ratio (audio_branch / text_output)")
    print("=" * 60)

    if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
        # Re-run with audio to get fresh diagnostics
        model.kv_hook_manager.inject_audio(audio_tokens, gate=1.0)
        with torch.no_grad():
            _ = model.vl_model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        diagnostics = model.kv_hook_manager.get_diagnostics()
        if diagnostics:
            print("\nRMS diagnostics per layer:")
            print("  (Want rms_ratio in 1%-10% range)")
            for layer_idx, diag in sorted(diagnostics.items()):
                rms_ratio = diag["rms_ratio"]
                text_rms = diag["text_rms"]
                audio_rms = diag["audio_rms"]

                if rms_ratio < 0.001:
                    status = "⚠️ TOO LOW - audio won't matter"
                elif rms_ratio > 0.5:
                    status = "⚠️ TOO HIGH - may destabilize"
                elif 0.01 <= rms_ratio <= 0.1:
                    status = "✓ GOOD"
                else:
                    status = "~ OK"

                print(f"  Layer {layer_idx}:")
                print(f"    text_rms:  {text_rms:.4f}")
                print(f"    audio_rms: {audio_rms:.4f}")
                print(f"    rms_ratio: {rms_ratio:.4f} ({rms_ratio*100:.2f}%) {status}")

        model.kv_hook_manager.clear_audio()

    # Also check adapter parameters
    if hasattr(model, 'kv_adapters'):
        print("\nAdapter parameters:")
        for layer_idx, adapter in model.kv_adapters.items():
            scale = adapter.audio_scale.item()
            q_scale = adapter.audio_query_adapter.scale.item() if hasattr(adapter, 'audio_query_adapter') else 0
            print(f"  Layer {layer_idx}: audio_scale={scale:.4f}, query_adapter.scale={q_scale:.4f}")

    # =========================================================================
    # CHECK 3: Delta logits with vs without audio
    # =========================================================================
    print("\n" + "=" * 60)
    print("CHECK 3: Delta Logits (with audio vs null audio)")
    print("=" * 60)

    with torch.no_grad():
        # Forward WITH audio
        if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
            model.kv_hook_manager.inject_audio(audio_tokens, gate=1.0)

        outputs_with_audio = model.vl_model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits_with_audio = outputs_with_audio.logits[:, -1, :]  # Last token

        if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
            model.kv_hook_manager.clear_audio()

        # Forward WITHOUT audio (null audio = zeros)
        null_audio = torch.zeros_like(audio_tokens)
        if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
            model.kv_hook_manager.inject_audio(null_audio, gate=1.0)

        outputs_null_audio = model.vl_model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits_null_audio = outputs_null_audio.logits[:, -1, :]

        if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
            model.kv_hook_manager.clear_audio()

        # Forward with NO audio injection at all (baseline)
        outputs_no_injection = model.vl_model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits_no_injection = outputs_no_injection.logits[:, -1, :]

        # Compute differences
        delta_audio_vs_null = (logits_with_audio - logits_null_audio).abs().mean().item()
        delta_audio_vs_none = (logits_with_audio - logits_no_injection).abs().mean().item()
        delta_null_vs_none = (logits_null_audio - logits_no_injection).abs().mean().item()

        print(f"\nLogits comparison (mean absolute difference):")
        print(f"  |logits(audio) - logits(null)|: {delta_audio_vs_null:.6f}")
        print(f"  |logits(audio) - logits(none)|: {delta_audio_vs_none:.6f}")
        print(f"  |logits(null) - logits(none)|:  {delta_null_vs_none:.6f}")

        if delta_audio_vs_null < 1e-6:
            print("\n  ⚠️  CRITICAL: Audio has NO effect on logits!")
            print("      The audio branch is not connected properly.")
        elif delta_audio_vs_null < 0.01:
            print("\n  ⚠️  WARNING: Audio effect is very small.")
            print("      May need stronger initialization or gate.")
        else:
            print("\n  ✓ Audio affects logits - branch is connected.")

        # Check cosine similarity
        cos_sim = F.cosine_similarity(
            logits_with_audio.flatten().unsqueeze(0),
            logits_null_audio.flatten().unsqueeze(0)
        ).item()
        print(f"\n  Cosine similarity (audio vs null): {cos_sim:.6f}")
        if cos_sim > 0.9999:
            print("    ⚠️  Too similar - audio not influencing output")

    # =========================================================================
    # CHECK 4: ΔQ on/off test (query adapter must matter)
    # =========================================================================
    print("\n" + "=" * 60)
    print("CHECK 4: ΔQ On/Off Test (Query Adapter Effect)")
    print("=" * 60)

    with torch.no_grad():
        # Forward with ΔQ enabled (normal)
        if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
            model.kv_hook_manager.inject_audio(audio_tokens, gate=1.0)
        outputs_dq_on = model.vl_model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_dq_on = outputs_dq_on.hidden_states[-1][:, -8:, :]  # Last 8 positions
        logits_dq_on = outputs_dq_on.logits[:, -1, :]
        if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
            model.kv_hook_manager.clear_audio()

        # Temporarily zero out query adapter
        original_scales = {}
        if hasattr(model, 'kv_adapters'):
            for layer_idx, adapter in model.kv_adapters.items():
                if hasattr(adapter, 'audio_query_adapter'):
                    original_scales[layer_idx] = adapter.audio_query_adapter.scale.data.clone()
                    adapter.audio_query_adapter.scale.data.zero_()

        # Forward with ΔQ disabled (zeroed)
        if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
            model.kv_hook_manager.inject_audio(audio_tokens, gate=1.0)
        outputs_dq_off = model.vl_model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_dq_off = outputs_dq_off.hidden_states[-1][:, -8:, :]
        logits_dq_off = outputs_dq_off.logits[:, -1, :]
        if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
            model.kv_hook_manager.clear_audio()

        # Restore query adapter
        if hasattr(model, 'kv_adapters'):
            for layer_idx, scale in original_scales.items():
                model.kv_adapters[layer_idx].audio_query_adapter.scale.data = scale

        # Compare
        delta_logits_dq = (logits_dq_on - logits_dq_off).abs().mean().item()
        delta_hidden_dq = (hidden_dq_on - hidden_dq_off).abs().mean().item()
        cos_hidden_dq = F.cosine_similarity(
            hidden_dq_on.flatten().unsqueeze(0),
            hidden_dq_off.flatten().unsqueeze(0)
        ).item()

        print(f"\nΔQ enabled vs ΔQ=0 comparison:")
        print(f"  |logits_on - logits_off|: {delta_logits_dq:.6f}")
        print(f"  |hidden_on - hidden_off|: {delta_hidden_dq:.6f}")
        print(f"  cosine(hidden_on, hidden_off): {cos_hidden_dq:.6f}")

        if delta_logits_dq < 1e-6 and cos_hidden_dq > 0.9999:
            print("\n  ⚠️  CRITICAL: ΔQ has NO effect!")
            print("      Query adapter is not wired correctly.")
            dq_works = False
        elif delta_logits_dq < 0.001:
            print("\n  ⚠️  WARNING: ΔQ effect is very small.")
            dq_works = False
        else:
            print("\n  ✓ ΔQ affects output - query adapter is working.")
            dq_works = True

    # =========================================================================
    # CHECK 5: Attention Entropy (non-uniform = learning alignment)
    # =========================================================================
    print("\n" + "=" * 60)
    print("CHECK 5: Attention Entropy (Uniformity Check)")
    print("=" * 60)

    with torch.no_grad():
        if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
            model.kv_hook_manager.set_return_attention_weights(True)
            model.kv_hook_manager.inject_audio(audio_tokens, gate=1.0)

            _ = model.vl_model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            attn_weights = model.kv_hook_manager.get_attention_weights()
            model.kv_hook_manager.clear_audio()

            if attn_weights:
                print("\nAttention entropy per layer:")
                print("  (High ratio = uniform/forced, Low ratio = learned alignment)")
                uniform_layers = 0
                for layer_idx, weights in sorted(attn_weights.items()):
                    # weights: (batch, heads, q_len, n_audio)
                    n_audio = weights.shape[-1]
                    max_entropy = float(torch.tensor(n_audio).float().log())

                    # Normalize to distribution over audio tokens
                    attn_dist = weights.mean(dim=1)  # (batch, q_len, n_audio)
                    attn_dist = attn_dist / (attn_dist.sum(dim=-1, keepdim=True) + 1e-8)

                    # Compute entropy
                    entropy = -(attn_dist * (attn_dist + 1e-8).log()).sum(dim=-1).mean()
                    entropy_ratio = entropy.item() / max_entropy

                    status = ""
                    if entropy_ratio > 0.95:
                        status = "⚠️ UNIFORM (forced by reg?)"
                        uniform_layers += 1
                    elif entropy_ratio > 0.8:
                        status = "~ somewhat uniform"
                    else:
                        status = "✓ non-uniform (good)"

                    print(f"  Layer {layer_idx}: H={entropy.item():.3f} / {max_entropy:.3f} = {entropy_ratio:.2f} {status}")

                if uniform_layers == len(attn_weights):
                    print("\n  ⚠️  All layers have uniform attention!")
                    print("      Min-attn reg may be forcing mass without learning.")
            else:
                print("  ⚠️  No attention weights available")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("SANITY CHECK SUMMARY")
    print("=" * 60)

    issues = []
    warnings = []

    # Check diagnostics using new API
    if hasattr(model, 'kv_hook_manager') and model.kv_hook_manager is not None:
        # Re-run to get fresh diagnostics
        model.kv_hook_manager.inject_audio(audio_tokens, gate=1.0)
        with torch.no_grad():
            _ = model.vl_model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        diagnostics = model.kv_hook_manager.get_diagnostics()
        model.kv_hook_manager.clear_audio()

        if not diagnostics:
            issues.append("No diagnostics captured - forward hooks not working")
        else:
            # Check attention mass
            attn_masses = [d["audio_attn_mass"] for d in diagnostics.values()]
            min_attn = min(attn_masses)
            mean_attn = sum(attn_masses) / len(attn_masses)

            if min_attn < 0.001:
                issues.append(f"Audio attention too low: min={min_attn:.6f}")
            elif min_attn < 0.01:
                warnings.append(f"Audio attention low: min={min_attn:.4f}, mean={mean_attn:.4f}")

            # Check RMS ratio
            rms_ratios = [d["rms_ratio"] for d in diagnostics.values()]
            min_ratio = min(rms_ratios)
            max_ratio = max(rms_ratios)

            if max_ratio < 0.001:
                issues.append(f"RMS ratio too low: max={max_ratio:.6f} - audio won't affect output")
            elif max_ratio > 0.5:
                warnings.append(f"RMS ratio high: max={max_ratio:.4f} - may destabilize")

    # Check logits difference
    if delta_audio_vs_null < 1e-6:
        issues.append("Audio has NO effect on logits - branch disconnected")
    elif delta_audio_vs_null < 0.01:
        warnings.append(f"Audio effect small: {delta_audio_vs_null:.6f}")

    # Check ΔQ effect
    if not dq_works:
        issues.append("ΔQ has no effect - query adapter not wired correctly")

    if issues:
        print("\n❌ CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n⚠️  DO NOT proceed with full training until these are fixed!")
        return False
    elif warnings:
        print("\n⚠️  WARNINGS (may be OK, monitor closely):")
        for warning in warnings:
            print(f"   - {warning}")
        print("\n✓ Basic checks passed, but monitor these during training.")
        return True
    else:
        print("\n✓ All checks passed!")
        print("  - Audio attention is being computed")
        print("  - RMS ratio in acceptable range")
        print("  - Audio affects model output")
        print("\n✓ Safe to proceed with training.")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check for KV augmentation")
    parser.add_argument("--data-path", type=str, required=True, help="Path to AVE dataset")
    args = parser.parse_args()

    success = run_sanity_check(args)
    sys.exit(0 if success else 1)
