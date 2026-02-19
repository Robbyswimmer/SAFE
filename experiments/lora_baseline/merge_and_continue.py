#!/usr/bin/env python3
"""
LoRA merge utility for stage transition.

After Stage 1 (audio LoRA) training:
  1. Loads base model + audio LoRA checkpoint
  2. Calls model.merge_and_unload() to permanently merge LoRA into base weights
  3. Saves the merged model
  4. (Optionally) verifies the merge preserved predictions

Usage:
  python merge_and_continue.py \
    --base-model models/Qwen_Qwen3-8B \
    --lora-checkpoint checkpoints/lora_stage1/best_lora \
    --output-dir checkpoints/lora_stage1_merged
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    raise ImportError("peft is required. Install with: pip install peft")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    p.add_argument("--base-model", type=str, required=True,
                    help="Path to base LLM (e.g., models/Qwen_Qwen3-8B)")
    p.add_argument("--lora-checkpoint", type=str, required=True,
                    help="Path to trained LoRA adapter directory")
    p.add_argument("--output-dir", type=Path, required=True,
                    help="Directory to save merged model")
    p.add_argument("--verify", action="store_true",
                    help="Verify merge by comparing outputs on a test prompt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load base model ---
    print(f"[Merge] Loading base model: {args.base_model}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load LoRA adapter ---
    print(f"[Merge] Loading LoRA adapter: {args.lora_checkpoint}", flush=True)
    model = PeftModel.from_pretrained(base_model, args.lora_checkpoint)

    # --- Optional: verify pre-merge output ---
    pre_merge_output = None
    if args.verify:
        model.to(device).eval()
        test_prompt = (
            "<|im_start|>user\n"
            "Answer with exactly one short answer token (single word or number).\n"
            "Question: What instrument is being played?\n"
            "Answer:<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        enc = tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=8, do_sample=False)
        pre_merge_output = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"[Merge] Pre-merge output: {pre_merge_output!r}", flush=True)
        model.cpu()

    # --- Merge LoRA into base weights ---
    print("[Merge] Merging LoRA weights into base model...", flush=True)
    merged_model = model.merge_and_unload()
    print("[Merge] Merge complete.", flush=True)

    # --- Save merged model ---
    print(f"[Merge] Saving merged model to: {args.output_dir}", flush=True)
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Count parameters to verify
    total_params = sum(p.numel() for p in merged_model.parameters())
    print(f"[Merge] Merged model parameters: {total_params:,}", flush=True)

    # --- Optional: verify post-merge output ---
    if args.verify and pre_merge_output is not None:
        merged_model.to(device).eval()
        enc = tokenizer(
            "<|im_start|>user\n"
            "Answer with exactly one short answer token (single word or number).\n"
            "Question: What instrument is being played?\n"
            "Answer:<|im_end|>\n"
            "<|im_start|>assistant\n",
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = merged_model.generate(**enc, max_new_tokens=8, do_sample=False)
        post_merge_output = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"[Merge] Post-merge output: {post_merge_output!r}", flush=True)

        match = pre_merge_output == post_merge_output
        print(f"[Merge] Outputs match: {match}", flush=True)
        if not match:
            print("[Merge] WARNING: Pre/post merge outputs differ! "
                  "This may indicate a merge issue.", flush=True)

    # --- Save metadata ---
    metadata = {
        "base_model": args.base_model,
        "lora_checkpoint": args.lora_checkpoint,
        "total_params": total_params,
    }
    if args.verify:
        metadata["pre_merge_output"] = pre_merge_output
        metadata["post_merge_output"] = post_merge_output if args.verify else None
        metadata["outputs_match"] = match if args.verify else None

    with open(args.output_dir / "merge_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Merge] Done. Merged model saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
