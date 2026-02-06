#!/usr/bin/env python3
"""
Post-hoc gate calibration for KV-augment models.

Sweeps fusion gate values on validation data and picks the best gate
for AV composition performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.epic_sounds_avqa_composition.train_epic_sounds_avqa import (
    EpicSoundsAVQADataset,
    collate_epic,
    build_model_config,
    evaluate,
    set_seed,
)
from safe.models.safe_model import SAFEModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate KV gate on validation set")
    p.add_argument("--data-root", type=Path, default=Path("experiments/epic_sounds_avqa_composition/data"))
    p.add_argument("--val-manifest", type=Path, default=Path("experiments/epic_sounds_avqa_composition/data/manifests/validation.jsonl"))
    p.add_argument("--media-root", type=Path, default=Path("experiments/epic_sounds_avqa_composition/data/processed"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("experiments/epic_sounds_avqa_composition/results/gate_calibration.json"))
    p.add_argument("--architecture", type=str, default="kv_augment", choices=["pre_ffn", "kv_augment"])
    p.add_argument("--fusion-layers", type=str, default="1,5,9,13,17,21")
    p.add_argument("--num-audio-tokens", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-answer-tokens", type=int, default=16)
    p.add_argument("--llm-model", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gates", type=str, default="0.2,0.4,0.6,0.8,1.0,1.2")
    p.add_argument("--objective", type=str, default="av_composition_exact", choices=["av_composition_exact", "both_exact"])
    return p.parse_args()


def _score(metrics: dict, objective: str) -> float:
    if objective == "both_exact":
        return float(metrics.get("exact_match", 0.0))
    by_type = metrics.get("by_question_type", {})
    av = by_type.get("av_composition", {})
    return float(av.get("exact_match", 0.0))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reuse training script config builder
    model_args = argparse.Namespace(
        architecture=args.architecture,
        llm_model=args.llm_model,
        fusion_layers=args.fusion_layers,
        num_audio_tokens=args.num_audio_tokens,
        kv_query_rank=16,
        kv_bottleneck_dim=64,
        freeze_audio_encoder=True,
    )
    cfg = build_model_config(model_args)

    model = SAFEModel(**cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to_device(device)
    model.eval()

    tokenizer = model.base_vl.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = EpicSoundsAVQADataset(args.val_manifest, args.media_root)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_epic,
        pin_memory=torch.cuda.is_available(),
    )

    gates = [float(x.strip()) for x in args.gates.split(",") if x.strip()]
    results = []
    best = None

    for gate in gates:
        metrics = evaluate(
            model=model,
            dataloader=loader,
            tokenizer=tokenizer,
            device=device,
            modality="both",
            args=argparse.Namespace(fusion_gate=gate, max_answer_tokens=args.max_answer_tokens),
        )
        score = _score(metrics, args.objective)
        row = {"gate": gate, "score": score, "metrics": metrics}
        results.append(row)
        if best is None or score > best["score"]:
            best = row
        print(f"[gate={gate:.3f}] score={score:.3f} objective={args.objective}", flush=True)

    payload = {
        "checkpoint": str(args.checkpoint),
        "objective": args.objective,
        "best": best,
        "runs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] wrote calibration report to {args.output}", flush=True)


if __name__ == "__main__":
    main()
