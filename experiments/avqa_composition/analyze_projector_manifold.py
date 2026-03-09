#!/usr/bin/env python3
"""
Analyze whether projector tokens land in backbone-compatible, modality-specific regions.

Primary use case:
  - concat / RKCA checkpoints on MUSIC-AVQA
  - compare audio projector token summaries to audio-text anchors
  - compare vision projector token summaries to vision-text anchors
  - measure role separation, retrieval, linear CKA, and subspace overlap
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.avqa_composition.train_avqa_composition import (  # type: ignore
    AVQA_ANSWER_VOCAB,
    ManifestAVQADataset,
    build_modality_aware_questions,
    build_model_config,
    collate_avqa,
    extract_answer,
    parse_args as _unused_parse_args,  # noqa: F401
    resolve_modality_batch,
    set_seed,
)
from safe.models.safe_model import SAFEModel


def pool_masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.float()
    y = y.float()
    n = min(int(x.size(0)), int(y.size(0)))
    if n <= 1:
        return 0.0
    x = x[:n]
    y = y[:n]
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    hsic = (x.t().matmul(y)).pow(2).sum()
    norm_x = (x.t().matmul(x)).pow(2).sum().clamp_min(eps).sqrt()
    norm_y = (y.t().matmul(y)).pow(2).sum().clamp_min(eps).sqrt()
    return float((hsic / (norm_x * norm_y + eps)).item())


def fit_basis(samples: torch.Tensor, rank: int) -> Optional[torch.Tensor]:
    if samples.ndim != 2 or samples.size(0) < 2:
        return None
    centered = samples.float() - samples.float().mean(dim=0, keepdim=True)
    max_rank = min(int(rank), int(centered.size(0) - 1), int(centered.size(1)))
    if max_rank <= 0:
        return None
    try:
        _, _, v = torch.pca_lowrank(centered, q=max_rank, center=False, niter=2)
        basis = v[:, :max_rank].contiguous()
    except Exception:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:max_rank].transpose(0, 1).contiguous()
    return F.normalize(basis, dim=0)


def subspace_overlap(a: torch.Tensor, b: torch.Tensor, rank: int) -> Optional[float]:
    basis_a = fit_basis(a, rank=rank)
    basis_b = fit_basis(b, rank=rank)
    if basis_a is None or basis_b is None:
        return None
    gram = basis_a.transpose(0, 1).matmul(basis_b)
    svals = torch.linalg.svdvals(gram.float())
    return float((svals.pow(2).mean()).item())


def make_anchor_statement(answer: str) -> str:
    ans = str(answer).replace("_", " ")
    return f"The answer is {ans}."


def build_analysis_namespace(args: argparse.Namespace) -> argparse.Namespace:
    # Reuse the training config builder by creating a compatible namespace.
    ns = argparse.Namespace(
        model_config=args.model_config,
        llm_model=args.llm_model,
        num_audio_tokens=args.num_audio_tokens,
        fusion_layers=None,
        audio_fusion_layers=None,
        vision_fusion_layers=None,
        freeze_audio_encoder=True,
        label_smoothing=None,
        bottleneck_dim=None,
        learned_gate=False,
        learned_gate_init=0.0,
        delta_norm_cap_ratio=0.0,
        delta_norm_cap_eps=1e-6,
        gate_depth_decay=1.0,
        audio_gate_depth_decay=1.0,
        vision_gate_depth_decay=1.0,
        icm_enable=False,
        icm_dim=512,
        icm_heads=8,
        icm_layers=1,
        icm_dropout=0.1,
        icm_gate_init=-2.0,
        icm_min_modalities=2,
        icm_util_target=0.7,
        slim_projector=args.slim_projector,
        fusion_gate=args.fusion_gate,
        modality_aware_prompts=bool(args.modality_aware_prompts),
        text_prompt_prefix=getattr(args, "text_prompt_prefix", ""),
        audio_prompt_prefix=getattr(args, "audio_prompt_prefix", ""),
        image_prompt_prefix=getattr(args, "image_prompt_prefix", ""),
        both_prompt_prefix=getattr(args, "both_prompt_prefix", ""),
    )
    return ns


def collect_projector_summaries(
    model: SAFEModel,
    dataloader: DataLoader,
    device: torch.device,
    prompt_args: argparse.Namespace,
    modality: str,
    max_samples: int,
) -> Tuple[torch.Tensor, List[str]]:
    summaries: List[torch.Tensor] = []
    answers: List[str] = []
    model.eval()
    collected = 0
    with torch.no_grad():
        for batch in dataloader:
            if max_samples > 0 and collected >= max_samples:
                break
            mm = resolve_modality_batch(batch, modality)
            inputs = model.prepare_multimodal_inputs(
                text=build_modality_aware_questions(batch["questions"], modality, prompt_args),
                images=mm["images"],
                audio=mm["audio"],
                answers=None,
                device=str(device),
                training_mode=False,
            )
            audio_tokens = inputs.pop("audio_tokens", None)
            audio_mask = inputs.pop("audio_attention_mask", None)
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=None,
                pixel_values=inputs.get("pixel_values"),
                audio_tokens=audio_tokens,
                audio_attention_mask=audio_mask,
                gate=prompt_args.fusion_gate,
                output_hidden_states=False,
            )
            key = "audio_projector_tokens" if modality == "audio" else "vision_projector_tokens"
            token_states = outputs.get(key) if isinstance(outputs, dict) else None
            if token_states is None:
                raise RuntimeError(
                    f"Projector token outputs unavailable for modality='{modality}'. "
                    "Use a concat/RKCA model config for this analysis."
                )
            pooled = token_states.float().mean(dim=1).cpu()
            batch_answers = [extract_answer(a) for a in batch["answers"]]
            for vec, ans in zip(pooled, batch_answers):
                summaries.append(vec)
                answers.append(ans)
                collected += 1
                if max_samples > 0 and collected >= max_samples:
                    break

    if not summaries:
        raise RuntimeError(f"No projector summaries collected for modality='{modality}'")
    return torch.stack(summaries, dim=0), answers


def collect_anchor_summaries(
    model: SAFEModel,
    answers: Sequence[str],
    device: torch.device,
    prompt_args: argparse.Namespace,
    modality: str,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    unique_answers = sorted({extract_answer(a) for a in answers})
    prompts = [build_modality_aware_questions(make_anchor_statement(ans), modality, prompt_args) for ans in unique_answers]
    out: Dict[str, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            batch_answers = unique_answers[start:start + batch_size]
            inputs = model.prepare_multimodal_inputs(
                text=batch_prompts,
                images=None,
                audio=None,
                answers=None,
                device=str(device),
                training_mode=False,
            )
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=None,
                pixel_values=inputs.get("pixel_values"),
                audio_tokens=None,
                audio_attention_mask=None,
                gate=0.0,
                output_hidden_states=True,
            )
            last_hidden = outputs.get("hidden_states") if isinstance(outputs, dict) else None
            if last_hidden is None:
                raise RuntimeError("Anchor hidden states unavailable from model forward pass")
            pooled = pool_masked_mean(last_hidden.float(), inputs["attention_mask"]).cpu()
            for ans, vec in zip(batch_answers, pooled):
                out[ans] = vec
    return out


def compute_alignment_metrics(
    summaries: torch.Tensor,
    answers: Sequence[str],
    same_anchor: Dict[str, torch.Tensor],
    wrong_anchor: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], torch.Tensor]:
    same_keys = sorted(same_anchor.keys())
    same_matrix = torch.stack([same_anchor[k] for k in same_keys], dim=0).float()
    same_norm = F.normalize(same_matrix, dim=-1)

    wrong_keys = sorted(wrong_anchor.keys())
    wrong_matrix = torch.stack([wrong_anchor[k] for k in wrong_keys], dim=0).float()
    wrong_norm = F.normalize(wrong_matrix, dim=-1)

    x = F.normalize(summaries.float(), dim=-1)
    sims_same = x.matmul(same_norm.t())
    sims_wrong = x.matmul(wrong_norm.t())

    per_sample: List[Dict[str, Any]] = []
    matched_same_rows: List[torch.Tensor] = []
    same_correct = 0
    combined_correct = 0
    same_cos_vals: List[float] = []
    wrong_cos_vals: List[float] = []
    margin_vals: List[float] = []

    combined_keys = [(k, "same") for k in same_keys] + [(k, "wrong") for k in wrong_keys]
    combined_matrix = torch.cat([same_norm, wrong_norm], dim=0)

    for idx, ans in enumerate(answers):
        ans = extract_answer(ans)
        same_idx = same_keys.index(ans)
        wrong_idx = wrong_keys.index(ans) if ans in wrong_anchor else None
        correct_same = float(sims_same[idx, same_idx].item())
        correct_wrong = float(sims_wrong[idx, wrong_idx].item()) if wrong_idx is not None else float("nan")
        top_same_idx = int(torch.argmax(sims_same[idx]).item())
        top_combined_idx = int(torch.argmax(x[idx].unsqueeze(0).matmul(combined_matrix.t())).item())
        top_same_answer = same_keys[top_same_idx]
        top_combined_answer, top_combined_domain = combined_keys[top_combined_idx]
        if top_same_answer == ans:
            same_correct += 1
        if top_combined_answer == ans and top_combined_domain == "same":
            combined_correct += 1
        if ans in same_anchor:
            matched_same_rows.append(same_anchor[ans].float())
        same_cos_vals.append(correct_same)
        if not math.isnan(correct_wrong):
            wrong_cos_vals.append(correct_wrong)
            margin_vals.append(correct_same - correct_wrong)
        per_sample.append({
            "answer": ans,
            "cosine_same_anchor": correct_same,
            "cosine_wrong_modality_anchor": correct_wrong,
            "margin_same_minus_wrong": (correct_same - correct_wrong) if not math.isnan(correct_wrong) else None,
            "top1_same_answer": top_same_answer,
            "top1_combined_answer": top_combined_answer,
            "top1_combined_domain": top_combined_domain,
        })

    matched_anchor_matrix = torch.stack(matched_same_rows, dim=0) if matched_same_rows else torch.zeros_like(summaries)
    metrics = {
        "num_samples": int(len(answers)),
        "mean_cosine_same_anchor": sum(same_cos_vals) / float(max(1, len(same_cos_vals))),
        "mean_cosine_wrong_modality_anchor": (
            sum(wrong_cos_vals) / float(max(1, len(wrong_cos_vals))) if wrong_cos_vals else None
        ),
        "mean_role_margin": sum(margin_vals) / float(max(1, len(margin_vals))) if margin_vals else None,
        "top1_same_retrieval": 100.0 * same_correct / float(max(1, len(answers))),
        "top1_combined_role_correct": 100.0 * combined_correct / float(max(1, len(answers))),
        "linear_cka_to_same_anchor": linear_cka(summaries, matched_anchor_matrix),
        "subspace_overlap_to_same_anchor": subspace_overlap(summaries, matched_anchor_matrix, rank=min(8, summaries.size(0) - 1)),
    }
    return metrics, per_sample, matched_anchor_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze projector manifold alignment against text anchors")
    p.add_argument("--model-config", type=str, default="rkca")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=512)
    p.add_argument("--num-audio-tokens", type=int, default=8)
    p.add_argument("--fusion-gate", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--llm-model", type=str, default=None)
    p.add_argument("--slim-projector", action="store_true",
                   help="Use config defaults as-is. By default this script assumes concat checkpoints use full-output projectors.")
    p.add_argument("--modality-aware-prompts", action="store_true")
    p.add_argument("--text-prompt-prefix", type=str, default="")
    p.add_argument("--audio-prompt-prefix", type=str, default="")
    p.add_argument("--image-prompt-prefix", type=str, default="")
    p.add_argument("--both-prompt-prefix", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analysis_ns = build_analysis_namespace(args)

    model_cfg = build_model_config(analysis_ns)
    model = SAFEModel(**model_cfg)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"[load] checkpoint={args.checkpoint}", flush=True)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to_device(device)
    if hasattr(model, "get_runtime_device"):
        device = model.get_runtime_device()
    model.eval()

    dataset = ManifestAVQADataset(args.val_manifest, args.media_root)
    if args.max_samples > 0 and len(dataset) > args.max_samples:
        dataset = Subset(dataset, list(range(args.max_samples)))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_avqa,
        pin_memory=torch.cuda.is_available(),
    )

    audio_summaries, audio_answers = collect_projector_summaries(
        model=model,
        dataloader=dataloader,
        device=device,
        prompt_args=analysis_ns,
        modality="audio",
        max_samples=args.max_samples,
    )
    vision_summaries, vision_answers = collect_projector_summaries(
        model=model,
        dataloader=dataloader,
        device=device,
        prompt_args=analysis_ns,
        modality="image",
        max_samples=args.max_samples,
    )

    union_answers = sorted(set(audio_answers) | set(vision_answers) | set(AVQA_ANSWER_VOCAB))
    audio_anchors = collect_anchor_summaries(
        model=model,
        answers=union_answers,
        device=device,
        prompt_args=analysis_ns,
        modality="audio",
        batch_size=args.batch_size,
    )
    vision_anchors = collect_anchor_summaries(
        model=model,
        answers=union_answers,
        device=device,
        prompt_args=analysis_ns,
        modality="image",
        batch_size=args.batch_size,
    )

    audio_metrics, audio_rows, audio_anchor_matrix = compute_alignment_metrics(
        summaries=audio_summaries,
        answers=audio_answers,
        same_anchor=audio_anchors,
        wrong_anchor=vision_anchors,
    )
    vision_metrics, vision_rows, vision_anchor_matrix = compute_alignment_metrics(
        summaries=vision_summaries,
        answers=vision_answers,
        same_anchor=vision_anchors,
        wrong_anchor=audio_anchors,
    )

    cross_metrics = {
        "audio_vs_vision_projector_cka": linear_cka(audio_summaries, vision_summaries),
        "audio_vs_vision_projector_subspace_overlap": subspace_overlap(
            audio_summaries, vision_summaries, rank=min(8, min(audio_summaries.size(0), vision_summaries.size(0)) - 1)
        ),
        "audio_anchor_vs_vision_anchor_cka": linear_cka(audio_anchor_matrix, vision_anchor_matrix),
        "audio_anchor_vs_vision_anchor_subspace_overlap": subspace_overlap(
            audio_anchor_matrix, vision_anchor_matrix, rank=min(8, min(audio_anchor_matrix.size(0), vision_anchor_matrix.size(0)) - 1)
        ),
    }

    summary = {
        "checkpoint": str(args.checkpoint),
        "model_config": args.model_config,
        "modality_aware_prompts": bool(args.modality_aware_prompts),
        "num_samples_audio": int(audio_summaries.size(0)),
        "num_samples_vision": int(vision_summaries.size(0)),
        "audio": audio_metrics,
        "vision": vision_metrics,
        "cross": cross_metrics,
    }

    summary_path = args.output_dir / "projector_manifold_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for name, rows in (("audio", audio_rows), ("vision", vision_rows)):
        csv_path = args.output_dir / f"projector_manifold_{name}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("[summary] " + json.dumps(summary, indent=2), flush=True)
    print(f"[save] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
