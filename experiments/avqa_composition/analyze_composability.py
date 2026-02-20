#!/usr/bin/env python3
"""
Single-run composability diagnostics for AVQA composition checkpoints.

This script is designed to answer a concrete question quickly:
  "Do we need more overlap or more orthogonality between modality adapters?"

It reports:
- Output-level composition metrics (text/audio/image/both, synergy, gain vs best single)
- Per-layer output additivity probe (existing extracted-match probe)
- Per-layer representation diagnostics:
  - shift norms for audio / vision / both
  - hidden-space additivity residual ||d_av - d_a - d_v||
  - audio/vision subspace overlap (principal-angle proxies)
  - projected overlap ratios
  - fitted overlap scale m* that best explains observed composed shift

All results are saved to JSON for paper tables/plots.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.avqa_composition.train_avqa_composition import (
    ManifestAVQADataset,
    _extract_pooled_layer_states,
    _fit_shift_basis,
    build_model_config,
    collate_avqa,
    evaluate,
    resolve_modality_batch,
    run_layer_additivity_probe,
    set_seed,
)
from safe.models.safe_model import SAFEModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Composability diagnostics for AVQA checkpoints")
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--model-config", type=str, default="composition_independent")
    p.add_argument("--llm-model", type=str, default=None)
    p.add_argument("--fusion-layers", type=str, default=None)
    p.add_argument("--num-audio-tokens", type=int, default=8)
    p.add_argument("--fusion-gate", type=float, default=0.2)
    p.add_argument("--max-answer-tokens", type=int, default=16)
    p.add_argument("--bottleneck-dim", type=int, default=None)
    p.add_argument("--no-slim-projector", dest="slim_projector", action="store_false")
    p.set_defaults(slim_projector=True)

    p.add_argument("--compose-audio-ckpt", type=Path, required=True)
    p.add_argument("--compose-vision-ckpt", type=Path, required=True)

    p.add_argument("--max-samples", type=int, default=1000,
                   help="Truncate val set before all diagnostics (0 = full val set)")
    p.add_argument("--probe-samples", type=int, default=512,
                   help="Subset size for representation diagnostics and layer probe")
    p.add_argument("--subspace-rank", type=int, default=8)

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--eval-debug-samples", type=int, default=0)
    p.add_argument("--skip-layer-probe", action="store_true",
                   help="Skip generation-based per-layer additivity probe")

    p.add_argument("--output-json", type=Path, default=None,
                   help="Defaults to <output-dir>/composability_diagnostics.json")
    return p.parse_args()


@torch.no_grad()
def _run_forward(
    model: SAFEModel,
    prepared_inputs: Dict[str, Any],
    gate: float,
) -> Optional[Dict[str, Any]]:
    local_inputs = dict(prepared_inputs)
    audio_tokens = local_inputs.pop("audio_tokens", None)
    audio_mask = local_inputs.pop("audio_attention_mask", None)

    out = model(
        input_ids=local_inputs["input_ids"],
        attention_mask=local_inputs.get("attention_mask"),
        labels=local_inputs.get("labels"),
        pixel_values=local_inputs.get("pixel_values"),
        audio_tokens=audio_tokens,
        audio_attention_mask=audio_mask,
        gate=gate,
        output_hidden_states=True,
    )
    if not isinstance(out, dict):
        return None
    return out


@torch.no_grad()
def collect_representation_diagnostics(
    model: SAFEModel,
    dataset: ManifestAVQADataset,
    device: torch.device,
    args: argparse.Namespace,
    layers: Sequence[int],
) -> Dict[str, Any]:
    start = time.time()

    if args.probe_samples > 0 and args.probe_samples < len(dataset):
        probe_ds = Subset(dataset, list(range(args.probe_samples)))
    else:
        probe_ds = dataset

    loader = DataLoader(
        probe_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_avqa,
        pin_memory=torch.cuda.is_available(),
    )

    da_rows: Dict[int, List[torch.Tensor]] = {int(l): [] for l in layers}
    dv_rows: Dict[int, List[torch.Tensor]] = {int(l): [] for l in layers}
    dav_rows: Dict[int, List[torch.Tensor]] = {int(l): [] for l in layers}
    residual_rows: Dict[int, List[torch.Tensor]] = {int(l): [] for l in layers}

    processed = 0
    for batch in loader:
        prepared: Dict[str, Dict[str, Any]] = {}
        for modality in ("text", "audio", "image", "both"):
            mm = resolve_modality_batch(batch, modality)
            prepared[modality] = model.prepare_multimodal_inputs(
                text=batch["questions"],
                images=mm["images"],
                audio=mm["audio"],
                answers=batch["answers"],
                device=str(device),
                training_mode=True,
            )

        out_text = _run_forward(model, prepared["text"], gate=0.0)
        out_audio = _run_forward(model, prepared["audio"], gate=args.fusion_gate)
        out_image = _run_forward(model, prepared["image"], gate=args.fusion_gate)
        out_both = _run_forward(model, prepared["both"], gate=args.fusion_gate)
        if out_text is None or out_audio is None or out_image is None or out_both is None:
            continue

        hs_text = out_text.get("all_hidden_states")
        hs_audio = out_audio.get("all_hidden_states")
        hs_image = out_image.get("all_hidden_states")
        hs_both = out_both.get("all_hidden_states")
        if hs_text is None or hs_audio is None or hs_image is None or hs_both is None:
            continue

        pooled_text = _extract_pooled_layer_states(
            hs_text,
            layer_indices=layers,
            labels=prepared["text"].get("labels"),
            attention_mask=prepared["text"].get("attention_mask"),
        )
        pooled_audio = _extract_pooled_layer_states(
            hs_audio,
            layer_indices=layers,
            labels=prepared["audio"].get("labels"),
            attention_mask=prepared["audio"].get("attention_mask"),
        )
        pooled_image = _extract_pooled_layer_states(
            hs_image,
            layer_indices=layers,
            labels=prepared["image"].get("labels"),
            attention_mask=prepared["image"].get("attention_mask"),
        )
        pooled_both = _extract_pooled_layer_states(
            hs_both,
            layer_indices=layers,
            labels=prepared["both"].get("labels"),
            attention_mask=prepared["both"].get("attention_mask"),
        )

        for layer in layers:
            if layer not in pooled_text or layer not in pooled_audio or layer not in pooled_image or layer not in pooled_both:
                continue

            p0 = pooled_text[layer].detach().float().cpu()
            pa = pooled_audio[layer].detach().float().cpu()
            pv = pooled_image[layer].detach().float().cpu()
            pav = pooled_both[layer].detach().float().cpu()

            da = pa - p0
            dv = pv - p0
            dav = pav - p0
            residual = dav - da - dv

            da_rows[layer].append(da)
            dv_rows[layer].append(dv)
            dav_rows[layer].append(dav)
            residual_rows[layer].append(residual)

        processed += len(batch["questions"])

    per_layer: Dict[int, Dict[str, float]] = {}
    for layer in layers:
        if not da_rows[layer] or not dv_rows[layer] or not dav_rows[layer] or not residual_rows[layer]:
            continue

        da_mat = torch.cat(da_rows[layer], dim=0)
        dv_mat = torch.cat(dv_rows[layer], dim=0)
        dav_mat = torch.cat(dav_rows[layer], dim=0)
        res_mat = torch.cat(residual_rows[layer], dim=0)

        n = min(da_mat.size(0), dv_mat.size(0), dav_mat.size(0), res_mat.size(0))
        da_mat = da_mat[:n]
        dv_mat = dv_mat[:n]
        dav_mat = dav_mat[:n]
        res_mat = res_mat[:n]

        da_norm = da_mat.norm(dim=1)
        dv_norm = dv_mat.norm(dim=1)
        dav_norm = dav_mat.norm(dim=1)
        res_norm = res_mat.norm(dim=1)

        cos_num = (da_mat * dv_mat).sum(dim=1)
        cos_den = (da_norm * dv_norm).clamp(min=1e-8)
        cos_da_dv = cos_num / cos_den

        normalized_add_err = res_norm / (da_norm + dv_norm).clamp(min=1e-8)

        Ua = _fit_shift_basis(da_mat, rank=args.subspace_rank)
        Uv = _fit_shift_basis(dv_mat, rank=args.subspace_rank)

        subspace_overlap_fro = float("nan")
        principal_cos_max = float("nan")
        principal_cos_mean = float("nan")
        vision_parallel_ratio = float("nan")
        m_opt = float("nan")
        residual_norm_curr = float(res_norm.mean().item())
        residual_norm_opt = float("nan")
        residual_opt_improve_pct = float("nan")

        if Ua is not None and Uv is not None and Ua.numel() > 0 and Uv.numel() > 0:
            cross = Ua.transpose(0, 1).matmul(Uv)
            svals = torch.linalg.svdvals(cross)
            denom = math.sqrt(float(max(1, min(Ua.size(1), Uv.size(1)))))
            subspace_overlap_fro = float(torch.norm(cross, p="fro").item() / denom)
            principal_cos_max = float(svals.max().item()) if svals.numel() else float("nan")
            principal_cos_mean = float(svals.mean().item()) if svals.numel() else float("nan")

            proj = dv_mat.matmul(Ua).matmul(Ua.transpose(0, 1))
            proj_norm = proj.norm(dim=1)
            vision_parallel_ratio = float((proj_norm / dv_norm.clamp(min=1e-8)).mean().item())

            dv_perp = dv_mat - proj
            target = dav_mat - da_mat - dv_perp
            num = float((target * proj).sum().item())
            den = float((proj * proj).sum().item()) + 1e-8
            m_opt = num / den

            recon_curr = da_mat + dv_mat
            recon_opt = da_mat + dv_perp + (m_opt * proj)
            residual_norm_curr = float((dav_mat - recon_curr).norm(dim=1).mean().item())
            residual_norm_opt = float((dav_mat - recon_opt).norm(dim=1).mean().item())
            if residual_norm_curr > 1e-8:
                residual_opt_improve_pct = 100.0 * (residual_norm_curr - residual_norm_opt) / residual_norm_curr

        per_layer[int(layer)] = {
            "num_samples": int(n),
            "audio_shift_norm_mean": float(da_norm.mean().item()),
            "vision_shift_norm_mean": float(dv_norm.mean().item()),
            "both_shift_norm_mean": float(dav_norm.mean().item()),
            "additivity_residual_norm_mean": float(res_norm.mean().item()),
            "normalized_additivity_error_mean": float(normalized_add_err.mean().item()),
            "mean_cosine_audio_vs_vision": float(cos_da_dv.mean().item()),
            "subspace_overlap_fro": subspace_overlap_fro,
            "principal_cosine_max": principal_cos_max,
            "principal_cosine_mean": principal_cos_mean,
            "vision_parallel_to_audio_ratio": vision_parallel_ratio,
            "overlap_scale_m_opt": float(m_opt),
            "residual_norm_current": float(residual_norm_curr),
            "residual_norm_opt": float(residual_norm_opt),
            "residual_opt_improve_pct": float(residual_opt_improve_pct),
        }

    elapsed = time.time() - start
    return {
        "num_samples": int(min(len(probe_ds), processed)),
        "elapsed_sec": float(elapsed),
        "layers": [int(l) for l in layers],
        "per_layer": per_layer,
    }


def _build_eval_namespace(args: argparse.Namespace) -> SimpleNamespace:
    # Minimal namespace expected by evaluate()/probe helpers.
    return SimpleNamespace(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fp16=bool(args.fp16),
        fusion_gate=float(args.fusion_gate),
        max_answer_tokens=int(args.max_answer_tokens),
        eval_debug_samples=int(args.eval_debug_samples),
        layer_additivity_probe=not bool(args.skip_layer_probe),
        layer_probe_samples=int(args.probe_samples),
        layer_probe_every=1,
    )


def _summarize_direction(rep_diag: Dict[str, Any]) -> Dict[str, Any]:
    layer_stats = rep_diag.get("per_layer", {})
    if not layer_stats:
        return {
            "recommendation": "insufficient_data",
            "reason": "No per-layer diagnostics were computed.",
        }

    valid = list(layer_stats.values())
    mean_parallel = sum(float(s.get("vision_parallel_to_audio_ratio", 0.0) or 0.0) for s in valid) / len(valid)
    mean_mopt = sum(float(s.get("overlap_scale_m_opt", 1.0) or 1.0) for s in valid) / len(valid)
    mean_add_err = sum(float(s.get("normalized_additivity_error_mean", 0.0) or 0.0) for s in valid) / len(valid)

    if mean_mopt > 1.15:
        rec = "increase_overlap"
        reason = "Observed composed shifts look under-powered along audio-aligned directions (m*>1)."
    elif mean_mopt < 0.85:
        rec = "decrease_overlap"
        reason = "Observed composed shifts look over-coupled along overlap directions (m*<1)."
    elif mean_parallel < 0.10:
        rec = "add_controlled_overlap"
        reason = "Very low vision projection into audio subspace suggests adapters are too orthogonal."
    else:
        rec = "balanced_overlap"
        reason = "Overlap appears moderate; focus on no-harm/routing/calibration rather than stronger geometry changes."

    return {
        "recommendation": rec,
        "reason": reason,
        "mean_parallel_ratio": float(mean_parallel),
        "mean_m_opt": float(mean_mopt),
        "mean_normalized_additivity_error": float(mean_add_err),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_json = args.output_json or (args.output_dir / "composability_diagnostics.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diag] device={device}", flush=True)

    val_ds = ManifestAVQADataset(args.val_manifest, args.media_root)
    if args.max_samples > 0:
        val_ds.rows = val_ds.rows[: args.max_samples]
    print(f"[diag] val_samples={len(val_ds)}", flush=True)
    print(f"[diag] val_media_stats={val_ds.media_stats}", flush=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_avqa,
        pin_memory=torch.cuda.is_available(),
    )

    cfg_args = SimpleNamespace(
        model_config=args.model_config,
        llm_model=args.llm_model,
        fusion_layers=args.fusion_layers,
        num_audio_tokens=args.num_audio_tokens,
        freeze_audio_encoder=True,
        label_smoothing=None,
        bottleneck_dim=args.bottleneck_dim,
        learned_gate=False,
        learned_gate_init=0.0,
        slim_projector=args.slim_projector,
    )
    model_cfg = build_model_config(cfg_args)

    model = SAFEModel(**model_cfg)
    model.load_modality_adapters(str(args.compose_audio_ckpt), "audio")
    model.load_modality_adapters(str(args.compose_vision_ckpt), "vision")
    model.to_device(device)
    model.eval()
    if hasattr(model, "get_runtime_device"):
        device = model.get_runtime_device()
    print(f"[diag] runtime_device={device}", flush=True)

    tokenizer = model.base_vl.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_args = _build_eval_namespace(args)

    overall_eval: Dict[str, Any] = {}
    for modality in ("text", "audio", "image", "both"):
        metrics = evaluate(model, val_loader, tokenizer, device, modality, eval_args)
        overall_eval[modality] = metrics
        print(
            f"[diag:{modality}] extracted={metrics['extracted_match']:.2f} "
            f"raw={metrics['exact_match']:.2f} n={metrics['num_samples']}",
            flush=True,
        )

    y0 = overall_eval["text"]["extracted_match"]
    ya = overall_eval["audio"]["extracted_match"]
    yv = overall_eval["image"]["extracted_match"]
    yav = overall_eval["both"]["extracted_match"]
    summary = {
        "text_extracted_match": float(y0),
        "audio_extracted_match": float(ya),
        "image_extracted_match": float(yv),
        "both_extracted_match": float(yav),
        "synergy_extracted_pp": float(yav - (ya + yv - y0)),
        "gain_vs_best_single_pp": float(yav - max(ya, yv)),
        "gain_vs_text_pp": float(yav - y0),
    }
    print(
        "[diag:summary] "
        f"text={y0:.2f} audio={ya:.2f} image={yv:.2f} both={yav:.2f} "
        f"synergy={summary['synergy_extracted_pp']:+.2f} "
        f"gain_vs_best_single={summary['gain_vs_best_single_pp']:+.2f}",
        flush=True,
    )

    layer_probe = None
    if not args.skip_layer_probe:
        print("[diag] running layer additivity probe...", flush=True)
        layer_probe = run_layer_additivity_probe(model, val_ds, tokenizer, device, eval_args)

    fusion_layers: List[int] = []
    if hasattr(model, "fusion_adapter") and model.fusion_adapter is not None:
        fusion_layers = list(sorted(getattr(model.fusion_adapter, "fusion_layer_indices", [])))
    print(f"[diag] fusion_layers={fusion_layers}", flush=True)

    rep_diag = collect_representation_diagnostics(
        model=model,
        dataset=val_ds,
        device=device,
        args=args,
        layers=fusion_layers,
    )
    direction = _summarize_direction(rep_diag)
    print(
        f"[diag:direction] recommendation={direction.get('recommendation')} "
        f"mean_m_opt={direction.get('mean_m_opt', float('nan')):.3f} "
        f"mean_parallel={direction.get('mean_parallel_ratio', float('nan')):.3f}",
        flush=True,
    )

    payload = {
        "config": {
            "model_config": args.model_config,
            "llm_model": args.llm_model,
            "fusion_layers": args.fusion_layers,
            "num_audio_tokens": args.num_audio_tokens,
            "fusion_gate": args.fusion_gate,
            "max_samples": args.max_samples,
            "probe_samples": args.probe_samples,
            "subspace_rank": args.subspace_rank,
            "audio_ckpt": str(args.compose_audio_ckpt),
            "vision_ckpt": str(args.compose_vision_ckpt),
        },
        "eval": overall_eval,
        "summary": summary,
        "layer_additivity_probe": layer_probe,
        "representation_diagnostics": rep_diag,
        "direction": direction,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[diag] wrote {output_json}", flush=True)


if __name__ == "__main__":
    main()
