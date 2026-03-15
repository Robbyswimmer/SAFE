#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.avqa_composition.train_avqa_composition import (
    ManifestAVQADataset,
    categorical_f1,
    collate_avqa,
    extract_answer,
    normalize_answer,
    token_f1,
)
from experiments.internvl_benchmark.eval_music_avqa_text_caption_baseline import (
    RawInternVLEvalEngine,
    build_prompt,
)
from experiments.internvl_benchmark.train_clap_qwen_caption_decoder import (
    CLAPQwenCaptionDecoder,
)
from safe.models.audio_encoders import CLAPAudioEncoder
from torch.utils.data import DataLoader, Dataset, Subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine CLAP->text decoder targets with frozen InternVL AVQA utility.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--media-root", type=str, required=True)
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--clip-cache-manifest", type=str, default="")
    parser.add_argument("--caption-field", type=str, default="refined_decoder_caption")
    parser.add_argument("--llm-model", type=str, default="models/OpenGVLab_InternVL3_5-8B")
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--max-rows-per-clip", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _audio_dedup_key(row: Dict[str, Any], fallback_idx: int) -> str:
    audio_path = str(row.get("audio_path", "") or "").strip()
    if audio_path:
        return f"audio_path:{audio_path}"
    sample_id = str(row.get("sample_id", "") or "").strip()
    if sample_id:
        return f"sample_id:{sample_id}"
    return f"row:{fallback_idx}"


def _resolve_image_path(media_root: Path, path_value: str) -> Optional[Path]:
    if not path_value:
        return None
    p = Path(path_value).expanduser()
    if p.is_absolute() and p.exists():
        return p
    candidate = media_root / p
    if candidate.exists():
        return candidate
    repo_candidate = REPO_ROOT / p
    if repo_candidate.exists():
        return repo_candidate
    stem = p.stem
    for subdir in ("frames", "image", "image_31", "images"):
        for ext in (".jpg", ".jpeg", ".png"):
            c = media_root / subdir / f"{stem}{ext}"
            if c.exists():
                return c
    return None


def _load_image(media_root: Path, path_value: str) -> Optional[Image.Image]:
    path = _resolve_image_path(media_root, path_value)
    if path is None:
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _load_allowed_token_ids(checkpoint_path: Path, device: torch.device) -> Optional[torch.Tensor]:
    token_path = checkpoint_path.parent / "allowed_token_ids.json"
    if not token_path.exists():
        return None
    payload = json.loads(token_path.read_text(encoding="utf-8"))
    token_ids = payload.get("token_ids", [])
    if not token_ids:
        return None
    return torch.tensor([int(x) for x in token_ids], dtype=torch.long, device=device)


def _load_decoder(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[CLAPQwenCaptionDecoder, Any, Optional[torch.Tensor]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm_model"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    vocab_size = int(max(tokenizer.get_vocab().values()) + 1)

    model = CLAPQwenCaptionDecoder(
        vocab_size=vocab_size,
        pad_token_id=int(tokenizer.pad_token_id),
        start_token_id=int(start_token_id),
        d_model=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        ff_mult=int(cfg["ff_mult"]),
        dropout=float(cfg["dropout"]),
        max_length=int(cfg["max_length"]),
        num_memory_tokens=int(cfg["num_memory_tokens"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    allowed_token_ids = _load_allowed_token_ids(checkpoint_path, device)
    return model, tokenizer, allowed_token_ids


def _sample_candidate_captions(
    decoder: CLAPQwenCaptionDecoder,
    decoder_tokenizer: Any,
    audio_embedding: torch.Tensor,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    allowed_token_ids: Optional[torch.Tensor],
) -> List[str]:
    expanded = audio_embedding.expand(num_candidates, -1)
    generated_ids = decoder.generate(
        expanded,
        decoder_tokenizer,
        max_new_tokens=max_new_tokens,
        allowed_token_ids=allowed_token_ids,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    captions = decoder_tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    cleaned = [" ".join(x.strip().split()) for x in captions]
    deduped: List[str] = []
    seen = set()
    for cap in cleaned:
        if cap and cap not in seen:
            seen.add(cap)
            deduped.append(cap)
    return deduped or [""]


def _score_caption_for_clip(
    engine: RawInternVLEvalEngine,
    media_root: Path,
    clip_rows: Sequence[Dict[str, Any]],
    caption: str,
    device: torch.device,
    eval_batch_size: int,
) -> Dict[str, float]:
    raw_correct = 0
    extracted_correct = 0
    total = 0
    total_f1 = 0.0
    total_cat_f1 = 0.0

    for start in range(0, len(clip_rows), eval_batch_size):
        chunk = clip_rows[start : start + eval_batch_size]
        prompts = [
            build_prompt(caption=caption, question=str(row.get("question", "")), input_mode="both")
            for row in chunk
        ]
        images = [_load_image(media_root, str(row.get("image_path", ""))) for row in chunk]
        preds = engine.generate_batch(
            prompts=prompts,
            images=images,
            device=device,
            max_new_tokens=8,
            num_beams=1,
        )
        for pred, row in zip(preds, chunk):
            ref = str(row.get("answer", "") or "")
            pred_norm = normalize_answer(pred)
            ref_norm = normalize_answer(ref)
            pred_extracted = extract_answer(pred)
            ref_extracted = extract_answer(ref)
            raw_correct += int(pred_norm == ref_norm)
            extracted_correct += int(pred_extracted == ref_extracted)
            total_f1 += token_f1(pred, ref)
            total_cat_f1 += categorical_f1(pred, ref)
            total += 1

    return {
        "raw_em": 100.0 * raw_correct / max(total, 1),
        "extracted_em": 100.0 * extracted_correct / max(total, 1),
        "f1": 100.0 * total_f1 / max(total, 1),
        "cat_f1": 100.0 * total_cat_f1 / max(total, 1),
        "n": total,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    media_root = Path(args.media_root)
    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.clip_cache_manifest) if args.clip_cache_manifest else None
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, Any]] = []
    with Path(args.manifest).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_rows.append(json.loads(line))

    clip_to_indices: Dict[str, List[int]] = {}
    ordered_keys: List[str] = []
    for idx, row in enumerate(raw_rows):
        key = _audio_dedup_key(row, idx)
        if key not in clip_to_indices:
            clip_to_indices[key] = []
            ordered_keys.append(key)
        clip_to_indices[key].append(idx)
    if args.max_clips and args.max_clips > 0:
        ordered_keys = ordered_keys[: args.max_clips]

    unique_indices = [clip_to_indices[key][0] for key in ordered_keys]
    dataset: Dataset = ManifestAVQADataset(Path(args.manifest), media_root)
    dataset = Subset(dataset, unique_indices)  # type: ignore[assignment]
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_avqa,
    )

    decoder, decoder_tokenizer, allowed_token_ids = _load_decoder(Path(args.checkpoint), device)
    clap = CLAPAudioEncoder(freeze=True).to(device)
    clap.eval()
    engine = RawInternVLEvalEngine(llm_model=args.llm_model, device=device)

    clip_results: Dict[str, Dict[str, Any]] = {}
    with torch.no_grad():
        for clip_idx, batch in enumerate(dataloader):
            key = ordered_keys[clip_idx]
            member_indices = clip_to_indices[key]
            clip_rows = [raw_rows[i] for i in member_indices]
            if args.max_rows_per_clip and args.max_rows_per_clip > 0:
                clip_rows = clip_rows[: args.max_rows_per_clip]
            audio_embeddings = clap(batch["audio"]).to(device)
            candidates = _sample_candidate_captions(
                decoder=decoder,
                decoder_tokenizer=decoder_tokenizer,
                audio_embedding=audio_embeddings,
                num_candidates=args.num_candidates,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                allowed_token_ids=allowed_token_ids,
            )

            best_caption = candidates[0]
            best_score = -math.inf
            candidate_scores: List[Dict[str, Any]] = []
            for caption in candidates:
                metrics = _score_caption_for_clip(
                    engine=engine,
                    media_root=media_root,
                    clip_rows=clip_rows,
                    caption=caption,
                    device=device,
                    eval_batch_size=args.eval_batch_size,
                )
                score = float(metrics["cat_f1"])
                candidate_scores.append({"caption": caption, "metrics": metrics})
                if score > best_score:
                    best_score = score
                    best_caption = caption

            clip_results[key] = {
                "dedup_key": key,
                "audio_path": clip_rows[0].get("audio_path", ""),
                "num_rows": len(member_indices),
                "best_caption": best_caption,
                "best_score": best_score,
                "candidates": candidate_scores,
            }
            if (clip_idx + 1) % 10 == 0 or clip_idx == 0:
                print(
                    f"[refine] clip={clip_idx + 1}/{len(ordered_keys)} "
                    f"rows={len(member_indices)} best_cat_f1={best_score:.2f} "
                    f"caption={best_caption[:200]!r}",
                    flush=True,
                )

    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(raw_rows):
            key = _audio_dedup_key(row, idx)
            result = clip_results.get(key)
            out = dict(row)
            if result is not None:
                out[args.caption_field] = result["best_caption"]
                out[f"{args.caption_field}_score"] = result["best_score"]
                out["dedup_key"] = key
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    if cache_path is not None:
        with cache_path.open("w", encoding="utf-8") as f:
            for key in ordered_keys:
                if key in clip_results:
                    f.write(json.dumps(clip_results[key], ensure_ascii=False) + "\n")

    summary = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "output_manifest": str(output_path),
        "clip_cache_manifest": str(cache_path) if cache_path is not None else "",
        "caption_field": args.caption_field,
        "unique_clips_refined": len(clip_results),
        "rows_written": len(raw_rows),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
