#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.avqa_composition.train_avqa_composition import ManifestAVQADataset, collate_avqa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MUSIC-AVQA teacher captions with an external audio captioning model.")
    parser.add_argument("--teacher-model", type=str, default="models/audio_caption_teacher")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--media-root", type=str, required=True)
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--caption-field", type=str, default="teacher_caption_raw")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _clean_caption(text: str) -> str:
    text = " ".join((text or "").strip().split())
    return text


def _load_teacher(model_path: Path, device: torch.device) -> tuple[Any, Any]:
    from transformers import AutoModelForSpeechSeq2Seq, AutoModelForSeq2SeqLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model: Optional[Any] = None
    load_errors: List[str] = []

    for cls in (AutoModelForSpeechSeq2Seq, AutoModelForSeq2SeqLM):
        try:
            model = cls.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            )
            break
        except Exception as exc:  # pragma: no cover - fallback path
            load_errors.append(f"{cls.__name__}: {exc}")

    if model is None:
        raise RuntimeError(
            "Failed to load teacher model from "
            f"{model_path}. Errors: {' | '.join(load_errors)}"
        )

    model.to(device)
    model.eval()
    return processor, model


def _prepare_audio_batch(
    audio_batch: Sequence[Any],
    target_sr: int,
) -> List[List[float]]:
    import torchaudio

    features: List[List[float]] = []
    for audio in audio_batch:
        if audio is None:
            waveform = torch.zeros(target_sr, dtype=torch.float32)
            sr = target_sr
        else:
            waveform, sr = audio
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform, dtype=torch.float32)
            waveform = waveform.float().cpu()
            if waveform.dim() == 2 and waveform.size(0) > 1:
                waveform = waveform.mean(dim=0)
            elif waveform.dim() == 2:
                waveform = waveform.squeeze(0)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0), sr, target_sr
            ).squeeze(0)
        features.append(waveform.tolist())
    return features


@torch.no_grad()
def _generate_batch(
    processor: Any,
    model: Any,
    audio_batch: Sequence[Any],
    device: torch.device,
    max_new_tokens: int,
    num_beams: int,
) -> List[str]:
    target_sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
    inputs_audio = _prepare_audio_batch(audio_batch, target_sr)
    # Whisper-style processors expect `audio=...`; some other speech processors also
    # accept the singular form. Try that first, then fall back to `audios=...`.
    try:
        proc = processor(
            audio=inputs_audio,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
    except TypeError:
        proc = processor(
            audios=inputs_audio,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
    model_dtype = getattr(model, "dtype", None)
    cast_proc: Dict[str, Any] = {}
    for k, v in proc.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if model_dtype is not None and torch.is_floating_point(v):
                v = v.to(dtype=model_dtype)
        cast_proc[k] = v
    proc = cast_proc
    generated = model.generate(
        **proc,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
    )
    decoded = processor.batch_decode(generated, skip_special_tokens=True)
    return [_clean_caption(x) for x in decoded]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*load_with_torchcodec.*")
    warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*deprecated.*")

    model_path = Path(args.teacher_model)
    if not model_path.exists():
        raise FileNotFoundError(f"Teacher model path not found: {model_path}")

    processor, model = _load_teacher(model_path, device)

    raw_rows: List[Dict[str, Any]] = []
    with Path(args.manifest).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_rows.append(json.loads(line))

    dataset: Dataset = ManifestAVQADataset(Path(args.manifest), Path(args.media_root))
    if args.max_samples and args.max_samples > 0:
        limit = min(args.max_samples, len(raw_rows))
        raw_rows = raw_rows[:limit]
        dataset = Subset(dataset, list(range(limit)))  # type: ignore[assignment]

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_avqa,
    )

    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    row_cursor = 0
    with output_path.open("w", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(dataloader):
            captions = _generate_batch(
                processor=processor,
                model=model,
                audio_batch=batch["audio"],
                device=device,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )

            for i, caption in enumerate(captions):
                source_row = dict(raw_rows[row_cursor])
                source_row["sample_id"] = batch["sample_ids"][i]
                source_row["question"] = batch["questions"][i]
                source_row["answer"] = batch["answers"][i]
                source_row["question_type"] = batch["question_types"][i]
                source_row[args.caption_field] = caption
                f.write(json.dumps(source_row, ensure_ascii=False) + "\n")
                rows_written += 1
                row_cursor += 1

            if (batch_idx + 1) % 25 == 0:
                print(
                    f"[external-teacher] batches={batch_idx + 1} rows={rows_written} "
                    f"last_caption={captions[0][:160]!r}",
                    flush=True,
                )

    print(
        json.dumps(
            {
                "teacher_model": str(model_path),
                "manifest": args.manifest,
                "output_manifest": str(output_path),
                "caption_field": args.caption_field,
                "rows_written": rows_written,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
