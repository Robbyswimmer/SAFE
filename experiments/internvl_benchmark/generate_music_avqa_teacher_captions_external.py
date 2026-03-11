#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.avqa_composition.train_avqa_composition import ManifestAVQADataset, collate_avqa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MUSIC-AVQA teacher captions with an external audio captioning model.")
    parser.add_argument("--teacher-model", type=str, default="models/audio_caption_teacher")
    parser.add_argument("--teacher-backend", type=str, default="auto", choices=["auto", "transformers", "conette", "qwen_omni"])
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
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit quantization (fits 30B MoE in 48GB)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit quantization")
    return parser.parse_args()


def _clean_caption(text: str) -> str:
    text = " ".join((text or "").strip().split())
    return text


def _looks_like_conette(model_path: Path) -> bool:
    name = str(model_path).lower()
    if "conette" in name:
        return True
    cfg_path = model_path / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            joined = json.dumps(cfg).lower()
            return "conette" in joined
        except Exception:
            return False
    return False


def _looks_like_qwen_omni(model_path: Path) -> bool:
    name = str(model_path).lower()
    if "qwen" in name and "omni" in name:
        return True
    cfg_path = model_path / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            model_type = cfg.get("model_type", "").lower()
            archs = " ".join(cfg.get("architectures", [])).lower()
            combined = model_type + " " + archs
            return "qwen" in combined and "omni" in combined
        except Exception:
            return False
    return False


def _load_teacher_qwen_omni(
    model_path: Path, device: torch.device,
    load_in_8bit: bool = False, load_in_4bit: bool = False,
) -> Tuple[str, Any, Any]:
    try:
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    except ImportError:
        raise RuntimeError(
            "Qwen3-Omni requires transformers >= 4.51. "
            "Install with: pip install --upgrade transformers"
        )

    # Prefer flash_attention_2, fall back to sdpa if flash_attn not installed
    attn_impl = "flash_attention_2"
    if importlib.util.find_spec("flash_attn") is None:
        attn_impl = "sdpa"
        print("[qwen_omni] flash_attn not installed, using SDPA attention", flush=True)

    print(f"[qwen_omni] Loading processor from {model_path} ...", flush=True)
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path))

    load_kwargs: Dict[str, Any] = dict(
        dtype="auto",
        device_map="auto",
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    if load_in_8bit or load_in_4bit:
        from transformers import BitsAndBytesConfig
        if load_in_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            print("[qwen_omni] Loading in 8-bit quantization", flush=True)
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            print("[qwen_omni] Loading in 4-bit (nf4) quantization", flush=True)

    print(f"[qwen_omni] Loading model from {model_path} (attn={attn_impl}) ...", flush=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(model_path), **load_kwargs,
    )
    model.eval()
    print(f"[qwen_omni] Model loaded (device={model.device}, dtype={model.dtype})", flush=True)
    return "qwen_omni", processor, model


def _load_teacher_transformers(model_path: Path, device: torch.device) -> Tuple[str, Any, Any]:
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
    return "transformers", processor, model


def _load_teacher_conette(model_path: Path, device: torch.device) -> Tuple[str, Any, Any]:
    if importlib.util.find_spec("conette") is None:
        raise RuntimeError("Python package 'conette' is not installed in this environment.")

    from conette import CoNeTTEModel  # type: ignore

    model = CoNeTTEModel.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    return "conette", None, model


def _load_teacher(
    model_path: Path, device: torch.device, backend: str,
    load_in_8bit: bool = False, load_in_4bit: bool = False,
) -> Tuple[str, Any, Any]:
    load_errors: List[str] = []

    if backend in {"auto", "qwen_omni"} and _looks_like_qwen_omni(model_path):
        try:
            return _load_teacher_qwen_omni(
                model_path, device, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit,
            )
        except Exception as exc:
            load_errors.append(f"qwen_omni: {exc}")
            if backend == "qwen_omni":
                raise RuntimeError(
                    f"Failed to load Qwen-Omni teacher from {model_path}: {' | '.join(load_errors)}"
                ) from exc

    if backend in {"auto", "conette"} and _looks_like_conette(model_path):
        try:
            return _load_teacher_conette(model_path, device)
        except Exception as exc:
            load_errors.append(f"conette: {exc}")
            if backend == "conette":
                raise RuntimeError(
                    f"Failed to load CoNeTTE teacher from {model_path}: {' | '.join(load_errors)}"
                ) from exc

    if backend in {"auto", "transformers"}:
        try:
            return _load_teacher_transformers(model_path, device)
        except Exception as exc:
            load_errors.append(f"transformers: {exc}")
            raise RuntimeError(
                f"Failed to load external teacher from {model_path}: {' | '.join(load_errors)}"
            ) from exc

    raise RuntimeError(f"Unsupported teacher backend {backend!r} for model path {model_path}")


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
def _generate_batch_transformers(
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


@torch.no_grad()
def _generate_batch_conette(
    model: Any,
    audio_batch: Sequence[Any],
    max_new_tokens: int,
    num_beams: int,
) -> List[str]:
    # Best-effort adapter for CoNeTTE package APIs. We normalize audio to 32kHz,
    # then try common generation call surfaces.
    target_sr = 32000
    inputs_audio = _prepare_audio_batch(audio_batch, target_sr)
    waveforms = [torch.tensor(x, dtype=torch.float32) for x in inputs_audio]

    attempts = []
    attempts.append(lambda: model.generate(waveforms, sample_rate=target_sr, max_new_tokens=max_new_tokens, num_beams=num_beams))
    attempts.append(lambda: model.generate(audio=waveforms, sample_rate=target_sr, max_new_tokens=max_new_tokens, num_beams=num_beams))
    attempts.append(lambda: model.generate_from_audio(waveforms, sample_rate=target_sr))
    attempts.append(lambda: model.generate_from_audio(audio=waveforms, sample_rate=target_sr))
    attempts.append(lambda: model.caption_audio(waveforms, sample_rate=target_sr))
    attempts.append(lambda: model.caption_audio(audio=waveforms, sample_rate=target_sr))

    last_exc: Optional[Exception] = None
    outputs: Any = None
    for fn in attempts:
        try:
            outputs = fn()
            break
        except Exception as exc:  # pragma: no cover - best effort API probing
            last_exc = exc
            continue
    if outputs is None:
        raise RuntimeError(
            "CoNeTTE generation API probing failed. Last error: "
            f"{last_exc}"
        )

    captions: List[str] = []
    if isinstance(outputs, dict):
        for key in ("captions", "predictions", "texts", "sequences"):
            if key in outputs:
                outputs = outputs[key]
                break
    if isinstance(outputs, (list, tuple)):
        for item in outputs:
            if isinstance(item, dict):
                text = item.get("caption") or item.get("text") or item.get("prediction") or ""
            else:
                text = str(item)
            captions.append(_clean_caption(text))
    else:
        raise RuntimeError(f"Unexpected CoNeTTE output type: {type(outputs)}")
    return captions


@torch.no_grad()
def _generate_batch_qwen_omni(
    processor: Any,
    model: Any,
    audio_batch: Sequence[Any],
    max_new_tokens: int,
    num_beams: int,
) -> List[str]:
    """Generate captions with Qwen3-Omni-Captioner (audio-only, single-turn)."""
    import numpy as np
    import torchaudio

    target_sr = getattr(
        getattr(processor, "feature_extractor", None), "sampling_rate", 16000
    )

    # Build chat template text once (audio-only captioner, no text prompt)
    conversation = [
        {"role": "user", "content": [{"type": "audio", "audio": "input.wav"}]},
    ]
    template_text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    captions: List[str] = []
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

        audio_np = waveform.numpy().astype(np.float32)

        inputs = processor(
            text=template_text,
            audio=[audio_np],
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            thinker_return_dict_in_generate=True,
        )
        # Qwen3-Omni generate returns (text_output, audio_output)
        text_output = output[0] if isinstance(output, tuple) else output
        seqs = text_output.sequences if hasattr(text_output, "sequences") else text_output

        decoded = processor.batch_decode(
            seqs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        captions.append(_clean_caption(decoded[0] if decoded else ""))

    return captions


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*load_with_torchcodec.*")
    warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*deprecated.*")

    model_path = Path(args.teacher_model)
    if not model_path.exists():
        raise FileNotFoundError(f"Teacher model path not found: {model_path}")

    backend, processor, model = _load_teacher(
        model_path, device, args.teacher_backend,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit,
    )

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
            if backend == "qwen_omni":
                captions = _generate_batch_qwen_omni(
                    processor=processor,
                    model=model,
                    audio_batch=batch["audio"],
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )
            elif backend == "conette":
                captions = _generate_batch_conette(
                    model=model,
                    audio_batch=batch["audio"],
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )
            else:
                captions = _generate_batch_transformers(
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
                "teacher_backend": backend,
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
