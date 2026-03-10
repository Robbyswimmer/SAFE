#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from configs.model_configs import get_config
from experiments.avqa_composition.train_avqa_composition import ManifestAVQADataset, collate_avqa
from train_safe import _strip_generation_artifacts, create_model, load_checkpoint


SIMPLE_CAPTION_PROMPT = "Name the main sound or event in a very short phrase."
RICH_CAPTION_PROMPT = (
    "Describe what you hear in one detailed sentence, including the main sound source, "
    "actions, count, intensity, tempo or rhythm, and acoustic context when relevant."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached simple/rich audio captions for MUSIC-AVQA.")
    parser.add_argument("--model-config", type=str, default="rkca_joint_caption16")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--media-root", type=str, required=True)
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def clean_caption(text: str) -> str:
    text = _strip_generation_artifacts(text or "")
    text = " ".join(text.strip().split())
    return text


def _generate_batch(
    model: Any,
    prompts: Sequence[str],
    audio_batch: Sequence[Any],
    device: torch.device,
    max_new_tokens: int,
    num_beams: int,
) -> List[str]:
    base_model = model.module if hasattr(model, "module") else model
    tokenizer = base_model.base_vl.tokenizer

    generation_inputs = base_model.prepare_multimodal_inputs(
        text=list(prompts),
        audio=list(audio_batch),
        answers=None,
        device=str(device),
        training_mode=False,
        llava_audio_prompt_style="plain",
    )

    input_ids = generation_inputs["input_ids"].to(device)
    attention_mask = generation_inputs["attention_mask"].to(device)
    audio_tokens = generation_inputs.get("audio_tokens")
    if audio_tokens is not None:
        audio_tokens = audio_tokens.to(device)
    audio_attention_mask = generation_inputs.get("audio_attention_mask")
    if audio_attention_mask is not None:
        audio_attention_mask = audio_attention_mask.to(device)

    generated_ids = base_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio_tokens=audio_tokens,
        audio_attention_mask=audio_attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,
        num_beams=num_beams,
        repetition_penalty=1.05,
        no_repeat_ngram_size=3,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = int(input_ids.shape[1])
    if generated_ids.dim() == 2 and generated_ids.size(1) > prompt_len:
        decoded_ids = generated_ids[:, prompt_len:]
    else:
        decoded_ids = generated_ids

    decoded = tokenizer.batch_decode(
        decoded_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return [clean_caption(x) for x in decoded]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_config = get_config(args.model_config)
    model = create_model(model_config).to(device)
    load_checkpoint(
        model=model,
        optimizer=None,
        scheduler=None,
        checkpoint_path=Path(args.checkpoint),
        device=device,
    )
    model.eval()

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
            simple = _generate_batch(
                model=model,
                prompts=[SIMPLE_CAPTION_PROMPT] * len(batch["sample_ids"]),
                audio_batch=batch["audio"],
                device=device,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            rich = _generate_batch(
                model=model,
                prompts=[RICH_CAPTION_PROMPT] * len(batch["sample_ids"]),
                audio_batch=batch["audio"],
                device=device,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )

            for i in range(len(batch["sample_ids"])):
                source_row = raw_rows[row_cursor]
                row = {
                    "sample_id": batch["sample_ids"][i],
                    "question": batch["questions"][i],
                    "answer": batch["answers"][i],
                    "question_type": batch["question_types"][i],
                    "audio_path": source_row.get("audio_path", ""),
                    "image_path": source_row.get("image_path", ""),
                    "simple_audio_caption": simple[i],
                    "rich_audio_caption": rich[i],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1
                row_cursor += 1

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"[caption-cache] batches={batch_idx + 1} rows={rows_written} "
                    f"last_simple={simple[0][:80]!r} last_rich={rich[0][:120]!r}",
                    flush=True,
                )

    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "manifest": args.manifest,
                "output_manifest": str(output_path),
                "rows_written": rows_written,
                "caption_prompts": {
                    "simple": SIMPLE_CAPTION_PROMPT,
                    "rich": RICH_CAPTION_PROMPT,
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
