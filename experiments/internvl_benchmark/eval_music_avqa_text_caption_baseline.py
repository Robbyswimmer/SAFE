#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from configs.model_configs import get_config
from experiments.avqa_composition.train_avqa_composition import (
    categorical_f1,
    extract_answer,
    normalize_answer,
    token_f1,
)
from train_safe import create_model, load_checkpoint


class CachedCaptionAVQADataset(Dataset):
    def __init__(self, manifest_path: Path, media_root: Path, caption_field: str):
        self.media_root = media_root
        self.caption_field = caption_field
        self.rows: List[Dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_media_path(self, path_value: str) -> Optional[Path]:
        if not path_value:
            return None
        p = Path(path_value).expanduser()
        if p.is_absolute() and p.exists():
            return p
        candidate = self.media_root / p
        if candidate.exists():
            return candidate
        repo_candidate = Path.cwd() / p
        if repo_candidate.exists():
            return repo_candidate
        stem = p.stem
        for subdir in ("frames", "image", "image_31", "images"):
            for ext in (".jpg", ".jpeg", ".png"):
                c = self.media_root / subdir / f"{stem}{ext}"
                if c.exists():
                    return c
        for subdir in ("frames", "image", "image_31", "images"):
            base = self.media_root / subdir
            if not base.exists():
                continue
            matches = list(base.rglob(f"{stem}.*"))
            if matches:
                return matches[0]
        return None

    def _load_image(self, path_value: str) -> Optional[Image.Image]:
        path = self._resolve_media_path(path_value)
        if path is None:
            return None
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        return {
            "sample_id": row.get("sample_id", f"sample_{idx}"),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "question_type": row.get("question_type", "unknown"),
            "caption": row.get(self.caption_field, ""),
            "image": self._load_image(row.get("image_path", "")),
        }


def collate_cached_caption_avqa(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sample_ids": [x["sample_id"] for x in batch],
        "questions": [x["question"] for x in batch],
        "answers": [x["answer"] for x in batch],
        "question_types": [x["question_type"] for x in batch],
        "captions": [x["caption"] for x in batch],
        "images": [x["image"] for x in batch],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MUSIC-AVQA using cached audio captions as text.")
    parser.add_argument("--model-config", type=str, default="rkca_joint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--media-root", type=str, required=True)
    parser.add_argument("--caption-field", type=str, required=True)
    parser.add_argument("--input-mode", type=str, choices=["both", "caption", "image"], default="both")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_prompt(caption: str, question: str, input_mode: str) -> str:
    if input_mode == "caption":
        return (
            f"From audio: {caption}\n"
            f"Question: {question}\n"
            "Answer with exactly one short answer token (single word or number)."
        )
    if input_mode == "image":
        return (
            f"Question: {question}\n"
            "Answer with exactly one short answer token (single word or number)."
        )
    return (
        f"From audio: {caption}\n"
        f"Question: {question}\n"
        "Answer with exactly one short answer token (single word or number)."
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    base_model = model.module if hasattr(model, "module") else model
    tokenizer = base_model.base_vl.tokenizer

    dataset: Dataset = CachedCaptionAVQADataset(
        manifest_path=Path(args.manifest),
        media_root=Path(args.media_root),
        caption_field=args.caption_field,
    )
    if args.max_samples and args.max_samples > 0:
        dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))  # type: ignore[assignment]
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_cached_caption_avqa,
    )

    predictions_out: List[Dict[str, Any]] = []
    raw_correct = 0
    extracted_correct = 0
    total = 0
    total_f1 = 0.0
    total_cat_f1 = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            prompts = [
                build_prompt(caption=cap, question=q, input_mode=args.input_mode)
                for cap, q in zip(batch["captions"], batch["questions"])
            ]
            images = batch["images"] if args.input_mode in {"both", "image"} else None

            generation_inputs = base_model.prepare_multimodal_inputs(
                text=prompts,
                images=images,
                audio=None,
                answers=None,
                device=str(device),
                training_mode=False,
            )

            input_ids = generation_inputs["input_ids"].to(device)
            attention_mask = generation_inputs["attention_mask"].to(device)
            pixel_values = generation_inputs.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)

            generated_ids = base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=1,
                num_beams=args.num_beams,
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

            batch_predictions = tokenizer.batch_decode(
                decoded_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            for i, pred in enumerate(batch_predictions):
                pred_norm = normalize_answer(pred)
                ref_norm = normalize_answer(batch["answers"][i])
                pred_extracted = extract_answer(pred)
                ref_extracted = extract_answer(batch["answers"][i])

                raw_correct += int(pred_norm == ref_norm)
                extracted_correct += int(pred_extracted == ref_extracted)
                total_f1 += token_f1(pred, batch["answers"][i])
                total_cat_f1 += categorical_f1(pred, batch["answers"][i])
                total += 1

                predictions_out.append(
                    {
                        "sample_id": batch["sample_ids"][i],
                        "question_type": batch["question_types"][i],
                        "question": batch["questions"][i],
                        "answer": batch["answers"][i],
                        "caption_field": args.caption_field,
                        "caption": batch["captions"][i],
                        "prediction_raw": pred,
                        "prediction_extracted": pred_extracted,
                    }
                )

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"[eval-caption] step={batch_idx + 1} raw_em={100.0 * raw_correct / max(total, 1):.2f}% "
                    f"extracted_em={100.0 * extracted_correct / max(total, 1):.2f}% "
                    f"caption_field={args.caption_field} input_mode={args.input_mode}",
                    flush=True,
                )

    summary = {
        "checkpoint": args.checkpoint,
        "model_config": args.model_config,
        "manifest": args.manifest,
        "caption_field": args.caption_field,
        "input_mode": args.input_mode,
        "n": total,
        "raw_em": 100.0 * raw_correct / max(total, 1),
        "extracted_em": 100.0 * extracted_correct / max(total, 1),
        "f1": 100.0 * total_f1 / max(total, 1),
        "cat_f1": 100.0 * total_cat_f1 / max(total, 1),
    }

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[summary] {json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
