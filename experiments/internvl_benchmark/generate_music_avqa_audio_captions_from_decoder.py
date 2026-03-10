#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.avqa_composition.train_avqa_composition import ManifestAVQADataset, collate_avqa
from safe.models.audio_encoders import CLAPAudioEncoder
from experiments.internvl_benchmark.train_clap_qwen_caption_decoder import CLAPQwenCaptionDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MUSIC-AVQA audio captions from lightweight CLAP->Qwen decoder.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--media-root", type=str, required=True)
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["llm_model"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    model = CLAPQwenCaptionDecoder(
        vocab_size=len(tokenizer),
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

    clap = CLAPAudioEncoder(freeze=True).to(device)
    clap.eval()

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

    out_path = Path(args.output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_cursor = 0
    with out_path.open("w", encoding="utf-8") as f, torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            audio_embeddings = clap(batch["audio"]).to(device)
            generated_ids = model.generate(audio_embeddings, tokenizer, max_new_tokens=args.max_new_tokens)
            captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            captions = [" ".join(x.strip().split()) for x in captions]

            for i in range(len(captions)):
                source_row = raw_rows[row_cursor]
                row = {
                    "sample_id": batch["sample_ids"][i],
                    "question": batch["questions"][i],
                    "answer": batch["answers"][i],
                    "question_type": batch["question_types"][i],
                    "audio_path": source_row.get("audio_path", ""),
                    "image_path": source_row.get("image_path", ""),
                    "decoder_audio_caption": captions[i],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_cursor += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"[decoder-captions] batches={batch_idx + 1} rows={row_cursor} last={captions[0][:120]!r}", flush=True)

    print(f"[save] caption manifest -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
