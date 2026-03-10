#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safe.data.datasets import AudioCapsDataset, ClothoDataset, WavCapsDataset
from safe.models.audio_encoders import CLAPAudioEncoder
from train_safe import compute_caption_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_caption(answer: Any, train: bool) -> Optional[str]:
    if answer is None:
        return None
    if isinstance(answer, str):
        text = answer.strip()
        return text if text else None
    if isinstance(answer, (list, tuple)):
        cleaned = [str(x).strip() for x in answer if str(x).strip()]
        if not cleaned:
            return None
        return random.choice(cleaned) if train else cleaned[0]
    return None


class CaptionBatchCollator:
    def __init__(self, tokenizer: Any, max_length: int, train: bool) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train = train
        self.start_token_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else tokenizer.pad_token_id
        )
        if self.start_token_id is None:
            raise ValueError("Tokenizer must have bos/eos/pad token id")

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        audio: List[Any] = []
        captions: List[str] = []
        references: List[List[str]] = []
        sample_ids: List[str] = []

        for sample in batch:
            cap = _pick_caption(sample.get("answers"), train=self.train)
            if sample.get("audio") is None or not cap:
                continue
            refs_raw = sample.get("answers")
            refs: List[str] = []
            if isinstance(refs_raw, str):
                refs = [refs_raw.strip()] if refs_raw.strip() else []
            elif isinstance(refs_raw, (list, tuple)):
                refs = [str(x).strip() for x in refs_raw if str(x).strip()]
            if not refs:
                refs = [cap]
            audio.append(sample["audio"])
            captions.append(cap)
            references.append(refs)
            sample_ids.append(str(sample.get("sample_id", "")))

        if not audio:
            return {
                "audio": [],
                "captions": [],
                "references": [],
                "sample_ids": [],
                "decoder_input_ids": torch.empty(0, 0, dtype=torch.long),
                "labels": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
            }

        encoded = self.tokenizer(
            captions,
            padding=False,
            truncation=True,
            max_length=self.max_length - 1,
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"]
        decoder_sequences: List[torch.Tensor] = []
        label_sequences: List[torch.Tensor] = []
        for ids in encoded:
            seq = list(ids)
            decoder_seq = torch.tensor([int(self.start_token_id)] + seq, dtype=torch.long)
            label_seq = torch.tensor(seq + [int(self.tokenizer.eos_token_id)], dtype=torch.long)
            decoder_sequences.append(decoder_seq)
            label_sequences.append(label_seq)

        decoder_input_ids = nn.utils.rnn.pad_sequence(
            decoder_sequences,
            batch_first=True,
            padding_value=int(self.tokenizer.pad_token_id),
        )
        labels = nn.utils.rnn.pad_sequence(
            label_sequences,
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = decoder_input_ids.ne(int(self.tokenizer.pad_token_id)).long()

        return {
            "audio": audio,
            "captions": captions,
            "references": references,
            "sample_ids": sample_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class CLAPQwenCaptionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        start_token_id: int,
        d_model: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_length: int = 48,
        num_memory_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.max_length = max_length
        self.num_memory_tokens = num_memory_tokens

        self.audio_to_memory = nn.Sequential(
            nn.Linear(512, d_model * num_memory_tokens),
            nn.GELU(),
            nn.Linear(d_model * num_memory_tokens, d_model * num_memory_tokens),
        )
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_length, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def forward(self, audio_embeddings: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        bsz, tgt_len = decoder_input_ids.shape
        memory = self.audio_to_memory(audio_embeddings).view(bsz, self.num_memory_tokens, -1)
        pos = torch.arange(tgt_len, device=decoder_input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_embed(decoder_input_ids) + self.pos_embed(pos)
        causal_mask = torch.triu(
            torch.full((tgt_len, tgt_len), float("-inf"), device=decoder_input_ids.device),
            diagonal=1,
        )
        x = self.decoder(tgt=x, memory=memory, tgt_mask=causal_mask)
        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        audio_embeddings: torch.Tensor,
        tokenizer: Any,
        max_new_tokens: int = 32,
        num_beams: int = 1,
    ) -> torch.Tensor:
        if num_beams != 1:
            raise NotImplementedError("Beam search not implemented for lightweight decoder yet")
        bsz = audio_embeddings.size(0)
        generated = torch.full(
            (bsz, 1),
            int(self.start_token_id),
            dtype=torch.long,
            device=audio_embeddings.device,
        )
        eos_id = int(tokenizer.eos_token_id)
        finished = torch.zeros(bsz, dtype=torch.bool, device=audio_embeddings.device)
        for _ in range(max_new_tokens):
            logits = self.forward(audio_embeddings, generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            finished |= next_token.squeeze(1).eq(eos_id)
            if finished.all():
                break
        return generated[:, 1:]


def build_datasets(data_path: str, use_wavcaps: bool) -> Tuple[Dataset, Dataset]:
    train_parts: List[Dataset] = [AudioCapsDataset(data_path, split="train"), ClothoDataset(data_path, split="train")]
    if use_wavcaps:
        try:
            train_parts.append(WavCapsDataset(data_path, split="train"))
        except Exception as exc:
            print(f"[warn] failed to load WavCaps train split: {exc}", flush=True)
    train_dataset: Dataset = ConcatDataset(train_parts)

    val_parts: List[Dataset] = [AudioCapsDataset(data_path, split="val"), ClothoDataset(data_path, split="val")]
    if use_wavcaps:
        try:
            val_parts.append(WavCapsDataset(data_path, split="val"))
        except Exception as exc:
            print(f"[warn] failed to load WavCaps val split: {exc}", flush=True)
    val_dataset: Dataset = ConcatDataset(val_parts)
    return train_dataset, val_dataset


def evaluate(
    model: CLAPQwenCaptionDecoder,
    clap: CLAPAudioEncoder,
    dataloader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int,
) -> Dict[str, float]:
    model.eval()
    predictions: List[str] = []
    references: List[List[str]] = []
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            if len(batch["audio"]) == 0:
                continue
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)
            audio_embeddings = clap(batch["audio"]).to(device)
            logits = model(audio_embeddings, decoder_input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            total_loss += float(loss.item())
            total_batches += 1

            generated_ids = model.generate(audio_embeddings, tokenizer, max_new_tokens=max_new_tokens)
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions.extend([" ".join(p.strip().split()) for p in preds])
            references.extend(batch["references"])

    metrics = compute_caption_metrics(predictions, references, compute_bertscore=False)
    metrics["loss"] = total_loss / max(total_batches, 1)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight CLAP->Qwen caption decoder.")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--llm-model", type=str, default="models/OpenGVLab_InternVL3_5-8B")
    parser.add_argument("--use-wavcaps", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-memory-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    return parser.parse_args()


def maybe_subset(dataset: Dataset, max_samples: int) -> Dataset:
    if max_samples and max_samples > 0:
        return torch.utils.data.Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must provide eos_token_id")
    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    vocab = tokenizer.get_vocab()
    vocab_size = int(max(vocab.values()) + 1)

    train_dataset, val_dataset = build_datasets(args.data_path, use_wavcaps=args.use_wavcaps)
    train_dataset = maybe_subset(train_dataset, args.max_train_samples)
    val_dataset = maybe_subset(val_dataset, args.max_val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=CaptionBatchCollator(tokenizer, args.max_length, train=True),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=CaptionBatchCollator(tokenizer, args.max_length, train=False),
    )

    clap = CLAPAudioEncoder(freeze=True).to(device)
    clap.eval()

    model = CLAPQwenCaptionDecoder(
        vocab_size=vocab_size,
        pad_token_id=int(tokenizer.pad_token_id),
        start_token_id=int(start_token_id),
        d_model=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        max_length=args.max_length,
        num_memory_tokens=args.num_memory_tokens,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.num_epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_cider = -1.0
    global_step = 0
    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        seen_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            if len(batch["audio"]) == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)
            audio_embeddings = clap(batch["audio"]).to(device)
            logits = model(audio_embeddings, decoder_input_ids)
            valid_labels = labels[labels >= 0]
            if valid_labels.numel() > 0:
                max_label = int(valid_labels.max().item())
                if max_label >= logits.size(-1):
                    raise RuntimeError(
                        f"Label id out of range for decoder vocab: max_label={max_label} logits_vocab={logits.size(-1)}"
                    )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            seen_batches += 1
            global_step += 1

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"[train] epoch={epoch + 1} step={batch_idx + 1}/{len(train_loader)} "
                    f"loss={running_loss / max(seen_batches, 1):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        metrics = evaluate(
            model=model,
            clap=clap,
            dataloader=val_loader,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"[eval] epoch={epoch + 1} {json.dumps(metrics, indent=2)}", flush=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "config": {
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "ff_mult": args.ff_mult,
                "dropout": args.dropout,
                "max_length": args.max_length,
                "num_memory_tokens": args.num_memory_tokens,
                "llm_model": args.llm_model,
            },
        }
        torch.save(checkpoint, output_dir / "checkpoint_last.pt")
        if metrics.get("cider", 0.0) > best_cider:
            best_cider = float(metrics["cider"])
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")
            print(f"[save] best checkpoint -> {output_dir / 'checkpoint_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
