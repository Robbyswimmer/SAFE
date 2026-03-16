#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
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

from configs.model_configs import get_config
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
    SafeEvalEngine,
)
from safe.data.datasets import AudioCapsDataset, ClothoDataset, WavCapsDataset
from safe.models.audio_encoders import CLAPAudioEncoder
from train_safe import compute_caption_metrics, create_model, load_checkpoint


warnings.filterwarnings(
    "ignore",
    message=".*torchaudio.load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*`torch_dtype` is deprecated! Use `dtype` instead!.*",
    category=UserWarning,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_caption(value: Any, train: bool) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, (list, tuple)):
        cleaned = [str(x).strip() for x in value if str(x).strip()]
        if not cleaned:
            return None
        return random.choice(cleaned) if train else cleaned[0]
    return None


class MusicAVQATextTargetDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        media_root: Path,
        target_field: str,
        dedup_by_audio: bool = False,
    ) -> None:
        self.manifest_path = manifest_path
        self.media_root = media_root
        self.target_field = target_field
        self.base = ManifestAVQADataset(manifest_path, media_root)
        self.indices = list(range(len(self.base)))
        if dedup_by_audio:
            unique_indices: List[int] = []
            seen_keys = set()
            for idx, row in enumerate(self.base.rows):
                key = str(row.get("audio_path", "") or "").strip()
                if not key:
                    key = str(row.get("sample_id", f"row_{idx}") or f"row_{idx}")
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                unique_indices.append(idx)
            self.indices = unique_indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx = self.indices[idx]
        item = self.base[base_idx]
        row = self.base.rows[base_idx]
        item["target_text"] = row.get(self.target_field, "")
        item["audio_path"] = row.get("audio_path", "")
        item["image_path"] = row.get("image_path", "")
        return item


class CaptionBatchCollator:
    def __init__(self, tokenizer: Any, max_length: int, train: bool, target_field: str = "answer") -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train = train
        self.target_field = target_field
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
            raw_answers = sample.get("answers")
            if raw_answers is None:
                raw_answers = sample.get("answer")
            raw_target = sample.get("target_text")
            if raw_target in (None, ""):
                raw_target = raw_answers
            cap = _pick_caption(raw_target, train=self.train)
            audio_source = sample.get("audio")
            if audio_source is None:
                audio_source = sample.get("audio_path")
            if audio_source is None or not cap:
                continue
            refs_raw = raw_target
            refs: List[str] = []
            if isinstance(refs_raw, str):
                refs = [refs_raw.strip()] if refs_raw.strip() else []
            elif isinstance(refs_raw, (list, tuple)):
                refs = [str(x).strip() for x in refs_raw if str(x).strip()]
            if not refs:
                refs = [cap]
            audio.append(audio_source)
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
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        for module in self.audio_to_memory:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

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
        allowed_token_ids: Optional[torch.Tensor] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
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
            next_logits = logits[:, -1, :]
            if temperature <= 0:
                temperature = 1.0
            if do_sample:
                next_logits = next_logits / temperature
            if allowed_token_ids is not None:
                constrained = torch.full_like(next_logits, float("-inf"))
                constrained[:, allowed_token_ids] = next_logits[:, allowed_token_ids]
                next_logits = constrained
            if do_sample:
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_mask = cumulative_probs > top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
                    filtered = torch.full_like(next_logits, float("-inf"))
                    filtered.scatter_(1, sorted_indices, sorted_logits)
                    next_logits = filtered
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            finished |= next_token.squeeze(1).eq(eos_id)
            if finished.all():
                break
        return generated[:, 1:]


def build_datasets(
    data_path: str,
    use_wavcaps: bool,
    dataset_mode: str,
    train_manifest: Optional[str],
    val_manifest: Optional[str],
    media_root: Optional[str],
    target_field: str,
    dedup_by_audio: bool,
) -> Tuple[Dataset, Dataset]:
    if dataset_mode == "music_avqa":
        if not train_manifest or not val_manifest or not media_root:
            raise ValueError("music_avqa mode requires --train-manifest, --val-manifest, and --media-root")
        train_dataset = MusicAVQATextTargetDataset(
            Path(train_manifest), Path(media_root), target_field, dedup_by_audio=dedup_by_audio
        )
        val_dataset = MusicAVQATextTargetDataset(
            Path(val_manifest), Path(media_root), target_field, dedup_by_audio=dedup_by_audio
        )
        return train_dataset, val_dataset

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
    allowed_token_ids: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.eval()
    predictions: List[str] = []
    references: List[List[str]] = []
    total_loss = 0.0
    total_batches = 0
    printed_examples = 0
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

            generated_ids = model.generate(
                audio_embeddings,
                tokenizer,
                max_new_tokens=max_new_tokens,
                allowed_token_ids=allowed_token_ids,
            )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions.extend([" ".join(p.strip().split()) for p in preds])
            references.extend(batch["references"])
            if printed_examples < 2:
                for pred, refs in zip(preds[: 2 - printed_examples], batch["references"][: 2 - printed_examples]):
                    ref_preview = ", ".join(refs[:2])
                    print(f"[eval-sample] pred={pred[:160]!r} refs={ref_preview[:200]}", flush=True)
                    printed_examples += 1
                    if printed_examples >= 2:
                        break

    metrics = compute_caption_metrics(predictions, references, compute_bertscore=False)
    metrics["loss"] = total_loss / max(total_batches, 1)
    return metrics


def evaluate_music_avqa_answers(
    model: CLAPQwenCaptionDecoder,
    clap: CLAPAudioEncoder,
    dataloader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int,
    allowed_token_ids: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.eval()
    raw_correct = 0
    extracted_correct = 0
    total = 0
    total_f1 = 0.0
    total_cat_f1 = 0.0
    total_loss = 0.0
    total_batches = 0
    printed_examples = 0

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

            generated_ids = model.generate(
                audio_embeddings,
                tokenizer,
                max_new_tokens=max_new_tokens,
                allowed_token_ids=allowed_token_ids,
            )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            preds = [" ".join(p.strip().split()) for p in preds]

            for pred, refs in zip(preds, batch["references"]):
                ref = refs[0] if refs else ""
                pred_norm = normalize_answer(pred)
                ref_norm = normalize_answer(ref)
                pred_extracted = extract_answer(pred)
                ref_extracted = extract_answer(ref)
                raw_correct += int(pred_norm == ref_norm)
                extracted_correct += int(pred_extracted == ref_extracted)
                total_f1 += token_f1(pred, ref)
                total_cat_f1 += categorical_f1(pred, ref)
                total += 1
                if printed_examples < 2:
                    print(f"[decoder-eval-sample] pred={pred[:160]!r} refs={ref[:200]}", flush=True)
                    printed_examples += 1

    return {
        "raw_em": 100.0 * raw_correct / max(total, 1),
        "extracted_em": 100.0 * extracted_correct / max(total, 1),
        "f1": 100.0 * total_f1 / max(total, 1),
        "cat_f1": 100.0 * total_cat_f1 / max(total, 1),
        "loss": total_loss / max(total_batches, 1),
        "n": total,
    }


def build_holdout_prompt(question: str, caption: str, mode: str) -> str:
    if mode == "audio":
        return (
            f"From audio: {caption}\n"
            f"Question: {question}\n"
            "Answer with exactly one short answer token (single word or number)."
        )
    if mode == "image":
        return (
            f"Question: {question}\n"
            "Answer with exactly one short answer token (single word or number)."
        )
    return (
        f"From audio: {caption}\n"
        f"Question: {question}\n"
        "Answer with exactly one short answer token (single word or number)."
    )


def evaluate_music_avqa_holdouts(
    decoder: CLAPQwenCaptionDecoder,
    clap: CLAPAudioEncoder,
    dataloader: DataLoader,
    engine: Any,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int,
    output_dir: Path,
    epoch_index: int,
    allowed_token_ids: Optional[torch.Tensor] = None,
) -> None:
    summaries: Dict[str, Dict[str, float]] = {}
    sample_print_limits = {"image": 2, "audio": 6, "both": 6}
    phase_label = "init" if epoch_index < 0 else str(epoch_index + 1)
    for mode in ("image", "audio", "both"):
        raw_correct = 0
        extracted_correct = 0
        total = 0
        total_f1 = 0.0
        total_cat_f1 = 0.0
        printed_examples = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                audio_embeddings = clap(batch["audio"]).to(device)
                generated_ids = decoder.generate(
                    audio_embeddings,
                    tokenizer,
                    max_new_tokens=max_new_tokens,
                    allowed_token_ids=allowed_token_ids,
                )
                captions = tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                captions = [" ".join(x.strip().split()) for x in captions]
                prompts = [
                    build_holdout_prompt(question=q, caption=cap, mode=mode)
                    for q, cap in zip(batch["questions"], captions)
                ]
                images = batch["images"] if mode in {"image", "both"} else None
                preds = engine.generate_batch(
                    prompts=prompts,
                    images=images,
                    device=device,
                    max_new_tokens=8,
                    num_beams=1,
                )
                for i, (pred, ref) in enumerate(zip(preds, batch["answers"])):
                    pred_norm = normalize_answer(pred)
                    ref_norm = normalize_answer(ref)
                    pred_extracted = extract_answer(pred)
                    ref_extracted = extract_answer(ref)
                    raw_correct += int(pred_norm == ref_norm)
                    extracted_correct += int(pred_extracted == ref_extracted)
                    total_f1 += token_f1(pred, ref)
                    total_cat_f1 += categorical_f1(pred, ref)
                    total += 1
                    if printed_examples < sample_print_limits[mode]:
                        question = batch["questions"][i]
                        caption = captions[i]
                        image_flag = batch["images"][i] is not None
                        print(
                            f"[holdout-sample:{mode}] q={question[:120]!r} "
                            f"audio_text={caption[:120]!r} "
                            f"pred={pred[:80]!r} ref={ref[:80]!r} "
                            f"has_image={image_flag}",
                            flush=True,
                        )
                        printed_examples += 1

                if (batch_idx + 1) % 100 == 0:
                    print(
                        f"[eval:{mode}] step={batch_idx + 1}/{len(dataloader)} "
                        f"raw_em={100.0 * raw_correct / max(total, 1):.2f}% "
                        f"extracted_em={100.0 * extracted_correct / max(total, 1):.2f}% "
                        f"cat_f1={100.0 * total_cat_f1 / max(total, 1):.2f}",
                        flush=True,
                    )

        summaries[mode] = {
            "raw_em": 100.0 * raw_correct / max(total, 1),
            "extracted_em": 100.0 * extracted_correct / max(total, 1),
            "f1": 100.0 * total_f1 / max(total, 1),
            "cat_f1": 100.0 * total_cat_f1 / max(total, 1),
            "n": total,
        }
        print(
            f"[eval:{mode}] complete phase={phase_label} "
            f"raw_em={summaries[mode]['raw_em']:.2f} "
            f"extracted_em={summaries[mode]['extracted_em']:.2f} "
            f"cat_f1={summaries[mode]['cat_f1']:.2f} "
            f"n={summaries[mode]['n']}",
            flush=True,
        )

    holdout_name = "holdout_epoch_init.json" if epoch_index < 0 else f"holdout_epoch_{epoch_index + 1}.json"
    with (output_dir / holdout_name).open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight CLAP->Qwen caption decoder.")
    parser.add_argument("--dataset-mode", type=str, default="audiocaption", choices=["audiocaption", "music_avqa"])
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--llm-model", type=str, default="models/OpenGVLab_InternVL3_5-8B")
    parser.add_argument("--train-manifest", type=str, default="")
    parser.add_argument("--val-manifest", type=str, default="")
    parser.add_argument("--media-root", type=str, default="")
    parser.add_argument("--target-field", type=str, default="answer")
    parser.add_argument("--dedup-by-audio", action="store_true")
    parser.add_argument("--constrained-decoding", action="store_true")
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
    parser.add_argument("--holdout-model-config", type=str, default="")
    parser.add_argument("--holdout-checkpoint", type=str, default="")
    parser.add_argument("--holdout-batch-size", type=int, default=2)
    parser.add_argument("--holdout-backend", type=str, default="raw_internvl", choices=["raw_internvl", "safe"])
    return parser.parse_args()


def maybe_subset(dataset: Dataset, max_samples: int) -> Dataset:
    if max_samples and max_samples > 0:
        return torch.utils.data.Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def iter_target_texts(dataset: Dataset) -> List[str]:
    if isinstance(dataset, torch.utils.data.Subset):
        parent = dataset.dataset
        subset_indices = list(dataset.indices)
        if isinstance(parent, MusicAVQATextTargetDataset):
            texts = []
            for rel_idx in subset_indices:
                base_idx = parent.indices[rel_idx]
                text = str(parent.base.rows[base_idx].get(parent.target_field, "") or "").strip()
                if text:
                    texts.append(text)
            return texts
        return []
    if isinstance(dataset, MusicAVQATextTargetDataset):
        texts = []
        for base_idx in dataset.indices:
            text = str(dataset.base.rows[base_idx].get(dataset.target_field, "") or "").strip()
            if text:
                texts.append(text)
        return texts
    return []


def build_allowed_token_ids(
    tokenizer: Any,
    texts: Sequence[str],
) -> torch.Tensor:
    allowed = set()
    special_ids = [
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
    ]
    for token_id in special_ids:
        if token_id is not None:
            allowed.add(int(token_id))
    for text in texts:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        allowed.update(int(x) for x in encoded)
    return torch.tensor(sorted(allowed), dtype=torch.long)


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

    train_dataset, val_dataset = build_datasets(
        data_path=args.data_path,
        use_wavcaps=args.use_wavcaps,
        dataset_mode=args.dataset_mode,
        train_manifest=args.train_manifest or None,
        val_manifest=args.val_manifest or None,
        media_root=args.media_root or None,
        target_field=args.target_field,
        dedup_by_audio=args.dedup_by_audio,
    )
    train_dataset = maybe_subset(train_dataset, args.max_train_samples)
    val_dataset = maybe_subset(val_dataset, args.max_val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=CaptionBatchCollator(tokenizer, args.max_length, train=True, target_field=args.target_field),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=CaptionBatchCollator(tokenizer, args.max_length, train=False, target_field=args.target_field),
    )
    holdout_loader: Optional[DataLoader] = None
    holdout_engine: Optional[Any] = None
    if args.dataset_mode == "music_avqa" and args.holdout_model_config:
        holdout_dataset = ManifestAVQADataset(Path(args.val_manifest), Path(args.media_root))
        holdout_dataset = maybe_subset(holdout_dataset, args.max_val_samples)
        holdout_loader = DataLoader(
            holdout_dataset,
            batch_size=args.holdout_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_avqa,
        )
        if args.holdout_backend == "raw_internvl":
            print(f"[holdout] loading raw InternVL directly from {args.llm_model}", flush=True)
            holdout_engine = RawInternVLEvalEngine(llm_model=args.llm_model, device=device)
        else:
            holdout_cfg_name = args.holdout_model_config
            if holdout_cfg_name == "internvl":
                holdout_cfg_name = "rkca_joint"
                print(
                    "[holdout] remapping holdout_model_config=internvl -> rkca_joint "
                    "to use the raw frozen InternVL text+image path",
                    flush=True,
                )
            llm_cfg = get_config(holdout_cfg_name)
            holdout_model = create_model(llm_cfg).to(device)
            holdout_ckpt = Path(args.holdout_checkpoint) if args.holdout_checkpoint else None
            if holdout_ckpt is not None and holdout_ckpt.exists():
                load_checkpoint(
                    model=holdout_model,
                    optimizer=None,
                    scheduler=None,
                    checkpoint_path=holdout_ckpt,
                    device=device,
                )
                print(f"[holdout] loaded checkpoint: {holdout_ckpt}", flush=True)
            else:
                print("[holdout] using raw frozen base model (no holdout checkpoint provided)", flush=True)
            holdout_model.eval()
            holdout_base_model = holdout_model.module if hasattr(holdout_model, "module") else holdout_model
            holdout_engine = SafeEvalEngine(
                base_model=holdout_base_model,
                tokenizer=holdout_base_model.base_vl.tokenizer,
            )

    clap = CLAPAudioEncoder(freeze=True).to(device)
    clap.eval()

    allowed_token_ids: Optional[torch.Tensor] = None
    if args.constrained_decoding:
        target_texts = iter_target_texts(train_dataset) + iter_target_texts(val_dataset)
        allowed_token_ids = build_allowed_token_ids(tokenizer, target_texts).to(device)
        allowed_vocab_preview = tokenizer.batch_decode(
            allowed_token_ids[: min(32, allowed_token_ids.numel())].unsqueeze(1),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        print(
            f"[constraint] enabled allowed_tokens={allowed_token_ids.numel()} "
            f"preview={allowed_vocab_preview[:12]}",
            flush=True,
        )
        with (output_dir / "allowed_token_ids.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "count": int(allowed_token_ids.numel()),
                    "token_ids": [int(x) for x in allowed_token_ids.tolist()],
                },
                f,
                indent=2,
            )

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

    if args.dataset_mode == "music_avqa":
        best_metric_name = "extracted_em"
        best_metric = -1.0
    else:
        best_metric_name = "cider"
        best_metric = -1.0
    global_step = 0
    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Initial validation before any training update.
    if args.dataset_mode == "music_avqa":
        init_metrics = evaluate_music_avqa_answers(
            model=model,
            clap=clap,
            dataloader=val_loader,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
            allowed_token_ids=allowed_token_ids,
        )
    else:
        init_metrics = evaluate(
            model=model,
            clap=clap,
            dataloader=val_loader,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
            allowed_token_ids=allowed_token_ids,
        )
    print(f"[init-eval] {json.dumps(init_metrics, indent=2)}", flush=True)
    if holdout_loader is not None and holdout_engine is not None:
        evaluate_music_avqa_holdouts(
            decoder=model,
            clap=clap,
            dataloader=holdout_loader,
            engine=holdout_engine,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
            output_dir=output_dir,
            epoch_index=-1,
            allowed_token_ids=allowed_token_ids,
        )

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
            if batch_idx == 0 and epoch == 0:
                min_label = int(valid_labels.min().item()) if valid_labels.numel() > 0 else -1
                unique_labels = int(torch.unique(valid_labels).numel()) if valid_labels.numel() > 0 else 0
                print(
                    f"[debug:init] decoder_input_shape={tuple(decoder_input_ids.shape)} "
                    f"labels_shape={tuple(labels.shape)} vocab={logits.size(-1)} "
                    f"label_min={min_label} label_max={max_label if valid_labels.numel() > 0 else -1} "
                    f"label_unique={unique_labels}",
                    flush=True,
                )
                print(
                    f"[debug:init] logits_mean={float(logits.mean().item()):.4f} "
                    f"logits_std={float(logits.std().item()):.4f} "
                    f"logits_absmax={float(logits.abs().max().item()):.4f}",
                    flush=True,
                )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            seen_batches += 1
            global_step += 1

            if (batch_idx + 1) % 100 == 0:
                logits_detached = logits.detach()
                print(
                    f"[train] epoch={epoch + 1} step={batch_idx + 1}/{len(train_loader)} "
                    f"loss={running_loss / max(seen_batches, 1):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"grad={grad_norm:.2f} "
                    f"logit_mean={float(logits_detached.mean().item()):.2f} "
                    f"logit_std={float(logits_detached.std().item()):.2f} "
                    f"logit_absmax={float(logits_detached.abs().max().item()):.2f}",
                    flush=True,
                )

        if args.dataset_mode == "music_avqa":
            metrics = evaluate_music_avqa_answers(
                model=model,
                clap=clap,
                dataloader=val_loader,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=args.max_new_tokens,
                allowed_token_ids=allowed_token_ids,
            )
        else:
            metrics = evaluate(
                model=model,
                clap=clap,
                dataloader=val_loader,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=args.max_new_tokens,
                allowed_token_ids=allowed_token_ids,
            )
        print(f"[eval] epoch={epoch + 1} {json.dumps(metrics, indent=2)}", flush=True)
        if holdout_loader is not None and holdout_engine is not None:
            evaluate_music_avqa_holdouts(
                decoder=model,
                clap=clap,
                dataloader=holdout_loader,
                engine=holdout_engine,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=args.max_new_tokens,
                output_dir=output_dir,
                epoch_index=epoch,
                allowed_token_ids=allowed_token_ids,
            )

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
                "target_field": args.target_field,
                "dedup_by_audio": args.dedup_by_audio,
                "constrained_decoding": args.constrained_decoding,
            },
        }
        torch.save(checkpoint, output_dir / "checkpoint_last.pt")
        metric_value = float(metrics.get(best_metric_name, 0.0))
        if metric_value > best_metric:
            best_metric = metric_value
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")
            print(
                f"[save] best checkpoint -> {output_dir / 'checkpoint_best.pt'} "
                f"({best_metric_name}={metric_value:.4f})",
                flush=True,
            )


if __name__ == "__main__":
    main()
