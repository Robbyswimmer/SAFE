#!/usr/bin/env python3
"""
Train/evaluate SAFE on EPIC-SOUNDS AV-QA composition.

Supports architecture ablation:
- `pre_ffn` (residual fusion before FFN)
- `kv_augment` (audio injected as additional K/V memory)

Supports modality utilization checks:
- both (vision + audio)
- image (vision only)
- audio (audio only)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs.model_configs import get_config
from safe.models.safe_model import SAFEModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    return " ".join(text.split())


def token_f1(pred: str, ref: str) -> float:
    p = normalize_answer(pred).split()
    r = normalize_answer(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    common = 0
    r_counts: Dict[str, int] = defaultdict(int)
    for t in r:
        r_counts[t] += 1
    for t in p:
        if r_counts[t] > 0:
            common += 1
            r_counts[t] -= 1
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(r)
    return 2 * precision * recall / (precision + recall)


class EpicSoundsAVQADataset(Dataset):
    def __init__(self, manifest_path: Path, media_root: Path):
        self.manifest_path = manifest_path
        self.media_root = media_root
        self.rows: List[Dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, rel_path: str) -> Optional[Image.Image]:
        if not rel_path:
            return None
        image_path = self.media_root / rel_path
        if not image_path.exists():
            return None
        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            return None

    def _load_audio(self, rel_path: str) -> Optional[Sequence[Any]]:
        if not rel_path:
            return None
        audio_path = self.media_root / rel_path
        if not audio_path.exists():
            return None
        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(audio_path))
            if waveform.dim() == 2 and waveform.size(0) > 1:
                waveform = waveform.mean(dim=0)
            elif waveform.dim() == 2:
                waveform = waveform.squeeze(0)

            target_sr = 48000
            if sr != target_sr:
                waveform = torchaudio.functional.resample(
                    waveform.unsqueeze(0), sr, target_sr
                ).squeeze(0)
                sr = target_sr
            return (waveform, sr)
        except Exception:
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        return {
            "sample_id": row.get("sample_id", f"sample_{idx}"),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "question_type": row.get("question_type", "unknown"),
            "audio": self._load_audio(row.get("audio_path", "")),
            "image": self._load_image(row.get("image_path", "")),
        }


def collate_epic(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_ids": [],
        "questions": [],
        "answers": [],
        "question_types": [],
        "audio": [],
        "images": [],
    }
    for item in batch:
        out["sample_ids"].append(item["sample_id"])
        out["questions"].append(item["question"])
        out["answers"].append(item["answer"])
        out["question_types"].append(item["question_type"])
        out["audio"].append(item["audio"])
        out["images"].append(item["image"])
    return out


def build_model_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.architecture == "kv_augment":
        cfg = get_config("kv_augment")
    else:
        cfg = get_config("phase1")

    if args.llm_model:
        cfg["llm_model_name"] = args.llm_model

    cfg["num_audio_tokens"] = args.num_audio_tokens
    cfg["fusion_layer_indices"] = [int(x) for x in args.fusion_layers.split(",")]
    cfg["freeze_base_vl"] = True
    cfg["freeze_audio_encoder"] = args.freeze_audio_encoder

    fusion_cfg = dict(cfg.get("fusion_config", {}))
    if args.architecture == "pre_ffn":
        fusion_cfg["fusion_mode"] = "residual"
        fusion_cfg["injection_point"] = "pre_ffn"
        fusion_cfg.setdefault("use_bottleneck", True)
        fusion_cfg.setdefault("bottleneck_dim", 256)
    else:
        fusion_cfg["fusion_mode"] = "kv_augment"
        fusion_cfg.setdefault("query_adapter_rank", args.kv_query_rank)
        fusion_cfg.setdefault("bottleneck_dim", args.kv_bottleneck_dim)
        fusion_cfg.setdefault("head_dim", 128)
        fusion_cfg.setdefault("num_attention_heads", 40)
    cfg["fusion_config"] = fusion_cfg

    return cfg


def resolve_modality_batch(batch: Dict[str, Any], modality: str) -> Dict[str, Any]:
    if modality == "audio":
        return {"audio": batch["audio"], "images": None}
    if modality == "image":
        return {"audio": None, "images": batch["images"]}
    return {"audio": batch["audio"], "images": batch["images"]}


def train_epoch(
    model: SAFEModel,
    dataloader: DataLoader,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    pbar = tqdm(dataloader, desc="train")
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        mm = resolve_modality_batch(batch, args.train_modality)

        inputs = model.prepare_multimodal_inputs(
            text=batch["questions"],
            images=mm["images"],
            audio=mm["audio"],
            answers=batch["answers"],
            device=str(device),
            training_mode=True,
        )

        audio_tokens = inputs.pop("audio_tokens", None)
        audio_mask = inputs.pop("audio_attention_mask", None)

        with autocast(enabled=args.fp16):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=inputs.get("labels"),
                pixel_values=inputs.get("pixel_values"),
                audio_tokens=audio_tokens,
                audio_attention_mask=audio_mask,
                gate=args.fusion_gate,
            )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss = loss / args.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * args.gradient_accumulation_steps
        total_batches += 1
        pbar.set_postfix(loss=f"{total_loss / max(total_batches, 1):.4f}")

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(
    model: SAFEModel,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    modality: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model.eval()

    exact_total = 0.0
    f1_total = 0.0
    count = 0

    by_type: Dict[str, Dict[str, float]] = defaultdict(lambda: {"exact": 0.0, "f1": 0.0, "n": 0.0})

    for batch in tqdm(dataloader, desc=f"eval:{modality}"):
        mm = resolve_modality_batch(batch, modality)
        inputs = model.prepare_multimodal_inputs(
            text=batch["questions"],
            images=mm["images"],
            audio=mm["audio"],
            answers=None,
            device=str(device),
            training_mode=False,
        )

        audio_tokens = inputs.pop("audio_tokens", None)
        audio_mask = inputs.pop("audio_attention_mask", None)

        output_ids = model.generate(
            text=None,
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            audio_tokens=audio_tokens,
            audio_attention_mask=audio_mask,
            gate=args.fusion_gate,
            max_new_tokens=args.max_answer_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )

        prompt_mask = inputs.get("attention_mask")
        for i in range(output_ids.size(0)):
            prompt_len = int(prompt_mask[i].sum().item()) if prompt_mask is not None else inputs["input_ids"].size(1)
            gen = output_ids[i, prompt_len:]
            pred = tokenizer.decode(gen, skip_special_tokens=True).strip()
            ref = batch["answers"][i]
            qtype = batch["question_types"][i]

            exact = float(normalize_answer(pred) == normalize_answer(ref))
            f1 = token_f1(pred, ref)

            exact_total += exact
            f1_total += f1
            count += 1

            by_type[qtype]["exact"] += exact
            by_type[qtype]["f1"] += f1
            by_type[qtype]["n"] += 1.0

    result = {
        "modality": modality,
        "exact_match": 100.0 * exact_total / max(1, count),
        "token_f1": 100.0 * f1_total / max(1, count),
        "num_samples": count,
        "by_question_type": {},
    }

    for k, v in by_type.items():
        n = max(1.0, v["n"])
        result["by_question_type"][k] = {
            "exact_match": 100.0 * v["exact"] / n,
            "token_f1": 100.0 * v["f1"] / n,
            "num_samples": int(v["n"]),
        }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EPIC-SOUNDS AV-QA training")

    parser.add_argument("--data-root", type=Path, default=Path("experiments/epic_sounds_avqa_composition/data"))
    parser.add_argument("--train-manifest", type=Path, default=Path("experiments/epic_sounds_avqa_composition/data/manifests/train.jsonl"))
    parser.add_argument("--val-manifest", type=Path, default=Path("experiments/epic_sounds_avqa_composition/data/manifests/validation.jsonl"))
    parser.add_argument("--media-root", type=Path, default=Path("experiments/epic_sounds_avqa_composition/data/processed"))
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--architecture", type=str, default="pre_ffn", choices=["pre_ffn", "kv_augment"])
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--fusion-layers", type=str, default="1,5,9,13,17,21")
    parser.add_argument("--num-audio-tokens", type=int, default=8)
    parser.add_argument("--kv-query-rank", type=int, default=16)
    parser.add_argument("--kv-bottleneck-dim", type=int, default=64)

    parser.add_argument("--train-modality", type=str, default="both", choices=["audio", "image", "both"])
    parser.add_argument("--eval-modalities", type=str, default="both,audio,image")
    parser.add_argument("--fusion-gate", type=float, default=1.0)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-answer-tokens", type=int, default=16)

    parser.add_argument("--freeze-audio-encoder", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    train_ds = EpicSoundsAVQADataset(args.train_manifest, args.media_root)
    val_ds = EpicSoundsAVQADataset(args.val_manifest, args.media_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_epic,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_epic,
        pin_memory=torch.cuda.is_available(),
    )

    model_cfg = build_model_config(args)
    model = SAFEModel(**model_cfg)
    model.enable_audio_training()
    model.to_device(device)

    tokenizer = model.base_vl.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable_params = list(model.get_trainable_parameters())
    trainable_count = sum(p.numel() for p in trainable_params if p.requires_grad)
    print(f"[info] trainable parameters: {trainable_count:,}")

    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.fp16)

    best_score = -1.0
    history: List[Dict[str, Any]] = []
    eval_modalities = [m.strip() for m in args.eval_modalities.split(",") if m.strip()]

    for epoch in range(args.num_epochs):
        print(f"\n[epoch {epoch + 1}/{args.num_epochs}]")
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
        )

        epoch_result: Dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "eval": {},
        }

        for modality in eval_modalities:
            metrics = evaluate(
                model=model,
                dataloader=val_loader,
                tokenizer=tokenizer,
                device=device,
                modality=modality,
                args=args,
            )
            epoch_result["eval"][modality] = metrics
            print(
                f"  [eval:{modality}] exact={metrics['exact_match']:.2f} "
                f"f1={metrics['token_f1']:.2f} n={metrics['num_samples']}"
            )

        history.append(epoch_result)

        score = epoch_result["eval"].get("both", {}).get("exact_match", -1.0)
        if score > best_score:
            best_score = score
            ckpt_path = args.output_dir / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [save] best checkpoint -> {ckpt_path}")

        with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    final_path = args.output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n[done] final checkpoint -> {final_path}")

    result = {
        "architecture": args.architecture,
        "train_modality": args.train_modality,
        "eval_modalities": eval_modalities,
        "best_both_exact_match": best_score,
        "history_path": str(args.output_dir / "history.json"),
    }
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("[summary] " + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
