#!/usr/bin/env python3
"""
Train/evaluate SAFE on AVQA/MUSIC-AVQA composition QA.

This is a Pre-FFN-first experiment runner for ECCV composition work.
It expects train/val manifests in JSONL where each row has:
  - question
  - answer
  - audio_path (optional for image-only ablations)
  - image_path (optional for audio-only ablations)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
# tqdm removed — use explicit print logging for clean stdout/stderr separation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.model_configs import get_config
from safe.models.safe_model import SAFEModel

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    # Strip common LLM preamble patterns
    for prefix in ("the answer is", "answer:", "a:", "it is", "this is"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.strip().rstrip(".")
    return " ".join(text.split())


# Known MUSIC-AVQA answer vocabulary (from official dataset)
AVQA_ANSWER_VOCAB = [
    "yes", "no",
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "left", "right",
    "accordion", "acoustic_guitar", "bagpipe", "banjo", "bassoon",
    "cello", "clarinet", "congas", "drum", "electric_bass", "erhu",
    "flute", "guzheng", "piano", "pipa", "saxophone", "trumpet",
    "tuba", "ukulele", "violin", "xylophone",
]

# Aliases: common LLM outputs that map to canonical answers
_ANSWER_ALIASES = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven",
    "guitar": "acoustic_guitar", "electric guitar": "electric_bass",
    "drums": "drum", "conga": "congas",
}


def extract_answer(raw_pred: str, vocab: List[str] = AVQA_ANSWER_VOCAB) -> str:
    """Extract the best matching answer from generated text against known vocabulary."""
    norm = normalize_answer(raw_pred)
    # Exact match first
    if norm in vocab:
        return norm
    # Check aliases
    if norm in _ANSWER_ALIASES:
        return _ANSWER_ALIASES[norm]
    # Check if any vocab answer appears as substring in the prediction
    # Prefer longer matches (e.g., "acoustic_guitar" over "guitar")
    found = []
    for ans in vocab:
        # Match with underscores replaced by spaces too
        ans_space = ans.replace("_", " ")
        if ans in norm or ans_space in norm:
            found.append(ans)
    if found:
        return max(found, key=len)
    # Check aliases as substrings
    for alias, canonical in _ANSWER_ALIASES.items():
        if alias in norm:
            return canonical
    # No match — return normalized prediction as-is
    return norm


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


class ManifestAVQADataset(Dataset):
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
        self.media_stats = self._compute_media_stats(max_samples=512)

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_media_path(self, path_value: str) -> Optional[Path]:
        if not path_value:
            return None
        p = Path(path_value).expanduser()
        # Try absolute path as-is
        if p.is_absolute():
            if p.exists():
                return p
            return None
        # Try relative to media_root
        candidate = self.media_root / p
        if candidate.exists():
            return candidate
        # Try relative to repo root (common when manifests store repo-relative paths)
        repo_candidate = REPO_ROOT / p
        if repo_candidate.exists():
            return repo_candidate
        # Try just the filename under common AVQA/MUSIC-AVQA subdirs.
        stem = p.stem
        for subdir in ("audio", "audio_old", "frames", "image", "image_31", "images"):
            for ext in (".wav", ".mp3", ".m4a", ".flac", ".jpg", ".jpeg", ".png"):
                c = self.media_root / subdir / f"{stem}{ext}"
                if c.exists():
                    return c
        # Try recursive stem search as last resort for heterogeneous layouts.
        for subdir in ("audio", "audio_old", "frames", "image", "image_31", "images"):
            base = self.media_root / subdir
            if not base.exists():
                continue
            for ext in (".wav", ".mp3", ".m4a", ".flac", ".jpg", ".jpeg", ".png"):
                matches = list(base.rglob(f"{stem}{ext}"))
                if matches:
                    return matches[0]
        return None

    def _compute_media_stats(self, max_samples: int = 512) -> Dict[str, int]:
        """
        Lightweight sanity stats to verify manifest/media alignment before training.
        Uses path resolution only (no audio/image decoding).
        """
        n = min(len(self.rows), max_samples)
        audio_ok = 0
        image_ok = 0
        both_ok = 0
        for i in range(n):
            row = self.rows[i]
            has_audio = self._resolve_media_path(row.get("audio_path", "")) is not None
            has_image = self._resolve_media_path(row.get("image_path", "")) is not None
            if has_audio:
                audio_ok += 1
            if has_image:
                image_ok += 1
            if has_audio and has_image:
                both_ok += 1
        return {
            "checked_samples": n,
            "audio_paths_found": audio_ok,
            "image_paths_found": image_ok,
            "both_paths_found": both_ok,
        }

    def _load_image(self, path_value: str) -> Optional[Image.Image]:
        image_path = self._resolve_media_path(path_value)
        if image_path is None:
            return None
        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            return None

    def _load_audio(self, path_value: str) -> Optional[Sequence[Any]]:
        audio_path = self._resolve_media_path(path_value)
        if audio_path is None:
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


def collate_avqa(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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
    cfg = get_config("phase1")
    if args.llm_model:
        cfg["llm_model_name"] = args.llm_model

    cfg["num_audio_tokens"] = args.num_audio_tokens
    cfg["fusion_type"] = "multilayer"  # Required for mid-layer hook-based Pre-FFN fusion
    cfg["fusion_layer_indices"] = [int(x) for x in args.fusion_layers.split(",")]
    cfg["freeze_base_vl"] = True
    cfg["freeze_audio_encoder"] = args.freeze_audio_encoder

    fusion_cfg = dict(cfg.get("fusion_config", {}))
    fusion_cfg["fusion_mode"] = "residual"
    fusion_cfg["injection_point"] = "pre_ffn"
    fusion_cfg.setdefault("use_bottleneck", True)
    fusion_cfg.setdefault("bottleneck_dim", 256)
    cfg["fusion_config"] = fusion_cfg
    # Only pass keys that SAFEModel.__init__ accepts
    valid_keys = {
        "llm_model_name", "vision_model_name",
        "audio_encoder_type", "audio_encoder_config",
        "projector_type", "num_audio_tokens", "projector_config",
        "fusion_type", "fusion_layer_indices", "lora_rank", "fusion_config",
        "freeze_base_vl", "freeze_audio_encoder", "label_smoothing",
        "llm_hidden_size", "audio_embed_dim",
    }
    return {k: v for k, v in cfg.items() if k in valid_keys}


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
    num_batches = len(dataloader)
    log_every = max(1, num_batches // 20)  # Log ~20 times per epoch
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        mm = resolve_modality_batch(batch, args.train_modality)

        # Debug: log first batch modality presence
        if step == 0:
            if mm["audio"] is None:
                print("  [debug] batch 0: audio input disabled by train_modality", flush=True)
            else:
                audio_ok = sum(1 for a in mm["audio"] if a is not None)
                print(f"  [debug] batch 0: {audio_ok}/{len(mm['audio'])} samples have audio", flush=True)
            if mm["images"] is None:
                print("  [debug] batch 0: image input disabled by train_modality", flush=True)
            else:
                image_ok = sum(1 for img in mm["images"] if img is not None)
                print(f"  [debug] batch 0: {image_ok}/{len(mm['images'])} samples have images", flush=True)

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
        if step == 0:
            pv = inputs.get("pixel_values")
            if pv is None:
                print("  [debug] batch 0: pixel_values=None (vision not entering model)", flush=True)
            else:
                print(f"  [debug] batch 0: pixel_values shape={tuple(pv.shape)}", flush=True)

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

            # Skip batch if loss has no gradient (e.g., all audio failed to load)
            if loss is None or not loss.requires_grad:
                if step == 0:
                    print("  [warn] first batch loss has no grad — check audio/image loading", flush=True)
                continue

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

        if (step + 1) % log_every == 0 or step == 0:
            avg_loss = total_loss / max(total_batches, 1)
            print(f"  [train] step {step + 1}/{num_batches} loss={avg_loss:.4f}", flush=True)

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
    extracted_total = 0.0
    f1_total = 0.0
    count = 0
    by_type: Dict[str, Dict[str, float]] = defaultdict(lambda: {"exact": 0.0, "extracted": 0.0, "f1": 0.0, "n": 0.0})

    eval_batches = len(dataloader)
    eval_log_every = max(1, eval_batches // 5)  # Log ~5 times per eval
    for eval_step, batch in enumerate(dataloader):
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
            qtype = batch["question_types"][i] if "question_types" in batch else "unknown"

            norm_ref = normalize_answer(ref)
            norm_pred = normalize_answer(pred)
            exact = float(norm_pred == norm_ref)
            # Extracted match: map generated text to closest known answer
            extracted_pred = extract_answer(pred)
            extracted = float(extracted_pred == norm_ref)
            f1 = token_f1(pred, ref)
            exact_total += exact
            extracted_total += extracted
            f1_total += f1
            count += 1

            by_type[qtype]["exact"] += exact
            by_type[qtype]["extracted"] += extracted
            by_type[qtype]["f1"] += f1
            by_type[qtype]["n"] += 1.0

        if (eval_step + 1) % eval_log_every == 0:
            running_em = 100.0 * exact_total / max(1, count)
            running_ext = 100.0 * extracted_total / max(1, count)
            print(f"  [eval:{modality}] step {eval_step + 1}/{eval_batches} raw_em={running_em:.2f}% extracted_em={running_ext:.2f}%", flush=True)

    result = {
        "modality": modality,
        "exact_match": 100.0 * exact_total / max(1, count),
        "extracted_match": 100.0 * extracted_total / max(1, count),
        "token_f1": 100.0 * f1_total / max(1, count),
        "num_samples": count,
        "by_question_type": {},
    }
    for k, v in by_type.items():
        n = max(1.0, v["n"])
        result["by_question_type"][k] = {
            "exact_match": 100.0 * v["exact"] / n,
            "extracted_match": 100.0 * v["extracted"] / n,
            "token_f1": 100.0 * v["f1"] / n,
            "num_samples": int(v["n"]),
        }
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AVQA/MUSIC-AVQA composition training")
    p.add_argument("--dataset", type=str, default="music_avqa", choices=["avqa", "music_avqa"])
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--llm-model", type=str, default=None)
    p.add_argument("--fusion-layers", type=str, default="1,5,9,13,17,21")
    p.add_argument("--num-audio-tokens", type=int, default=8)

    p.add_argument("--train-modality", type=str, default="both", choices=["audio", "image", "both"])
    p.add_argument("--eval-modalities", type=str, default="both,audio,image")
    p.add_argument("--fusion-gate", type=float, default=1.0)

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-answer-tokens", type=int, default=5)

    p.add_argument("--freeze-audio-encoder", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="SAFE-AVQA-Composition")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] dataset={args.dataset} device={device}")

    train_ds = ManifestAVQADataset(args.train_manifest, args.media_root)
    val_ds = ManifestAVQADataset(args.val_manifest, args.media_root)
    print(f"[info] train_samples={len(train_ds)} val_samples={len(val_ds)}")
    print(f"[info] train_media_stats={train_ds.media_stats}")
    print(f"[info] val_media_stats={val_ds.media_stats}")

    wandb_run = None
    if args.wandb:
        if wandb is None:
            print("[warn] --wandb requested but wandb is not installed. Continuing without wandb.")
        else:
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                tags=tags if tags else None,
                config={
                    "dataset": args.dataset,
                    "architecture": "pre_ffn",
                    "train_modality": args.train_modality,
                    "eval_modalities": args.eval_modalities,
                    "fusion_layers": args.fusion_layers,
                    "num_audio_tokens": args.num_audio_tokens,
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "seed": args.seed,
                    "train_manifest": str(args.train_manifest),
                    "val_manifest": str(args.val_manifest),
                },
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_avqa,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_avqa,
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
    print(f"[info] trainable_parameters={sum(p.numel() for p in trainable_params if p.requires_grad):,}")
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.fp16)

    best_score = -1.0
    history: List[Dict[str, Any]] = []
    eval_modalities = [m.strip() for m in args.eval_modalities.split(",") if m.strip()]

    # Image-only = eval-only baseline (frozen LLaVA, no trainable params touch loss)
    if args.train_modality == "image":
        print("\n[image-only] No trainable params in image path — running eval-only baseline")
        model.eval()
        epoch_result: Dict[str, Any] = {"epoch": 0, "train_loss": 0.0, "eval": {}}
        for modality in eval_modalities:
            metrics = evaluate(model, val_loader, tokenizer, device, modality, args)
            epoch_result["eval"][modality] = metrics
            print(f"  [eval:{modality}] raw_em={metrics['exact_match']:.2f} extracted_em={metrics['extracted_match']:.2f} f1={metrics['token_f1']:.2f} n={metrics['num_samples']}")
        history.append(epoch_result)
        best_score = epoch_result["eval"].get("image", {}).get("extracted_match", -1.0)
        if wandb_run is not None:
            log_payload: Dict[str, Any] = {"epoch": 0}
            for modality, metrics in epoch_result["eval"].items():
                log_payload[f"val/{modality}/exact_match"] = metrics["exact_match"]
                log_payload[f"val/{modality}/extracted_match"] = metrics["extracted_match"]
                log_payload[f"val/{modality}/token_f1"] = metrics["token_f1"]
            wandb_run.log(log_payload, step=0)
    else:
        for epoch in range(args.num_epochs):
            print(f"\n[epoch {epoch + 1}/{args.num_epochs}]")
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device, args)
            epoch_result = {"epoch": epoch + 1, "train_loss": float(train_loss), "eval": {}}

            for modality in eval_modalities:
                metrics = evaluate(model, val_loader, tokenizer, device, modality, args)
                epoch_result["eval"][modality] = metrics
                print(f"  [eval:{modality}] raw_em={metrics['exact_match']:.2f} extracted_em={metrics['extracted_match']:.2f} f1={metrics['token_f1']:.2f} n={metrics['num_samples']}")

            history.append(epoch_result)
            if wandb_run is not None:
                log_payload = {
                    "epoch": epoch + 1,
                    "train/loss": float(train_loss),
                }
                for modality, metrics in epoch_result["eval"].items():
                    log_payload[f"val/{modality}/exact_match"] = metrics["exact_match"]
                    log_payload[f"val/{modality}/extracted_match"] = metrics["extracted_match"]
                    log_payload[f"val/{modality}/token_f1"] = metrics["token_f1"]
                wandb_run.log(log_payload, step=epoch + 1)

            # Track best score using extracted_match (more fair for generative models)
            score_key = args.train_modality  # "both" or "audio"
            score = epoch_result["eval"].get(score_key, {}).get("extracted_match", -1.0)
            if score > best_score:
                best_score = score
                ckpt_path = args.output_dir / "best_model.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"  [save] best checkpoint -> {ckpt_path}")

            with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        final_path = args.output_dir / "final_model.pt"
        torch.save(model.state_dict(), final_path)

    results = {
        "dataset": args.dataset,
        "architecture": "pre_ffn",
        "train_modality": args.train_modality,
        "eval_modalities": eval_modalities,
        "best_exact_match": best_score,
        "best_modality": args.train_modality,
        "history_path": str(args.output_dir / "history.json"),
    }
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("[summary] " + json.dumps(results, indent=2))

    if wandb_run is not None:
        wandb_run.summary["best_exact_match"] = best_score
        wandb_run.summary["results_path"] = str(args.output_dir / "results.json")
        wandb_run.finish()


if __name__ == "__main__":
    main()
