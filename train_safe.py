#!/usr/bin/env python3
"""
train_safe.py - Clean, optimized training script for SAFE audio captioning

Focuses on essentials:
- Audio captioning task (cross-entropy loss)
- CIDEr/BLEU evaluation metrics
- Memory-efficient training (gradient accumulation, mixed precision)
- Clear result tracking

Removed:
- Retention loss mechanisms (no longer needed)
- Curriculum learning
- Null-space projection
- SCST fine-tuning
- Complex dataset mixing
"""

import argparse
import json
import os
import platform
import random
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# Optional: Weights & Biases
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

# SAFE imports
from configs.model_configs import get_config
from safe.data.datasets import AudioCapsDataset, WavCapsDataset, create_safe_dataloader
from safe.models.safe_model import SAFEModel


# ============================================================================
# SECTION 1: UTILITIES
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------------------------------------------------------
# Weights & Biases helpers (cluster-friendly)
# ----------------------------------------------------------------------------

def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _get_rank() -> int:
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        value = _env_int(key)
        if value is not None:
            return value
    return 0


def _is_main_process() -> bool:
    return _get_rank() == 0


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _wandb_log(wandb_run: Any, data: Dict[str, Any], *, step: Optional[int] = None):
    if wandb_run is None:
        return
    try:
        if step is None:
            wandb_run.log(data)
        else:
            wandb_run.log(data, step=step)
    except Exception:
        pass


def _wandb_log_artifact(
    wandb_run: Any,
    *,
    artifact_name: str,
    artifact_type: str,
    files: List[Path],
    aliases: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    if wandb_run is None or wandb is None:
        return
    try:
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=_jsonify(metadata or {}),
        )
        for file_path in files:
            artifact.add_file(str(file_path))
        wandb_run.log_artifact(artifact, aliases=aliases or [])
    except Exception:
        pass


def _maybe_init_wandb(
    args: argparse.Namespace,
    *,
    train_config: Dict[str, Any],
    model_config: Dict[str, Any],
    output_dir: Path,
    model: Optional[nn.Module] = None,
    total_params: Optional[int] = None,
    trainable_params: Optional[int] = None,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
) -> Any:
    if not getattr(args, "wandb", False):
        return None
    if not _is_main_process():
        return None
    if wandb is None:
        print("⚠️  wandb failed to import; continuing without W&B logging.", flush=True)
        return None

    wandb_mode = getattr(args, "wandb_mode", None) or os.environ.get("WANDB_MODE")
    if wandb_mode is None:
        # Default to offline if no key is provided (prevents interactive login prompts on clusters).
        wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    if wandb_mode == "disabled":
        return None

    wandb_dir = getattr(args, "wandb_dir", None) or os.environ.get("WANDB_DIR") or str(output_dir / "wandb")
    tags: List[str] = []
    raw_tags = getattr(args, "wandb_tags", None)
    if raw_tags:
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    run_name = getattr(args, "wandb_name", None) or (f"{output_dir.name}-{slurm_job_id}" if slurm_job_id else output_dir.name)
    run_group = getattr(args, "wandb_group", None) or slurm_job_id

    try:
        settings = wandb.Settings(start_method="thread")
    except Exception:
        settings = None

    wandb_config = {
        "args": _jsonify(vars(args)),
        "train_config": _jsonify(train_config),
        "model_config": _jsonify(model_config),
        "data": _jsonify({"train_size": train_size, "val_size": val_size}),
        "params": _jsonify({"total": total_params, "trainable": trainable_params}),
        "system": _jsonify(
            {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "slurm_job_id": slurm_job_id,
            }
        ),
    }

    init_kwargs: Dict[str, Any] = dict(
        project=getattr(args, "wandb_project", None) or os.environ.get("WANDB_PROJECT") or "SAFE",
        entity=getattr(args, "wandb_entity", None) or os.environ.get("WANDB_ENTITY"),
        name=run_name,
        group=run_group,
        tags=tags if tags else None,
        dir=wandb_dir,
        mode=wandb_mode,
        config=wandb_config,
        save_code=bool(getattr(args, "wandb_log_code", False)),
        notes=getattr(args, "wandb_notes", None),
    )
    if settings is not None:
        init_kwargs["settings"] = settings

    try:
        wandb_run = wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"⚠️  wandb.init failed ({exc}); continuing without W&B logging.", flush=True)
        return None

    # Make charts use our optimizer step when provided.
    try:
        wandb.define_metric("train/optimizer_step")
        for prefix in ("train/*", "val/*", "train_subset/*", "eval/*"):
            wandb.define_metric(prefix, step_metric="train/optimizer_step")
    except Exception:
        pass

    watch_mode = getattr(args, "wandb_watch", "false")
    if model is not None and watch_mode and watch_mode.lower() != "false":
        try:
            wandb_run.watch(
                model,
                log=watch_mode,
                log_freq=int(getattr(args, "wandb_watch_log_freq", 500)),
            )
        except Exception:
            pass

    return wandb_run


def _is_valid_caption_reference(answer: Any) -> bool:
    """
    Return True if `answer` looks like a usable caption reference.

    We treat None/empty strings/empty lists and common sentinel strings like
    "None"/"nan"/"null" as invalid. Lists/dicts are considered valid if they
    contain at least one valid caption-like value.
    """
    if answer is None:
        return False

    if isinstance(answer, str):
        candidate = answer.strip().lower()
        return bool(candidate) and candidate not in {"none", "nan", "null"}

    if isinstance(answer, dict):
        candidate = (
            answer.get("answer")
            or answer.get("text")
            or answer.get("caption")
            or answer.get("captions")
            or answer.get("answers")
        )
        return _is_valid_caption_reference(candidate)

    if isinstance(answer, (list, tuple)):
        if not answer:
            return False
        return any(_is_valid_caption_reference(item) for item in answer)

    candidate = str(answer).strip().lower()
    return bool(candidate) and candidate not in {"none", "nan", "null"}


# Global cache for HuggingFace evaluate metrics (to avoid re-loading)
_EVALUATE_METRIC_CACHE: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Any] = {}


def _normalize_audio_caption(text: Any) -> str:
    """
    Normalization logic copied from StageATrainer._normalize_audio_caption
    so that metrics match the main training pipeline.
    """
    if text is None:
        return ""

    if isinstance(text, (list, tuple)):
        text = " ".join(str(t) for t in text if t)
    elif isinstance(text, dict):
        value = text.get("answer") or text.get("text")
        text = value if value is not None else ""

    import re
    import unicodedata

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\u2019", "'")  # Normalize curly apostrophes
    normalized = normalized.lower()

    # Collapse possessives before stripping punctuation so "dog's" -> "dogs"
    normalized = re.sub(r"'s\b", "s", normalized)

    # Remove residual apostrophes and punctuation (keep alphanumerics + whitespace)
    normalized = re.sub(r"'", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)

    tokens = [tok for tok in normalized.split() if tok]
    if not tokens:
        return ""

    number_map = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
    }

    cleaned_tokens: List[str] = []
    for tok in tokens:
        cleaned_tokens.append(number_map.get(tok, tok))

    if not cleaned_tokens:
        return ""

    return " ".join(cleaned_tokens)


def _embed_texts_for_contrastive(
    model: SAFEModel,
    texts: List[str],
    device: torch.device,
    max_length: int = 48,
) -> torch.Tensor:
    """
    Embed texts using the base LLM embedding layer, pooled over tokens.
    Mirrors StageATrainer._embed_texts behavior.
    """
    tokenizer = model.base_vl.tokenizer
    embedding_layer = model.base_vl.llm.get_input_embeddings()
    hidden_size = embedding_layer.weight.size(1)

    if not texts:
        return torch.empty(0, hidden_size, device=device)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        text_embeds = embedding_layer(input_ids)

    mask = attention_mask.unsqueeze(-1)
    pooled = (text_embeds * mask).sum(dim=1).float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    return pooled / denom

class MixedAudioCaptionDataset(Dataset):
    """
    Simple wrapper to mix AudioCaps and WavCaps with a given ratio.

    - Always uses all AudioCaps samples.
    - Uses wavcaps_ratio * len(WavCaps) samples (clipped to [0, len]).
    - Shuffles combined index map with a fixed seed.

    Memory-optimized: Uses numpy arrays instead of Python lists for index mapping.
    This reduces memory from ~40MB to ~4MB for 500K samples.
    """

    def __init__(
        self,
        audiocaps_dataset: Dataset,
        wavcaps_dataset: Optional[Dataset] = None,
        wavcaps_ratio: float = 0.8,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if audiocaps_dataset is None and wavcaps_dataset is None:
            raise ValueError("MixedAudioCaptionDataset requires at least one dataset")

        self.audiocaps_dataset = audiocaps_dataset
        self.wavcaps_dataset = wavcaps_dataset

        if wavcaps_dataset is None:
            # Only AudioCaps - use simple range
            a_count = len(audiocaps_dataset)
            self._dataset_ids = np.zeros(a_count, dtype=np.uint8)  # 0 = audiocaps
            self._local_indices = np.arange(a_count, dtype=np.int32)
        else:
            a_count = len(audiocaps_dataset)
            w_count = len(wavcaps_dataset)
            ratio = max(0.0, min(1.0, float(wavcaps_ratio)))
            w_samples = int(w_count * ratio)
            total = a_count + w_samples

            # Use numpy arrays for memory efficiency
            # dataset_ids: 0 = audiocaps, 1 = wavcaps (1 byte per sample vs ~50 bytes for string)
            # local_indices: int32 (4 bytes per sample vs 28 bytes for Python int)
            self._dataset_ids = np.concatenate([
                np.zeros(a_count, dtype=np.uint8),
                np.ones(w_samples, dtype=np.uint8),
            ])
            self._local_indices = np.concatenate([
                np.arange(a_count, dtype=np.int32),
                np.arange(w_samples, dtype=np.int32),
            ])

        if shuffle:
            rng = np.random.Generator(np.random.PCG64(seed))
            perm = rng.permutation(len(self._dataset_ids))
            self._dataset_ids = self._dataset_ids[perm]
            self._local_indices = self._local_indices[perm]

    def __len__(self) -> int:
        return len(self._dataset_ids)

    def __getitem__(self, idx: int):
        dataset_id = self._dataset_ids[idx]
        local_idx = int(self._local_indices[idx])

        if dataset_id == 0:
            return self.audiocaps_dataset[local_idx]
        else:
            return self.wavcaps_dataset[local_idx]


def _extract_answer_from_generation(generated_text: str) -> str:
    """
    Extract answer text from a chat-style generation.

    Mirrors StageATrainer._extract_answer(..., mode="audio"):
    - Prefer content after an "ASSISTANT:" marker.
    - Strip assistant-style prefixes.
    - Keep full caption (no sentence truncation).
    """
    if not generated_text:
        return ""

    import re

    answer = str(generated_text).strip()
    answer = re.sub(r"[\r\n]+", " ", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    if not answer:
        return ""

    lower_answer = answer.lower()

    # Extract text after ASSISTANT: marker if present
    assistant_match = re.search(r"(?:assistant|ssistant)\s*[:\-]\s*(.+)", lower_answer, re.IGNORECASE)
    if assistant_match:
        match_in_original = re.search(r"(?:assistant|ssistant)\s*[:\-]\s*(.+)", answer, re.IGNORECASE)
        if match_in_original:
            answer = match_in_original.group(1).strip()
            lower_answer = answer.lower()

    # Remove obvious assistant-style prefixes
    prefix_patterns = [
        r"^assistant\s*[:\-]\s*",
        r"^ssistant\s*[:\-]\s*",
        r"^ans(?:wer)?\s*[:\-]\s*",
        r"^ant\s*[:\-]\s*",
        r"^response\s*[:\-]\s*",
        r"^reply\s*[:\-]\s*",
    ]
    for pattern in prefix_patterns:
        if re.match(pattern, lower_answer):
            answer = re.sub(pattern, "", answer, count=1, flags=re.IGNORECASE).strip()
            lower_answer = answer.lower()
            break

    # Generic "prefix: value" handling
    if ":" in answer:
        prefix, remainder = answer.split(":", 1)
        prefix_clean = prefix.strip().lower()
        if (
            prefix_clean
            and len(prefix_clean.split()) <= 3
            and all(ch.isalpha() for ch in prefix_clean.replace(" ", ""))
        ):
            answer = remainder.strip()
            lower_answer = answer.lower()

    # Remove leading bullets / numbering
    answer = re.sub(r"^(?:[\-\*\u2022]+|\d+\.)\s*", "", answer)

    return answer.strip()

# ============================================================================
# SECTION 2: METRICS
# ============================================================================

def compute_cider(predictions: List[str], references: List[List[str]]) -> float:
    """
    Compute CIDEr score using pycocoevalcap

    Args:
        predictions: List of predicted captions
        references: List of reference caption lists (multiple refs per sample)

    Returns:
        CIDEr score (0-100 scale)
    """
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        print("⚠️  pycocoevalcap not available, CIDEr score will be 0.0")
        return 0.0

    # Convert to pycocoevalcap format: {id: [captions]}
    gts = {i: refs for i, refs in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}

    try:
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts, res)
        return float(score) * 100  # Scale to 0-100
    except Exception as e:
        print(f"⚠️  CIDEr computation failed: {e}")
        return 0.0


def compute_bleu(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Compute BLEU scores using pycocoevalcap

    Returns:
        Dict with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
    except ImportError:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    gts = {i: refs for i, refs in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}

    try:
        bleu_scorer = Bleu(4)
        scores, _ = bleu_scorer.compute_score(gts, res)
        return {
            "bleu1": float(scores[0]),
            "bleu2": float(scores[1]),
            "bleu3": float(scores[2]),
            "bleu4": float(scores[3]),
        }
    except Exception:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}


def compute_meteor(predictions: List[str], references: List[List[str]]) -> float:
    """Compute METEOR score"""
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        meteor_scorer = Meteor()
        score, _ = meteor_scorer.compute_score(gts, res)
        return float(score)
    except Exception:
        return 0.0


def compute_rouge(predictions: List[str], references: List[List[str]]) -> float:
    """Compute ROUGE-L score"""
    try:
        from pycocoevalcap.rouge.rouge import Rouge
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        rouge_scorer = Rouge()
        score, _ = rouge_scorer.compute_score(gts, res)
        return float(score)
    except Exception:
        return 0.0


def compute_caption_metrics(
    predictions: List[str],
    references: List[List[str]],
    compute_bertscore: bool = False,
    light_metrics: bool = False,
) -> Dict[str, float]:
    """
    Compute caption metrics, mirroring StageATrainer's metric stack as closely
    as possible while keeping training-time evaluation lightweight when
    `light_metrics=True`.
    """
    # Default structure so callers can always rely on these keys
    metrics: Dict[str, float] = {
        "cider": 0.0,
        "spice": 0.0,
        "spider": 0.0,
        "bleu1": 0.0,
        "bleu2": 0.0,
        "bleu3": 0.0,
        "bleu4": 0.0,
        "meteor": 0.0,
        "rouge_l": 0.0,
        "bertscore_f1": 0.0,
    }

    if not predictions or not references:
        return metrics

    # Normalize and filter empty pairs (match StageATrainer behaviour)
    paired: List[Tuple[str, List[str]]] = []
    for pred, refs in zip(predictions, references):
        pred_clean = _normalize_audio_caption(str(pred).strip())
        refs_clean = [
            _normalize_audio_caption(str(ref).strip())
            for ref in refs
            if str(ref).strip()
        ]
        pred_clean = pred_clean.strip()
        refs_clean = [r for r in refs_clean if r.strip()]
        if pred_clean and refs_clean:
            paired.append((pred_clean, refs_clean))

    if not paired:
        return metrics

    preds_list, refs_list = zip(*paired)
    preds_list = list(preds_list)
    refs_list = [list(r) for r in refs_list]

    # Reference statistics (useful sanity check; mirrors StageATrainer logs)
    ref_counts = [len(r) for r in refs_list]
    if ref_counts:
        avg_refs = sum(ref_counts) / len(ref_counts)
        min_refs = min(ref_counts)
        max_refs = max(ref_counts)
        print(
            f"[RefValidation] References per sample: avg={avg_refs:.1f}, "
            f"min={min_refs}, max={max_refs}, total_samples={len(refs_list)}",
            flush=True,
        )
        if avg_refs < 2.0:
            print(
                f"⚠️  WARNING: Low reference count (avg={avg_refs:.1f}). "
                f"AudioCaps-style CIDEr expects ~5 refs/sample.",
                flush=True,
            )

    # HuggingFace evaluate metrics (BLEU/METEOR/ROUGE and optional BERTScore)
    try:
        import evaluate

        def _metric(name: str, **load_kwargs):
            key = (name, tuple(sorted(load_kwargs.items())))
            if key not in _EVALUATE_METRIC_CACHE:
                _EVALUATE_METRIC_CACHE[key] = evaluate.load(name, **load_kwargs)
            return _EVALUATE_METRIC_CACHE[key]

        # BLEU
        try:
            bleu_metric = _metric("bleu")
            bleu_result = bleu_metric.compute(predictions=preds_list, references=refs_list)
            if bleu_result:
                precisions = bleu_result.get("precisions", [])
                for n in range(min(4, len(precisions))):
                    metrics[f"bleu{n + 1}"] = float(precisions[n])
        except Exception as exc:
            print(f"⚠️  BLEU metric failed: {exc}", flush=True)

        # METEOR (always computed if available)
        try:
            meteor_metric = _metric("meteor")
            meteor_result = meteor_metric.compute(predictions=preds_list, references=refs_list)
            if meteor_result and "meteor" in meteor_result:
                metrics["meteor"] = float(meteor_result["meteor"])
        except Exception as exc:
            print(f"⚠️  METEOR metric failed: {exc}", flush=True)

        # ROUGE-L (best over references per sample, always computed if available)
        try:
            rouge_metric = _metric("rouge")
            rouge_scores: List[float] = []
            for pred, refs in zip(preds_list, refs_list):
                best = 0.0
                for ref in refs:
                    try:
                        result = rouge_metric.compute(predictions=[pred], references=[ref])
                        best = max(best, float(result.get("rougeL", 0.0)))
                    except Exception as rouge_exc:
                        print(f"⚠️  ROUGE-L metric failed on sample: {rouge_exc}", flush=True)
                rouge_scores.append(best)
            if rouge_scores:
                metrics["rouge_l"] = float(sum(rouge_scores) / len(rouge_scores))
        except Exception as exc:
            print(f"⚠️  ROUGE-L metric failed: {exc}", flush=True)

        # Optional BERTScore via evaluate (only in heavy eval mode)
        if compute_bertscore and not light_metrics:
            try:
                bert_metric = _metric("bertscore")
                bert_result = bert_metric.compute(
                    predictions=preds_list,
                    references=[r[0] for r in refs_list],
                    lang="en",
                )
                if "f1" in bert_result:
                    f1_scores = bert_result["f1"]
                    if isinstance(f1_scores, (list, tuple)) and len(f1_scores) > 0:
                        metrics["bertscore_f1"] = float(sum(f1_scores) / len(f1_scores))
            except Exception as exc:
                print(f"⚠️  BERTScore metric failed: {exc}", flush=True)

    except Exception as exc:
        # If evaluate is unavailable, we still compute CIDEr/SPICE below
        print(
            f"⚠️  evaluate library unavailable for BLEU/METEOR/ROUGE/BERTScore: {exc}",
            flush=True,
        )

    # CIDEr + SPICE (pycocoevalcap) – match StageATrainer behaviour
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice

        gts = {str(i): refs for i, refs in enumerate(refs_list)}
        res = {str(i): [pred] for i, pred in enumerate(preds_list)}

        try:
            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(gts, res)
            metrics["cider"] = float(cider_score) * 100.0
        except Exception as exc:
            print(f"⚠️  CIDEr metric failed: {exc}", flush=True)

        # SPICE is heavy (Java CoreNLP); only compute when light_metrics=False
        if not light_metrics:
            try:
                spice_scorer = Spice()
                spice_score, _ = spice_scorer.compute_score(gts, res)
                metrics["spice"] = float(spice_score) * 100.0
            except Exception as exc:
                print(f"⚠️  SPICE metric failed: {exc}", flush=True)

        if metrics["cider"] > 0.0 and metrics["spice"] > 0.0:
            metrics["spider"] = (metrics["cider"] + metrics["spice"]) / 2.0

    except ImportError as exc:
        print(f"⚠️  pycocoevalcap unavailable for CIDEr/SPICE: {exc}", flush=True)

    return metrics


# ============================================================================
# SECTION 3: EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate(
    model: SAFEModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    max_new_tokens: int = 20,
    num_beams: int = 1,
    compute_bertscore: bool = False,
    light_metrics: bool = False,
    suppress_eos_for_audio: bool = True,
    sample_output: Optional[List[Dict[str, Any]]] = None,
    sample_limit: int = 0,
) -> Dict[str, float]:
    """
    Evaluate model on audio captioning task

    Args:
        model: SAFE model
        dataloader: Validation dataloader
        device: Device to run on
        max_batches: Maximum batches to evaluate (None = all)
        max_new_tokens: Max tokens to generate
        num_beams: Beam search size
        compute_bertscore: Whether to compute BERTScore

    Returns:
        Dict with metrics (loss, cider, bleu4, etc.)
    """
    model.eval()

    # Ensure audio fusion is fully enabled during evaluation
    if hasattr(model, "set_gate"):
        try:
            model.set_gate(1.0)
        except Exception:
            pass

    # CRITICAL: Configure generation parameters to prevent hanging
    tokenizer = model.base_vl.tokenizer

    # Ensure pad_token exists
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    # Set generation config on the LLM to prevent conflicts
    if hasattr(model.base_vl.llm, 'config'):
        model.base_vl.llm.config.pad_token_id = tokenizer.pad_token_id
        model.base_vl.llm.config.eos_token_id = tokenizer.eos_token_id

    if hasattr(model.base_vl.llm, 'generation_config'):
        model.base_vl.llm.generation_config.pad_token_id = tokenizer.pad_token_id
        model.base_vl.llm.generation_config.eos_token_id = tokenizer.eos_token_id
        # Override max_length to respect max_new_tokens limit
        model.base_vl.llm.generation_config.max_length = None

    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_references = []

    print(f"Running evaluation (max_batches={max_batches})...", flush=True)
    start_time = time.time()
    skipped_batches = 0
    last_skip_log_time = start_time

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Move batch to device
        questions = batch["questions"]
        answers = batch["answers"]
        audio = batch["audio"]
        has_audio_flags = batch.get("has_audio", None)

        # Filter out samples with missing audio OR missing caption references.
        # - Missing audio can cause generation to hang (zero-filled audio tokens).
        # - Missing captions produce meaningless loss/metrics and can hide training failure.
        valid_indices = [
            i
            for i, (a, ans) in enumerate(zip(audio, answers))
            if a is not None and _is_valid_caption_reference(ans)
        ]

        if not valid_indices:
            skipped_batches += 1
            missing_audio = sum(1 for a in audio if a is None)
            missing_caps = sum(1 for ans in answers if not _is_valid_caption_reference(ans))
            now = time.time()
            if batch_idx < 5 or (now - last_skip_log_time) > 60:
                print(
                    f"  ⚠️  Skipping batch {batch_idx} - "
                    f"missing_audio={missing_audio}/{len(audio)} missing_captions={missing_caps}/{len(answers)} "
                    f"(skipped_batches={skipped_batches})",
                    flush=True,
                )
                last_skip_log_time = now
            continue

        if len(valid_indices) < len(audio):
            if batch_idx < 5:
                missing_audio = sum(1 for a in audio if a is None)
                missing_caps = sum(1 for ans in answers if not _is_valid_caption_reference(ans))
                print(
                    f"  ⚠️  Filtering batch {batch_idx} - "
                    f"kept={len(valid_indices)}/{len(audio)} "
                    f"missing_audio={missing_audio} missing_captions={missing_caps}",
                    flush=True,
                )
            questions = [questions[i] for i in valid_indices]
            answers = [answers[i] for i in valid_indices]
            audio = [audio[i] for i in valid_indices]

        # Prepare inputs (ensure correct device)
        inputs = model.prepare_multimodal_inputs(
            text=questions,
            audio=audio,
            answers=answers,
            device=device,
            training_mode=True,  # For loss computation
        )

        # Move inputs to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs["labels"].to(device)
        audio_tokens = inputs.get("audio_tokens")
        if audio_tokens is not None:
            audio_tokens = audio_tokens.to(device)
        audio_attention_mask = inputs.get("audio_attention_mask")
        if audio_attention_mask is not None:
            audio_attention_mask = audio_attention_mask.to(device)

        # Compute loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            audio_tokens=audio_tokens,
            audio_attention_mask=audio_attention_mask,
        )

        loss = outputs.get("loss")
        if loss is not None:
            total_loss += loss.item()
            num_batches += 1

        # Generate predictions (reuse same device)
        generation_inputs = model.prepare_multimodal_inputs(
            text=questions,
            audio=audio,
            answers=None,  # No answers for generation
            device=device,
            training_mode=False,
        )

        gen_input_ids = generation_inputs["input_ids"].to(device)
        gen_attention_mask = generation_inputs["attention_mask"].to(device)
        gen_audio_tokens = generation_inputs.get("audio_tokens")
        if gen_audio_tokens is not None:
            gen_audio_tokens = gen_audio_tokens.to(device)
        gen_audio_attention_mask = generation_inputs.get("audio_attention_mask")
        if gen_audio_attention_mask is not None:
            gen_audio_attention_mask = gen_audio_attention_mask.to(device)

        # Build generation kwargs and optionally suppress EOS for audio batches
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 1,
            "num_beams": num_beams,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if suppress_eos_for_audio and has_audio_flags is not None:
            if torch.is_tensor(has_audio_flags):
                has_audio_any = bool(has_audio_flags.any().item())
            else:
                has_audio_any = any(bool(x) for x in has_audio_flags)
            if has_audio_any and tokenizer.eos_token_id is not None:
                suppress_tokens = [tokenizer.eos_token_id]
                if (
                    tokenizer.pad_token_id is not None
                    and tokenizer.pad_token_id != tokenizer.eos_token_id
                ):
                    suppress_tokens.append(tokenizer.pad_token_id)
                generation_kwargs["suppress_tokens"] = suppress_tokens

        # Generate captions
        generated_ids = model.generate(
            input_ids=gen_input_ids,
            attention_mask=gen_attention_mask,
            audio_tokens=gen_audio_tokens,
            audio_attention_mask=gen_audio_attention_mask,
            **generation_kwargs,
        )

        # Decode predictions
        tokenizer = model.base_vl.tokenizer
        batch_predictions = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Clean predictions (remove question prompt and extract answer content) + collect references
        for i, pred in enumerate(batch_predictions):
            question = questions[i]
            if question and question in pred:
                pred = pred.replace(question, "").strip()
            pred_answer = _extract_answer_from_generation(pred)
            cleaned_pred = pred_answer if pred_answer else pred.strip()
            all_predictions.append(cleaned_pred)

            answer = answers[i]
            if isinstance(answer, str):
                refs = [answer]
            elif isinstance(answer, list):
                refs = [str(a) for a in answer]
            else:
                refs = [str(answer)]
            all_references.append(refs)

            if sample_output is not None and sample_limit > 0 and len(sample_output) < sample_limit:
                sample_output.append(
                    {
                        "question": str(question),
                        "prediction": str(cleaned_pred),
                        "references": [str(r) for r in refs],
                    }
                )

        if (batch_idx + 1) % 10 == 0:
            print(f"  Evaluated {batch_idx + 1} batches...", flush=True)

    elapsed = time.time() - start_time

    # Compute metrics
    print(f"[Metrics] Computing caption metrics on {len(all_predictions)} predictions...", flush=True)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    caption_metrics = compute_caption_metrics(
        all_predictions,
        all_references,
        compute_bertscore=compute_bertscore,
        light_metrics=light_metrics,
    )

    metrics = {
        "loss": avg_loss,
        **caption_metrics,
        "num_samples": len(all_predictions),
        "eval_time": elapsed,
    }

    print(f"[Metrics] Caption metrics computed.", flush=True)
    print(f"✓ Evaluation complete ({format_time(elapsed)})", flush=True)
    print(f"  Loss: {avg_loss:.4f}", flush=True)
    # Print all key caption metrics in a consistent order
    metric_keys = [
        "bleu1",
        "bleu2",
        "bleu3",
        "bleu4",
        "meteor",
        "rouge_l",
        "cider",
        "spice",
        "spider",
        "bertscore_f1",
    ]
    for key in metric_keys:
        if key in metrics:
            value = metrics[key]
            if key in ["cider", "spice", "spider"]:
                print(f"  {key.upper()}: {value:.2f}", flush=True)
            elif key == "bertscore_f1":
                print(f"  BERTSCORE_F1: {value:.4f}", flush=True)
            else:
                print(f"  {key.upper()}: {value:.4f}", flush=True)

    # Log sample predictions
    print(f"\n📝 Sample predictions:", flush=True)
    for i in range(min(3, len(all_predictions))):
        print(f"  [{i+1}] Pred: {all_predictions[i]}", flush=True)
        print(f"      Refs: {all_references[i]}", flush=True)

    # Explicit memory cleanup to prevent OOM during long training runs
    del all_predictions, all_references
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return metrics


# ============================================================================
# SECTION 4: TRAINING
# ============================================================================

def train_epoch(
    model: SAFEModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    scaler: Optional[GradScaler] = None,
    wandb_run: Any = None,
    optimizer_step: int = 0,
) -> Tuple[Dict[str, float], int]:
    """
    Train for one epoch

    Args:
        model: SAFE model
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        config: Training configuration
        scaler: GradScaler for mixed precision (optional)

    Returns:
        Dict with training metrics
    """
    # Ensure audio components are in training mode while keeping base VL frozen
    if hasattr(model, "enable_audio_training"):
        model.enable_audio_training()
    else:
        model.train()

    total_loss = 0.0
    num_batches = 0
    num_samples = 0

    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    use_amp = config.get("fp16", False) and scaler is not None

    optimizer.zero_grad()

    start_time = time.time()
    last_log_time = start_time
    step_start_time = start_time
    step_loss_sum = 0.0
    step_samples = 0
    step_micro_batches = 0
    last_proj_grad_norm: Optional[float] = None
    last_fuse_grad_norm: Optional[float] = None
    skipped_batches = 0
    skipped_samples = 0
    filtered_samples = 0
    missing_audio_samples = 0
    missing_caption_samples = 0
    last_skip_log_time = start_time

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        questions = batch["questions"]
        answers = batch["answers"]
        audio = batch["audio"]

        # Skip samples with missing audio OR missing captions.
        if isinstance(audio, list):
            valid_indices = [
                i
                for i, (a, ans) in enumerate(zip(audio, answers))
                if a is not None and _is_valid_caption_reference(ans)
            ]
            if not valid_indices:
                skipped_batches += 1
                skipped_samples += len(audio)
                missing_audio = sum(1 for a in audio if a is None)
                missing_caps = sum(1 for ans in answers if not _is_valid_caption_reference(ans))
                missing_audio_samples += missing_audio
                missing_caption_samples += missing_caps

                now = time.time()
                if batch_idx < 5 or (now - last_skip_log_time) > 60:
                    print(
                        f"  ⚠️  Skipping batch {batch_idx} - "
                        f"missing_audio={missing_audio}/{len(audio)} missing_captions={missing_caps}/{len(answers)} "
                        f"(skipped_batches={skipped_batches})",
                        flush=True,
                    )
                    last_skip_log_time = now
                continue
            if len(valid_indices) < len(audio):
                filtered_samples += len(audio) - len(valid_indices)
                missing_audio = sum(1 for a in audio if a is None)
                missing_caps = sum(1 for ans in answers if not _is_valid_caption_reference(ans))
                missing_audio_samples += missing_audio
                missing_caption_samples += missing_caps

                if batch_idx < 5:
                    print(
                        f"  ⚠️  Filtering batch {batch_idx} - "
                        f"kept={len(valid_indices)}/{len(audio)} "
                        f"missing_audio={missing_audio} missing_captions={missing_caps}",
                        flush=True,
                    )
                questions = [questions[i] for i in valid_indices]
                answers = [answers[i] for i in valid_indices]
                audio = [audio[i] for i in valid_indices]

        # Prepare inputs
        inputs = model.prepare_multimodal_inputs(
            text=questions,
            audio=audio,
            answers=answers,
            device=device,
            training_mode=True
        )

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs["labels"].to(device)
        audio_tokens = inputs.get("audio_tokens")
        if audio_tokens is not None:
            audio_tokens = audio_tokens.to(device)
        audio_attention_mask = inputs.get("audio_attention_mask")
        if audio_attention_mask is not None:
            audio_attention_mask = audio_attention_mask.to(device)

        # Forward pass with optional mixed precision
        if use_amp:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    audio_tokens=audio_tokens,
                    audio_attention_mask=audio_attention_mask,
                )
                loss = outputs["loss"]
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                audio_tokens=audio_tokens,
                audio_attention_mask=audio_attention_mask,
            )
            loss = outputs["loss"]

        # Optional audio-text contrastive loss (InfoNCE-style) on this batch
        contrastive_weight = float(config.get("audio_contrastive_weight", 0.0) or 0.0)
        contrastive_temp = float(config.get("audio_contrastive_temperature", 0.07) or 0.07)
        if contrastive_weight > 0.0 and audio_tokens is not None:
            try:
                # Pool audio tokens to a single vector per sample
                # audio_tokens: (B, T, H) -> (B, H)
                audio_vecs = audio_tokens.mean(dim=1)

                # Resolve textual answers for each sample using SAFEModel helper
                resolved_answers: List[str] = []
                for ans in answers:
                    if hasattr(model, "_select_training_answer"):
                        text = model._select_training_answer(ans)
                    else:
                        text = ans if isinstance(ans, str) else (ans[0] if isinstance(ans, list) and ans else "")
                    resolved_answers.append(str(text or "").strip())

                # Filter out empty pairs
                paired_indices = [i for i, t in enumerate(resolved_answers) if t]
                if paired_indices:
                    audio_vecs = audio_vecs[paired_indices]
                    text_list = [resolved_answers[i] for i in paired_indices]

                    if audio_vecs.size(0) > 1:
                        text_vecs = _embed_texts_for_contrastive(
                            model,
                            text_list,
                            device=device,
                            max_length=int(config.get("audio_contrastive_max_length", 48) or 48),
                        )
                        if text_vecs.numel() > 0:
                            # Normalize
                            audio_norm = torch.nn.functional.normalize(audio_vecs, dim=-1)
                            text_norm = torch.nn.functional.normalize(text_vecs, dim=-1)
                            # Similarity matrix: (B, B)
                            sim = audio_norm @ text_norm.t() / max(contrastive_temp, 1e-5)
                            targets = torch.arange(sim.size(0), device=device)
                            contrastive_loss = torch.nn.functional.cross_entropy(sim, targets)
                            loss = loss + contrastive_weight * contrastive_loss
            except Exception:
                # Fail-safe: ignore contrastive errors to keep training running
                pass

        # If loss has no gradient path (e.g., SAFE gate effectively off),
        # skip this batch to avoid autograd errors.
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            # Optionally log once, but keep silent in normal operation
            continue

        # Normalize by gradient accumulation steps
        loss = loss / gradient_accumulation_steps

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Accumulate per-optimizer-step stats for W&B
        with torch.no_grad():
            try:
                loss_unscaled_value = float(loss.detach().item() * gradient_accumulation_steps)
            except Exception:
                loss_unscaled_value = 0.0
        step_loss_sum += loss_unscaled_value
        step_samples += len(questions)
        step_micro_batches += 1

        # Gradient health check (first 3 epochs, every 50 batches)
        # Detects learning failures early by monitoring audio component gradients
        if epoch <= 3 and batch_idx % 50 == 0:
            proj_grad_norm = 0.0
            fuse_grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if "audio_projector" in name:
                        proj_grad_norm += grad_norm
                    if "fusion_adapter" in name:
                        fuse_grad_norm += grad_norm
            print(f"[GradCheck] epoch={epoch} batch={batch_idx} proj={proj_grad_norm:.6f} fuse={fuse_grad_norm:.6f}", flush=True)
            last_proj_grad_norm = float(proj_grad_norm)
            last_fuse_grad_norm = float(fuse_grad_norm)

        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if use_amp:
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            optimizer_step += 1

            # W&B logging at optimizer-step granularity
            if wandb_run is not None:
                step_time = max(time.time() - step_start_time, 1e-6)
                lrs = scheduler.get_last_lr()
                log_dict: Dict[str, Any] = {
                    "train/optimizer_step": optimizer_step,
                    "epoch": epoch,
                    "train/batch_idx": batch_idx,
                    "train/micro_batches": step_micro_batches,
                    "train/loss_step": (step_loss_sum / max(step_micro_batches, 1)),
                    "train/samples_per_sec_step": step_samples / step_time,
                    "train/grad_norm": float(grad_norm) if grad_norm is not None else None,
                }
                for group_idx, (group, lr) in enumerate(zip(optimizer.param_groups, lrs)):
                    group_name = str(group.get("name") or f"group{group_idx}")
                    log_dict[f"train/lr/{group_name}"] = float(lr)

                if use_amp and scaler is not None:
                    try:
                        log_dict["train/amp_scale"] = float(scaler.get_scale())
                    except Exception:
                        pass

                if last_proj_grad_norm is not None:
                    log_dict["train/gradcheck/audio_projector_sum_norm"] = last_proj_grad_norm
                if last_fuse_grad_norm is not None:
                    log_dict["train/gradcheck/fusion_adapter_sum_norm"] = last_fuse_grad_norm

                if audio_tokens is not None:
                    try:
                        with torch.no_grad():
                            log_dict["train/audio_token_norm"] = float(audio_tokens.norm(dim=-1).mean().item())
                    except Exception:
                        pass

                try:
                    if hasattr(model, "get_last_attention_summary"):
                        summary = model.get_last_attention_summary()
                        if isinstance(summary, dict):
                            if summary.get("overall_mean", None) is not None:
                                log_dict["train/attn_mean"] = float(summary["overall_mean"])
                            if summary.get("overall_max", None) is not None:
                                log_dict["train/attn_max"] = float(summary["overall_max"])
                except Exception:
                    pass

                if torch.cuda.is_available():
                    try:
                        log_dict["train/gpu_mem_allocated_mb"] = float(torch.cuda.memory_allocated() / (1024**2))
                        log_dict["train/gpu_mem_reserved_mb"] = float(torch.cuda.memory_reserved() / (1024**2))
                        log_dict["train/gpu_max_mem_allocated_mb"] = float(
                            torch.cuda.max_memory_allocated() / (1024**2)
                        )
                    except Exception:
                        pass

                _wandb_log(wandb_run, log_dict, step=optimizer_step)

            # Reset step accumulators
            step_start_time = time.time()
            step_loss_sum = 0.0
            step_samples = 0
            step_micro_batches = 0

        # Logging
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        num_samples += len(questions)

        # Periodic logging
        current_time = time.time()
        if current_time - last_log_time > 60:  # Log every minute
            avg_loss = total_loss / num_batches
            elapsed = current_time - start_time
            samples_per_sec = num_samples / elapsed
            lr = scheduler.get_last_lr()[0]

            # Optional diagnostics for audio fusion strength
            audio_token_norm = None
            if audio_tokens is not None:
                try:
                    with torch.no_grad():
                        audio_token_norm = float(audio_tokens.norm(dim=-1).mean().item())
                except Exception:
                    audio_token_norm = None

            attn_mean = None
            attn_max = None
            try:
                if hasattr(model, "get_last_attention_summary"):
                    summary = model.get_last_attention_summary()
                    if isinstance(summary, dict):
                        attn_mean = summary.get("overall_mean", None)
                        attn_max = summary.get("overall_max", None)
            except Exception:
                attn_mean = attn_max = None

            log_msg = (
                f"[Epoch {epoch}] Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                f"Speed: {samples_per_sec:.1f} samples/s"
            )
            extras = []
            if audio_token_norm is not None:
                extras.append(f"audio_norm={audio_token_norm:.2f}")
            if attn_mean is not None and attn_max is not None:
                extras.append(f"attn_mean={attn_mean:.4f} attn_max={attn_max:.4f}")
            if extras:
                log_msg = f"{log_msg} | " + " ".join(extras)

            print(log_msg, flush=True)
            last_log_time = current_time

    # Final statistics
    if num_batches == 0:
        raise RuntimeError(
            f"No valid training batches were processed in epoch {epoch}. "
            f"Skipped_batches={skipped_batches}, dropped_samples={skipped_samples + filtered_samples}. "
            "This usually means your dataset has missing/empty captions and/or missing audio paths."
        )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    elapsed = time.time() - start_time

    metrics = {
        "loss": avg_loss,
        "num_samples": num_samples,
        "train_time": elapsed,
        "samples_per_sec": num_samples / elapsed,
        "skipped_batches": skipped_batches,
        "dropped_samples": skipped_samples + filtered_samples,
        "missing_audio_samples": missing_audio_samples,
        "missing_caption_samples": missing_caption_samples,
    }

    # Memory cleanup after training epoch to prevent OOM during long runs
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return metrics, optimizer_step


def train(
    model: SAFEModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    wandb_run: Any = None,
    wandb_log_checkpoints: bool = False,
    wandb_sample_count: int = 0,
) -> Dict[str, Any]:
    """
    Main training loop

    Args:
        model: SAFE model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration
        output_dir: Output directory for checkpoints
        device: Device to train on

    Returns:
        Training history dict
    """
    # Setup optimizer with different learning rates (Stage-A style defaults)
    lr_projector = config.get("learning_rate_projector", 2e-4)
    lr_adapter = config.get("learning_rate_adapter", 1e-4)
    weight_decay = config.get("weight_decay", 0.01)

    # Group parameters by component
    projector_params = []
    adapter_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "audio_projector" in name:
            projector_params.append(param)
        elif "fusion_adapter" in name or "lora" in name.lower():
            adapter_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": projector_params, "lr": lr_projector, "name": "projector"},
        {"params": adapter_params, "lr": lr_adapter, "name": "adapter"},
    ]

    if other_params:
        param_groups.append({"params": other_params, "lr": lr_adapter, "name": "other"})

    optimizer = AdamW(param_groups, weight_decay=weight_decay)

    # Learning rate scheduler
    num_epochs = config.get("num_epochs", 20)
    warmup_steps = config.get("warmup_steps", 1000)
    total_steps = len(train_loader) * num_epochs

    # Cosine schedule with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = GradScaler() if config.get("fp16", False) else None

    # Training loop
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_cider": [],
        "val_bleu4": [],
        "epochs": [],
    }

    best_cider = 0.0
    patience = config.get("early_stopping_patience", 5)
    patience_counter = 0

    # Global step approximation for scheduling (e.g., gate warmup)
    global_step = 0
    optimizer_step = 0

    print(f"\n{'='*80}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"  Projector LR: {lr_projector}")
    print(f"  Adapter LR: {lr_adapter}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Mixed precision: {config.get('fp16', False)}")
    print(f"{'='*80}\n")

    # Optional initial evaluation before training (sanity check)
    initial_max_eval = config.get("max_eval_batches", None)
    if initial_max_eval is not None and initial_max_eval <= 0:
        initial_max_eval = None
    print(f"[InitEval] Running initial evaluation on validation set (max_batches={initial_max_eval})", flush=True)
    init_samples: Optional[List[Dict[str, Any]]] = [] if (wandb_run is not None and wandb_sample_count > 0) else None
    init_metrics = evaluate(
        model,
        val_loader,
        device,
        max_batches=initial_max_eval,
        max_new_tokens=config.get("max_new_tokens", 20),
        num_beams=config.get("num_beams", 1),
        # Training-time eval: use light metrics (BLEU, METEOR, ROUGE, CIDEr), skip SPICE/BERTScore
        compute_bertscore=False,
        light_metrics=True,
        suppress_eos_for_audio=True,
        sample_output=init_samples,
        sample_limit=wandb_sample_count,
    )
    print(f"[InitEval] CIDEr={init_metrics.get('cider', 0.0):.2f} BLEU-4={init_metrics.get('bleu4', 0.0):.4f}", flush=True)
    if wandb_run is not None:
        init_log = {"train/optimizer_step": optimizer_step, "epoch": 0}
        init_log.update({f"val/{k}": v for k, v in init_metrics.items() if isinstance(v, (int, float))})
        _wandb_log(wandb_run, init_log, step=optimizer_step)
        if init_samples and wandb is not None:
            try:
                table = wandb.Table(columns=["question", "prediction", "references"])
                for row in init_samples:
                    table.add_data(row["question"], row["prediction"], "\n".join(row["references"]))
                _wandb_log(wandb_run, {"train/optimizer_step": optimizer_step, "val/samples": table}, step=optimizer_step)
            except Exception:
                pass

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}\n")

        # Train
        train_metrics, optimizer_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            config,
            scaler,
            wandb_run=wandb_run,
            optimizer_step=optimizer_step,
        )
        if wandb_run is not None:
            train_log = {"train/optimizer_step": optimizer_step, "epoch": epoch}
            train_log.update({f"train/epoch_{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))})
            _wandb_log(wandb_run, train_log, step=optimizer_step)

        # Gate warmup: ramp SAFE gate from 0 → 1 over a configured number of steps
        gate_warmup_steps = int(config.get("gate_warmup_steps", 0) or 0)
        if gate_warmup_steps > 0 and hasattr(model, "set_gate_warmup"):
            global_step += len(train_loader)
            model.set_gate_warmup(global_step, warmup_steps=gate_warmup_steps)

        print(f"\n✓ Training complete:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Time: {format_time(train_metrics['train_time'])}")
        print(f"  Speed: {train_metrics['samples_per_sec']:.1f} samples/sec")

        # Evaluate
        eval_frequency = config.get("eval_frequency", 1)
        if epoch % eval_frequency == 0:
            print(f"\n{'='*80}")
            print(f"Running validation...")
            print(f"{'='*80}\n")

            val_samples: Optional[List[Dict[str, Any]]] = [] if (wandb_run is not None and wandb_sample_count > 0) else None
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                max_batches=config.get("max_eval_batches"),
                max_new_tokens=config.get("max_new_tokens", 20),
                num_beams=config.get("num_beams", 1),
                # Training-time eval: light metrics (no SPICE/BERTScore).
                compute_bertscore=False,
                light_metrics=True,
                suppress_eos_for_audio=True,
                sample_output=val_samples,
                sample_limit=wandb_sample_count,
            )

            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_cider"].append(val_metrics["cider"])
            history["val_bleu4"].append(val_metrics["bleu4"])
            history["epochs"].append(epoch)

            # Save checkpoint
            is_best = val_metrics["cider"] > best_cider
            if is_best:
                best_cider = val_metrics["cider"]
                patience_counter = 0
                print(f"\n🎉 New best CIDEr: {best_cider:.2f}")
            else:
                patience_counter += 1

            ckpt_paths = save_checkpoint(
                model,
                optimizer,
                scheduler,
                {**train_metrics, **val_metrics, "epoch": epoch},
                output_dir,
                is_best=is_best,
            )

            # Save history
            with open(output_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            if wandb_run is not None:
                val_log = {
                    "train/optimizer_step": optimizer_step,
                    "epoch": epoch,
                    "val/best_cider": float(best_cider),
                    "val/is_best": int(bool(is_best)),
                    "early_stopping/patience_counter": int(patience_counter),
                }
                val_log.update({f"val/{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})
                _wandb_log(wandb_run, val_log, step=optimizer_step)
                if val_samples and wandb is not None:
                    try:
                        table = wandb.Table(columns=["question", "prediction", "references"])
                        for row in val_samples:
                            table.add_data(row["question"], row["prediction"], "\n".join(row["references"]))
                        _wandb_log(wandb_run, {"train/optimizer_step": optimizer_step, "val/samples": table}, step=optimizer_step)
                    except Exception:
                        pass
                if (
                    wandb_log_checkpoints
                    and is_best
                    and isinstance(ckpt_paths, dict)
                    and ckpt_paths.get("best") is not None
                ):
                    try:
                        run_id = getattr(wandb_run, "id", None) or "run"
                        _wandb_log_artifact(
                            wandb_run,
                            artifact_name=f"checkpoint-{run_id}",
                            artifact_type="model",
                            files=[ckpt_paths["best"]],
                            aliases=["best", f"epoch{epoch}"],
                            metadata={
                                "epoch": epoch,
                                "optimizer_step": optimizer_step,
                                "metrics": {**train_metrics, **val_metrics},
                                "output_dir": str(output_dir),
                            },
                        )
                    except Exception:
                        pass

            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered (patience={patience})")
                break

    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"  Best CIDEr: {best_cider:.2f}")
    print(f"  Checkpoints saved to: {output_dir}")
    print(f"{'='*80}\n")

    return history


# ============================================================================
# SECTION 5: CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(
    model: SAFEModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, float],
    output_dir: Path,
    is_best: bool = False,
):
    """Save model checkpoint"""
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }

    # Save last checkpoint
    checkpoint_path = output_dir / "checkpoint_last.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Saved checkpoint: {checkpoint_path}")

    # Save best checkpoint
    if is_best:
        best_path = output_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"💾 Saved BEST checkpoint: {best_path}")


def load_checkpoint(
    model: SAFEModel,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, float]:
    """Load model checkpoint"""
    print(f"📂 Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    metrics = checkpoint.get("metrics", {})
    print(f"✓ Checkpoint loaded (epoch={metrics.get('epoch', 'unknown')})")

    return metrics


# ============================================================================
# SECTION 6: MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SAFE model on audio captioning")

    # Model configuration
    parser.add_argument("--model-config", type=str, default="phase1",
                        choices=["demo", "full", "multimodal", "phase1"],
                        help="Model configuration name")

    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--train-split", type=str, default="train",
                        help="Training split name")
    parser.add_argument("--val-split", type=str, default="val",
                        help="Validation split name")
    parser.add_argument("--use-wavcaps", action="store_true",
                        help="Include WavCaps in training mix")
    parser.add_argument("--wavcaps-ratio", type=float, default=0.8,
                        help="Fraction of WavCaps train samples to include (0.0-1.0)")
    parser.add_argument("--wavcaps-split", type=str, default="train",
                        help="WavCaps split to use for training")

    # Training
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--num-epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=8,
                        help="Validation batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32,
                        help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")

    # Optimization
    # Match Phase 1 defaults from README: 1e-3 / 5e-4
    parser.add_argument("--learning-rate-projector", type=float, default=1e-3,
                        help="Learning rate for audio projector")
    parser.add_argument("--learning-rate-adapter", type=float, default=5e-4,
                        help="Learning rate for fusion adapter")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Warmup steps for learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")

    # Optional audio contrastive loss (Stage-A style)
    parser.add_argument("--audio-contrastive-weight", type=float, default=0.0,
                        help="Weight for optional audio-text contrastive loss (0.0 = disabled)")
    parser.add_argument("--audio-contrastive-temperature", type=float, default=0.07,
                        help="Temperature for audio-text contrastive loss")
    parser.add_argument("--audio-contrastive-max-length", type=int, default=48,
                        help="Max caption length (tokens) for contrastive text embeddings")
    parser.add_argument("--gate-warmup-steps", type=int, default=0,
                        help="If >0, ramp SAFE gate 0→1 over this many optimizer steps")

    # Evaluation
    parser.add_argument("--eval-frequency", type=int, default=1,
                        help="Evaluate every N epochs")
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="Max batches for validation (None = all)")
    parser.add_argument("--max-new-tokens", type=int, default=20,
                        help="Max new tokens for generation")
    parser.add_argument("--num-beams", type=int, default=1,
                        help="Beam search size")

    # Checkpointing
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only (requires --resume)")
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Early stopping patience (epochs)")

    # Memory optimization
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory (trades compute for memory)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model config
    print(f"\nLoading model config: {args.model_config}")
    model_config = get_config(args.model_config)

    # Whitelist of valid SAFEModel constructor arguments
    safe_model_keys = {
        "llm_model_name",
        "vision_model_name",
        "audio_encoder_type",
        "audio_encoder_config",
        "projector_type",
        "num_audio_tokens",
        "projector_config",
        "fusion_type",
        "fusion_layer_indices",
        "lora_rank",
        "fusion_config",
        "freeze_base_vl",
        "freeze_audio_encoder",
        "llm_hidden_size",
        "audio_embed_dim",
    }

    # Filter config to only include valid constructor arguments
    constructor_config = {k: v for k, v in model_config.items() if k in safe_model_keys}

    # Initialize model
    print(f"\nInitializing SAFE model...")
    print(f"  LLM: {constructor_config.get('llm_model_name', 'N/A')}")
    print(f"  Vision: {constructor_config.get('vision_model_name', 'N/A')}")
    print(f"  Audio: {constructor_config.get('audio_encoder_type', 'N/A')}")

    model = SAFEModel(**constructor_config)
    model = model.to(device)

    # Enable gradient checkpointing if requested (saves ~10-15GB memory)
    if args.gradient_checkpointing:
        print(f"  Enabling gradient checkpointing for memory optimization...")
        if hasattr(model.base_vl, 'llm') and hasattr(model.base_vl.llm, 'gradient_checkpointing_enable'):
            model.base_vl.llm.gradient_checkpointing_enable()
            print(f"  ✓ Gradient checkpointing enabled on LLM")
        else:
            print(f"  ⚠️  LLM does not support gradient checkpointing")

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\n📊 Model parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Load datasets
    print(f"\n📂 Loading datasets from: {args.data_path}")
    audiocaps_train = AudioCapsDataset(args.data_path, split=args.train_split)
    val_dataset = AudioCapsDataset(args.data_path, split=args.val_split)

    wavcaps_train = None
    if args.use_wavcaps:
        try:
            wavcaps_train = WavCapsDataset(args.data_path, split=args.wavcaps_split)
            print(f"  WavCaps train: {len(wavcaps_train)} samples (split='{args.wavcaps_split}')")
            print(f"  WavCaps ratio: {args.wavcaps_ratio:.2f}")
        except Exception as e:
            print(f"⚠️  Failed to load WavCaps dataset: {e}. Continuing with AudioCaps only.", flush=True)
            wavcaps_train = None

    if wavcaps_train is not None:
        train_dataset = MixedAudioCaptionDataset(
            audiocaps_dataset=audiocaps_train,
            wavcaps_dataset=wavcaps_train,
            wavcaps_ratio=args.wavcaps_ratio,
            shuffle=True,
            seed=args.seed,
        )
        print(f"  AudioCaps train: {len(audiocaps_train)} samples")
        print(f"  Mixed train samples: {len(train_dataset)} (AudioCaps + WavCaps subset)")
    else:
        train_dataset = audiocaps_train
        print(f"  Train (AudioCaps only): {len(train_dataset)} samples")

    print(f"  Val (AudioCaps): {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = create_safe_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = create_safe_dataloader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Training config
    config = {
        "num_epochs": args.num_epochs,
        "learning_rate_projector": args.learning_rate_projector,
        "learning_rate_adapter": args.learning_rate_adapter,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "fp16": args.fp16,
        "eval_frequency": args.eval_frequency,
        # Default eval cap if not specified to keep metrics manageable
        "max_eval_batches": args.max_eval_batches if args.max_eval_batches is not None else 50,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "early_stopping_patience": args.early_stopping_patience,
        "audio_contrastive_weight": args.audio_contrastive_weight,
        "audio_contrastive_temperature": args.audio_contrastive_temperature,
        "audio_contrastive_max_length": args.audio_contrastive_max_length,
        "gate_warmup_steps": args.gate_warmup_steps,
    }

    # Evaluation only mode
    if args.eval_only:
        if args.resume is None:
            raise ValueError("--eval-only requires --resume")

        load_checkpoint(model, None, None, Path(args.resume), device)

        print(f"\n{'='*80}")
        print(f"Running evaluation only")
        print(f"{'='*80}\n")

        metrics = evaluate(
            model,
            val_loader,
            device,
            max_batches=args.max_eval_batches,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            compute_bertscore=True,   # Full metrics in eval-only mode
            light_metrics=False,
            suppress_eos_for_audio=False,
        )

        # Save results
        results_path = output_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n💾 Results saved to: {results_path}")
        return

    # Resume from checkpoint if specified
    if args.resume:
        load_checkpoint(model, None, None, Path(args.resume), device)

    # Train
    train(
        model,
        train_loader,
        val_loader,
        config,
        output_dir,
        device,
    )

    print(f"\n✅ Training complete! Results saved to: {output_dir}")


# ============================================================================
# SECTION 5: CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(
    model: SAFEModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, float],
    output_dir: Path,
    is_best: bool = False,
) -> Dict[str, Path]:
    """Save model checkpoint"""
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }

    # Save last checkpoint
    checkpoint_path = output_dir / "checkpoint_last.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Saved checkpoint: {checkpoint_path}")
    paths: Dict[str, Path] = {"last": checkpoint_path}

    # Save best checkpoint
    if is_best:
        best_path = output_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"💾 Saved BEST checkpoint: {best_path}")
        paths["best"] = best_path

    return paths


def load_checkpoint(
    model: SAFEModel,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, float]:
    """Load model checkpoint"""
    print(f"📂 Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    metrics = checkpoint.get("metrics", {})
    print(f"✓ Checkpoint loaded (epoch={metrics.get('epoch', 'unknown')})")

    return metrics


# ============================================================================
# SECTION 6: MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SAFE model on audio captioning")

    # Model configuration
    parser.add_argument("--model-config", type=str, default="phase1",
                        choices=["demo", "full", "multimodal", "phase1"],
                        help="Model configuration name")

    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--train-split", type=str, default="train",
                        help="Training split name")
    parser.add_argument("--val-split", type=str, default="val",
                        help="Validation split name")
    parser.add_argument("--use-wavcaps", action="store_true",
                        help="Include WavCaps in training mix")
    parser.add_argument("--wavcaps-ratio", type=float, default=0.8,
                        help="Fraction of WavCaps train samples to include (0.0-1.0)")
    parser.add_argument("--wavcaps-split", type=str, default="train",
                        help="WavCaps split to use for training")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit number of training samples for smoke testing")

    # Training
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--num-epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=8,
                        help="Validation batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32,
                        help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")

    # Optimization
    # Match Phase 1 defaults from README: 1e-3 / 5e-4
    parser.add_argument("--learning-rate-projector", type=float, default=1e-3,
                        help="Learning rate for audio projector")
    parser.add_argument("--learning-rate-adapter", type=float, default=5e-4,
                        help="Learning rate for fusion adapter")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Warmup steps for learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")

    # Optional audio contrastive loss (Stage-A style)
    parser.add_argument("--audio-contrastive-weight", type=float, default=0.0,
                        help="Weight for optional audio-text contrastive loss (0.0 = disabled)")
    parser.add_argument("--audio-contrastive-temperature", type=float, default=0.07,
                        help="Temperature for audio-text contrastive loss")
    parser.add_argument("--audio-contrastive-max-length", type=int, default=48,
                        help="Max caption length (tokens) for contrastive text embeddings")
    parser.add_argument("--gate-warmup-steps", type=int, default=0,
                        help="If >0, ramp SAFE gate 0→1 over this many optimizer steps")

    # Evaluation
    parser.add_argument("--eval-frequency", type=int, default=1,
                        help="Evaluate every N epochs")
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="Max batches for validation (None = all)")
    parser.add_argument("--max-new-tokens", type=int, default=20,
                        help="Max new tokens for generation")
    parser.add_argument("--num-beams", type=int, default=1,
                        help="Beam search size")

    # Checkpointing
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only (requires --resume)")
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Early stopping patience (epochs)")

    # Memory optimization
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory (trades compute for memory)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")

    # Weights & Biases logging
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (defaults to $WANDB_PROJECT or 'SAFE')")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity (user/team).")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="W&B run name (defaults to output dir name).")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="W&B group (defaults to $SLURM_JOB_ID when set).")
    parser.add_argument("--wandb-tags", type=str, default=None,
                        help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-mode", type=str, default=None,
                        choices=["online", "offline", "disabled"],
                        help="W&B mode override. If unset, defaults to offline when WANDB_API_KEY is missing.")
    parser.add_argument("--wandb-dir", type=str, default=None,
                        help="Directory for W&B run files (recommended on clusters, e.g. $SCRATCH/wandb).")
    parser.add_argument("--wandb-notes", type=str, default=None,
                        help="Optional W&B notes.")
    parser.add_argument("--wandb-log-code", action="store_true",
                        help="Log a code snapshot to W&B (can be slow/large).")
    parser.add_argument("--wandb-watch", type=str, default="false",
                        choices=["false", "gradients", "parameters", "all"],
                        help="Enable wandb.watch (expensive).")
    parser.add_argument("--wandb-watch-log-freq", type=int, default=500,
                        help="wandb.watch log frequency.")
    parser.add_argument("--wandb-log-checkpoints", action="store_true",
                        help="Log best checkpoints as W&B artifacts.")
    parser.add_argument("--wandb-sample-count", type=int, default=3,
                        help="How many sample predictions to log per evaluation.")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model config
    print(f"\nLoading model config: {args.model_config}")
    model_config = get_config(args.model_config)

    # Whitelist of valid SAFEModel constructor arguments
    safe_model_keys = {
        "llm_model_name",
        "vision_model_name",
        "audio_encoder_type",
        "audio_encoder_config",
        "projector_type",
        "num_audio_tokens",
        "projector_config",
        "fusion_type",
        "fusion_layer_indices",
        "lora_rank",
        "fusion_config",
        "freeze_base_vl",
        "freeze_audio_encoder",
        "llm_hidden_size",
        "audio_embed_dim",
    }

    # Filter config to only include valid constructor arguments
    constructor_config = {k: v for k, v in model_config.items() if k in safe_model_keys}

    # Initialize model
    print(f"\nInitializing SAFE model...")
    print(f"  LLM: {constructor_config.get('llm_model_name', 'N/A')}")
    print(f"  Vision: {constructor_config.get('vision_model_name', 'N/A')}")
    print(f"  Audio: {constructor_config.get('audio_encoder_type', 'N/A')}")

    model = SAFEModel(**constructor_config)
    model = model.to(device)

    # Enable gradient checkpointing if requested (saves ~10-15GB memory)
    if args.gradient_checkpointing:
        print(f"  Enabling gradient checkpointing for memory optimization...")
        if hasattr(model.base_vl, 'llm') and hasattr(model.base_vl.llm, 'gradient_checkpointing_enable'):
            model.base_vl.llm.gradient_checkpointing_enable()
            print(f"  ✓ Gradient checkpointing enabled on LLM")
        else:
            print(f"  ⚠️  LLM does not support gradient checkpointing")

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\n📊 Model parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Load datasets
    print(f"\n📂 Loading datasets from: {args.data_path}")
    audiocaps_train = AudioCapsDataset(args.data_path, split=args.train_split)
    if args.max_train_samples is not None:
        print(f"  ⚠️  Limiting training samples to {args.max_train_samples} for smoke testing")
        indices = list(range(min(len(audiocaps_train), args.max_train_samples)))
        audiocaps_train = torch.utils.data.Subset(audiocaps_train, indices)
    
    val_dataset = AudioCapsDataset(args.data_path, split=args.val_split)

    wavcaps_train = None
    if args.use_wavcaps:
        try:
            wavcaps_train = WavCapsDataset(args.data_path, split=args.wavcaps_split)
            print(f"  WavCaps train: {len(wavcaps_train)} samples (split='{args.wavcaps_split}')")
            print(f"  WavCaps ratio: {args.wavcaps_ratio:.2f}")
        except Exception as e:
            print(f"⚠️  Failed to load WavCaps dataset: {e}. Continuing with AudioCaps only.", flush=True)
            wavcaps_train = None

    if wavcaps_train is not None:
        train_dataset = MixedAudioCaptionDataset(
            audiocaps_dataset=audiocaps_train,
            wavcaps_dataset=wavcaps_train,
            wavcaps_ratio=args.wavcaps_ratio,
            shuffle=True,
            seed=args.seed,
        )
        print(f"  AudioCaps train: {len(audiocaps_train)} samples")
        print(f"  Mixed train samples: {len(train_dataset)} (AudioCaps + WavCaps subset)")
    else:
        train_dataset = audiocaps_train
        print(f"  Train (AudioCaps only): {len(train_dataset)} samples")

    print(f"  Val (AudioCaps): {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = create_safe_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = create_safe_dataloader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Training config
    config = {
        "num_epochs": args.num_epochs,
        "learning_rate_projector": args.learning_rate_projector,
        "learning_rate_adapter": args.learning_rate_adapter,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "fp16": args.fp16,
        "eval_frequency": args.eval_frequency,
        # Default eval cap if not specified to keep metrics manageable
        "max_eval_batches": args.max_eval_batches if args.max_eval_batches is not None else 50,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "early_stopping_patience": args.early_stopping_patience,
        "audio_contrastive_weight": args.audio_contrastive_weight,
        "audio_contrastive_temperature": args.audio_contrastive_temperature,
        "audio_contrastive_max_length": args.audio_contrastive_max_length,
        "gate_warmup_steps": args.gate_warmup_steps,
    }

    # Optional W&B init (after model + data are available so config is complete)
    wandb_run = _maybe_init_wandb(
        args,
        train_config=config,
        model_config=model_config,
        output_dir=output_dir,
        model=model,
        total_params=total_params,
        trainable_params=trainable_params,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
    )

    # Evaluation only mode
    if args.eval_only:
        if args.resume is None:
            raise ValueError("--eval-only requires --resume")

        load_checkpoint(model, None, None, Path(args.resume), device)

        print(f"\n{'='*80}")
        print(f"Running evaluation only")
        print(f"{'='*80}\n")

        eval_samples: Optional[List[Dict[str, Any]]] = (
            [] if (wandb_run is not None and args.wandb_sample_count > 0) else None
        )
        metrics = evaluate(
            model,
            val_loader,
            device,
            max_batches=args.max_eval_batches,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            compute_bertscore=True,   # Full metrics in eval-only mode
            light_metrics=False,
            suppress_eos_for_audio=False,
            sample_output=eval_samples,
            sample_limit=args.wandb_sample_count,
        )

        # Save results
        results_path = output_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

        if wandb_run is not None:
            eval_log = {"train/optimizer_step": 0, "epoch": 0}
            eval_log.update({f"eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
            _wandb_log(wandb_run, eval_log, step=0)
            if eval_samples and wandb is not None:
                try:
                    table = wandb.Table(columns=["question", "prediction", "references"])
                    for row in eval_samples:
                        table.add_data(row["question"], row["prediction"], "\n".join(row["references"]))
                    _wandb_log(wandb_run, {"train/optimizer_step": 0, "eval/samples": table}, step=0)
                except Exception:
                    pass
            try:
                wandb_run.finish()
            except Exception:
                pass

        print(f"\n💾 Results saved to: {results_path}")
        return

    # Resume from checkpoint if specified
    if args.resume:
        load_checkpoint(model, None, None, Path(args.resume), device)

    # Train
    train(
        model,
        train_loader,
        val_loader,
        config,
        output_dir,
        device,
        wandb_run=wandb_run,
        wandb_log_checkpoints=bool(args.wandb_log_checkpoints),
        wandb_sample_count=int(args.wandb_sample_count),
    )

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    print(f"\n✅ Training complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
