#!/usr/bin/env python3
"""
ScanQA Composition Training Script.

Tests point cloud + image composition for 3D question answering.

Usage:
    # Point cloud only
    python train_scanqa_composition.py --modality pointcloud

    # Image only
    python train_scanqa_composition.py --modality image

    # Both (composition)
    python train_scanqa_composition.py --modality both
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from functools import lru_cache

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: nltk not available. Install for BLEU/METEOR metrics.")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from safe.data.scanqa_dataset import ScanQADataset, collate_scanqa_batch
from configs.pointcloud_configs import get_pointcloud_config


def normalize_answer(text: str) -> str:
    """Normalize LLM-generated answer by stripping preamble, lowercasing, etc."""
    text = (text or "").strip().lower()
    # Strip common LLM preamble patterns
    for prefix in ("the answer is", "answer:", "a:", "it is", "this is"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.strip().rstrip(".")
    return " ".join(text.split())


def token_f1(pred: str, ref: str) -> float:
    """Compute token-level F1 between normalized prediction and reference."""
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


@lru_cache(maxsize=100000)
def _lcs_length(pred_tokens: tuple[str, ...], ref_tokens: tuple[str, ...]) -> int:
    """Longest common subsequence length for token tuples."""
    m = len(pred_tokens)
    n = len(ref_tokens)
    if m == 0 or n == 0:
        return 0

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = 0
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


def rouge_l_score(pred: str, ref: str) -> float:
    """Compute ROUGE-L F1 on whitespace-tokenized, normalized strings."""
    p = tuple(normalize_answer(pred).split())
    r = tuple(normalize_answer(ref).split())
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0

    lcs = _lcs_length(p, r)
    if lcs == 0:
        return 0.0

    precision = lcs / len(p)
    recall = lcs / len(r)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_cider(predictions: List[str], references: List[List[str]]) -> float:
    """Compute CIDEr (0-100 scale) when pycocoevalcap is available."""
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        return 0.0

    gts = {str(i): [str(r) for r in refs if str(r).strip()] for i, refs in enumerate(references)}
    res = {str(i): [str(pred)] for i, pred in enumerate(predictions)}

    try:
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts, res)
        return float(score) * 100.0
    except Exception:
        return 0.0


def _dedupe_texts(values: List[str]) -> List[str]:
    """Deduplicate strings while preserving order."""
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def select_training_answers(
    batch: Dict[str, Any],
    strategy: str = "random",
) -> List[str]:
    """Select one supervision target per sample from the available references."""
    selected: List[str] = []
    primary_answers = batch.get("answers", [])
    all_answers = batch.get("all_answers", [])

    for primary, refs in zip(primary_answers, all_answers):
        candidates = _dedupe_texts(list(refs) if isinstance(refs, list) else [refs])
        if not candidates:
            candidates = _dedupe_texts([primary])
        if not candidates:
            candidates = [""]

        if strategy == "first":
            chosen = candidates[0]
        elif strategy == "shortest":
            chosen = min(candidates, key=lambda text: (len(text.split()), len(text)))
        else:
            chosen = random.choice(candidates)
        selected.append(chosen)

    return selected


def _pad_tokenized_sequences(
    sequences: List[List[int]],
    pad_value: int,
    *,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Pad a jagged list of token ID sequences into a dense tensor."""
    if not sequences:
        return torch.empty(0, 0, dtype=dtype)

    max_len = max(len(seq) for seq in sequences)
    output = torch.full((len(sequences), max_len), pad_value, dtype=dtype)
    for row_idx, seq in enumerate(sequences):
        if seq:
            output[row_idx, :len(seq)] = torch.tensor(seq, dtype=dtype)
    return output


def compute_qa_metrics(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Compute QA evaluation metrics.

    Args:
        predictions: List of predicted answer strings
        references: List of lists of reference answer strings

    Returns:
        Dictionary with BLEU-1, BLEU-4, METEOR, exact_match, norm_em, token_f1
    """
    _zero = {"bleu1": 0.0, "bleu4": 0.0, "meteor": 0.0, "rouge_l": 0.0, "cider": 0.0,
             "exact_match": 0.0, "norm_em": 0.0, "token_f1": 0.0}

    if len(predictions) == 0:
        return _zero

    bleu1_scores = []
    bleu4_scores = []
    meteor_scores = []
    rouge_l_scores = []
    exact_matches = []
    norm_em_scores = []
    token_f1_scores = []

    smoother = SmoothingFunction()

    for pred, refs in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens_list = [ref.lower().split() for ref in refs if ref]

        # Raw exact match (against any reference)
        exact = any(pred.lower().strip() == ref.lower().strip() for ref in refs if ref)
        exact_matches.append(float(exact))

        # Normalized exact match (strips LLM preamble)
        norm_exact = any(normalize_answer(pred) == normalize_answer(ref) for ref in refs if ref)
        norm_em_scores.append(float(norm_exact))

        # Token F1 (max across references)
        best_f1 = max((token_f1(pred, ref) for ref in refs if ref), default=0.0)
        token_f1_scores.append(best_f1)

        # ROUGE-L (best across references)
        best_rouge_l = max((rouge_l_score(pred, ref) for ref in refs if ref), default=0.0)
        rouge_l_scores.append(best_rouge_l)

        # BLEU scores
        if pred_tokens and ref_tokens_list and NLTK_AVAILABLE:
            bleu1 = sentence_bleu(ref_tokens_list, pred_tokens,
                                  weights=(1.0, 0, 0, 0),
                                  smoothing_function=smoother.method1)
            bleu4 = sentence_bleu(ref_tokens_list, pred_tokens,
                                  weights=(0.25, 0.25, 0.25, 0.25),
                                  smoothing_function=smoother.method1)
            bleu1_scores.append(bleu1)
            bleu4_scores.append(bleu4)

            # METEOR (max across references)
            meteor = max(meteor_score([ref.split()], pred_tokens) for ref in refs if ref)
            meteor_scores.append(meteor)
        else:
            bleu1_scores.append(0.0)
            bleu4_scores.append(0.0)
            meteor_scores.append(0.0)

    n = len(bleu1_scores)
    cider = compute_cider(predictions, references)
    return {
        "bleu1": sum(bleu1_scores) / n * 100 if n > 0 else 0.0,
        "bleu4": sum(bleu4_scores) / n * 100 if n > 0 else 0.0,
        "meteor": sum(meteor_scores) / n * 100 if n > 0 else 0.0,
        "rouge_l": sum(rouge_l_scores) / len(rouge_l_scores) * 100 if rouge_l_scores else 0.0,
        "cider": cider,
        "exact_match": sum(exact_matches) / len(exact_matches) * 100 if exact_matches else 0.0,
        "norm_em": sum(norm_em_scores) / len(norm_em_scores) * 100 if norm_em_scores else 0.0,
        "token_f1": sum(token_f1_scores) / len(token_f1_scores) * 100 if token_f1_scores else 0.0,
    }


class ScanQACompositionModel(nn.Module):
    """
    Model for ScanQA with optional modality composition.

    Training approach:
    - Image-only: Use LLaVA as-is (already trained for vision+language)
    - PC-only: Train SAFE adapter (PointBERT -> fusion -> LLM)
    - Both: Use SAFE model with images processed through LLaVA's vision encoder
            PC tokens injected via SAFE fusion, images via LLaVA's native path

    Key insight: SAFE's base_vl IS LLaVA, so "both" mode uses a single model
    that processes images natively AND receives PC tokens via fusion.
    """

    def __init__(
        self,
        modality: str = "both",
        llm_model_name: str = "llava-hf/llava-1.5-7b-hf",
        pointcloud_encoder_checkpoint: Optional[str] = None,
        num_tokens: int = 8,
        fusion_layer_indices: List[int] = [1, 5, 9, 13, 17, 21],
        freeze_llm: bool = True,  # Freeze LLM, only train SAFE adapter
        freeze_encoder: bool = True,
        unfreeze_encoder_last_n: int = 0,
        config: Optional[Dict] = None,
    ):
        super().__init__()
        self.modality = modality
        self.llm_model_name = config["llm_model_name"] if config else llm_model_name

        if modality == "image":
            # Image-only: Use standalone LLaVA (already trained)
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            print("Loading LLaVA model for image-only mode...")
            self.llava = LlavaForConditionalGeneration.from_pretrained(
                self.llm_model_name,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            self.processor = AutoProcessor.from_pretrained(self.llm_model_name)

            # Freeze LLaVA - it's already trained
            for param in self.llava.parameters():
                param.requires_grad = False

            self.llm_hidden_size = self.llava.config.text_config.hidden_size

        else:
            # PC-only or Both: Use SAFE model (which contains LLaVA as base_vl)
            from safe.models.safe_pointcloud_model import SAFEPointCloudModel
            print(f"Loading SAFE point cloud model for {modality} mode...")

            if config is not None:
                safe_config = {
                    "llm_model_name": config["llm_model_name"],
                    "vision_model_name": config.get("vision_model_name", "openai/clip-vit-large-patch14"),
                    "pointcloud_encoder_type": config.get("pointcloud_encoder_type", "pointbert"),
                    "pointcloud_encoder_config": config.get("pointcloud_encoder_config", {}),
                    "num_tokens": config.get("num_tokens", num_tokens),
                    "fusion_type": config.get("fusion_type", "multilayer"),
                    "fusion_layer_indices": config.get("fusion_layer_indices", fusion_layer_indices),
                    "lora_rank": config.get("lora_rank", 8),
                    "fusion_config": config.get("fusion_config", {}),
                    "freeze_base_vl": config.get("freeze_base_vl", True),
                    "freeze_pointcloud_encoder": config.get("freeze_pointcloud_encoder", True),
                    "llm_hidden_size": config.get("llm_hidden_size", 5120),
                    "pointcloud_embed_dim": config.get("pointcloud_embed_dim", 768),
                    "label_smoothing": config.get("label_smoothing", 0.0),
                    "enable_gradient_checkpointing": bool(
                        config.get(
                            "enable_gradient_checkpointing",
                            str(os.environ.get("SAFE_GRAD_CKPT", "0")).strip().lower() in {"1", "true", "yes", "on"},
                        )
                    ),
                }
            else:
                safe_config = {
                    "llm_model_name": self.llm_model_name,
                    "pointcloud_encoder_type": "pointbert",
                    "pointcloud_encoder_config": {
                        "model_name": "pointbert-base",
                        "checkpoint_path": pointcloud_encoder_checkpoint,
                        "unfreeze_last_n_blocks": unfreeze_encoder_last_n,
                    },
                    "num_tokens": num_tokens,
                    "fusion_layer_indices": fusion_layer_indices,
                    "freeze_base_vl": freeze_llm,
                    "freeze_pointcloud_encoder": freeze_encoder,
                    "enable_gradient_checkpointing": str(os.environ.get("SAFE_GRAD_CKPT", "0")).strip().lower() in {
                        "1", "true", "yes", "on"
                    },
                }

            self.safe_model = SAFEPointCloudModel(**safe_config)
            self.safe_model.enable_pointcloud_training()

            # For "both" mode, we'll use SAFE's processor for images too
            base_vl = self.safe_model.base_vl
            if hasattr(base_vl, "processor"):
                self.processor = base_vl.processor
            elif hasattr(base_vl, "internvl_image_processor") and base_vl.internvl_image_processor is not None:
                self.processor = base_vl.internvl_image_processor
            else:
                self.processor = None

            # InternVLChatConfig nests hidden_size under llm_config / text_config
            _cfg = self.safe_model.base_vl.llm.config
            self.llm_hidden_size = getattr(
                _cfg, "hidden_size",
                getattr(
                    getattr(_cfg, "llm_config", getattr(_cfg, "text_config", None)),
                    "hidden_size", 4096
                ),
            )

    def _process_images(self, images: List, device) -> torch.Tensor:
        """Process PIL images to pixel_values tensor."""
        debug_image_processor = not hasattr(self, "_image_processor_debug_logged")
        if debug_image_processor:
            self._image_processor_debug_logged = True
            try:
                first_sizes = [getattr(img, "size", None) for img in images[:4]]
                print(f"[ImageDebug] Input image sizes: {first_sizes}", flush=True)
            except Exception:
                pass
        processed = self.processor(images=images, return_tensors="pt")
        # Match model dtype (InternVL uses bfloat16, not float16)
        model_dtype = torch.bfloat16
        if hasattr(self, "safe_model"):
            model_dtype = next(self.safe_model.base_vl.llm.parameters()).dtype
        elif hasattr(self, "llava"):
            model_dtype = next(self.llava.parameters()).dtype
        pixel_values = processed["pixel_values"].to(device=device, dtype=model_dtype)
        if debug_image_processor:
            print(f"[ImageDebug] pixel_values shape={tuple(pixel_values.shape)} dtype={pixel_values.dtype}", flush=True)
        return pixel_values

    @staticmethod
    def _extract_loss_logits(outputs):
        """Extract loss/logits from either a dict or HF model output."""
        if isinstance(outputs, dict):
            return {
                "loss": outputs.get("loss"),
                "logits": outputs.get("logits"),
            }
        return {
            "loss": getattr(outputs, "loss", None),
            "logits": getattr(outputs, "logits", None),
        }

    def forward(
        self,
        pointclouds: Optional[torch.Tensor] = None,
        images: Optional[List] = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        gate: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training (with labels) or inference."""

        if self.modality == "image":
            # Image-only: Use LLaVA directly (zero-shot / frozen)
            if pixel_values is None and images is not None:
                pixel_values = self._process_images(images, input_ids.device)

            outputs = self.llava(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
            return {"loss": outputs.loss, "logits": outputs.logits}

        elif self.modality == "pointcloud":
            # PC-only: Use SAFE with point cloud fusion (no images)
            outputs = self.safe_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
                labels=labels,
                gate=gate,
            )
            return self._extract_loss_logits(outputs)

        else:  # "both" - TRUE COMPOSITION
            # Process images if needed
            if pixel_values is None and images is not None:
                pixel_values = self._process_images(images, input_ids.device)

            outputs = self.safe_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
                pixel_values=pixel_values,
                labels=labels,
                gate=gate,
            )
            return self._extract_loss_logits(outputs)

    @torch.no_grad()
    def generate(
        self,
        pointclouds: Optional[torch.Tensor] = None,
        images: Optional[List] = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate answers."""

        if self.modality == "image":
            if pixel_values is None and images is not None:
                pixel_values = self._process_images(images, input_ids.device)

            outputs = self.llava.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )
            return outputs

        elif self.modality == "pointcloud":
            outputs = self.safe_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )
            return outputs

        else:  # "both" - TRUE COMPOSITION
            if pixel_values is None and images is not None:
                pixel_values = self._process_images(images, input_ids.device)

            # Generate with both image and PC
            outputs = self.safe_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )
            return outputs

    def get_trainable_params(self):
        """Get trainable parameters (only SAFE adapter components, LLM is frozen)."""
        return [p for p in self.parameters() if p.requires_grad]

    def supported_eval_modalities(self) -> List[str]:
        """Return the eval modality ablations this model can run."""
        if hasattr(self, "safe_model"):
            return ["text", "image", "pointcloud", "both"]
        return ["text", "image"]

    def eval_prompt_prefix_length(self, eval_modality: str) -> int:
        """Extra prompt tokens prepended internally before generation."""
        eval_modality = str(eval_modality).lower()
        if eval_modality in {"image", "both"} and hasattr(self, "safe_model"):
            return self.safe_model.get_image_prompt_prefix_length()
        return 0

    @torch.no_grad()
    def generate_for_eval(
        self,
        eval_modality: str,
        pointclouds: Optional[torch.Tensor] = None,
        images: Optional[List] = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate under an explicit evaluation condition."""
        eval_modality = str(eval_modality).lower()

        if pixel_values is None and images is not None and eval_modality in {"image", "both"}:
            pixel_values = self._process_images(images, input_ids.device)

        if eval_modality == "text":
            if hasattr(self, "safe_model"):
                return self.safe_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )
            return self.llava.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )

        if eval_modality == "image":
            if hasattr(self, "safe_model"):
                return self.safe_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )
            return self.llava.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )

        if eval_modality == "pointcloud":
            if not hasattr(self, "safe_model"):
                raise ValueError("pointcloud eval requested, but model has no SAFE pointcloud path")
            return self.safe_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )

        if eval_modality == "both":
            if not hasattr(self, "safe_model"):
                raise ValueError("both-modality eval requested, but model has no SAFE pointcloud path")
            return self.safe_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pointcloud=pointclouds,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )

        raise ValueError(f"Unsupported eval_modality: {eval_modality}")


def parse_args():
    parser = argparse.ArgumentParser(description="ScanQA Composition Training")

    # Data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--modality", type=str, default="both", choices=["pointcloud", "image", "both"])
    parser.add_argument("--num-points", type=int, default=8192)

    # Model
    parser.add_argument("--model-config", type=str, default=None,
                        help="Config name from pointcloud_configs.py (e.g. scanqa_internvl)")
    parser.add_argument("--llm-model", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--fusion-layer-indices", type=str, default=None)
    parser.add_argument("--num-pointcloud-tokens", type=int, default=None)
    parser.add_argument("--encoder-checkpoint", type=str, default=None)
    parser.add_argument("--unfreeze-encoder-last-n", type=int, default=0)
    parser.add_argument("--freeze-llm", action="store_true", help="Freeze LLM (use linear probe)")

    # Training
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--safe-lr", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument(
        "--train-answer-mode",
        type=str,
        default="random",
        choices=["random", "first", "shortest"],
        help="How to select one supervision target from ScanQA's multiple references",
    )

    # Generation
    parser.add_argument("--max-answer-tokens", type=int, default=32)
    parser.add_argument("--eval-num-beams", type=int, default=1)
    parser.add_argument("--eval-repetition-penalty", type=float, default=1.1)
    parser.add_argument("--eval-no-repeat-ngram-size", type=int, default=3)

    # Gate warmup (matches AVQA setup)
    parser.add_argument("--gate-warmup-epochs", type=int, default=2,
                        help="Gradually ramp fusion gate from 0→1 over N epochs")

    # Eval
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Cap both train and val datasets to N samples (0=full datasets)")
    parser.add_argument("--train-max-samples", type=int, default=0,
                        help="Cap only the train dataset to N samples (0=full train set)")
    parser.add_argument("--val-max-samples", type=int, default=0,
                        help="Cap only the val dataset to N samples (0=full val set)")
    parser.add_argument(
        "--eval-modalities",
        type=str,
        default="auto",
        help="Comma-separated eval ablations to run each epoch: auto,text,image,pointcloud,both",
    )

    # Logging
    parser.add_argument("--log-every", type=int, default=50,
                        help="Print loss every N steps")

    # Smoke test
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick 10-step train + eval to verify everything works")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ScanQA-Composition")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)

    return parser.parse_args()


def prepare_qa_inputs(batch, tokenizer, device, max_length=256):
    """Prepare inputs for QA training/inference."""
    questions = batch["questions"]
    answers = batch["answers"]

    # Format: "Question: {q}\nAnswer: {a}"
    prompts = [f"Question: {q}\nAnswer:" for q in questions]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    prompt_token_lists: List[List[int]] = []
    full_token_lists: List[List[int]] = []
    label_token_lists: List[List[int]] = []

    for prompt, answer in zip(prompts, answers):
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_text = str(answer).strip()
        answer_prefix = f" {answer_text}" if answer_text else ""
        answer_ids = tokenizer(answer_prefix, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

        prompt_ids = list(prompt_ids)
        answer_ids = list(answer_ids)

        if len(prompt_ids) >= max_length:
            prompt_ids = prompt_ids[:max_length]
            answer_ids = []
        else:
            remaining = max_length - len(prompt_ids)
            answer_ids = answer_ids[:remaining]

        input_ids = prompt_ids + answer_ids
        labels = ([-100] * len(prompt_ids)) + answer_ids

        prompt_token_lists.append(prompt_ids[:max_length])
        full_token_lists.append(input_ids)
        label_token_lists.append(labels)

    input_ids = _pad_tokenized_sequences(full_token_lists, pad_token_id)
    attention_mask = _pad_tokenized_sequences(
        [[1] * len(seq) for seq in full_token_lists],
        0,
    )
    labels = _pad_tokenized_sequences(label_token_lists, -100)
    prompt_input_ids = _pad_tokenized_sequences(prompt_token_lists, pad_token_id)
    prompt_attention_mask = _pad_tokenized_sequences(
        [[1] * len(seq) for seq in prompt_token_lists],
        0,
    )

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
        "prompt_input_ids": prompt_input_ids.to(device),
        "prompt_attention_mask": prompt_attention_mask.to(device),
    }


def compute_answer_ce_loss(
    logits: Optional[torch.Tensor],
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> Optional[torch.Tensor]:
    """Compute causal LM loss over answer tokens only."""
    if logits is None:
        return None

    if logits.dim() < 3 or labels.dim() < 2:
        return None

    seq_diff = int(logits.size(1) - labels.size(1))
    if seq_diff > 0:
        pad = torch.full(
            (labels.size(0), seq_diff),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        labels = torch.cat([pad, labels], dim=1)
    elif seq_diff < 0:
        labels = labels[:, -logits.size(1):]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = (shift_labels != -100).any()
    if not bool(valid.item() if torch.is_tensor(valid) else valid):
        return None

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        label_smoothing=float(max(label_smoothing, 0.0)),
    )


def _compute_gate_value(epoch, batch_idx, total_batches, gate_warmup_epochs=2):
    """Gradually ramp fusion gate from 0→1 over warmup epochs (matches AVQA gate warmup)."""
    total_warmup_steps = gate_warmup_epochs * total_batches
    global_step = epoch * total_batches + batch_idx
    if global_step >= total_warmup_steps:
        return 1.0
    return global_step / max(1, total_warmup_steps)


def train_epoch(model, dataloader, optimizer, scheduler, device, args, epoch, tokenizer):
    """Train one epoch."""
    model.train()

    total_loss = 0.0
    interval_loss = 0.0
    num_batches = 0
    interval_batches = 0
    log_every = getattr(args, "log_every", 50)
    max_steps = getattr(args, "_smoke_max_steps", None)
    # InternVL runs in bf16 natively — AMP autocast causes dtype conflicts
    # between vit_embeds (float32 under autocast) and input_embeds (bf16).
    # Disable AMP; the model handles its own mixed precision.
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gate_warmup_epochs = getattr(args, "gate_warmup_epochs", 2)
    label_smoothing = float(getattr(args, "label_smoothing", 0.0) or 0.0)
    train_answer_mode = getattr(args, "train_answer_mode", "random")

    total_steps = len(dataloader)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    accum_steps = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        if max_steps is not None and batch_idx >= max_steps:
            print(f"[smoke-test] Stopping training after {max_steps} steps", flush=True)
            break

        if not batch:
            continue

        # Gate warmup: gradually enable fusion (matches AVQA)
        gate = _compute_gate_value(epoch, batch_idx, total_steps, gate_warmup_epochs)

        # Prepare inputs
        train_answers = select_training_answers(batch, strategy=train_answer_mode)
        train_batch = dict(batch)
        train_batch["answers"] = train_answers
        qa_inputs = prepare_qa_inputs(train_batch, tokenizer, device)

        kwargs = {
            "input_ids": qa_inputs["input_ids"],
            "attention_mask": qa_inputs["attention_mask"],
            "labels": qa_inputs["labels"],
        }

        if args.modality in ["pointcloud", "both"] and batch.get("pointclouds") is not None:
            kwargs["pointclouds"] = batch["pointclouds"].to(device)
        if args.modality in ["image", "both"] and batch.get("images") is not None:
            kwargs["images"] = batch["images"]
        kwargs["gate"] = gate

        # Forward with AMP autocast (matches AVQA)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(**kwargs)
            loss = outputs["loss"]
            logits = outputs.get("logits")

        manual_loss = compute_answer_ce_loss(
            logits,
            kwargs["labels"],
            label_smoothing=label_smoothing,
        )
        if manual_loss is not None:
            loss = manual_loss

        if loss is None:
            continue

        loss = loss / args.gradient_accumulation_steps
        scaler.scale(loss).backward()
        accum_steps += 1

        if accum_steps % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        step_loss = loss.item() * args.gradient_accumulation_steps
        total_loss += step_loss
        interval_loss += step_loss
        num_batches += 1
        interval_batches += 1

        avg_loss = total_loss / num_batches
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "gate": f"{gate:.2f}"})

        # Detailed interval logging
        if num_batches % log_every == 0:
            cur_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.safe_lr
            int_avg = interval_loss / interval_batches
            print(
                f"\n[train] epoch={epoch+1} step={batch_idx+1}/{total_steps} "
                f"loss={avg_loss:.4f} interval_loss={int_avg:.4f} "
                f"lr={cur_lr:.2e} gate={gate:.3f}",
                flush=True,
            )
            interval_loss = 0.0
            interval_batches = 0

    if accum_steps % args.gradient_accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    avg = total_loss / max(num_batches, 1)
    print(f"\n[train] Epoch {epoch+1} complete — avg_loss={avg:.4f} steps={num_batches}", flush=True)
    return {"loss": avg}


@torch.no_grad()
def evaluate(model, dataloader, device, args, tokenizer, max_samples=None, eval_modality: Optional[str] = None):
    """Evaluate model with generation."""
    model.eval()
    modality_key = str(eval_modality or args.modality).lower()

    all_predictions = []
    all_references = []

    num_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        if not batch:
            continue
        if max_samples and num_samples >= max_samples:
            break

        # Prepare inputs for generation
        questions = batch["questions"]
        prompts = [f"Question: {q}\nAnswer:" for q in questions]

        prompt_encodings = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        kwargs = {
            "input_ids": prompt_encodings["input_ids"].to(device),
            "attention_mask": prompt_encodings["attention_mask"].to(device),
            "max_new_tokens": args.max_answer_tokens,
            "do_sample": False,
            "num_beams": args.eval_num_beams,
            "repetition_penalty": args.eval_repetition_penalty,
            "no_repeat_ngram_size": args.eval_no_repeat_ngram_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if modality_key in ["pointcloud", "both"] and batch.get("pointclouds") is not None:
            kwargs["pointclouds"] = batch["pointclouds"].to(device)
        if modality_key in ["image", "both"] and batch.get("images") is not None:
            kwargs["images"] = batch["images"]

        # Generate
        output_ids = model.generate_for_eval(eval_modality=modality_key, **kwargs)
        extra_prefix_len = int(model.eval_prompt_prefix_length(modality_key))
        prompt_width = int(prompt_encodings["input_ids"].size(1))

        # Decode predictions
        for i, ids in enumerate(output_ids):
            # Some generation paths return full sequence (prompt + generation),
            # while others return only the newly generated tokens.
            full_prompt_width = prompt_width + extra_prefix_len
            if ids.size(0) > full_prompt_width:
                generated_ids = ids[full_prompt_width:]
            else:
                generated_ids = ids
            pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            all_predictions.append(pred)
            all_references.append(batch["all_answers"][i])

        num_samples += len(batch["questions"])

    # Compute metrics
    metrics = compute_qa_metrics(all_predictions, all_references)

    return metrics, all_predictions, all_references


def resolve_eval_modalities(args, model: ScanQACompositionModel) -> List[str]:
    """Resolve requested eval ablations against model capabilities."""
    supported = model.supported_eval_modalities()
    requested = str(getattr(args, "eval_modalities", "auto") or "auto").strip().lower()
    if requested == "auto":
        return supported

    selected = []
    for item in requested.split(","):
        key = item.strip().lower()
        if key and key in supported and key not in selected:
            selected.append(key)
    return selected or supported


def maybe_limit_dataset(dataset, max_samples: int):
    """Optionally truncate a dataset for faster debugging runs."""
    max_samples = int(max_samples or 0)
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, list(range(max_samples)))


def main():
    args = parse_args()

    # Smoke test overrides
    if args.smoke_test:
        print("=" * 60)
        print("SMOKE TEST MODE — quick validation run")
        print("=" * 60)
        args.num_epochs = 1
        args.max_eval_samples = 10
        args._smoke_max_steps = 10
        args.log_every = 2
        args.wandb = False
    else:
        args._smoke_max_steps = None

    if not getattr(args, "train_max_samples", 0):
        args.train_max_samples = int(getattr(args, "max_samples", 0) or 0)
    if not getattr(args, "val_max_samples", 0):
        args.val_max_samples = int(getattr(args, "max_samples", 0) or 0)

    config = get_pointcloud_config(args.model_config) if args.model_config else None

    if args.batch_size is None:
        args.batch_size = int(config.get("recommended_batch_size", 4)) if config else 4
    if args.safe_lr is None:
        args.safe_lr = float(config.get("safe_lr", 1e-5)) if config else 1e-5
    if args.label_smoothing is None:
        args.label_smoothing = float(config.get("label_smoothing", 0.0)) if config else 0.0
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = (
            int(config.get("gradient_accumulation_steps", 4)) if config else 4
        )
    if args.num_pointcloud_tokens is None:
        args.num_pointcloud_tokens = int(config.get("num_tokens", 8)) if config else 8

    if args.fusion_layer_indices is None:
        default_layers = config.get("fusion_layer_indices") if config else None
        if not default_layers:
            default_layers = [1, 5, 9, 13, 17, 21]
        fusion_layers = [int(x) for x in default_layers]
        args.fusion_layer_indices = ",".join(str(x) for x in fusion_layers)
    else:
        fusion_layers = [int(x.strip()) for x in args.fusion_layer_indices.split(",") if x.strip()]

    print("=" * 60)
    print("ScanQA Composition Training")
    print("=" * 60)
    if config:
        print(f"Model config:     {config['name']}")
    print(f"Modality:        {args.modality}")
    print(f"Data path:       {args.data_path}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Epochs:          {args.num_epochs}")
    print(f"LR:              {args.safe_lr}")
    print(f"Label smooth:    {args.label_smoothing}")
    print(f"Grad accum:      {args.gradient_accumulation_steps}")
    print(f"Effective batch: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"PC tokens:       {args.num_pointcloud_tokens}")
    print(f"Fusion layers:   {fusion_layers}")
    print(f"Train answers:   {args.train_answer_mode}")
    print(f"Log every:       {args.log_every}")
    print(f"Eval every:      {args.eval_every} epoch(s)")
    print(f"Max eval samp:   {args.max_eval_samples}")
    print(f"Train subset:    {args.train_max_samples or 'full'}")
    print(f"Val subset:      {args.val_max_samples or 'full'}")
    print(f"Gate warmup:     {args.gate_warmup_epochs} epoch(s)")
    print(f"Optimizer:       AdamW (grouped, betas=0.9/0.95)")
    print(f"AMP:             disabled")
    print(f"Smoke test:      {args.smoke_test}")
    print("=" * 60)

    # Create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ScanQADataset(
        args.data_path,
        split="train",
        modality=args.modality,
        num_points=args.num_points,
    )
    train_dataset = maybe_limit_dataset(train_dataset, args.train_max_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_scanqa_batch,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")

    if config:
        print(f"Using config: {config['name']} — {config.get('description', '')}")

    # Create model
    print("\nCreating model...")
    model = ScanQACompositionModel(
        modality=args.modality,
        llm_model_name=args.llm_model,
        pointcloud_encoder_checkpoint=args.encoder_checkpoint,
        num_tokens=args.num_pointcloud_tokens,
        fusion_layer_indices=fusion_layers,
        freeze_llm=args.freeze_llm,
        unfreeze_encoder_last_n=args.unfreeze_encoder_last_n,
        config=config,
    )
    model = model.to(args.device)

    if args.fp16:
        model = model.half()

    eval_modalities = resolve_eval_modalities(args, model)
    eval_dataset_modality = "both" if "both" in eval_modalities or "pointcloud" in eval_modalities else "image"
    val_dataset = ScanQADataset(
        args.data_path,
        split="val",
        modality=eval_dataset_modality,
        num_points=args.num_points,
        augment=False,
    )
    val_dataset = maybe_limit_dataset(val_dataset, args.val_max_samples)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_scanqa_batch,
        pin_memory=True,
    )
    print(f"Val samples: {len(val_dataset)}")
    print(f"Eval modalities: {eval_modalities}")
    print(f"Eval subset:     {eval_dataset_modality}")

    # Get tokenizer
    if args.modality in ["pointcloud", "both"]:
        tokenizer = model.safe_model.base_vl.tokenizer
    else:
        tokenizer = model.processor.tokenizer

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Optimizer — parameter-grouped (match AVQA setup)
    trainable_params = model.get_trainable_params()
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in trainable_params)
    print(f"\n[params] Total: {total_params:,} | Trainable: {train_params:,} ({100*train_params/total_params:.2f}%)")
    if hasattr(model, "safe_model"):
        for name, mod in [
            ("projector", getattr(model.safe_model, "pointcloud_projector", None)),
            ("fusion_adapter", getattr(model.safe_model, "fusion_adapter", None)),
            ("pc_encoder", getattr(model.safe_model, "pointcloud_encoder", None)),
        ]:
            if mod is not None:
                t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
                a = sum(p.numel() for p in mod.parameters())
                print(f"  {name}: {t:,} trainable / {a:,} total")
    print()

    # Separate bias/norm (no weight_decay) from other params (matches AVQA)
    decay_params = []
    no_decay_params = []
    for p in trainable_params:
        if p.dim() <= 1:  # bias, norm, embedding
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.safe_lr,
        betas=(0.9, 0.95),
    )

    # Scheduler
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        if args.lr_scheduler == "constant":
            return 1.0
        else:  # cosine
            progress = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # W&B
    if args.wandb and wandb:
        tags = args.wandb_tags.split(",") if args.wandb_tags else []
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=tags,
            config=vars(args),
        )

    # Training loop
    print("\nStarting training...")
    best_bleu4 = 0.0

    for epoch in range(args.num_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            args.device, args, epoch, tokenizer
        )

        print(f"\nEpoch {epoch+1} - Train Loss: {train_metrics['loss']:.4f}")

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            eval_results: Dict[str, Dict[str, float]] = {}
            sample_predictions: List[str] = []
            sample_references: List[List[str]] = []
            sample_modality = "both" if "both" in eval_modalities else eval_modalities[0]

            for eval_key in eval_modalities:
                val_metrics, predictions, references = evaluate(
                    model, val_loader, args.device, args,
                    tokenizer, args.max_eval_samples, eval_modality=eval_key
                )
                eval_results[eval_key] = val_metrics
                if eval_key == sample_modality:
                    sample_predictions = predictions
                    sample_references = references

                print(f"Val Metrics [{eval_key}]:")
                print(f"  BLEU-1: {val_metrics['bleu1']:.2f}")
                print(f"  BLEU-4: {val_metrics['bleu4']:.2f}")
                print(f"  METEOR: {val_metrics['meteor']:.2f}")
                print(f"  ROUGE-L: {val_metrics['rouge_l']:.2f}")
                print(f"  CIDEr:   {val_metrics['cider']:.2f}")
                print(f"  Exact Match: {val_metrics['exact_match']:.2f}")
                print(f"  Norm EM:     {val_metrics['norm_em']:.2f}")
                print(f"  Token F1:    {val_metrics['token_f1']:.2f}")

            # Show some examples from the composition condition when available.
            print(f"\nSample predictions [{sample_modality}]:")
            for i in range(min(3, len(sample_predictions))):
                print(f"  Pred: {sample_predictions[i]}")
                print(f"  Refs: {sample_references[i][:2]}")
                print()

            if "both" in eval_results:
                score_modality = "both"
            elif args.modality in eval_results:
                score_modality = args.modality
            else:
                score_modality = next(iter(eval_results))
            score_metrics = eval_results[score_modality]

            # Save best using the composition condition when available.
            if score_metrics["bleu4"] > best_bleu4:
                best_bleu4 = score_metrics["bleu4"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_bleu4": best_bleu4,
                    "score_modality": score_modality,
                    "eval_results": eval_results,
                    "args": vars(args),
                }, Path(args.output_dir) / "best_model.pt")
                print(f"New best! Saved checkpoint ({score_modality} BLEU-4: {best_bleu4:.2f})")

            if args.wandb and wandb:
                log_payload = {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "best_bleu4": best_bleu4,
                    "lr": scheduler.get_last_lr()[0],
                }
                for eval_key, val_metrics in eval_results.items():
                    prefix = f"val/{eval_key}"
                    log_payload[f"{prefix}/bleu1"] = val_metrics["bleu1"]
                    log_payload[f"{prefix}/bleu4"] = val_metrics["bleu4"]
                    log_payload[f"{prefix}/meteor"] = val_metrics["meteor"]
                    log_payload[f"{prefix}/rouge_l"] = val_metrics["rouge_l"]
                    log_payload[f"{prefix}/cider"] = val_metrics["cider"]
                    log_payload[f"{prefix}/exact_match"] = val_metrics["exact_match"]
                    log_payload[f"{prefix}/norm_em"] = val_metrics["norm_em"]
                    log_payload[f"{prefix}/token_f1"] = val_metrics["token_f1"]
                if "both" in eval_results:
                    strongest_single_bleu4 = max(
                        eval_results.get("image", {}).get("bleu4", float("-inf")),
                        eval_results.get("pointcloud", {}).get("bleu4", float("-inf")),
                        eval_results.get("text", {}).get("bleu4", float("-inf")),
                    )
                    if strongest_single_bleu4 != float("-inf"):
                        log_payload["val/composition_gain_bleu4"] = (
                            eval_results["both"]["bleu4"] - strongest_single_bleu4
                        )
                wandb.log(log_payload)

    print(f"\nTraining complete! Best BLEU-4: {best_bleu4:.2f}")

    if args.wandb and wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
