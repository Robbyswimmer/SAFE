#!/usr/bin/env python3
"""
LoRA baseline for incremental modality comparison (Table 1).

Demonstrates the degradation problem when adding modalities via weight
modification (LoRA on LLM self-attention) vs MICA's frozen-backbone
cross-attention approach.

Protocol:
  Stage 0: Evaluate text-only Qwen3-8B baseline
  Stage 1: Train audio LoRA + audio projector -> evaluate text, audio
  Stage 2: Merge audio LoRA -> Train vision LoRA + vision projector
           -> evaluate text, audio, vision, both
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LORA_BASELINE_DIR = Path(__file__).resolve().parent
if str(LORA_BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(LORA_BASELINE_DIR))

from configs.model_configs import get_config
from _audio_embedding_proxy import AudioTokenEmbeddingProxy
from safe.models.audio_encoders import CLAPAudioEncoder
from safe.models.base_vl import BaseVLModel
from safe.models.projectors import AudioProjector, TokenSetProjector

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError:
    raise ImportError("peft is required for LoRA baseline. Install with: pip install peft")

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        CLIPVisionModel,
        CLIPImageProcessor,
    )
except ImportError:
    raise ImportError("transformers is required. Install with: pip install transformers")

try:
    import wandb as _wandb
except Exception:
    _wandb = None

# ---------------------------------------------------------------------------
# Reuse answer utilities from the composition experiment
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO_ROOT / "experiments" / "avqa_composition"))
from train_avqa_composition import (
    ManifestAVQADataset,
    collate_avqa,
    normalize_answer,
    extract_answer,
    token_f1,
    categorical_f1,
    AVQA_ANSWER_VOCAB,
    set_seed,
)


# ===================================================================
# LoRA Baseline Model
# ===================================================================

class LoRABaselineModel(nn.Module):
    """
    LoRA baseline: prepend modality tokens + LoRA on Qwen3-8B self-attention.

    Architecture:
        Audio:  waveform -> CLAP(frozen) -> AudioProjector -> 8 tokens @ 4096-d
        Vision: image -> CLIP(frozen) -> VisionProjector -> 8 tokens @ 4096-d
        Text:   question -> tokenizer -> embeddings @ 4096-d
                => [modality_tokens | text_tokens] -> Qwen3-8B + LoRA -> answer
    """

    def __init__(self, cfg: Dict[str, Any], stage: int = 1, apply_lora: bool = True):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.apply_lora = apply_lora
        self.llm_hidden_size = cfg["llm_hidden_size"]
        self.use_native_vision = "internvl" in str(cfg["llm_model_name"]).lower()
        self.base_vl: Optional[BaseVLModel] = None
        self.audio_context_token = "<AUDIO_CONTEXT>"
        self.audio_context_token_id: Optional[int] = None
        self._pending_audio_token_embeddings: Optional[torch.Tensor] = None

        # --- Load LLM + tokenizer ---
        llm_path = cfg["llm_model_name"]
        print(f"[LoRA] Loading LLM: {llm_path}", flush=True)
        if self.use_native_vision:
            self.base_vl = BaseVLModel(
                llm_model_name=llm_path,
                vision_model_name="built-in",
                llm_hidden_size=self.llm_hidden_size,
                freeze_vision=True,
                freeze_llm=False,
                prefer_flash_attention_2=False,
            )
            self.llm = self.base_vl.llm
            self.tokenizer = self.base_vl.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_path, trust_remote_code=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Match Qwen composition scripts (causal generation with batched prompts).
            self.tokenizer.padding_side = "left"

            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.use_native_vision:
            self._init_native_audio_placeholders()

        # Freeze base LLM before applying LoRA
        for p in self.llm.parameters():
            p.requires_grad = False

        # --- Apply PEFT LoRA ---
        if self.apply_lora:
            lora_cfg = LoraConfig(
                r=cfg.get("lora_rank", 8),
                lora_alpha=cfg.get("lora_alpha", 16),
                target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
                lora_dropout=cfg.get("lora_dropout", 0.05),
                bias="none",
                task_type="CAUSAL_LM",
            )
            print(f"[LoRA] Applying LoRA: r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, "
                  f"targets={lora_cfg.target_modules}", flush=True)
            self.llm = get_peft_model(self.llm, lora_cfg)
            self.llm.print_trainable_parameters()
        else:
            print("[LoRA] Stage 0 baseline: no LoRA adapters attached", flush=True)

        # --- Audio encoder + projector (always created, loaded in stage 2) ---
        audio_cfg = cfg.get("audio_encoder_config", {})
        print("[LoRA] Loading CLAP audio encoder...", flush=True)
        self.audio_encoder = CLAPAudioEncoder(
            model_name=audio_cfg.get("model_name", "laion/larger_clap_music_and_speech"),
            freeze=cfg.get("freeze_audio_encoder", True),
            sample_rate=audio_cfg.get("sample_rate", 48000),
            max_length=audio_cfg.get("max_length", 10.0),
        )
        self.audio_projector = AudioProjector(
            audio_embed_dim=cfg.get("audio_embed_dim", 512),
            llm_hidden_size=self.llm_hidden_size,
            num_audio_tokens=cfg.get("num_audio_tokens", 8),
            dropout=0.1,
            bottleneck_dim=1024,
            use_swiglu=True,
            use_positional_embedding=True,
        )

        # --- Vision encoder + projector (only used in stage 2) ---
        self.vision_encoder = None
        self.vision_processor = None
        self.vision_projector = None
        if stage >= 2 and not self.use_native_vision:
            self._init_vision(cfg)

        self.label_smoothing = cfg.get("label_smoothing", 0.1)

    def _init_native_audio_placeholders(self) -> None:
        added = self.tokenizer.add_tokens([self.audio_context_token])
        if added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))
        token_id = self.tokenizer.convert_tokens_to_ids(self.audio_context_token)
        if token_id == self.tokenizer.unk_token_id:
            raise RuntimeError("Failed to register native audio placeholder token.")
        self.audio_context_token_id = int(token_id)

        host = getattr(self.llm, "language_model", self.llm)
        base_embedding = host.get_input_embeddings()
        proxy = AudioTokenEmbeddingProxy(base_embedding, self)
        host.set_input_embeddings(proxy)

    def _apply_pending_audio_embeddings(
        self,
        input_ids: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        pending = self._pending_audio_token_embeddings
        token_id = self.audio_context_token_id
        if pending is None or token_id is None:
            return embeddings

        output = embeddings.clone()
        batch_size = min(output.size(0), pending.size(0))
        for batch_idx in range(batch_size):
            positions = (input_ids[batch_idx] == token_id).nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            replace_count = min(int(positions.numel()), int(pending.size(1)))
            output[batch_idx, positions[:replace_count], :] = pending[batch_idx, :replace_count, :].to(
                device=output.device,
                dtype=output.dtype,
            )
        return output

    def _init_vision(self, cfg: Dict[str, Any]) -> None:
        vision_name = cfg.get("vision_model_name", "openai/clip-vit-large-patch14")
        print(f"[LoRA] Loading CLIP vision encoder: {vision_name}", flush=True)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_name)
        self.vision_processor = CLIPImageProcessor.from_pretrained(vision_name)
        if cfg.get("freeze_vision_encoder", True):
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            self.vision_encoder.eval()

        self.vision_projector = TokenSetProjector(
            input_dim=cfg.get("vision_embed_dim", 1024),
            num_tokens=cfg.get("num_vision_tokens", 8),
            output_dim=self.llm_hidden_size,
            dropout=0.1,
            bottleneck_dim=1024,
            use_positional_embedding=True,
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_audio(self, audio_list: List[Any]) -> Optional[torch.Tensor]:
        """CLAP -> projector -> (B, num_audio_tokens, llm_hidden_size)."""
        if audio_list is None:
            return None
        valid = [a for a in audio_list if a is not None]
        if not valid:
            return None

        device = next(self.audio_projector.parameters()).device
        with torch.no_grad():
            audio_emb = self.audio_encoder(valid)  # (B, 512)
        audio_emb = audio_emb.to(device)
        out_dtype = next(self.llm.parameters()).dtype
        tokens = self.audio_projector(audio_emb, out_dtype=out_dtype)  # (B, T, D)
        return tokens

    def encode_vision(self, image_list: List[Any]) -> Optional[torch.Tensor]:
        """CLIP -> projector -> (B, num_vision_tokens, llm_hidden_size)."""
        if self.use_native_vision:
            return None
        if self.vision_encoder is None or self.vision_projector is None:
            return None
        if image_list is None:
            return None
        valid = [img for img in image_list if img is not None]
        if not valid:
            return None

        device = next(self.vision_projector.parameters()).device
        pixel_values = self.vision_processor(
            images=valid, return_tensors="pt",
        )["pixel_values"].to(device, dtype=self.vision_encoder.dtype)

        with torch.no_grad():
            vision_out = self.vision_encoder(pixel_values=pixel_values)
        # Use last_hidden_state (all patch tokens)
        patch_tokens = vision_out.last_hidden_state  # (B, G, 1024)
        patch_tokens = patch_tokens.to(device)
        out_dtype = next(self.llm.parameters()).dtype
        tokens = self.vision_projector(patch_tokens, out_dtype=out_dtype)  # (B, T, D)
        return tokens

    def make_null_audio_tokens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return zero audio tokens to control for extra token slots without evidence."""
        out_dtype = next(self.llm.parameters()).dtype
        return torch.zeros(
            batch_size,
            int(self.cfg.get("num_audio_tokens", 8)),
            self.llm_hidden_size,
            device=device,
            dtype=out_dtype,
        )

    def make_null_vision_tokens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return zero vision tokens to control for extra token slots without evidence."""
        out_dtype = next(self.llm.parameters()).dtype
        return torch.zeros(
            batch_size,
            int(self.cfg.get("num_vision_tokens", 8)),
            self.llm_hidden_size,
            device=device,
            dtype=out_dtype,
        )

    # ------------------------------------------------------------------
    # Prompt formatting (Qwen3 chat template, thinking disabled)
    # ------------------------------------------------------------------

    def format_prompt(self, question: str) -> str:
        if self.use_native_vision:
            return (
                "Answer with exactly one short answer token (single word or number).\n"
                f"Question: {question}\n"
                "Answer:"
            )
        return (
            "<|im_start|>user\n"
            "Answer with exactly one short answer token (single word or number).\n"
            f"Question: {question}\n"
            "Answer:<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _tokenize_batch(
        self,
        questions: List[str],
        answers: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize prompts (+ optional answers for training)."""
        if answers is not None:
            texts = [self.format_prompt(q) + a + "<|im_end|>" for q, a in zip(questions, answers)]
        else:
            texts = [self.format_prompt(q) for q in questions]

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc

    def _prepare_native_vision_inputs(
        self,
        questions: List[str],
        answers: Optional[List[str]] = None,
        image_list: Optional[List[Any]] = None,
        device: Optional[torch.device] = None,
        audio_placeholder_count: int = 0,
    ) -> Dict[str, torch.Tensor]:
        if self.base_vl is None:
            raise RuntimeError("Native-vision path requires base_vl to be initialized.")
        if device is None:
            device = next(self.llm.parameters()).device

        audio_prefix = ""
        if audio_placeholder_count > 0:
            audio_prefix = (" ".join([self.audio_context_token] * audio_placeholder_count)).strip() + "\n"

        prompt_texts = [audio_prefix + self.format_prompt(q) for q in questions]
        if answers is None:
            full_texts = prompt_texts
        else:
            full_texts = [f"{prompt_text} {a}" for prompt_text, a in zip(prompt_texts, answers)]

        images_arg = image_list if image_list is not None and any(img is not None for img in image_list) else None
        full_inputs = self.base_vl.prepare_inputs_for_training(
            full_texts,
            images=images_arg,
            device=str(device),
        )

        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]
        pixel_values = full_inputs.get("pixel_values")

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        if answers is not None:
            prompt_inputs = self.base_vl.prepare_inputs_for_training(
                prompt_texts,
                images=images_arg,
                device=str(device),
            )
            prompt_attention = prompt_inputs["attention_mask"]
            for i in range(labels.size(0)):
                full_nonpad = int(attention_mask[i].sum().item())
                prompt_nonpad = int(prompt_attention[i].sum().item())
                seq_len = int(labels.size(1))
                start = seq_len - full_nonpad
                prompt_end = min(seq_len, start + prompt_nonpad)
                labels[i, start:prompt_end] = -100

        if self.audio_context_token_id is not None:
            labels[input_ids == self.audio_context_token_id] = -100

        result: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        return result

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        questions: List[str],
        answers: List[str],
        audio_list: Optional[List[Any]] = None,
        image_list: Optional[List[Any]] = None,
        use_null_audio: bool = False,
        use_null_image: bool = False,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.llm.parameters()).device

        if self.use_native_vision:
            pending_audio_tokens = None
            if use_null_audio:
                pending_audio_tokens = self.make_null_audio_tokens(len(questions), device)
            elif audio_list is not None:
                pending_audio_tokens = self.encode_audio(audio_list)

            native_inputs = self._prepare_native_vision_inputs(
                questions=questions,
                answers=answers,
                image_list=image_list,
                device=device,
                audio_placeholder_count=0 if pending_audio_tokens is None else pending_audio_tokens.size(1),
            )
            input_ids = native_inputs["input_ids"]
            attention_mask = native_inputs["attention_mask"]
            labels = native_inputs["labels"]
            pixel_values = native_inputs.get("pixel_values")
            self._pending_audio_token_embeddings = pending_audio_tokens
            try:
                outputs = self.base_vl.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
            finally:
                self._pending_audio_token_embeddings = None
            return {"loss": outputs["loss"], "logits": outputs["logits"]}

        # Tokenize text
        enc = self._tokenize_batch(questions, answers=answers, device=device)
        input_ids = enc["input_ids"]           # (B, T_text)
        attention_mask = enc["attention_mask"]  # (B, T_text)

        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, T_text, D)

        # Build labels: -100 everywhere except answer tokens
        labels = input_ids.clone()
        # Mask out prompt tokens (everything before the assistant section)
        for i in range(labels.size(0)):
            prompt_only = self.format_prompt(questions[i])
            prompt_ids = self.tokenizer.encode(prompt_only, add_special_tokens=False)
            labels[i, :len(prompt_ids)] = -100
        # Mask padding
        labels[attention_mask == 0] = -100

        # Encode modality tokens
        modality_tokens_list = []
        if use_null_audio:
            modality_tokens_list.append(self.make_null_audio_tokens(len(questions), device))
        elif audio_list is not None:
            audio_tokens = self.encode_audio(audio_list)
            if audio_tokens is not None:
                modality_tokens_list.append(audio_tokens)
        if use_null_image:
            modality_tokens_list.append(self.make_null_vision_tokens(len(questions), device))
        elif image_list is not None:
            vision_tokens = self.encode_vision(image_list)
            if vision_tokens is not None:
                modality_tokens_list.append(vision_tokens)

        # Prepend modality tokens if present
        if modality_tokens_list:
            mod_tokens = torch.cat(modality_tokens_list, dim=1)  # (B, M, D)
            num_mod = mod_tokens.size(1)

            # Prepend to embeddings
            text_embeds = torch.cat([mod_tokens, text_embeds], dim=1)

            # Extend attention mask
            mod_mask = torch.ones(
                input_ids.size(0), num_mod,
                dtype=attention_mask.dtype, device=device,
            )
            attention_mask = torch.cat([mod_mask, attention_mask], dim=1)

            # Extend labels with -100 for modality positions
            mod_labels = torch.full(
                (input_ids.size(0), num_mod),
                -100, dtype=labels.dtype, device=device,
            )
            labels = torch.cat([mod_labels, labels], dim=1)

        # Forward through LoRA-adapted LLM
        outputs = self.llm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        if self.label_smoothing > 0 and loss is not None:
            # Apply label smoothing manually if needed
            # The HF loss already computes CE; label smoothing is a minor refinement
            pass

        return {"loss": loss, "logits": outputs.logits}

    # ------------------------------------------------------------------
    # Generate (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        questions: List[str],
        audio_list: Optional[List[Any]] = None,
        image_list: Optional[List[Any]] = None,
        max_new_tokens: int = 16,
        use_null_audio: bool = False,
        use_null_image: bool = False,
    ) -> List[str]:
        device = next(self.llm.parameters()).device

        if self.use_native_vision:
            pending_audio_tokens = None
            if use_null_audio:
                pending_audio_tokens = self.make_null_audio_tokens(len(questions), device)
            elif audio_list is not None:
                pending_audio_tokens = self.encode_audio(audio_list)

            native_inputs = self._prepare_native_vision_inputs(
                questions=questions,
                answers=None,
                image_list=image_list,
                device=device,
                audio_placeholder_count=0 if pending_audio_tokens is None else pending_audio_tokens.size(1),
            )
            input_ids = native_inputs["input_ids"]
            attention_mask = native_inputs["attention_mask"]
            pixel_values = native_inputs.get("pixel_values")
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "num_beams": 1,
            }
            if pixel_values is not None:
                generate_kwargs["pixel_values"] = pixel_values
                generate_kwargs["image_flags"] = torch.ones(
                    (pixel_values.size(0), 1),
                    dtype=torch.long,
                    device=pixel_values.device,
                )

            self._pending_audio_token_embeddings = pending_audio_tokens
            try:
                output_ids = self.llm.generate(**generate_kwargs)
            finally:
                self._pending_audio_token_embeddings = None
            prompt_width = int(attention_mask.size(1))
            preds = []
            for i in range(output_ids.size(0)):
                seq = output_ids[i]
                gen = seq[prompt_width:] if seq.size(0) > prompt_width else seq
                pred_text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
                preds.append(pred_text)
            return preds

        enc = self._tokenize_batch(questions, device=device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Encode modality tokens
        modality_tokens_list = []
        if use_null_audio:
            modality_tokens_list.append(self.make_null_audio_tokens(len(questions), device))
        elif audio_list is not None:
            audio_tokens = self.encode_audio(audio_list)
            if audio_tokens is not None:
                modality_tokens_list.append(audio_tokens)
        if use_null_image:
            modality_tokens_list.append(self.make_null_vision_tokens(len(questions), device))
        elif image_list is not None:
            vision_tokens = self.encode_vision(image_list)
            if vision_tokens is not None:
                modality_tokens_list.append(vision_tokens)

        if modality_tokens_list:
            mod_tokens = torch.cat(modality_tokens_list, dim=1)
            num_mod = mod_tokens.size(1)
            text_embeds = torch.cat([mod_tokens, text_embeds], dim=1)
            mod_mask = torch.ones(
                input_ids.size(0), num_mod,
                dtype=attention_mask.dtype, device=device,
            )
            attention_mask = torch.cat([mod_mask, attention_mask], dim=1)

        output_ids = self.llm.generate(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

        # Decode generated tokens robustly across HF paths:
        # some return full sequence (prompt + generation), others new tokens only.
        prompt_width = int(attention_mask.size(1))
        preds = []
        for i in range(output_ids.size(0)):
            seq = output_ids[i]
            gen = seq[prompt_width:] if seq.size(0) > prompt_width else seq
            pred_text = self.tokenizer.decode(gen, skip_special_tokens=True)
            preds.append(pred_text)
        return preds

    # ------------------------------------------------------------------
    # Trainable parameter helpers
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = []
        # LoRA parameters
        for p in self.llm.parameters():
            if p.requires_grad:
                params.append(p)
        # Audio projector
        for p in self.audio_projector.parameters():
            if p.requires_grad:
                params.append(p)
        # Vision projector (stage 2)
        if self.vision_projector is not None:
            for p in self.vision_projector.parameters():
                if p.requires_grad:
                    params.append(p)
        return params

    def print_trainable_summary(self) -> None:
        lora_params = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        audio_proj_params = sum(p.numel() for p in self.audio_projector.parameters())
        vision_proj_params = 0
        if self.vision_projector is not None:
            vision_proj_params = sum(p.numel() for p in self.vision_projector.parameters())
        total = lora_params + audio_proj_params + vision_proj_params
        print(f"[LoRA] Trainable parameters:", flush=True)
        print(f"  LoRA:             {lora_params:>12,}", flush=True)
        print(f"  Audio projector:  {audio_proj_params:>12,}", flush=True)
        print(f"  Vision projector: {vision_proj_params:>12,}", flush=True)
        print(f"  Total:            {total:>12,}", flush=True)


# ===================================================================
# Training loop
# ===================================================================

def train_epoch(
    model: LoRABaselineModel,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: Optional[LambdaLR],
    scaler: GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    global_step: int = 0,
    wandb_run: Any = None,
) -> Tuple[float, int]:
    model.train()
    # Keep frozen modules in eval
    model.audio_encoder.eval()
    if model.vision_encoder is not None:
        model.vision_encoder.eval()

    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # Resolve modality based on training stage
        audio = batch["audio"] if args.train_modality in ("audio", "both") else None
        images = batch["images"] if args.train_modality in ("image", "both") else None

        amp_ctx = autocast(dtype=torch.bfloat16) if args.bf16 else nullcontext()
        with amp_ctx:
            outputs = model(
                questions=batch["questions"],
                answers=batch["answers"],
                audio_list=audio,
                image_list=images,
            )
            loss = outputs["loss"]

        if loss is None:
            continue

        loss = loss / args.gradient_accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            trainable = model.get_trainable_params()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  step={global_step} loss={loss.item() * args.gradient_accumulation_steps:.4f} lr={lr:.2e}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "train/loss": loss.item() * args.gradient_accumulation_steps,
                        "train/lr": lr,
                    }, step=global_step)

        total_loss += loss.item() * args.gradient_accumulation_steps
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, global_step


# ===================================================================
# Evaluation
# ===================================================================

@torch.no_grad()
def evaluate(
    model: LoRABaselineModel,
    dataloader: DataLoader,
    device: torch.device,
    modality: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model.eval()
    eval_start = time.time()

    exact_total = 0.0
    extracted_total = 0.0
    f1_total = 0.0
    cat_f1_total = 0.0
    count = 0
    by_type: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total_batches = max(1, len(dataloader))
    log_every = max(1, min(200, total_batches // 5))

    for step, batch in enumerate(dataloader, start=1):
        # Resolve modality
        use_null_audio = False
        use_null_image = False
        if modality == "text":
            audio, images = None, None
        elif modality == "audio":
            audio, images = batch["audio"], None
        elif modality == "image":
            audio, images = None, batch["images"]
        elif modality == "both_null":
            audio, images = None, batch["images"]
            use_null_audio = True
        else:  # both
            audio, images = batch["audio"], batch["images"]

        preds = model.generate(
            questions=batch["questions"],
            audio_list=audio,
            image_list=images,
            max_new_tokens=args.max_answer_tokens,
            use_null_audio=use_null_audio,
            use_null_image=use_null_image,
        )

        for i, (pred, ref) in enumerate(zip(preds, batch["answers"])):
            exact = float(normalize_answer(pred) == normalize_answer(ref))
            extracted = float(extract_answer(pred) == extract_answer(ref))
            f1 = token_f1(pred, ref)
            cat_f1 = categorical_f1(pred, ref)

            exact_total += exact
            extracted_total += extracted
            f1_total += f1
            cat_f1_total += cat_f1
            count += 1

            qtype = batch["question_types"][i] if i < len(batch["question_types"]) else "unknown"
            by_type[qtype]["exact"] += exact
            by_type[qtype]["extracted"] += extracted
            by_type[qtype]["f1"] += f1
            by_type[qtype]["cat_f1"] += cat_f1
            by_type[qtype]["count"] += 1

        if step % log_every == 0 or step == total_batches:
            elapsed = time.time() - eval_start
            eta = (elapsed / step) * max(0, total_batches - step)
            running_extracted = 100.0 * extracted_total / max(1, count)
            print(
                f"  [eval:{modality}] step {step}/{total_batches} "
                f"extracted_em={running_extracted:.2f}% "
                f"elapsed={elapsed/60.0:.1f}m eta={eta/60.0:.1f}m",
                flush=True,
            )

    n = max(1, count)
    results = {
        "modality": modality,
        "exact_match": 100.0 * exact_total / n,
        "extracted_match": 100.0 * extracted_total / n,
        "token_f1": 100.0 * f1_total / n,
        "categorical_f1": 100.0 * cat_f1_total / n,
        "num_samples": count,
    }

    # Per question-type breakdown
    by_type_results = {}
    for qtype, vals in by_type.items():
        qn = max(1, vals["count"])
        by_type_results[qtype] = {
            "exact_match": 100.0 * vals["exact"] / qn,
            "extracted_match": 100.0 * vals["extracted"] / qn,
            "categorical_f1": 100.0 * vals["cat_f1"] / qn,
            "count": int(vals["count"]),
        }
    results["by_question_type"] = by_type_results

    return results


# ===================================================================
# Optimizer / scheduler builders
# ===================================================================

def build_optimizer(model: LoRABaselineModel, args: argparse.Namespace) -> AdamW:
    no_decay_terms = ("bias", "norm.weight", "layer_norm.weight", "LayerNorm.weight")
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(term in name for term in no_decay_terms):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)


def build_lr_scheduler(
    optimizer: AdamW,
    total_update_steps: int,
    args: argparse.Namespace,
) -> Tuple[Optional[LambdaLR], int]:
    if args.lr_scheduler == "none" or total_update_steps <= 0:
        return None, 0

    warmup_steps = int(total_update_steps * args.warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, max(0, total_update_steps - 1)))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_update_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if args.lr_scheduler == "linear":
            return max(0.01, 1.0 - progress)
        # cosine
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.01 + 0.99 * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda), warmup_steps


# ===================================================================
# Argument parser
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA baseline for modality composition comparison")

    # Data
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    # Stage
    p.add_argument("--stage", type=int, required=True, choices=[0, 1, 2],
                    help="0=baseline eval, 1=audio LoRA, 2=legacy vision LoRA on merged Qwen model")

    # Model
    p.add_argument("--llm-model", type=str, default=None,
                    help="Override LLM path (default: from config)")
    p.add_argument("--model-config", default="lora_baseline",
                    help="Config name from model_configs.py")

    # LoRA
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-target-modules", type=str, default="q_proj,v_proj",
                    help="Comma-separated target modules for LoRA")
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Stage 2: paths to merged model + audio projector from stage 1
    p.add_argument("--merged-model-path", type=Path, default=None,
                    help="Path to audio-merged LLM (stage 2 only)")
    p.add_argument("--audio-projector-path", type=Path, default=None,
                    help="Path to trained audio projector checkpoint (stage 2 only)")

    # Training
    p.add_argument("--train-modality", choices=["audio", "image", "both"], default="audio")
    p.add_argument("--eval-modalities", type=str, default="text,audio",
                    help="Comma-separated modalities to evaluate")
    p.add_argument("--stage0-results-path", type=Path, default=None,
                    help="Optional path to stage0_results.json for retention summaries")
    p.add_argument("--stage1-results-path", type=Path, default=None,
                    help="Optional path to stage1_final_results.json for stage-2 retention summaries")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lr-scheduler", choices=["none", "cosine", "linear"], default="cosine")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-answer-tokens", type=int, default=16)
    p.add_argument("--label-smoothing", type=float, default=0.1)

    # Precision
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="SAFE-LoRA-Baseline")

    return p.parse_args()


def _load_json_if_exists(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_summary_metrics(
    stage: int,
    final_results: Dict[str, Any],
    stage0_results: Optional[Dict[str, Any]] = None,
    stage1_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"stage": stage}

    text_score = final_results.get("text", {}).get("extracted_match")
    audio_score = final_results.get("audio", {}).get("extracted_match")
    image_score = final_results.get("image", {}).get("extracted_match")
    both_score = final_results.get("both", {}).get("extracted_match")
    both_null_score = final_results.get("both_null", {}).get("extracted_match")

    if text_score is not None:
        summary["text_extracted_match"] = text_score
    if audio_score is not None:
        summary["audio_extracted_match"] = audio_score
    if image_score is not None:
        summary["image_extracted_match"] = image_score
    if both_score is not None:
        summary["both_extracted_match"] = both_score
    if both_null_score is not None:
        summary["both_null_extracted_match"] = both_null_score

    if both_score is not None and audio_score is not None and image_score is not None:
        summary["best_single_extracted_match"] = max(audio_score, image_score)
        summary["composition_gain_vs_best_single"] = both_score - max(audio_score, image_score)
    if (
        both_score is not None
        and audio_score is not None
        and image_score is not None
        and text_score is not None
    ):
        summary["synergy_extracted_match"] = both_score - audio_score - image_score + text_score
    if both_score is not None and both_null_score is not None:
        summary["composition_gain_vs_both_null"] = both_score - both_null_score

    if stage0_results is not None and text_score is not None:
        stage0_text = stage0_results.get("text", {}).get("extracted_match", stage0_results.get("extracted_match"))
        if stage0_text is not None:
            summary["text_retention_drop_vs_stage0"] = text_score - stage0_text

    if stage == 1 and stage0_results is not None and audio_score is not None:
        stage0_text = stage0_results.get("text", {}).get("extracted_match", stage0_results.get("extracted_match"))
        if stage0_text is not None:
            summary["audio_gain_vs_stage0_text"] = audio_score - stage0_text

    if stage == 2 and stage1_results is not None:
        stage1_audio = stage1_results.get("audio", {}).get("extracted_match")
        stage1_text = stage1_results.get("text", {}).get("extracted_match")
        if stage1_audio is not None and audio_score is not None:
            summary["audio_retention_drop_vs_stage1"] = audio_score - stage1_audio
        if stage1_text is not None and text_score is not None:
            summary["text_retention_drop_vs_stage1"] = text_score - stage1_text

    return summary


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage0_results = _load_json_if_exists(args.stage0_results_path)
    stage1_results = _load_json_if_exists(args.stage1_results_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LoRA] Device: {device}", flush=True)
    print(f"[LoRA] Stage: {args.stage}", flush=True)

    # --- Build config ---
    try:
        cfg = get_config(args.model_config)
    except ValueError:
        # Fallback: build config inline if lora_baseline not yet registered
        cfg = {
            "llm_model_name": "models/OpenGVLab_InternVL3_5-8B",
            "vision_model_name": "built-in",
            "audio_encoder_type": "clap",
            "audio_encoder_config": {
                "model_name": "laion/larger_clap_music_and_speech",
                "sample_rate": 48000,
                "max_length": 10.0,
            },
            "llm_hidden_size": 4096,
            "audio_embed_dim": 512,
            "vision_embed_dim": 1024,
            "num_audio_tokens": 8,
            "freeze_audio_encoder": True,
            "label_smoothing": 0.1,
        }

    # Override from CLI
    if args.llm_model:
        cfg["llm_model_name"] = args.llm_model
    if args.merged_model_path and args.stage == 2:
        cfg["llm_model_name"] = str(args.merged_model_path)

    cfg["lora_rank"] = args.lora_rank
    cfg["lora_alpha"] = args.lora_alpha
    cfg["lora_target_modules"] = [m.strip() for m in args.lora_target_modules.split(",")]
    cfg["lora_dropout"] = args.lora_dropout
    cfg["label_smoothing"] = args.label_smoothing

    native_vision_mode = "internvl" in str(cfg["llm_model_name"]).lower()

    # --- Stage 0: baseline evaluation ---
    if args.stage == 0:
        print("=" * 60, flush=True)
        print("[Stage 0] Baseline evaluation", flush=True)
        print("=" * 60, flush=True)
        model = LoRABaselineModel(cfg, stage=0, apply_lora=False)
        model.to(device)
        val_ds = ManifestAVQADataset(args.val_manifest, args.media_root)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_avqa,
        )

        baseline_modalities = ["text"]
        if native_vision_mode:
            baseline_modalities.append("image")
        results = {"stage": 0}
        for modality in baseline_modalities:
            metrics = evaluate(model, val_loader, device, modality, args)
            results[modality] = metrics
            print(f"  [Stage0:{modality}] exact={metrics['exact_match']:.1f}% "
                  f"extracted={metrics['extracted_match']:.1f}% "
                  f"cat_f1={metrics['categorical_f1']:.1f}%", flush=True)
        if "text" in results:
            results["extracted_match"] = results["text"]["extracted_match"]
        print(f"\n[Stage 0] Results: {json.dumps(results, indent=2)}", flush=True)
        with open(args.output_dir / "stage0_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # --- Stage 1 or 2: Build model and train ---
    print("=" * 60, flush=True)
    print(f"[Stage {args.stage}] Building LoRA baseline model", flush=True)
    print("=" * 60, flush=True)

    model = LoRABaselineModel(cfg, stage=args.stage)

    # Stage 2: load trained audio projector from stage 1
    if args.stage == 2 and native_vision_mode:
        raise ValueError("Stage 2 legacy vision-LoRA path is not needed for InternVL native-vision mode.")

    if args.stage == 2 and args.audio_projector_path:
        print(f"[Stage 2] Loading audio projector from {args.audio_projector_path}", flush=True)
        state = torch.load(args.audio_projector_path, map_location="cpu")
        model.audio_projector.load_state_dict(state)
        # Freeze audio projector during stage 2 (only train vision LoRA + vision projector)
        for p in model.audio_projector.parameters():
            p.requires_grad = False
        print("[Stage 2] Audio projector loaded and frozen", flush=True)

    model.to(device)
    model.print_trainable_summary()

    # --- Data ---
    train_ds = ManifestAVQADataset(args.train_manifest, args.media_root)
    val_ds = ManifestAVQADataset(args.val_manifest, args.media_root)
    print(f"[Data] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_avqa,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_avqa,
    )

    # --- Optimizer & scheduler ---
    optimizer = build_optimizer(model, args)
    steps_per_epoch = max(1, len(train_loader) // args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_epochs
    scheduler, warmup_steps = build_lr_scheduler(optimizer, total_steps, args)
    scaler = GradScaler(enabled=False)  # bf16 doesn't use GradScaler

    print(f"[Training] epochs={args.num_epochs}, steps/epoch={steps_per_epoch}, "
          f"total_steps={total_steps}, warmup={warmup_steps}", flush=True)

    # --- WandB ---
    wandb_run = None
    if args.wandb and _wandb is not None:
        wandb_run = _wandb.init(
            project=args.wandb_project,
            name=f"lora_stage{args.stage}_{time.strftime('%Y%m%d_%H%M%S')}",
            config=vars(args),
        )

    # --- Training loop ---
    eval_modalities = [m.strip() for m in args.eval_modalities.split(",") if m.strip()]
    best_score = -1.0
    global_step = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(args.num_epochs):
        print(f"\n{'='*60}", flush=True)
        print(f"Epoch {epoch + 1}/{args.num_epochs}", flush=True)
        print(f"{'='*60}", flush=True)

        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, args, global_step=global_step, wandb_run=wandb_run,
        )
        print(f"  Epoch {epoch+1} train_loss={train_loss:.4f}", flush=True)

        # Evaluate
        epoch_results = {"epoch": epoch + 1, "train_loss": train_loss}
        for modality in eval_modalities:
            metrics = evaluate(model, val_loader, device, modality, args)
            print(f"  [{modality}] exact={metrics['exact_match']:.1f}% "
                  f"extracted={metrics['extracted_match']:.1f}% "
                  f"cat_f1={metrics['categorical_f1']:.1f}%", flush=True)
            epoch_results[modality] = metrics

            if wandb_run is not None:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        wandb_run.log({f"eval/{modality}/{k}": v}, step=global_step)

        history.append(epoch_results)

        # Save best checkpoint (use primary training modality score)
        primary_mod = args.train_modality if args.train_modality != "both" else "audio"
        if primary_mod in epoch_results:
            score = epoch_results[primary_mod].get("extracted_match", 0.0)
            if score > best_score:
                best_score = score
                print(f"  New best {primary_mod} extracted_match: {score:.1f}%", flush=True)
                # Save LoRA adapter
                model.llm.save_pretrained(args.output_dir / "best_lora")
                # Save projectors
                torch.save(
                    model.audio_projector.state_dict(),
                    args.output_dir / "best_audio_projector.pt",
                )
                if model.vision_projector is not None:
                    torch.save(
                        model.vision_projector.state_dict(),
                        args.output_dir / "best_vision_projector.pt",
                    )

    # --- Save final results ---
    with open(args.output_dir / f"stage{args.stage}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation on all modalities
    print(f"\n{'='*60}", flush=True)
    print(f"Final evaluation (Stage {args.stage})", flush=True)
    print(f"{'='*60}", flush=True)
    final_results = {"stage": args.stage}
    all_modalities = ["text", "audio"]
    if native_vision_mode or args.stage >= 2:
        all_modalities.extend(["image", "both_null", "both"])
    for modality in all_modalities:
        metrics = evaluate(model, val_loader, device, modality, args)
        final_results[modality] = metrics
        print(f"  [{modality}] exact={metrics['exact_match']:.1f}% "
              f"extracted={metrics['extracted_match']:.1f}% "
              f"cat_f1={metrics['categorical_f1']:.1f}%", flush=True)

    summary_metrics = _compute_summary_metrics(
        stage=args.stage,
        final_results=final_results,
        stage0_results=stage0_results,
        stage1_results=stage1_results,
    )
    final_results["summary"] = summary_metrics

    with open(args.output_dir / f"stage{args.stage}_final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    with open(args.output_dir / f"stage{args.stage}_summary.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"\n[Summary] {json.dumps(summary_metrics, indent=2)}", flush=True)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"\n[Done] Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
