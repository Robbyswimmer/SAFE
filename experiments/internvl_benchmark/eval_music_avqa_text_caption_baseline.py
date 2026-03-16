#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    parser.add_argument(
        "--model-backend",
        type=str,
        default="raw_internvl",
        choices=["raw_internvl", "safe"],
        help="Use raw InternVL directly or route through SAFEModel.",
    )
    parser.add_argument("--model-config", type=str, default="rkca_joint")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.environ.get("LLM_MODEL_PATH", "models/OpenGVLab_InternVL3_5-8B"),
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--media-root", type=str, required=True)
    parser.add_argument("--caption-field", type=str, required=True)
    parser.add_argument(
        "--input-mode",
        type=str,
        default="both",
        help="One mode or a comma-separated list from {image,caption,both_null,both}.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_prompt(caption: str, question: str, input_mode: str) -> str:
    SYSTEM_INSTRUCTION = (
        "You are answering questions about a music performance. "
        "You may be given a video frame and/or a description of what is heard in the audio. "
        "Use all available information to answer. "
        "If one modality is missing, rely on what you have."
    )

    if input_mode in ("caption", "both"):
        audio_line = f"Audio description: {caption}"
    else:
        audio_line = "Audio description: not available"

    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"{audio_line}\n"
        f"Question: {question}\n"
        "Answer with exactly one short answer token (single word or number)."
    )


def parse_input_modes(input_mode_arg: str) -> List[str]:
    valid = {"both", "caption", "image", "both_null"}
    modes = [x.strip() for x in input_mode_arg.split(",") if x.strip()]
    if not modes:
        raise ValueError("No input modes provided.")
    invalid = [m for m in modes if m not in valid]
    if invalid:
        raise ValueError(f"Invalid input modes: {invalid}. Valid modes: {sorted(valid)}")
    return modes


def _convert_to_pil(image: Any) -> Optional[Image.Image]:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        tensor = image
        if tensor.dim() == 4:
            tensor = tensor[0]
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        if tensor.dim() != 3:
            return None
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).clamp(0, 255)
        return Image.fromarray(tensor.cpu().numpy().astype("uint8"))
    return None


class SafeEvalEngine:
    def __init__(self, base_model: Any, tokenizer: Any) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer

    def generate_batch(
        self,
        prompts: Sequence[str],
        images: Optional[Sequence[Optional[Image.Image]]],
        device: torch.device,
        max_new_tokens: int,
        num_beams: int,
    ) -> List[str]:
        generation_inputs = self.base_model.prepare_multimodal_inputs(
            text=list(prompts),
            images=list(images) if images is not None else None,
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

        generated_ids = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            num_beams=num_beams,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        prompt_len = int(input_ids.shape[1])
        if generated_ids.dim() == 2 and generated_ids.size(1) > prompt_len:
            decoded_ids = generated_ids[:, prompt_len:]
        else:
            decoded_ids = generated_ids

        return self.tokenizer.batch_decode(
            decoded_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )


class RawInternVLEvalEngine:
    def __init__(self, llm_model: str, device: torch.device) -> None:
        self.device = device
        model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        try:
            self.model = AutoModel.from_pretrained(
                llm_model,
                trust_remote_code=True,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=False,
            )
        except RuntimeError as exc:
            if "meta tensors" not in str(exc).lower():
                raise
            print(
                "[RawInternVL] Meta-tensor init detected, retrying with "
                "numpy-backed torch.linspace to bypass meta .item() errors",
                flush=True,
            )
            # Root cause: InternVisionEncoder.__init__ does
            #   [x.item() for x in torch.linspace(0, rate, n)]
            # but a TorchFunctionMode from transformers/accelerate forces
            # torch.linspace onto the meta device where .item() is invalid.
            #
            # Previous attempts to pass device='cpu' through the original
            # torch.linspace failed because TorchFunctionMode intercepts
            # at C++ dispatch level, overriding explicit kwargs.
            #
            # Fix: replace torch.linspace with a plain Python function that
            # computes values via numpy and converts with torch.from_numpy.
            # Because it is NOT a torch op, TorchFunctionMode dispatch never
            # fires.  And torch.from_numpy is not a "device constructor" so
            # device modes leave it alone -- it always returns a CPU tensor.
            import numpy as _np

            _orig_linspace = torch.linspace

            def _np_linspace(start, end, steps, **_kw):
                arr = _np.linspace(float(start), float(end), int(steps),
                                   dtype=_np.float64)
                return torch.from_numpy(arr.copy())

            prev_default = None
            try:
                if hasattr(torch, "get_default_device"):
                    try:
                        prev_default = torch.get_default_device()
                    except Exception:
                        pass
                if hasattr(torch, "set_default_device"):
                    torch.set_default_device(None)
                torch.linspace = _np_linspace
                self.model = AutoModel.from_pretrained(
                    llm_model,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                    device_map=None,
                )
            finally:
                torch.linspace = _orig_linspace
                if prev_default is not None and hasattr(torch, "set_default_device"):
                    try:
                        torch.set_default_device(prev_default)
                    except Exception:
                        pass
        self.model = self.model.to(device)
        self.model.eval()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_model,
                trust_remote_code=True,
                fix_mistral_regex=True,
            )
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(llm_model)
        except Exception:
            self.image_processor = AutoProcessor.from_pretrained(llm_model, trust_remote_code=True)
        self._ensure_img_context_token()
        self._prep_logged = False

    def _ensure_img_context_token(self) -> None:
        if hasattr(self.model, "img_context_token_id") and self.model.img_context_token_id is None:
            img_token_id = getattr(self.model.config, "image_token_id", None)
            if img_token_id is None:
                try:
                    img_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
                    if img_token_id == self.tokenizer.unk_token_id:
                        img_token_id = None
                except Exception:
                    img_token_id = None
            if img_token_id is not None:
                self.model.img_context_token_id = img_token_id

    def _lookup_token_id(self, config_names: Sequence[str], token_candidates: Sequence[str]) -> Optional[int]:
        for name in config_names:
            value = getattr(self.model.config, name, None)
            if isinstance(value, int) and value >= 0:
                return value
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        for token in token_candidates:
            try:
                value = self.tokenizer.convert_tokens_to_ids(token)
            except Exception:
                continue
            if isinstance(value, list):
                value = value[0] if value else None
            if isinstance(value, int) and value >= 0 and (unk_id is None or value != unk_id):
                return value
        return None

    def _build_text_ids(self, prompt: str) -> List[int]:
        user_text = f"/no_think\n{prompt.strip()}"
        if bool(getattr(self.tokenizer, "chat_template", None)) and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                message = [{"role": "user", "content": user_text}]
                try:
                    prompt_ids = self.tokenizer.apply_chat_template(
                        message,
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    prompt_ids = self.tokenizer.apply_chat_template(
                        message,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                if torch.is_tensor(prompt_ids):
                    prompt_ids = prompt_ids.tolist()
                if prompt_ids and isinstance(prompt_ids[0], list):
                    prompt_ids = prompt_ids[0]
                if isinstance(prompt_ids, list) and prompt_ids:
                    return [int(tok) for tok in prompt_ids]
            except Exception:
                pass
        return self.tokenizer.encode(f"USER: {user_text}\nASSISTANT:", add_special_tokens=True)

    def _prepare_inputs(
        self,
        prompts: Sequence[str],
        images: Optional[Sequence[Optional[Image.Image]]],
    ) -> Dict[str, torch.Tensor]:
        image_token_id = getattr(self.model, "img_context_token_id", None)
        if not isinstance(image_token_id, int) or image_token_id < 0:
            image_token_id = getattr(self.model.config, "image_token_id", 151667)
        image_seq_length = getattr(self.model.config, "image_seq_length", None)
        runtime_image_tokens = getattr(self.model, "num_image_token", None)
        if isinstance(runtime_image_tokens, int) and runtime_image_tokens > 0:
            image_seq_length = runtime_image_tokens
        if not isinstance(image_seq_length, int) or image_seq_length <= 0:
            image_seq_length = 256
        image_start_token_id = self._lookup_token_id(
            ("img_start_token_id", "image_start_token_id", "vision_start_token_id"),
            ("<img>", "<image_start>", "<|vision_start|>"),
        )
        image_end_token_id = self._lookup_token_id(
            ("img_end_token_id", "image_end_token_id", "vision_end_token_id"),
            ("</img>", "<image_end>", "<|vision_end|>"),
        )
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)

        pil_images = [_convert_to_pil(img) for img in (images or [None] * len(prompts))]
        if len(pil_images) < len(prompts):
            pil_images.extend([None] * (len(prompts) - len(pil_images)))
        pil_images = pil_images[: len(prompts)]

        valid_pixel_images: List[Image.Image] = []
        image_indices: List[int] = []
        all_input_ids: List[torch.Tensor] = []
        for idx, prompt in enumerate(prompts):
            has_image = pil_images[idx] is not None
            text_ids = self._build_text_ids(prompt)
            if has_image:
                sample_ids: List[int] = []
                if image_start_token_id is not None:
                    sample_ids.append(image_start_token_id)
                sample_ids.extend([int(image_token_id)] * int(image_seq_length))
                if image_end_token_id is not None:
                    sample_ids.append(image_end_token_id)
                if newline_ids:
                    sample_ids.extend(newline_ids)
                sample_ids.extend(text_ids)
                valid_pixel_images.append(pil_images[idx])  # type: ignore[arg-type]
                image_indices.append(idx)
            else:
                sample_ids = text_ids
            all_input_ids.append(torch.tensor(sample_ids, dtype=torch.long))

        max_len = max(ids.size(0) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(prompts), max_len), dtype=torch.long)
        for idx, ids in enumerate(all_input_ids):
            start = max_len - ids.size(0)
            input_ids[idx, start:] = ids
            attention_mask[idx, start:] = 1

        result: Dict[str, torch.Tensor] = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }
        if valid_pixel_images:
            pixel_inputs = self.image_processor(images=valid_pixel_images, return_tensors="pt")
            pixel_values = pixel_inputs["pixel_values"]
            if len(valid_pixel_images) < len(prompts):
                full_pv = torch.zeros(
                    (len(prompts),) + tuple(pixel_values.shape[1:]),
                    dtype=pixel_values.dtype,
                )
                for pv_idx, batch_idx in enumerate(image_indices):
                    full_pv[batch_idx] = pixel_values[pv_idx]
                pixel_values = full_pv
            result["pixel_values"] = pixel_values.to(self.device)
        if not self._prep_logged:
            print(
                f"[RawInternVLPrep] image_token_id={image_token_id} image_seq_length={image_seq_length} "
                f"img_start_id={image_start_token_id} img_end_id={image_end_token_id}",
                flush=True,
            )
            self._prep_logged = True
        return result

    def generate_batch(
        self,
        prompts: Sequence[str],
        images: Optional[Sequence[Optional[Image.Image]]],
        device: torch.device,
        max_new_tokens: int,
        num_beams: int,
    ) -> List[str]:
        del device  # the raw engine owns its device placement
        inputs = self._prepare_inputs(prompts, images)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs.get("pixel_values")
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            num_beams=num_beams,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            if pixel_values is not None:
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    **gen_kwargs,
                )
            else:
                gen_model = getattr(self.model, "language_model", self.model)
                generated_ids = gen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
        prompt_len = int(input_ids.shape[1])
        if generated_ids.dim() == 2 and generated_ids.size(1) > prompt_len:
            decoded_ids = generated_ids[:, prompt_len:]
        else:
            decoded_ids = generated_ids
        return self.tokenizer.batch_decode(
            decoded_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )


def evaluate_mode(
    *,
    engine: Any,
    dataloader: DataLoader,
    device: torch.device,
    input_mode: str,
    caption_field: str,
    max_new_tokens: int,
    num_beams: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    predictions_out: List[Dict[str, Any]] = []
    raw_correct = 0
    extracted_correct = 0
    total = 0
    total_f1 = 0.0
    total_cat_f1 = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            prompts = [
                build_prompt(caption=cap, question=q, input_mode=input_mode)
                for cap, q in zip(batch["captions"], batch["questions"])
            ]
            images = batch["images"] if input_mode in {"both", "image", "both_null"} else None
            batch_predictions = engine.generate_batch(
                prompts=prompts,
                images=images,
                device=device,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
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
                        "caption_field": caption_field,
                        "caption": batch["captions"][i],
                        "input_mode": input_mode,
                        "prediction_raw": pred,
                        "prediction_extracted": pred_extracted,
                    }
                )

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"[eval:{input_mode}] step={batch_idx + 1} raw_em={100.0 * raw_correct / max(total, 1):.2f}% "
                    f"extracted_em={100.0 * extracted_correct / max(total, 1):.2f}% "
                    f"cat_f1={100.0 * total_cat_f1 / max(total, 1):.2f}",
                    flush=True,
                )

    summary = {
        "caption_field": caption_field,
        "input_mode": input_mode,
        "n": total,
        "raw_em": 100.0 * raw_correct / max(total, 1),
        "extracted_em": 100.0 * extracted_correct / max(total, 1),
        "f1": 100.0 * total_f1 / max(total, 1),
        "cat_f1": 100.0 * total_cat_f1 / max(total, 1),
    }
    print(
        f"[eval:{input_mode}] complete raw_em={summary['raw_em']:.2f} extracted_em={summary['extracted_em']:.2f} "
        f"cat_f1={summary['cat_f1']:.2f} n={summary['n']}",
        flush=True,
    )
    return summary, predictions_out


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_backend == "safe":
        from configs.model_configs import get_config

        model_config = get_config(args.model_config)
        model = create_model(model_config).to(device)
        if args.checkpoint:
            load_checkpoint(
                model=model,
                optimizer=None,
                scheduler=None,
                checkpoint_path=Path(args.checkpoint),
                device=device,
            )
        else:
            print("[eval] using raw frozen base model (no checkpoint provided)", flush=True)
        model.eval()
        base_model = model.module if hasattr(model, "module") else model
        engine: Any = SafeEvalEngine(base_model=base_model, tokenizer=base_model.base_vl.tokenizer)
    else:
        if args.checkpoint:
            raise ValueError("raw_internvl backend does not support SAFE checkpoints; use --model-backend safe")
        print(f"[eval] loading raw InternVL directly from {args.llm_model}", flush=True)
        engine = RawInternVLEvalEngine(llm_model=args.llm_model, device=device)

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

    mode_summaries: Dict[str, Any] = {}
    modes = parse_input_modes(args.input_mode)
    for mode in modes:
        mode_output_dir = output_dir / mode if len(modes) > 1 else output_dir
        mode_output_dir.mkdir(parents=True, exist_ok=True)
        summary, predictions_out = evaluate_mode(
            engine=engine,
            dataloader=dataloader,
            device=device,
            input_mode=mode,
            caption_field=args.caption_field,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        summary.update(
            {
                "checkpoint": args.checkpoint,
                "model_config": args.model_config,
                "manifest": args.manifest,
            }
        )
        mode_summaries[mode] = summary
        with (mode_output_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with (mode_output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
            for row in predictions_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if len(modes) > 1:
        combined = {
            "checkpoint": args.checkpoint,
            "model_config": args.model_config,
            "manifest": args.manifest,
            "caption_field": args.caption_field,
            "modes": mode_summaries,
        }
        with (output_dir / "results_all_modes.json").open("w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"[summary] {json.dumps(combined, indent=2)}", flush=True)
    else:
        print(f"[summary] {json.dumps(mode_summaries[modes[0]], indent=2)}", flush=True)


if __name__ == "__main__":
    main()
