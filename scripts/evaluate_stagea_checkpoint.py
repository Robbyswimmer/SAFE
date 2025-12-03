import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safe.models.safe_model import SAFEModel
from safe.data.datasets import AudioCapsDataset, create_safe_dataloader
from safe.training.stage_a import StageATrainer
from configs.model_configs import get_config as get_model_config


def load_checkpoint(run_id: str, checkpoint_name: str | None = None, experiments_dir: str = "experiments/full_training") -> Path:
    """Locate checkpoint path for a given run ID (supports full_training and phase1)."""
    runs_dir = Path(experiments_dir) / "runs" / run_id

    if not runs_dir.exists():
        # Try phase1 directory if not found in full_training
        phase1_runs_dir = Path("experiments/phase1") / "runs" / run_id
        if phase1_runs_dir.exists():
            runs_dir = phase1_runs_dir
        else:
            # Try searching for the run ID in subdirectories if exact match fails
            matches = list(Path(experiments_dir).rglob(run_id))
            if not matches:
                matches = list(Path("experiments/phase1").rglob(run_id))
            if matches:
                runs_dir = matches[0]
            else:
                raise FileNotFoundError(
                    f"Run directory not found for ID: {run_id} in experiments/full_training or experiments/phase1"
                )

    timestamp_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        raise FileNotFoundError(f"No timestamp directory found in {runs_dir}")

    latest_run = sorted(timestamp_dirs)[-1]
    checkpoints_dir = latest_run / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    if checkpoint_name:
        checkpoint_path = checkpoints_dir / checkpoint_name
    else:
        checkpoints = list(checkpoints_dir.glob("*.pt")) + list(checkpoints_dir.glob("*.safetensors"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
        checkpoint_path = sorted(checkpoints, key=os.path.getmtime)[-1]

    print(f"Loading checkpoint: {checkpoint_path}")
    return checkpoint_path


SAFE_MODEL_KEYS = {
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


def _infer_hidden_size(model_name: str, fallback: int) -> int:
    name = (model_name or "").lower()
    if "t5-base" in name:
        return 768
    if "t5-large" in name:
        return 1024
    if "llava" in name and "13b" in name:
        return 5120
    if "llava" in name and "7b" in name:
        return 4096
    return fallback


def build_safe_model_kwargs(model_config_name: str, overrides: dict) -> dict:
    model_config = get_model_config(model_config_name)
    model_config.update(overrides)

    llm_name = model_config.get("llm_model_name")
    if llm_name:
        current_hidden = model_config.get("llm_hidden_size", 4096)
        model_config["llm_hidden_size"] = _infer_hidden_size(llm_name, current_hidden)

    kwargs = {key: model_config[key] for key in SAFE_MODEL_KEYS if key in model_config}
    if "llm_model_name" not in kwargs:
        raise SystemExit("Model configuration missing 'llm_model_name'.")
    return kwargs


def main():
    parser = argparse.ArgumentParser(description="Stage-A style evaluation with StageATrainer (AudioCaps subset).")
    parser.add_argument("--run_id", required=True, help="Run ID (e.g. 232363)")
    parser.add_argument("--checkpoint", help="Specific checkpoint filename")
    parser.add_argument("--data_root", default="experiments/full_training/data")
    parser.add_argument("--split", default="test", help="Dataset split (val/test)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model_config", default="phase1", help="SAFE model preset (demo/full/multimodal/phase1)")
    parser.add_argument("--num_audio_tokens", type=int, default=32, help="Number of audio tokens (match training)")
    parser.add_argument("--max_audio_eval_samples", type=int, default=800, help="Max AudioCaps samples for eval subset")
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # 1. Locate checkpoint
    checkpoint_path = load_checkpoint(args.run_id, args.checkpoint)
    run_dir = checkpoint_path.parent.parent

    # 2. Load checkpoint object and model weights
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["model_state_dict"]
    elif isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["state_dict"]
    else:
        state_dict = checkpoint_obj

    model_overrides = {"num_audio_tokens": args.num_audio_tokens}
    safe_model_kwargs = build_safe_model_kwargs(args.model_config, model_overrides)
    print(
        "Initializing SAFEModel with preset '{}' (LLM: {} hidden_size={} num_audio_tokens={})".format(
            args.model_config,
            safe_model_kwargs.get("llm_model_name"),
            safe_model_kwargs.get("llm_hidden_size"),
            safe_model_kwargs.get("num_audio_tokens"),
        )
    )
    model = SAFEModel(**safe_model_kwargs)

    # Remap fusion keys if needed (handle PEFT wrapping)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("safe_model."):
            new_state_dict[k.replace("safe_model.", "")] = v
        else:
            new_state_dict[k] = v

    remapped_count = 0
    for k in list(new_state_dict.keys()):
        if "fusion_adapter.cross_attention" in k and "base_model.model" not in k:
            suffix = k.split("fusion_adapter.cross_attention.")[1]
            new_key = f"fusion_adapter.cross_attention.base_model.model.{suffix}"
            new_state_dict[new_key] = new_state_dict.pop(k)
            remapped_count += 1
    if remapped_count > 0:
        print(f"Remapped {remapped_count} fusion_adapter keys to match PEFT structure.")

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
        relevant_missing = [k for k in missing_keys if "audio_projector" in k or "fusion_adapter" in k]
        if relevant_missing:
            print(f"CRITICAL: Missing relevant keys: {relevant_missing}")

    # Let StageATrainer handle device placement via SAFEModel.to_device

    # 3. Load AudioCaps subset
    print(f"Loading {args.split} dataset from {args.data_root}...")
    split_aliases = {"train": "train", "val": "val", "validation": "val", "test": "test"}
    normalized_split = split_aliases.get(args.split.lower(), args.split)
    dataset = AudioCapsDataset(data_path=args.data_root, split=normalized_split)

    # Subsample for memory-friendly eval
    max_samples = args.max_audio_eval_samples
    if max_samples > 0 and len(dataset) > max_samples:
        indices = list(range(max_samples))
        dataset = Subset(dataset, indices)
        print(f"Using subset of {max_samples} samples out of {len(dataset)} for evaluation.")
    else:
        print(f"Using all {len(dataset)} samples for evaluation.")

    train_loader = create_safe_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    val_loader = train_loader  # For eval we only need one stream

    # 4. Build StageATrainer config (start from training config if available)
    train_config = {}
    if isinstance(checkpoint_obj, dict) and "config" in checkpoint_obj:
        cfg = checkpoint_obj["config"]
        if isinstance(cfg, dict):
            train_config.update(cfg)

    # Ensure core fields for eval
    train_config.setdefault("num_epochs", 1)
    train_config.setdefault("output_dir", str(run_dir))
    train_config.setdefault("eval_steps", 1_000)
    train_config.setdefault("logging_steps", 100)

    # Memory-friendly overrides
    train_config["disable_bertscore"] = True  # Avoid loading RoBERTa
    train_config["max_audio_eval_samples"] = max_samples
    train_config["max_vl_eval_samples"] = 0  # AudioCaps only
    train_config["max_eval_batches"] = None  # We already subset dataset
    train_config["gradient_accumulation_steps"] = 1

    print("\nStage-A style evaluation config overrides:")
    print(f"  disable_bertscore: {train_config['disable_bertscore']}")
    print(f"  max_audio_eval_samples: {train_config['max_audio_eval_samples']}")
    print(f"  max_vl_eval_samples: {train_config['max_vl_eval_samples']}")
    print(f"  batch_size: {args.batch_size}")

    # 5. Run evaluation via StageATrainer
    trainer = StageATrainer(
        safe_model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=train_config,
        curriculum_config=None,
    )

    max_eval_batches = train_config.get("max_eval_batches", None)
    metrics = trainer.evaluate(
        max_batches=max_eval_batches,
        description=f"AudioCaps-{normalized_split}-StageA",
    )

    print("\n" + "=" * 40)
    print(f"STAGE-A STYLE RESULTS for {args.run_id} on {normalized_split}")
    print("=" * 40)
    for k, v in metrics.items():
        # Only print scalar floats nicely
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("=" * 40)

    results_path = run_dir / f"stagea_eval_results_{normalized_split}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved Stage-A style eval results to {results_path}")


if __name__ == "__main__":
    main()

