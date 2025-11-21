import argparse
import json
import os
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from safe.models.safe_model import SAFEModel
from safe.data.datasets import AudioCapsDataset
from configs.model_configs import get_config as get_model_config

def _normalize_audio_caption(text) -> str:
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
    normalized = normalized.replace("\u2019", "'")  # Normalise curly apostrophes
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

    cleaned_tokens = []
    for tok in tokens:
        cleaned_tokens.append(number_map.get(tok, tok))

    if not cleaned_tokens:
        return ""

    return " ".join(cleaned_tokens)

def load_checkpoint(run_id, checkpoint_name=None, experiments_dir="experiments/full_training"):
    """Find and load the specified checkpoint."""
    runs_dir = Path(experiments_dir) / "runs" / run_id
    
    if not runs_dir.exists():
        # Try searching for the run ID in subdirectories if exact match fails
        matches = list(Path(experiments_dir).rglob(run_id))
        if matches:
            runs_dir = matches[0]
        else:
            raise FileNotFoundError(f"Run directory not found for ID: {run_id}")
            
    # Find timestamp directory (usually just one)
    timestamp_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        raise FileNotFoundError(f"No timestamp directory found in {runs_dir}")
    
    # Sort by name (timestamp) and take latest
    latest_run = sorted(timestamp_dirs)[-1]
    checkpoints_dir = latest_run / "checkpoints"
    
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
        
    if checkpoint_name:
        checkpoint_path = checkpoints_dir / checkpoint_name
    else:
        # Find latest checkpoint
        checkpoints = list(checkpoints_dir.glob("*.pt")) + list(checkpoints_dir.glob("*.safetensors"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
        # Sort by modification time
        checkpoint_path = sorted(checkpoints, key=os.path.getmtime)[-1]
        
    print(f"Loading checkpoint: {checkpoint_path}")
    return checkpoint_path

def evaluate(model, dataloader, device, generation_kwargs):
    """Run evaluation loop."""
    model.eval()
    
    predictions = {}
    references = {}
    
    print(f"Evaluating on {len(dataloader.dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Prepare inputs using SAFEModel's method (handles chat templates & audio encoding)
            inputs = model.prepare_multimodal_inputs(
                text=batch["questions"],
                audio=batch["audio"],
                device=device
            )
            
            # Generate
            gen_outputs = model.generate(
                **inputs,
                **generation_kwargs
            )
            
            decoded_preds = model.base_vl.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            
            batch_answers = batch.get("answers")
            
            for i, pred in enumerate(decoded_preds):
                sample_id = str(len(predictions))
                
                pred_norm = _normalize_audio_caption(pred)
                predictions[sample_id] = [pred_norm]
                
                if batch_answers:
                    refs = batch_answers[i]
                    if isinstance(refs, str): refs = [refs]
                    refs_norm = [_normalize_audio_caption(r) for r in refs]
                    references[sample_id] = refs_norm
                else:
                    refs_norm = ["<No Reference>"]
                    pass

                # Log every 50 samples
                if int(sample_id) % 50 == 0:
                    print(f"\n[Sample {sample_id}]")
                    print(f"  Pred: {pred_norm}")
                    print(f"  Ref:  {refs_norm[0]}")

    return predictions, references
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
    if "llava" in name and "7b" in name:
        return 4096
    return fallback


def build_safe_model_kwargs(model_config_name: str, overrides: dict) -> dict:
    try:
        model_config = get_model_config(model_config_name)
    except ValueError as exc:
        raise SystemExit(f"Unknown model config '{model_config_name}'.") from exc

    model_config.update(overrides)

    llm_name = model_config.get("llm_model_name")
    if llm_name:
        current_hidden = model_config.get("llm_hidden_size", 4096)
        model_config["llm_hidden_size"] = _infer_hidden_size(llm_name, current_hidden)

    kwargs = {key: model_config[key] for key in SAFE_MODEL_KEYS if key in model_config}
    if "llm_model_name" not in kwargs:
        raise SystemExit("Model configuration missing 'llm_model_name'.")
    return kwargs


from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

def compute_metrics(predictions, references):
    """Compute all standard captioning metrics."""
    print("Computing metrics...")
    
    metrics = {}
    
    # CIDEr
    print("Computing CIDEr...")
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(references, predictions)
    metrics["CIDEr"] = cider_score * 100.0
    
    # SPICE
    print("Computing SPICE...")
    try:
        spice_scorer = Spice()
        spice_score, _ = spice_scorer.compute_score(references, predictions)
        metrics["SPICE"] = spice_score * 100.0
    except Exception as e:
        print(f"Warning: SPICE computation failed: {e}")
        metrics["SPICE"] = 0.0

    # BLEU
    print("Computing BLEU...")
    try:
        bleu_scorer = Bleu(4)
        bleu_score, _ = bleu_scorer.compute_score(references, predictions)
        for i, score in enumerate(bleu_score):
            metrics[f"BLEU-{i+1}"] = score * 100.0
    except Exception as e:
        print(f"Warning: BLEU computation failed: {e}")

    # ROUGE-L
    print("Computing ROUGE-L...")
    try:
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(references, predictions)
        metrics["ROUGE_L"] = rouge_score * 100.0
    except Exception as e:
        print(f"Warning: ROUGE computation failed: {e}")

    # METEOR
    print("Computing METEOR...")
    try:
        meteor_scorer = Meteor()
        meteor_score, _ = meteor_scorer.compute_score(references, predictions)
        metrics["METEOR"] = meteor_score * 100.0
    except Exception as e:
        print(f"Warning: METEOR computation failed: {e}")
        
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, help="Run ID (e.g. 232228)")
    parser.add_argument("--checkpoint", help="Specific checkpoint filename")
    parser.add_argument("--data_root", default="experiments/full_training/data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--llm_model_name", help="Override LLM model name (e.g. google/flan-t5-base)")
    parser.add_argument("--num_audio_tokens", type=int, default=16, help="Number of audio tokens")
    parser.add_argument("--model_config", default="full", help="SAFE model preset (demo/full/multimodal)")
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # 1. Locate Checkpoint
    checkpoint_path = load_checkpoint(args.run_id, args.checkpoint)
    run_dir = checkpoint_path.parent.parent
    config_path = run_dir / "config.json" 
    
    if config_path.exists():
        print(f"Loading trainer config from {config_path}")
    else:
        print(f"WARNING: Trainer config not found at {config_path}. Continuing with defaults.")

    model_overrides = {}
    if args.llm_model_name:
        model_overrides["llm_model_name"] = args.llm_model_name
    model_overrides["num_audio_tokens"] = args.num_audio_tokens

    safe_model_kwargs = build_safe_model_kwargs(args.model_config, model_overrides)
    print(
        "Initializing model with preset '{}' (LLM: {} hidden_size={} num_audio_tokens={})".format(
            args.model_config,
            safe_model_kwargs.get("llm_model_name"),
            safe_model_kwargs.get("llm_hidden_size"),
            safe_model_kwargs.get("num_audio_tokens"),
        )
    )
    model = SAFEModel(**safe_model_kwargs)
    
    print("Loading weights...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
        
    # Handle prefix 'safe_model.' if present (from Lightning)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("safe_model."):
            new_state_dict[k.replace("safe_model.", "")] = v
        else:
            new_state_dict[k] = v
            
    # Check for audio_projector keys
    projector_keys = [k for k in new_state_dict.keys() if "audio_projector" in k]
    if not projector_keys:
        print("WARNING: No 'audio_projector' keys found in checkpoint! Audio will be random noise.")
    else:
        print(f"Found {len(projector_keys)} 'audio_projector' keys in checkpoint.")

    # Remap fusion keys if needed (handle PEFT wrapping)
    fusion_keys = [k for k in new_state_dict.keys() if "fusion_adapter" in k]
    remapped_count = 0
    
    # Create a copy of keys to iterate safely
    for k in list(new_state_dict.keys()):
        if "fusion_adapter.cross_attention" in k and "base_model.model" not in k:
            # Checkpoint has flattened keys (e.g. fusion_adapter.cross_attention.query.weight)
            # Model expects nested keys (e.g. fusion_adapter.cross_attention.base_model.model.query.weight)
            suffix = k.split("fusion_adapter.cross_attention.")[1]
            new_key = f"fusion_adapter.cross_attention.base_model.model.{suffix}"
            new_state_dict[new_key] = new_state_dict.pop(k)
            remapped_count += 1
            
    if remapped_count > 0:
        print(f"Remapped {remapped_count} fusion_adapter keys to match PEFT structure.")

    # Load with strict=False but report missing keys
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False) 
    
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
        # Filter for relevant keys (projector, adapter)
        relevant_missing = [k for k in missing_keys if "audio_projector" in k or "fusion_adapter" in k]
        if relevant_missing:
            print(f"CRITICAL: Missing relevant keys: {relevant_missing}")
            
    model.to(args.device)
    
    # 3. Load Data
    print(f"Loading {args.split} dataset from {args.data_root}...")
    dataset = AudioCapsDataset(data_path=args.data_root, split=args.split)
    
    def smart_collate(batch):
        audio_data = []
        for x in batch:
            audio_entry = x.get("audio")
            if isinstance(audio_entry, tuple):
                # Unpack (waveform, sr) -> waveform
                audio_data.append(audio_entry[0])
            else:
                audio_data.append(audio_entry)
                
        return {
            "questions": [x.get("question", "describe the audio") for x in batch],
            "audio": audio_data,
            "answers": [x.get("answers") for x in batch]
        }

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=smart_collate,
        num_workers=4
    )
    
    # Debug: Check first batch
    first_batch = next(iter(dataloader))
    print(f"[Debug] First batch answers: {first_batch['answers'][:2]}")
    
    # 4. Evaluate
    gen_kwargs = {
        "max_new_tokens": 40,
        "num_beams": 4,
        "length_penalty": 1.0,
        "repetition_penalty": 1.2
    }
    
    preds, refs = evaluate(model, dataloader, args.device, gen_kwargs)
    
    # 5. Metrics
    metrics = compute_metrics(preds, refs)
    
    print("\n" + "="*40)
    print(f"RESULTS for {args.run_id} on {args.split}")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
    print("="*40)
    
    # Save results
    results_path = run_dir / f"eval_results_{args.split}.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved results to {results_path}")



if __name__ == "__main__":
    main()
