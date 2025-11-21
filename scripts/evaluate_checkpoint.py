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
            # Move inputs to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio_embeds = batch["audio_embeds"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            
            # Project audio embeddings if projector exists
            if hasattr(model, "audio_projector"):
                # Ensure audio_embeds is correct dtype
                base_dtype = next(model.parameters()).dtype
                audio_embeds = audio_embeds.to(dtype=base_dtype)
                
                # Project: (B, 1, 512) -> (B, 1, 5120)
                audio_tokens = model.audio_projector(audio_embeds)
            else:
                audio_tokens = audio_embeds

            gen_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_tokens=audio_tokens,
                audio_attention_mask=audio_mask,
                **generation_kwargs
            )
            
            decoded_preds = model.base_vl.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            
            batch_answers = batch.get("answers") # List of lists of strings
            
            for i, pred in enumerate(decoded_preds):
                # Use a global index or try to get ID
                sample_id = str(len(predictions))
                
                pred_norm = _normalize_audio_caption(pred)
                predictions[sample_id] = [pred_norm]
                
                if batch_answers:
                    refs = batch_answers[i]
                    if isinstance(refs, str): refs = [refs]
                    refs_norm = [_normalize_audio_caption(r) for r in refs]
                    references[sample_id] = refs_norm
                else:
                    pass

    return predictions, references

def compute_metrics(predictions, references):
    """Compute CIDEr and SPICE."""
    print("Computing metrics...")
    
    cider_scorer = Cider()
    cider_score, cider_scores = cider_scorer.compute_score(references, predictions)
    
    return {
        "CIDEr": cider_score * 100.0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, help="Run ID (e.g. 232228)")
    parser.add_argument("--checkpoint", help="Specific checkpoint filename")
    parser.add_argument("--data_root", default="experiments/full_training/data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--llm_model_name", help="Override LLM model name (e.g. google/flan-t5-base)")
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # 1. Locate Checkpoint
    checkpoint_path = load_checkpoint(args.run_id, args.checkpoint)
    run_dir = checkpoint_path.parent.parent
    config_path = run_dir / "config.json" 
    
    # Default config (Match FULL_CONFIG from model_configs.py)
    model_config = {
        "llm_model_name": args.llm_model_name or "llava-hf/llava-1.5-13b-hf", 
        "audio_encoder_type": "clap",
        "llm_hidden_size": 5120 # Default for LLaVA-13B
    }
    
    if config_path.exists():
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
            # Map 'variant' to 'llm_model_name' if present
            if "variant" in loaded_config:
                 variant = loaded_config["variant"]
                 if variant == "t5-base":
                     model_config["llm_model_name"] = "t5-base"
                     model_config["llm_hidden_size"] = 768
                 elif variant == "flan-t5-base":
                     model_config["llm_model_name"] = "google/flan-t5-base"
                     model_config["llm_hidden_size"] = 768
                 else:
                     model_config["llm_model_name"] = variant
            
            # Allow config to override if not explicitly set by arg
            if not args.llm_model_name and "llm_model_name" in loaded_config:
                model_config["llm_model_name"] = loaded_config["llm_model_name"]
            
            if "llm_hidden_size" in loaded_config:
                model_config["llm_hidden_size"] = loaded_config["llm_hidden_size"]
    else:
        print(f"WARNING: Config not found at {config_path}. Using default: {model_config['llm_model_name']}")
        # Adjust hidden size if user overrode model name via CLI but no config file
        if "t5-base" in model_config["llm_model_name"]:
            model_config["llm_hidden_size"] = 768
        elif "t5-large" in model_config["llm_model_name"]:
            model_config["llm_hidden_size"] = 1024
        elif "llava" in model_config["llm_model_name"] and "7b" in model_config["llm_model_name"]:
             model_config["llm_hidden_size"] = 4096

    print(f"Initializing model with LLM: {model_config['llm_model_name']} (hidden_size={model_config['llm_hidden_size']})")
    model = SAFEModel(
        llm_model_name=model_config['llm_model_name'],
        audio_encoder_type=model_config['audio_encoder_type'],
        llm_hidden_size=model_config['llm_hidden_size']
    )
    
    print("Loading weights...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        
    # Handle prefix 'safe_model.' if present (from Lightning)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("safe_model."):
            new_state_dict[k.replace("safe_model.", "")] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False) 
    model.to(args.device)
    
    # 3. Load Data
    print(f"Loading {args.split} dataset from {args.data_root}...")
    dataset = AudioCapsDataset(data_path=args.data_root, split=args.split)
    
    def smart_collate(batch):
        tokenizer = model.base_vl.tokenizer
        input_texts = ["describe the audio" for _ in batch] 
        
        inputs = tokenizer(input_texts, padding=True, return_tensors="pt")
        
        audio_tensors = []
        for x in batch:
            if 'audio_embed' in x:
                audio_tensors.append(torch.from_numpy(x['audio_embed']))
            else:
                pass
                
        if audio_tensors:
            audio_embeds = torch.stack(audio_tensors)
            if audio_embeds.dim() == 2: 
                audio_embeds = audio_embeds.unsqueeze(1)
        else:
            audio_embeds = torch.zeros(len(batch), 1, 512)
            
        audio_mask = torch.ones(audio_embeds.shape[0], audio_embeds.shape[1])

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "audio_embeds": audio_embeds,
            "audio_mask": audio_mask,
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
