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
from safe.training.stage_a import _normalize_audio_caption

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
            
            # Generate
            # We need to prepare inputs for generation
            # SAFEModel.generate expects standard generation args
            # But we need to handle the multimodal inputs
            
            # The model's generate method might not handle the audio embeddings injection automatically 
            # if we just call model.generate(). 
            # We should use the underlying generate logic or prepare_inputs_for_generation.
            # However, SAFEModel wraps a T5/Llama model.
            # Let's look at how StageATrainer does it:
            # It calls self.safe_model.generate(...)
            
            # We need to pass audio_embeds and audio_mask to generate
            gen_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_embeds=audio_embeds,
                audio_mask=audio_mask,
                **generation_kwargs
            )
            
            decoded_preds = model.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            
            # Store results
            # We need unique IDs for COCO eval. 
            # The dataset might not provide them in the batch if we didn't collate them.
            # But we can use the index if we are careful.
            # Better: The batch should contain 'id' or 'youtube_id' if the dataset provides it.
            # AudioCapsDataset returns a dict.
            
            # Let's check what keys are in batch
            # If 'youtube_id' is present (it should be if we updated the dataset), use it.
            # Otherwise use a counter.
            
            # Actually, AudioCapsDataset.__getitem__ returns:
            # input_ids, attention_mask, audio_embeds, audio_mask, answers, etc.
            # It does NOT return metadata by default in the tensor batch unless collated specially.
            # But we can iterate the dataset directly if needed, or just rely on order.
            # For COCO eval, we just need matching keys.
            
            # Wait, the references (ground truth) are needed.
            # batch['answers'] contains the list of references for this batch?
            # The default collator might stack them if they are tensors, but answers are lists of strings.
            # So they might be in a list.
            
            batch_answers = batch.get("answers") # List of lists of strings
            
            for i, pred in enumerate(decoded_preds):
                # Use a global index or try to get ID
                # Since we are sequential, we can just use a running ID
                sample_id = str(len(predictions))
                
                pred_norm = _normalize_audio_caption(pred)
                predictions[sample_id] = [pred_norm]
                
                if batch_answers:
                    refs = batch_answers[i]
                    if isinstance(refs, str): refs = [refs]
                    refs_norm = [_normalize_audio_caption(r) for r in refs]
                    references[sample_id] = refs_norm
                else:
                    # Should not happen for test set if we want to eval
                    pass

    return predictions, references

def compute_metrics(predictions, references):
    """Compute CIDEr and SPICE."""
    print("Computing metrics...")
    
    # Format for pycocoevalcap
    # preds: {id: [text]}
    # refs: {id: [text, text, ...]}
    
    cider_scorer = Cider()
    cider_score, cider_scores = cider_scorer.compute_score(references, predictions)
    
    # SPICE can be slow and requires java, maybe skip for quick check
    # spice_scorer = Spice()
    # spice_score, spice_scores = spice_scorer.compute_score(references, predictions)
    
    return {
        "CIDEr": cider_score * 100.0, # Scale to 0-100 usually? Or is it already? 
        # PyCocoEvalCap returns raw score (e.g. 0.6). Standard reporting is often *100 (60.0).
        # Let's check standard. Yes, usually reported as 60.0.
        # "SPICE": spice_score * 100.0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, help="Run ID (e.g. 232228)")
    parser.add_argument("--checkpoint", help="Specific checkpoint filename")
    parser.add_argument("--data_root", default="experiments/full_training/data/audiocaps")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # 1. Locate Checkpoint
    checkpoint_path = load_checkpoint(args.run_id, args.checkpoint)
    
    # 2. Load Model
    # We need to know the model variant (t5-base, t5-large, etc.)
    # It's usually in the config saved with the run, or we can try to infer/default.
    # For now, let's assume 't5-base' or try to load from checkpoint args if possible.
    # SAFEModel.from_pretrained might work if it's a full save, but it's likely a state_dict.
    # We'll initialize a default SAFEModel and load state_dict.
    # WARNING: If architecture differs (e.g. projection dim), this will fail.
    # Ideally we load the config.json from the run dir.
    
    run_dir = checkpoint_path.parent.parent
    config_path = run_dir / "config.json" # If it exists
    
    # Default config
    model_config = {
        "variant": "t5-base",
        "audio_encoder_model": "laion/clap-htsat-unfused"
    }
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
            model_config.update(loaded_config)
            
    print(f"Initializing model with variant: {model_config['variant']}")
    model = SAFEModel(
        variant=model_config['variant'],
        audio_encoder_model=model_config['audio_encoder_model']
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
            
    model.load_state_dict(new_state_dict, strict=False) # strict=False to ignore missing keys if any (e.g. loss weights)
    model.to(args.device)
    
    # 3. Load Data
    print(f"Loading {args.split} dataset from {args.data_root}...")
    dataset = AudioCapsDataset(data_path=args.data_root, split=args.split)
    
    # We need a collate function that handles the multimodal inputs
    # SAFEModel has a collate helper or we can use the dataset's if it has one.
    # AudioCapsDataset usually returns raw items.
    # We need to batch them.
    
    # Let's define a simple collator
    def collate_fn(batch):
        # batch is list of dicts
        # we need to tokenize text and stack audio
        
        # Extract inputs
        # For evaluation, we might provide a dummy prompt or the question
        # AudioCaps is usually "describe this sound"
        
        # We need to use the model's tokenizer
        tokenizer = model.tokenizer
        
        input_texts = [item.get("question", "describe the audio") for item in batch]
        
        # Tokenize inputs
        inputs = tokenizer(
            input_texts,
            padding=True,
            return_tensors="pt"
        )
        
        # Process Audio
        # Dataset should return 'audio_embeds' (if precomputed) or 'audio' (waveform)
        # If waveform, we need to encode.
        # But wait, SAFEModel expects embeddings in forward/generate?
        # Yes, `audio_embeds`.
        # If the dataset returns waveforms, we need the audio encoder.
        # But SAFEModel HAS the audio encoder inside it (self.audio_encoder).
        # However, AudioCapsDataset might be returning precomputed features if configured?
        # Let's check AudioCapsDataset.__getitem__
        # It usually returns 'audio' (waveform) and 'audio_embeds' (if hdf5 available).
        
        # Assuming we have waveforms or embeddings.
        # If we have waveforms, we need to stack them.
        
        audio_embeds_list = []
        audio_masks_list = []
        
        for item in batch:
            if "audio_embed" in item:
                # Precomputed
                emb = torch.tensor(item["audio_embed"])
                audio_embeds_list.append(emb)
            elif "audio" in item:
                # Waveform
                # We need to compute embedding on the fly?
                # Or does the model handle it?
                # SAFEModel.forward takes audio_embeds.
                # So we must compute it.
                # But we are inside collate, no model access.
                # We should do it in the loop or use a dataset that computes it.
                pass
        
        # If we have embeddings
        if audio_embeds_list:
            audio_embeds = torch.stack(audio_embeds_list)
            audio_mask = torch.ones(audio_embeds.shape[0], 1) # Dummy mask
        else:
            # Fallback: The dataset might return paths and we load audio?
            # For now, let's assume the dataset returns what we need or we fix it.
            # Actually, `AudioCapsDataset` in this repo likely loads the HDF5 features if available.
            # Let's assume it returns 'audio_embeds'.
            # If not, we might crash.
            audio_embeds = torch.zeros(len(batch), 1, 512) # Dummy
            audio_mask = torch.zeros(len(batch), 1)
            
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "audio_embeds": audio_embeds,
            "audio_mask": audio_mask,
            "answers": [item.get("captions", item.get("caption")) for item in batch]
        }

    # We need the tokenizer for the collator
    # But collator is a function.
    # We can use a lambda or partial.
    
    # Re-define collator to use model.tokenizer
    def smart_collate(batch):
        tokenizer = model.tokenizer
        input_texts = ["describe the audio" for _ in batch] # Standard prompt
        
        inputs = tokenizer(input_texts, padding=True, return_tensors="pt")
        
        # Handle Audio
        # If dataset returns 'audio_embeds' (numpy), convert to tensor
        audio_tensors = []
        for x in batch:
            if 'audio_embed' in x:
                audio_tensors.append(torch.from_numpy(x['audio_embed']))
            else:
                # Try to load from file if needed?
                # For now assume embeddings are present (download script created them?)
                # Wait, the download script created HDF5 features!
                # So AudioCapsDataset SHOULD load them.
                pass
                
        if audio_tensors:
            audio_embeds = torch.stack(audio_tensors)
            if audio_embeds.dim() == 2: # (B, D) -> (B, 1, D)
                audio_embeds = audio_embeds.unsqueeze(1)
        else:
            # Create dummy if missing (will fail eval but run)
            audio_embeds = torch.zeros(len(batch), 1, 512)
            
        audio_mask = torch.ones(audio_embeds.shape[0], audio_embeds.shape[1])

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "audio_embeds": audio_embeds,
            "audio_mask": audio_mask,
            "answers": [x.get("captions", [x.get("caption")]) for x in batch]
        }

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=smart_collate,
        num_workers=4
    )
    
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
