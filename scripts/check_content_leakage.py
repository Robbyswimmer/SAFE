import json
import os
import argparse
from pathlib import Path
from collections import defaultdict

def load_captions_and_filenames(file_path):
    """
    Load all captions and audio filenames from a JSON/JSONL file.
    Returns:
        captions: Set of normalized caption strings
        filenames: Set of normalized audio filenames (stems)
        samples: List of raw sample dicts for debugging
    """
    captions = set()
    filenames = set()
    samples = []
    
    path = Path(file_path)
    if not path.exists():
        print(f"Warning: File not found: {path}")
        return captions, filenames, samples

    print(f"Loading {path.name}...")
    
    data = []
    if path.suffix == '.jsonl':
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(path, 'r') as f:
            content = json.load(f)
            if isinstance(content, dict) and 'data' in content:
                data = content['data']
            else:
                data = content

    for item in data:
        # Extract Captions
        caps = []
        if 'captions' in item:
            caps.extend(item['captions'])
        elif 'caption' in item:
            caps.append(item['caption'])
        elif 'answers' in item:
            if isinstance(item['answers'], list):
                caps.extend(item['answers'])
            else:
                caps.append(item['answers'])
        elif 'answer' in item:
            if isinstance(item['answer'], list):
                caps.extend(item['answer'])
            else:
                caps.append(item['answer'])
        
        # Normalize and add
        for c in caps:
            if isinstance(c, str):
                captions.add(c.strip().lower())
        
        # Extract Filename Stem
        fpath = item.get('file_path') or item.get('audio_path') or item.get('audio')
        sname = item.get('sound_name') or item.get('id') or item.get('ytid')
        
        fname = None
        if fpath:
            fname = Path(fpath).stem
        elif sname:
            fname = Path(sname).stem
            
        if fname:
            # Remove timestamp suffix if present (common in AudioCaps)
            # e.g. "rqu8iB22I_Y_5000" -> "rqu8iB22I_Y"
            import re
            fname_clean = re.sub(r'_\d+$', '', fname)
            filenames.add(fname_clean)
            
        samples.append(item)
        
    return captions, filenames, samples

def main():
    parser = argparse.ArgumentParser(description="Check for content leakage between Train and Val")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to audiocaps data directory")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Define files to check
    train_file = data_dir / "audiocaps_train.json" # Or whatever the user's train file is
    val_file = data_dir / "audiocaps_val.json"     # The fixed val file
    
    # Fallback for train file names
    if not train_file.exists():
        candidates = ["train.json", "AudioCaps_train.json", "train.jsonl"]
        for c in candidates:
            if (data_dir / c).exists():
                train_file = data_dir / c
                break
    
    print(f"Checking Train: {train_file}")
    print(f"Checking Val:   {val_file}")
    
    train_caps, train_files, _ = load_captions_and_filenames(train_file)
    val_caps, val_files, val_samples = load_captions_and_filenames(val_file)
    
    print(f"\nStats:")
    print(f"Train: {len(train_caps)} unique captions, {len(train_files)} unique audio files")
    print(f"Val:   {len(val_caps)} unique captions, {len(val_files)} unique audio files")
    
    # Check Overlaps
    print(f"\n--- LEAKAGE REPORT ---")
    
    # 1. Caption Overlap
    cap_overlap = train_caps.intersection(val_caps)
    print(f"Caption Overlap: {len(cap_overlap)} strings found in both Train and Val")
    if cap_overlap:
        print("Examples:")
        for i, c in enumerate(list(cap_overlap)[:5]):
            print(f"  - '{c}'")
            
    # 2. Audio Filename Overlap
    file_overlap = train_files.intersection(val_files)
    print(f"Audio File Overlap: {len(file_overlap)} filenames found in both Train and Val")
    if file_overlap:
        print("Examples:")
        for i, f in enumerate(list(file_overlap)[:5]):
            print(f"  - {f}")
            
    if len(cap_overlap) > 0 or len(file_overlap) > 0:
        print("\n🚨 CONCLUSION: DATA LEAKAGE DETECTED!")
        print("The model is seeing validation data during training.")
    else:
        print("\n✅ CONCLUSION: No direct content leakage found.")

if __name__ == "__main__":
    main()
