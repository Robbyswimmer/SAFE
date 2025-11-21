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
    ytids = set()
    samples = []
    
    path = Path(file_path)
    if not path.exists():
        print(f"Warning: File not found: {path}")
        return captions, filenames, ytids, samples

    print(f"Loading {path.name}...")
    
    data = []
    if path.suffix == '.jsonl':
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif path.suffix == '.csv':
        import csv
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # CSV format: id, youtube_id, start_time, caption
                if len(row) >= 4:
                    data.append({
                        "id": row[0],
                        "youtube_id": row[1],
                        "caption": row[3]
                    })
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
            
        # Extract YouTube ID
        ytid = item.get('youtube_id') or item.get('ytid')
        
        # Check metadata dictionary (common in AudioCaps)
        if not ytid and 'metadata' in item and isinstance(item['metadata'], dict):
            ytid = item['metadata'].get('youtube_id') or item['metadata'].get('ytid')

        # Check wavcaps_id (e.g. "Yb0RFKhbpFJA.wav")
        if not ytid and 'wavcaps_id' in item:
            val = item['wavcaps_id']
            if isinstance(val, str):
                # Remove .wav or .flac extension
                import re
                val = re.sub(r'\.(wav|flac|mp3)$', '', val, flags=re.IGNORECASE)
                val = val.strip()
                
                # Handle 'Y' prefix (common in AudioSet/WavCaps)
                if len(val) == 12 and val.startswith('Y'):
                    val = val[1:]
                    
                if len(val) == 11:
                    ytid = val
                elif len(samples) < 5: # Debug print for first few failures
                    print(f"DEBUG: Rejected wavcaps_id: '{val}' (len {len(val)})")

        if not ytid and 'id' in item:
            # Sometimes ID is the YTID
            val = item['id']
            if isinstance(val, str):
                val = val.strip()
                if len(val) == 11: # Basic YTID check
                    ytid = val

        # Fallback: Use filename as ID if it looks like one
        if not ytid and fname:
            fname_clean = fname.strip()
            # Remove 'Y' prefix if present in filename too
            if len(fname_clean) == 12 and fname_clean.startswith('Y'):
                fname_clean = fname_clean[1:]
            
            if len(fname_clean) == 11:
                ytid = fname_clean

        if ytid:
            ytids.add(ytid.strip())

        samples.append(item)
        
    if len(samples) > 0:
        print(f"DEBUG: First item keys: {list(samples[0].keys())}")
        if 'metadata' in samples[0]:
            print(f"DEBUG: First item metadata: {samples[0]['metadata']}")
        
    return captions, filenames, ytids, samples

def main():
    parser = argparse.ArgumentParser(description="Check for content leakage between Train and Val")
    parser.add_argument('--train_file', type=str, required=True, help="Path to training metadata file")
    parser.add_argument('--val_file', type=str, required=True, help="Path to validation metadata file")
    args = parser.parse_args()
    
    train_file = Path(args.train_file)
    val_file = Path(args.val_file)
    
    if not train_file.exists():
        print(f"Error: Train file not found: {train_file}")
        return
    if not val_file.exists():
        print(f"Error: Val file not found: {val_file}")
        return
    
    print(f"Checking Train: {train_file}")
    print(f"Checking Val:   {val_file}")
    
    train_caps, train_files, train_ytids, _ = load_captions_and_filenames(train_file)
    val_caps, val_files, val_ytids, val_samples = load_captions_and_filenames(val_file)
    
    print(f"\nStats:")
    print(f"Train: {len(train_caps)} unique captions, {len(train_files)} unique audio files, {len(train_ytids)} unique YTIDs")
    print(f"Val:   {len(val_caps)} unique captions, {len(val_files)} unique audio files, {len(val_ytids)} unique YTIDs")
    
    print("\n--- DEBUG: First 5 YTIDs ---")
    print(f"Train: {list(train_ytids)[:5]}")
    print(f"Val:   {list(val_ytids)[:5]}")
    
    # Check Overlaps
    print(f"\n--- LEAKAGE REPORT ---")
    
    # 1. Caption Overlap
    cap_overlap = train_caps.intersection(val_caps)
    print(f"Caption Overlap: {len(cap_overlap)} strings found in both Train and Val")
    if cap_overlap:
        print("Examples:")
        for i, c in enumerate(list(cap_overlap)[:5]):
            print(f"  - '{c}'")
            
    # 2. YouTube ID Overlap
    ytid_overlap = train_ytids.intersection(val_ytids)
    print(f"YouTube ID Overlap: {len(ytid_overlap)} IDs found in both Train and Val")
    if ytid_overlap:
        print("Examples:")
        for i, y in enumerate(list(ytid_overlap)[:5]):
            print(f"  - {y}")

    # 3. Audio Filename Overlap
    file_overlap = train_files.intersection(val_files)
    print(f"Audio File Overlap: {len(file_overlap)} filenames found in both Train and Val")
    if file_overlap:
        print("Examples:")
        for i, f in enumerate(list(file_overlap)[:5]):
            print(f"  - {f}")
            
    if len(cap_overlap) > 0 or len(file_overlap) > 0 or len(ytid_overlap) > 0:
        print("\n🚨 CONCLUSION: DATA LEAKAGE DETECTED!")
        if len(ytid_overlap) > 0:
            print("CRITICAL: The same YouTube videos are in both Train and Val.")
        elif len(cap_overlap) > 0:
            print("Warning: Caption overlap detected. If audio is different, this might be benign.")
    else:
        print("\n✅ CONCLUSION: No direct content leakage found.")

if __name__ == "__main__":
    main()
