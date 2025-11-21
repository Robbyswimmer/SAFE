import json
import argparse
from pathlib import Path
import re

def get_ytid(item):
    """Extract YouTube ID from a data item."""
    # 1. Explicit ID
    ytid = item.get('youtube_id') or item.get('ytid')
    
    # 2. Metadata dict
    if not ytid and 'metadata' in item and isinstance(item['metadata'], dict):
        ytid = item['metadata'].get('youtube_id') or item['metadata'].get('ytid')

    # 3. WavCaps ID (e.g. "Yb0RFKhbpFJA.wav")
    if not ytid and 'wavcaps_id' in item:
        val = item['wavcaps_id']
        if isinstance(val, str):
            val = re.sub(r'\.(wav|flac|mp3)$', '', val, flags=re.IGNORECASE).strip()
            if len(val) == 12 and val.startswith('Y'):
                val = val[1:]
            if len(val) == 11:
                ytid = val

    # 4. Generic ID field
    if not ytid and 'id' in item:
        val = item['id']
        if isinstance(val, str):
            val = val.strip()
            if len(val) == 11:
                ytid = val

    # 5. Filename fallback
    if not ytid:
        fpath = item.get('file_path') or item.get('audio_path') or item.get('audio')
        sname = item.get('sound_name')
        fname = None
        if fpath:
            fname = Path(fpath).stem
        elif sname:
            fname = Path(sname).stem
            
        if fname:
            fname = re.sub(r'_\d+$', '', fname).strip()
            if len(fname) == 12 and fname.startswith('Y'):
                fname = fname[1:]
            if len(fname) == 11:
                ytid = fname
                
    return ytid

def load_forbidden_ids(val_file):
    """Load all YouTube IDs from the validation file."""
    forbidden = set()
    path = Path(val_file)
    print(f"Loading forbidden IDs from {path.name}...")
    
    with open(path, 'r') as f:
        content = json.load(f)
        data = content['data'] if isinstance(content, dict) and 'data' in content else content
        
    for item in data:
        ytid = get_ytid(item)
        if ytid:
            forbidden.add(ytid)
            
    print(f"Found {len(forbidden)} forbidden IDs.")
    return forbidden

def filter_file(input_file, forbidden_ids, replace=True):
    """Filter a JSONL file, removing entries with forbidden IDs."""
    input_path = Path(input_file)
    # Create temp file for writing
    output_path = input_path.with_name(f"{input_path.stem}_clean{input_path.suffix}")
    
    print(f"Filtering {input_path.name}...")
    
    kept = 0
    removed = 0
    
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            
            try:
                item = json.loads(line)
                ytid = get_ytid(item)
                
                if ytid and ytid in forbidden_ids:
                    removed += 1
                    # print(f"  Removing {ytid}")
                else:
                    kept += 1
                    fout.write(line)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line in {input_path.name}")
                continue
                
    print(f"Done. Kept: {kept}, Removed: {removed}")
    
    if replace:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        if backup_path.exists():
            print(f"Warning: Backup {backup_path.name} already exists. Overwriting.")
        
        print(f"Backing up original to {backup_path.name}...")
        input_path.rename(backup_path)
        
        print(f"Replacing original {input_path.name} with clean version...")
        output_path.rename(input_path)
        return input_path
        
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_file', required=True, help="Path to validation file (source of forbidden IDs)")
    parser.add_argument('--target_files', nargs='+', required=True, help="List of WavCaps JSONL files to filter")
    parser.add_argument('--no-replace', action='store_true', help="Do not replace original files (create _clean copies instead)")
    args = parser.parse_args()
    
    forbidden = load_forbidden_ids(args.val_file)
    
    for target in args.target_files:
        filter_file(target, forbidden, replace=not args.no_replace)

if __name__ == "__main__":
    main()
