import json
import argparse
import csv
from pathlib import Path

def load_train_captions(file_path):
    captions = set()
    print(f"Loading training captions from {file_path}...")
    path = Path(file_path)
    
    if path.suffix == '.csv':
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 4:
                    cap = row[3].strip().lower()
                    captions.add(cap)
    else:
        # JSONL or JSON
        with open(path, 'r') as f:
            if path.suffix == '.jsonl':
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # Extract caption
                        caps = []
                        if 'caption' in item: caps.append(item['caption'])
                        if 'captions' in item: caps.extend(item['captions'])
                        for c in caps:
                            captions.add(c.strip().lower())
    return captions

def filter_val(val_file, train_captions):
    print(f"Filtering {val_file}...")
    with open(val_file, 'r') as f:
        content = json.load(f)
        
    data = content['data'] if isinstance(content, dict) and 'data' in content else content
    
    clean_data = []
    removed = 0
    
    for item in data:
        # Check if ANY of the validation captions exist in training
        is_overlap = False
        val_caps = item.get('captions', [])
        if isinstance(val_caps, str): val_caps = [val_caps]
        
        for cap in val_caps:
            if cap.strip().lower() in train_captions:
                is_overlap = True
                break
        
        if is_overlap:
            removed += 1
        else:
            clean_data.append(item)
            
    print(f"Original: {len(data)}")
    print(f"Removed:  {removed} (Exact caption matches)")
    print(f"Remaining: {len(clean_data)}")
    
    output_path = Path(val_file).with_name(f"{Path(val_file).stem}_no_overlap.json")
    
    # Preserve original structure
    output_content = {"data": clean_data} if isinstance(content, dict) and 'data' in content else clean_data
    
    with open(output_path, 'w') as f:
        json.dump(output_content, f, indent=2)
        
    print(f"Saved clean validation set to {output_path.name}")
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--val_file', required=True)
    args = parser.parse_args()
    
    train_caps = load_train_captions(args.train_file)
    filter_val(args.val_file, train_caps)

if __name__ == "__main__":
    main()
