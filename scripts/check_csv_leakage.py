import csv
import json
import argparse
from pathlib import Path

def load_csv_captions(file_path):
    captions = set()
    print(f"Loading {file_path}...")
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Format seems to be: id, youtube_id, start_time, caption
            # We'll try to grab the last column or index 3
            if len(row) >= 4:
                cap = row[3]
                captions.add(cap.strip().lower())
    return captions

def load_json_captions(file_path):
    captions = set()
    print(f"Loading {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, dict) and 'data' in data:
            data = data['data']
        
        for item in data:
            # Handle various JSON formats
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
            
            for c in caps:
                if isinstance(c, str):
                    captions.add(c.strip().lower())
    return captions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_json', required=True)
    args = parser.parse_args()

    train_caps = load_csv_captions(args.train_csv)
    val_caps = load_json_captions(args.val_json)

    print(f"Stats:")
    print(f"Train (CSV): {len(train_caps)} unique captions")
    print(f"Val (JSON):  {len(val_caps)} unique captions")

    overlap = train_caps.intersection(val_caps)
    print(f"\n--- LEAKAGE REPORT ---")
    print(f"Overlap: {len(overlap)} exact matches found")
    
    if overlap:
        print("Examples:")
        for i, c in enumerate(list(overlap)[:5]):
            print(f"  - '{c}'")
        print("\n🚨 CONCLUSION: MASSIVE LEAKAGE CONFIRMED IN CSV")
    else:
        print("\n✅ No leakage found in CSV")

if __name__ == "__main__":
    main()
