import json
import argparse
from pathlib import Path
import os

def check_files(metadata_file, audio_dir):
    print(f"Checking audio files for {metadata_file} in {audio_dir}...")
    
    missing = 0
    found = 0
    total = 0
    
    audio_path = Path(audio_dir)
    
    with open(metadata_file, 'r') as f:
        # Handle JSONL
        if metadata_file.endswith('.jsonl'):
            for line in f:
                if line.strip():
                    total += 1
                    item = json.loads(line)
                    # Try to find the file
                    # Filename usually derived from youtube_id or file_path
                    fname = None
                    if 'file_path' in item:
                        fname = Path(item['file_path']).name
                    elif 'audio_path' in item:
                        fname = Path(item['audio_path']).name
                    elif 'youtube_id' in item:
                        fname = f"{item['youtube_id']}.wav" # Assumption
                    elif 'id' in item:
                         fname = f"{item['id']}.wav" # Assumption

                    if fname:
                        # Check various extensions
                        if (audio_path / fname).exists():
                            found += 1
                            continue
                        
                        # Try with .flac or .mp3
                        stem = Path(fname).stem
                        if (audio_path / f"{stem}.flac").exists():
                            found += 1
                            continue
                        if (audio_path / f"{stem}.mp3").exists():
                            found += 1
                            continue
                            
                        # Try recursive search if flat check fails (slow but thorough)
                        # Actually, let's just check the specific path if provided
                        if 'file_path' in item and (audio_path / item['file_path']).exists():
                             found += 1
                             continue
                             
                        missing += 1
                        if missing <= 5:
                            print(f"Missing: {fname}")
                    else:
                        print(f"Warning: Could not determine filename for item: {item}")
                        
    print(f"Total: {total}")
    print(f"Found: {found}")
    print(f"Missing: {missing}")
    
    if missing == 0 and total > 0:
        print("✅ All test files are present.")
    elif found > 0:
        print(f"⚠️  Partial data: {found}/{total} files present.")
    else:
        print("❌ No test files found.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--audio_dir', required=True)
    args = parser.parse_args()
    
    check_files(args.metadata, args.audio_dir)

if __name__ == "__main__":
    main()
