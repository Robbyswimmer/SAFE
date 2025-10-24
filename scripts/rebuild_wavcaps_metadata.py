#!/usr/bin/env python3
"""
Rebuild WavCaps metadata by matching audio file IDs to JSON captions.
Handles the case where JSON has 'wav_path' placeholders but correct IDs and captions.
"""

import json
from pathlib import Path
import argparse


def load_caption_map(json_path):
    """Load JSON and create ID -> metadata map."""
    print(f"Loading {json_path}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Handle both list and dict formats
    if isinstance(data, dict) and 'data' in data:
        samples = data['data']
    else:
        samples = data if isinstance(data, list) else []

    # Build ID -> metadata map
    caption_map = {}
    for sample in samples:
        file_id = str(sample.get('id', ''))
        if file_id:
            caption_map[file_id] = {
                'caption': sample.get('caption', ''),
                'description': sample.get('description', ''),
                'duration': sample.get('duration'),
                'tags': sample.get('tags', []),
            }

    print(f"  Loaded {len(caption_map)} captions")
    return caption_map


def scan_audio_files(audio_dir, subset_name):
    """Scan audio directory and return list of (file_id, file_path) tuples."""
    print(f"Scanning {audio_dir} for {subset_name}...")

    if not audio_dir.exists():
        print(f"  Warning: Directory not found: {audio_dir}")
        return []

    audio_files = []
    for ext in ['.flac', '.wav', '.mp3', '.ogg']:
        audio_files.extend(audio_dir.rglob(f'*{ext}'))

    # Filter by subset name in path
    subset_keywords = {
        'FreeSound': ['freesound'],
        'BBC_Sound_Effects': ['bbc', 'sound_effects'],
    }

    keywords = subset_keywords.get(subset_name, [subset_name.lower()])

    # Extract ID from filename and filter by path
    id_path_pairs = []
    for file_path in audio_files:
        path_lower = str(file_path).lower()
        # Check if path contains subset keywords
        if any(kw in path_lower for kw in keywords):
            file_id = file_path.stem  # filename without extension
            id_path_pairs.append((file_id, file_path))

    print(f"  Found {len(id_path_pairs)} {subset_name} audio files")
    return id_path_pairs


def build_jsonl_entries(id_path_pairs, caption_map, subset_name, data_root):
    """Build JSONL entries by matching IDs to captions."""
    entries = []
    missing_captions = 0

    for idx, (file_id, file_path) in enumerate(id_path_pairs):
        # Get caption metadata
        metadata = caption_map.get(file_id, {})
        caption = metadata.get('caption', '')

        if not caption:
            missing_captions += 1

        # Build relative path from data root
        try:
            relative_path = file_path.relative_to(data_root)
        except ValueError:
            # File outside data_root, use as-is
            relative_path = file_path

        entry = {
            'id': f"{subset_name}_{idx:06d}",
            'question': 'What is happening in the audio?',
            'answer': caption,
            'audio': str(relative_path),
            'audio_path': str(relative_path),
            'subset': subset_name,
            'wavcaps_id': file_id,
        }

        # Add optional fields
        if metadata.get('duration'):
            entry['duration'] = metadata['duration']

        entries.append(entry)

    if missing_captions > 0:
        print(f"  Warning: {missing_captions} files without captions")

    return entries


def main():
    parser = argparse.ArgumentParser(description='Rebuild WavCaps metadata from JSON and audio files')
    parser.add_argument('--wavcaps-dir', type=str, required=True, help='WavCaps directory')
    parser.add_argument('--data-root', type=str, help='Data root for relative paths (default: wavcaps-dir parent)')
    args = parser.parse_args()

    wavcaps_dir = Path(args.wavcaps_dir).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else wavcaps_dir.parent

    print(f"WavCaps directory: {wavcaps_dir}")
    print(f"Data root: {data_root}")
    print()

    all_entries = []

    # Process FreeSound
    freesound_json = wavcaps_dir / 'json_files/FreeSound/FreeSound.json'
    freesound_audio = wavcaps_dir / 'audio'

    if freesound_json.exists():
        caption_map = load_caption_map(freesound_json)
        id_path_pairs = scan_audio_files(freesound_audio, 'FreeSound')
        entries = build_jsonl_entries(id_path_pairs, caption_map, 'FreeSound', data_root)
        all_entries.extend(entries)

        # Save FreeSound-specific file
        output_file = wavcaps_dir / 'wavcaps_freesound.jsonl'
        with open(output_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        print(f"✓ Wrote {len(entries)} FreeSound entries to {output_file.name}\n")

    # Process BBC
    bbc_json = wavcaps_dir / 'json_files/BBC_Sound_Effects/bbc_final.json'
    if not bbc_json.exists():
        bbc_json = wavcaps_dir / 'json_files/BBC_Sound_Effects/BBC_Sound_Effects.json'

    bbc_audio = wavcaps_dir / 'audio'

    if bbc_json.exists():
        caption_map = load_caption_map(bbc_json)
        id_path_pairs = scan_audio_files(bbc_audio, 'BBC_Sound_Effects')
        entries = build_jsonl_entries(id_path_pairs, caption_map, 'BBC_Sound_Effects', data_root)
        all_entries.extend(entries)

        # Save BBC-specific file
        output_file = wavcaps_dir / 'wavcaps_bbc_sound_effects.jsonl'
        with open(output_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        print(f"✓ Wrote {len(entries)} BBC entries to {output_file.name}\n")

    # Write combined file
    if all_entries:
        output_file = wavcaps_dir / 'wavcaps_train.jsonl'
        with open(output_file, 'w') as f:
            for entry in all_entries:
                f.write(json.dumps(entry) + '\n')
        print(f"✓ Wrote {len(all_entries)} total entries to {output_file.name}")

        # Show sample
        print("\nSample entry:")
        print(json.dumps(all_entries[0], indent=2))


if __name__ == '__main__':
    main()
