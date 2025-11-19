#!/usr/bin/env python3
"""
Convert AudioSetCaps CSV files to JSONL format for training.

Reads the three CSV files from HuggingFace AudioSetCaps dataset:
- AudioSetCaps_caption.csv (original AudioSet)
- VGGSound_AudioSetCaps_caption.csv (VGGSound subset)
- YouTube-8M_AudioSetCaps_caption.csv (YouTube-8M subset)

Converts to JSONL with format:
{"id": "youtube_id", "question": "What is happening in the audio?", "answer": "caption_text"}
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def parse_audiosetcaps_id(row_id: str) -> str:
    """
    Parse YouTube ID from AudioSetCaps CSV ID field.

    Three formats:
    1. AudioSetCaps: Y{youtube_id} (e.g., Y---1_cCGK4M) -> remove Y prefix
    2. VGGSound: {youtube_id}_{timestamp} (e.g., OxPnZzn1_L8_000883) -> keep as-is
    3. YouTube-8M: {youtube_id}_{start}_{duration} (e.g., 9eUrEyAR3xk_84_10) -> keep as-is
    """
    if row_id.startswith('Y'):
        return row_id[1:]  # Remove Y prefix
    return row_id


def convert_csv_to_jsonl(csv_path: Path, output_path: Path, question: str = "What is happening in the audio?"):
    """Convert single CSV file to JSONL format."""
    df = pd.read_csv(csv_path)

    print(f"Processing {csv_path.name}: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    entries = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Converting {csv_path.name}"):
        youtube_id = parse_audiosetcaps_id(str(row['id']))
        caption = str(row['caption'])

        entry = {
            "id": youtube_id,
            "question": question,
            "answer": caption
        }
        entries.append(entry)

    # Write to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    print(f"✓ Wrote {len(entries)} entries to {output_path}")
    return len(entries)


def main():
    parser = argparse.ArgumentParser(
        description="Convert AudioSetCaps CSV files to JSONL format"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/audiosetcaps"),
        help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--download-csvs",
        action="store_true",
        help="Download CSV files from HuggingFace if not present"
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        help="Directory containing CSV files (alternative to downloading)"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is happening in the audio?",
        help="Question to use for all entries"
    )

    args = parser.parse_args()

    csv_files = [
        "AudioSetCaps_caption.csv",
        "VGGSound_AudioSetCaps_caption.csv",
        "YouTube-8M_AudioSetCaps_caption.csv",
    ]

    total_entries = 0

    for csv_file in csv_files:
        # Determine CSV path
        if args.csv_dir:
            csv_path = args.csv_dir / csv_file
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found, skipping")
                continue
        else:
            # Download from HuggingFace
            print(f"Downloading {csv_file} from HuggingFace...")
            csv_path = Path(hf_hub_download(
                'baijs/AudioSetCaps',
                filename=f'Dataset/{csv_file}',
                repo_type='dataset'
            ))

        # Determine output filename
        if csv_file == "AudioSetCaps_caption.csv":
            output_name = "audiosetcaps_train.jsonl"
        elif csv_file == "VGGSound_AudioSetCaps_caption.csv":
            output_name = "vggsound_audiosetcaps_train.jsonl"
        else:  # YouTube-8M
            output_name = "youtube8m_audiosetcaps_train.jsonl"

        output_path = args.output_dir / output_name

        # Convert
        count = convert_csv_to_jsonl(csv_path, output_path, args.question)
        total_entries += count

    print("\n" + "="*70)
    print(f"✓ Conversion complete!")
    print(f"Total entries: {total_entries:,}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
