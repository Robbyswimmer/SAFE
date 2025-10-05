#!/usr/bin/env python3
"""
Extract CLAP audio features from WAV files and store in HDF5 format.

This script:
1. Loads existing WAV files from data/audiocaps/audio/
2. Extracts CLAP embeddings using the frozen CLAP encoder
3. Saves embeddings to HDF5 file (space-efficient)
4. Logs all source YouTube IDs and timestamps
5. DELETES original WAV files after successful extraction

Usage:
    python scripts/extract_audio_features.py --split train --delete-wavs
    python scripts/extract_audio_features.py --split val --dry-run

Ethical Note:
    This implements fair use by storing only derived features (embeddings),
    not the original audio. Raw WAV files are deleted after processing.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from safe.models.audio_encoders import CLAPAudioEncoder


def compute_checksum(data: np.ndarray) -> str:
    """Compute SHA256 checksum of embedding data."""
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


def load_metadata(csv_path: Path) -> List[Dict]:
    """Load AudioCaps metadata CSV."""
    metadata = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append({
                'audiocap_id': row['audiocap_id'],
                'youtube_id': row['youtube_id'],
                'start_time': int(row['start_time']),
                'caption': row['caption']
            })
    return metadata


def extract_clap_embedding(
    audio_path: Path,
    encoder: CLAPAudioEncoder,
    device: str = 'cuda'
) -> Tuple[np.ndarray, bool]:
    """
    Extract CLAP embedding from audio file.

    Returns:
        (embedding, success): embedding as numpy array, success flag
    """
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Move to device
        waveform = waveform.to(device)

        # Extract embedding
        with torch.no_grad():
            embedding = encoder(waveform)  # (1, embed_dim)

        # Convert to numpy
        embedding_np = embedding.cpu().numpy().squeeze()  # (embed_dim,)

        return embedding_np, True

    except Exception as e:
        print(f"  ⚠️  Failed to extract from {audio_path.name}: {e}")
        return np.zeros(512, dtype=np.float32), False


def main():
    parser = argparse.ArgumentParser(description='Extract CLAP features from AudioCaps WAV files')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train',
                        help='Dataset split to process')
    parser.add_argument('--data-root', type=str, default='experiments/full_training/data/audiocaps',
                        help='Root directory for AudioCaps data')
    parser.add_argument('--delete-wavs', action='store_true',
                        help='Delete WAV files after successful extraction')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run without deleting files or saving (test mode)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for CLAP encoder')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Currently only batch_size=1 supported')

    args = parser.parse_args()

    # Setup paths
    data_root = Path(args.data_root).expanduser().resolve()
    audio_dir = data_root / 'audio' / args.split
    metadata_path = data_root / f'{args.split}.csv'  # CSV is directly in data_root
    features_dir = data_root / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)

    h5_path = features_dir / f'{args.split}_clap_embeddings.h5'
    sources_path = features_dir / f'{args.split}_sources.json'

    # Validate paths
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        return 1

    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        return 1

    # Load metadata
    print(f"📋 Loading metadata from {metadata_path}")
    metadata = load_metadata(metadata_path)
    print(f"   Found {len(metadata)} entries in metadata")

    # Find existing WAV files
    wav_files = list(audio_dir.glob('*.wav'))
    print(f"🎵 Found {len(wav_files)} WAV files in {audio_dir}")

    if args.dry_run:
        print("🧪 DRY RUN MODE - no files will be deleted or saved")

    # Initialize CLAP encoder
    print(f"🔧 Initializing CLAP encoder on {args.device}...")
    encoder = CLAPAudioEncoder(freeze=True)
    encoder = encoder.to(args.device)
    encoder.eval()
    print(f"   CLAP embedding dimension: {encoder.audio_embed_dim}")

    # Create mapping from YouTube ID to metadata
    id_to_metadata = {entry['youtube_id']: entry for entry in metadata}

    # Process files
    embeddings_dict = {}
    sources_dict = {}
    failed_files = []
    deleted_count = 0

    print(f"\n🔄 Processing {len(wav_files)} files...")

    for wav_path in tqdm(wav_files, desc="Extracting embeddings"):
        youtube_id = wav_path.stem  # Filename is YouTube ID

        # Extract embedding
        embedding, success = extract_clap_embedding(wav_path, encoder, args.device)

        if not success:
            failed_files.append(wav_path)
            continue

        # Get metadata if available
        meta = id_to_metadata.get(youtube_id, {})

        # Store embedding
        embeddings_dict[youtube_id] = embedding

        # Store source info
        sources_dict[youtube_id] = {
            'youtube_id': youtube_id,
            'audiocap_id': meta.get('audiocap_id', 'unknown'),
            'start_time': meta.get('start_time', -1),
            'caption': meta.get('caption', ''),
            'wav_filename': wav_path.name,
            'checksum': compute_checksum(embedding)
        }

    print(f"\n✅ Successfully extracted {len(embeddings_dict)} embeddings")
    print(f"⚠️  Failed: {len(failed_files)} files")

    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files[:10]:
            print(f"  - {f.name}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    # Save to HDF5
    if not args.dry_run and embeddings_dict:
        print(f"\n💾 Saving embeddings to {h5_path}")

        with h5py.File(h5_path, 'w') as hf:
            # Create datasets
            embedding_dim = encoder.audio_embed_dim
            num_samples = len(embeddings_dict)

            # Store embeddings
            embeddings_array = np.stack([embeddings_dict[k] for k in sorted(embeddings_dict.keys())])
            hf.create_dataset('embeddings', data=embeddings_array, compression='gzip')

            # Store YouTube IDs as index
            youtube_ids = sorted(embeddings_dict.keys())
            hf.create_dataset('youtube_ids', data=np.array(youtube_ids, dtype='S11'))

            # Store metadata as attributes
            hf.attrs['split'] = args.split
            hf.attrs['num_samples'] = num_samples
            hf.attrs['embedding_dim'] = embedding_dim
            hf.attrs['encoder'] = 'CLAP'
            hf.attrs['encoder_model'] = 'laion/clap-htsat-unfused'

        file_size_mb = h5_path.stat().st_size / (1024 * 1024)
        print(f"   Saved {num_samples} embeddings ({embedding_dim}D) - {file_size_mb:.2f} MB")

        # Save sources
        print(f"💾 Saving source log to {sources_path}")
        with open(sources_path, 'w') as f:
            json.dump(sources_dict, f, indent=2)
        print(f"   Logged {len(sources_dict)} sources")

    # Delete WAV files
    if args.delete_wavs and not args.dry_run:
        print(f"\n🗑️  Deleting WAV files...")

        # Only delete files that were successfully processed
        for youtube_id in embeddings_dict.keys():
            wav_path = audio_dir / f"{youtube_id}.wav"
            if wav_path.exists():
                try:
                    wav_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"  ⚠️  Failed to delete {wav_path.name}: {e}")

        print(f"   Deleted {deleted_count}/{len(embeddings_dict)} WAV files")

        # Calculate space savings
        if deleted_count > 0:
            # Estimate average file size from remaining files
            remaining_wavs = list(audio_dir.glob('*.wav'))
            if remaining_wavs:
                avg_size_mb = sum(f.stat().st_size for f in remaining_wavs[:100]) / len(remaining_wavs[:100]) / (1024 * 1024)
            else:
                avg_size_mb = 44  # Estimated average

            space_freed_gb = (deleted_count * avg_size_mb) / 1024
            print(f"   💰 Freed approximately {space_freed_gb:.2f} GB of storage")

    # Summary
    print(f"\n" + "="*60)
    print(f"SUMMARY - {args.split.upper()} split")
    print(f"="*60)
    print(f"Total files processed:  {len(wav_files)}")
    print(f"Successful extractions: {len(embeddings_dict)}")
    print(f"Failed extractions:     {len(failed_files)}")
    if not args.dry_run:
        print(f"HDF5 file:              {h5_path}")
        print(f"Sources log:            {sources_path}")
    if args.delete_wavs and not args.dry_run:
        print(f"WAV files deleted:      {deleted_count}")
    print(f"="*60)

    if args.dry_run:
        print("\n🧪 DRY RUN COMPLETE - no files were modified")
    else:
        print("\n✅ COMPLETE - features extracted and saved")
        if args.delete_wavs:
            print("   ⚠️  Original WAV files have been deleted")
            print("   ✓  CLAP embeddings preserved in HDF5 format")

    return 0


if __name__ == '__main__':
    sys.exit(main())
