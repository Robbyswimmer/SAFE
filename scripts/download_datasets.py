#!/usr/bin/env python3
"""
Download and setup real datasets for SAFE training.
This script downloads MUSIC-AVQA, AudioCaps, and VQA v2 datasets.
"""

import os
import sys
import requests
import zipfile
import tarfile
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import subprocess

def download_file(url, destination, description="Downloading"):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=description,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            pbar.update(size)
    
    print(f"✓ Downloaded: {destination}")

def extract_archive(archive_path, extract_to):
    """Extract zip or tar archive."""
    print(f"Extracting {archive_path}...")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
    
    print(f"✓ Extracted to: {extract_to}")

def download_music_avqa(data_dir):
    """Download MUSIC-AVQA dataset."""
    print("\n" + "="*60)
    print("DOWNLOADING MUSIC-AVQA DATASET")
    print("="*60)
    
    avqa_dir = data_dir / "avqa"
    avqa_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset URLs (these are example URLs - you'll need to update with actual URLs)
    urls = {
        "metadata": "https://github.com/gewu-lab/MUSIC-AVQA/releases/download/dataset/avqa_metadata.json",
        "videos_part1": "https://example.com/music-avqa-videos-part1.tar.gz",  # Replace with actual URL
        "videos_part2": "https://example.com/music-avqa-videos-part2.tar.gz",  # Replace with actual URL
    }
    
    try:
        # Download metadata
        print("1. Downloading MUSIC-AVQA metadata...")
        metadata_path = avqa_dir / "avqa_metadata.json"
        
        # For MUSIC-AVQA, you typically need to:
        print("MUSIC-AVQA requires manual download from:")
        print("https://github.com/gewu-lab/MUSIC-AVQA")
        print("\nPlease:")
        print("1. Visit the GitHub repository")
        print("2. Follow their download instructions")
        print("3. Extract videos to: ./data/avqa/videos/")
        print("4. Extract metadata to: ./data/avqa/metadata/")
        
        # Create placeholder structure
        (avqa_dir / "videos").mkdir(exist_ok=True)
        (avqa_dir / "metadata").mkdir(exist_ok=True)
        
        return False  # Indicate manual download needed
        
    except Exception as e:
        print(f"Error downloading MUSIC-AVQA: {e}")
        return False

def download_audiocaps(data_dir):
    """Download AudioCaps dataset."""
    print("\n" + "="*60)
    print("DOWNLOADING AUDIOCAPS DATASET") 
    print("="*60)
    
    audiocaps_dir = data_dir / "audiocaps"
    audiocaps_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # AudioCaps metadata URLs
        metadata_urls = {
            "train": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/train.csv",
            "val": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/val.csv",
            "test": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/test.csv"
        }
        
        print("1. Downloading AudioCaps metadata...")
        for split, url in metadata_urls.items():
            try:
                download_file(url, audiocaps_dir / f"{split}.csv", f"AudioCaps {split} metadata")
            except Exception as e:
                print(f"Warning: Could not download {split} metadata: {e}")
        
        # Audio files need to be downloaded from YouTube
        print("\n2. AudioCaps audio files:")
        print("AudioCaps audio files are sourced from YouTube.")
        print("You'll need to use youtube-dl or yt-dlp to download them.")
        print("\nExample command:")
        print("pip install yt-dlp")
        print("# Then use the YouTube IDs from the CSV files to download audio")
        
        # Create audio directory
        (audiocaps_dir / "audio").mkdir(exist_ok=True)
        
        return True
        
    except Exception as e:
        print(f"Error setting up AudioCaps: {e}")
        return False

def download_vqa_v2(data_dir):
    """Download VQA v2 dataset."""
    print("\n" + "="*60)
    print("DOWNLOADING VQA v2 DATASET")
    print("="*60)
    
    vqa_dir = data_dir / "vqa"
    vqa_dir.mkdir(parents=True, exist_ok=True)
    
    # VQA v2 URLs
    base_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa"
    
    files_to_download = {
        "questions": {
            "train": f"{base_url}/v2_Questions_Train_mscoco.zip",
            "val": f"{base_url}/v2_Questions_Val_mscoco.zip"
        },
        "annotations": {
            "train": f"{base_url}/v2_Annotations_Train_mscoco.zip", 
            "val": f"{base_url}/v2_Annotations_Val_mscoco.zip"
        }
    }
    
    try:
        print("1. Downloading VQA v2 questions and annotations...")
        
        for data_type, urls in files_to_download.items():
            for split, url in urls.items():
                filename = f"vqa_v2_{data_type}_{split}.zip"
                file_path = vqa_dir / filename
                
                if not file_path.exists():
                    download_file(url, file_path, f"VQA v2 {data_type} {split}")
                    extract_archive(str(file_path), str(vqa_dir))
                    # Clean up zip file
                    file_path.unlink()
                else:
                    print(f"✓ Already exists: {filename}")
        
        print("\n2. VQA v2 images:")
        print("VQA v2 uses COCO images. Download from:")
        print("http://cocodataset.org/#download")
        print("You need:")
        print("- 2014 Train images [83K/13GB]") 
        print("- 2014 Val images [41K/6GB]")
        print("Extract to: ./data/vqa/images/")
        
        # Create images directory
        (vqa_dir / "images").mkdir(exist_ok=True)
        
        return True
        
    except Exception as e:
        print(f"Error downloading VQA v2: {e}")
        return False

def create_dataset_structure(data_dir):
    """Create the expected dataset directory structure."""
    print("\n" + "="*60)
    print("CREATING DATASET DIRECTORY STRUCTURE")
    print("="*60)
    
    # Create main directories
    directories = [
        "avqa/videos",
        "avqa/metadata", 
        "audiocaps/audio",
        "audiocaps/metadata",
        "vqa/images/train2014",
        "vqa/images/val2014",
        "vqa/questions",
        "vqa/annotations"
    ]
    
    for dir_path in directories:
        full_path = data_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {full_path}")

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = ['requests', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Main download function."""
    print("SAFE Dataset Downloader")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Set up data directory
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    print(f"📁 Data directory: {data_dir.absolute()}")
    
    # Create directory structure
    create_dataset_structure(data_dir)
    
    # Track download results
    results = {}
    
    # Download datasets
    try:
        results['audiocaps'] = download_audiocaps(data_dir)
        results['vqa'] = download_vqa_v2(data_dir) 
        results['music_avqa'] = download_music_avqa(data_dir)
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        return
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    for dataset, success in results.items():
        status = "✓ Complete" if success else "⚠ Manual steps required"
        print(f"{dataset.upper():<15}: {status}")
    
    print(f"\n📁 All data saved to: {data_dir.absolute()}")
    
    # Final instructions
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Complete any manual downloads mentioned above")
    print("2. Verify dataset structure matches expected layout")
    print("3. Test with: python train_stage_a_curriculum.py --config demo --use-real-data")
    
    # Check current data status
    print(f"\n📊 Current data status:")
    for subdir in ["avqa", "audiocaps", "vqa"]:
        path = data_dir / subdir
        if path.exists():
            file_count = len(list(path.rglob("*"))) 
            print(f"  {subdir:<12}: {file_count} files")
        else:
            print(f"  {subdir:<12}: Not created")

if __name__ == "__main__":
    main()