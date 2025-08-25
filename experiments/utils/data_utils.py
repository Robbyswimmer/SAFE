"""
Common data loading utilities for experiments.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import librosa
from PIL import Image
import torch

def load_audiocaps_metadata(data_path: str, split: str) -> pd.DataFrame:
    """Load AudioCaps CSV metadata."""
    csv_path = os.path.join(data_path, "audiocaps", "metadata", f"{split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"AudioCaps {split} metadata not found: {csv_path}")
    return pd.read_csv(csv_path)

def load_vqa_metadata(data_path: str, split: str) -> Tuple[List[Dict], List[Dict]]:
    """Load VQA questions and annotations."""
    questions_file = os.path.join(data_path, "vqa", f"v2_OpenEnded_mscoco_{split}2014_questions.json")
    annotations_file = os.path.join(data_path, "vqa", f"v2_mscoco_{split}2014_annotations.json")
    
    questions, annotations = [], []
    
    if os.path.exists(questions_file):
        with open(questions_file, 'r') as f:
            questions = json.load(f).get('questions', [])
    
    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            annotations = json.load(f).get('annotations', [])
    
    return questions, annotations

def check_file_loadable(file_path: str, file_type: str = "auto") -> Dict[str, Any]:
    """
    Check if a file can be loaded successfully.
    
    Returns:
        Dict with 'loadable', 'error', 'metadata' keys
    """
    result = {
        'loadable': False,
        'error': None,
        'metadata': {}
    }
    
    if not os.path.exists(file_path):
        result['error'] = "File not found"
        return result
    
    try:
        # Auto-detect file type
        if file_type == "auto":
            ext = Path(file_path).suffix.lower()
            if ext in ['.wav', '.mp3', '.m4a', '.flac']:
                file_type = "audio"
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                file_type = "image"
            else:
                file_type = "unknown"
        
        if file_type == "audio":
            # Try loading audio
            audio, sr = librosa.load(file_path, sr=None)
            result['metadata'] = {
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'channels': 1 if audio.ndim == 1 else audio.shape[0],
                'samples': len(audio)
            }
            result['loadable'] = True
            
        elif file_type == "image":
            # Try loading image
            img = Image.open(file_path)
            result['metadata'] = {
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format
            }
            result['loadable'] = True
            
        else:
            # Just check if file is readable
            with open(file_path, 'rb') as f:
                f.read(1024)  # Read first 1KB
            result['loadable'] = True
            
    except Exception as e:
        result['error'] = str(e)
    
    return result

def get_dataset_stats(data_path: str) -> Dict[str, Any]:
    """Get overview statistics for all datasets."""
    stats = {
        'audiocaps': {},
        'vqa': {},
        'avqa': {}
    }
    
    # AudioCaps stats
    for split in ['train', 'val', 'test']:
        try:
            df = load_audiocaps_metadata(data_path, split)
            audio_dir = os.path.join(data_path, "audiocaps", "audio", split)
            audio_files = len(list(Path(audio_dir).glob("*.wav"))) if os.path.exists(audio_dir) else 0
            
            stats['audiocaps'][split] = {
                'metadata_samples': len(df),
                'audio_files': audio_files,
                'coverage': audio_files / len(df) if len(df) > 0 else 0
            }
        except Exception as e:
            stats['audiocaps'][split] = {'error': str(e)}
    
    # VQA stats
    for split in ['train', 'val']:
        try:
            questions, annotations = load_vqa_metadata(data_path, split)
            images_dir = os.path.join(data_path, "vqa", "images", f"{split}2014")
            image_files = len(list(Path(images_dir).glob("*.jpg"))) if os.path.exists(images_dir) else 0
            
            stats['vqa'][split] = {
                'questions': len(questions),
                'annotations': len(annotations),
                'image_files': image_files
            }
        except Exception as e:
            stats['vqa'][split] = {'error': str(e)}
    
    return stats

def sample_dataset_files(data_path: str, n_samples: int = 100) -> Dict[str, List[str]]:
    """Sample files from each dataset for validation."""
    samples = {
        'audiocaps_audio': [],
        'vqa_images': [],
        'avqa_videos': []
    }
    
    # Sample AudioCaps audio files
    for split in ['train', 'val', 'test']:
        audio_dir = Path(data_path) / "audiocaps" / "audio" / split
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.wav"))
            sample_size = min(n_samples // 3, len(audio_files))
            samples['audiocaps_audio'].extend(
                np.random.choice(audio_files, sample_size, replace=False).tolist()
            )
    
    # Sample VQA images
    for split in ['train', 'val']:
        images_dir = Path(data_path) / "vqa" / "images" / f"{split}2014"
        if images_dir.exists():
            image_files = list(images_dir.glob("*.jpg"))
            sample_size = min(n_samples // 2, len(image_files))
            if len(image_files) > 0:
                samples['vqa_images'].extend(
                    np.random.choice(image_files, sample_size, replace=False).tolist()
                )
    
    return samples

def calculate_audio_features(audio_path: str) -> Dict[str, float]:
    """Calculate audio features for analysis."""
    try:
        audio, sr = librosa.load(audio_path)
        
        # Basic features
        duration = len(audio) / sr
        rms = librosa.feature.rms(y=audio)[0]
        snr_estimate = np.mean(rms) / (np.std(rms) + 1e-8)  # Simple SNR estimate
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        return {
            'duration': duration,
            'snr_estimate': float(snr_estimate),
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
            'rms_mean': float(np.mean(rms)),
            'sample_rate': sr
        }
    except Exception as e:
        return {'error': str(e)}