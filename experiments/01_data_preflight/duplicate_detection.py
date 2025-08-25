"""
Cross-Dataset Duplicate Detection

Detects duplicates across datasets using multiple methods:
- Perceptual hash comparison for images
- Audio fingerprinting for audio files  
- Text similarity for captions/questions
- Metadata overlap detection
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict
import imagehash
from PIL import Image
import librosa
from scipy.spatial.distance import hamming

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.utils.data_utils import load_audiocaps_metadata, load_vqa_metadata, sample_dataset_files
from experiments.utils.validation_metrics import ValidationResult, check_duplicate_rate

class DuplicateDetector:
    """Detects duplicates across and within datasets."""
    
    def __init__(self, data_path: str, output_dir: str = "./experiments/reports"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
        # Hash storage
        self.image_hashes = {}  # {hash: [dataset, split, id, file_path]}
        self.audio_hashes = {}  # {hash: [dataset, split, id, file_path]}
        self.text_hashes = {}   # {hash: [dataset, split, id, text]}
    
    def calculate_image_hash(self, image_path: str) -> str:
        """Calculate perceptual hash for an image."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate average hash (good for detecting similar images)
                img_hash = imagehash.average_hash(img, hash_size=8)
                return str(img_hash)
        except Exception as e:
            return f"ERROR_{hash(str(e))}"
    
    def calculate_audio_fingerprint(self, audio_path: str) -> str:
        """Calculate audio fingerprint using spectral features."""
        try:
            # Load audio (first 30 seconds to avoid memory issues)
            y, sr = librosa.load(audio_path, duration=30.0, sr=22050)
            
            # Extract spectral features that are somewhat invariant to small changes
            # Use chromagram for harmonic content
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Use spectral centroid for timbral characteristics
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_mean = np.mean(spectral_centroids)
            
            # Use tempo for rhythmic characteristics
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Create a simple fingerprint by quantizing features
            chroma_quantized = (chroma_mean * 8).astype(int)  # 0-7 range
            spectral_quantized = int(spectral_mean / 500)     # Quantize spectral centroid
            tempo_quantized = int(tempo / 20)                 # Quantize tempo
            
            # Combine into a hash string
            fingerprint = f"{tempo_quantized:02d}_{''.join(map(str, chroma_quantized))[:12]}_{spectral_quantized:03d}"
            return fingerprint
            
        except Exception as e:
            return f"ERROR_{hash(str(e))}"
    
    def calculate_text_hash(self, text: str, fuzzy: bool = False) -> str:
        """Calculate text hash with optional fuzzy matching preparation."""
        if not text or not text.strip():
            return "EMPTY"
        
        # Normalize text
        normalized = text.lower().strip()
        
        if fuzzy:
            # Remove common words and punctuation for fuzzy matching
            import re
            # Remove punctuation and common words
            normalized = re.sub(r'[^\w\s]', '', normalized)
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            words = [w for w in normalized.split() if w not in common_words]
            normalized = ' '.join(sorted(words))  # Sort for order-invariant comparison
        
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def process_vqa_images(self) -> Dict[str, Any]:
        """Process VQA images for duplicate detection."""
        print("🖼️  Processing VQA images...")
        
        duplicates = defaultdict(list)
        processed_count = 0
        error_count = 0
        
        for split in ['train', 'val']:
            try:
                questions, annotations = load_vqa_metadata(self.data_path, split)
                images_dir = Path(self.data_path) / "vqa" / "images" / f"{split}2014"
                
                if not images_dir.exists():
                    print(f"    ⚠️  Images directory not found: {images_dir}")
                    continue
                
                # Get unique image IDs from questions/annotations
                image_ids = set()
                for q in questions:
                    if 'image_id' in q:
                        image_ids.add(q['image_id'])
                
                # Process sample of images (to avoid overwhelming computation)
                sample_ids = list(image_ids)[:500]  # Limit to 500 per split
                
                for image_id in sample_ids:
                    image_file = images_dir / f"COCO_{split}2014_{image_id:012d}.jpg"
                    
                    if image_file.exists():
                        img_hash = self.calculate_image_hash(str(image_file))
                        
                        if not img_hash.startswith("ERROR"):
                            hash_info = {
                                'dataset': 'vqa',
                                'split': split,
                                'id': image_id,
                                'file_path': str(image_file),
                                'hash': img_hash
                            }
                            
                            if img_hash in self.image_hashes:
                                # Found duplicate
                                duplicates[img_hash].append(hash_info)
                                if len(duplicates[img_hash]) == 1:
                                    # Add the original
                                    duplicates[img_hash].insert(0, self.image_hashes[img_hash])
                            else:
                                self.image_hashes[img_hash] = hash_info
                            
                            processed_count += 1
                        else:
                            error_count += 1
            
            except Exception as e:
                print(f"    ❌ Error processing {split}: {e}")
                error_count += 1
        
        duplicate_pairs = []
        for img_hash, duplicate_list in duplicates.items():
            if len(duplicate_list) > 1:
                # Convert to pairs for reporting
                for i in range(len(duplicate_list)):
                    for j in range(i + 1, len(duplicate_list)):
                        duplicate_pairs.append({
                            'hash': img_hash,
                            'item1': duplicate_list[i],
                            'item2': duplicate_list[j],
                            'type': 'image'
                        })
        
        return {
            'processed_count': processed_count,
            'error_count': error_count,
            'duplicate_hashes': len(duplicates),
            'duplicate_pairs': duplicate_pairs[:20]  # Limit stored pairs
        }
    
    def process_audiocaps_audio(self) -> Dict[str, Any]:
        """Process AudioCaps audio files for duplicate detection."""
        print("🎵 Processing AudioCaps audio files...")
        
        duplicates = defaultdict(list)
        processed_count = 0
        error_count = 0
        
        for split in ['train', 'val']:
            try:
                df = load_audiocaps_metadata(self.data_path, split)
                audio_dir = Path(self.data_path) / "audiocaps" / "audio" / split
                
                if not audio_dir.exists():
                    print(f"    ⚠️  Audio directory not found: {audio_dir}")
                    continue
                
                # Process sample of audio files
                sample_size = min(200, len(df))  # Limit to 200 per split
                sample_indices = np.random.choice(len(df), sample_size, replace=False)
                
                for idx in sample_indices:
                    row = df.iloc[idx]
                    youtube_id = row['youtube_id']
                    audio_file = audio_dir / f"{youtube_id}.wav"
                    
                    if audio_file.exists():
                        audio_hash = self.calculate_audio_fingerprint(str(audio_file))
                        
                        if not audio_hash.startswith("ERROR"):
                            hash_info = {
                                'dataset': 'audiocaps',
                                'split': split,
                                'id': youtube_id,
                                'file_path': str(audio_file),
                                'hash': audio_hash,
                                'caption': row.get('caption', '')
                            }
                            
                            if audio_hash in self.audio_hashes:
                                # Found duplicate
                                duplicates[audio_hash].append(hash_info)
                                if len(duplicates[audio_hash]) == 1:
                                    duplicates[audio_hash].insert(0, self.audio_hashes[audio_hash])
                            else:
                                self.audio_hashes[audio_hash] = hash_info
                            
                            processed_count += 1
                        else:
                            error_count += 1
            
            except Exception as e:
                print(f"    ❌ Error processing {split}: {e}")
                error_count += 1
        
        duplicate_pairs = []
        for audio_hash, duplicate_list in duplicates.items():
            if len(duplicate_list) > 1:
                for i in range(len(duplicate_list)):
                    for j in range(i + 1, len(duplicate_list)):
                        duplicate_pairs.append({
                            'hash': audio_hash,
                            'item1': duplicate_list[i],
                            'item2': duplicate_list[j],
                            'type': 'audio'
                        })
        
        return {
            'processed_count': processed_count,
            'error_count': error_count,
            'duplicate_hashes': len(duplicates),
            'duplicate_pairs': duplicate_pairs[:20]
        }
    
    def process_text_content(self) -> Dict[str, Any]:
        """Process text content (captions, questions) for duplicates."""
        print("📝 Processing text content...")
        
        exact_duplicates = defaultdict(list)
        fuzzy_duplicates = defaultdict(list)
        processed_count = 0
        
        # Process AudioCaps captions
        for split in ['train', 'val']:
            try:
                df = load_audiocaps_metadata(self.data_path, split)
                
                for _, row in df.iterrows():
                    caption = str(row.get('caption', '')).strip()
                    if caption:
                        # Exact hash
                        exact_hash = self.calculate_text_hash(caption, fuzzy=False)
                        # Fuzzy hash
                        fuzzy_hash = self.calculate_text_hash(caption, fuzzy=True)
                        
                        text_info = {
                            'dataset': 'audiocaps',
                            'split': split,
                            'id': row['youtube_id'],
                            'text': caption[:100],  # Truncate for storage
                            'text_type': 'caption'
                        }
                        
                        # Check exact duplicates
                        if exact_hash in self.text_hashes:
                            exact_duplicates[exact_hash].append(text_info)
                            if len(exact_duplicates[exact_hash]) == 1:
                                exact_duplicates[exact_hash].insert(0, self.text_hashes[exact_hash])
                        else:
                            self.text_hashes[exact_hash] = text_info
                        
                        # Check fuzzy duplicates (different from exact)
                        if fuzzy_hash != exact_hash:
                            fuzzy_key = f"fuzzy_{fuzzy_hash}"
                            if fuzzy_key in self.text_hashes:
                                fuzzy_duplicates[fuzzy_key].append(text_info)
                                if len(fuzzy_duplicates[fuzzy_key]) == 1:
                                    fuzzy_duplicates[fuzzy_key].insert(0, self.text_hashes[fuzzy_key])
                            else:
                                self.text_hashes[fuzzy_key] = text_info
                        
                        processed_count += 1
            
            except Exception as e:
                print(f"    ❌ Error processing AudioCaps {split}: {e}")
        
        # Process VQA questions
        for split in ['train', 'val']:
            try:
                questions, _ = load_vqa_metadata(self.data_path, split)
                
                for q in questions:
                    question_text = str(q.get('question', '')).strip()
                    if question_text:
                        exact_hash = self.calculate_text_hash(question_text, fuzzy=False)
                        fuzzy_hash = self.calculate_text_hash(question_text, fuzzy=True)
                        
                        text_info = {
                            'dataset': 'vqa',
                            'split': split,
                            'id': q.get('question_id', 'unknown'),
                            'text': question_text[:100],
                            'text_type': 'question'
                        }
                        
                        # Check exact duplicates
                        if exact_hash in self.text_hashes:
                            exact_duplicates[exact_hash].append(text_info)
                            if len(exact_duplicates[exact_hash]) == 1:
                                exact_duplicates[exact_hash].insert(0, self.text_hashes[exact_hash])
                        else:
                            self.text_hashes[exact_hash] = text_info
                        
                        # Check fuzzy duplicates
                        if fuzzy_hash != exact_hash:
                            fuzzy_key = f"fuzzy_{fuzzy_hash}"
                            if fuzzy_key in self.text_hashes:
                                fuzzy_duplicates[fuzzy_key].append(text_info)
                                if len(fuzzy_duplicates[fuzzy_key]) == 1:
                                    fuzzy_duplicates[fuzzy_key].insert(0, self.text_hashes[fuzzy_key])
                            else:
                                self.text_hashes[fuzzy_key] = text_info
                        
                        processed_count += 1
            
            except Exception as e:
                print(f"    ❌ Error processing VQA {split}: {e}")
        
        # Convert to pairs for reporting
        exact_pairs = []
        for text_hash, duplicate_list in exact_duplicates.items():
            if len(duplicate_list) > 1:
                for i in range(len(duplicate_list)):
                    for j in range(i + 1, len(duplicate_list)):
                        exact_pairs.append({
                            'hash': text_hash,
                            'item1': duplicate_list[i],
                            'item2': duplicate_list[j],
                            'type': 'text_exact'
                        })
        
        fuzzy_pairs = []
        for text_hash, duplicate_list in fuzzy_duplicates.items():
            if len(duplicate_list) > 1:
                for i in range(len(duplicate_list)):
                    for j in range(i + 1, len(duplicate_list)):
                        fuzzy_pairs.append({
                            'hash': text_hash,
                            'item1': duplicate_list[i],
                            'item2': duplicate_list[j],
                            'type': 'text_fuzzy'
                        })
        
        return {
            'processed_count': processed_count,
            'exact_duplicates': len(exact_duplicates),
            'fuzzy_duplicates': len(fuzzy_duplicates),
            'exact_pairs': exact_pairs[:10],  # Limit stored pairs
            'fuzzy_pairs': fuzzy_pairs[:10]
        }
    
    def run_full_detection(self) -> Dict[str, Any]:
        """Run complete duplicate detection."""
        print("🔍 Starting Cross-Dataset Duplicate Detection...")
        print("=" * 50)
        
        results = {
            'image_duplicates': {},
            'audio_duplicates': {},
            'text_duplicates': {},
            'validation_summary': {}
        }
        
        # Process different modalities
        results['image_duplicates'] = self.process_vqa_images()
        results['audio_duplicates'] = self.process_audiocaps_audio()
        results['text_duplicates'] = self.process_text_content()
        
        # Create validation results
        validations = []
        
        # Validate image duplicate rate
        img_results = results['image_duplicates']
        if img_results['processed_count'] > 0:
            img_duplicate_rate = len(img_results['duplicate_pairs']) / img_results['processed_count']
            img_validation = check_duplicate_rate({
                'total_samples': img_results['processed_count'],
                'duplicates_found': len(img_results['duplicate_pairs'])
            }, max_duplicate_rate=0.05)  # Allow up to 5% duplicates
            validations.append(img_validation)
        
        # Validate audio duplicate rate
        audio_results = results['audio_duplicates']
        if audio_results['processed_count'] > 0:
            audio_duplicate_rate = len(audio_results['duplicate_pairs']) / audio_results['processed_count']
            audio_validation = check_duplicate_rate({
                'total_samples': audio_results['processed_count'],
                'duplicates_found': len(audio_results['duplicate_pairs'])
            }, max_duplicate_rate=0.03)  # Allow up to 3% duplicates
            validations.append(audio_validation)
        
        # Validate text duplicate rate (exact matches only)
        text_results = results['text_duplicates']
        if text_results['processed_count'] > 0:
            text_duplicate_rate = len(text_results['exact_pairs']) / text_results['processed_count']
            text_validation = check_duplicate_rate({
                'total_samples': text_results['processed_count'],
                'duplicates_found': len(text_results['exact_pairs'])
            }, max_duplicate_rate=0.02)  # Allow up to 2% exact text duplicates
            validations.append(text_validation)
        
        # Overall summary
        passed_tests = sum(1 for v in validations if v.passed)
        total_tests = len(validations)
        
        results['validation_summary'] = {
            'overall_passed': passed_tests == total_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'validations': validations
        }
        
        # Save results
        with open(self.output_dir / 'duplicate_detection_results.json', 'w') as f:
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.results = results
        return results
    
    def _make_serializable(self, obj):
        """Convert ValidationResult objects to dicts for JSON serialization."""
        if isinstance(obj, ValidationResult):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def print_summary(self):
        """Print duplicate detection summary."""
        if not self.results:
            print("No detection results available. Run detection first.")
            return
        
        print("\n" + "=" * 50)
        print("DUPLICATE DETECTION SUMMARY")
        print("=" * 50)
        
        summary = self.results.get('validation_summary', {})
        if summary.get('overall_passed'):
            print("✅ Overall Status: DUPLICATE RATES ACCEPTABLE")
        else:
            print("❌ Overall Status: HIGH DUPLICATE RATES DETECTED")
        
        print(f"Pass Rate: {summary.get('pass_rate', 0):.1%}")
        
        for validation in summary.get('validations', []):
            print(f"  {validation}")
        
        # Modality-specific summaries
        modalities = [
            ('image_duplicates', '🖼️  Images'),
            ('audio_duplicates', '🎵 Audio'),
            ('text_duplicates', '📝 Text')
        ]
        
        for key, label in modalities:
            if key in self.results:
                results = self.results[key]
                processed = results.get('processed_count', 0)
                
                if key == 'text_duplicates':
                    exact_pairs = len(results.get('exact_pairs', []))
                    fuzzy_pairs = len(results.get('fuzzy_pairs', []))
                    print(f"\n{label}: {processed} processed")
                    print(f"  Exact duplicates: {exact_pairs}")
                    print(f"  Fuzzy duplicates: {fuzzy_pairs}")
                else:
                    duplicate_pairs = len(results.get('duplicate_pairs', []))
                    duplicate_rate = duplicate_pairs / processed if processed > 0 else 0
                    print(f"\n{label}: {processed} processed, {duplicate_pairs} duplicates ({duplicate_rate:.1%})")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run cross-dataset duplicate detection")
    parser.add_argument("--data-path", required=True, help="Path to data directory")
    parser.add_argument("--output-dir", default="./experiments/reports", 
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Run detection
    detector = DuplicateDetector(args.data_path, args.output_dir)
    results = detector.run_full_detection()
    detector.print_summary()
    
    # Return exit code based on results
    if results['validation_summary']['overall_passed']:
        print("\n🎉 Duplicate detection passed!")
        return 0
    else:
        print("\n⚠️  Duplicate detection found high duplicate rates.")
        return 1

if __name__ == "__main__":
    exit(main())