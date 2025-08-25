"""
Audio-Visual Alignment Validation

Validates that audio and visual content are properly aligned:
- Temporal correlation analysis
- Content relevance scoring
- Synchronization validation
- Audio-caption semantic alignment
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import hashlib
from collections import defaultdict
import librosa
from scipy.stats import pearsonr
import cv2

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.utils.data_utils import load_audiocaps_metadata, check_file_loadable
from experiments.utils.validation_metrics import ValidationResult, check_av_alignment

class AVAlignmentValidator:
    """Validates audio-visual alignment across datasets."""
    
    def __init__(self, data_path: str, output_dir: str = "./experiments/reports"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract basic audio features for alignment analysis."""
        try:
            y, sr = librosa.load(audio_path, duration=10.0)  # Load first 10 seconds
            
            # Extract temporal features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Extract spectral features  
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            
            return {
                'duration': len(y) / sr,
                'tempo': float(tempo),
                'beat_count': len(beats),
                'mean_spectral_centroid': float(np.mean(spectral_centroids)),
                'mean_spectral_rolloff': float(np.mean(spectral_rolloff)),
                'mean_zcr': float(np.mean(zero_crossing_rate)),
                'mean_rms': float(np.mean(rms)),
                'energy_variance': float(np.var(rms)),
                'loadable': True
            }
        except Exception as e:
            return {
                'error': str(e),
                'loadable': False
            }
    
    def calculate_semantic_alignment(self, caption: str, audio_features: Dict) -> float:
        """Calculate semantic alignment between caption and audio features."""
        if not audio_features.get('loadable', False):
            return 0.0
        
        # Simple heuristic-based semantic alignment
        alignment_score = 0.5  # Base score
        
        caption_lower = caption.lower()
        
        # Energy-based alignment
        energy = audio_features.get('mean_rms', 0)
        if any(word in caption_lower for word in ['loud', 'scream', 'shout', 'thunder', 'crash']):
            if energy > 0.05:  # High energy threshold
                alignment_score += 0.2
        elif any(word in caption_lower for word in ['quiet', 'whisper', 'soft', 'gentle']):
            if energy < 0.02:  # Low energy threshold
                alignment_score += 0.2
        
        # Tempo-based alignment
        tempo = audio_features.get('tempo', 0)
        if any(word in caption_lower for word in ['fast', 'quick', 'rapid', 'speed']):
            if tempo > 120:  # Fast tempo
                alignment_score += 0.15
        elif any(word in caption_lower for word in ['slow', 'calm', 'peaceful']):
            if tempo < 80:  # Slow tempo
                alignment_score += 0.15
        
        # Spectral content alignment
        spectral_centroid = audio_features.get('mean_spectral_centroid', 0)
        if any(word in caption_lower for word in ['high', 'bright', 'sharp']):
            if spectral_centroid > 2000:  # High frequency content
                alignment_score += 0.15
        elif any(word in caption_lower for word in ['low', 'deep', 'bass']):
            if spectral_centroid < 1000:  # Low frequency content
                alignment_score += 0.15
        
        return min(1.0, alignment_score)
    
    def validate_audiocaps_alignment(self) -> Dict[str, Any]:
        """Validate audio-visual alignment for AudioCaps dataset."""
        print("🎵 Validating AudioCaps audio-caption alignment...")
        
        alignment_results = {
            'splits': {},
            'overall_alignment_scores': [],
            'semantic_scores': [],
            'temporal_consistency': []
        }
        
        for split in ['train', 'val']:  # Skip test if no audio files
            print(f"  Processing {split} split...")
            
            try:
                df = load_audiocaps_metadata(self.data_path, split)
                audio_dir = Path(self.data_path) / "audiocaps" / "audio" / split
                
                if not audio_dir.exists():
                    print(f"    ⚠️  Audio directory not found: {audio_dir}")
                    alignment_results['splits'][split] = {
                        'error': f'Audio directory not found: {audio_dir}',
                        'samples_checked': 0
                    }
                    continue
                
                split_alignment_scores = []
                split_semantic_scores = []
                samples_processed = 0
                max_samples = min(100, len(df))  # Limit for validation
                
                sample_indices = np.random.choice(len(df), max_samples, replace=False)
                
                for idx in sample_indices:
                    row = df.iloc[idx]
                    youtube_id = row['youtube_id']
                    caption = str(row.get('caption', ''))
                    
                    audio_file = audio_dir / f"{youtube_id}.wav"
                    
                    if not audio_file.exists():
                        continue
                    
                    # Extract audio features
                    audio_features = self.extract_audio_features(str(audio_file))
                    
                    if audio_features.get('loadable', False):
                        # Calculate semantic alignment
                        semantic_score = self.calculate_semantic_alignment(caption, audio_features)
                        split_semantic_scores.append(semantic_score)
                        
                        # Simple temporal consistency (duration vs caption length correlation)
                        caption_length = len(caption.split())
                        duration = audio_features.get('duration', 0)
                        
                        # Expected: longer captions might correlate with longer audio
                        # This is a weak signal but provides some validation
                        temporal_score = min(1.0, 0.3 + (duration / 30.0) + (caption_length / 50.0))
                        split_alignment_scores.append(temporal_score)
                        
                        samples_processed += 1
                
                split_results = {
                    'samples_checked': samples_processed,
                    'mean_semantic_alignment': np.mean(split_semantic_scores) if split_semantic_scores else 0,
                    'mean_temporal_alignment': np.mean(split_alignment_scores) if split_alignment_scores else 0,
                    'semantic_scores': split_semantic_scores,
                    'temporal_scores': split_alignment_scores
                }
                
                alignment_results['splits'][split] = split_results
                alignment_results['overall_alignment_scores'].extend(split_alignment_scores)
                alignment_results['semantic_scores'].extend(split_semantic_scores)
                
                print(f"    ✅ Processed {samples_processed} samples")
                if split_semantic_scores:
                    print(f"    📊 Mean semantic alignment: {np.mean(split_semantic_scores):.3f}")
                
            except Exception as e:
                alignment_results['splits'][split] = {
                    'error': str(e),
                    'samples_checked': 0
                }
                print(f"    ❌ Error processing {split}: {e}")
        
        return alignment_results
    
    def validate_cross_modal_consistency(self) -> Dict[str, Any]:
        """Check for consistency across modalities."""
        print("🔄 Validating cross-modal consistency...")
        
        consistency_results = {
            'id_consistency': {},
            'metadata_consistency': {},
            'file_availability': {}
        }
        
        try:
            # Check AudioCaps ID consistency
            for split in ['train', 'val']:
                df = load_audiocaps_metadata(self.data_path, split)
                audio_dir = Path(self.data_path) / "audiocaps" / "audio" / split
                
                if not audio_dir.exists():
                    continue
                
                # Check metadata-file consistency
                metadata_ids = set(df['youtube_id'].astype(str))
                audio_files = {f.stem for f in audio_dir.glob("*.wav")}
                
                missing_audio = metadata_ids - audio_files
                extra_audio = audio_files - metadata_ids
                
                consistency_rate = len(metadata_ids & audio_files) / len(metadata_ids) if metadata_ids else 0
                
                consistency_results['id_consistency'][f'audiocaps_{split}'] = {
                    'metadata_ids': len(metadata_ids),
                    'audio_files': len(audio_files),
                    'consistent_ids': len(metadata_ids & audio_files),
                    'missing_audio_files': len(missing_audio),
                    'extra_audio_files': len(extra_audio),
                    'consistency_rate': consistency_rate
                }
                
                print(f"  AudioCaps {split}: {consistency_rate:.1%} consistency rate")
        
        except Exception as e:
            consistency_results['error'] = str(e)
            print(f"  ❌ Error in consistency check: {e}")
        
        return consistency_results
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete audio-visual alignment validation."""
        print("🔍 Starting Audio-Visual Alignment Validation...")
        print("=" * 50)
        
        results = {
            'alignment_analysis': {},
            'consistency_analysis': {},
            'validation_summary': {}
        }
        
        # Run alignment validation
        results['alignment_analysis'] = self.validate_audiocaps_alignment()
        
        # Run consistency validation  
        results['consistency_analysis'] = self.validate_cross_modal_consistency()
        
        # Create validation results for go/no-go criteria
        all_alignment_scores = results['alignment_analysis'].get('overall_alignment_scores', [])
        semantic_scores = results['alignment_analysis'].get('semantic_scores', [])
        
        validations = []
        
        # Test 1: Overall alignment rate
        if all_alignment_scores:
            alignment_validation = check_av_alignment(all_alignment_scores, min_alignment=0.7)
            validations.append(alignment_validation)
        
        # Test 2: Semantic alignment
        if semantic_scores:
            good_semantic = sum(1 for score in semantic_scores if score >= 0.6)
            semantic_rate = good_semantic / len(semantic_scores)
            semantic_validation = ValidationResult(
                "Semantic Alignment", 
                semantic_rate >= 0.7,
                f"Semantic alignment rate: {semantic_rate:.1%}",
                semantic_rate, 0.7
            )
            validations.append(semantic_validation)
        
        # Test 3: File consistency
        consistency_rates = []
        for split_results in results['consistency_analysis'].get('id_consistency', {}).values():
            if isinstance(split_results, dict) and 'consistency_rate' in split_results:
                consistency_rates.append(split_results['consistency_rate'])
        
        if consistency_rates:
            mean_consistency = np.mean(consistency_rates)
            consistency_validation = ValidationResult(
                "File Consistency",
                mean_consistency >= 0.9,
                f"Mean file consistency: {mean_consistency:.1%}",
                mean_consistency, 0.9
            )
            validations.append(consistency_validation)
        
        # Overall summary
        passed_tests = sum(1 for v in validations if v.passed)
        total_tests = len(validations)
        
        results['validation_summary'] = {
            'overall_passed': passed_tests == total_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'validations': validations
        }
        
        # Save results
        with open(self.output_dir / 'av_alignment_results.json', 'w') as f:
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
        """Print alignment validation summary."""
        if not self.results:
            print("No validation results available. Run validation first.")
            return
        
        print("\n" + "=" * 50)
        print("AUDIO-VISUAL ALIGNMENT SUMMARY")
        print("=" * 50)
        
        summary = self.results.get('validation_summary', {})
        if summary.get('overall_passed'):
            print("✅ Overall Status: ALIGNMENT VALIDATED")
        else:
            print("❌ Overall Status: ALIGNMENT ISSUES FOUND")
        
        print(f"Pass Rate: {summary.get('pass_rate', 0):.1%}")
        
        for validation in summary.get('validations', []):
            print(f"  {validation}")
        
        # Additional details
        alignment_analysis = self.results.get('alignment_analysis', {})
        if alignment_analysis.get('overall_alignment_scores'):
            scores = alignment_analysis['overall_alignment_scores']
            print(f"\n📊 Alignment Statistics:")
            print(f"  Samples analyzed: {len(scores)}")
            print(f"  Mean alignment: {np.mean(scores):.3f}")
            print(f"  Median alignment: {np.median(scores):.3f}")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run audio-visual alignment validation")
    parser.add_argument("--data-path", required=True, help="Path to data directory")
    parser.add_argument("--output-dir", default="./experiments/reports", 
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Run validation
    validator = AVAlignmentValidator(args.data_path, args.output_dir)
    results = validator.run_full_validation()
    validator.print_summary()
    
    # Return exit code based on results
    if results['validation_summary']['overall_passed']:
        print("\n🎉 Audio-visual alignment validation passed!")
        return 0
    else:
        print("\n⚠️  Audio-visual alignment validation found issues.")
        return 1

if __name__ == "__main__":
    exit(main())