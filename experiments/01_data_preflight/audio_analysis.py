"""
Audio Quality Analysis

Comprehensive audio quality assessment:
- SNR (Signal-to-Noise Ratio) analysis  
- Duration distribution analysis
- Spectral quality assessment
- Format and encoding validation
- Audio level and dynamic range analysis
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple
import librosa
from scipy import signal
from scipy.stats import describe
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.utils.data_utils import load_audiocaps_metadata, calculate_audio_features
from experiments.utils.validation_metrics import ValidationResult

class AudioQualityAnalyzer:
    """Analyzes audio quality across datasets."""
    
    def __init__(self, data_path: str, output_dir: str = "./experiments/reports"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def extract_detailed_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract comprehensive audio features for quality analysis."""
        try:
            # Load full audio file
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            # Basic properties
            features = {
                'duration': duration,
                'sample_rate': sr,
                'channels': 1,  # librosa loads as mono by default
                'samples': len(y),
                'file_size_mb': os.path.getsize(audio_path) / (1024 * 1024)
            }
            
            # Audio level analysis
            rms = librosa.feature.rms(y=y)[0]
            features.update({
                'rms_mean': float(np.mean(rms)),
                'rms_std': float(np.std(rms)),
                'rms_max': float(np.max(rms)),
                'peak_amplitude': float(np.max(np.abs(y))),
                'dynamic_range_db': float(20 * np.log10(np.max(rms) / (np.min(rms) + 1e-10)))
            })
            
            # Signal-to-noise ratio estimation
            # Method 1: Use spectral subtraction approach
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # Estimate noise from quietest portions
            frame_energy = np.mean(magnitude**2, axis=0)
            noise_threshold = np.percentile(frame_energy, 10)  # Bottom 10% as noise estimate
            noise_frames = frame_energy <= noise_threshold
            
            if np.any(noise_frames) and np.any(~noise_frames):
                noise_power = np.mean(frame_energy[noise_frames])
                signal_power = np.mean(frame_energy[~noise_frames])
                snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
            else:
                snr_db = 10.0  # Default reasonable value
            
            features['snr_estimate_db'] = float(snr_db)
            
            # Spectral analysis
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            features.update({
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate))
            })
            
            # Harmonic/percussive separation for content analysis
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_ratio = np.sum(y_harmonic**2) / (np.sum(y**2) + 1e-10)
            percussive_ratio = np.sum(y_percussive**2) / (np.sum(y**2) + 1e-10)
            
            features.update({
                'harmonic_ratio': float(harmonic_ratio),
                'percussive_ratio': float(percussive_ratio)
            })
            
            # Tempo and rhythm analysis
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features.update({
                    'tempo_bpm': float(tempo),
                    'beat_count': len(beats),
                    'rhythm_regularity': float(np.std(np.diff(beats))) if len(beats) > 1 else 0.0
                })
            except:
                features.update({
                    'tempo_bpm': 0.0,
                    'beat_count': 0,
                    'rhythm_regularity': 0.0
                })
            
            # Frequency content analysis
            # Check for clipping or distortion indicators
            clipping_threshold = 0.95
            clipped_samples = np.sum(np.abs(y) > clipping_threshold)
            clipping_rate = clipped_samples / len(y)
            
            features['clipping_rate'] = float(clipping_rate)
            
            # Silence detection
            silence_threshold = 0.01
            silent_samples = np.sum(np.abs(y) < silence_threshold)
            silence_rate = silent_samples / len(y)
            
            features['silence_rate'] = float(silence_rate)
            
            features['loadable'] = True
            features['error'] = None
            
            return features
            
        except Exception as e:
            return {
                'error': str(e),
                'loadable': False,
                'duration': 0,
                'sample_rate': 0
            }
    
    def analyze_audiocaps_quality(self) -> Dict[str, Any]:
        """Analyze audio quality for AudioCaps dataset."""
        print("🎵 Analyzing AudioCaps audio quality...")
        
        analysis_results = {
            'splits': {},
            'quality_metrics': {},
            'distributions': {}
        }
        
        all_features = []
        
        for split in ['train', 'val']:
            print(f"  Processing {split} split...")
            
            try:
                df = load_audiocaps_metadata(self.data_path, split)
                audio_dir = Path(self.data_path) / "audiocaps" / "audio" / split
                
                if not audio_dir.exists():
                    print(f"    ⚠️  Audio directory not found: {audio_dir}")
                    analysis_results['splits'][split] = {
                        'error': f'Audio directory not found: {audio_dir}',
                        'samples_analyzed': 0
                    }
                    continue
                
                # Sample files for analysis (limit for performance)
                available_files = list(audio_dir.glob("*.wav"))
                max_samples = min(100, len(available_files))
                
                if max_samples == 0:
                    analysis_results['splits'][split] = {
                        'error': 'No audio files found',
                        'samples_analyzed': 0
                    }
                    continue
                
                sample_files = np.random.choice(available_files, max_samples, replace=False)
                
                split_features = []
                successful_analyses = 0
                
                for audio_file in sample_files:
                    features = self.extract_detailed_audio_features(str(audio_file))
                    if features.get('loadable', False):
                        split_features.append(features)
                        all_features.append(features)
                        successful_analyses += 1
                
                if split_features:
                    # Calculate split statistics
                    split_stats = self._calculate_quality_statistics(split_features)
                    analysis_results['splits'][split] = {
                        'samples_analyzed': successful_analyses,
                        'total_files_available': len(available_files),
                        'statistics': split_stats
                    }
                    
                    print(f"    ✅ Analyzed {successful_analyses} files")
                    print(f"    📊 Mean SNR: {split_stats['snr_estimate_db']['mean']:.1f} dB")
                    print(f"    ⏱️  Mean duration: {split_stats['duration']['mean']:.1f}s")
                else:
                    analysis_results['splits'][split] = {
                        'error': 'No successful analyses',
                        'samples_analyzed': 0
                    }
            
            except Exception as e:
                analysis_results['splits'][split] = {
                    'error': str(e),
                    'samples_analyzed': 0
                }
                print(f"    ❌ Error processing {split}: {e}")
        
        # Overall quality metrics
        if all_features:
            analysis_results['quality_metrics'] = self._calculate_quality_statistics(all_features)
            analysis_results['distributions'] = self._extract_distributions(all_features)
        
        return analysis_results
    
    def _calculate_quality_statistics(self, features_list: List[Dict]) -> Dict[str, Any]:
        """Calculate statistical summaries of audio features."""
        if not features_list:
            return {}
        
        stats = {}
        
        # Numerical features to analyze
        numerical_features = [
            'duration', 'snr_estimate_db', 'rms_mean', 'peak_amplitude',
            'dynamic_range_db', 'spectral_centroid_mean', 'tempo_bpm',
            'clipping_rate', 'silence_rate', 'harmonic_ratio', 'percussive_ratio'
        ]
        
        for feature in numerical_features:
            values = [f.get(feature, 0) for f in features_list if f.get(feature) is not None]
            if values:
                stats[feature] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75))
                }
        
        # Sample rate analysis
        sample_rates = [f.get('sample_rate', 0) for f in features_list]
        unique_srs, sr_counts = np.unique(sample_rates, return_counts=True)
        stats['sample_rates'] = {
            'unique_rates': unique_srs.tolist(),
            'counts': sr_counts.tolist(),
            'most_common': int(unique_srs[np.argmax(sr_counts)]) if len(unique_srs) > 0 else 0
        }
        
        return stats
    
    def _extract_distributions(self, features_list: List[Dict]) -> Dict[str, List]:
        """Extract feature distributions for plotting."""
        distributions = {}
        
        features_to_plot = [
            'duration', 'snr_estimate_db', 'spectral_centroid_mean', 
            'dynamic_range_db', 'clipping_rate', 'silence_rate'
        ]
        
        for feature in features_to_plot:
            values = [f.get(feature, 0) for f in features_list if f.get(feature) is not None]
            distributions[feature] = values
        
        return distributions
    
    def generate_quality_plots(self):
        """Generate audio quality visualization plots."""
        print("📊 Generating audio quality plots...")
        
        if 'distributions' not in self.results or not self.results['distributions']:
            print("    ⚠️  No distribution data available for plotting")
            return
        
        distributions = self.results['distributions']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('AudioCaps Audio Quality Analysis', fontsize=16)
        
        plot_configs = [
            ('duration', 'Duration (seconds)', axes[0, 0]),
            ('snr_estimate_db', 'SNR Estimate (dB)', axes[0, 1]),
            ('spectral_centroid_mean', 'Spectral Centroid (Hz)', axes[0, 2]),
            ('dynamic_range_db', 'Dynamic Range (dB)', axes[1, 0]),
            ('clipping_rate', 'Clipping Rate', axes[1, 1]),
            ('silence_rate', 'Silence Rate', axes[1, 2])
        ]
        
        for feature, title, ax in plot_configs:
            if feature in distributions and distributions[feature]:
                values = distributions[feature]
                
                ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(title)
                ax.set_xlabel(title)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                mean_val = np.mean(values)
                median_val = np.median(values)
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='blue', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
                ax.legend()
            else:
                ax.text(0.5, 0.5, f'No data for\\n{title}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'audio_quality_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Plots saved to: {self.output_dir / 'audio_quality_analysis.png'}")
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete audio quality analysis."""
        print("🔍 Starting Audio Quality Analysis...")
        print("=" * 50)
        
        results = {
            'audiocaps_analysis': {},
            'validation_summary': {}
        }
        
        # Analyze AudioCaps
        results['audiocaps_analysis'] = self.analyze_audiocaps_quality()
        
        # Create validation results
        validations = []
        
        quality_metrics = results['audiocaps_analysis'].get('quality_metrics', {})
        
        # Validation 1: SNR quality
        if 'snr_estimate_db' in quality_metrics:
            snr_stats = quality_metrics['snr_estimate_db']
            mean_snr = snr_stats.get('mean', 0)
            
            snr_validation = ValidationResult(
                "Audio SNR Quality",
                mean_snr >= 10.0,  # Minimum 10dB SNR
                f"Mean SNR: {mean_snr:.1f} dB",
                mean_snr, 10.0
            )
            validations.append(snr_validation)
        
        # Validation 2: Duration distribution
        if 'duration' in quality_metrics:
            duration_stats = quality_metrics['duration']
            mean_duration = duration_stats.get('mean', 0)
            std_duration = duration_stats.get('std', 0)
            
            # Check if durations are reasonable (not too short/long, not too variable)
            duration_reasonable = (2.0 <= mean_duration <= 60.0 and std_duration <= 30.0)
            
            duration_validation = ValidationResult(
                "Duration Distribution",
                duration_reasonable,
                f"Mean duration: {mean_duration:.1f}s ± {std_duration:.1f}s",
                mean_duration, (2.0, 60.0)
            )
            validations.append(duration_validation)
        
        # Validation 3: Clipping rate
        if 'clipping_rate' in quality_metrics:
            clipping_stats = quality_metrics['clipping_rate']
            mean_clipping = clipping_stats.get('mean', 0)
            
            clipping_validation = ValidationResult(
                "Audio Clipping Rate",
                mean_clipping <= 0.01,  # Max 1% clipping
                f"Mean clipping rate: {mean_clipping:.1%}",
                mean_clipping, 0.01
            )
            validations.append(clipping_validation)
        
        # Validation 4: Excessive silence
        if 'silence_rate' in quality_metrics:
            silence_stats = quality_metrics['silence_rate']
            mean_silence = silence_stats.get('mean', 0)
            
            silence_validation = ValidationResult(
                "Silence Rate",
                mean_silence <= 0.3,  # Max 30% silence
                f"Mean silence rate: {mean_silence:.1%}",
                mean_silence, 0.3
            )
            validations.append(silence_validation)
        
        # Validation 5: Sample rate consistency
        if 'sample_rates' in quality_metrics:
            sr_info = quality_metrics['sample_rates']
            unique_rates = sr_info.get('unique_rates', [])
            
            # Check if we have reasonable sample rate(s)
            acceptable_rates = [16000, 22050, 44100, 48000]
            has_acceptable_rate = any(rate in acceptable_rates for rate in unique_rates)
            
            sr_validation = ValidationResult(
                "Sample Rate Quality",
                has_acceptable_rate and len(unique_rates) <= 3,  # Not too many different rates
                f"Sample rates found: {unique_rates}",
                len(unique_rates), 3
            )
            validations.append(sr_validation)
        
        # Overall summary
        passed_tests = sum(1 for v in validations if v.passed)
        total_tests = len(validations)
        
        results['validation_summary'] = {
            'overall_passed': passed_tests == total_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'validations': validations
        }
        
        # Generate plots
        self.results = results
        self.generate_quality_plots()
        
        # Save results
        with open(self.output_dir / 'audio_analysis_results.json', 'w') as f:
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2, default=str)
        
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
        """Print audio analysis summary."""
        if not self.results:
            print("No analysis results available. Run analysis first.")
            return
        
        print("\n" + "=" * 50)
        print("AUDIO QUALITY ANALYSIS SUMMARY")
        print("=" * 50)
        
        summary = self.results.get('validation_summary', {})
        if summary.get('overall_passed'):
            print("✅ Overall Status: AUDIO QUALITY ACCEPTABLE")
        else:
            print("❌ Overall Status: AUDIO QUALITY ISSUES FOUND")
        
        print(f"Pass Rate: {summary.get('pass_rate', 0):.1%}")
        
        for validation in summary.get('validations', []):
            print(f"  {validation}")
        
        # Dataset-specific summaries
        audiocaps_analysis = self.results.get('audiocaps_analysis', {})
        if 'quality_metrics' in audiocaps_analysis:
            metrics = audiocaps_analysis['quality_metrics']
            
            print(f"\n🎵 AudioCaps Quality Metrics:")
            
            if 'snr_estimate_db' in metrics:
                snr = metrics['snr_estimate_db']
                print(f"  SNR: {snr['mean']:.1f} ± {snr['std']:.1f} dB")
            
            if 'duration' in metrics:
                duration = metrics['duration']
                print(f"  Duration: {duration['mean']:.1f} ± {duration['std']:.1f} seconds")
            
            if 'sample_rates' in metrics:
                sr = metrics['sample_rates']
                print(f"  Sample rates: {sr['unique_rates']} (most common: {sr['most_common']} Hz)")
        
        # Split summaries
        for split, split_data in audiocaps_analysis.get('splits', {}).items():
            if isinstance(split_data, dict) and 'samples_analyzed' in split_data:
                analyzed = split_data['samples_analyzed']
                available = split_data.get('total_files_available', 'unknown')
                print(f"  {split}: {analyzed} analyzed / {available} available")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run audio quality analysis")
    parser.add_argument("--data-path", required=True, help="Path to data directory")
    parser.add_argument("--output-dir", default="./experiments/reports", 
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = AudioQualityAnalyzer(args.data_path, args.output_dir)
    results = analyzer.run_full_analysis()
    analyzer.print_summary()
    
    # Return exit code based on results
    if results['validation_summary']['overall_passed']:
        print("\n🎉 Audio quality analysis passed!")
        return 0
    else:
        print("\n⚠️  Audio quality analysis found issues.")
        return 1

if __name__ == "__main__":
    exit(main())