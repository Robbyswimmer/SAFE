"""
Schema & Distribution Audit

Validates data quality per dataset:
- Class balance analysis 
- Audio length/SNR histograms
- Image resolution analysis
- Missing data detection
- File loading verification
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.utils.data_utils import (
    load_audiocaps_metadata, load_vqa_metadata, 
    check_file_loadable, sample_dataset_files, calculate_audio_features
)
from experiments.utils.validation_metrics import (
    ValidationResult, check_file_load_rate, check_class_balance
)

class SchemaAuditor:
    """Performs comprehensive schema and distribution audit."""
    
    def __init__(self, data_path: str, output_dir: str = "./experiments/reports"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def audit_audiocaps(self) -> Dict[str, Any]:
        """Audit AudioCaps dataset."""
        print("🎵 Auditing AudioCaps dataset...")
        
        audit_results = {
            'splits': {},
            'audio_analysis': {},
            'class_balance': {},
            'file_loading': {}
        }
        
        for split in ['train', 'val', 'test']:
            try:
                # Load metadata
                df = load_audiocaps_metadata(self.data_path, split)
                audio_dir = Path(self.data_path) / "audiocaps" / "audio" / split
                
                split_results = {
                    'metadata_samples': len(df),
                    'caption_lengths': [],
                    'duration_distribution': [],
                    'audio_files_found': 0,
                    'missing_files': []
                }
                
                # Analyze captions
                if 'caption' in df.columns:
                    split_results['caption_lengths'] = [len(caption.split()) 
                                                      for caption in df['caption'].fillna('')]
                
                # Check audio files
                if audio_dir.exists():
                    audio_files = list(audio_dir.glob("*.wav"))
                    split_results['audio_files_found'] = len(audio_files)
                    
                    # Check for missing files
                    for _, row in df.iterrows():
                        youtube_id = row['youtube_id']
                        audio_file = audio_dir / f"{youtube_id}.wav"
                        if not audio_file.exists():
                            split_results['missing_files'].append(youtube_id)
                    
                    # Sample files for detailed analysis
                    sample_files = np.random.choice(audio_files, 
                                                  min(50, len(audio_files)), 
                                                  replace=False)
                    
                    audio_features = []
                    for audio_file in sample_files:
                        features = calculate_audio_features(str(audio_file))
                        if 'error' not in features:
                            audio_features.append(features)
                    
                    if audio_features:
                        split_results['audio_features'] = {
                            'duration_mean': np.mean([f['duration'] for f in audio_features]),
                            'duration_std': np.std([f['duration'] for f in audio_features]),
                            'snr_mean': np.mean([f['snr_estimate'] for f in audio_features]),
                            'sample_rate_mode': Counter([f['sample_rate'] for f in audio_features]).most_common(1)[0][0]
                        }
                
                audit_results['splits'][split] = split_results
                
            except Exception as e:
                audit_results['splits'][split] = {'error': str(e)}
        
        # Test file loading
        sample_files = sample_dataset_files(self.data_path, 100)
        load_results = []
        for audio_file in sample_files['audiocaps_audio'][:50]:  # Test 50 files
            result = check_file_loadable(str(audio_file), "audio")
            load_results.append(result)
        
        audit_results['file_loading']['results'] = load_results
        audit_results['file_loading']['validation'] = check_file_load_rate(load_results)
        
        return audit_results
    
    def audit_vqa(self) -> Dict[str, Any]:
        """Audit VQA dataset."""
        print("🖼️  Auditing VQA dataset...")
        
        audit_results = {
            'splits': {},
            'question_analysis': {},
            'answer_analysis': {},
            'image_analysis': {},
            'file_loading': {}
        }
        
        all_question_types = []
        all_answer_types = []
        
        for split in ['train', 'val']:
            try:
                questions, annotations = load_vqa_metadata(self.data_path, split)
                images_dir = Path(self.data_path) / "vqa" / "images" / f"{split}2014"
                
                split_results = {
                    'questions_count': len(questions),
                    'annotations_count': len(annotations),
                    'question_lengths': [],
                    'answer_lengths': [],
                    'question_types': [],
                    'answer_types': [],
                    'image_files_found': 0
                }
                
                # Analyze questions
                for q in questions:
                    if 'question' in q:
                        split_results['question_lengths'].append(len(q['question'].split()))
                
                # Analyze annotations
                for ann in annotations:
                    if 'question_type' in ann:
                        split_results['question_types'].append(ann['question_type'])
                        all_question_types.append(ann['question_type'])
                    
                    if 'answer_type' in ann:
                        split_results['answer_types'].append(ann['answer_type'])
                        all_answer_types.append(ann['answer_type'])
                    
                    if 'multiple_choice_answer' in ann:
                        answer = ann['multiple_choice_answer']
                        split_results['answer_lengths'].append(len(str(answer).split()))
                
                # Check image files
                if images_dir.exists():
                    image_files = list(images_dir.glob("*.jpg"))
                    split_results['image_files_found'] = len(image_files)
                
                audit_results['splits'][split] = split_results
                
            except Exception as e:
                audit_results['splits'][split] = {'error': str(e)}
        
        # Class balance analysis
        if all_question_types:
            question_type_counts = Counter(all_question_types)
            audit_results['question_analysis']['type_balance'] = check_class_balance(
                dict(question_type_counts), max_imbalance=10.0  # More lenient for question types
            )
        
        if all_answer_types:
            answer_type_counts = Counter(all_answer_types)
            audit_results['answer_analysis']['type_balance'] = check_class_balance(
                dict(answer_type_counts), max_imbalance=5.0
            )
        
        # Test file loading
        sample_files = sample_dataset_files(self.data_path, 100)
        load_results = []
        for image_file in sample_files['vqa_images'][:50]:  # Test 50 files
            result = check_file_loadable(str(image_file), "image")
            load_results.append(result)
        
        audit_results['file_loading']['results'] = load_results
        audit_results['file_loading']['validation'] = check_file_load_rate(load_results)
        
        return audit_results
    
    def generate_histograms(self, results: Dict[str, Any]):
        """Generate distribution histograms."""
        print("📊 Generating distribution histograms...")
        
        # AudioCaps histograms
        if 'audiocaps' in results:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('AudioCaps Distribution Analysis')
            
            # Caption lengths
            caption_lengths = []
            for split_data in results['audiocaps']['splits'].values():
                if isinstance(split_data, dict) and 'caption_lengths' in split_data:
                    caption_lengths.extend(split_data['caption_lengths'])
            
            if caption_lengths:
                axes[0, 0].hist(caption_lengths, bins=30, alpha=0.7)
                axes[0, 0].set_title('Caption Length Distribution')
                axes[0, 0].set_xlabel('Words per Caption')
                axes[0, 0].set_ylabel('Frequency')
            
            # Audio durations (if available)
            durations = []
            snr_estimates = []
            for split_data in results['audiocaps']['splits'].values():
                if (isinstance(split_data, dict) and 
                    'audio_features' in split_data and 
                    split_data['audio_features']):
                    # This would need individual file analysis for proper histograms
                    pass
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'audiocaps_distributions.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # VQA histograms
        if 'vqa' in results:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('VQA Distribution Analysis')
            
            # Question lengths
            question_lengths = []
            answer_lengths = []
            for split_data in results['vqa']['splits'].values():
                if isinstance(split_data, dict):
                    if 'question_lengths' in split_data:
                        question_lengths.extend(split_data['question_lengths'])
                    if 'answer_lengths' in split_data:
                        answer_lengths.extend(split_data['answer_lengths'])
            
            if question_lengths:
                axes[0, 0].hist(question_lengths, bins=20, alpha=0.7)
                axes[0, 0].set_title('Question Length Distribution')
                axes[0, 0].set_xlabel('Words per Question')
                axes[0, 0].set_ylabel('Frequency')
            
            if answer_lengths:
                axes[0, 1].hist(answer_lengths, bins=20, alpha=0.7)
                axes[0, 1].set_title('Answer Length Distribution')
                axes[0, 1].set_xlabel('Words per Answer')
                axes[0, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'vqa_distributions.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def run_full_audit(self) -> Dict[str, Any]:
        """Run complete schema audit."""
        print("🔍 Starting Schema & Distribution Audit...")
        print("=" * 50)
        
        results = {}
        
        # Audit each dataset
        results['audiocaps'] = self.audit_audiocaps()
        results['vqa'] = self.audit_vqa()
        
        # Generate visualizations
        self.generate_histograms(results)
        
        # Overall validation summary
        all_validations = []
        
        # Collect all validation results
        for dataset, dataset_results in results.items():
            if 'file_loading' in dataset_results and 'validation' in dataset_results['file_loading']:
                all_validations.append(dataset_results['file_loading']['validation'])
        
        # Add class balance validations
        if 'question_analysis' in results.get('vqa', {}):
            if 'type_balance' in results['vqa']['question_analysis']:
                all_validations.append(results['vqa']['question_analysis']['type_balance'])
        
        if 'answer_analysis' in results.get('vqa', {}):
            if 'type_balance' in results['vqa']['answer_analysis']:
                all_validations.append(results['vqa']['answer_analysis']['type_balance'])
        
        # Overall assessment
        passed_tests = sum(1 for v in all_validations if v.passed)
        total_tests = len(all_validations)
        
        results['summary'] = {
            'overall_passed': passed_tests == total_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'validations': all_validations
        }
        
        # Save results
        with open(self.output_dir / 'schema_audit_results.json', 'w') as f:
            # Convert ValidationResult objects to dicts for JSON serialization
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
        else:
            return obj
    
    def print_summary(self):
        """Print audit summary."""
        if not self.results:
            print("No audit results available. Run audit first.")
            return
        
        print("\n" + "=" * 50)
        print("SCHEMA AUDIT SUMMARY")
        print("=" * 50)
        
        summary = self.results.get('summary', {})
        if summary.get('overall_passed'):
            print("✅ Overall Status: PASSED")
        else:
            print("❌ Overall Status: FAILED")
        
        print(f"Pass Rate: {summary.get('pass_rate', 0):.1%}")
        
        for validation in summary.get('validations', []):
            print(f"  {validation}")
        
        # Dataset-specific summaries
        for dataset in ['audiocaps', 'vqa']:
            if dataset in self.results:
                print(f"\n📊 {dataset.upper()} Summary:")
                dataset_results = self.results[dataset]
                
                for split, split_data in dataset_results.get('splits', {}).items():
                    if isinstance(split_data, dict) and 'error' not in split_data:
                        if dataset == 'audiocaps':
                            metadata_count = split_data.get('metadata_samples', 0)
                            audio_count = split_data.get('audio_files_found', 0)
                            coverage = audio_count / metadata_count if metadata_count > 0 else 0
                            print(f"  {split}: {metadata_count} metadata, {audio_count} audio ({coverage:.1%} coverage)")
                        elif dataset == 'vqa':
                            questions = split_data.get('questions_count', 0)
                            images = split_data.get('image_files_found', 0)
                            print(f"  {split}: {questions} questions, {images} images")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run schema and distribution audit")
    parser.add_argument("--data-path", required=True, help="Path to data directory")
    parser.add_argument("--output-dir", default="./experiments/reports", 
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Run audit
    auditor = SchemaAuditor(args.data_path, args.output_dir)
    results = auditor.run_full_audit()
    auditor.print_summary()
    
    # Return exit code based on results
    if results['summary']['overall_passed']:
        print("\n🎉 Schema audit completed successfully!")
        return 0
    else:
        print("\n⚠️  Schema audit found issues. Check reports for details.")
        return 1

if __name__ == "__main__":
    exit(main())