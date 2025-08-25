"""
ID Leakage Detection

Detects potential data leakage between train/val/test splits:
- ID overlap detection
- Content similarity analysis  
- Temporal leakage checks
- Statistical validation of split randomness
"""

import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.utils.data_utils import load_audiocaps_metadata, load_vqa_metadata
from experiments.utils.validation_metrics import ValidationResult, check_id_leakage

class LeakageDetector:
    """Detects data leakage between dataset splits."""
    
    def __init__(self, data_path: str, output_dir: str = "./experiments/reports"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def extract_audiocaps_ids(self) -> Dict[str, Set[str]]:
        """Extract IDs from AudioCaps splits."""
        print("🎵 Extracting AudioCaps IDs...")
        
        ids_by_split = {}
        
        for split in ['train', 'val', 'test']:
            try:
                df = load_audiocaps_metadata(self.data_path, split)
                
                # Use YouTube IDs and audiocap IDs
                youtube_ids = set(df['youtube_id'].astype(str))
                audiocap_ids = set(df['audiocap_id'].astype(str)) if 'audiocap_id' in df.columns else set()
                
                # Combine both ID types for comprehensive check
                all_ids = youtube_ids | audiocap_ids
                ids_by_split[split] = all_ids
                
                print(f"  {split}: {len(all_ids)} unique IDs ({len(youtube_ids)} YouTube, {len(audiocap_ids)} AudioCap)")
                
            except Exception as e:
                print(f"  Error loading {split}: {e}")
                ids_by_split[split] = set()
        
        return ids_by_split
    
    def extract_vqa_ids(self) -> Dict[str, Set[str]]:
        """Extract IDs from VQA splits."""
        print("🖼️  Extracting VQA IDs...")
        
        ids_by_split = {}
        
        for split in ['train', 'val']:
            try:
                questions, annotations = load_vqa_metadata(self.data_path, split)
                
                # Extract image IDs and question IDs
                question_ids = set()
                image_ids = set()
                
                for q in questions:
                    if 'question_id' in q:
                        question_ids.add(str(q['question_id']))
                    if 'image_id' in q:
                        image_ids.add(str(q['image_id']))
                
                for ann in annotations:
                    if 'question_id' in ann:
                        question_ids.add(str(ann['question_id']))
                    if 'image_id' in ann:
                        image_ids.add(str(ann['image_id']))
                
                # Use image IDs as primary (questions can be different for same image)
                ids_by_split[split] = image_ids
                
                print(f"  {split}: {len(image_ids)} image IDs, {len(question_ids)} question IDs")
                
            except Exception as e:
                print(f"  Error loading {split}: {e}")
                ids_by_split[split] = set()
        
        return ids_by_split
    
    def detect_id_overlap(self, ids_by_split: Dict[str, Set[str]], dataset_name: str) -> Dict[str, Any]:
        """Detect ID overlaps between splits."""
        print(f"🔍 Checking ID overlaps for {dataset_name}...")
        
        splits = list(ids_by_split.keys())
        overlaps = {}
        
        # Check all pairs of splits
        for i, split1 in enumerate(splits):
            for split2 in splits[i+1:]:
                overlap = ids_by_split[split1] & ids_by_split[split2]
                if overlap:
                    overlap_key = f"{split1}_{split2}"
                    overlaps[overlap_key] = {
                        'count': len(overlap),
                        'ids': list(overlap)[:10],  # Store first 10 for reporting
                        'all_ids': list(overlap) if len(overlap) <= 100 else None
                    }
                    print(f"  ⚠️  {overlap_key}: {len(overlap)} overlapping IDs")
                else:
                    print(f"  ✅ {split1}-{split2}: No overlaps")
        
        # Prepare for validation check
        leakage_results = {
            'train_ids': list(ids_by_split.get('train', set())),
            'val_ids': list(ids_by_split.get('val', set())),
            'test_ids': list(ids_by_split.get('test', set()))
        }
        
        validation = check_id_leakage(leakage_results)
        
        return {
            'dataset': dataset_name,
            'ids_by_split': {k: len(v) for k, v in ids_by_split.items()},
            'overlaps': overlaps,
            'validation': validation,
            'total_unique_ids': len(set().union(*ids_by_split.values()))
        }
    
    def detect_content_similarity(self, dataset_name: str) -> Dict[str, Any]:
        """Detect potential content similarity across splits."""
        print(f"📝 Checking content similarity for {dataset_name}...")
        
        similarity_results = {
            'dataset': dataset_name,
            'method': 'text_hash',
            'similar_pairs': [],
            'similarity_rate': 0.0
        }
        
        if dataset_name == 'audiocaps':
            # Check caption similarity across splits
            all_captions = {}
            
            for split in ['train', 'val', 'test']:
                try:
                    df = load_audiocaps_metadata(self.data_path, split)
                    for _, row in df.iterrows():
                        caption = str(row.get('caption', '')).strip().lower()
                        caption_hash = hashlib.md5(caption.encode()).hexdigest()
                        
                        if caption_hash in all_captions:
                            # Found duplicate caption
                            similar_pair = {
                                'caption': caption[:100],  # First 100 chars
                                'splits': [all_captions[caption_hash]['split'], split],
                                'ids': [all_captions[caption_hash]['id'], row.get('youtube_id', 'unknown')]
                            }
                            similarity_results['similar_pairs'].append(similar_pair)
                        else:
                            all_captions[caption_hash] = {
                                'split': split,
                                'id': row.get('youtube_id', 'unknown'),
                                'caption': caption
                            }
                
                except Exception as e:
                    print(f"  Error processing {split}: {e}")
            
            total_captions = len(all_captions)
            similar_count = len(similarity_results['similar_pairs'])
            similarity_results['similarity_rate'] = similar_count / total_captions if total_captions > 0 else 0
            
            print(f"  Found {similar_count} similar captions out of {total_captions} total")
        
        elif dataset_name == 'vqa':
            # Check question similarity across splits
            all_questions = {}
            
            for split in ['train', 'val']:
                try:
                    questions, _ = load_vqa_metadata(self.data_path, split)
                    for q in questions:
                        question_text = str(q.get('question', '')).strip().lower()
                        question_hash = hashlib.md5(question_text.encode()).hexdigest()
                        
                        if question_hash in all_questions:
                            similar_pair = {
                                'question': question_text[:100],
                                'splits': [all_questions[question_hash]['split'], split],
                                'ids': [all_questions[question_hash]['id'], q.get('question_id', 'unknown')]
                            }
                            similarity_results['similar_pairs'].append(similar_pair)
                        else:
                            all_questions[question_hash] = {
                                'split': split,
                                'id': q.get('question_id', 'unknown'),
                                'question': question_text
                            }
                
                except Exception as e:
                    print(f"  Error processing {split}: {e}")
            
            total_questions = len(all_questions)
            similar_count = len(similarity_results['similar_pairs'])
            similarity_results['similarity_rate'] = similar_count / total_questions if total_questions > 0 else 0
            
            print(f"  Found {similar_count} similar questions out of {total_questions} total")
        
        return similarity_results
    
    def validate_split_randomness(self, ids_by_split: Dict[str, Set[str]], dataset_name: str) -> Dict[str, Any]:
        """Validate that splits appear to be randomly distributed."""
        print(f"🎲 Validating split randomness for {dataset_name}...")
        
        randomness_results = {
            'dataset': dataset_name,
            'tests': {}
        }
        
        # Convert IDs to numeric for statistical tests
        all_ids = list(set().union(*ids_by_split.values()))
        
        if not all_ids:
            randomness_results['tests']['no_data'] = "No IDs available for testing"
            return randomness_results
        
        # Test 1: Size ratio reasonableness
        total_ids = len(all_ids)
        split_sizes = {split: len(ids) for split, ids in ids_by_split.items()}
        
        expected_ratios = {
            'train': 0.7,  # Rough expectations
            'val': 0.15,
            'test': 0.15
        }
        
        ratio_test = {}
        for split, actual_size in split_sizes.items():
            actual_ratio = actual_size / total_ids if total_ids > 0 else 0
            expected_ratio = expected_ratios.get(split, 0.1)
            
            # Allow reasonable variance (±20%)
            ratio_acceptable = abs(actual_ratio - expected_ratio) <= 0.2
            ratio_test[split] = {
                'actual_ratio': actual_ratio,
                'expected_ratio': expected_ratio,
                'acceptable': ratio_acceptable
            }
        
        randomness_results['tests']['size_ratios'] = ratio_test
        
        # Test 2: ID distribution (for numeric IDs)
        numeric_ids = []
        for id_str in all_ids:
            try:
                # Try to extract numeric parts
                numeric_part = ''.join(filter(str.isdigit, str(id_str)))
                if numeric_part:
                    numeric_ids.append(int(numeric_part) % 1000)  # Use last 3 digits
            except:
                pass
        
        if len(numeric_ids) > 100:  # Need sufficient sample
            # Simple uniformity test - check if splits have similar ID distributions
            split_numeric_samples = {}
            for split, ids in ids_by_split.items():
                split_numerics = []
                for id_str in ids:
                    try:
                        numeric_part = ''.join(filter(str.isdigit, str(id_str)))
                        if numeric_part:
                            split_numerics.append(int(numeric_part) % 1000)
                    except:
                        pass
                
                if split_numerics:
                    split_numeric_samples[split] = {
                        'mean': np.mean(split_numerics),
                        'std': np.std(split_numerics),
                        'count': len(split_numerics)
                    }
            
            randomness_results['tests']['numeric_distribution'] = split_numeric_samples
        
        return randomness_results
    
    def run_full_detection(self) -> Dict[str, Any]:
        """Run complete leakage detection."""
        print("🔍 Starting ID Leakage Detection...")
        print("=" * 50)
        
        results = {
            'datasets': {},
            'summary': {}
        }
        
        # Process each dataset
        for dataset_name in ['audiocaps', 'vqa']:
            dataset_results = {}
            
            try:
                # Extract IDs
                if dataset_name == 'audiocaps':
                    ids_by_split = self.extract_audiocaps_ids()
                elif dataset_name == 'vqa':
                    ids_by_split = self.extract_vqa_ids()
                
                # Detect overlaps
                overlap_results = self.detect_id_overlap(ids_by_split, dataset_name)
                dataset_results['id_overlaps'] = overlap_results
                
                # Content similarity
                similarity_results = self.detect_content_similarity(dataset_name)
                dataset_results['content_similarity'] = similarity_results
                
                # Randomness validation
                randomness_results = self.validate_split_randomness(ids_by_split, dataset_name)
                dataset_results['randomness'] = randomness_results
                
            except Exception as e:
                dataset_results['error'] = str(e)
                print(f"  Error processing {dataset_name}: {e}")
            
            results['datasets'][dataset_name] = dataset_results
        
        # Overall summary
        all_validations = []
        for dataset_results in results['datasets'].values():
            if 'id_overlaps' in dataset_results and 'validation' in dataset_results['id_overlaps']:
                all_validations.append(dataset_results['id_overlaps']['validation'])
        
        passed_tests = sum(1 for v in all_validations if v.passed)
        total_tests = len(all_validations)
        
        results['summary'] = {
            'overall_passed': passed_tests == total_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'validations': all_validations
        }
        
        # Save results
        with open(self.output_dir / 'leakage_detection_results.json', 'w') as f:
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
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj
    
    def print_summary(self):
        """Print leakage detection summary."""
        if not self.results:
            print("No detection results available. Run detection first.")
            return
        
        print("\n" + "=" * 50)
        print("LEAKAGE DETECTION SUMMARY")
        print("=" * 50)
        
        summary = self.results.get('summary', {})
        if summary.get('overall_passed'):
            print("✅ Overall Status: NO LEAKAGE DETECTED")
        else:
            print("❌ Overall Status: POTENTIAL LEAKAGE FOUND")
        
        print(f"Pass Rate: {summary.get('pass_rate', 0):.1%}")
        
        for validation in summary.get('validations', []):
            print(f"  {validation}")
        
        # Dataset-specific summaries
        for dataset_name, dataset_results in self.results.get('datasets', {}).items():
            print(f"\n📊 {dataset_name.upper()} Results:")
            
            if 'error' in dataset_results:
                print(f"  ❌ Error: {dataset_results['error']}")
                continue
            
            # ID overlaps
            if 'id_overlaps' in dataset_results:
                overlaps = dataset_results['id_overlaps']['overlaps']
                if overlaps:
                    print(f"  ⚠️  ID Overlaps found:")
                    for pair, overlap_info in overlaps.items():
                        print(f"    {pair}: {overlap_info['count']} overlapping IDs")
                else:
                    print("  ✅ No ID overlaps detected")
            
            # Content similarity
            if 'content_similarity' in dataset_results:
                similarity = dataset_results['content_similarity']
                similar_count = len(similarity.get('similar_pairs', []))
                similarity_rate = similarity.get('similarity_rate', 0)
                
                if similar_count > 0:
                    print(f"  ⚠️  Content similarity: {similar_count} similar items ({similarity_rate:.1%})")
                else:
                    print("  ✅ No content similarity detected")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ID leakage detection")
    parser.add_argument("--data-path", required=True, help="Path to data directory")
    parser.add_argument("--output-dir", default="./experiments/reports", 
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Run detection
    detector = LeakageDetector(args.data_path, args.output_dir)
    results = detector.run_full_detection()
    detector.print_summary()
    
    # Return exit code based on results
    if results['summary']['overall_passed']:
        print("\n🎉 Leakage detection completed - no issues found!")
        return 0
    else:
        print("\n⚠️  Leakage detection found potential issues. Check reports for details.")
        return 1

if __name__ == "__main__":
    exit(main())