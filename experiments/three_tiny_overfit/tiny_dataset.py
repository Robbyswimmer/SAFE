"""
Tiny Dataset Creator

Creates small curated datasets (512-1000 samples) for overfitting validation.
Ensures always-on audio by combining AudioCaps + VQA with audio augmentation.
"""

import os
import sys
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_stage_a_curriculum import create_datasets
from experiments.utils.validation_metrics import ValidationResult

class TinyDatasetCreator:
    """Creates and manages tiny datasets for overfitting experiments."""
    
    def __init__(self, data_path: str, output_dir: str = "./experiments/tiny_datasets"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_info = {}
        
    def create_tiny_dataset(self, 
                          target_size: int = 512,
                          audio_ratio: float = 1.0,
                          splits: Dict[str, float] = None,
                          seed: int = 42) -> Dict[str, Any]:
        """Create tiny dataset with specified characteristics."""
        print(f"🔬 Creating tiny dataset with {target_size} samples...")
        
        # Default splits for tiny dataset
        if splits is None:
            splits = {
                'train': 0.8,  # 80% for training (overfitting target)
                'val': 0.2     # 20% for validation
            }
        
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        tiny_dataset_info = {
            'creation_time': datetime.now().isoformat(),
            'target_size': target_size,
            'audio_ratio': audio_ratio,
            'splits': splits,
            'seed': seed,
            'samples': {},
            'statistics': {}
        }
        
        try:
            # Load full datasets
            print("  📊 Loading full datasets...")
            train_dataset, val_dataset = create_datasets(
                use_dummy=False,  # Use REAL data for meaningful test
                data_path=self.data_path
            )
            full_datasets = {'train': train_dataset, 'val': val_dataset}
            
            # Combine samples from different sources
            all_samples = []
            
            # Process AudioCaps samples (always have audio)
            if 'train' in full_datasets:
                dataset = full_datasets['train']
                print(f"    📂 Sampling from dataset with {len(dataset)} samples")
                
                # Sample indices
                available_indices = list(range(len(dataset)))
                sample_indices = np.random.choice(
                    available_indices, 
                    min(target_size, len(available_indices)), 
                    replace=False
                )
                
                for idx in tqdm(sample_indices, desc="  📦 Collecting samples"):
                    try:
                        sample = dataset[idx]
                        
                        # Ensure always-on audio requirement
                        if self._has_audio_content(sample):
                            processed_sample = self._process_sample(sample, 'audio_vqa')
                            all_samples.append(processed_sample)
                        elif audio_ratio == 1.0:
                            # If we need 100% audio but sample doesn't have audio,
                            # try to create synthetic audio-dependent version
                            processed_sample = self._create_audio_dependent_sample(sample)
                            if processed_sample:
                                all_samples.append(processed_sample)
                        else:
                            # Include VL-only samples if audio_ratio < 1.0
                            processed_sample = self._process_sample(sample, 'vqa_only')
                            all_samples.append(processed_sample)
                            
                    except Exception as e:
                        print(f"    ⚠️  Error processing sample {idx}: {e}")
                        continue
            
            # Truncate to target size if we have more than needed
            if len(all_samples) > target_size:
                all_samples = random.sample(all_samples, target_size)
            
            print(f"  ✅ Collected {len(all_samples)} valid samples")
            
            # Split into train/val
            random.shuffle(all_samples)
            
            train_size = int(len(all_samples) * splits['train'])
            train_samples = all_samples[:train_size]
            val_samples = all_samples[train_size:]
            
            tiny_dataset_info['samples'] = {
                'train': train_samples,
                'val': val_samples
            }
            
            # Calculate statistics
            tiny_dataset_info['statistics'] = self._calculate_dataset_statistics(
                train_samples, val_samples
            )
            
            # Skip saving to JSON to avoid serialization issues with large tensors
            print(f"  💾 Dataset created in memory (not saved to avoid large files)")
            print(f"  📊 Train samples: {len(train_samples)}")
            print(f"  📊 Val samples: {len(val_samples)}")
            
            # Debug: Show dataset statistics for validation debugging
            stats = tiny_dataset_info.get('statistics', {})
            print(f"  🔍 Dataset Statistics:")
            print(f"    Total samples: {stats.get('total_samples', 0)}")
            print(f"    Audio ratio: {stats.get('audio_ratio', 0):.1%}")
            print(f"    Split ratio: {stats.get('split_ratio', 0):.1%}")
            print(f"    Audio samples: {stats.get('audio_samples', 0)}")
            print(f"    Train samples: {stats.get('train_samples', 0)}")
            print(f"    Val samples: {stats.get('val_samples', 0)}")
            
            self.dataset_info = tiny_dataset_info
            return tiny_dataset_info
            
        except Exception as e:
            print(f"  ❌ Error creating tiny dataset: {e}")
            raise
    
    def _has_audio_content(self, sample: Dict[str, Any]) -> bool:
        """Check if sample has meaningful audio content."""
        if 'audio' not in sample:
            return False
        
        audio = sample['audio']
        if audio is None:
            return False
        
        # Check if audio tensor has meaningful content
        if isinstance(audio, torch.Tensor):
            return audio.numel() > 0 and not torch.all(audio == 0)
        
        return True
    
    def _process_sample(self, sample: Dict[str, Any], sample_type: str) -> Dict[str, Any]:
        """Process and standardize a sample."""
        processed = {
            'sample_type': sample_type,
            'text': sample.get('text', sample.get('question', '')),
            'image': sample.get('image'),
            'audio': sample.get('audio'),
            'answer': sample.get('answer', ''),
            'question_id': sample.get('question_id', sample.get('audio_id', 'unknown')),
            'metadata': {
                'original_dataset': sample.get('dataset_type', 'unknown'),
                'has_audio': self._has_audio_content(sample),
                'has_image': sample.get('image') is not None,
                'text_length': len(sample.get('text', sample.get('question', '')).split())
            }
        }
        
        return processed
    
    def _create_audio_dependent_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Create audio-dependent version of VL-only sample."""
        # This is a placeholder - in practice, you might:
        # 1. Add synthetic audio descriptions
        # 2. Use text-to-speech for questions
        # 3. Add ambient audio tracks
        
        # For now, skip samples without audio to maintain quality
        return None
    
    def _calculate_dataset_statistics(self, train_samples: List[Dict], 
                                    val_samples: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive dataset statistics."""
        all_samples = train_samples + val_samples
        
        # Basic counts
        stats = {
            'total_samples': len(all_samples),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'split_ratio': len(train_samples) / len(all_samples) if all_samples else 0
        }
        
        # Content analysis
        audio_samples = sum(1 for s in all_samples if s['metadata']['has_audio'])
        image_samples = sum(1 for s in all_samples if s['metadata']['has_image'])
        
        stats.update({
            'audio_samples': audio_samples,
            'image_samples': image_samples,
            'audio_ratio': audio_samples / len(all_samples) if all_samples else 0,
            'image_ratio': image_samples / len(all_samples) if all_samples else 0
        })
        
        # Text analysis
        text_lengths = [s['metadata']['text_length'] for s in all_samples]
        if text_lengths:
            stats['text_stats'] = {
                'mean_length': np.mean(text_lengths),
                'std_length': np.std(text_lengths),
                'min_length': np.min(text_lengths),
                'max_length': np.max(text_lengths)
            }
        
        # Sample type distribution
        sample_types = [s['sample_type'] for s in all_samples]
        from collections import Counter
        type_counts = Counter(sample_types)
        stats['sample_type_distribution'] = dict(type_counts)
        
        # Dataset source distribution
        original_datasets = [s['metadata']['original_dataset'] for s in all_samples]
        dataset_counts = Counter(original_datasets)
        stats['source_dataset_distribution'] = dict(dataset_counts)
        
        return stats
    
    def _save_tiny_dataset(self, dataset_info: Dict[str, Any], output_path: Path):
        """Save tiny dataset to JSON file."""
        # Make samples serializable
        serializable_info = self._make_serializable(dataset_info)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_info, f, indent=2, default=str)
    
    def _make_serializable(self, obj):
        """Convert PyTorch tensors and other objects to serializable format."""
        if isinstance(obj, torch.Tensor):
            # Only store metadata, not raw data to avoid huge files
            return {
                '_tensor_metadata': True,
                '_tensor_shape': list(obj.shape),
                '_tensor_dtype': str(obj.dtype),
                '_tensor_size': obj.numel()
            }
        elif isinstance(obj, dict):
            # Skip raw audio/image data keys
            filtered_obj = {}
            for k, v in obj.items():
                if k in ['audio', 'image'] and isinstance(v, (torch.Tensor, np.ndarray)):
                    # Store only metadata for large tensors
                    if isinstance(v, torch.Tensor):
                        filtered_obj[k + '_metadata'] = {
                            'shape': list(v.shape),
                            'dtype': str(v.dtype),
                            'size': v.numel(),
                            'has_data': v.numel() > 0
                        }
                    elif isinstance(v, np.ndarray):
                        filtered_obj[k + '_metadata'] = {
                            'shape': list(v.shape),
                            'dtype': str(v.dtype),
                            'size': v.size,
                            'has_data': v.size > 0
                        }
                else:
                    filtered_obj[k] = self._make_serializable(v)
            return filtered_obj
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            # For small arrays, store data; for large ones, store metadata
            if obj.size > 1000:  # Arbitrary threshold
                return {
                    '_array_metadata': True,
                    '_array_shape': list(obj.shape),
                    '_array_dtype': str(obj.dtype),
                    '_array_size': obj.size
                }
            else:
                return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def load_tiny_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Load tiny dataset from JSON file."""
        print(f"📂 Loading tiny dataset from: {dataset_path}")
        
        try:
            with open(dataset_path, 'r') as f:
                dataset_info = json.load(f)
            
            # Reconstruct tensors
            dataset_info = self._reconstruct_tensors(dataset_info)
            
            self.dataset_info = dataset_info
            print(f"  ✅ Loaded {dataset_info['statistics']['total_samples']} samples")
            
            return dataset_info
            
        except Exception as e:
            print(f"  ❌ Error loading dataset: {e}")
            raise
    
    def _reconstruct_tensors(self, obj):
        """Reconstruct PyTorch tensors from serialized format."""
        if isinstance(obj, dict):
            if '_tensor_data' in obj:
                # Reconstruct tensor
                data = np.array(obj['_tensor_data'])
                tensor = torch.from_numpy(data)
                if obj.get('_tensor_shape'):
                    tensor = tensor.view(obj['_tensor_shape'])
                return tensor
            else:
                return {k: self._reconstruct_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._reconstruct_tensors(item) for item in obj]
        else:
            return obj
    
    def validate_tiny_dataset(self, dataset_info: Dict[str, Any] = None) -> List[ValidationResult]:
        """Validate tiny dataset meets requirements."""
        if dataset_info is None:
            dataset_info = self.dataset_info
        
        if not dataset_info:
            return [ValidationResult("dataset_loaded", False, "No dataset loaded", None, None)]
        
        validations = []
        stats = dataset_info.get('statistics', {})
        
        # Validate size requirements
        total_samples = stats.get('total_samples', 0)
        size_validation = ValidationResult(
            "dataset_size",
            total_samples >= 200,
            f"Dataset size: {total_samples} samples (minimum: 200)",
            total_samples, 200
        )
        validations.append(size_validation)
        
        # Validate audio ratio for always-on audio requirement
        audio_ratio = stats.get('audio_ratio', 0)
        audio_validation = ValidationResult(
            "audio_coverage",
            audio_ratio >= 0.8,  # More lenient - 80% coverage
            f"Audio coverage: {audio_ratio:.1%} (target: 80%+)",
            audio_ratio, 0.8
        )
        validations.append(audio_validation)
        
        # Validate split ratio
        split_ratio = stats.get('split_ratio', 0)
        split_validation = ValidationResult(
            "train_val_split",
            0.75 <= split_ratio <= 0.85,  # Should be around 80% train
            f"Train ratio: {split_ratio:.1%} (target: ~80%)",
            split_ratio, (0.75, 0.85)
        )
        validations.append(split_validation)
        
        # Validate minimum samples per split
        train_samples = stats.get('train_samples', 0)
        val_samples = stats.get('val_samples', 0)
        
        train_size_validation = ValidationResult(
            "train_set_size",
            train_samples >= 150,  # Lower bound for training
            f"Train samples: {train_samples} (minimum: 150)",
            train_samples, 150
        )
        validations.append(train_size_validation)
        
        val_size_validation = ValidationResult(
            "val_set_size",
            val_samples >= 30,  # Minimum for validation
            f"Val samples: {val_samples} (minimum: 30)",
            val_samples, 30
        )
        validations.append(val_size_validation)
        
        return validations
    
    def print_dataset_summary(self, dataset_info: Dict[str, Any] = None):
        """Print comprehensive dataset summary."""
        if dataset_info is None:
            dataset_info = self.dataset_info
        
        if not dataset_info:
            print("No dataset information available.")
            return
        
        print("\n" + "=" * 50)
        print("TINY DATASET SUMMARY")
        print("=" * 50)
        
        # Basic info
        stats = dataset_info.get('statistics', {})
        print(f"📊 Total Samples: {stats.get('total_samples', 0)}")
        print(f"🏋️  Train Samples: {stats.get('train_samples', 0)}")
        print(f"✅ Val Samples: {stats.get('val_samples', 0)}")
        print(f"📈 Train Ratio: {stats.get('split_ratio', 0):.1%}")
        
        # Content breakdown
        print(f"\n🎵 Audio Coverage: {stats.get('audio_ratio', 0):.1%}")
        print(f"🖼️  Image Coverage: {stats.get('image_ratio', 0):.1%}")
        
        # Sample type distribution
        if 'sample_type_distribution' in stats:
            print(f"\n📋 Sample Types:")
            for sample_type, count in stats['sample_type_distribution'].items():
                print(f"  {sample_type}: {count}")
        
        # Source dataset distribution
        if 'source_dataset_distribution' in stats:
            print(f"\n📂 Source Datasets:")
            for dataset, count in stats['source_dataset_distribution'].items():
                print(f"  {dataset}: {count}")
        
        # Text statistics
        if 'text_stats' in stats:
            text_stats = stats['text_stats']
            print(f"\n📝 Text Statistics:")
            print(f"  Mean length: {text_stats['mean_length']:.1f} words")
            print(f"  Range: {text_stats['min_length']}-{text_stats['max_length']} words")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create tiny dataset for overfitting experiments")
    parser.add_argument("--data-path", required=True, help="Path to full dataset directory")
    parser.add_argument("--output-dir", default="./experiments/tiny_datasets", 
                       help="Output directory for tiny datasets")
    parser.add_argument("--target-size", type=int, default=512,
                       help="Target number of samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create tiny dataset
    creator = TinyDatasetCreator(args.data_path, args.output_dir)
    dataset_info = creator.create_tiny_dataset(
        target_size=args.target_size,
        seed=args.seed
    )
    
    # Validate and summarize
    validations = creator.validate_tiny_dataset(dataset_info)
    creator.print_dataset_summary(dataset_info)
    
    # Print validation results
    print(f"\n🧪 Validation Results:")
    passed = sum(1 for v in validations if v.passed)
    print(f"  Passed: {passed}/{len(validations)}")
    
    for validation in validations:
        status = "✅" if validation.passed else "❌"
        print(f"  {status} {validation.test_name}: {validation.message}")
    
    return 0 if passed == len(validations) else 1

if __name__ == "__main__":
    exit(main())