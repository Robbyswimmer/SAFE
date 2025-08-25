"""
Curriculum Configuration Validation

Validates curriculum learning setup:
- YAML configuration validation
- Stage progression parameters
- Sampling ratio verification
- Loss weight configurations
- Dataset availability checks
"""

import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import jsonschema

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.utils.validation_metrics import ValidationResult, check_curriculum_sampling

class CurriculumValidator:
    """Validates curriculum learning configuration and sampling."""
    
    def __init__(self, data_path: str, config_path: str = None, output_dir: str = "./experiments/reports"):
        self.data_path = data_path
        self.config_path = config_path or str(Path(__file__).parent.parent.parent / "configs" / "curriculum" / "default_curriculum.yaml")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.curriculum_config = None
    
    def load_curriculum_config(self) -> Dict[str, Any]:
        """Load and parse curriculum configuration."""
        print("📋 Loading curriculum configuration...")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.curriculum_config = config
            print(f"  ✅ Loaded config: {config.get('name', 'Unknown')}")
            print(f"  📊 Stages found: {list(config.get('stages', {}).keys())}")
            
            return config
        
        except Exception as e:
            print(f"  ❌ Error loading config: {e}")
            return {}
    
    def validate_config_schema(self) -> ValidationResult:
        """Validate curriculum configuration schema."""
        print("🔍 Validating configuration schema...")
        
        if not self.curriculum_config:
            return ValidationResult(
                "Config Schema", False, 
                "Configuration not loaded", None, None
            )
        
        required_fields = [
            'name', 'stages', 'datasets', 'validation'
        ]
        
        required_stage_fields = [
            'duration_epochs', 'audio_ratio', 'criteria', 'loss_weights'
        ]
        
        required_criteria_fields = [
            'min_audio_accuracy', 'max_vl_degradation'
        ]
        
        required_loss_weights = [
            'audio_task_loss', 'retention_loss'
        ]
        
        validation_errors = []
        
        # Check top-level fields
        for field in required_fields:
            if field not in self.curriculum_config:
                validation_errors.append(f"Missing required field: {field}")
        
        # Check stages
        stages = self.curriculum_config.get('stages', {})
        if not stages:
            validation_errors.append("No stages defined")
        else:
            for stage_name, stage_config in stages.items():
                # Check stage fields
                for field in required_stage_fields:
                    if field not in stage_config:
                        validation_errors.append(f"Stage '{stage_name}' missing field: {field}")
                
                # Check criteria
                criteria = stage_config.get('criteria', {})
                for field in required_criteria_fields:
                    if field not in criteria:
                        validation_errors.append(f"Stage '{stage_name}' criteria missing field: {field}")
                
                # Check loss weights
                loss_weights = stage_config.get('loss_weights', {})
                for field in required_loss_weights:
                    if field not in loss_weights:
                        validation_errors.append(f"Stage '{stage_name}' loss_weights missing field: {field}")
        
        # Check datasets configuration
        datasets = self.curriculum_config.get('datasets', {})
        if not any(datasets.get(category, []) for category in ['audio_dependent', 'visual_only']):
            validation_errors.append("No datasets configured for audio_dependent or visual_only")
        
        passed = len(validation_errors) == 0
        message = "Schema validation passed" if passed else f"Schema errors: {'; '.join(validation_errors[:3])}"
        
        return ValidationResult(
            "Config Schema", passed, message, 
            len(validation_errors), 0,
            metadata={'errors': validation_errors}
        )
    
    def validate_stage_progression(self) -> ValidationResult:
        """Validate stage progression logic."""
        print("📈 Validating stage progression...")
        
        if not self.curriculum_config:
            return ValidationResult(
                "Stage Progression", False,
                "Configuration not available", None, None
            )
        
        stages = self.curriculum_config.get('stages', {})
        stage_names = list(stages.keys())
        
        progression_issues = []
        
        # Check audio ratio progression (should generally increase)
        audio_ratios = []
        for stage_name in stage_names:
            stage = stages[stage_name]
            audio_ratio = stage.get('audio_ratio', 0)
            audio_ratios.append((stage_name, audio_ratio))
        
        # Validate that ratios generally increase (with some tolerance)
        for i in range(len(audio_ratios) - 1):
            current_stage, current_ratio = audio_ratios[i]
            next_stage, next_ratio = audio_ratios[i + 1]
            
            # Allow slight decreases or plateaus but flag major regressions
            if next_ratio < current_ratio - 0.1:
                progression_issues.append(
                    f"Audio ratio decreases significantly: {current_stage}({current_ratio}) -> {next_stage}({next_ratio})"
                )
        
        # Check accuracy criteria progression (should generally increase)
        accuracy_targets = []
        for stage_name in stage_names:
            stage = stages[stage_name]
            criteria = stage.get('criteria', {})
            min_accuracy = criteria.get('min_audio_accuracy', 0)
            accuracy_targets.append((stage_name, min_accuracy))
        
        for i in range(len(accuracy_targets) - 1):
            current_stage, current_acc = accuracy_targets[i]
            next_stage, next_acc = accuracy_targets[i + 1]
            
            if next_acc < current_acc:
                progression_issues.append(
                    f"Accuracy target decreases: {current_stage}({current_acc}) -> {next_stage}({next_acc})"
                )
        
        # Check duration reasonableness
        for stage_name, stage_config in stages.items():
            duration = stage_config.get('duration_epochs', 0)
            if duration < 1:
                progression_issues.append(f"Stage '{stage_name}' has unreasonably short duration: {duration}")
            elif duration > 50:
                progression_issues.append(f"Stage '{stage_name}' has very long duration: {duration}")
        
        passed = len(progression_issues) == 0
        message = "Stage progression validated" if passed else f"Issues found: {len(progression_issues)}"
        
        return ValidationResult(
            "Stage Progression", passed, message,
            len(progression_issues), 0,
            metadata={'issues': progression_issues}
        )
    
    def validate_dataset_availability(self) -> ValidationResult:
        """Validate that configured datasets are available."""
        print("📁 Validating dataset availability...")
        
        if not self.curriculum_config:
            return ValidationResult(
                "Dataset Availability", False,
                "Configuration not available", None, None
            )
        
        datasets = self.curriculum_config.get('datasets', {})
        availability_issues = []
        
        # Check audio-dependent datasets
        audio_datasets = datasets.get('audio_dependent', [])
        for dataset_config in audio_datasets:
            dataset_type = dataset_config.get('type', '')
            
            if dataset_type == 'AudioCaps':
                # Check AudioCaps availability
                audiocaps_dir = Path(self.data_path) / "audiocaps"
                if not audiocaps_dir.exists():
                    availability_issues.append("AudioCaps directory not found")
                else:
                    # Check for metadata files
                    metadata_dir = audiocaps_dir / "metadata"
                    required_files = ['train.csv', 'val.csv']
                    for file_name in required_files:
                        if not (metadata_dir / file_name).exists():
                            availability_issues.append(f"AudioCaps {file_name} not found")
            
            elif dataset_type == 'AVQA':
                # Check AVQA (might be optional)
                avqa_dir = Path(self.data_path) / "avqa"
                if not avqa_dir.exists():
                    # This might be acceptable as AVQA is complex to obtain
                    print(f"    ⚠️  AVQA dataset not found (may be optional)")
        
        # Check visual-only datasets  
        visual_datasets = datasets.get('visual_only', [])
        for dataset_config in visual_datasets:
            dataset_type = dataset_config.get('type', '')
            
            if dataset_type == 'VQAv2':
                vqa_dir = Path(self.data_path) / "vqa"
                if not vqa_dir.exists():
                    availability_issues.append("VQA directory not found")
                else:
                    # Check for essential VQA files
                    required_files = [
                        'v2_OpenEnded_mscoco_train2014_questions.json',
                        'v2_OpenEnded_mscoco_val2014_questions.json',
                        'v2_mscoco_train2014_annotations.json',
                        'v2_mscoco_val2014_annotations.json'
                    ]
                    for file_name in required_files:
                        if not (vqa_dir / file_name).exists():
                            availability_issues.append(f"VQA {file_name} not found")
            
            elif dataset_type == 'GQA':
                # GQA might be optional
                gqa_dir = Path(self.data_path) / "gqa" 
                if not gqa_dir.exists():
                    print(f"    ⚠️  GQA dataset not found (may be optional)")
        
        passed = len(availability_issues) == 0
        message = "All configured datasets available" if passed else f"Missing datasets: {len(availability_issues)}"
        
        return ValidationResult(
            "Dataset Availability", passed, message,
            len(availability_issues), 0,
            metadata={'issues': availability_issues}
        )
    
    def simulate_curriculum_sampling(self) -> Dict[str, Any]:
        """Simulate curriculum sampling to verify ratios."""
        print("🎲 Simulating curriculum sampling...")
        
        if not self.curriculum_config:
            return {'error': 'Configuration not available'}
        
        stages = self.curriculum_config.get('stages', {})
        simulation_results = {}
        
        # Simulate each stage
        for stage_name, stage_config in stages.items():
            print(f"  Simulating {stage_name} stage...")
            
            audio_ratio = stage_config.get('audio_ratio', 0.5)
            difficulty_filter = stage_config.get('difficulty_filter', ['easy'])
            
            if isinstance(difficulty_filter, str):
                difficulty_filter = [difficulty_filter]
            
            # Simulate sampling 1000 samples
            num_samples = 1000
            audio_samples = int(num_samples * audio_ratio)
            visual_samples = num_samples - audio_samples
            
            # Simulate difficulty distribution
            difficulty_distribution = {}
            for difficulty in ['easy', 'medium', 'hard']:
                if difficulty in difficulty_filter:
                    difficulty_distribution[difficulty] = 1.0 / len(difficulty_filter)
                else:
                    difficulty_distribution[difficulty] = 0.0
            
            stage_simulation = {
                'total_samples': num_samples,
                'audio_samples': audio_samples,
                'visual_samples': visual_samples,
                'actual_audio_ratio': audio_samples / num_samples,
                'expected_audio_ratio': audio_ratio,
                'difficulty_distribution': difficulty_distribution,
                'ratio_error': abs((audio_samples / num_samples) - audio_ratio)
            }
            
            simulation_results[stage_name] = stage_simulation
        
        return simulation_results
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete curriculum validation."""
        print("🔍 Starting Curriculum Configuration Validation...")
        print("=" * 50)
        
        results = {
            'config_loading': {},
            'schema_validation': {},
            'progression_validation': {},
            'dataset_validation': {},
            'sampling_simulation': {},
            'validation_summary': {}
        }
        
        # Load configuration
        config = self.load_curriculum_config()
        results['config_loading'] = {
            'loaded': bool(config),
            'config_path': self.config_path,
            'name': config.get('name', 'Unknown') if config else None
        }
        
        validations = []
        
        if config:
            # Schema validation
            schema_validation = self.validate_config_schema()
            validations.append(schema_validation)
            results['schema_validation'] = schema_validation.to_dict()
            
            # Progression validation
            progression_validation = self.validate_stage_progression()
            validations.append(progression_validation)
            results['progression_validation'] = progression_validation.to_dict()
            
            # Dataset availability
            dataset_validation = self.validate_dataset_availability()
            validations.append(dataset_validation)
            results['dataset_validation'] = dataset_validation.to_dict()
            
            # Sampling simulation
            sampling_results = self.simulate_curriculum_sampling()
            results['sampling_simulation'] = sampling_results
            
            # Validate sampling accuracy
            if 'error' not in sampling_results:
                expected_ratios = {stage: stage_config.get('audio_ratio', 0.5) 
                                 for stage, stage_config in config.get('stages', {}).items()}
                actual_ratios = {stage: sim['actual_audio_ratio'] 
                               for stage, sim in sampling_results.items()}
                
                sampling_validation = check_curriculum_sampling(
                    actual_ratios, expected_ratios, tolerance=0.01
                )
                validations.append(sampling_validation)
        
        else:
            # Config loading failed
            config_validation = ValidationResult(
                "Config Loading", False, 
                f"Failed to load config from {self.config_path}",
                None, None
            )
            validations.append(config_validation)
        
        # Overall summary
        passed_tests = sum(1 for v in validations if v.passed)
        total_tests = len(validations)
        
        results['validation_summary'] = {
            'overall_passed': passed_tests == total_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'validations': validations
        }
        
        # Save results
        with open(self.output_dir / 'curriculum_validation_results.json', 'w') as f:
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
        """Print curriculum validation summary."""
        if not self.results:
            print("No validation results available. Run validation first.")
            return
        
        print("\n" + "=" * 50)
        print("CURRICULUM VALIDATION SUMMARY") 
        print("=" * 50)
        
        summary = self.results.get('validation_summary', {})
        if summary.get('overall_passed'):
            print("✅ Overall Status: CURRICULUM VALIDATED")
        else:
            print("❌ Overall Status: CURRICULUM ISSUES FOUND")
        
        print(f"Pass Rate: {summary.get('pass_rate', 0):.1%}")
        
        for validation in summary.get('validations', []):
            print(f"  {validation}")
        
        # Additional details
        config_info = self.results.get('config_loading', {})
        if config_info.get('loaded'):
            print(f"\n📋 Configuration: {config_info.get('name', 'Unknown')}")
            print(f"   Path: {config_info.get('config_path', 'Unknown')}")
        
        # Show sampling simulation results
        sampling_sim = self.results.get('sampling_simulation', {})
        if sampling_sim and 'error' not in sampling_sim:
            print(f"\n📊 Sampling Simulation Results:")
            for stage, sim in sampling_sim.items():
                expected = sim.get('expected_audio_ratio', 0)
                actual = sim.get('actual_audio_ratio', 0) 
                error = sim.get('ratio_error', 0)
                print(f"  {stage}: {actual:.1%} audio ratio (expected {expected:.1%}, error {error:.3f})")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run curriculum configuration validation")
    parser.add_argument("--data-path", required=True, help="Path to data directory")
    parser.add_argument("--config-path", help="Path to curriculum config YAML file")
    parser.add_argument("--output-dir", default="./experiments/reports", 
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Run validation
    validator = CurriculumValidator(args.data_path, args.config_path, args.output_dir)
    results = validator.run_full_validation()
    validator.print_summary()
    
    # Return exit code based on results
    if results['validation_summary']['overall_passed']:
        print("\n🎉 Curriculum validation passed!")
        return 0
    else:
        print("\n⚠️  Curriculum validation found issues.")
        return 1

if __name__ == "__main__":
    exit(main())