"""
Tiny Overfit (Learning Path Sanity) - Main Orchestration Script

Coordinates complete tiny overfitting validation:
1. Create/load tiny dataset
2. Setup SAFE model for tiny-scale training
3. Execute intensive training with monitoring
4. Validate learning against go/no-go criteria  
5. Generate comprehensive learning validation report

Part of Experiment #3 in the SAFE training validation framework.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from safe.models.safe_model import SAFEModel
from experiments.three_tiny_overfit.tiny_dataset import TinyDatasetCreator
from experiments.three_tiny_overfit.tiny_trainer import TinyTrainer
from experiments.three_tiny_overfit.learning_validator import LearningValidator
from experiments.utils.validation_metrics import ValidationResult, aggregate_validation_results

class TinyOverfitValidator:
    """Orchestrates complete tiny overfitting validation experiment."""
    
    def __init__(self, data_path: str, output_dir: str = "./experiments/reports"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.dataset_creator = TinyDatasetCreator(
            data_path=data_path,
            output_dir=self.output_dir / "tiny_datasets"
        )
        self.learning_validator = LearningValidator(output_dir=output_dir)
        
        self.results = {
            'run_info': {
                'start_time': datetime.now().isoformat(),
                'data_path': str(data_path),
                'output_dir': str(output_dir)
            },
            'phases': {}
        }
    
    def create_or_load_tiny_dataset(self, 
                                  target_size: int = 512,
                                  seed: int = 42,
                                  force_recreate: bool = False) -> Dict[str, Any]:
        """Create new tiny dataset or load existing one."""
        print("\n🔍 Phase 1: Tiny Dataset Preparation")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Always create dataset fresh to avoid serialization issues
            print(f"  🔨 Creating fresh tiny dataset...")
            dataset_info = self.dataset_creator.create_tiny_dataset(
                target_size=target_size,
                audio_ratio=1.0,  # Always-on audio requirement
                seed=seed
            )
            
            # Validate dataset
            validations = self.dataset_creator.validate_tiny_dataset(dataset_info)
            
            # Debug: Show all validation results
            print(f"  🔍 Validation Results:")
            for validation in validations:
                status = "✅" if validation.passed else "❌"
                print(f"    {status} {validation.test_name}: {validation.message}")
            
            all_passed = all(v.passed for v in validations)
            print(f"  🔍 Overall validation: {'✅ PASSED' if all_passed else '❌ FAILED'}")
            
            # Store actual samples in memory for training (don't rely on JSON serialization)
            self.train_samples = dataset_info['samples']['train'] if 'samples' in dataset_info else []
            self.val_samples = dataset_info['samples']['val'] if 'samples' in dataset_info else []
            
            phase_result = {
                'status': 'completed',
                'passed': all(v.passed for v in validations),
                'duration': time.time() - start_time,
                'dataset_info': {
                    'total_samples': dataset_info['statistics']['total_samples'],
                    'train_samples': dataset_info['statistics']['train_samples'],
                    'val_samples': dataset_info['statistics']['val_samples'],
                    'audio_ratio': dataset_info['statistics']['audio_ratio']
                },
                'validations': validations
            }
            
            if phase_result['passed']:
                print("  ✅ Tiny dataset ready")
                self.dataset_creator.print_dataset_summary(dataset_info)
            else:
                print("  ❌ Dataset validation issues found")
                # Print specific validation failures
                for validation in validations:
                    if not validation.passed:
                        print(f"    ❌ {validation.test_name}: {validation.message}")
            
            return phase_result
            
        except Exception as e:
            return {
                'status': 'error',
                'passed': False,
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    def setup_safe_model(self, model_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Setup SAFE model for tiny-scale training."""
        print("\n🔍 Phase 2: SAFE Model Setup")
        print("-" * 40)
        
        start_time = time.time()
        
        # Default configuration optimized for tiny-scale training
        default_config = {
            "llm_model_name": "microsoft/DialoGPT-medium",
            "vision_model_name": "openai/clip-vit-large-patch14", 
            "audio_encoder_type": "clap",
            "projector_type": "standard",
            "fusion_type": "lora",
            "num_audio_tokens": 8,
            "lora_rank": 8,
            "freeze_base_vl": True,
            "freeze_audio_encoder": True
        }
        
        # Merge with provided config
        safe_config = default_config.copy()
        if model_config:
            safe_config.update(model_config)
        
        try:
            print("  🤖 Loading SAFE model...")
            safe_model = SAFEModel(**safe_config)
            
            # Verify model setup
            total_params = sum(p.numel() for p in safe_model.parameters())
            trainable_params = sum(p.numel() for p in safe_model.parameters() if p.requires_grad)
            
            phase_result = {
                'status': 'completed',
                'passed': True,
                'duration': time.time() - start_time,
                'model_info': {
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
                    'config': safe_config
                },
                'safe_model': safe_model
            }
            
            print(f"  ✅ Model loaded: {trainable_params:,} trainable / {total_params:,} total params")
            print(f"  📊 Trainable ratio: {phase_result['model_info']['trainable_ratio']:.2%}")
            
            return phase_result
            
        except Exception as e:
            return {
                'status': 'error',
                'passed': False,
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    def run_tiny_training(self, 
                         safe_model: SAFEModel,
                         dataset_info: Dict[str, Any],
                         training_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute tiny-scale training with intensive monitoring."""
        print("\n🔍 Phase 3: Tiny-Scale Training")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Use in-memory samples (not from JSON which only has metadata)
            train_samples = self.train_samples
            val_samples = self.val_samples
            
            print(f"  📊 Training on {len(train_samples)} samples")
            print(f"  📊 Validating on {len(val_samples)} samples")
            
            # Debug: Check sample content
            if train_samples:
                sample = train_samples[0]
                print(f"  🔍 Sample keys: {list(sample.keys())}")
                print(f"  🔍 Audio type: {type(sample.get('audio'))}")
                print(f"  🔍 Image type: {type(sample.get('image'))}")
                print(f"  🔍 Has audio: {sample.get('audio') is not None}")
                print(f"  🔍 Has image: {sample.get('image') is not None}")
                if sample.get('audio') is not None:
                    print(f"  🔍 Audio shape: {sample['audio'].shape if hasattr(sample['audio'], 'shape') else 'No shape'}")
                
                # Check a few more samples
                none_audio_count = sum(1 for s in train_samples[:10] if s.get('audio') is None)
                none_image_count = sum(1 for s in train_samples[:10] if s.get('image') is None)
                print(f"  🔍 In first 10 samples: {none_audio_count} have None audio, {none_image_count} have None image")
            
            # Initialize trainer
            trainer = TinyTrainer(
                safe_model=safe_model,
                train_samples=train_samples,
                val_samples=val_samples,
                config=training_config
            )
            
            # Execute training
            training_results = trainer.train()
            
            # Save learning curves
            curves_path = self.output_dir / 'tiny_training_curves.png'
            trainer.save_learning_curves(curves_path)
            
            phase_result = {
                'status': 'completed',
                'passed': True,  # Will be validated in next phase
                'duration': time.time() - start_time,
                'training_results': training_results,
                'curves_path': str(curves_path)
            }
            
            print(f"  ✅ Training completed in {phase_result['duration']:.1f}s")
            
            # Show key results
            final_metrics = training_results.get('final_metrics', {})
            print(f"  📈 Final train accuracy: {final_metrics.get('train_eval_accuracy', 0):.1%}")
            print(f"  📊 Final val accuracy: {final_metrics.get('val_audio_accuracy', 0):.1%}")
            
            return phase_result
            
        except Exception as e:
            print(f"  ❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'passed': False,
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    def validate_learning_criteria(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate learning against go/no-go criteria."""
        print("\n🔍 Phase 4: Learning Criteria Validation")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Run validation
            validations = self.learning_validator.validate_training_results(training_results)
            
            # Save validation results
            validation_report_path = self.learning_validator.save_validation_results(
                training_results, validations
            )
            
            # Aggregate results
            validation_summary = aggregate_validation_results(validations)
            
            phase_result = {
                'status': 'completed', 
                'passed': validation_summary['overall_passed'],
                'duration': time.time() - start_time,
                'validation_summary': validation_summary,
                'validations': validations,
                'report_path': str(validation_report_path)
            }
            
            print(f"  🧪 Validation completed: {validation_summary['pass_rate']:.1%} pass rate")
            
            if phase_result['passed']:
                print("  🎉 All learning criteria met!")
            else:
                print("  ⚠️  Some learning criteria failed")
            
            return phase_result
            
        except Exception as e:
            return {
                'status': 'error',
                'passed': False,
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive experiment report."""
        print("\n📊 Generating Final Experiment Report...")
        print("-" * 40)
        
        # Collect all phase results  
        all_validations = []
        phase_summaries = []
        
        for phase_name, phase_result in self.results['phases'].items():
            passed = phase_result.get('passed', False)
            duration = phase_result.get('duration', 0)
            
            # Create phase validation
            phase_validation = ValidationResult(
                test_name=f"phase_{phase_name}",
                passed=passed,
                message=f"Phase {phase_name}: {'Passed' if passed else 'Failed'}",
                metadata={'duration': duration}
            )
            all_validations.append(phase_validation)
            
            phase_summaries.append({
                'phase_name': phase_name,
                'passed': passed,
                'duration': duration,
                'status': phase_result.get('status', 'unknown')
            })
        
        # Overall experiment summary
        experiment_summary = aggregate_validation_results(all_validations)
        
        # Collect key metrics
        training_phase = self.results['phases'].get('tiny_training', {})
        validation_phase = self.results['phases'].get('learning_validation', {})
        
        key_metrics = {}
        if 'training_results' in training_phase:
            final_metrics = training_phase['training_results'].get('final_metrics', {})
            convergence_info = training_phase['training_results'].get('convergence_info', {})
            
            key_metrics = {
                'final_train_loss': convergence_info.get('final_loss'),
                'train_accuracy': final_metrics.get('train_eval_accuracy', 0),
                'val_audio_accuracy': final_metrics.get('val_audio_accuracy', 0),
                'val_vl_accuracy': final_metrics.get('val_vl_accuracy', 0),
                'learning_criteria_passed': validation_phase.get('passed', False)
            }
        
        # Generate recommendations
        recommendations = self._generate_experiment_recommendations()
        
        final_report = {
            'experiment_info': {
                'name': 'Tiny Overfit (Learning Path Sanity)',
                'completion_time': datetime.now().isoformat(),
                'total_duration': sum(p.get('duration', 0) for p in self.results['phases'].values())
            },
            'experiment_summary': experiment_summary,
            'phase_summaries': phase_summaries,
            'key_metrics': key_metrics,
            'go_no_go_result': experiment_summary['overall_passed'],
            'recommendations': recommendations
        }
        
        # Save report
        report_path = self.output_dir / 'tiny_overfit_experiment_report.json'
        with open(report_path, 'w') as f:
            serializable_report = self._make_serializable(final_report)
            json.dump(serializable_report, f, indent=2, default=str)
        
        print(f"  📄 Final report saved: {report_path}")
        return final_report
    
    def _generate_experiment_recommendations(self) -> List[str]:
        """Generate experiment-level recommendations."""
        recommendations = []
        
        # Check each phase
        for phase_name, phase_result in self.results['phases'].items():
            if not phase_result.get('passed', False):
                if phase_name == 'dataset_preparation':
                    recommendations.append(
                        "Dataset issues detected: Review data quality and sampling strategy"
                    )
                elif phase_name == 'model_setup':
                    recommendations.append(
                        "Model setup issues: Check architecture configuration and parameter counts"
                    )
                elif phase_name == 'tiny_training':
                    recommendations.append(
                        "Training issues detected: Review training configuration and monitor convergence"
                    )
                elif phase_name == 'learning_validation':
                    recommendations.append(
                        "Learning criteria not met: Address specific validation failures before scaling"
                    )
        
        # Overall assessment
        passed_phases = sum(1 for p in self.results['phases'].values() if p.get('passed', False))
        total_phases = len(self.results['phases'])
        
        if passed_phases == total_phases:
            recommendations.append(
                "🎉 All phases completed successfully! SAFE model demonstrates clear learning capability. Ready to proceed with larger-scale training experiments (Experiment #4: Projector + Fusion Pilot)."
            )
        elif passed_phases >= total_phases * 0.75:
            recommendations.append(
                "Most phases completed successfully. Address remaining issues before proceeding to scaled experiments."
            )
        else:
            recommendations.append(
                "Multiple experiment phases failed. Comprehensive review of model architecture, training setup, and data pipeline required before proceeding."
            )
        
        return recommendations
    
    def run_full_experiment(self,
                          target_dataset_size: int = 512,
                          dataset_seed: int = 42,
                          model_config: Dict[str, Any] = None,
                          training_config: Dict[str, Any] = None,
                          force_recreate_dataset: bool = False) -> Dict[str, Any]:
        """Run complete tiny overfitting validation experiment."""
        print("🔍 Starting Tiny Overfit (Learning Path Sanity) Experiment")
        print("=" * 60)
        print(f"Data Path: {self.data_path}")
        print(f"Output Dir: {self.output_dir}")
        print(f"Target Dataset Size: {target_dataset_size}")
        print(f"Start Time: {self.results['run_info']['start_time']}")
        
        # Phase 1: Dataset Preparation
        dataset_result = self.create_or_load_tiny_dataset(
            target_size=target_dataset_size,
            seed=dataset_seed,
            force_recreate=force_recreate_dataset
        )
        self.results['phases']['dataset_preparation'] = dataset_result
        
        if not dataset_result['passed']:
            print("❌ Dataset preparation failed - aborting experiment")
            return self.results
        
        # Phase 2: Model Setup
        model_result = self.setup_safe_model(model_config)
        self.results['phases']['model_setup'] = model_result
        
        if not model_result['passed']:
            print("❌ Model setup failed - aborting experiment")
            return self.results
        
        # Phase 3: Training
        training_result = self.run_tiny_training(
            model_result['safe_model'],
            dataset_result['dataset_info'],
            training_config
        )
        self.results['phases']['tiny_training'] = training_result
        
        if not training_result['passed']:
            print("❌ Training failed - aborting experiment") 
            return self.results
        
        # Phase 4: Learning Validation
        validation_result = self.validate_learning_criteria(
            training_result['training_results']
        )
        self.results['phases']['learning_validation'] = validation_result
        
        # Generate final report
        final_report = self.generate_final_report()
        self.results['final_report'] = final_report
        
        # Add completion info
        self.results['run_info']['end_time'] = datetime.now().isoformat()
        self.results['run_info']['total_duration'] = final_report['experiment_info']['total_duration']
        
        # Save complete results
        results_path = self.output_dir / 'tiny_overfit_complete_results.json'
        with open(results_path, 'w') as f:
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        return self.results
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects for JSON output."""
        if isinstance(obj, ValidationResult):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'state_dict'):  # PyTorch models
            return f"<PyTorch Model: {type(obj).__name__}>"
        else:
            return obj
    
    def print_final_summary(self):
        """Print final experiment summary."""
        if not self.results:
            print("No results available.")
            return
        
        print("\n" + "=" * 60)
        print("TINY OVERFIT EXPERIMENT SUMMARY")
        print("=" * 60)
        
        final_report = self.results.get('final_report', {})
        experiment_summary = final_report.get('experiment_summary', {})
        
        # Overall status
        if experiment_summary.get('overall_passed'):
            print("🎉 Overall Status: EXPERIMENT PASSED")
        else:
            print("❌ Overall Status: EXPERIMENT FAILED")
        
        print(f"Pass Rate: {experiment_summary.get('pass_rate', 0):.1%}")
        print(f"Total Duration: {final_report.get('experiment_info', {}).get('total_duration', 0):.1f}s")
        
        # Phase results
        phase_summaries = final_report.get('phase_summaries', [])
        if phase_summaries:
            print("\nPhase Results:")
            for phase in phase_summaries:
                status = "✅ PASS" if phase['passed'] else "❌ FAIL"
                duration = phase.get('duration', 0)
                print(f"  {status} {phase['phase_name']:<25} ({duration:.1f}s)")
        
        # Key metrics
        key_metrics = final_report.get('key_metrics', {})
        if key_metrics:
            print("\n📊 Key Metrics:")
            print(f"  Final Train Loss: {key_metrics.get('final_train_loss', 'N/A')}")
            print(f"  Train Accuracy: {key_metrics.get('train_accuracy', 0):.1%}")
            print(f"  Val Audio Accuracy: {key_metrics.get('val_audio_accuracy', 0):.1%}")
            print(f"  Val VL Accuracy: {key_metrics.get('val_vl_accuracy', 0):.1%}")
        
        # Recommendations
        recommendations = final_report.get('recommendations', [])
        if recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print(f"\n📁 Detailed reports saved to: {self.output_dir}")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tiny overfit learning validation experiment")
    parser.add_argument("--data-path", required=True, help="Path to data directory")
    parser.add_argument("--output-dir", default="./experiments/reports",
                       help="Output directory for reports")
    parser.add_argument("--dataset-size", type=int, default=750,
                       help="Target tiny dataset size")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for dataset creation")
    parser.add_argument("--model-config", help="JSON file with model configuration")
    parser.add_argument("--training-config", help="JSON file with training configuration")
    parser.add_argument("--force-recreate-dataset", action="store_true",
                       help="Force recreation of tiny dataset")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_path):
        print(f"❌ Data path does not exist: {args.data_path}")
        return 1
    
    # Load configurations
    model_config = None
    if args.model_config:
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
    
    training_config = None
    if args.training_config:
        with open(args.training_config, 'r') as f:
            training_config = json.load(f)
    
    # Run experiment
    validator = TinyOverfitValidator(args.data_path, args.output_dir)
    results = validator.run_full_experiment(
        target_dataset_size=args.dataset_size,
        dataset_seed=args.seed,
        model_config=model_config,
        training_config=training_config,
        force_recreate_dataset=args.force_recreate_dataset
    )
    validator.print_final_summary()
    
    # Return exit code based on results
    final_report = results.get('final_report', {})
    experiment_passed = final_report.get('go_no_go_result', False)
    
    if experiment_passed:
        print("\n🎉 Tiny overfit experiment completed successfully!")
        print("✅ Ready to proceed with Experiment #4: Projector + Fusion Pilot.")
        return 0
    else:
        print("\n⚠️  Tiny overfit experiment found issues.")
        print("❌ Please address the issues before proceeding with scaled training.")
        return 1

if __name__ == "__main__":
    exit(main())