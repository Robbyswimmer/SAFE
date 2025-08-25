"""
Model Safety & Robustness Baseline - Main Orchestration Script

Coordinates all safety validation tests:
1. Parameter freeze audit
2. Forward pass equivalence testing
3. Comprehensive safety validation reporting

Part of Experiment #2 in the SAFE training validation framework.
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

from experiments.two_model_safety.parameter_audit import ParameterAuditor
from experiments.two_model_safety.equivalence_test import EquivalenceValidator
from experiments.utils.validation_metrics import ValidationResult, aggregate_validation_results

class SafetyValidator:
    """Orchestrates all model safety validation tests."""
    
    def __init__(self, data_path: str, output_dir: str = "./experiments/reports"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            'run_info': {
                'start_time': datetime.now().isoformat(),
                'data_path': str(data_path),
                'output_dir': str(output_dir)
            },
            'tests': {}
        }
    
    def run_parameter_audit(self, model_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run parameter freeze audit test."""
        print("\n🔍 Running Parameter Freeze Audit...")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            auditor = ParameterAuditor(self.output_dir)
            results = auditor.run_full_audit(model_config)
            
            test_result = {
                'status': 'completed',
                'passed': results['validation_summary']['overall_passed'],
                'duration': time.time() - start_time,
                'results': results,
                'summary': f"Pass rate: {results['validation_summary']['pass_rate']:.1%}"
            }
            
            if results['validation_summary']['overall_passed']:
                print("✅ Parameter audit PASSED")
            else:
                print("❌ Parameter audit FAILED")
            
        except Exception as e:
            test_result = {
                'status': 'error',
                'passed': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'summary': f"Error: {e}"
            }
            print(f"❌ Parameter audit ERROR: {e}")
        
        return test_result
    
    def run_equivalence_test(self, 
                           model_config: Dict[str, Any] = None,
                           num_test_samples: int = 1000,
                           batch_size: int = 8) -> Dict[str, Any]:
        """Run forward pass equivalence test."""
        print("\n🔍 Running Forward Pass Equivalence Test...")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            validator = EquivalenceValidator(self.data_path, self.output_dir)
            results = validator.run_full_test(model_config, num_test_samples, batch_size)
            
            test_result = {
                'status': 'completed',
                'passed': results['validation_summary']['overall_passed'],
                'duration': time.time() - start_time,
                'results': results,
                'summary': f"Pass rate: {results['validation_summary']['pass_rate']:.1%}"
            }
            
            if results['validation_summary']['overall_passed']:
                print("✅ Equivalence test PASSED")
            else:
                print("❌ Equivalence test FAILED")
            
        except Exception as e:
            test_result = {
                'status': 'error',
                'passed': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'summary': f"Error: {e}"
            }
            print(f"❌ Equivalence test ERROR: {e}")
        
        return test_result
    
    def validate_model_configuration(self, model_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Basic validation of model configuration."""
        print("\n🔍 Validating Model Configuration...")
        print("-" * 40)
        
        start_time = time.time()
        
        # Default configuration for validation
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
        
        config_to_validate = model_config or default_config
        
        config_validations = []
        
        # Check critical freeze settings
        freeze_base_vl = config_to_validate.get('freeze_base_vl', False)
        freeze_audio_encoder = config_to_validate.get('freeze_audio_encoder', False)
        
        base_freeze_validation = ValidationResult(
            "freeze_base_vl_setting",
            freeze_base_vl,
            f"Base VL freeze setting: {freeze_base_vl}",
            freeze_base_vl, True
        )
        config_validations.append(base_freeze_validation)
        
        audio_freeze_validation = ValidationResult(
            "freeze_audio_encoder_setting",
            freeze_audio_encoder,
            f"Audio encoder freeze setting: {freeze_audio_encoder}",
            freeze_audio_encoder, True
        )
        config_validations.append(audio_freeze_validation)
        
        # Check reasonable parameter ranges
        lora_rank = config_to_validate.get('lora_rank', 8)
        num_audio_tokens = config_to_validate.get('num_audio_tokens', 8)
        
        lora_validation = ValidationResult(
            "lora_rank_range",
            1 <= lora_rank <= 64,
            f"LoRA rank: {lora_rank} (should be 1-64)",
            lora_rank, (1, 64)
        )
        config_validations.append(lora_validation)
        
        token_validation = ValidationResult(
            "audio_token_range",
            1 <= num_audio_tokens <= 32,
            f"Audio tokens: {num_audio_tokens} (should be 1-32)",
            num_audio_tokens, (1, 32)
        )
        config_validations.append(token_validation)
        
        # Check supported model types
        audio_encoder_type = config_to_validate.get('audio_encoder_type', 'clap')
        supported_audio_encoders = ['clap', 'whisper', 'multimodal']
        
        audio_encoder_validation = ValidationResult(
            "audio_encoder_support",
            audio_encoder_type in supported_audio_encoders,
            f"Audio encoder type: {audio_encoder_type} (supported: {supported_audio_encoders})",
            audio_encoder_type, supported_audio_encoders
        )
        config_validations.append(audio_encoder_validation)
        
        # Aggregate results
        passed_tests = sum(1 for v in config_validations if v.passed)
        total_tests = len(config_validations)
        
        test_result = {
            'status': 'completed',
            'passed': passed_tests == total_tests,
            'duration': time.time() - start_time,
            'results': {
                'config_tested': config_to_validate,
                'validations': config_validations,
                'summary': {
                    'overall_passed': passed_tests == total_tests,
                    'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
                    'total_tests': total_tests,
                    'passed_tests': passed_tests
                }
            },
            'summary': f"Pass rate: {passed_tests / total_tests if total_tests > 0 else 0:.1%}"
        }
        
        if passed_tests == total_tests:
            print("✅ Configuration validation PASSED")
        else:
            print("❌ Configuration validation FAILED")
        
        for validation in config_validations:
            status = "✅" if validation.passed else "❌"
            print(f"  {status} {validation.test_name}: {validation.message}")
        
        return test_result
    
    def generate_overall_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety validation report."""
        print("\n📊 Generating Overall Safety Report...")
        print("-" * 40)
        
        # Collect all validation results
        all_validations = []
        test_summaries = []
        
        for test_name, test_result in self.results['tests'].items():
            passed = test_result.get('passed', False)
            summary = test_result.get('summary', 'No summary')
            
            # Create a ValidationResult for consistency
            validation = ValidationResult(
                test_name=test_name,
                passed=passed,
                message=summary,
                metadata={'duration': test_result.get('duration', 0)}
            )
            all_validations.append(validation)
            
            test_summaries.append({
                'test_name': test_name,
                'passed': passed,
                'duration': test_result.get('duration', 0),
                'summary': summary
            })
        
        # Aggregate results
        overall_summary = aggregate_validation_results(all_validations)
        
        # Add timing information
        total_duration = sum(t.get('duration', 0) for t in self.results['tests'].values())
        
        report = {
            'safety_summary': overall_summary,
            'test_summaries': test_summaries,
            'timing': {
                'total_duration': total_duration,
                'individual_tests': {name: result.get('duration', 0) 
                                   for name, result in self.results['tests'].items()}
            },
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        for test_name, test_result in self.results['tests'].items():
            if not test_result.get('passed', False):
                if test_name == 'model_configuration':
                    recommendations.append(
                        "Review model configuration settings, ensure proper freeze parameters are set"
                    )
                elif test_name == 'parameter_audit':
                    recommendations.append(
                        "Review parameter freeze audit report and ensure backbone models are properly frozen"
                    )
                elif test_name == 'equivalence_test':
                    recommendations.append(
                        "Investigate forward pass equivalence issues - SAFE with gate=0 should match base VL model"
                    )
        
        if not recommendations:
            recommendations.append(
                "All model safety checks passed! SAFE model architecture is properly configured for safe training."
            )
        
        return recommendations
    
    def run_full_safety_validation(self, 
                                 model_config: Dict[str, Any] = None,
                                 num_test_samples: int = 1000,
                                 batch_size: int = 8) -> Dict[str, Any]:
        """Run all model safety validation tests."""
        print("🔍 Starting Model Safety & Robustness Baseline Validation")
        print("=" * 60)
        print(f"Data Path: {self.data_path}")
        print(f"Output Dir: {self.output_dir}")
        print(f"Start Time: {self.results['run_info']['start_time']}")
        
        # Run individual tests
        tests = [
            ('model_configuration', lambda: self.validate_model_configuration(model_config)),
            ('parameter_audit', lambda: self.run_parameter_audit(model_config)),
            ('equivalence_test', lambda: self.run_equivalence_test(model_config, num_test_samples, batch_size))
        ]
        
        for test_name, test_func in tests:
            try:
                self.results['tests'][test_name] = test_func()
            except Exception as e:
                print(f"❌ Failed to run {test_name}: {e}")
                self.results['tests'][test_name] = {
                    'status': 'error',
                    'passed': False,
                    'error': str(e),
                    'summary': f"Test failed: {e}"
                }
        
        # Generate overall report
        overall_report = self.generate_overall_report()
        self.results['overall_report'] = overall_report
        
        # Add completion info
        self.results['run_info']['end_time'] = datetime.now().isoformat()
        self.results['run_info']['total_duration'] = overall_report['timing']['total_duration']
        
        # Save comprehensive results
        with open(self.output_dir / 'model_safety_results.json', 'w') as f:
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        return self.results
    
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
    
    def print_final_summary(self):
        """Print final safety validation summary."""
        if not self.results:
            print("No results available.")
            return
        
        print("\n" + "=" * 60)
        print("MODEL SAFETY VALIDATION SUMMARY")
        print("=" * 60)
        
        overall_report = self.results.get('overall_report', {})
        safety_summary = overall_report.get('safety_summary', {})
        
        # Overall status
        if safety_summary.get('overall_passed'):
            print("🎉 Overall Status: ALL SAFETY TESTS PASSED")
        else:
            print("⚠️  Overall Status: SOME SAFETY TESTS FAILED")
        
        print(f"Pass Rate: {safety_summary.get('pass_rate', 0):.1%}")
        print(f"Total Duration: {overall_report.get('timing', {}).get('total_duration', 0):.1f}s")
        
        # Individual test results
        print("\nIndividual Test Results:")
        for test_summary in overall_report.get('test_summaries', []):
            status = "✅ PASS" if test_summary['passed'] else "❌ FAIL"
            duration = test_summary.get('duration', 0)
            print(f"  {status} {test_summary['test_name']:<25} ({duration:.1f}s) - {test_summary.get('summary', '')}")
        
        # Recommendations
        recommendations = overall_report.get('recommendations', [])
        if recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print(f"\n📁 Detailed reports saved to: {self.output_dir}")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run complete model safety validation")
    parser.add_argument("--data-path", required=True, help="Path to data directory")
    parser.add_argument("--output-dir", default="./experiments/reports", 
                       help="Output directory for reports")
    parser.add_argument("--config", help="JSON file with model configuration")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of test samples for equivalence test")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for equivalence test")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_path):
        print(f"❌ Data path does not exist: {args.data_path}")
        return 1
    
    # Load model config if provided
    model_config = None
    if args.config:
        with open(args.config, 'r') as f:
            model_config = json.load(f)
    
    # Run safety validation
    validator = SafetyValidator(args.data_path, args.output_dir)
    results = validator.run_full_safety_validation(
        model_config, args.num_samples, args.batch_size
    )
    validator.print_final_summary()
    
    # Return appropriate exit code
    overall_passed = results.get('overall_report', {}).get('safety_summary', {}).get('overall_passed', False)
    
    if overall_passed:
        print("\n🎉 Model safety validation completed successfully!")
        print("✅ Ready to proceed with learning validation experiments.")
        return 0
    else:
        print("\n⚠️  Model safety validation found issues.")
        print("❌ Please address the issues before proceeding with training.")
        return 1

if __name__ == "__main__":
    exit(main())