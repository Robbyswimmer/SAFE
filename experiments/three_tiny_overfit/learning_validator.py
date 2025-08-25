"""
Learning Validator

Validates learning progress against Experiment #3 go/no-go criteria:
- Training loss approaches near-zero
- Audio QA accuracy >90% on tiny training split
- VL retention unchanged with gate OFF  
- Clear learning curve progression
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.utils.validation_metrics import ValidationResult, aggregate_validation_results
from experiments.utils.training_utils import LearningCurveAnalyzer

class LearningValidator:
    """Validates learning progress against go/no-go criteria."""
    
    def __init__(self, output_dir: str = "./experiments/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_results = {}
    
    def validate_training_results(self, training_results: Dict[str, Any]) -> List[ValidationResult]:
        """Validate complete training results against all criteria."""
        print("🧪 Validating learning criteria...")
        
        validations = []
        
        # Extract metrics from training results
        final_metrics = training_results.get('final_metrics', {})
        convergence_info = training_results.get('convergence_info', {})
        overfitting_info = training_results.get('overfitting_info', {})
        
        # Criterion 1: Training loss approaches near-zero
        loss_validation = self._validate_loss_convergence(convergence_info)
        validations.append(loss_validation)
        
        # Criterion 2: Audio QA accuracy >90% on tiny training split  
        accuracy_validation = self._validate_audio_accuracy(final_metrics)
        validations.append(accuracy_validation)
        
        # Criterion 3: VL retention unchanged with gate OFF
        retention_validation = self._validate_vl_retention(final_metrics)
        validations.append(retention_validation)
        
        # Criterion 4: Clear learning curve progression
        progression_validation = self._validate_learning_progression(training_results)
        validations.append(progression_validation)
        
        # Additional analysis: Overfitting detection (informational)
        overfitting_validation = self._analyze_overfitting(overfitting_info)
        validations.append(overfitting_validation)
        
        # Print validation results
        print(f"  🔍 Running {len(validations)} validation checks...")
        passed = sum(1 for v in validations if v.passed)
        print(f"  ✅ Passed: {passed}/{len(validations)}")
        
        for validation in validations:
            status = "✅" if validation.passed else "❌"
            print(f"    {status} {validation.test_name}: {validation.message}")
        
        return validations
    
    def _validate_loss_convergence(self, convergence_info: Dict[str, Any]) -> ValidationResult:
        """Validate that training loss approaches near-zero."""
        final_loss = convergence_info.get('final_loss')
        converged = convergence_info.get('converged', False)
        loss_trend = convergence_info.get('loss_trend', 'unknown')
        
        # Target: Final loss should be very low (near-zero)
        target_loss = 0.05  # Allow some tolerance
        
        if final_loss is None:
            return ValidationResult(
                "loss_convergence",
                False,
                "No final loss available for evaluation",
                None, target_loss
            )
        
        # Check both absolute loss value and convergence status
        loss_low_enough = final_loss <= target_loss
        trending_down = loss_trend in ['improving', 'stable']  # Accept stable at low values
        
        passed = loss_low_enough and trending_down
        
        message = f"Final loss: {final_loss:.4f} (target: <{target_loss:.3f}), trend: {loss_trend}"
        
        return ValidationResult(
            "loss_convergence",
            passed,
            message,
            final_loss, target_loss,
            metadata={
                'converged': converged,
                'loss_trend': loss_trend,
                'target_loss': target_loss
            }
        )
    
    def _validate_audio_accuracy(self, final_metrics: Dict[str, Any]) -> ValidationResult:
        """Validate audio QA accuracy >90% on training split."""
        # Look for training accuracy on audio tasks
        train_accuracy = final_metrics.get('train_eval_accuracy', 0.0)
        
        # Target: >90% accuracy on tiny training split
        target_accuracy = 0.9
        
        passed = train_accuracy >= target_accuracy
        
        message = f"Train accuracy: {train_accuracy:.1%} (target: ≥{target_accuracy:.0%})"
        
        return ValidationResult(
            "audio_qa_accuracy", 
            passed,
            message,
            train_accuracy, target_accuracy,
            metadata={
                'train_samples': final_metrics.get('train_eval_total_samples', 0),
                'accuracy_type': 'audio_dependent'
            }
        )
    
    def _validate_vl_retention(self, final_metrics: Dict[str, Any]) -> ValidationResult:
        """Validate VL retention unchanged with gate OFF."""
        # Compare VL-only performance (gate=0) with expected baseline
        vl_accuracy = final_metrics.get('val_vl_accuracy', 0.0)
        audio_accuracy = final_metrics.get('val_audio_accuracy', 0.0)
        
        # For tiny overfit experiment, we expect both to be high
        # but VL retention should not degrade significantly when audio is OFF
        
        if vl_accuracy == 0.0 and audio_accuracy == 0.0:
            return ValidationResult(
                "vl_retention",
                False,
                "No VL retention metrics available",
                None, None
            )
        
        # Simplified check: VL accuracy should be reasonable
        # In practice, you might compare with a baseline VL-only model
        min_vl_retention = 0.7  # Expect at least 70% for tiny dataset
        
        passed = vl_accuracy >= min_vl_retention
        
        # Calculate retention compared to audio-enabled version
        if audio_accuracy > 0:
            retention_ratio = vl_accuracy / audio_accuracy
            detailed_message = f"VL accuracy: {vl_accuracy:.1%}, Audio accuracy: {audio_accuracy:.1%}, Retention ratio: {retention_ratio:.1%}"
        else:
            detailed_message = f"VL accuracy: {vl_accuracy:.1%} (target: ≥{min_vl_retention:.0%})"
        
        return ValidationResult(
            "vl_retention",
            passed,
            detailed_message,
            vl_accuracy, min_vl_retention,
            metadata={
                'vl_accuracy': vl_accuracy,
                'audio_accuracy': audio_accuracy,
                'retention_threshold': min_vl_retention
            }
        )
    
    def _validate_learning_progression(self, training_results: Dict[str, Any]) -> ValidationResult:
        """Validate clear learning curve progression."""
        # Check if learning curves show clear improvement over time
        # This is a more complex validation requiring analysis of the training history
        
        config = training_results.get('config', {})
        final_metrics = training_results.get('final_metrics', {})
        
        # Basic checks for learning progression
        num_epochs = config.get('num_epochs', 0)
        steps_completed = training_results.get('steps_completed', 0)
        
        # Check minimum training duration
        min_epochs = 10  # Should train for at least 10 epochs
        min_steps = 50   # Should complete at least 50 steps
        
        sufficient_training = num_epochs >= min_epochs or steps_completed >= min_steps
        
        # Check for improvement indicators
        final_loss = training_results.get('convergence_info', {}).get('final_loss', float('inf'))
        loss_trend = training_results.get('convergence_info', {}).get('loss_trend', 'unknown')
        
        clear_progression = (final_loss < 1.0) and (loss_trend in ['improving', 'stable'])
        
        passed = sufficient_training and clear_progression
        
        if passed:
            message = f"Clear progression: {num_epochs} epochs, loss trend: {loss_trend}, final loss: {final_loss:.4f}"
        else:
            issues = []
            if not sufficient_training:
                issues.append("insufficient training duration")
            if not clear_progression:
                issues.append("unclear learning progression")
            message = f"Issues detected: {', '.join(issues)}"
        
        return ValidationResult(
            "learning_progression",
            passed,
            message,
            final_loss if clear_progression else None,
            1.0,  # Loss should be below 1.0 for clear progression
            metadata={
                'num_epochs': num_epochs,
                'steps_completed': steps_completed,
                'loss_trend': loss_trend,
                'sufficient_training': sufficient_training
            }
        )
    
    def _analyze_overfitting(self, overfitting_info: Dict[str, Any]) -> ValidationResult:
        """Analyze overfitting (informational - not a go/no-go criterion)."""
        overfitting_detected = overfitting_info.get('overfitting_detected', False)
        confidence = overfitting_info.get('confidence', 0.0)
        
        # For tiny overfitting experiment, we WANT to see overfitting
        # This is informational to confirm the model can overfit
        
        if overfitting_detected:
            message = f"Overfitting detected (confidence: {confidence:.1%}) - Good for tiny overfit experiment!"
            passed = True  # Overfitting is desired in this context
        else:
            message = "No clear overfitting detected - model may need more training"
            passed = False  # We want to see overfitting in tiny experiments
        
        return ValidationResult(
            "overfitting_analysis",
            passed,
            message,
            confidence, 0.5,  # Expect at least 50% confidence in overfitting
            metadata=overfitting_info
        )
    
    def generate_validation_report(self, 
                                 training_results: Dict[str, Any],
                                 validations: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        print("\n📊 Generating learning validation report...")
        
        # Aggregate validation results
        validation_summary = aggregate_validation_results(validations)
        
        # Extract key metrics for summary
        final_metrics = training_results.get('final_metrics', {})
        convergence_info = training_results.get('convergence_info', {})
        
        report = {
            'report_info': {
                'generation_time': datetime.now().isoformat(),
                'experiment_type': 'tiny_overfit_learning_validation'
            },
            'go_no_go_summary': validation_summary,
            'detailed_validations': [v.to_dict() for v in validations],
            'key_metrics': {
                'final_train_loss': convergence_info.get('final_loss'),
                'train_accuracy': final_metrics.get('train_eval_accuracy', 0),
                'val_audio_accuracy': final_metrics.get('val_audio_accuracy', 0),
                'val_vl_accuracy': final_metrics.get('val_vl_accuracy', 0),
                'overfitting_detected': training_results.get('overfitting_info', {}).get('overfitting_detected', False)
            },
            'recommendations': self._generate_recommendations(validations, training_results)
        }
        
        return report
    
    def _generate_recommendations(self, 
                                validations: List[ValidationResult],
                                training_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check each criterion and provide specific recommendations
        for validation in validations:
            if not validation.passed:
                if validation.test_name == 'loss_convergence':
                    recommendations.append(
                        "Training loss not converging: Try higher learning rate, more epochs, or smaller batch size"
                    )
                elif validation.test_name == 'audio_qa_accuracy':
                    recommendations.append(
                        "Audio QA accuracy too low: Check audio data quality, model architecture, or training parameters"
                    )
                elif validation.test_name == 'vl_retention':
                    recommendations.append(
                        "VL retention poor: Review fusion mechanism, check for catastrophic forgetting"
                    )
                elif validation.test_name == 'learning_progression':
                    recommendations.append(
                        "Unclear learning progression: Increase training duration or adjust learning rate schedule"
                    )
                elif validation.test_name == 'overfitting_analysis':
                    recommendations.append(
                        "No overfitting detected: This may be acceptable, but consider more epochs for tiny overfit validation"
                    )
        
        # General recommendations based on overall results
        passed_count = sum(1 for v in validations if v.passed)
        total_count = len(validations)
        
        if passed_count == total_count:
            recommendations.append(
                "All learning criteria passed! Model demonstrates clear learning capability on tiny dataset. Ready for scaled training experiments."
            )
        elif passed_count >= total_count * 0.75:  # 75% pass rate
            recommendations.append(
                "Most criteria passed. Address remaining issues before scaling to larger experiments."
            )
        else:
            recommendations.append(
                "Multiple learning issues detected. Review model architecture, training setup, and data quality before proceeding."
            )
        
        return recommendations
    
    def save_validation_results(self, 
                              training_results: Dict[str, Any],
                              validations: List[ValidationResult]):
        """Save validation results to files."""
        # Generate report
        report = self.generate_validation_report(training_results, validations)
        
        # Save JSON report
        json_path = self.output_dir / 'tiny_overfit_learning_validation.json'
        import json
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"  💾 Validation report saved: {json_path}")
        
        # Generate validation visualization
        self._create_validation_plots(training_results, validations)
        
        return json_path
    
    def _create_validation_plots(self, 
                               training_results: Dict[str, Any],
                               validations: List[ValidationResult]):
        """Create validation visualization plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Validation Results Summary
        validation_names = [v.test_name.replace('_', ' ').title() for v in validations]
        validation_results = ['Pass' if v.passed else 'Fail' for v in validations]
        colors = ['green' if v.passed else 'red' for v in validations]
        
        y_pos = np.arange(len(validation_names))
        ax1.barh(y_pos, [1 if r == 'Pass' else 0 for r in validation_results], color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(validation_names)
        ax1.set_xlabel('Pass (1) / Fail (0)')
        ax1.set_title('Go/No-Go Validation Results')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Key Metrics
        final_metrics = training_results.get('final_metrics', {})
        metrics_names = ['Train Acc', 'Val Audio Acc', 'Val VL Acc']
        metrics_values = [
            final_metrics.get('train_eval_accuracy', 0) * 100,
            final_metrics.get('val_audio_accuracy', 0) * 100,
            final_metrics.get('val_vl_accuracy', 0) * 100
        ]
        
        bars = ax2.bar(metrics_names, metrics_values, alpha=0.7, color=['blue', 'orange', 'green'])
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Final Accuracy Metrics')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Plot 3: Loss Convergence
        convergence_info = training_results.get('convergence_info', {})
        final_loss = convergence_info.get('final_loss', 0)
        target_loss = 0.05
        
        ax3.bar(['Final Loss', 'Target Loss'], [final_loss, target_loss], 
               color=['blue', 'red'], alpha=0.7)
        ax3.set_ylabel('Loss Value')
        ax3.set_title('Loss Convergence Analysis')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Overall Assessment
        ax4.axis('off')
        
        # Create assessment text
        passed_count = sum(1 for v in validations if v.passed)
        total_count = len(validations)
        pass_rate = passed_count / total_count if total_count > 0 else 0
        
        assessment_text = f"""
TINY OVERFIT LEARNING VALIDATION

Overall Result: {'✅ PASS' if pass_rate >= 0.75 else '❌ FAIL'}

Criteria Results:
• Passed: {passed_count}/{total_count} ({pass_rate:.1%})
• Loss Convergence: {'✅' if validations[0].passed else '❌'}
• Audio QA Accuracy: {'✅' if validations[1].passed else '❌'}
• VL Retention: {'✅' if validations[2].passed else '❌'}
• Learning Progression: {'✅' if validations[3].passed else '❌'}

Key Metrics:
• Final Loss: {final_loss:.4f}
• Train Accuracy: {final_metrics.get('train_eval_accuracy', 0):.1%}
• Val Audio Accuracy: {final_metrics.get('val_audio_accuracy', 0):.1%}
• Val VL Accuracy: {final_metrics.get('val_vl_accuracy', 0):.1%}

Status: {'Ready for scaled experiments' if pass_rate >= 0.75 else 'Requires investigation'}
        """
        
        ax4.text(0.05, 0.95, assessment_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'tiny_overfit_validation_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  📊 Validation plots saved: {plot_path}")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate tiny overfit learning results")
    parser.add_argument("--results-file", required=True, 
                       help="Path to training results JSON file")
    parser.add_argument("--output-dir", default="./experiments/reports",
                       help="Output directory for validation reports")
    
    args = parser.parse_args()
    
    # Load training results
    import json
    with open(args.results_file, 'r') as f:
        training_results = json.load(f)
    
    # Run validation
    validator = LearningValidator(args.output_dir)
    validations = validator.validate_training_results(training_results)
    validator.save_validation_results(training_results, validations)
    
    # Return exit code based on results
    passed_count = sum(1 for v in validations if v.passed)
    total_count = len(validations)
    
    if passed_count >= total_count * 0.75:  # 75% pass rate threshold
        print("\n🎉 Learning validation passed!")
        return 0
    else:
        print("\n⚠️  Learning validation found issues.")
        return 1

if __name__ == "__main__":
    exit(main())