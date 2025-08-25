"""
Parameter Freeze Audit

Validates that only intended parameters are trainable in SAFE model:
- Verifies backbone models (BLIP-2, CLAP) are 100% frozen
- Confirms only projector + LoRA fusion layers are trainable
- Validates trainable parameter count is within expected range
"""

import os
import sys
import json
import torch
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from safe.models.safe_model import SAFEModel
from experiments.utils.model_validation import ModelParameterAnalyzer, count_model_parameters, format_parameter_count
from experiments.utils.validation_metrics import ValidationResult, aggregate_validation_results

class ParameterAuditor:
    """Comprehensive parameter freeze audit for SAFE model."""
    
    def __init__(self, output_dir: str = "./experiments/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.model = None
        self.analyzer = None
    
    def load_safe_model(self, config: Dict[str, Any] = None) -> SAFEModel:
        """Load SAFE model with specified configuration."""
        print("🤖 Loading SAFE model for parameter audit...")
        
        # Default configuration for testing
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
        
        # Merge provided config with defaults
        model_config = default_config.copy()
        if config:
            model_config.update(config)
        
        try:
            self.model = SAFEModel(**model_config)
            print(f"  ✅ Model loaded successfully")
            print(f"  📊 Model type: SAFE with {model_config['audio_encoder_type']} audio encoder")
            
            # Initialize analyzer
            self.analyzer = ModelParameterAnalyzer(self.model)
            
            return self.model
            
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            raise
    
    def audit_parameter_structure(self) -> Dict[str, Any]:
        """Analyze complete parameter structure."""
        print("🔍 Analyzing parameter structure...")
        
        if not self.model or not self.analyzer:
            raise ValueError("Model not loaded. Call load_safe_model() first.")
        
        # Analyze all parameters
        param_info = self.analyzer.analyze_parameters()
        
        # Print summary
        totals = param_info['totals']
        print(f"  📊 Total parameters: {format_parameter_count(totals['total_params'])}")
        print(f"  🔓 Trainable parameters: {format_parameter_count(totals['trainable_params'])} ({totals.get('trainable_ratio', 0):.1%})")
        print(f"  🔒 Frozen parameters: {format_parameter_count(totals['frozen_params'])} ({totals.get('frozen_ratio', 0):.1%})")
        
        # Component breakdown
        print(f"\n📋 Component Breakdown:")
        for component, data in param_info['components'].items():
            trainable_count = data['trainable_params']
            total_count = data['total_params']
            ratio = data.get('trainable_ratio', 0)
            status = "🔓 TRAINABLE" if ratio > 0 else "🔒 FROZEN"
            
            print(f"  {status} {component}: {format_parameter_count(trainable_count)}/{format_parameter_count(total_count)} ({ratio:.1%})")
        
        return param_info
    
    def validate_freeze_requirements(self) -> List[ValidationResult]:
        """Validate parameter freeze requirements against SAFE specifications."""
        print("✅ Validating parameter freeze requirements...")
        
        if not self.analyzer or not self.analyzer.parameter_info:
            raise ValueError("Parameter analysis not performed. Call audit_parameter_structure() first.")
        
        validations = []
        
        # Expected frozen components (should be 100% frozen)
        expected_frozen = [
            'base_vl_language',  # LLM backbone
            'base_vl_vision',    # Vision backbone  
            'vision_encoder',    # CLIP/ViT encoder
            'audio_encoder'      # CLAP/Whisper encoder
        ]
        
        # Expected trainable components (should have some trainable parameters)
        expected_trainable = [
            'audio_projector',   # Audio projection layer
            'fusion_adapter',    # LoRA fusion adapter
            'embeddings'         # Token embeddings (may be resized)
        ]
        
        # Validate freeze requirements
        freeze_validations = self.analyzer.validate_freeze_requirements(
            expected_frozen, expected_trainable
        )
        validations.extend(freeze_validations)
        
        # Validate total trainable ratio (should be ~0.2% of total parameters)
        ratio_validation = self.analyzer.validate_trainable_ratio(
            min_ratio=0.001,  # 0.1% minimum
            max_ratio=0.010   # 1.0% maximum (more lenient for different model sizes)
        )
        validations.append(ratio_validation)
        
        # Validate absolute parameter count (10M-50M trainable)
        totals = self.analyzer.parameter_info['totals']
        trainable_count = totals['trainable_params']
        
        count_validation = ValidationResult(
            "trainable_parameter_count",
            10_000_000 <= trainable_count <= 50_000_000,
            f"Trainable parameter count: {format_parameter_count(trainable_count)} ({trainable_count:,})",
            trainable_count, (10_000_000, 50_000_000)
        )
        validations.append(count_validation)
        
        # Print validation results
        print(f"  🧪 Running {len(validations)} validation checks...")
        passed = sum(1 for v in validations if v.passed)
        print(f"  ✅ Passed: {passed}/{len(validations)}")
        
        for validation in validations:
            status = "✅" if validation.passed else "❌"
            print(f"    {status} {validation.test_name}: {validation.message}")
        
        return validations
    
    def generate_parameter_visualization(self) -> None:
        """Generate parameter distribution visualization."""
        print("📊 Generating parameter visualization...")
        
        if not self.analyzer or not self.analyzer.parameter_info:
            return
        
        param_info = self.analyzer.parameter_info
        components = param_info['components']
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Component parameter counts (log scale)
        component_names = list(components.keys())
        total_params = [components[comp]['total_params'] for comp in component_names]
        trainable_params = [components[comp]['trainable_params'] for comp in component_names]
        
        x = range(len(component_names))
        ax1.bar(x, total_params, alpha=0.7, label='Total', color='lightblue')
        ax1.bar(x, trainable_params, alpha=0.9, label='Trainable', color='orange')
        ax1.set_yscale('log')
        ax1.set_xlabel('Model Components')
        ax1.set_ylabel('Parameter Count (log scale)')
        ax1.set_title('Parameter Count by Component')
        ax1.set_xticks(x)
        ax1.set_xticklabels(component_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Trainable ratio by component
        trainable_ratios = [components[comp].get('trainable_ratio', 0) * 100 for comp in component_names]
        bars = ax2.bar(x, trainable_ratios, color=['red' if r == 0 else 'green' for r in trainable_ratios])
        ax2.set_xlabel('Model Components')
        ax2.set_ylabel('Trainable Percentage (%)')
        ax2.set_title('Trainable Parameter Percentage by Component')
        ax2.set_xticks(x)
        ax2.set_xticklabels(component_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, ratio in zip(bars, trainable_ratios):
            if ratio > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{ratio:.1f}%', ha='center', va='bottom')
        
        # 3. Overall parameter distribution (pie chart)
        total_trainable = param_info['totals']['trainable_params']
        total_frozen = param_info['totals']['frozen_params']
        
        ax3.pie([total_frozen, total_trainable], 
               labels=[f'Frozen\\n{format_parameter_count(total_frozen)}', 
                      f'Trainable\\n{format_parameter_count(total_trainable)}'],
               autopct='%1.2f%%', colors=['lightcoral', 'lightgreen'],
               startangle=90)
        ax3.set_title('Overall Parameter Distribution')
        
        # 4. Parameter count table (text)
        ax4.axis('off')
        table_data = []
        
        for comp_name, comp_data in components.items():
            table_data.append([
                comp_name,
                format_parameter_count(comp_data['total_params']),
                format_parameter_count(comp_data['trainable_params']),
                f"{comp_data.get('trainable_ratio', 0)*100:.1f}%"
            ])
        
        # Add total row
        table_data.append([
            'TOTAL',
            format_parameter_count(param_info['totals']['total_params']),
            format_parameter_count(param_info['totals']['trainable_params']),
            f"{param_info['totals'].get('trainable_ratio', 0)*100:.1f}%"
        ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Component', 'Total', 'Trainable', 'Ratio'],
                         cellLoc='left',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Parameter Summary Table', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'parameter_audit_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✅ Visualization saved to: {plot_path}")
    
    def run_full_audit(self, model_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run complete parameter freeze audit."""
        print("🔍 Starting Parameter Freeze Audit...")
        print("=" * 50)
        
        results = {
            'audit_info': {
                'start_time': datetime.now().isoformat(),
                'model_config': model_config or {}
            },
            'parameter_analysis': {},
            'freeze_validation': {},
            'validation_summary': {}
        }
        
        try:
            # Load model
            self.load_safe_model(model_config)
            
            # Analyze parameter structure
            param_analysis = self.audit_parameter_structure()
            results['parameter_analysis'] = param_analysis
            
            # Validate freeze requirements
            validations = self.validate_freeze_requirements()
            results['freeze_validation'] = {
                'validations': validations,
                'individual_results': [v.to_dict() for v in validations]
            }
            
            # Generate visualizations
            self.generate_parameter_visualization()
            
            # Overall validation summary
            validation_summary = aggregate_validation_results(validations)
            results['validation_summary'] = validation_summary
            
            # Add completion info
            results['audit_info']['end_time'] = datetime.now().isoformat()
            results['audit_info']['success'] = True
            
        except Exception as e:
            results['audit_info']['error'] = str(e)
            results['audit_info']['success'] = False
            print(f"❌ Audit failed: {e}")
        
        # Save results
        with open(self.output_dir / 'parameter_audit_results.json', 'w') as f:
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
        """Print parameter audit summary."""
        if not self.results:
            print("No audit results available. Run audit first.")
            return
        
        print("\n" + "=" * 50)
        print("PARAMETER FREEZE AUDIT SUMMARY")
        print("=" * 50)
        
        # Overall status
        summary = self.results.get('validation_summary', {})
        if summary.get('overall_passed'):
            print("✅ Overall Status: PARAMETER FREEZE VALIDATED")
        else:
            print("❌ Overall Status: PARAMETER FREEZE ISSUES FOUND")
        
        print(f"Pass Rate: {summary.get('pass_rate', 0):.1%}")
        
        # Parameter statistics
        param_analysis = self.results.get('parameter_analysis', {})
        if 'totals' in param_analysis:
            totals = param_analysis['totals']
            print(f"\n📊 Parameter Statistics:")
            print(f"  Total: {format_parameter_count(totals['total_params'])}")
            print(f"  Trainable: {format_parameter_count(totals['trainable_params'])} ({totals.get('trainable_ratio', 0):.2%})")
            print(f"  Frozen: {format_parameter_count(totals['frozen_params'])} ({totals.get('frozen_ratio', 0):.2%})")
        
        # Validation details
        validations = summary.get('results', [])
        if validations:
            print(f"\n🧪 Validation Results:")
            for validation in validations:
                status = "✅" if validation.get('passed', False) else "❌"
                test_name = validation.get('test_name', 'Unknown')
                message = validation.get('message', '')
                print(f"  {status} {test_name}: {message}")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run parameter freeze audit")
    parser.add_argument("--output-dir", default="./experiments/reports", 
                       help="Output directory for reports")
    parser.add_argument("--config", help="JSON file with model configuration")
    
    args = parser.parse_args()
    
    # Load model config if provided
    model_config = None
    if args.config:
        with open(args.config, 'r') as f:
            model_config = json.load(f)
    
    # Run audit
    auditor = ParameterAuditor(args.output_dir)
    results = auditor.run_full_audit(model_config)
    auditor.print_summary()
    
    # Return exit code based on results
    if results['validation_summary']['overall_passed']:
        print("\n🎉 Parameter freeze audit passed!")
        return 0
    else:
        print("\n⚠️  Parameter freeze audit found issues.")
        return 1

if __name__ == "__main__":
    exit(main())