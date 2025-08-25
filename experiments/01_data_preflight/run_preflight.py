"""
Data & Pipeline Preflight - Main Orchestration Script

Runs all data validation tests in sequence:
1. Schema & Distribution Audit
2. ID Leakage Detection
3. Configuration Validation
4. Generate comprehensive report

Each test has clear go/no-go criteria that must be met.
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

from experiments.01_data_preflight.schema_audit import SchemaAuditor
from experiments.01_data_preflight.leakage_detection import LeakageDetector
from experiments.01_data_preflight.av_alignment_check import AVAlignmentValidator
from experiments.01_data_preflight.curriculum_validation import CurriculumValidator
from experiments.01_data_preflight.duplicate_detection import DuplicateDetector
from experiments.01_data_preflight.audio_analysis import AudioQualityAnalyzer
from experiments.01_data_preflight.report_generator import ComprehensiveReportGenerator
from experiments.utils.validation_metrics import ValidationResult, aggregate_validation_results

class PreflightRunner:
    """Orchestrates all preflight validation tests."""
    
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
    
    def run_schema_audit(self) -> Dict[str, Any]:
        """Run schema and distribution audit."""
        print("\n🔍 Running Schema & Distribution Audit...")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            auditor = SchemaAuditor(self.data_path, self.output_dir)
            results = auditor.run_full_audit()
            
            test_result = {
                'status': 'completed',
                'passed': results['summary']['overall_passed'],
                'duration': time.time() - start_time,
                'results': results,
                'summary': f"Pass rate: {results['summary']['pass_rate']:.1%}"
            }
            
            if results['summary']['overall_passed']:
                print("✅ Schema audit PASSED")
            else:
                print("❌ Schema audit FAILED")
            
        except Exception as e:
            test_result = {
                'status': 'error',
                'passed': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'summary': f"Error: {e}"
            }
            print(f"❌ Schema audit ERROR: {e}")
        
        return test_result
    
    def run_leakage_detection(self) -> Dict[str, Any]:
        """Run ID leakage detection."""
        print("\n🔍 Running ID Leakage Detection...")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            detector = LeakageDetector(self.data_path, self.output_dir)
            results = detector.run_full_detection()
            
            test_result = {
                'status': 'completed',
                'passed': results['summary']['overall_passed'],
                'duration': time.time() - start_time,
                'results': results,
                'summary': f"Pass rate: {results['summary']['pass_rate']:.1%}"
            }
            
            if results['summary']['overall_passed']:
                print("✅ Leakage detection PASSED")
            else:
                print("❌ Leakage detection FAILED")
            
        except Exception as e:
            test_result = {
                'status': 'error',
                'passed': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'summary': f"Error: {e}"
            }
            print(f"❌ Leakage detection ERROR: {e}")
        
        return test_result
    
    def validate_data_availability(self) -> Dict[str, Any]:
        """Basic validation that required data exists."""
        print("\n🔍 Validating Data Availability...")
        print("-" * 40)
        
        start_time = time.time()
        
        required_paths = {
            'AudioCaps metadata (train)': Path(self.data_path) / "audiocaps" / "metadata" / "train.csv",
            'AudioCaps metadata (val)': Path(self.data_path) / "audiocaps" / "metadata" / "val.csv",
            'VQA questions (train)': Path(self.data_path) / "vqa" / "v2_OpenEnded_mscoco_train2014_questions.json",
            'VQA questions (val)': Path(self.data_path) / "vqa" / "v2_OpenEnded_mscoco_val2014_questions.json",
            'VQA annotations (train)': Path(self.data_path) / "vqa" / "v2_mscoco_train2014_annotations.json",
            'VQA annotations (val)': Path(self.data_path) / "vqa" / "v2_mscoco_val2014_annotations.json"
        }\n        \n        optional_paths = {\n            'AudioCaps audio (train)': Path(self.data_path) / \"audiocaps\" / \"audio\" / \"train\",\n            'AudioCaps audio (val)': Path(self.data_path) / \"audiocaps\" / \"audio\" / \"val\",\n            'VQA images (train)': Path(self.data_path) / \"vqa\" / \"images\" / \"train2014\",\n            'VQA images (val)': Path(self.data_path) / \"vqa\" / \"images\" / \"val2014\"\n        }\n        \n        availability_results = {\n            'required': {},\n            'optional': {},\n            'all_required_present': True\n        }\n        \n        # Check required paths\n        for name, path in required_paths.items():\n            exists = path.exists()\n            availability_results['required'][name] = {\n                'path': str(path),\n                'exists': exists,\n                'is_file': path.is_file() if exists else False\n            }\n            \n            if exists:\n                print(f\"✅ {name}: Found\")\n            else:\n                print(f\"❌ {name}: Missing\")\n                availability_results['all_required_present'] = False\n        \n        # Check optional paths (directories with files)\n        for name, path in optional_paths.items():\n            exists = path.exists() and path.is_dir()\n            file_count = len(list(path.iterdir())) if exists else 0\n            \n            availability_results['optional'][name] = {\n                'path': str(path),\n                'exists': exists,\n                'file_count': file_count\n            }\n            \n            if exists and file_count > 0:\n                print(f\"✅ {name}: Found ({file_count} files)\")\n            else:\n                print(f\"⚠️  {name}: {'Empty' if exists else 'Missing'}\")\n        \n        test_result = {\n            'status': 'completed',\n            'passed': availability_results['all_required_present'],\n            'duration': time.time() - start_time,\n            'results': availability_results,\n            'summary': 'All required data available' if availability_results['all_required_present'] else 'Missing required data'\n        }\n        \n        return test_result\n    \n    def generate_overall_report(self) -> Dict[str, Any]:\n        \"\"\"Generate comprehensive preflight report.\"\"\"\n        print(\"\\n📊 Generating Overall Report...\")\n        print(\"-\" * 40)\n        \n        # Collect all validation results\n        all_validations = []\n        test_summaries = []\n        \n        for test_name, test_result in self.results['tests'].items():\n            passed = test_result.get('passed', False)\n            summary = test_result.get('summary', 'No summary')\n            \n            # Create a ValidationResult for consistency\n            validation = ValidationResult(\n                test_name=test_name,\n                passed=passed,\n                message=summary,\n                metadata={'duration': test_result.get('duration', 0)}\n            )\n            all_validations.append(validation)\n            \n            test_summaries.append({\n                'test_name': test_name,\n                'passed': passed,\n                'duration': test_result.get('duration', 0),\n                'summary': summary\n            })\n        \n        # Aggregate results\n        overall_summary = aggregate_validation_results(all_validations)\n        \n        # Add timing information\n        total_duration = sum(t.get('duration', 0) for t in self.results['tests'].values())\n        \n        report = {\n            'preflight_summary': overall_summary,\n            'test_summaries': test_summaries,\n            'timing': {\n                'total_duration': total_duration,\n                'individual_tests': {name: result.get('duration', 0) \n                                   for name, result in self.results['tests'].items()}\n            },\n            'recommendations': self.generate_recommendations()\n        }\n        \n        return report\n    \n    def generate_recommendations(self) -> List[str]:\n        \"\"\"Generate recommendations based on test results.\"\"\"\n        recommendations = []\n        \n        for test_name, test_result in self.results['tests'].items():\n            if not test_result.get('passed', False):\n                if test_name == 'data_availability':\n                    recommendations.append(\n                        \"Download missing required datasets before proceeding with training\"\n                    )\n                elif test_name == 'schema_audit':\n                    recommendations.append(\n                        \"Review schema audit report and fix data quality issues\"\n                    )\n                elif test_name == 'leakage_detection':\n                    recommendations.append(\n                        \"Investigate and resolve data leakage issues between splits\"\n                    )\n        \n        if not recommendations:\n            recommendations.append(\n                \"All preflight checks passed! Ready to proceed with training experiments.\"\n            )\n        \n        return recommendations\n    \n    def run_full_preflight(self) -> Dict[str, Any]:\n        \"\"\"Run all preflight validation tests.\"\"\"\n        print(\"🚀 Starting Data & Pipeline Preflight Validation\")\n        print(\"=\" * 60)\n        print(f\"Data Path: {self.data_path}\")\n        print(f\"Output Dir: {self.output_dir}\")\n        print(f\"Start Time: {self.results['run_info']['start_time']}\")\n        \n        # Run individual tests\n        tests = [\n            ('data_availability', self.validate_data_availability),\n            ('schema_audit', self.run_schema_audit),\n            ('leakage_detection', self.run_leakage_detection)\n        ]\n        \n        for test_name, test_func in tests:\n            try:\n                self.results['tests'][test_name] = test_func()\n            except Exception as e:\n                print(f\"❌ Failed to run {test_name}: {e}\")\n                self.results['tests'][test_name] = {\n                    'status': 'error',\n                    'passed': False,\n                    'error': str(e),\n                    'summary': f\"Test failed: {e}\"\n                }\n        \n        # Generate overall report\n        overall_report = self.generate_overall_report()\n        self.results['overall_report'] = overall_report\n        \n        # Add completion info\n        self.results['run_info']['end_time'] = datetime.now().isoformat()\n        self.results['run_info']['total_duration'] = overall_report['timing']['total_duration']\n        \n        # Save comprehensive results\n        with open(self.output_dir / 'preflight_results.json', 'w') as f:\n            serializable_results = self._make_serializable(self.results)\n            json.dump(serializable_results, f, indent=2, default=str)\n        \n        return self.results\n    \n    def _make_serializable(self, obj):\n        \"\"\"Convert ValidationResult objects to dicts for JSON serialization.\"\"\"\n        if isinstance(obj, ValidationResult):\n            return obj.to_dict()\n        elif isinstance(obj, dict):\n            return {k: self._make_serializable(v) for k, v in obj.items()}\n        elif isinstance(obj, list):\n            return [self._make_serializable(item) for item in obj]\n        elif isinstance(obj, set):\n            return list(obj)\n        else:\n            return obj\n    \n    def print_final_summary(self):\n        \"\"\"Print final preflight summary.\"\"\"\n        if not self.results:\n            print(\"No results available.\")\n            return\n        \n        print(\"\\n\" + \"=\" * 60)\n        print(\"PREFLIGHT VALIDATION SUMMARY\")\n        print(\"=\" * 60)\n        \n        overall_report = self.results.get('overall_report', {})\n        preflight_summary = overall_report.get('preflight_summary', {})\n        \n        # Overall status\n        if preflight_summary.get('overall_passed'):\n            print(\"🎉 Overall Status: ALL TESTS PASSED\")\n        else:\n            print(\"⚠️  Overall Status: SOME TESTS FAILED\")\n        \n        print(f\"Pass Rate: {preflight_summary.get('pass_rate', 0):.1%}\")\n        print(f\"Total Duration: {overall_report.get('timing', {}).get('total_duration', 0):.1f}s\")\n        \n        # Individual test results\n        print(\"\\nIndividual Test Results:\")\n        for test_summary in overall_report.get('test_summaries', []):\n            status = \"✅ PASS\" if test_summary['passed'] else \"❌ FAIL\"\n            duration = test_summary.get('duration', 0)\n            print(f\"  {status} {test_summary['test_name']:<20} ({duration:.1f}s) - {test_summary.get('summary', '')}\")\n        \n        # Recommendations\n        recommendations = overall_report.get('recommendations', [])\n        if recommendations:\n            print(\"\\nRecommendations:\")\n            for i, rec in enumerate(recommendations, 1):\n                print(f\"  {i}. {rec}\")\n        \n        print(f\"\\n📁 Detailed reports saved to: {self.output_dir}\")\n\ndef main():\n    \"\"\"Main execution function.\"\"\"\n    import argparse\n    \n    parser = argparse.ArgumentParser(description=\"Run complete data preflight validation\")\n    parser.add_argument(\"--data-path\", required=True, help=\"Path to data directory\")\n    parser.add_argument(\"--output-dir\", default=\"./experiments/reports\", \n                       help=\"Output directory for reports\")\n    \n    args = parser.parse_args()\n    \n    # Validate inputs\n    if not os.path.exists(args.data_path):\n        print(f\"❌ Data path does not exist: {args.data_path}\")\n        return 1\n    \n    # Run preflight validation\n    runner = PreflightRunner(args.data_path, args.output_dir)\n    results = runner.run_full_preflight()\n    runner.print_final_summary()\n    \n    # Return appropriate exit code\n    overall_passed = results.get('overall_report', {}).get('preflight_summary', {}).get('overall_passed', False)\n    \n    if overall_passed:\n        print(\"\\n🎉 Preflight validation completed successfully!\")\n        print(\"✅ Ready to proceed with SAFE training experiments.\")\n        return 0\n    else:\n        print(\"\\n⚠️  Preflight validation found issues.\")\n        print(\"❌ Please address the issues before proceeding with training.\")\n        return 1\n\nif __name__ == \"__main__\":\n    exit(main())"
        }
        
        optional_paths = {
            'AudioCaps audio (train)': Path(self.data_path) / "audiocaps" / "audio" / "train",
            'AudioCaps audio (val)': Path(self.data_path) / "audiocaps" / "audio" / "val",
            'VQA images (train)': Path(self.data_path) / "vqa" / "images" / "train2014",
            'VQA images (val)': Path(self.data_path) / "vqa" / "images" / "val2014"
        }
        
        availability_results = {
            'required': {},
            'optional': {},
            'all_required_present': True
        }
        
        # Check required paths
        for name, path in required_paths.items():
            exists = path.exists()
            availability_results['required'][name] = {
                'path': str(path),
                'exists': exists,
                'is_file': path.is_file() if exists else False
            }
            
            if exists:
                print(f"✅ {name}: Found")
            else:
                print(f"❌ {name}: Missing")
                availability_results['all_required_present'] = False
        
        # Check optional paths (directories with files)
        for name, path in optional_paths.items():
            exists = path.exists() and path.is_dir()
            file_count = len(list(path.iterdir())) if exists else 0
            
            availability_results['optional'][name] = {
                'path': str(path),
                'exists': exists,
                'file_count': file_count
            }
            
            if exists and file_count > 0:
                print(f"✅ {name}: Found ({file_count} files)")
            else:
                print(f"⚠️  {name}: {'Empty' if exists else 'Missing'}")
        
        test_result = {
            'status': 'completed',
            'passed': availability_results['all_required_present'],
            'duration': time.time() - start_time,
            'results': availability_results,
            'summary': 'All required data available' if availability_results['all_required_present'] else 'Missing required data'
        }
        
        return test_result
    
    def generate_overall_report(self) -> Dict[str, Any]:
        """Generate comprehensive preflight report."""
        print("\n📊 Generating Overall Report...")
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
            'preflight_summary': overall_summary,
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
                if test_name == 'data_availability':
                    recommendations.append(
                        "Download missing required datasets before proceeding with training"
                    )
                elif test_name == 'schema_audit':
                    recommendations.append(
                        "Review schema audit report and fix data quality issues"
                    )
                elif test_name == 'leakage_detection':
                    recommendations.append(
                        "Investigate and resolve data leakage issues between splits"
                    )
        
        if not recommendations:
            recommendations.append(
                "All preflight checks passed! Ready to proceed with training experiments."
            )
        
        return recommendations
    
    def run_av_alignment_check(self) -> Dict[str, Any]:
        """Run audio-visual alignment validation."""
        print("\n🔍 Running Audio-Visual Alignment Check...")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            validator = AVAlignmentValidator(self.data_path, self.output_dir)
            results = validator.run_full_validation()
            
            test_result = {
                'status': 'completed',
                'passed': results['validation_summary']['overall_passed'],
                'duration': time.time() - start_time,
                'results': results,
                'summary': f"Pass rate: {results['validation_summary']['pass_rate']:.1%}"
            }
            
            if results['validation_summary']['overall_passed']:
                print("✅ A/V alignment check PASSED")
            else:
                print("❌ A/V alignment check FAILED")
            
        except Exception as e:
            test_result = {
                'status': 'error',
                'passed': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'summary': f"Error: {e}"
            }
            print(f"❌ A/V alignment check ERROR: {e}")
        
        return test_result
    
    def run_curriculum_validation(self) -> Dict[str, Any]:
        """Run curriculum configuration validation."""
        print("\n🔍 Running Curriculum Configuration Validation...")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            validator = CurriculumValidator(self.data_path, output_dir=self.output_dir)
            results = validator.run_full_validation()
            
            test_result = {
                'status': 'completed',
                'passed': results['validation_summary']['overall_passed'],
                'duration': time.time() - start_time,
                'results': results,
                'summary': f"Pass rate: {results['validation_summary']['pass_rate']:.1%}"
            }
            
            if results['validation_summary']['overall_passed']:
                print("✅ Curriculum validation PASSED")
            else:
                print("❌ Curriculum validation FAILED")
            
        except Exception as e:
            test_result = {
                'status': 'error',
                'passed': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'summary': f"Error: {e}"
            }
            print(f"❌ Curriculum validation ERROR: {e}")
        
        return test_result
    
    def run_duplicate_detection(self) -> Dict[str, Any]:
        """Run cross-dataset duplicate detection."""
        print("\n🔍 Running Cross-Dataset Duplicate Detection...")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            detector = DuplicateDetector(self.data_path, self.output_dir)
            results = detector.run_full_detection()
            
            test_result = {
                'status': 'completed',
                'passed': results['validation_summary']['overall_passed'],
                'duration': time.time() - start_time,
                'results': results,
                'summary': f"Pass rate: {results['validation_summary']['pass_rate']:.1%}"
            }
            
            if results['validation_summary']['overall_passed']:
                print("✅ Duplicate detection PASSED")
            else:
                print("❌ Duplicate detection FAILED")
            
        except Exception as e:
            test_result = {
                'status': 'error',
                'passed': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'summary': f"Error: {e}"
            }
            print(f"❌ Duplicate detection ERROR: {e}")
        
        return test_result
    
    def run_audio_analysis(self) -> Dict[str, Any]:
        """Run comprehensive audio quality analysis."""
        print("\n🔍 Running Audio Quality Analysis...")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            analyzer = AudioQualityAnalyzer(self.data_path, self.output_dir)
            results = analyzer.run_full_analysis()
            
            test_result = {
                'status': 'completed',
                'passed': results['validation_summary']['overall_passed'],
                'duration': time.time() - start_time,
                'results': results,
                'summary': f"Pass rate: {results['validation_summary']['pass_rate']:.1%}"
            }
            
            if results['validation_summary']['overall_passed']:
                print("✅ Audio analysis PASSED")
            else:
                print("❌ Audio analysis FAILED")
            
        except Exception as e:
            test_result = {
                'status': 'error',
                'passed': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'summary': f"Error: {e}"
            }
            print(f"❌ Audio analysis ERROR: {e}")
        
        return test_result
    
    def run_full_preflight(self) -> Dict[str, Any]:
        """Run all preflight validation tests."""
        print("🚀 Starting Data & Pipeline Preflight Validation")
        print("=" * 60)
        print(f"Data Path: {self.data_path}")
        print(f"Output Dir: {self.output_dir}")
        print(f"Start Time: {self.results['run_info']['start_time']}")
        
        # Run individual tests
        tests = [
            ('data_availability', self.validate_data_availability),
            ('schema_audit', self.run_schema_audit),
            ('leakage_detection', self.run_leakage_detection),
            ('av_alignment', self.run_av_alignment_check),
            ('curriculum_validation', self.run_curriculum_validation),
            ('duplicate_detection', self.run_duplicate_detection),
            ('audio_analysis', self.run_audio_analysis)
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
        with open(self.output_dir / 'preflight_results.json', 'w') as f:
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Generate comprehensive HTML report
        print("\n📊 Generating comprehensive HTML report...")
        try:
            report_generator = ComprehensiveReportGenerator(self.data_path, self.output_dir)
            report_paths = report_generator.run_report_generation()
            
            self.results['report_paths'] = {
                'html_report': str(report_paths.get('html_report', '')),
                'json_summary': str(report_paths.get('json_summary', ''))
            }
        except Exception as e:
            print(f"⚠️  Could not generate HTML report: {e}")
            self.results['report_paths'] = {'error': str(e)}
        
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
        """Print final preflight summary."""
        if not self.results:
            print("No results available.")
            return
        
        print("\n" + "=" * 60)
        print("PREFLIGHT VALIDATION SUMMARY")
        print("=" * 60)
        
        overall_report = self.results.get('overall_report', {})
        preflight_summary = overall_report.get('preflight_summary', {})
        
        # Overall status
        if preflight_summary.get('overall_passed'):
            print("🎉 Overall Status: ALL TESTS PASSED")
        else:
            print("⚠️  Overall Status: SOME TESTS FAILED")
        
        print(f"Pass Rate: {preflight_summary.get('pass_rate', 0):.1%}")
        print(f"Total Duration: {overall_report.get('timing', {}).get('total_duration', 0):.1f}s")
        
        # Individual test results
        print("\nIndividual Test Results:")
        for test_summary in overall_report.get('test_summaries', []):
            status = "✅ PASS" if test_summary['passed'] else "❌ FAIL"
            duration = test_summary.get('duration', 0)
            print(f"  {status} {test_summary['test_name']:<20} ({duration:.1f}s) - {test_summary.get('summary', '')}")
        
        # Recommendations
        recommendations = overall_report.get('recommendations', [])
        if recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print(f"\n📁 Detailed reports saved to: {self.output_dir}")
        
        # Show HTML report link if available
        if 'report_paths' in self.results and 'html_report' in self.results['report_paths']:
            html_report_path = self.results['report_paths']['html_report']
            if html_report_path and not html_report_path.startswith('error'):
                print(f"\n🌐 Interactive HTML report available:")
                print(f"   file://{Path(html_report_path).absolute()}")
                print(f"   Open this link in your web browser for detailed interactive analysis.")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run complete data preflight validation")
    parser.add_argument("--data-path", required=True, help="Path to data directory")
    parser.add_argument("--output-dir", default="./experiments/reports", 
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_path):
        print(f"❌ Data path does not exist: {args.data_path}")
        return 1
    
    # Run preflight validation
    runner = PreflightRunner(args.data_path, args.output_dir)
    results = runner.run_full_preflight()
    runner.print_final_summary()
    
    # Return appropriate exit code
    overall_passed = results.get('overall_report', {}).get('preflight_summary', {}).get('overall_passed', False)
    
    if overall_passed:
        print("\n🎉 Preflight validation completed successfully!")
        print("✅ Ready to proceed with SAFE training experiments.")
        return 0
    else:
        print("\n⚠️  Preflight validation found issues.")
        print("❌ Please address the issues before proceeding with training.")
        return 1

if __name__ == "__main__":
    exit(main())