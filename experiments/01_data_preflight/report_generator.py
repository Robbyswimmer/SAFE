"""
Comprehensive Report Generator

Generates interactive HTML reports with visualizations:
- Data quality dashboard
- Validation results summary  
- Interactive charts and plots
- Detailed findings and recommendations
- Export capabilities (JSON, PDF)
"""

import os
import sys
import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

class ComprehensiveReportGenerator:
    """Generates comprehensive HTML reports from validation results."""
    
    def __init__(self, data_path: str, output_dir: str = "./experiments/reports"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing results
        self.results = self._load_all_results()
    
    def _load_all_results(self) -> Dict[str, Any]:
        """Load all available validation results."""
        results = {
            'preflight_overall': {},
            'schema_audit': {},
            'leakage_detection': {},
            'av_alignment': {},
            'curriculum_validation': {},
            'duplicate_detection': {},
            'audio_analysis': {}
        }
        
        result_files = {
            'preflight_overall': 'preflight_results.json',
            'schema_audit': 'schema_audit_results.json',
            'leakage_detection': 'leakage_detection_results.json',
            'av_alignment': 'av_alignment_results.json',
            'curriculum_validation': 'curriculum_validation_results.json',
            'duplicate_detection': 'duplicate_detection_results.json',
            'audio_analysis': 'audio_analysis_results.json'
        }
        
        for key, filename in result_files.items():
            filepath = self.output_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        results[key] = json.load(f)
                        print(f"✅ Loaded {key} results")
                except Exception as e:
                    print(f"⚠️  Could not load {key}: {e}")
            else:
                print(f"📁 {key} results not found at {filepath}")
        
        return results
    
    def _encode_image_to_base64(self, image_path: Path) -> Optional[str]:
        """Convert image to base64 for embedding in HTML."""
        try:
            if image_path.exists():
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                    return base64.b64encode(image_data).decode('utf-8')
        except Exception:
            pass
        return None
    
    def _create_summary_chart(self) -> str:
        """Create overall validation summary chart."""
        # Collect all validation results
        all_validations = []
        test_categories = []
        
        for category, data in self.results.items():
            if not data:
                continue
                
            # Extract validations from different result structures
            validations = []
            if 'validation_summary' in data and 'validations' in data['validation_summary']:
                validations = data['validation_summary']['validations']
            elif 'summary' in data and 'validations' in data['summary']:
                validations = data['summary']['validations']
            elif 'overall_report' in data and 'preflight_summary' in data['overall_report']:
                # For preflight results, extract from test summaries
                test_summaries = data['overall_report'].get('test_summaries', [])
                for test in test_summaries:
                    validations.append({
                        'test_name': test.get('test_name', 'Unknown'),
                        'passed': test.get('passed', False)
                    })
            
            for validation in validations:
                if isinstance(validation, dict):
                    all_validations.append(validation)
                    test_categories.append(category.replace('_', ' ').title())
        
        if not all_validations:
            return "<p>No validation data available for summary chart.</p>"
        
        # Create summary visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart of overall pass/fail
        passed = sum(1 for v in all_validations if v.get('passed', False))
        failed = len(all_validations) - passed
        
        if passed + failed > 0:
            ax1.pie([passed, failed], labels=['Passed', 'Failed'], 
                   colors=['#28a745', '#dc3545'], autopct='%1.1f%%', startangle=90)
            ax1.set_title(f'Overall Validation Results\\n({passed + failed} total tests)')
        
        # Bar chart by category
        if test_categories:
            category_results = {}
            for cat, val in zip(test_categories, all_validations):
                if cat not in category_results:
                    category_results[cat] = {'passed': 0, 'failed': 0}
                if val.get('passed', False):
                    category_results[cat]['passed'] += 1
                else:
                    category_results[cat]['failed'] += 1
            
            categories = list(category_results.keys())
            passed_counts = [category_results[cat]['passed'] for cat in categories]
            failed_counts = [category_results[cat]['failed'] for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax2.bar(x - width/2, passed_counts, width, label='Passed', color='#28a745', alpha=0.8)
            ax2.bar(x + width/2, failed_counts, width, label='Failed', color='#dc3545', alpha=0.8)
            
            ax2.set_xlabel('Test Category')
            ax2.set_ylabel('Number of Tests')
            ax2.set_title('Results by Category')
            ax2.set_xticks(x)
            ax2.set_xticklabels(categories, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save and encode
        chart_path = self.output_dir / 'summary_chart.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        encoded_image = self._encode_image_to_base64(chart_path)
        if encoded_image:
            return f'<img src="data:image/png;base64,{encoded_image}" alt="Summary Chart" style="max-width: 100%; height: auto;">'
        else:
            return "<p>Could not generate summary chart.</p>"
    
    def _format_validation_results(self, results: Dict, title: str) -> str:
        """Format validation results into HTML."""
        if not results:
            return f"<h3>{title}</h3><p>No results available.</p>"
        
        html = f"<h3>{title}</h3>"
        
        # Extract validation summary
        summary = {}
        validations = []
        
        if 'validation_summary' in results:
            summary = results['validation_summary']
            validations = summary.get('validations', [])
        elif 'summary' in results:
            summary = results['summary']
            validations = summary.get('validations', [])
        
        # Overall status
        if summary:
            overall_passed = summary.get('overall_passed', False)
            pass_rate = summary.get('pass_rate', 0)
            
            status_color = '#28a745' if overall_passed else '#dc3545'
            status_text = 'PASSED' if overall_passed else 'FAILED'
            
            html += f"""
            <div class="alert alert-{'success' if overall_passed else 'danger'}">
                <strong>Overall Status:</strong> <span style="color: {status_color};">{status_text}</span><br>
                <strong>Pass Rate:</strong> {pass_rate:.1%}
            </div>
            """
        
        # Individual validations
        if validations:
            html += "<h4>Individual Tests:</h4><ul class='list-group'>"
            for validation in validations:
                if isinstance(validation, dict):
                    test_name = validation.get('test_name', 'Unknown Test')
                    passed = validation.get('passed', False)
                    message = validation.get('message', 'No details available')
                    
                    status_icon = '✅' if passed else '❌'
                    list_class = 'list-group-item-success' if passed else 'list-group-item-danger'
                    
                    html += f"""
                    <li class="list-group-item {list_class}">
                        {status_icon} <strong>{test_name}:</strong> {message}
                    </li>
                    """
            html += "</ul>"
        
        return html
    
    def _create_dataset_overview(self) -> str:
        """Create dataset overview section."""
        html = "<h2>Dataset Overview</h2>"
        
        # Extract dataset information from schema audit
        schema_results = self.results.get('schema_audit', {})
        
        if schema_results:
            html += "<h3>Dataset Statistics</h3><div class='row'>"
            
            # AudioCaps
            if 'audiocaps' in schema_results:
                audiocaps = schema_results['audiocaps']
                html += "<div class='col-md-6'><h4>AudioCaps</h4>"
                
                for split, split_data in audiocaps.get('splits', {}).items():
                    if isinstance(split_data, dict) and 'error' not in split_data:
                        metadata_count = split_data.get('metadata_samples', 0)
                        audio_count = split_data.get('audio_files_found', 0)
                        coverage = audio_count / metadata_count if metadata_count > 0 else 0
                        
                        html += f"""
                        <p><strong>{split.title()}:</strong><br>
                        Metadata: {metadata_count} samples<br>
                        Audio files: {audio_count} files<br>
                        Coverage: {coverage:.1%}</p>
                        """
                
                html += "</div>"
            
            # VQA
            if 'vqa' in schema_results:
                vqa = schema_results['vqa']
                html += "<div class='col-md-6'><h4>VQA v2</h4>"
                
                for split, split_data in vqa.get('splits', {}).items():
                    if isinstance(split_data, dict) and 'error' not in split_data:
                        questions = split_data.get('questions_count', 0)
                        annotations = split_data.get('annotations_count', 0)
                        images = split_data.get('image_files_found', 0)
                        
                        html += f"""
                        <p><strong>{split.title()}:</strong><br>
                        Questions: {questions}<br>
                        Annotations: {annotations}<br>
                        Images: {images}</p>
                        """
                
                html += "</div>"
            
            html += "</div>"
        
        return html
    
    def _create_recommendations_section(self) -> str:
        """Create recommendations based on all results."""
        html = "<h2>Recommendations</h2>"
        
        recommendations = []
        priority_issues = []
        
        # Analyze all results for recommendations
        for category, results in self.results.items():
            if not results:
                continue
            
            # Extract failed validations
            validations = []
            if 'validation_summary' in results:
                validations = results['validation_summary'].get('validations', [])
            elif 'summary' in results:
                validations = results['summary'].get('validations', [])
            
            failed_validations = [v for v in validations if isinstance(v, dict) and not v.get('passed', False)]
            
            if failed_validations:
                category_name = category.replace('_', ' ').title()
                
                for validation in failed_validations:
                    test_name = validation.get('test_name', 'Unknown')
                    message = validation.get('message', '')
                    
                    # Categorize recommendations
                    if 'leakage' in test_name.lower() or 'duplicate' in test_name.lower():
                        priority_issues.append(f"{category_name}: {test_name} - {message}")
                    elif 'load' in test_name.lower() or 'availability' in test_name.lower():
                        priority_issues.append(f"{category_name}: {test_name} - {message}")
                    else:
                        recommendations.append(f"{category_name}: {test_name} - {message}")
        
        # Priority issues first
        if priority_issues:
            html += "<h3>🚨 Priority Issues</h3><div class='alert alert-danger'>"
            html += "<p>These issues should be resolved before proceeding with training:</p><ul>"
            for issue in priority_issues:
                html += f"<li>{issue}</li>"
            html += "</ul></div>"
        
        # General recommendations
        if recommendations:
            html += "<h3>💡 Recommendations</h3><div class='alert alert-warning'>"
            html += "<p>Consider addressing these issues to improve data quality:</p><ul>"
            for rec in recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul></div>"
        
        if not priority_issues and not recommendations:
            html += "<div class='alert alert-success'>✅ No issues found! Data quality looks good for training.</div>"
        
        # Add general best practices
        html += """
        <h3>📋 General Best Practices</h3>
        <ul>
            <li><strong>Data Backup:</strong> Ensure all datasets are backed up before training</li>
            <li><strong>Version Control:</strong> Track dataset versions and preprocessing steps</li>
            <li><strong>Monitoring:</strong> Monitor training metrics for signs of data quality issues</li>
            <li><strong>Validation:</strong> Regularly validate model outputs on held-out test sets</li>
        </ul>
        """
        
        return html
    
    def generate_html_report(self) -> str:
        """Generate comprehensive HTML report."""
        print("📊 Generating comprehensive HTML report...")
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create summary chart
        summary_chart = self._create_summary_chart()
        
        # Format individual sections
        sections = {
            'Schema Audit': self._format_validation_results(self.results.get('schema_audit', {}), 'Schema & Distribution Audit'),
            'Leakage Detection': self._format_validation_results(self.results.get('leakage_detection', {}), 'ID Leakage Detection'),
            'A/V Alignment': self._format_validation_results(self.results.get('av_alignment', {}), 'Audio-Visual Alignment'),
            'Curriculum Validation': self._format_validation_results(self.results.get('curriculum_validation', {}), 'Curriculum Configuration'),
            'Duplicate Detection': self._format_validation_results(self.results.get('duplicate_detection', {}), 'Cross-Dataset Duplicates'),
            'Audio Quality': self._format_validation_results(self.results.get('audio_analysis', {}), 'Audio Quality Analysis')
        }
        
        # Create dataset overview
        dataset_overview = self._create_dataset_overview()
        
        # Create recommendations
        recommendations = self._create_recommendations_section()
        
        # Include existing plots
        plot_images = {}
        plot_files = [
            ('audiocaps_distributions.png', 'AudioCaps Distributions'),
            ('vqa_distributions.png', 'VQA Distributions'),
            ('audio_quality_analysis.png', 'Audio Quality Analysis')
        ]
        
        for filename, title in plot_files:
            plot_path = self.output_dir / filename
            encoded = self._encode_image_to_base64(plot_path)
            if encoded:
                plot_images[title] = f'<img src="data:image/png;base64,{encoded}" alt="{title}" style="max-width: 100%; height: auto;">'
        
        # HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SAFE Data Preflight Validation Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ font-family: 'Arial', sans-serif; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; }}
                .section {{ margin: 2rem 0; }}
                .alert {{ border-radius: 8px; }}
                .card {{ box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); border: none; margin-bottom: 1rem; }}
                .nav-pills .nav-link.active {{ background-color: #667eea; }}
                .plot-container {{ text-align: center; margin: 1rem 0; }}
                .timestamp {{ color: #6c757d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1 class="display-4">🛡️ SAFE Data Preflight Validation Report</h1>
                    <p class="lead">Comprehensive data quality assessment for multimodal training</p>
                    <p class="timestamp">Generated: {timestamp}</p>
                </div>
            </div>
            
            <div class="container mt-4">
                <!-- Navigation -->
                <ul class="nav nav-pills justify-content-center mb-4" id="reportTabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="summary-tab" data-bs-toggle="pill" data-bs-target="#summary" type="button">Summary</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="datasets-tab" data-bs-toggle="pill" data-bs-target="#datasets" type="button">Datasets</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="validations-tab" data-bs-toggle="pill" data-bs-target="#validations" type="button">Validations</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="plots-tab" data-bs-toggle="pill" data-bs-target="#plots" type="button">Visualizations</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="recommendations-tab" data-bs-toggle="pill" data-bs-target="#recommendations" type="button">Recommendations</button>
                    </li>
                </ul>
                
                <div class="tab-content">
                    <!-- Summary Tab -->
                    <div class="tab-pane fade show active" id="summary" role="tabpanel">
                        <div class="card">
                            <div class="card-body">
                                <h2>Executive Summary</h2>
                                <p>This report presents the results of comprehensive data preflight validation for the SAFE (Safe Audio-Visual Enhancement) framework training pipeline.</p>
                                
                                <h3>Overall Results</h3>
                                <div class="plot-container">
                                    {summary_chart}
                                </div>
                                
                                <h3>Data Path</h3>
                                <p><code>{self.data_path}</code></p>
                                
                                <h3>Tests Performed</h3>
                                <ul>
                                    <li><strong>Schema & Distribution Audit:</strong> Dataset structure and quality validation</li>
                                    <li><strong>ID Leakage Detection:</strong> Cross-split data leakage identification</li>
                                    <li><strong>Audio-Visual Alignment:</strong> Multimodal content alignment verification</li>
                                    <li><strong>Curriculum Validation:</strong> Training configuration validation</li>
                                    <li><strong>Duplicate Detection:</strong> Cross-dataset duplicate identification</li>
                                    <li><strong>Audio Quality Analysis:</strong> Comprehensive audio quality assessment</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Datasets Tab -->
                    <div class="tab-pane fade" id="datasets" role="tabpanel">
                        <div class="card">
                            <div class="card-body">
                                {dataset_overview}
                            </div>
                        </div>
                    </div>
                    
                    <!-- Validations Tab -->
                    <div class="tab-pane fade" id="validations" role="tabpanel">
        """
        
        # Add validation sections
        for section_title, section_content in sections.items():
            html_content += f"""
                        <div class="card">
                            <div class="card-body">
                                {section_content}
                            </div>
                        </div>
            """
        
        html_content += """
                    </div>
                    
                    <!-- Plots Tab -->
                    <div class="tab-pane fade" id="plots" role="tabpanel">
        """
        
        # Add plot sections
        for plot_title, plot_html in plot_images.items():
            html_content += f"""
                        <div class="card">
                            <div class="card-body">
                                <h3>{plot_title}</h3>
                                <div class="plot-container">
                                    {plot_html}
                                </div>
                            </div>
                        </div>
            """
        
        if not plot_images:
            html_content += """
                        <div class="card">
                            <div class="card-body">
                                <p>No visualization plots available. Run individual validation scripts to generate plots.</p>
                            </div>
                        </div>
            """
        
        html_content += f"""
                    </div>
                    
                    <!-- Recommendations Tab -->
                    <div class="tab-pane fade" id="recommendations" role="tabpanel">
                        <div class="card">
                            <div class="card-body">
                                {recommendations}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <footer class="bg-light text-center py-3 mt-5">
                <p class="text-muted">SAFE Data Preflight Validation Report | Generated by Claude Code</p>
            </footer>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """
        
        return html_content
    
    def save_html_report(self) -> Path:
        """Save the HTML report to file."""
        html_content = self.generate_html_report()
        
        report_path = self.output_dir / 'preflight_validation_report.html'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 HTML report saved to: {report_path}")
        return report_path
    
    def export_json_summary(self) -> Path:
        """Export a JSON summary of all results."""
        summary = {
            'generation_time': datetime.now().isoformat(),
            'data_path': self.data_path,
            'results_summary': {}
        }
        
        for category, results in self.results.items():
            if not results:
                continue
            
            category_summary = {
                'available': True,
                'overall_passed': False,
                'pass_rate': 0.0,
                'total_tests': 0,
                'passed_tests': 0
            }
            
            # Extract summary information
            if 'validation_summary' in results:
                summary_data = results['validation_summary']
                category_summary.update({
                    'overall_passed': summary_data.get('overall_passed', False),
                    'pass_rate': summary_data.get('pass_rate', 0.0),
                    'total_tests': summary_data.get('total_tests', 0),
                    'passed_tests': summary_data.get('passed_tests', 0)
                })
            elif 'summary' in results:
                summary_data = results['summary']
                category_summary.update({
                    'overall_passed': summary_data.get('overall_passed', False),
                    'pass_rate': summary_data.get('pass_rate', 0.0)
                })
            
            summary['results_summary'][category] = category_summary
        
        # Overall summary
        all_categories = list(summary['results_summary'].values())
        if all_categories:
            total_passed = sum(1 for cat in all_categories if cat['overall_passed'])
            summary['overall_summary'] = {
                'categories_passed': total_passed,
                'total_categories': len(all_categories),
                'overall_pass_rate': total_passed / len(all_categories)
            }
        
        json_path = self.output_dir / 'preflight_summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 JSON summary saved to: {json_path}")
        return json_path
    
    def run_report_generation(self) -> Dict[str, Path]:
        """Run complete report generation."""
        print("🔍 Starting Comprehensive Report Generation...")
        print("=" * 50)
        
        # Generate reports
        html_path = self.save_html_report()
        json_path = self.export_json_summary()
        
        return {
            'html_report': html_path,
            'json_summary': json_path
        }

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive preflight validation report")
    parser.add_argument("--data-path", required=True, help="Path to data directory")
    parser.add_argument("--output-dir", default="./experiments/reports", 
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Generate reports
    generator = ComprehensiveReportGenerator(args.data_path, args.output_dir)
    report_paths = generator.run_report_generation()
    
    print("\n" + "=" * 50)
    print("REPORT GENERATION SUMMARY")
    print("=" * 50)
    
    for report_type, path in report_paths.items():
        print(f"✅ {report_type.replace('_', ' ').title()}: {path}")
    
    print(f"\n🌐 Open the HTML report in your browser:")
    print(f"   file://{report_paths['html_report'].absolute()}")
    
    return 0

if __name__ == "__main__":
    exit(main())