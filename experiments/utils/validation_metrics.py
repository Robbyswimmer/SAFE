"""
Go/No-go criteria checkers for validation experiments.
"""

from typing import Dict, List, Any, Tuple
import numpy as np

class ValidationResult:
    """Represents a validation test result with pass/fail status."""
    
    def __init__(self, test_name: str, passed: bool, message: str, 
                 value: Any = None, threshold: Any = None, metadata: Dict = None):
        self.test_name = test_name
        self.passed = passed
        self.message = message
        self.value = value
        self.threshold = threshold
        self.metadata = metadata or {}
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}: {self.test_name} - {self.message}"
    
    def to_dict(self):
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'metadata': self.metadata
        }

def check_file_load_rate(load_results: List[Dict], threshold: float = 0.95) -> ValidationResult:
    """Check if file loading success rate meets threshold."""
    total_files = len(load_results)
    if total_files == 0:
        return ValidationResult(
            "File Load Rate", False, 
            "No files to test", 0, threshold
        )
    
    successful_loads = sum(1 for result in load_results if result.get('loadable', False))
    load_rate = successful_loads / total_files
    
    passed = load_rate >= threshold
    message = f"Load rate: {load_rate:.1%} ({successful_loads}/{total_files})"
    
    return ValidationResult(
        "File Load Rate", passed, message, load_rate, threshold,
        metadata={'successful': successful_loads, 'total': total_files}
    )

def check_class_balance(class_counts: Dict[str, int], max_imbalance: float = 5.0) -> ValidationResult:
    """Check class balance within acceptable limits."""
    if not class_counts:
        return ValidationResult(
            "Class Balance", False, "No classes found", None, max_imbalance
        )
    
    counts = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    
    if min_count == 0:
        return ValidationResult(
            "Class Balance", False, 
            f"Empty classes found: {[k for k, v in class_counts.items() if v == 0]}",
            float('inf'), max_imbalance
        )
    
    imbalance_ratio = max_count / min_count
    passed = imbalance_ratio <= max_imbalance
    
    message = f"Imbalance ratio: {imbalance_ratio:.2f} (max={max_count}, min={min_count})"
    
    return ValidationResult(
        "Class Balance", passed, message, imbalance_ratio, max_imbalance,
        metadata={'class_counts': class_counts}
    )

def check_duplicate_rate(duplicate_results: Dict, max_duplicate_rate: float = 0.02) -> ValidationResult:
    """Check duplicate detection results."""
    total_samples = duplicate_results.get('total_samples', 0)
    duplicates_found = duplicate_results.get('duplicates_found', 0)
    
    if total_samples == 0:
        return ValidationResult(
            "Duplicate Rate", False, "No samples to check", None, max_duplicate_rate
        )
    
    duplicate_rate = duplicates_found / total_samples
    passed = duplicate_rate <= max_duplicate_rate
    
    message = f"Duplicate rate: {duplicate_rate:.1%} ({duplicates_found}/{total_samples})"
    
    return ValidationResult(
        "Duplicate Rate", passed, message, duplicate_rate, max_duplicate_rate,
        metadata=duplicate_results
    )

def check_id_leakage(leakage_results: Dict) -> ValidationResult:
    """Check for ID leakage between splits."""
    train_ids = set(leakage_results.get('train_ids', []))
    val_ids = set(leakage_results.get('val_ids', []))
    test_ids = set(leakage_results.get('test_ids', []))
    
    # Check for overlaps
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    
    total_overlaps = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    passed = total_overlaps == 0
    
    if passed:
        message = f"No ID leakage detected across {len(train_ids + val_ids + test_ids)} total IDs"
    else:
        overlaps = []
        if train_val_overlap:
            overlaps.append(f"train-val: {len(train_val_overlap)}")
        if train_test_overlap:
            overlaps.append(f"train-test: {len(train_test_overlap)}")
        if val_test_overlap:
            overlaps.append(f"val-test: {len(val_test_overlap)}")
        message = f"ID leakage found: {', '.join(overlaps)}"
    
    return ValidationResult(
        "ID Leakage", passed, message, total_overlaps, 0,
        metadata={
            'overlaps': {
                'train_val': list(train_val_overlap),
                'train_test': list(train_test_overlap),
                'val_test': list(val_test_overlap)
            }
        }
    )

def check_av_alignment(alignment_scores: List[float], min_alignment: float = 0.9) -> ValidationResult:
    """Check audio-visual alignment scores."""
    if not alignment_scores:
        return ValidationResult(
            "A/V Alignment", False, "No alignment scores provided", None, min_alignment
        )
    
    mean_alignment = np.mean(alignment_scores)
    good_alignments = sum(1 for score in alignment_scores if score >= min_alignment)
    alignment_rate = good_alignments / len(alignment_scores)
    
    passed = alignment_rate >= 0.9  # 90% of samples should have good alignment
    
    message = f"Alignment rate: {alignment_rate:.1%} (mean score: {mean_alignment:.3f})"
    
    return ValidationResult(
        "A/V Alignment", passed, message, alignment_rate, 0.9,
        metadata={
            'mean_score': mean_alignment,
            'scores': alignment_scores,
            'good_alignments': good_alignments
        }
    )

def check_curriculum_sampling(sampling_stats: Dict, expected_ratios: Dict, 
                            tolerance: float = 0.02) -> ValidationResult:
    """Check if curriculum sampling matches expected ratios."""
    all_passed = True
    details = {}
    
    for stage, expected_ratio in expected_ratios.items():
        actual_ratio = sampling_stats.get(stage, 0)
        diff = abs(actual_ratio - expected_ratio)
        stage_passed = diff <= tolerance
        all_passed &= stage_passed
        
        details[stage] = {
            'expected': expected_ratio,
            'actual': actual_ratio,
            'difference': diff,
            'passed': stage_passed
        }
    
    if all_passed:
        message = "All sampling ratios within tolerance"
    else:
        failed_stages = [k for k, v in details.items() if not v['passed']]
        message = f"Failed stages: {failed_stages}"
    
    return ValidationResult(
        "Curriculum Sampling", all_passed, message, None, tolerance,
        metadata={'stage_details': details}
    )

def aggregate_validation_results(results: List[ValidationResult]) -> Dict[str, Any]:
    """Aggregate multiple validation results into a summary."""
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    
    summary = {
        'overall_passed': passed_tests == total_tests,
        'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': total_tests - passed_tests,
        'results': [r.to_dict() for r in results]
    }
    
    return summary