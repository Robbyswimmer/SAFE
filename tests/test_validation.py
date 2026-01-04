"""
Tests for the SAFE dataset validation module.
"""
import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np

from safe.data.validation import (
    ValidationLevel,
    ValidationStatus,
    ValidationResult,
    DatasetStats,
    DatasetValidator,
    validate_safe_dataset,
)


class TestValidationLevel:
    """Tests for ValidationLevel enum."""

    def test_validation_levels_exist(self):
        """Test all validation levels are defined."""
        assert ValidationLevel.BASIC.value == "basic"
        assert ValidationLevel.STANDARD.value == "standard"
        assert ValidationLevel.COMPREHENSIVE.value == "comprehensive"


class TestValidationStatus:
    """Tests for ValidationStatus enum."""

    def test_validation_statuses_exist(self):
        """Test all validation statuses are defined."""
        assert ValidationStatus.PASSED.value == "passed"
        assert ValidationStatus.WARNING.value == "warning"
        assert ValidationStatus.FAILED.value == "failed"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_result_creation(self):
        """Test creating a ValidationResult."""
        result = ValidationResult(
            name="test_check",
            status=ValidationStatus.PASSED,
            message="Test passed successfully",
        )
        assert result.name == "test_check"
        assert result.status == ValidationStatus.PASSED
        assert result.message == "Test passed successfully"
        assert result.details == {}
        assert result.suggestions == []

    def test_result_with_details_and_suggestions(self):
        """Test ValidationResult with details and suggestions."""
        result = ValidationResult(
            name="audio_check",
            status=ValidationStatus.WARNING,
            message="Audio quality issues detected",
            details={"sample_rate": 16000, "channels": 1},
            suggestions=["Normalize audio", "Check sample rate"],
        )
        assert result.details["sample_rate"] == 16000
        assert len(result.suggestions) == 2


class TestDatasetStats:
    """Tests for DatasetStats dataclass."""

    def test_stats_initialization(self):
        """Test DatasetStats initializes with defaults."""
        stats = DatasetStats()
        assert stats.total_samples == 0
        assert stats.audio_samples == 0
        assert stats.visual_samples == 0
        assert stats.multimodal_samples == 0
        assert stats.invalid_samples == []
        assert stats.duplicate_samples == []

    def test_stats_modification(self):
        """Test DatasetStats can be modified."""
        stats = DatasetStats()
        stats.total_samples = 100
        stats.audio_samples = 50
        stats.visual_samples = 70
        stats.multimodal_samples = 30
        stats.invalid_samples.append(5)

        assert stats.total_samples == 100
        assert len(stats.invalid_samples) == 1


class MockDataset:
    """Mock dataset for testing validation."""

    def __init__(
        self,
        samples: List[Dict[str, Any]] = None,
        size: int = 10,
        include_audio: bool = True,
        include_images: bool = True,
    ):
        if samples is not None:
            self._samples = samples
        else:
            self._samples = []
            for i in range(size):
                sample = {
                    "question": f"Test question {i}?",
                    "answer": f"Test answer {i}",
                }
                if include_audio:
                    sample["audio"] = torch.randn(16000)
                else:
                    sample["audio"] = None
                if include_images:
                    sample["images"] = torch.randn(3, 224, 224)
                else:
                    sample["images"] = None
                self._samples.append(sample)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


class TestDatasetValidator:
    """Tests for DatasetValidator class."""

    def test_initialization(self):
        """Test DatasetValidator initializes correctly."""
        validator = DatasetValidator(validation_level=ValidationLevel.STANDARD)
        assert validator.validation_level == ValidationLevel.STANDARD
        assert validator.results == []

    def test_validate_empty_dataset(self):
        """Test validation of empty dataset."""
        validator = DatasetValidator()
        dataset = MockDataset(samples=[])

        results = validator.validate_dataset(dataset)

        # Should report failed for empty dataset
        failed_results = [r for r in results if r.status == ValidationStatus.FAILED]
        assert len(failed_results) > 0

    def test_validate_small_dataset(self):
        """Test validation of small dataset triggers warning."""
        validator = DatasetValidator()
        dataset = MockDataset(size=5)

        results = validator.validate_dataset(dataset)

        # Should have warning about small dataset
        warning_results = [r for r in results if r.status == ValidationStatus.WARNING]
        assert any("small" in r.message.lower() for r in warning_results)

    def test_validate_normal_dataset(self):
        """Test validation of normal dataset passes."""
        validator = DatasetValidator(validation_level=ValidationLevel.BASIC)
        dataset = MockDataset(size=20)

        results = validator.validate_dataset(dataset)

        # Should have passing results
        passed_results = [r for r in results if r.status == ValidationStatus.PASSED]
        assert len(passed_results) > 0

    def test_validate_audio_only_dataset(self):
        """Test validation of audio-only dataset."""
        validator = DatasetValidator()
        dataset = MockDataset(size=10, include_audio=True, include_images=False)

        results = validator.validate_dataset(dataset)

        # Check modality distribution is recorded
        assert validator.stats.audio_samples > 0
        assert validator.stats.visual_samples == 0

    def test_validate_visual_only_dataset(self):
        """Test validation of visual-only dataset."""
        validator = DatasetValidator()
        dataset = MockDataset(size=10, include_audio=False, include_images=True)

        results = validator.validate_dataset(dataset)

        assert validator.stats.audio_samples == 0
        assert validator.stats.visual_samples > 0

    def test_validate_multimodal_dataset(self):
        """Test validation of multimodal dataset."""
        validator = DatasetValidator()
        dataset = MockDataset(size=10, include_audio=True, include_images=True)

        results = validator.validate_dataset(dataset)

        assert validator.stats.multimodal_samples > 0

    def test_validate_with_sample_size(self):
        """Test validation with limited sample size."""
        validator = DatasetValidator()
        dataset = MockDataset(size=100)

        results = validator.validate_dataset(dataset, sample_size=20)

        # Should validate only 20 samples but stats reflect that
        assert validator.stats.total_samples == 100

    def test_validate_audio_with_nan(self):
        """Test validation detects NaN in audio."""
        validator = DatasetValidator()
        samples = [
            {
                "question": "Test?",
                "answer": "Answer",
                "audio": torch.tensor([float("nan"), 1.0, 2.0]),
                "images": None,
            }
        ]
        dataset = MockDataset(samples=samples)

        results = validator.validate_dataset(dataset)

        # Should report NaN issue
        failed_results = [r for r in results if r.status == ValidationStatus.FAILED]
        assert any("nan" in r.message.lower() for r in failed_results)

    def test_validate_audio_with_inf(self):
        """Test validation detects Inf in audio."""
        validator = DatasetValidator()
        samples = [
            {
                "question": "Test?",
                "answer": "Answer",
                "audio": torch.tensor([float("inf"), 1.0, 2.0]),
                "images": None,
            }
        ]
        dataset = MockDataset(samples=samples)

        results = validator.validate_dataset(dataset)

        # Should report Inf issue
        failed_results = [r for r in results if r.status == ValidationStatus.FAILED]
        assert any("inf" in r.message.lower() for r in failed_results)

    def test_validate_large_audio_values(self):
        """Test validation warns about large audio values."""
        validator = DatasetValidator()
        samples = [
            {
                "question": "Test?",
                "answer": "Answer",
                "audio": torch.tensor([100.0, 200.0, 300.0]),  # Very large
                "images": None,
            }
        ]
        dataset = MockDataset(samples=samples)

        results = validator.validate_dataset(dataset)

        # Should warn about large values
        warning_results = [r for r in results if r.status == ValidationStatus.WARNING]
        assert any("large" in r.message.lower() for r in warning_results)

    def test_validate_empty_question(self):
        """Test validation detects empty questions."""
        validator = DatasetValidator()
        samples = [
            {
                "question": "",
                "answer": "Answer",
                "audio": None,
                "images": None,
            }
        ]
        dataset = MockDataset(samples=samples)

        results = validator.validate_dataset(dataset)

        # Should report empty question
        failed_results = [r for r in results if r.status == ValidationStatus.FAILED]
        assert any("empty" in r.message.lower() for r in failed_results)

    def test_validate_empty_answer(self):
        """Test validation detects empty answers."""
        validator = DatasetValidator()
        samples = [
            {
                "question": "Test?",
                "answer": "",
                "audio": None,
                "images": None,
            }
        ]
        dataset = MockDataset(samples=samples)

        results = validator.validate_dataset(dataset)

        failed_results = [r for r in results if r.status == ValidationStatus.FAILED]
        assert any("empty" in r.message.lower() for r in failed_results)

    def test_validate_long_question_warning(self):
        """Test validation warns about very long questions."""
        validator = DatasetValidator()
        samples = [
            {
                "question": "Q" * 600,  # Very long
                "answer": "A",
                "audio": None,
                "images": None,
            }
        ]
        dataset = MockDataset(samples=samples)

        results = validator.validate_dataset(dataset)

        warning_results = [r for r in results if r.status == ValidationStatus.WARNING]
        assert any("long" in r.message.lower() for r in warning_results)

    def test_stats_updated_during_validation(self):
        """Test that stats are updated during validation."""
        validator = DatasetValidator()
        dataset = MockDataset(size=10, include_audio=True, include_images=True)

        validator.validate_dataset(dataset)

        assert validator.stats.total_samples == 10
        assert validator.stats.audio_samples == 10
        assert validator.stats.visual_samples == 10
        assert validator.stats.multimodal_samples == 10

    def test_save_report(self, tmp_path):
        """Test saving validation report to file."""
        validator = DatasetValidator()
        dataset = MockDataset(size=10)

        validator.validate_dataset(dataset)
        report_path = tmp_path / "report.json"
        validator.save_report(report_path)

        assert report_path.exists()

        with open(report_path) as f:
            report = json.load(f)

        assert "validation_level" in report
        assert "statistics" in report
        assert "results" in report

    def test_comprehensive_validation_includes_performance(self):
        """Test comprehensive validation includes performance checks."""
        validator = DatasetValidator(validation_level=ValidationLevel.COMPREHENSIVE)
        dataset = MockDataset(size=20)

        results = validator.validate_dataset(dataset)

        # Should include loading performance check
        result_names = [r.name for r in results]
        assert any("performance" in name.lower() for name in result_names)

    def test_standard_validation_includes_curriculum(self):
        """Test standard validation includes curriculum compatibility."""
        validator = DatasetValidator(validation_level=ValidationLevel.STANDARD)
        dataset = MockDataset(size=20)

        results = validator.validate_dataset(dataset)

        result_names = [r.name for r in results]
        assert any("curriculum" in name.lower() for name in result_names)

    def test_multimodal_alignment_check(self):
        """Test multimodal alignment validation."""
        validator = DatasetValidator()
        # Question mentions audio but no audio present
        samples = [
            {
                "question": "What sound do you hear?",
                "answer": "Bird",
                "audio": None,  # No audio!
                "images": None,
            }
        ]
        dataset = MockDataset(samples=samples)

        results = validator.validate_dataset(dataset)

        # Should detect alignment issue
        result_names = [r.name for r in results]
        assert any("alignment" in name.lower() for name in result_names)


class TestValidateSafeDataset:
    """Tests for validate_safe_dataset convenience function."""

    def test_basic_usage(self):
        """Test basic usage of validate_safe_dataset."""
        dataset = MockDataset(size=10)

        results = validate_safe_dataset(dataset)

        assert isinstance(results, list)
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_with_output_dir(self, tmp_path):
        """Test validate_safe_dataset saves report when output_dir provided."""
        dataset = MockDataset(size=10)

        results = validate_safe_dataset(dataset, output_dir=tmp_path)

        report_path = tmp_path / "validation_report.json"
        assert report_path.exists()

    def test_with_custom_validation_level(self):
        """Test validate_safe_dataset with custom validation level."""
        dataset = MockDataset(size=10)

        results = validate_safe_dataset(
            dataset, validation_level=ValidationLevel.COMPREHENSIVE
        )

        # Comprehensive should include more checks
        assert len(results) > 0

    def test_with_sample_size(self):
        """Test validate_safe_dataset with sample size limit."""
        dataset = MockDataset(size=100)

        results = validate_safe_dataset(dataset, sample_size=10)

        assert len(results) > 0


class TestValidatorEdgeCases:
    """Tests for edge cases in validation."""

    def test_dataset_missing_len_method(self):
        """Test handling of dataset without __len__."""
        class BadDataset:
            def __getitem__(self, idx):
                return {"question": "Q", "answer": "A"}

        validator = DatasetValidator()
        dataset = BadDataset()

        results = validator.validate_dataset(dataset)

        failed = [r for r in results if r.status == ValidationStatus.FAILED]
        assert len(failed) > 0

    def test_dataset_missing_getitem_method(self):
        """Test handling of dataset without __getitem__."""
        class BadDataset:
            def __len__(self):
                return 10

        validator = DatasetValidator()
        dataset = BadDataset()

        results = validator.validate_dataset(dataset)

        failed = [r for r in results if r.status == ValidationStatus.FAILED]
        assert len(failed) > 0

    def test_sample_raises_exception(self):
        """Test handling when sample loading raises exception."""
        class FailingDataset:
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                if idx == 1:
                    raise RuntimeError("Sample loading failed")
                return {"question": "Q", "answer": "A"}

        validator = DatasetValidator()
        dataset = FailingDataset()

        results = validator.validate_dataset(dataset)

        # Should record the failure but continue
        failed = [r for r in results if r.status == ValidationStatus.FAILED]
        assert any("sample_1" in r.name for r in failed)
        assert 1 in validator.stats.invalid_samples

    def test_non_tensor_audio(self):
        """Test handling of non-tensor audio."""
        validator = DatasetValidator()
        samples = [
            {
                "question": "Test?",
                "answer": "Answer",
                "audio": [1.0, 2.0, 3.0],  # List instead of tensor
                "images": None,
            }
        ]
        dataset = MockDataset(samples=samples)

        results = validator.validate_dataset(dataset)

        # Should warn about non-tensor audio
        warning_results = [r for r in results if r.status == ValidationStatus.WARNING]
        assert len(warning_results) > 0

    def test_non_tensor_images(self):
        """Test handling of non-tensor images."""
        validator = DatasetValidator()
        samples = [
            {
                "question": "Test?",
                "answer": "Answer",
                "audio": None,
                "images": [[1, 2], [3, 4]],  # List instead of tensor
            }
        ]
        dataset = MockDataset(samples=samples)

        results = validator.validate_dataset(dataset)

        warning_results = [r for r in results if r.status == ValidationStatus.WARNING]
        assert len(warning_results) > 0
