"""
Dataset Validation and Quality Assurance Framework for SAFE

This module provides comprehensive validation tools for ensuring dataset quality,
consistency, and suitability for multimodal learning.
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
from enum import Enum
import json
import hashlib
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation thoroughness levels."""
    BASIC = "basic"          # Quick validation
    STANDARD = "standard"    # Normal validation
    COMPREHENSIVE = "comprehensive"  # Full validation


class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"
    WARNING = "warning"  
    FAILED = "failed"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class DatasetStats:
    """Statistics about a dataset."""
    total_samples: int = 0
    audio_samples: int = 0
    visual_samples: int = 0
    multimodal_samples: int = 0
    
    # Audio statistics
    audio_duration_stats: Dict[str, float] = field(default_factory=dict)
    audio_sample_rate_distribution: Counter = field(default_factory=Counter)
    audio_channels_distribution: Counter = field(default_factory=Counter)
    
    # Visual statistics  
    image_size_distribution: Counter = field(default_factory=Counter)
    image_format_distribution: Counter = field(default_factory=Counter)
    
    # Text statistics
    question_length_stats: Dict[str, float] = field(default_factory=dict)
    answer_length_stats: Dict[str, float] = field(default_factory=dict)
    vocabulary_size: int = 0
    
    # Quality metrics
    invalid_samples: List[int] = field(default_factory=list)
    duplicate_samples: List[Tuple[int, int]] = field(default_factory=list)
    consistency_issues: List[Dict] = field(default_factory=list)


class DatasetValidator:
    """Comprehensive dataset validation and quality assessment."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """Initialize dataset validator.
        
        Args:
            validation_level: Thoroughness of validation checks
        """
        self.validation_level = validation_level
        self.results = []
        self.stats = DatasetStats()
        
    def validate_dataset(self, dataset, sample_size: Optional[int] = None) -> List[ValidationResult]:
        """Validate a complete dataset.
        
        Args:
            dataset: Dataset object to validate
            sample_size: Number of samples to check (None for all)
            
        Returns:
            List of validation results
        """
        self.results = []
        self.stats = DatasetStats()
        
        logger.info(f"Starting {self.validation_level.value} validation of dataset")
        
        # Determine sample size
        total_size = len(dataset)
        if sample_size is None:
            sample_size = total_size
        else:
            sample_size = min(sample_size, total_size)
            
        # Sample indices to validate
        if sample_size < total_size:
            indices = np.random.choice(total_size, sample_size, replace=False)
        else:
            indices = range(total_size)
            
        # Basic dataset structure validation
        self._validate_dataset_structure(dataset)
        
        # Sample-level validation
        self._validate_samples(dataset, indices)
        
        # Dataset-level validation
        self._validate_dataset_consistency(dataset, indices)
        
        # Multimodal-specific validation
        self._validate_multimodal_alignment(dataset, indices)
        
        # Curriculum learning compatibility
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
            self._validate_curriculum_compatibility(dataset, indices)
            
        # Performance and efficiency validation
        if self.validation_level == ValidationLevel.COMPREHENSIVE:
            self._validate_performance_characteristics(dataset, indices)
            
        self._summarize_validation()
        return self.results
        
    def _validate_dataset_structure(self, dataset):
        """Validate basic dataset structure and interface."""
        try:
            # Check required methods
            required_methods = ["__len__", "__getitem__"]
            for method in required_methods:
                if not hasattr(dataset, method):
                    self._add_result(ValidationResult(
                        "dataset_interface",
                        ValidationStatus.FAILED,
                        f"Dataset missing required method: {method}"
                    ))
                    return
                    
            # Check dataset size
            dataset_size = len(dataset)
            if dataset_size == 0:
                self._add_result(ValidationResult(
                    "dataset_size",
                    ValidationStatus.FAILED,
                    "Dataset is empty"
                ))
                return
            elif dataset_size < 10:
                self._add_result(ValidationResult(
                    "dataset_size",
                    ValidationStatus.WARNING,
                    f"Dataset very small ({dataset_size} samples)",
                    suggestions=["Consider using a larger dataset for meaningful training"]
                ))
            else:
                self._add_result(ValidationResult(
                    "dataset_size",
                    ValidationStatus.PASSED,
                    f"Dataset contains {dataset_size} samples"
                ))
                
            self.stats.total_samples = dataset_size
            
        except Exception as e:
            self._add_result(ValidationResult(
                "dataset_structure",
                ValidationStatus.FAILED,
                f"Error validating dataset structure: {str(e)}"
            ))
            
    def _validate_samples(self, dataset, indices):
        """Validate individual samples."""
        logger.info(f"Validating {len(indices)} samples")
        
        for i, idx in enumerate(indices):
            try:
                sample = dataset[idx]
                self._validate_sample_structure(sample, idx)
                self._validate_sample_content(sample, idx)
                
                # Update statistics
                self._update_sample_stats(sample)
                
            except Exception as e:
                self._add_result(ValidationResult(
                    f"sample_{idx}",
                    ValidationStatus.FAILED,
                    f"Error loading sample {idx}: {str(e)}"
                ))
                self.stats.invalid_samples.append(idx)
                
            # Progress logging
            if (i + 1) % max(1, len(indices) // 10) == 0:
                logger.info(f"Validated {i + 1}/{len(indices)} samples")
                
    def _validate_sample_structure(self, sample: Dict, idx: int):
        """Validate the structure of a single sample."""
        required_keys = ["question", "answer"]
        optional_keys = ["audio", "images", "question_type"]
        
        # Check required keys
        for key in required_keys:
            if key not in sample:
                self._add_result(ValidationResult(
                    f"sample_{idx}_structure",
                    ValidationStatus.FAILED,
                    f"Sample {idx} missing required key: {key}"
                ))
                
        # Check data types
        if "question" in sample and not isinstance(sample["question"], str):
            self._add_result(ValidationResult(
                f"sample_{idx}_types",
                ValidationStatus.WARNING,
                f"Sample {idx} question is not a string"
            ))
            
        if "answer" in sample and not isinstance(sample["answer"], str):
            self._add_result(ValidationResult(
                f"sample_{idx}_types",
                ValidationStatus.WARNING,
                f"Sample {idx} answer is not a string"
            ))
            
    def _validate_sample_content(self, sample: Dict, idx: int):
        """Validate the content quality of a single sample."""
        # Audio validation
        if sample.get("audio") is not None:
            self._validate_audio_content(sample["audio"], idx)
            
        # Image validation
        if sample.get("images") is not None:
            self._validate_image_content(sample["images"], idx)
            
        # Text validation
        self._validate_text_content(sample, idx)
        
    def _validate_audio_content(self, audio: torch.Tensor, idx: int):
        """Validate audio tensor content."""
        try:
            if not isinstance(audio, torch.Tensor):
                self._add_result(ValidationResult(
                    f"sample_{idx}_audio",
                    ValidationStatus.WARNING,
                    f"Audio in sample {idx} is not a torch.Tensor"
                ))
                return
                
            # Check dimensions
            if audio.dim() == 1:
                # Mono audio
                self.stats.audio_channels_distribution[1] += 1
            elif audio.dim() == 2 and audio.shape[0] <= 2:
                # Multi-channel audio
                self.stats.audio_channels_distribution[audio.shape[0]] += 1
            else:
                self._add_result(ValidationResult(
                    f"sample_{idx}_audio",
                    ValidationStatus.WARNING,
                    f"Audio in sample {idx} has unusual shape: {audio.shape}"
                ))
                
            # Check for valid audio data
            if torch.isnan(audio).any():
                self._add_result(ValidationResult(
                    f"sample_{idx}_audio",
                    ValidationStatus.FAILED,
                    f"Audio in sample {idx} contains NaN values"
                ))
                
            if torch.isinf(audio).any():
                self._add_result(ValidationResult(
                    f"sample_{idx}_audio",
                    ValidationStatus.FAILED,
                    f"Audio in sample {idx} contains infinite values"
                ))
                
            # Check audio range
            audio_max = audio.abs().max().item()
            if audio_max > 10.0:  # Suspiciously large values
                self._add_result(ValidationResult(
                    f"sample_{idx}_audio",
                    ValidationStatus.WARNING,
                    f"Audio in sample {idx} has large values (max: {audio_max:.2f})",
                    suggestions=["Consider normalizing audio data"]
                ))
                
        except Exception as e:
            self._add_result(ValidationResult(
                f"sample_{idx}_audio",
                ValidationStatus.FAILED,
                f"Error validating audio in sample {idx}: {str(e)}"
            ))
            
    def _validate_image_content(self, images: torch.Tensor, idx: int):
        """Validate image tensor content."""
        try:
            if not isinstance(images, torch.Tensor):
                self._add_result(ValidationResult(
                    f"sample_{idx}_images",
                    ValidationStatus.WARNING,
                    f"Images in sample {idx} is not a torch.Tensor"
                ))
                return
                
            # Check dimensions
            if images.dim() == 3:  # Single image (C, H, W)
                c, h, w = images.shape
                self.stats.image_size_distribution[(h, w)] += 1
            elif images.dim() == 4:  # Multiple images (N, C, H, W)
                n, c, h, w = images.shape
                self.stats.image_size_distribution[(h, w)] += n
            else:
                self._add_result(ValidationResult(
                    f"sample_{idx}_images",
                    ValidationStatus.WARNING,
                    f"Images in sample {idx} has unusual shape: {images.shape}"
                ))
                
            # Check for valid image data
            if torch.isnan(images).any():
                self._add_result(ValidationResult(
                    f"sample_{idx}_images",
                    ValidationStatus.FAILED,
                    f"Images in sample {idx} contains NaN values"
                ))
                
            if torch.isinf(images).any():
                self._add_result(ValidationResult(
                    f"sample_{idx}_images",
                    ValidationStatus.FAILED,
                    f"Images in sample {idx} contains infinite values"
                ))
                
            # Check image value range
            img_min, img_max = images.min().item(), images.max().item()
            if img_min < -2.0 or img_max > 2.0:  # Reasonable range for normalized images
                self._add_result(ValidationResult(
                    f"sample_{idx}_images",
                    ValidationStatus.WARNING,
                    f"Images in sample {idx} has unusual value range: [{img_min:.2f}, {img_max:.2f}]",
                    suggestions=["Check image normalization"]
                ))
                
        except Exception as e:
            self._add_result(ValidationResult(
                f"sample_{idx}_images",
                ValidationStatus.FAILED,
                f"Error validating images in sample {idx}: {str(e)}"
            ))
            
    def _validate_text_content(self, sample: Dict, idx: int):
        """Validate text content quality."""
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        
        # Check for empty text
        if not question.strip():
            self._add_result(ValidationResult(
                f"sample_{idx}_text",
                ValidationStatus.FAILED,
                f"Sample {idx} has empty question"
            ))
            
        if not answer.strip():
            self._add_result(ValidationResult(
                f"sample_{idx}_text",
                ValidationStatus.FAILED,
                f"Sample {idx} has empty answer"
            ))
            
        # Check for reasonable length
        if len(question) > 500:
            self._add_result(ValidationResult(
                f"sample_{idx}_text",
                ValidationStatus.WARNING,
                f"Sample {idx} has very long question ({len(question)} chars)"
            ))
            
        if len(answer) > 200:
            self._add_result(ValidationResult(
                f"sample_{idx}_text",
                ValidationStatus.WARNING,
                f"Sample {idx} has very long answer ({len(answer)} chars)"
            ))
            
        # Check for special characters that might cause issues
        problematic_chars = ['\x00', '\x01', '\x02', '\x03']
        for char in problematic_chars:
            if char in question or char in answer:
                self._add_result(ValidationResult(
                    f"sample_{idx}_text",
                    ValidationStatus.WARNING,
                    f"Sample {idx} contains problematic characters"
                ))
                break
                
    def _update_sample_stats(self, sample: Dict):
        """Update dataset statistics with sample information."""
        # Count modalities
        has_audio = sample.get("audio") is not None
        has_images = sample.get("images") is not None
        
        if has_audio:
            self.stats.audio_samples += 1
        if has_images:
            self.stats.visual_samples += 1
        if has_audio and has_images:
            self.stats.multimodal_samples += 1
            
        # Update text statistics
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        
        if not hasattr(self, '_question_lengths'):
            self._question_lengths = []
            self._answer_lengths = []
            
        self._question_lengths.append(len(question))
        self._answer_lengths.append(len(answer))
        
    def _validate_dataset_consistency(self, dataset, indices):
        """Validate consistency across the dataset."""
        logger.info("Validating dataset consistency")
        
        # Calculate text statistics
        if hasattr(self, '_question_lengths'):
            self.stats.question_length_stats = {
                'mean': np.mean(self._question_lengths),
                'std': np.std(self._question_lengths),
                'min': np.min(self._question_lengths),
                'max': np.max(self._question_lengths)
            }
            
        if hasattr(self, '_answer_lengths'):
            self.stats.answer_length_stats = {
                'mean': np.mean(self._answer_lengths),
                'std': np.std(self._answer_lengths),
                'min': np.min(self._answer_lengths),
                'max': np.max(self._answer_lengths)
            }
            
        # Check modality distribution
        total_validated = len(indices)
        audio_ratio = self.stats.audio_samples / total_validated
        visual_ratio = self.stats.visual_samples / total_validated
        multimodal_ratio = self.stats.multimodal_samples / total_validated
        
        self._add_result(ValidationResult(
            "modality_distribution",
            ValidationStatus.PASSED,
            f"Modality distribution - Audio: {audio_ratio:.2%}, Visual: {visual_ratio:.2%}, Multimodal: {multimodal_ratio:.2%}",
            details={
                "audio_ratio": audio_ratio,
                "visual_ratio": visual_ratio,
                "multimodal_ratio": multimodal_ratio
            }
        ))
        
        # Warn about imbalanced datasets
        if audio_ratio < 0.1:
            self._add_result(ValidationResult(
                "modality_balance",
                ValidationStatus.WARNING,
                "Very few audio samples in dataset",
                suggestions=["Consider adding more audio-dependent samples for SAFE training"]
            ))
            
        if visual_ratio < 0.1:
            self._add_result(ValidationResult(
                "modality_balance",
                ValidationStatus.WARNING,
                "Very few visual samples in dataset",
                suggestions=["Consider adding more visual samples for VL retention"]
            ))
            
    def _validate_multimodal_alignment(self, dataset, indices):
        """Validate alignment between different modalities."""
        logger.info("Validating multimodal alignment")
        
        # This is a placeholder for more sophisticated alignment checks
        # In practice, you might check:
        # - Audio-visual synchronization
        # - Semantic consistency between modalities and text
        # - Temporal alignment for video data
        
        alignment_issues = 0
        
        # Simple heuristic checks
        for idx in indices[:min(100, len(indices))]:  # Sample check
            try:
                sample = dataset[idx]
                
                # Check if question mentions audio/visual elements appropriately
                question = sample.get("question", "").lower()
                has_audio = sample.get("audio") is not None
                has_images = sample.get("images") is not None
                
                # Audio keywords but no audio data
                audio_keywords = ["sound", "hear", "audio", "noise", "music"]
                mentions_audio = any(keyword in question for keyword in audio_keywords)
                
                if mentions_audio and not has_audio:
                    alignment_issues += 1
                    
                # Visual keywords but no visual data
                visual_keywords = ["see", "look", "image", "picture", "visual", "color"]
                mentions_visual = any(keyword in question for keyword in visual_keywords)
                
                if mentions_visual and not has_images:
                    alignment_issues += 1
                    
            except Exception:
                continue
                
        if alignment_issues > 0:
            issue_rate = alignment_issues / min(100, len(indices))
            if issue_rate > 0.1:  # More than 10% issues
                self._add_result(ValidationResult(
                    "multimodal_alignment",
                    ValidationStatus.WARNING,
                    f"Potential modality-text misalignment detected ({issue_rate:.1%} of samples)",
                    suggestions=["Review questions for consistency with available modalities"]
                ))
            else:
                self._add_result(ValidationResult(
                    "multimodal_alignment",
                    ValidationStatus.PASSED,
                    f"Good modality-text alignment ({issue_rate:.1%} issues)"
                ))
        else:
            self._add_result(ValidationResult(
                "multimodal_alignment",
                ValidationStatus.PASSED,
                "No obvious modality-text misalignment detected"
            ))
            
    def _validate_curriculum_compatibility(self, dataset, indices):
        """Validate compatibility with curriculum learning."""
        logger.info("Validating curriculum learning compatibility")
        
        # Check if dataset supports difficulty levels
        has_difficulty_info = False
        if hasattr(dataset, 'get_difficulty'):
            has_difficulty_info = True
        elif hasattr(dataset, 'difficulty_levels'):
            has_difficulty_info = True
            
        if has_difficulty_info:
            self._add_result(ValidationResult(
                "curriculum_compatibility",
                ValidationStatus.PASSED,
                "Dataset supports difficulty-based curriculum learning"
            ))
        else:
            self._add_result(ValidationResult(
                "curriculum_compatibility",
                ValidationStatus.WARNING,
                "Dataset does not provide difficulty information",
                suggestions=["Implement difficulty scoring for curriculum learning"]
            ))
            
        # Check for question type diversity
        question_types = set()
        for idx in indices[:min(500, len(indices))]:
            try:
                sample = dataset[idx]
                qtype = sample.get("question_type", "unknown")
                question_types.add(qtype)
            except Exception:
                continue
                
        if len(question_types) > 1:
            self._add_result(ValidationResult(
                "question_type_diversity",
                ValidationStatus.PASSED,
                f"Dataset has {len(question_types)} question types: {list(question_types)}"
            ))
        else:
            self._add_result(ValidationResult(
                "question_type_diversity",
                ValidationStatus.WARNING,
                "Dataset has limited question type diversity",
                suggestions=["Add diverse question types for better curriculum learning"]
            ))
            
    def _validate_performance_characteristics(self, dataset, indices):
        """Validate performance characteristics for training."""
        logger.info("Validating performance characteristics")
        
        # Sample loading speed test
        import time
        
        sample_times = []
        for idx in indices[:min(50, len(indices))]:
            start_time = time.time()
            try:
                _ = dataset[idx]
                load_time = time.time() - start_time
                sample_times.append(load_time)
            except Exception:
                continue
                
        if sample_times:
            avg_load_time = np.mean(sample_times)
            max_load_time = np.max(sample_times)
            
            if avg_load_time > 1.0:  # More than 1 second per sample
                self._add_result(ValidationResult(
                    "loading_performance",
                    ValidationStatus.WARNING,
                    f"Slow sample loading (avg: {avg_load_time:.2f}s, max: {max_load_time:.2f}s)",
                    suggestions=[
                        "Consider preprocessing data",
                        "Use faster storage",
                        "Implement caching"
                    ]
                ))
            else:
                self._add_result(ValidationResult(
                    "loading_performance",
                    ValidationStatus.PASSED,
                    f"Good loading performance (avg: {avg_load_time:.3f}s)"
                ))
                
    def _add_result(self, result: ValidationResult):
        """Add a validation result."""
        self.results.append(result)
        
    def _summarize_validation(self):
        """Create validation summary."""
        passed = sum(1 for r in self.results if r.status == ValidationStatus.PASSED)
        warnings = sum(1 for r in self.results if r.status == ValidationStatus.WARNING)
        failed = sum(1 for r in self.results if r.status == ValidationStatus.FAILED)
        
        logger.info(f"Validation complete: {passed} passed, {warnings} warnings, {failed} failed")
        
        # Overall status
        if failed > 0:
            overall_status = ValidationStatus.FAILED
        elif warnings > 0:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASSED
            
        self._add_result(ValidationResult(
            "overall_validation",
            overall_status,
            f"Dataset validation summary: {passed}/{len(self.results)} checks passed",
            details={
                "passed": passed,
                "warnings": warnings,
                "failed": failed,
                "total_checks": len(self.results)
            }
        ))
        
    def save_report(self, output_path: Path):
        """Save validation report to file."""
        report = {
            "validation_level": self.validation_level.value,
            "timestamp": str(Path().cwd()),
            "statistics": {
                "total_samples": self.stats.total_samples,
                "audio_samples": self.stats.audio_samples,
                "visual_samples": self.stats.visual_samples,
                "multimodal_samples": self.stats.multimodal_samples,
                "invalid_samples": len(self.stats.invalid_samples),
                "question_length_stats": self.stats.question_length_stats,
                "answer_length_stats": self.stats.answer_length_stats
            },
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details,
                    "suggestions": r.suggestions
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Validation report saved to {output_path}")


def validate_safe_dataset(dataset, output_dir: Optional[Path] = None, 
                         validation_level: ValidationLevel = ValidationLevel.STANDARD,
                         sample_size: Optional[int] = None) -> List[ValidationResult]:
    """Convenience function to validate a SAFE dataset.
    
    Args:
        dataset: Dataset to validate
        output_dir: Directory to save validation report (optional)
        validation_level: Thoroughness of validation
        sample_size: Number of samples to validate (None for all)
        
    Returns:
        List of validation results
    """
    validator = DatasetValidator(validation_level)
    results = validator.validate_dataset(dataset, sample_size)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        validator.save_report(output_dir / "validation_report.json")
        
    return results