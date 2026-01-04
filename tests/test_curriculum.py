"""
Tests for the SAFE curriculum learning module.
"""
import pytest
import json
import yaml
import tempfile
from pathlib import Path

import torch

from safe.data.curriculum import (
    CurriculumConfig,
    CurriculumManager,
    CurriculumStage,
    DifficultyLevel,
    ProgressionStatus,
)


class TestDifficultyLevel:
    """Tests for DifficultyLevel enum."""

    def test_difficulty_levels_exist(self):
        """Test that all expected difficulty levels exist."""
        assert DifficultyLevel.EASY.value == "easy"
        assert DifficultyLevel.MEDIUM.value == "medium"
        assert DifficultyLevel.HARD.value == "hard"

    def test_difficulty_level_from_string(self):
        """Test creating DifficultyLevel from string."""
        assert DifficultyLevel("easy") == DifficultyLevel.EASY
        assert DifficultyLevel("medium") == DifficultyLevel.MEDIUM
        assert DifficultyLevel("hard") == DifficultyLevel.HARD


class TestProgressionStatus:
    """Tests for ProgressionStatus enum."""

    def test_progression_statuses_exist(self):
        """Test that all expected statuses exist."""
        assert ProgressionStatus.CONTINUE.value == "continue"
        assert ProgressionStatus.ADVANCE.value == "advance"
        assert ProgressionStatus.EXTEND.value == "extend"
        assert ProgressionStatus.FAILED.value == "failed"


class TestCurriculumStage:
    """Tests for CurriculumStage dataclass."""

    def test_stage_creation(self):
        """Test creating a CurriculumStage."""
        raw_config = {
            "duration_epochs": 5,
            "audio_ratio": 0.5,
            "difficulty_filter": "easy",
            "loss_weights": {"audio": 1.0, "retention": 0.5},
            "criteria": {"min_audio_accuracy": 0.7},
        }
        stage = CurriculumStage(name="stage1", idx=0, raw_config=raw_config)

        assert stage.name == "stage1"
        assert stage.idx == 0
        assert stage.duration_epochs == 5
        assert stage.audio_ratio == 0.5
        assert stage.difficulty_filter == "easy"
        assert stage.loss_weights == {"audio": 1.0, "retention": 0.5}
        assert stage.criteria == {"min_audio_accuracy": 0.7}

    def test_stage_default_values(self):
        """Test CurriculumStage uses defaults for missing config."""
        stage = CurriculumStage(name="minimal", idx=0, raw_config={})

        assert stage.duration_epochs == 1
        assert stage.audio_ratio == 0.0
        assert stage.difficulty_filter is None
        assert stage.loss_weights == {}
        assert stage.criteria == {}


class TestCurriculumConfig:
    """Tests for CurriculumConfig class."""

    @pytest.fixture
    def sample_config(self):
        """Sample curriculum configuration."""
        return {
            "stages": {
                "easy": {
                    "duration_epochs": 2,
                    "audio_ratio": 0.3,
                    "criteria": {"min_audio_accuracy": 0.6},
                },
                "medium": {
                    "duration_epochs": 3,
                    "audio_ratio": 0.5,
                    "criteria": {"min_audio_accuracy": 0.7},
                },
                "hard": {
                    "duration_epochs": 5,
                    "audio_ratio": 0.7,
                    "criteria": {"min_audio_accuracy": 0.8},
                },
            },
            "settings": {"auto_progression": True},
            "adaptation": {"max_stage_extensions": 2},
        }

    def test_config_from_dict(self, sample_config):
        """Test creating CurriculumConfig from dictionary."""
        config = CurriculumConfig(sample_config)

        assert config.get_num_stages() == 3
        assert config.get_stage(0).name == "easy"
        assert config.get_stage(1).name == "medium"
        assert config.get_stage(2).name == "hard"

    def test_config_empty_stages_raises(self):
        """Test that empty stages raises ValueError."""
        with pytest.raises(ValueError, match="must define at least one stage"):
            CurriculumConfig({"stages": {}})

    def test_config_missing_stages_raises(self):
        """Test that missing stages key raises ValueError."""
        with pytest.raises(ValueError):
            CurriculumConfig({})

    def test_config_stage_properties(self, sample_config):
        """Test accessing stage properties."""
        config = CurriculumConfig(sample_config)

        easy_stage = config.get_stage(0)
        assert easy_stage.duration_epochs == 2
        assert easy_stage.audio_ratio == 0.3
        assert easy_stage.criteria["min_audio_accuracy"] == 0.6

    def test_config_stage_index_out_of_range(self, sample_config):
        """Test accessing stage with invalid index raises IndexError."""
        config = CurriculumConfig(sample_config)

        with pytest.raises(IndexError):
            config.get_stage(5)

        with pytest.raises(IndexError):
            config.get_stage(-1)

    def test_config_stages_iterable(self, sample_config):
        """Test stages property returns iterable."""
        config = CurriculumConfig(sample_config)

        stages = list(config.stages)
        assert len(stages) == 3
        assert all(isinstance(s, CurriculumStage) for s in stages)

    def test_config_settings_property(self, sample_config):
        """Test settings property access."""
        config = CurriculumConfig(sample_config)

        assert config.settings["auto_progression"] is True

    def test_config_adaptation_property(self, sample_config):
        """Test adaptation property access."""
        config = CurriculumConfig(sample_config)

        assert config.adaptation["max_stage_extensions"] == 2

    def test_config_from_yaml_file(self, sample_config):
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(sample_config, f)
            f.flush()
            path = Path(f.name)

        try:
            config = CurriculumConfig.from_file(path)
            assert config.get_num_stages() == 3
        finally:
            path.unlink()

    def test_config_from_json_file(self, sample_config):
        """Test loading config from JSON file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(sample_config, f)
            f.flush()
            path = Path(f.name)

        try:
            config = CurriculumConfig.from_file(path)
            assert config.get_num_stages() == 3
        finally:
            path.unlink()

    def test_config_from_any_dict(self, sample_config):
        """Test from_any with dictionary input."""
        config = CurriculumConfig.from_any(sample_config)
        assert config.get_num_stages() == 3

    def test_config_from_any_existing_config(self, sample_config):
        """Test from_any with existing CurriculumConfig."""
        original = CurriculumConfig(sample_config)
        config = CurriculumConfig.from_any(original)
        assert config is original

    def test_config_from_any_invalid_type(self):
        """Test from_any raises for invalid type."""
        with pytest.raises(TypeError, match="Unsupported curriculum config type"):
            CurriculumConfig.from_any(12345)


class TestCurriculumManager:
    """Tests for CurriculumManager class."""

    @pytest.fixture
    def sample_config(self):
        """Sample curriculum configuration."""
        return {
            "stages": {
                "easy": {
                    "duration_epochs": 2,
                    "audio_ratio": 0.3,
                    "criteria": {
                        "min_audio_accuracy": 0.6,
                        "max_vl_degradation": 0.05,
                    },
                },
                "medium": {
                    "duration_epochs": 2,
                    "audio_ratio": 0.5,
                    "criteria": {"min_audio_accuracy": 0.7},
                },
                "hard": {
                    "duration_epochs": 2,
                    "audio_ratio": 0.7,
                    "criteria": {"min_audio_accuracy": 0.8},
                },
            },
            "settings": {"auto_progression": True},
            "adaptation": {
                "max_stage_extensions": 2,
                "extension_epochs": 1,
            },
        }

    @pytest.fixture
    def manager(self, sample_config):
        """Create a CurriculumManager instance."""
        return CurriculumManager(sample_config)

    def test_manager_initialization(self, manager):
        """Test CurriculumManager initializes correctly."""
        assert manager.current_stage_idx == 0
        assert manager.current_stage.name == "easy"
        assert manager.epochs_in_stage == 0
        assert manager.stage_extensions == 0
        assert manager.is_completed is False

    def test_manager_set_baseline_metrics(self, manager):
        """Test setting baseline metrics."""
        baseline = {"vl_retention": 0.95, "audio_accuracy": 0.7}
        manager.set_baseline_metrics(baseline)

        assert manager.baseline_metrics == baseline

    def test_manager_update_metrics(self, manager):
        """Test updating current metrics."""
        manager.update_metrics({"audio_accuracy": 0.65}, samples_processed=100)

        assert manager.current_metrics["audio_accuracy"] == 0.65
        assert manager.samples_in_stage == 100

    def test_advance_epoch_continue(self, manager):
        """Test advance_epoch returns CONTINUE when criteria not met."""
        manager.update_metrics({"audio_accuracy": 0.3})  # Below threshold

        status = manager.advance_epoch()

        assert status == ProgressionStatus.CONTINUE
        assert manager.epochs_in_stage == 1

    def test_advance_epoch_advance_stage(self, manager):
        """Test advance_epoch advances stage when criteria met."""
        manager.set_baseline_metrics({"vl_retention": 0.95})
        manager.update_metrics({"audio_accuracy": 0.7, "vl_retention": 0.94})

        # Complete required epochs
        manager.advance_epoch()  # epoch 1
        status = manager.advance_epoch()  # epoch 2 (duration)

        assert status == ProgressionStatus.ADVANCE
        assert manager.current_stage_idx == 1
        assert manager.current_stage.name == "medium"

    def test_advance_epoch_extend_stage(self, manager):
        """Test advance_epoch extends stage when criteria not met at duration."""
        manager.update_metrics({"audio_accuracy": 0.3})  # Below threshold

        # Complete duration epochs without meeting criteria
        manager.advance_epoch()  # epoch 1
        status = manager.advance_epoch()  # epoch 2 (duration reached, criteria not met)

        assert status == ProgressionStatus.EXTEND
        assert manager.stage_extensions == 1
        assert manager.epochs_in_stage == 0  # Reset for extension

    def test_advance_epoch_failed_after_max_extensions(self, manager):
        """Test advance_epoch returns FAILED after max extensions."""
        manager.update_metrics({"audio_accuracy": 0.3})

        # Exhaust duration and extensions
        for _ in range(2):  # Duration
            manager.advance_epoch()
        for _ in range(2):  # Extension 1 duration
            manager.advance_epoch()
        for _ in range(2):  # Extension 2 duration
            status = manager.advance_epoch()

        assert status == ProgressionStatus.FAILED
        assert manager.is_completed is True

    def test_advance_epoch_after_completion(self, manager):
        """Test advance_epoch returns FAILED after curriculum completed."""
        manager.is_completed = True

        status = manager.advance_epoch()

        assert status == ProgressionStatus.FAILED

    def test_criteria_satisfied_no_criteria(self, manager):
        """Test criteria satisfied when stage has no criteria."""
        manager.current_stage = CurriculumStage(
            name="no_criteria", idx=0, raw_config={}
        )

        assert manager._criteria_satisfied() is True

    def test_criteria_satisfied_audio_accuracy(self, manager):
        """Test criteria satisfaction for audio accuracy."""
        manager.update_metrics({"audio_accuracy": 0.7})
        assert manager._criteria_satisfied() is True

        manager.update_metrics({"audio_accuracy": 0.5})
        assert manager._criteria_satisfied() is False

    def test_criteria_satisfied_vl_degradation(self, manager):
        """Test criteria satisfaction for VL degradation."""
        manager.set_baseline_metrics({"vl_retention": 1.0})

        # Within tolerance (5% degradation allowed)
        manager.update_metrics({"vl_retention": 0.96, "audio_accuracy": 0.7})
        assert manager._criteria_satisfied() is True

        # Exceeds tolerance
        manager.update_metrics({"vl_retention": 0.9, "audio_accuracy": 0.7})
        assert manager._criteria_satisfied() is False

    def test_get_current_config(self, manager):
        """Test getting current stage configuration."""
        config = manager.get_current_config()

        assert config is not None
        assert config["stage_name"] == "easy"
        assert config["stage_idx"] == 0
        assert config["audio_ratio"] == 0.3

    def test_get_current_config_when_completed(self, manager):
        """Test get_current_config returns None when completed."""
        manager.is_completed = True

        assert manager.get_current_config() is None

    def test_get_progress_summary(self, manager):
        """Test getting progress summary."""
        manager.update_metrics({"audio_accuracy": 0.65}, samples_processed=500)
        manager.advance_epoch()

        summary = manager.get_progress_summary()

        assert summary["current_stage"] == "easy"
        assert summary["current_stage_idx"] == 0
        assert summary["total_stages"] == 3
        assert summary["is_completed"] is False
        assert summary["samples_in_stage"] == 500
        assert "audio_accuracy" in summary["current_metrics"]

    def test_checkpoint_save_load(self, manager, tmp_path):
        """Test saving and loading checkpoints."""
        # Progress through curriculum
        manager.set_baseline_metrics({"vl_retention": 0.95})
        manager.update_metrics({"audio_accuracy": 0.65}, samples_processed=1000)
        manager.advance_epoch()

        # Save checkpoint
        checkpoint_path = tmp_path / "curriculum_checkpoint.pt"
        manager.save_checkpoint(checkpoint_path)

        # Create new manager and load checkpoint
        new_manager = CurriculumManager(manager.config)
        new_manager.load_checkpoint(checkpoint_path)

        assert new_manager.current_stage_idx == manager.current_stage_idx
        assert new_manager.epochs_in_stage == manager.epochs_in_stage
        assert new_manager.samples_in_stage == manager.samples_in_stage
        assert new_manager.current_metrics == manager.current_metrics

    def test_manual_progression_mode(self, sample_config):
        """Test manual progression mode (auto_progression=False)."""
        sample_config["settings"]["auto_progression"] = False
        manager = CurriculumManager(sample_config)

        # Even with criteria met, should return CONTINUE until duration reached
        manager.update_metrics({"audio_accuracy": 0.9})
        status = manager.advance_epoch()

        assert status == ProgressionStatus.CONTINUE

    def test_history_tracking(self, manager):
        """Test that history is tracked properly."""
        manager.update_metrics({"audio_accuracy": 0.5}, samples_processed=100)
        manager.advance_epoch()
        manager.update_metrics({"audio_accuracy": 0.55}, samples_processed=200)
        manager.advance_epoch()

        assert len(manager.history) == 2
        assert manager.history[0]["epoch"] == 1
        assert manager.history[1]["epoch"] == 2
        assert manager.history[0]["stage"] == "easy"

    def test_complete_curriculum_successfully(self, manager):
        """Test completing entire curriculum successfully."""
        manager.set_baseline_metrics({"vl_retention": 0.95})

        stages_completed = 0
        for stage_idx in range(3):
            # Meet criteria for current stage
            min_acc = manager.current_stage.criteria.get("min_audio_accuracy", 0.5)
            manager.update_metrics({
                "audio_accuracy": min_acc + 0.1,
                "vl_retention": 0.94,
            })

            # Complete duration epochs
            for _ in range(manager.current_stage.duration_epochs):
                status = manager.advance_epoch()
                if status == ProgressionStatus.ADVANCE:
                    stages_completed += 1
                    break

            if manager.is_completed:
                break

        assert manager.is_completed is True
        assert stages_completed == 3  # All 3 stages completed
