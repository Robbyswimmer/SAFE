"""Curriculum learning utilities for SAFE training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml


class DifficultyLevel(Enum):
    """Difficulty levels used by the mock datasets and curriculum configs."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ProgressionStatus(Enum):
    """Status returned after each curriculum epoch."""

    CONTINUE = "continue"
    ADVANCE = "advance"
    EXTEND = "extend"
    FAILED = "failed"


@dataclass
class CurriculumStage:
    """Single curriculum stage configuration."""

    name: str
    idx: int
    raw_config: Dict[str, Any]

    @property
    def duration_epochs(self) -> int:
        return int(self.raw_config.get("duration_epochs", 1))

    @property
    def audio_ratio(self) -> float:
        return float(self.raw_config.get("audio_ratio", 0.0))

    @property
    def difficulty_filter(self) -> Any:
        return self.raw_config.get("difficulty_filter")

    @property
    def loss_weights(self) -> Dict[str, float]:
        return self.raw_config.get("loss_weights", {})

    @property
    def criteria(self) -> Dict[str, Any]:
        return self.raw_config.get("criteria", {})


class CurriculumConfig:
    """Parsed curriculum configuration loaded from YAML/JSON or dict."""

    def __init__(self, config: Dict[str, Any]):
        if "stages" not in config or not config["stages"]:
            raise ValueError("Curriculum configuration must define at least one stage")

        self.raw = config
        self._stages: List[CurriculumStage] = []
        for idx, (name, cfg) in enumerate(config["stages"].items()):
            self._stages.append(CurriculumStage(name=name, idx=idx, raw_config=cfg))

    @classmethod
    def from_file(cls, path: Path) -> "CurriculumConfig":
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in {".yml", ".yaml"}:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(data)

    @classmethod
    def from_any(cls, source: Any) -> "CurriculumConfig":
        if isinstance(source, CurriculumConfig):
            return source
        if isinstance(source, (str, Path)):
            return cls.from_file(Path(source))
        if isinstance(source, dict):
            return cls(source)
        raise TypeError(f"Unsupported curriculum config type: {type(source)!r}")

    def get_num_stages(self) -> int:
        return len(self._stages)

    def get_stage(self, idx: int) -> CurriculumStage:
        if idx < 0 or idx >= len(self._stages):
            raise IndexError(f"Stage index {idx} out of range (0-{len(self._stages)-1})")
        return self._stages[idx]

    @property
    def stages(self) -> Iterable[CurriculumStage]:
        return tuple(self._stages)

    @property
    def settings(self) -> Dict[str, Any]:
        return self.raw.get("settings", {})

    @property
    def adaptation(self) -> Dict[str, Any]:
        return self.raw.get("adaptation", {})


class CurriculumManager:
    """Stateful helper that orchestrates curriculum progression."""

    def __init__(self, config: Any):
        self.config = CurriculumConfig.from_any(config)
        self.current_stage_idx = 0
        self.current_stage: CurriculumStage = self.config.get_stage(0)
        self.epochs_in_stage = 0
        self.stage_extensions = 0
        self.samples_in_stage = 0
        self.current_metrics: Dict[str, float] = {}
        self.baseline_metrics: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []
        self.is_completed = False

        adaptation = self.config.adaptation
        self.max_stage_extensions = int(adaptation.get("max_stage_extensions", 0))
        self.extension_epochs = int(adaptation.get("extension_epochs", 1))
        self.auto_progression = bool(self.config.settings.get("auto_progression", True))

    def set_baseline_metrics(self, metrics: Dict[str, float]) -> None:
        self.baseline_metrics = metrics or {}

    # ------------------------------------------------------------------
    # Metrics + bookkeeping
    # ------------------------------------------------------------------
    def update_metrics(self, metrics: Dict[str, float], samples_processed: int = 0) -> None:
        """Update running metrics for the current stage."""

        if metrics:
            self.current_metrics.update(metrics)
        self.samples_in_stage = samples_processed or self.samples_in_stage

    # ------------------------------------------------------------------
    def _criteria_satisfied(self) -> bool:
        criteria = self.current_stage.criteria
        if not criteria:
            return True

        metrics = self.current_metrics or {}

        # Helper for VL degradation using baseline retention
        def retention_ok() -> bool:
            if "max_vl_degradation" not in criteria:
                return True
            baseline = self.baseline_metrics.get("vl_retention")
            if baseline is None or baseline <= 0:
                return True
            current = metrics.get("vl_retention")
            if current is None:
                return False
            degradation = max(0.0, (baseline - current) / max(baseline, 1e-8))
            return degradation <= float(criteria["max_vl_degradation"])

        checks = [retention_ok()]

        if "min_audio_accuracy" in criteria:
            target = float(criteria["min_audio_accuracy"])
            checks.append(metrics.get("audio_accuracy", 0.0) >= target)

        if "efficiency_min_skip_rate" in criteria:
            target = float(criteria["efficiency_min_skip_rate"])
            checks.append(metrics.get("efficiency_rate", 0.0) >= target)

        if "min_samples_processed" in criteria:
            checks.append(self.samples_in_stage >= int(criteria["min_samples_processed"]))

        return all(checks)

    # ------------------------------------------------------------------
    def advance_epoch(self) -> ProgressionStatus:
        """Advance the curriculum by one epoch and evaluate progression."""

        if self.is_completed:
            return ProgressionStatus.FAILED

        self.epochs_in_stage += 1
        self.history.append({
            "stage": self.current_stage.name,
            "stage_idx": self.current_stage_idx,
            "metrics": dict(self.current_metrics),
            "samples": self.samples_in_stage,
            "epoch": self.epochs_in_stage,
        })

        criteria_met = self._criteria_satisfied()

        duration = self.current_stage.duration_epochs
        if not self.auto_progression:
            # Manual mode only tracks epochs
            if self.epochs_in_stage >= duration:
                return ProgressionStatus.ADVANCE if criteria_met else ProgressionStatus.CONTINUE
            return ProgressionStatus.CONTINUE

        if criteria_met and self.epochs_in_stage >= duration:
            self._advance_stage()
            return ProgressionStatus.ADVANCE

        if criteria_met and self.epochs_in_stage >= max(1, duration - 1):
            # Allow early progression when clearly above targets
            early_threshold = float(self.config.adaptation.get("early_progression_threshold", 1.0))
            audio = self.current_metrics.get("audio_accuracy", 0.0)
            target = float(self.current_stage.criteria.get("min_audio_accuracy", 0.0))
            if target > 0 and audio >= early_threshold * target:
                self._advance_stage()
                return ProgressionStatus.ADVANCE

        if self.epochs_in_stage >= duration:
            if self.stage_extensions < self.max_stage_extensions:
                self.stage_extensions += 1
                self.epochs_in_stage = 0
                self.samples_in_stage = 0
                return ProgressionStatus.EXTEND

            # Maximum extensions reached -> fail curriculum
            self.is_completed = True
            return ProgressionStatus.FAILED

        return ProgressionStatus.CONTINUE

    # ------------------------------------------------------------------
    def _advance_stage(self) -> None:
        self.current_stage_idx += 1
        if self.current_stage_idx >= self.config.get_num_stages():
            self.is_completed = True
            return

        self.current_stage = self.config.get_stage(self.current_stage_idx)
        self.epochs_in_stage = 0
        self.stage_extensions = 0
        self.samples_in_stage = 0
        self.current_metrics = {}

    # ------------------------------------------------------------------
    def get_current_config(self) -> Optional[Dict[str, Any]]:
        if self.is_completed:
            return None
        stage_cfg = dict(self.current_stage.raw_config)
        stage_cfg.update({
            "stage_name": self.current_stage.name,
            "stage_idx": self.current_stage_idx,
        })
        return stage_cfg

    # ------------------------------------------------------------------
    def get_progress_summary(self) -> Dict[str, Any]:
        return {
            "current_stage": None if self.is_completed else self.current_stage.name,
            "current_stage_idx": self.current_stage_idx,
            "total_stages": self.config.get_num_stages(),
            "total_epochs": sum(h["epoch"] for h in self.history if h["stage_idx"] == self.current_stage_idx),
            "is_completed": self.is_completed,
            "current_metrics": dict(self.current_metrics),
            "samples_in_stage": self.samples_in_stage,
            "stage_extensions": self.stage_extensions,
        }

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: Path) -> None:
        data = {
            "current_stage_idx": self.current_stage_idx,
            "epochs_in_stage": self.epochs_in_stage,
            "stage_extensions": self.stage_extensions,
            "samples_in_stage": self.samples_in_stage,
            "current_metrics": self.current_metrics,
            "baseline_metrics": self.baseline_metrics,
            "history": self.history,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: Path) -> None:
        data = torch.load(path, map_location="cpu")
        self.current_stage_idx = data.get("current_stage_idx", 0)
        self.current_stage = self.config.get_stage(min(self.current_stage_idx, self.config.get_num_stages() - 1))
        self.epochs_in_stage = data.get("epochs_in_stage", 0)
        self.stage_extensions = data.get("stage_extensions", 0)
        self.samples_in_stage = data.get("samples_in_stage", 0)
        self.current_metrics = data.get("current_metrics", {})
        self.baseline_metrics = data.get("baseline_metrics", {})
        self.history = data.get("history", [])


__all__ = [
    "CurriculumConfig",
    "CurriculumManager",
    "ProgressionStatus",
    "DifficultyLevel",
]
