"""
Experiment #2: Model Safety & Robustness Baseline

This module contains validation scripts to ensure SAFE model safety:
- Parameter freeze audit
- Forward pass equivalence testing
- Model architecture validation

Part of the comprehensive SAFE training validation framework.
"""

__version__ = "1.0.0"

from .parameter_audit import ParameterAuditor
from .equivalence_test import EquivalenceValidator
from .run_safety_checks import SafetyValidator

__all__ = [
    "ParameterAuditor",
    "EquivalenceValidator", 
    "SafetyValidator"
]