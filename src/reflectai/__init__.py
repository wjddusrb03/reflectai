"""ReflectAI — Neuro-symbolic reasoning with abductive reflection.

First open-source implementation of ABL-Refl (AAAI 2026 Outstanding Paper):
"Efficient Rectification of Neuro-Symbolic Reasoning Inconsistencies
 by Abductive Reflection"

Architecture:
    Input → Body (CNN/MLP) → Output Head (predictions)
                           → Reflection Head (error flags)  ← core innovation
    Flagged positions → Abductive Solver → Corrected labels
"""

__version__ = "1.0.0"
__author__ = "wjddusrb03"

from .models import (
    Constraint,
    Correction,
    KnowledgeBase,
    Prediction,
    Reflection,
    SolveResult,
    SolverType,
    TaskType,
    TrainConfig,
    TrainStats,
)

__all__ = [
    # Models
    "Constraint",
    "Correction",
    "KnowledgeBase",
    "Prediction",
    "Reflection",
    "SolveResult",
    "SolverType",
    "TaskType",
    "TrainConfig",
    "TrainStats",
    # Version
    "__version__",
    "__author__",
]
