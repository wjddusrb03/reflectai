"""Core data models for ReflectAI.

Implements the data structures from ABL-Refl (AAAI 2026 Outstanding Paper):
- Prediction: neural network output with confidence
- Reflection: binary error flags from reflection head
- Correction: abductive reasoning output
- Task configuration and results
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# ══════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════

class TaskType(Enum):
    """Types of built-in tasks."""
    SUDOKU = "sudoku"
    MNIST_ADD = "mnist_add"
    EQUATION = "equation"
    CUSTOM = "custom"


class SolverType(Enum):
    """Constraint solver backends."""
    Z3 = "z3"
    SAT = "sat"
    BACKTRACK = "backtrack"


# ══════════════════════════════════════════════════════════════════
# Prediction & Reflection
# ══════════════════════════════════════════════════════════════════

@dataclass
class Prediction:
    """Output from the neural perception module.

    Attributes:
        labels: Predicted class labels [N]
        probabilities: Class probability distributions [N, C]
        confidence: Confidence score per prediction [N]
    """
    labels: np.ndarray                  # [N] int predicted labels
    probabilities: np.ndarray           # [N, C] float softmax outputs
    confidence: np.ndarray              # [N] float max probability per item

    @property
    def num_items(self) -> int:
        return len(self.labels)

    @property
    def num_classes(self) -> int:
        return self.probabilities.shape[1] if self.probabilities.ndim > 1 else 0

    def top_k(self, k: int = 3) -> np.ndarray:
        """Get top-k predictions for each item. Returns [N, k] indices."""
        return np.argsort(-self.probabilities, axis=1)[:, :k]


@dataclass
class Reflection:
    """Output from the reflection head — the core innovation.

    The reflection vector is a binary mask indicating which predictions
    are likely inconsistent with domain knowledge.

    Attributes:
        flags: Binary error flags [N] — 1=suspect, 0=trusted
        scores: Continuous reflection scores [N] (pre-threshold)
        threshold: Threshold used for binarization
    """
    flags: np.ndarray          # [N] binary: 1=flagged as potential error
    scores: np.ndarray         # [N] float: raw reflection scores (0.0-1.0)
    threshold: float = 0.5

    @property
    def num_flagged(self) -> int:
        return int(np.sum(self.flags))

    @property
    def flagged_indices(self) -> np.ndarray:
        return np.where(self.flags == 1)[0]

    @property
    def trusted_indices(self) -> np.ndarray:
        return np.where(self.flags == 0)[0]

    @property
    def flag_rate(self) -> float:
        if len(self.flags) == 0:
            return 0.0
        return float(np.mean(self.flags))


@dataclass
class Correction:
    """Output from the abductive reasoner.

    Contains the corrected labels and which positions were changed.

    Attributes:
        labels: Corrected labels [N]
        changed_mask: Which positions were changed [N] boolean
        consistent: Whether the final result satisfies all constraints
        solver_time_ms: Time taken by the constraint solver
    """
    labels: np.ndarray              # [N] int corrected labels
    changed_mask: np.ndarray        # [N] bool: True where correction differs from prediction
    consistent: bool = True
    solver_time_ms: float = 0.0

    @property
    def num_corrections(self) -> int:
        return int(np.sum(self.changed_mask))

    @property
    def correction_rate(self) -> float:
        if len(self.changed_mask) == 0:
            return 0.0
        return float(np.mean(self.changed_mask))


# ══════════════════════════════════════════════════════════════════
# Constraint / Knowledge Base
# ══════════════════════════════════════════════════════════════════

@dataclass
class Constraint:
    """A single logical constraint in the knowledge base.

    Example for Sudoku: "All values in row 0 must be distinct"
    -> name="row_0_distinct", indices=[0,1,2,...,8], rule="all_distinct"
    """
    name: str
    indices: list[int]          # Which positions this constraint covers
    rule: str                   # Rule type: "all_distinct", "sum_equals", "custom"
    params: dict = field(default_factory=dict)  # Rule-specific parameters

    def __str__(self) -> str:
        return f"{self.name}: {self.rule}({self.indices})"


@dataclass
class KnowledgeBase:
    """Collection of logical constraints defining domain knowledge.

    For Sudoku: 27 constraints (9 rows + 9 cols + 9 boxes, all_distinct)
    For MNIST Addition: 1 constraint (digit1 + digit2 == sum)
    """
    constraints: list[Constraint] = field(default_factory=list)
    num_classes: int = 10           # Number of possible values per position
    num_positions: int = 0          # Total number of positions

    def add_constraint(self, name: str, indices: list[int],
                       rule: str, **params) -> None:
        self.constraints.append(Constraint(name, indices, rule, params))

    def check_consistency(self, labels: np.ndarray) -> tuple[bool, list[str]]:
        """Check if labels satisfy all constraints.

        Returns (is_consistent, list_of_violated_constraint_names).
        """
        violations = []
        for c in self.constraints:
            values = labels[c.indices]
            if not self._check_rule(values, c.rule, c.params):
                violations.append(c.name)
        return len(violations) == 0, violations

    def _check_rule(self, values: np.ndarray, rule: str,
                    params: dict) -> bool:
        """Check a single rule against given values."""
        if rule == "all_distinct":
            # All values must be unique (and optionally within range)
            v = values[values > 0]  # Ignore zeros (unfilled)
            return len(v) == len(set(v))
        elif rule == "sum_equals":
            target = params.get("target", 0)
            return int(np.sum(values)) == target
        elif rule == "in_range":
            lo = params.get("min", 1)
            hi = params.get("max", 9)
            return np.all((values >= lo) & (values <= hi))
        return True


# ══════════════════════════════════════════════════════════════════
# Pipeline Result
# ══════════════════════════════════════════════════════════════════

@dataclass
class SolveResult:
    """Complete result from the ReflectAI pipeline."""
    prediction: Prediction
    reflection: Reflection
    correction: Correction
    knowledge_base: KnowledgeBase

    # Timing
    perception_time_ms: float = 0.0
    reflection_time_ms: float = 0.0
    reasoning_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def final_labels(self) -> np.ndarray:
        return self.correction.labels

    @property
    def is_consistent(self) -> bool:
        consistent, _ = self.knowledge_base.check_consistency(self.final_labels)
        return consistent

    @property
    def summary(self) -> dict:
        return {
            "num_items": self.prediction.num_items,
            "num_flagged": self.reflection.num_flagged,
            "num_corrections": self.correction.num_corrections,
            "is_consistent": self.is_consistent,
            "total_time_ms": round(self.total_time_ms, 2),
        }


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """Configuration for training the ReflectAI model."""
    # Model
    hidden_dim: int = 128
    num_classes: int = 10
    reflection_threshold: float = 0.5

    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Loss weights
    lambda_consistency: float = 1.0
    lambda_reflection_size: float = 0.1
    reflection_target_rate: float = 0.2  # C=0.8 → target ~20% flagging

    # Solver
    solver: SolverType = SolverType.BACKTRACK
    solver_timeout_ms: float = 1000.0

    # System
    device: str = "cpu"
    seed: int = 42


@dataclass
class TrainStats:
    """Statistics from a training run."""
    epoch: int = 0
    train_loss: float = 0.0
    supervised_loss: float = 0.0
    consistency_loss: float = 0.0
    reflection_loss: float = 0.0
    accuracy: float = 0.0
    reflection_rate: float = 0.0
    consistency_rate: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 4),
            "accuracy": round(self.accuracy, 4),
            "reflection_rate": round(self.reflection_rate, 4),
            "consistency_rate": round(self.consistency_rate, 4),
        }
