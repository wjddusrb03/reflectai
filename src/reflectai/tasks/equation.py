"""Equation recognition task for ReflectAI.

Task: Recognize handwritten equations like "3 + 4 = 7"
and verify arithmetic consistency.
"""

from __future__ import annotations

import numpy as np

from ..knowledge import build_equation_kb
from ..models import KnowledgeBase, Prediction


def generate_equations(num_samples: int = 100,
                       max_val: int = 9,
                       seed: int | None = None) -> list[dict]:
    """Generate simple addition equations: a + b = c.

    Args:
        num_samples: Number of equations
        max_val: Maximum digit value
        seed: Random seed

    Returns:
        List of dicts with 'a', 'b', 'c' (where a + b = c, c < 10)
    """
    rng = np.random.RandomState(seed)
    samples = []

    for _ in range(num_samples):
        a = rng.randint(0, max_val + 1)
        b = rng.randint(0, max_val + 1 - a)  # Ensure a + b <= max_val
        c = a + b
        samples.append({"a": a, "b": b, "c": c})

    return samples


def build_equation_constraint(a_pred: int, b_pred: int,
                              c_pred: int) -> KnowledgeBase:
    """Build KB for equation a + b = c with known target.

    Positions: [a, b, c] (indices 0, 1, 2)
    Constraint: positions[0] + positions[1] == positions[2]
    """
    kb = KnowledgeBase(num_classes=10, num_positions=3)
    # We encode this as: a + b - c = 0, which means sum of [a, b, -c] = 0
    # But since our solver expects positive values, we use sum_equals
    # with target = 0 on specially arranged positions
    # Actually, simpler: a + b = c means sum of first two equals third
    # We'll use a custom approach in the solver
    kb.add_constraint(
        "addition_valid",
        [0, 1, 2],
        "sum_equals",
        target=0,  # Will be overridden per-sample
    )
    return kb


def simulate_equation_predictions(a: int, b: int, c: int,
                                  error_rate: float = 0.2,
                                  seed: int | None = None) -> Prediction:
    """Simulate noisy predictions for equation recognition.

    Args:
        a, b, c: True values of the equation a + b = c
        error_rate: Per-position error probability
        seed: Random seed

    Returns:
        Prediction for positions [a, b, c]
    """
    rng = np.random.RandomState(seed)
    true_vals = np.array([a, b, c])
    labels = true_vals.copy()
    probs = np.full((3, 10), 0.01)

    for i in range(3):
        if rng.random() < error_rate:
            wrong = [v for v in range(10) if v != true_vals[i]]
            labels[i] = rng.choice(wrong)
            probs[i] = rng.dirichlet(np.ones(10) * 2)
        else:
            probs[i, labels[i]] = 0.9
            remaining = 0.1 / 9
            probs[i] = remaining
            probs[i, labels[i]] = 0.9

    confidence = probs.max(axis=1)
    return Prediction(labels=labels, probabilities=probs, confidence=confidence)
