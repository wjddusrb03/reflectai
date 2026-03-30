"""Sudoku task for ReflectAI.

Provides utilities for generating, validating, and solving Sudoku puzzles
using the ReflectAI reflection-based pipeline.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from ..knowledge import build_sudoku_kb
from ..models import KnowledgeBase, Prediction, Reflection


def generate_sudoku(difficulty: str = "easy",
                    seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Sudoku puzzle with solution.

    Args:
        difficulty: "easy" (40 given), "medium" (30 given), "hard" (25 given)
        seed: Random seed for reproducibility

    Returns:
        (puzzle, solution) — both [81] arrays with values 0-9 (0=empty)
    """
    rng = random.Random(seed)

    # Generate a valid complete Sudoku using backtracking
    grid = np.zeros(81, dtype=int)
    _fill_grid(grid, rng)
    solution = grid.copy()

    # Remove cells based on difficulty
    given_counts = {"easy": 40, "medium": 30, "hard": 25}
    num_given = given_counts.get(difficulty, 30)
    num_remove = 81 - num_given

    indices = list(range(81))
    rng.shuffle(indices)
    puzzle = solution.copy()
    for idx in indices[:num_remove]:
        puzzle[idx] = 0

    return puzzle, solution


def _fill_grid(grid: np.ndarray, rng: random.Random,
               pos: int = 0) -> bool:
    """Fill Sudoku grid using randomized backtracking."""
    if pos == 81:
        return True

    row, col = divmod(pos, 9)
    values = list(range(1, 10))
    rng.shuffle(values)

    for val in values:
        if _is_valid_placement(grid, row, col, val):
            grid[pos] = val
            if _fill_grid(grid, rng, pos + 1):
                return True
            grid[pos] = 0

    return False


def _is_valid_placement(grid: np.ndarray, row: int, col: int,
                        val: int) -> bool:
    """Check if placing val at (row, col) is valid."""
    # Row check
    row_vals = grid[row * 9:(row + 1) * 9]
    if val in row_vals:
        return False

    # Column check
    col_vals = grid[col::9]
    if val in col_vals:
        return False

    # Box check
    box_r, box_c = (row // 3) * 3, (col // 3) * 3
    for r in range(box_r, box_r + 3):
        for c in range(box_c, box_c + 3):
            if grid[r * 9 + c] == val:
                return False

    return True


def simulate_noisy_predictions(solution: np.ndarray,
                               error_rate: float = 0.15,
                               seed: int | None = None) -> Prediction:
    """Simulate neural network predictions with controlled error rate.

    Useful for testing the reflection + reasoning pipeline without
    training a real neural network.

    Args:
        solution: [81] ground truth Sudoku solution
        error_rate: Fraction of positions to corrupt
        seed: Random seed

    Returns:
        Prediction with noisy labels and synthetic probabilities
    """
    rng = np.random.RandomState(seed)
    N = len(solution)

    labels = solution.copy()
    num_errors = int(N * error_rate)
    error_indices = rng.choice(N, size=num_errors, replace=False)

    for idx in error_indices:
        # Pick a wrong value
        wrong_vals = [v for v in range(1, 10) if v != solution[idx]]
        labels[idx] = rng.choice(wrong_vals)

    # Synthetic probabilities: high confidence for correct, lower for errors
    probs = np.full((N, 10), 0.01)
    for i in range(N):
        if i in error_indices:
            # Spread probability more evenly for errors
            probs[i] = rng.dirichlet(np.ones(10) * 2)
        else:
            # High confidence for correct predictions
            probs[i, labels[i]] = 0.85
            remaining = (1 - 0.85) / 9
            probs[i] = remaining
            probs[i, labels[i]] = 0.85

    confidence = probs.max(axis=1)

    return Prediction(labels=labels, probabilities=probs, confidence=confidence)


def evaluate_sudoku_correction(prediction: Prediction,
                               solution: np.ndarray,
                               corrected_labels: np.ndarray) -> dict:
    """Evaluate how well the correction recovered the true solution.

    Args:
        prediction: Original (possibly noisy) predictions
        solution: Ground truth solution
        corrected_labels: Labels after reflection + reasoning

    Returns:
        Dict with accuracy metrics
    """
    N = len(solution)
    mask = solution > 0  # Only evaluate non-empty cells

    pred_correct = (prediction.labels[mask] == solution[mask]).sum()
    corr_correct = (corrected_labels[mask] == solution[mask]).sum()
    total = mask.sum()

    # Which errors were fixed?
    pred_errors = prediction.labels != solution
    fixed = pred_errors & (corrected_labels == solution)
    introduced = (~pred_errors) & (corrected_labels != solution)

    return {
        "prediction_accuracy": float(pred_correct / total),
        "correction_accuracy": float(corr_correct / total),
        "improvement": float((corr_correct - pred_correct) / total),
        "errors_fixed": int(fixed.sum()),
        "errors_introduced": int(introduced.sum()),
        "total_original_errors": int(pred_errors.sum()),
    }
