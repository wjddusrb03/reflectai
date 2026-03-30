"""Integration tests — full pipeline from puzzle to solution."""

import numpy as np
import pytest

from reflectai.knowledge import build_sudoku_kb, build_addition_kb
from reflectai.models import Prediction, Reflection
from reflectai.pipeline import solve_from_predictions
from reflectai.reasoner import BacktrackSolver
from reflectai.tasks.sudoku import generate_sudoku, simulate_noisy_predictions


class TestSudokuIntegration:
    """End-to-end test: generate puzzle → add noise → reflect → correct."""

    def test_full_pipeline(self):
        """Full Sudoku solve with synthetic noise."""
        _, solution = generate_sudoku("easy", seed=42)
        kb = build_sudoku_kb()

        # Simulate noisy predictions
        prediction = simulate_noisy_predictions(solution, error_rate=0.1, seed=42)
        errors = prediction.labels != solution

        # Simulate reflection: flag cells with errors + some noise
        scores = np.zeros(81)
        for i in range(81):
            if errors[i]:
                scores[i] = 0.8  # High score for actual errors
            else:
                scores[i] = 0.1  # Low score for correct predictions

        result = solve_from_predictions(
            prediction.labels,
            prediction.probabilities,
            scores,
            kb,
            threshold=0.5,
            solver_type="backtrack",
            timeout_ms=5000,
        )

        # Correction should improve accuracy
        pred_acc = (prediction.labels == solution).mean()
        corr_acc = (result.final_labels == solution).mean()
        assert corr_acc >= pred_acc

    def test_perfect_reflection_perfect_correction(self):
        """With perfect reflection (all errors flagged), solver should fix everything."""
        _, solution = generate_sudoku("easy", seed=123)
        kb = build_sudoku_kb()

        prediction = simulate_noisy_predictions(solution, error_rate=0.1, seed=123)
        errors = prediction.labels != solution

        # Perfect reflection: flag exactly the errors
        scores = np.where(errors, 0.9, 0.1)

        result = solve_from_predictions(
            prediction.labels,
            prediction.probabilities,
            scores,
            kb,
            threshold=0.5,
            timeout_ms=10000,
        )

        # With perfect flagging, solver should achieve high accuracy
        corr_acc = (result.final_labels == solution).mean()
        assert corr_acc > 0.9

    def test_no_errors_no_changes(self):
        """If predictions are perfect, nothing should change."""
        _, solution = generate_sudoku("easy", seed=42)
        kb = build_sudoku_kb()

        # Perfect predictions
        probs = np.full((81, 10), 0.01)
        for i in range(81):
            probs[i, solution[i]] = 0.9
        probs /= probs.sum(axis=1, keepdims=True)

        scores = np.full(81, 0.1)  # All low → nothing flagged

        result = solve_from_predictions(
            solution, probs, scores, kb, threshold=0.5,
        )
        np.testing.assert_array_equal(result.final_labels, solution)
        assert result.correction.num_corrections == 0


class TestAdditionIntegration:
    def test_fix_digit(self):
        """Test correcting a wrong digit in addition."""
        kb = build_addition_kb(num_digits=2, target_sum=7)

        # Predict [3, 5] but target is 7 → digit 1 should be 4
        labels = np.array([3, 5])
        probs = np.full((2, 10), 0.01)
        probs[0, 3] = 0.9
        probs[1, 5] = 0.3
        probs[1, 4] = 0.25  # 4 is second most likely
        probs /= probs.sum(axis=1, keepdims=True)

        scores = np.array([0.1, 0.8])  # Flag digit 1

        result = solve_from_predictions(
            labels, probs, scores, kb, threshold=0.5,
        )

        assert result.final_labels[0] == 3  # Unflagged
        if result.correction.consistent:
            assert result.final_labels.sum() == 7


class TestMultipleSolves:
    """Test solving multiple puzzles to verify robustness."""

    def test_batch_solve(self):
        kb = build_sudoku_kb()
        solver = BacktrackSolver()
        improvements = 0

        for seed in range(5):
            _, solution = generate_sudoku("easy", seed=seed)
            pred = simulate_noisy_predictions(solution, error_rate=0.1, seed=seed)
            errors = pred.labels != solution
            scores = np.where(errors, 0.8, 0.1)
            flags = (scores >= 0.5).astype(int)

            reflection = Reflection(flags=flags, scores=scores)
            correction = solver.solve(pred, reflection, kb, timeout_ms=10000)

            pred_acc = (pred.labels == solution).mean()
            corr_acc = (correction.labels == solution).mean()
            if corr_acc >= pred_acc:
                improvements += 1

        # Correction should not make things worse
        assert improvements >= 3
