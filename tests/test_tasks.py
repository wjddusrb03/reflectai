"""Tests for reflectai.tasks — built-in puzzle tasks."""

import numpy as np
import pytest

from reflectai.tasks.sudoku import (
    evaluate_sudoku_correction,
    generate_sudoku,
    simulate_noisy_predictions,
)
from reflectai.tasks.mnist_add import (
    evaluate_addition,
    generate_addition_samples,
    simulate_addition_predictions,
)
from reflectai.tasks.equation import (
    generate_equations,
    simulate_equation_predictions,
)


# ══════════════════════════════════════════════════════════════════
# Sudoku
# ══════════════════════════════════════════════════════════════════

class TestSudokuGeneration:
    def test_generate_easy(self):
        puzzle, solution = generate_sudoku("easy", seed=42)
        assert puzzle.shape == (81,)
        assert solution.shape == (81,)

    def test_solution_valid(self):
        _, solution = generate_sudoku("easy", seed=42)
        # All values 1-9
        assert np.all(solution >= 1) and np.all(solution <= 9)
        # Check rows
        for r in range(9):
            row = solution[r * 9:(r + 1) * 9]
            assert len(set(row)) == 9

    def test_puzzle_has_empty_cells(self):
        puzzle, _ = generate_sudoku("easy", seed=42)
        assert (puzzle == 0).sum() > 0

    def test_difficulty_levels(self):
        for diff in ["easy", "medium", "hard"]:
            puzzle, solution = generate_sudoku(diff, seed=42)
            assert puzzle.shape == (81,)

    def test_easy_more_given_than_hard(self):
        easy_puzzle, _ = generate_sudoku("easy", seed=42)
        hard_puzzle, _ = generate_sudoku("hard", seed=42)
        assert (easy_puzzle > 0).sum() > (hard_puzzle > 0).sum()

    def test_reproducible(self):
        p1, s1 = generate_sudoku("easy", seed=42)
        p2, s2 = generate_sudoku("easy", seed=42)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds(self):
        _, s1 = generate_sudoku("easy", seed=1)
        _, s2 = generate_sudoku("easy", seed=2)
        assert not np.array_equal(s1, s2)


class TestSudokuSimulation:
    def test_noisy_predictions(self):
        _, solution = generate_sudoku("easy", seed=42)
        pred = simulate_noisy_predictions(solution, error_rate=0.15, seed=42)
        assert pred.labels.shape == (81,)
        assert pred.probabilities.shape == (81, 10)

    def test_error_rate(self):
        _, solution = generate_sudoku("easy", seed=42)
        pred = simulate_noisy_predictions(solution, error_rate=0.15, seed=42)
        actual_errors = (pred.labels != solution).mean()
        # Should be roughly close to 0.15 (but stochastic)
        assert 0.05 < actual_errors < 0.30

    def test_zero_error_rate(self):
        _, solution = generate_sudoku("easy", seed=42)
        pred = simulate_noisy_predictions(solution, error_rate=0.0, seed=42)
        np.testing.assert_array_equal(pred.labels, solution)

    def test_probabilities_valid(self):
        _, solution = generate_sudoku("easy", seed=42)
        pred = simulate_noisy_predictions(solution, error_rate=0.15, seed=42)
        # Probabilities should sum to ~1
        sums = pred.probabilities.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=0.05)


class TestSudokuEvaluation:
    def test_perfect_correction(self):
        _, solution = generate_sudoku("easy", seed=42)
        pred = simulate_noisy_predictions(solution, error_rate=0.2, seed=42)
        metrics = evaluate_sudoku_correction(pred, solution, solution)
        assert metrics["correction_accuracy"] == 1.0
        assert metrics["errors_introduced"] == 0

    def test_no_improvement(self):
        _, solution = generate_sudoku("easy", seed=42)
        pred = simulate_noisy_predictions(solution, error_rate=0.2, seed=42)
        metrics = evaluate_sudoku_correction(pred, solution, pred.labels)
        assert metrics["improvement"] == 0.0

    def test_metrics_keys(self):
        _, solution = generate_sudoku("easy", seed=42)
        pred = simulate_noisy_predictions(solution, error_rate=0.1, seed=42)
        metrics = evaluate_sudoku_correction(pred, solution, solution)
        expected_keys = [
            "prediction_accuracy", "correction_accuracy", "improvement",
            "errors_fixed", "errors_introduced", "total_original_errors",
        ]
        for key in expected_keys:
            assert key in metrics


# ══════════════════════════════════════════════════════════════════
# MNIST Addition
# ══════════════════════════════════════════════════════════════════

class TestMNISTAddition:
    def test_generate_samples(self):
        samples = generate_addition_samples(num_samples=10, seed=42)
        assert len(samples) == 10
        for s in samples:
            assert "digits" in s
            assert "target_sum" in s
            assert s["target_sum"] == s["digits"].sum()

    def test_simulate_predictions(self):
        digits = np.array([3, 5])
        pred = simulate_addition_predictions(digits, error_rate=0.0, seed=42)
        np.testing.assert_array_equal(pred.labels, digits)

    def test_predictions_with_errors(self):
        digits = np.array([3, 5])
        pred = simulate_addition_predictions(digits, error_rate=1.0, seed=42)
        # With 100% error rate, at least one should differ
        assert not np.array_equal(pred.labels, digits)

    def test_evaluate_addition(self):
        digits = np.array([3, 5])
        pred = simulate_addition_predictions(digits, error_rate=0.0, seed=42)
        metrics = evaluate_addition(pred, digits, digits, target_sum=8)
        assert metrics["prediction_correct"]
        assert metrics["correction_correct"]
        assert metrics["digit_accuracy_before"] == 1.0


# ══════════════════════════════════════════════════════════════════
# Equation
# ══════════════════════════════════════════════════════════════════

class TestEquation:
    def test_generate(self):
        eqs = generate_equations(num_samples=20, seed=42)
        assert len(eqs) == 20
        for eq in eqs:
            assert eq["a"] + eq["b"] == eq["c"]
            assert 0 <= eq["c"] <= 9

    def test_simulate_predictions(self):
        pred = simulate_equation_predictions(3, 4, 7, error_rate=0.0, seed=42)
        np.testing.assert_array_equal(pred.labels, [3, 4, 7])

    def test_simulate_with_errors(self):
        pred = simulate_equation_predictions(3, 4, 7, error_rate=1.0, seed=42)
        assert not np.array_equal(pred.labels, [3, 4, 7])

    def test_probabilities_shape(self):
        pred = simulate_equation_predictions(1, 2, 3, seed=42)
        assert pred.probabilities.shape == (3, 10)
