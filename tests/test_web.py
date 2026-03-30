"""Tests for reflectai.web — web UI logic functions."""

import pytest
import numpy as np


class TestSudokuSolveFunction:
    def test_basic_call(self):
        from reflectai.web import solve_sudoku
        grids_html, result_text = solve_sudoku(
            difficulty="easy", error_rate=0.15, threshold=0.5,
            solver_type="backtrack", seed=42,
        )
        assert "<table>" in grids_html
        assert "Prediction Accuracy" in result_text
        assert "Correction Accuracy" in result_text

    def test_different_difficulties(self):
        from reflectai.web import solve_sudoku
        for diff in ["easy", "medium", "hard"]:
            grids, text = solve_sudoku(diff, 0.1, 0.5, "backtrack", 42)
            assert "<table>" in grids

    def test_high_error_rate(self):
        from reflectai.web import solve_sudoku
        grids, text = solve_sudoku("easy", 0.4, 0.5, "backtrack", 42)
        assert "Improvement" in text

    def test_z3_solver(self):
        try:
            import z3
        except ImportError:
            pytest.skip("z3 not installed")
        from reflectai.web import solve_sudoku
        grids, text = solve_sudoku("easy", 0.15, 0.5, "z3", 42)
        assert "<table>" in grids


class TestAdditionSolveFunction:
    def test_basic_call(self):
        from reflectai.web import solve_addition
        visual, text = solve_addition(3, 4, 0.5, "backtrack", 10)
        assert "Digit A" in visual
        assert "Digit B" in visual
        assert "Correction Correct" in text

    def test_no_error(self):
        from reflectai.web import solve_addition
        visual, text = solve_addition(2, 3, 0.0, "backtrack", 42)
        assert "Digit A" in visual

    def test_various_digits(self):
        from reflectai.web import solve_addition
        for a in range(0, 10, 3):
            for b in range(0, 10 - a, 3):
                visual, text = solve_addition(a, b, 0.3, "backtrack", 42)
                assert text is not None


class TestBenchmarkFunction:
    def test_basic_call(self):
        from reflectai.web import run_benchmark
        result = run_benchmark(5, "easy", 0.15, "backtrack")
        assert "Benchmark Results" in result
        assert "Prediction Accuracy" in result
        assert "Correction Accuracy" in result

    def test_more_puzzles(self):
        from reflectai.web import run_benchmark
        result = run_benchmark(10, "medium", 0.2, "backtrack")
        assert "Improvement" in result


class TestGridHTML:
    def test_grid_html(self):
        from reflectai.web import _grid_html
        values = np.arange(1, 82) % 9 + 1
        html = _grid_html(values, title="Test")
        assert "<table>" in html
        assert "Test" in html
        assert "<td>" in html

    def test_grid_with_highlights(self):
        from reflectai.web import _grid_html
        values = np.ones(81, dtype=int)
        highlights = {0: "cell-correct", 1: "cell-error"}
        html = _grid_html(values, highlights, "Highlighted")
        assert "cell-correct" in html
        assert "cell-error" in html

    def test_grid_empty_cells(self):
        from reflectai.web import _grid_html
        values = np.zeros(81, dtype=int)
        html = _grid_html(values, title="Empty")
        assert "&middot;" in html


class TestCreateApp:
    def test_app_creation(self):
        """Test that the Gradio app can be created (if gradio installed)."""
        try:
            import gradio
        except ImportError:
            pytest.skip("gradio not installed")

        from reflectai.web import create_app
        app = create_app()
        assert app is not None
