"""Tests for reflectai.knowledge — knowledge base builders."""

import numpy as np
import pytest

from reflectai.knowledge import (
    build_addition_kb,
    build_equation_kb,
    build_kb,
    build_nqueens_kb,
    build_sudoku_4x4_kb,
    build_sudoku_kb,
)


class TestSudokuKB:
    def test_9x9_constraints(self):
        kb = build_sudoku_kb(size=9)
        assert len(kb.constraints) == 27  # 9 rows + 9 cols + 9 boxes
        assert kb.num_classes == 10
        assert kb.num_positions == 81

    def test_4x4_constraints(self):
        kb = build_sudoku_4x4_kb()
        assert len(kb.constraints) == 12  # 4 rows + 4 cols + 4 boxes
        assert kb.num_classes == 5
        assert kb.num_positions == 16

    def test_valid_sudoku(self):
        kb = build_sudoku_kb()
        # Valid Sudoku solution (first row)
        solution = np.array([
            5, 3, 4, 6, 7, 8, 9, 1, 2,
            6, 7, 2, 1, 9, 5, 3, 4, 8,
            1, 9, 8, 3, 4, 2, 5, 6, 7,
            8, 5, 9, 7, 6, 1, 4, 2, 3,
            4, 2, 6, 8, 5, 3, 7, 9, 1,
            7, 1, 3, 9, 2, 4, 8, 5, 6,
            9, 6, 1, 5, 3, 7, 2, 8, 4,
            2, 8, 7, 4, 1, 9, 6, 3, 5,
            3, 4, 5, 2, 8, 6, 1, 7, 9,
        ])
        ok, violations = kb.check_consistency(solution)
        assert ok
        assert len(violations) == 0

    def test_invalid_sudoku_row(self):
        kb = build_sudoku_kb()
        solution = np.ones(81, dtype=int)  # All 1s — violates everything
        ok, violations = kb.check_consistency(solution)
        assert not ok
        assert len(violations) > 0

    def test_row_constraint_indices(self):
        kb = build_sudoku_kb()
        row_0 = kb.constraints[0]
        assert row_0.name == "row_0"
        assert row_0.indices == list(range(9))

    def test_col_constraint_indices(self):
        kb = build_sudoku_kb()
        col_0 = kb.constraints[9]  # After 9 row constraints
        assert col_0.name == "col_0"
        assert col_0.indices == list(range(0, 81, 9))

    def test_box_constraint_indices(self):
        kb = build_sudoku_kb()
        box_0_0 = kb.constraints[18]  # After rows and cols
        assert box_0_0.name == "box_0_0"
        expected = [0, 1, 2, 9, 10, 11, 18, 19, 20]
        assert box_0_0.indices == expected

    def test_invalid_size(self):
        with pytest.raises(AssertionError):
            build_sudoku_kb(size=7)


class TestAdditionKB:
    def test_with_target(self):
        kb = build_addition_kb(num_digits=2, target_sum=7)
        assert len(kb.constraints) == 1
        assert kb.num_classes == 10
        assert kb.num_positions == 2

    def test_without_target(self):
        kb = build_addition_kb(num_digits=3)
        assert len(kb.constraints) == 0
        assert kb.num_positions == 3

    def test_sum_check(self):
        kb = build_addition_kb(num_digits=2, target_sum=7)
        ok, _ = kb.check_consistency(np.array([3, 4]))
        assert ok
        ok, _ = kb.check_consistency(np.array([3, 5]))
        assert not ok


class TestEquationKB:
    def test_basic(self):
        kb = build_equation_kb(num_positions=5)
        assert len(kb.constraints) == 1
        assert kb.num_positions == 5


class TestNQueensKB:
    def test_8_queens(self):
        kb = build_nqueens_kb(n=8)
        assert len(kb.constraints) == 1
        assert kb.num_classes == 8
        assert kb.num_positions == 8

    def test_valid_placement(self):
        kb = build_nqueens_kb(n=4)
        # Valid 4-queens: columns [1, 3, 0, 2]
        ok, _ = kb.check_consistency(np.array([1, 3, 0, 2]))
        assert ok

    def test_invalid_placement(self):
        kb = build_nqueens_kb(n=4)
        # Same column twice
        ok, _ = kb.check_consistency(np.array([1, 1, 0, 2]))
        assert not ok


class TestBuildKB:
    def test_sudoku(self):
        kb = build_kb("sudoku")
        assert len(kb.constraints) == 27

    def test_sudoku_4x4(self):
        kb = build_kb("sudoku_4x4")
        assert len(kb.constraints) == 12

    def test_addition(self):
        kb = build_kb("addition", num_digits=3, target_sum=10)
        assert len(kb.constraints) == 1

    def test_equation(self):
        kb = build_kb("equation")
        assert len(kb.constraints) >= 1

    def test_nqueens(self):
        kb = build_kb("nqueens", n=6)
        assert kb.num_positions == 6

    def test_unknown_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            build_kb("chess")
