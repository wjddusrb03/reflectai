"""Tests for reflectai.reasoner — abductive reasoning solvers."""

import numpy as np
import pytest

from reflectai.models import (
    Correction, KnowledgeBase, Prediction, Reflection,
)
from reflectai.reasoner import BacktrackSolver, create_solver


class TestBacktrackSolver:
    def setup_method(self):
        self.solver = BacktrackSolver()

    def _make_prediction(self, labels, num_classes=10):
        N = len(labels)
        probs = np.full((N, num_classes), 0.01)
        for i, l in enumerate(labels):
            probs[i, l] = 0.9
        probs /= probs.sum(axis=1, keepdims=True)
        return Prediction(
            labels=np.array(labels),
            probabilities=probs,
            confidence=probs.max(axis=1),
        )

    def test_no_flags_returns_original(self):
        pred = self._make_prediction([1, 2, 3])
        ref = Reflection(np.zeros(3, dtype=int), np.zeros(3))
        kb = KnowledgeBase()
        kb.add_constraint("distinct", [0, 1, 2], "all_distinct")

        corr = self.solver.solve(pred, ref, kb)
        np.testing.assert_array_equal(corr.labels, [1, 2, 3])
        assert corr.consistent
        assert corr.num_corrections == 0

    def test_fixes_duplicate(self):
        """If two positions have same value, solver should fix flagged one."""
        pred = self._make_prediction([1, 1, 3], num_classes=4)
        # Flag position 1
        ref = Reflection(
            np.array([0, 1, 0]),
            np.array([0.1, 0.8, 0.1]),
        )
        kb = KnowledgeBase(num_classes=4)
        kb.add_constraint("distinct", [0, 1, 2], "all_distinct")

        corr = self.solver.solve(pred, ref, kb, timeout_ms=5000)
        # Position 1 should be changed to something != 1 and != 3
        assert corr.labels[0] == 1  # Unflagged, unchanged
        assert corr.labels[1] != 1  # Was flagged, should change
        assert corr.labels[2] == 3  # Unflagged, unchanged
        assert corr.consistent

    def test_sum_constraint(self):
        """Solver should fix values to satisfy sum constraint."""
        pred = self._make_prediction([3, 4], num_classes=10)
        # Flag position 1 — need sum to be 5
        ref = Reflection(np.array([0, 1]), np.array([0.1, 0.8]))
        kb = KnowledgeBase(num_classes=10)
        kb.add_constraint("sum", [0, 1], "sum_equals", target=5)

        corr = self.solver.solve(pred, ref, kb, timeout_ms=5000)
        assert corr.labels[0] == 3  # Unflagged
        assert corr.labels[0] + corr.labels[1] == 5
        assert corr.consistent

    def test_already_consistent(self):
        pred = self._make_prediction([1, 2, 3])
        ref = Reflection(np.array([0, 1, 0]), np.array([0.1, 0.8, 0.1]))
        kb = KnowledgeBase()
        kb.add_constraint("distinct", [0, 1, 2], "all_distinct")

        corr = self.solver.solve(pred, ref, kb)
        # Already consistent, might keep same values
        assert corr.consistent

    def test_timeout(self):
        """Very short timeout should still return a result."""
        pred = self._make_prediction([1, 1, 1, 1])
        ref = Reflection(np.ones(4, dtype=int), np.ones(4))
        kb = KnowledgeBase(num_classes=5)
        kb.add_constraint("distinct", [0, 1, 2, 3], "all_distinct")

        corr = self.solver.solve(pred, ref, kb, timeout_ms=0.001)
        # Should return something (possibly inconsistent due to timeout)
        assert len(corr.labels) == 4

    def test_solver_time_recorded(self):
        pred = self._make_prediction([1, 2, 3])
        ref = Reflection(np.zeros(3, dtype=int), np.zeros(3))
        kb = KnowledgeBase()

        corr = self.solver.solve(pred, ref, kb)
        assert corr.solver_time_ms >= 0

    def test_multiple_constraints(self):
        """Test with overlapping constraints."""
        pred = self._make_prediction([1, 1, 2], num_classes=4)
        ref = Reflection(np.array([0, 1, 0]), np.array([0.1, 0.8, 0.1]))
        kb = KnowledgeBase(num_classes=4)
        kb.add_constraint("c1", [0, 1], "all_distinct")
        kb.add_constraint("c2", [1, 2], "all_distinct")

        corr = self.solver.solve(pred, ref, kb, timeout_ms=5000)
        assert corr.labels[1] != corr.labels[0]
        assert corr.labels[1] != corr.labels[2]
        assert corr.consistent


class TestZ3Solver:
    """Tests for Z3 solver (skipped if z3 not installed)."""

    @pytest.fixture(autouse=True)
    def check_z3(self):
        try:
            import z3
        except ImportError:
            pytest.skip("z3-solver not installed")

    def _make_prediction(self, labels, num_classes=10):
        N = len(labels)
        probs = np.full((N, num_classes), 0.01)
        for i, l in enumerate(labels):
            probs[i, l] = 0.9
        probs /= probs.sum(axis=1, keepdims=True)
        return Prediction(
            labels=np.array(labels),
            probabilities=probs,
            confidence=probs.max(axis=1),
        )

    def test_z3_distinct(self):
        from reflectai.reasoner import Z3Solver
        solver = Z3Solver()

        pred = self._make_prediction([1, 1, 3], num_classes=4)
        ref = Reflection(np.array([0, 1, 0]), np.array([0.1, 0.8, 0.1]))
        kb = KnowledgeBase(num_classes=4)
        kb.add_constraint("distinct", [0, 1, 2], "all_distinct")

        corr = solver.solve(pred, ref, kb, timeout_ms=5000)
        assert corr.labels[0] == 1
        assert corr.labels[2] == 3
        assert corr.consistent

    def test_z3_sum(self):
        from reflectai.reasoner import Z3Solver
        solver = Z3Solver()

        pred = self._make_prediction([3, 4], num_classes=10)
        ref = Reflection(np.array([0, 1]), np.array([0.1, 0.8]))
        kb = KnowledgeBase(num_classes=10)
        kb.add_constraint("sum", [0, 1], "sum_equals", target=5)

        corr = solver.solve(pred, ref, kb, timeout_ms=5000)
        assert corr.labels[0] + corr.labels[1] == 5


class TestCreateSolver:
    def test_backtrack(self):
        solver = create_solver("backtrack")
        assert isinstance(solver, BacktrackSolver)

    def test_unknown(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            create_solver("nonexistent")

    def test_backtrack_with_params(self):
        solver = create_solver("backtrack", max_iterations=50000)
        assert solver.max_iterations == 50000
