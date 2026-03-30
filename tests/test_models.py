"""Tests for reflectai.models — core data structures."""

import numpy as np
import pytest

from reflectai.models import (
    Constraint, Correction, KnowledgeBase, Prediction, Reflection,
    SolveResult, SolverType, TaskType, TrainConfig, TrainStats,
)


# ══════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════

class TestEnums:
    def test_task_types(self):
        assert TaskType.SUDOKU.value == "sudoku"
        assert TaskType.MNIST_ADD.value == "mnist_add"
        assert TaskType.EQUATION.value == "equation"
        assert TaskType.CUSTOM.value == "custom"

    def test_solver_types(self):
        assert SolverType.Z3.value == "z3"
        assert SolverType.SAT.value == "sat"
        assert SolverType.BACKTRACK.value == "backtrack"


# ══════════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════════

class TestPrediction:
    def setup_method(self):
        self.labels = np.array([3, 7, 1])
        self.probs = np.random.dirichlet(np.ones(10), size=3).astype(np.float32)
        for i, l in enumerate(self.labels):
            self.probs[i, l] = 0.9
            self.probs[i] /= self.probs[i].sum()
        self.confidence = self.probs.max(axis=1)
        self.pred = Prediction(self.labels, self.probs, self.confidence)

    def test_num_items(self):
        assert self.pred.num_items == 3

    def test_num_classes(self):
        assert self.pred.num_classes == 10

    def test_top_k(self):
        top3 = self.pred.top_k(3)
        assert top3.shape == (3, 3)
        # Most likely class should be in top-3
        for i, label in enumerate(self.labels):
            assert label in top3[i]

    def test_top_k_default(self):
        top = self.pred.top_k()
        assert top.shape == (3, 3)

    def test_single_item(self):
        pred = Prediction(
            labels=np.array([5]),
            probabilities=np.random.dirichlet(np.ones(10), size=1),
            confidence=np.array([0.8]),
        )
        assert pred.num_items == 1

    def test_num_classes_1d(self):
        pred = Prediction(
            labels=np.array([1]),
            probabilities=np.array([0.9]),
            confidence=np.array([0.9]),
        )
        assert pred.num_classes == 0


# ══════════════════════════════════════════════════════════════════
# Reflection
# ══════════════════════════════════════════════════════════════════

class TestReflection:
    def setup_method(self):
        self.flags = np.array([0, 1, 0, 1, 0])
        self.scores = np.array([0.1, 0.8, 0.2, 0.7, 0.3])
        self.ref = Reflection(self.flags, self.scores, threshold=0.5)

    def test_num_flagged(self):
        assert self.ref.num_flagged == 2

    def test_flagged_indices(self):
        np.testing.assert_array_equal(self.ref.flagged_indices, [1, 3])

    def test_trusted_indices(self):
        np.testing.assert_array_equal(self.ref.trusted_indices, [0, 2, 4])

    def test_flag_rate(self):
        assert abs(self.ref.flag_rate - 0.4) < 1e-6

    def test_empty_flags(self):
        ref = Reflection(np.array([]), np.array([]))
        assert ref.flag_rate == 0.0
        assert ref.num_flagged == 0

    def test_all_flagged(self):
        ref = Reflection(np.ones(5, dtype=int), np.ones(5))
        assert ref.num_flagged == 5
        assert ref.flag_rate == 1.0

    def test_none_flagged(self):
        ref = Reflection(np.zeros(5, dtype=int), np.zeros(5))
        assert ref.num_flagged == 0
        assert ref.flag_rate == 0.0


# ══════════════════════════════════════════════════════════════════
# Correction
# ══════════════════════════════════════════════════════════════════

class TestCorrection:
    def test_basic(self):
        labels = np.array([1, 2, 3, 4])
        changed = np.array([False, True, False, True])
        corr = Correction(labels, changed, consistent=True, solver_time_ms=5.0)

        assert corr.num_corrections == 2
        assert abs(corr.correction_rate - 0.5) < 1e-6

    def test_no_corrections(self):
        corr = Correction(
            np.array([1, 2, 3]),
            np.array([False, False, False]),
        )
        assert corr.num_corrections == 0
        assert corr.correction_rate == 0.0

    def test_all_corrected(self):
        corr = Correction(
            np.array([1, 2]),
            np.array([True, True]),
        )
        assert corr.num_corrections == 2
        assert corr.correction_rate == 1.0

    def test_empty(self):
        corr = Correction(np.array([]), np.array([]))
        assert corr.correction_rate == 0.0


# ══════════════════════════════════════════════════════════════════
# Constraint & KnowledgeBase
# ══════════════════════════════════════════════════════════════════

class TestConstraint:
    def test_str(self):
        c = Constraint("row_0", [0, 1, 2], "all_distinct")
        assert "row_0" in str(c)
        assert "all_distinct" in str(c)

    def test_with_params(self):
        c = Constraint("sum_check", [0, 1], "sum_equals", {"target": 10})
        assert c.params["target"] == 10


class TestKnowledgeBase:
    def test_add_constraint(self):
        kb = KnowledgeBase()
        kb.add_constraint("test", [0, 1], "all_distinct")
        assert len(kb.constraints) == 1

    def test_all_distinct_pass(self):
        kb = KnowledgeBase()
        kb.add_constraint("row", [0, 1, 2], "all_distinct")
        ok, violations = kb.check_consistency(np.array([1, 2, 3]))
        assert ok
        assert len(violations) == 0

    def test_all_distinct_fail(self):
        kb = KnowledgeBase()
        kb.add_constraint("row", [0, 1, 2], "all_distinct")
        ok, violations = kb.check_consistency(np.array([1, 1, 3]))
        assert not ok
        assert "row" in violations

    def test_all_distinct_ignores_zeros(self):
        kb = KnowledgeBase()
        kb.add_constraint("row", [0, 1, 2], "all_distinct")
        ok, _ = kb.check_consistency(np.array([0, 0, 3]))
        assert ok  # Zeros are ignored

    def test_sum_equals_pass(self):
        kb = KnowledgeBase()
        kb.add_constraint("sum", [0, 1], "sum_equals", target=5)
        ok, _ = kb.check_consistency(np.array([2, 3]))
        assert ok

    def test_sum_equals_fail(self):
        kb = KnowledgeBase()
        kb.add_constraint("sum", [0, 1], "sum_equals", target=5)
        ok, violations = kb.check_consistency(np.array([2, 4]))
        assert not ok

    def test_in_range_pass(self):
        kb = KnowledgeBase()
        kb.add_constraint("range", [0, 1], "in_range", min=1, max=9)
        ok, _ = kb.check_consistency(np.array([5, 3]))
        assert ok

    def test_in_range_fail(self):
        kb = KnowledgeBase()
        kb.add_constraint("range", [0, 1], "in_range", min=1, max=9)
        ok, _ = kb.check_consistency(np.array([0, 3]))
        assert not ok

    def test_unknown_rule_passes(self):
        kb = KnowledgeBase()
        kb.add_constraint("custom", [0], "unknown_rule")
        ok, _ = kb.check_consistency(np.array([5]))
        assert ok  # Unknown rules pass by default

    def test_multiple_constraints(self):
        kb = KnowledgeBase()
        kb.add_constraint("row", [0, 1, 2], "all_distinct")
        kb.add_constraint("col", [0, 3, 6], "all_distinct")
        labels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        ok, _ = kb.check_consistency(labels)
        assert ok


# ══════════════════════════════════════════════════════════════════
# SolveResult
# ══════════════════════════════════════════════════════════════════

class TestSolveResult:
    def _make_result(self):
        pred = Prediction(
            np.array([1, 2, 3]),
            np.random.dirichlet(np.ones(10), 3),
            np.array([0.9, 0.8, 0.7]),
        )
        ref = Reflection(np.array([0, 1, 0]), np.array([0.1, 0.8, 0.2]))
        corr = Correction(np.array([1, 5, 3]), np.array([False, True, False]))
        kb = KnowledgeBase()
        kb.add_constraint("distinct", [0, 1, 2], "all_distinct")
        return SolveResult(pred, ref, corr, kb, total_time_ms=10.0)

    def test_final_labels(self):
        result = self._make_result()
        np.testing.assert_array_equal(result.final_labels, [1, 5, 3])

    def test_is_consistent(self):
        result = self._make_result()
        assert result.is_consistent  # [1, 5, 3] are all distinct

    def test_summary(self):
        result = self._make_result()
        s = result.summary
        assert s["num_items"] == 3
        assert s["num_flagged"] == 1
        assert s["num_corrections"] == 1
        assert s["is_consistent"] is True
        assert s["total_time_ms"] == 10.0


# ══════════════════════════════════════════════════════════════════
# TrainConfig & TrainStats
# ══════════════════════════════════════════════════════════════════

class TestTrainConfig:
    def test_defaults(self):
        config = TrainConfig()
        assert config.hidden_dim == 128
        assert config.num_classes == 10
        assert config.epochs == 50
        assert config.batch_size == 32
        assert config.learning_rate == 1e-3
        assert config.lambda_consistency == 1.0
        assert config.lambda_reflection_size == 0.1
        assert config.reflection_target_rate == 0.2
        assert config.solver == SolverType.BACKTRACK

    def test_custom(self):
        config = TrainConfig(hidden_dim=256, epochs=100, learning_rate=5e-4)
        assert config.hidden_dim == 256
        assert config.epochs == 100


class TestTrainStats:
    def test_to_dict(self):
        stats = TrainStats(epoch=5, train_loss=0.1234, accuracy=0.95)
        d = stats.to_dict()
        assert d["epoch"] == 5
        assert d["train_loss"] == 0.1234
        assert d["accuracy"] == 0.95

    def test_default_values(self):
        stats = TrainStats()
        assert stats.epoch == 0
        assert stats.train_loss == 0.0
