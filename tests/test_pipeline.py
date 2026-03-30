"""Tests for reflectai.pipeline — end-to-end pipeline."""

import numpy as np
import pytest
import torch

from reflectai.knowledge import build_sudoku_4x4_kb, build_addition_kb
from reflectai.models import KnowledgeBase, Prediction, Reflection
from reflectai.pipeline import Pipeline, solve_from_predictions
from reflectai.perception import MLPBody, PerceptionModule
from reflectai.reflection import ReflectionHead
from reflectai.trainer import ReflectAIModel


class TestSolveFromPredictions:
    def test_basic(self):
        kb = KnowledgeBase(num_classes=4, num_positions=3)
        kb.add_constraint("distinct", [0, 1, 2], "all_distinct")

        labels = np.array([1, 1, 3])
        probs = np.full((3, 4), 0.1)
        for i, l in enumerate(labels):
            probs[i, l] = 0.7
        scores = np.array([0.1, 0.8, 0.1])

        result = solve_from_predictions(
            labels, probs, scores, kb, threshold=0.5,
        )
        assert result.prediction.num_items == 3
        assert result.reflection.num_flagged == 1
        assert result.correction is not None

    def test_consistent_result(self):
        kb = KnowledgeBase(num_classes=5, num_positions=3)
        kb.add_constraint("distinct", [0, 1, 2], "all_distinct")

        labels = np.array([1, 2, 3])
        probs = np.full((3, 5), 0.05)
        for i, l in enumerate(labels):
            probs[i, l] = 0.8
        scores = np.array([0.1, 0.1, 0.1])

        result = solve_from_predictions(labels, probs, scores, kb, threshold=0.5)
        assert result.is_consistent

    def test_addition_task(self):
        kb = build_addition_kb(num_digits=2, target_sum=5)

        labels = np.array([3, 4])  # Sum is 7, not 5
        probs = np.full((2, 10), 0.01)
        probs[0, 3] = 0.9
        probs[1, 4] = 0.9
        scores = np.array([0.1, 0.8])  # Flag position 1

        result = solve_from_predictions(
            labels, probs, scores, kb, threshold=0.5
        )
        assert result.reflection.num_flagged == 1
        # After correction, sum should be 5
        if result.correction.consistent:
            assert result.final_labels.sum() == 5


class TestPipeline:
    def _make_pipeline(self, num_classes=5):
        body = MLPBody(input_dim=20, hidden_dim=32)
        perception = PerceptionModule(body, num_classes=num_classes)
        reflection_head = ReflectionHead(
            hidden_dim=32, num_classes=num_classes, threshold=0.5,
        )
        model = ReflectAIModel(perception, reflection_head)

        kb = KnowledgeBase(num_classes=num_classes, num_positions=1)
        return Pipeline(model, kb)

    def test_solve_single(self):
        pipe = self._make_pipeline()
        x = torch.randn(20)
        result = pipe.solve(x)
        assert result.prediction is not None
        assert result.reflection is not None
        assert result.correction is not None
        assert result.total_time_ms > 0

    def test_solve_batch(self):
        pipe = self._make_pipeline()
        x = torch.randn(4, 20)
        results = pipe.solve_batch(x)
        assert len(results) == 4

    def test_summary(self):
        pipe = self._make_pipeline()
        x = torch.randn(20)
        result = pipe.solve(x)
        s = result.summary
        assert "num_items" in s
        assert "total_time_ms" in s

    def test_timing(self):
        pipe = self._make_pipeline()
        x = torch.randn(20)
        result = pipe.solve(x)
        assert result.perception_time_ms >= 0
        assert result.reflection_time_ms >= 0
        assert result.reasoning_time_ms >= 0
