"""Tests for reflectai.reflection — the core innovation."""

import pytest
import torch

from reflectai.reflection import (
    MultiPositionReflectionHead,
    ReflectionHead,
    consistency_reward_loss,
    reflection_size_loss,
)


class TestReflectionHead:
    def setup_method(self):
        self.head = ReflectionHead(hidden_dim=64, num_classes=10, threshold=0.5)

    def test_forward_shape(self):
        features = torch.randn(4, 64)
        logits = torch.randn(4, 10)
        scores = self.head(features, logits)
        assert scores.shape == (4,)

    def test_scores_range(self):
        features = torch.randn(8, 64)
        logits = torch.randn(8, 10)
        scores = self.head(features, logits)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_reflect(self):
        features = torch.randn(4, 64)
        logits = torch.randn(4, 10)
        flags, scores = self.head.reflect(features, logits)
        assert flags.shape == (4,)
        assert scores.shape == (4,)
        # Flags should be binary
        assert set(flags.unique().tolist()).issubset({0.0, 1.0})

    def test_threshold_override(self):
        features = torch.randn(4, 64)
        logits = torch.randn(4, 10)
        # Very low threshold → more flags
        flags_low, _ = self.head.reflect(features, logits, threshold=0.01)
        # Very high threshold → fewer flags
        flags_high, _ = self.head.reflect(features, logits, threshold=0.99)
        assert flags_low.sum() >= flags_high.sum()

    def test_without_logits(self):
        head = ReflectionHead(hidden_dim=64, use_logits=False)
        features = torch.randn(4, 64)
        scores = head(features)
        assert scores.shape == (4,)

    def test_gradient_flows(self):
        features = torch.randn(4, 64, requires_grad=True)
        logits = torch.randn(4, 10)
        scores = self.head(features, logits)
        loss = scores.sum()
        loss.backward()
        assert features.grad is not None

    def test_batch_position_input(self):
        """Test with [B, N, hidden_dim] input."""
        features = torch.randn(2, 10, 64)
        logits = torch.randn(2, 10, 10)
        scores = self.head(features, logits)
        assert scores.shape == (2, 10)


class TestMultiPositionReflectionHead:
    def setup_method(self):
        self.head = MultiPositionReflectionHead(
            hidden_dim=64, num_classes=10, num_positions=9,
            use_context=True, threshold=0.5,
        )

    def test_forward_shape(self):
        features = torch.randn(2, 9, 64)
        logits = torch.randn(2, 9, 10)
        scores = self.head(features, logits)
        assert scores.shape == (2, 9)

    def test_scores_range(self):
        features = torch.randn(2, 9, 64)
        logits = torch.randn(2, 9, 10)
        scores = self.head(features, logits)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_reflect(self):
        features = torch.randn(2, 9, 64)
        logits = torch.randn(2, 9, 10)
        flags, scores = self.head.reflect(features, logits)
        assert flags.shape == (2, 9)
        assert set(flags.unique().tolist()).issubset({0.0, 1.0})

    def test_without_context(self):
        head = MultiPositionReflectionHead(
            hidden_dim=64, num_positions=9, use_context=False,
        )
        features = torch.randn(2, 9, 64)
        logits = torch.randn(2, 9, 10)
        scores = head(features, logits)
        assert scores.shape == (2, 9)

    def test_without_logits(self):
        head = MultiPositionReflectionHead(
            hidden_dim=64, num_positions=9, use_logits=False,
        )
        features = torch.randn(2, 9, 64)
        scores = head(features)
        assert scores.shape == (2, 9)


class TestReflectionSizeLoss:
    def test_at_target(self):
        scores = torch.full((100,), 0.2)
        loss = reflection_size_loss(scores, target_rate=0.2)
        assert loss.item() < 1e-6

    def test_above_target(self):
        scores = torch.full((100,), 0.5)
        loss = reflection_size_loss(scores, target_rate=0.2)
        assert loss.item() > 0

    def test_below_target(self):
        scores = torch.full((100,), 0.05)
        loss = reflection_size_loss(scores, target_rate=0.2)
        assert loss.item() > 0

    def test_gradient(self):
        scores = torch.randn(10).sigmoid().requires_grad_(True)
        loss = reflection_size_loss(scores, target_rate=0.2)
        loss.backward()
        assert scores.grad is not None

    def test_2d_input(self):
        scores = torch.full((4, 9), 0.2)
        loss = reflection_size_loss(scores, target_rate=0.2)
        assert loss.item() < 1e-6


class TestConsistencyRewardLoss:
    def test_no_reward(self):
        scores = torch.rand(4)
        flags = (scores > 0.5).float()
        rewards = torch.zeros(4)
        loss = consistency_reward_loss(scores, flags, rewards)
        # With zero rewards, loss should be zero (or very small)
        assert abs(loss.item()) < 1e-5

    def test_positive_reward(self):
        scores = torch.tensor([0.8, 0.2])
        flags = torch.tensor([1.0, 0.0])
        rewards = torch.tensor([1.0, 1.0])
        loss = consistency_reward_loss(scores, flags, rewards)
        assert isinstance(loss.item(), float)

    def test_2d_scores(self):
        scores = torch.rand(2, 9)
        flags = (scores > 0.5).float()
        rewards = torch.ones(2)
        loss = consistency_reward_loss(scores, flags, rewards)
        assert isinstance(loss.item(), float)

    def test_gradient(self):
        scores = torch.rand(4, requires_grad=True)
        flags = (scores.detach() > 0.5).float()
        rewards = torch.ones(4)
        loss = consistency_reward_loss(scores, flags, rewards)
        loss.backward()
        assert scores.grad is not None
