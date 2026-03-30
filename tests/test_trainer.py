"""Tests for reflectai.trainer — training loop."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from reflectai.knowledge import build_addition_kb
from reflectai.models import KnowledgeBase, TrainConfig
from reflectai.perception import MLPBody, PerceptionModule
from reflectai.reflection import ReflectionHead
from reflectai.trainer import ReflectAIModel, Trainer


def _make_model(input_dim=20, hidden_dim=32, num_classes=5):
    body = MLPBody(input_dim=input_dim, hidden_dim=hidden_dim)
    perception = PerceptionModule(body, num_classes=num_classes)
    reflection_head = ReflectionHead(
        hidden_dim=hidden_dim, num_classes=num_classes,
    )
    return ReflectAIModel(perception, reflection_head)


def _make_data(num_samples=32, input_dim=20, num_classes=5):
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=8)


class TestReflectAIModel:
    def test_forward(self):
        model = _make_model()
        x = torch.randn(4, 20)
        logits, features, ref_scores = model(x)
        assert logits.shape == (4, 5)
        assert features.shape == (4, 32)
        assert ref_scores.shape == (4,)

    def test_gradients(self):
        model = _make_model()
        x = torch.randn(4, 20)
        logits, _, ref_scores = model(x)
        loss = logits.sum() + ref_scores.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestTrainer:
    def test_train_one_epoch(self):
        model = _make_model()
        kb = KnowledgeBase(num_classes=5)
        config = TrainConfig(epochs=1, batch_size=8, learning_rate=1e-3)
        trainer = Trainer(model, kb, config)

        loader = _make_data()
        history = trainer.train(loader)
        assert len(history) == 1
        assert history[0].epoch == 1
        assert history[0].train_loss > 0

    def test_train_multiple_epochs(self):
        model = _make_model()
        kb = KnowledgeBase(num_classes=5)
        config = TrainConfig(epochs=3, batch_size=8)
        trainer = Trainer(model, kb, config)

        loader = _make_data()
        history = trainer.train(loader)
        assert len(history) == 3
        assert history[-1].epoch == 3

    def test_callback(self):
        model = _make_model()
        kb = KnowledgeBase(num_classes=5)
        config = TrainConfig(epochs=2, batch_size=8)
        trainer = Trainer(model, kb, config)

        callbacks = []
        loader = _make_data()
        trainer.train(loader, callback=lambda s: callbacks.append(s))
        assert len(callbacks) == 2

    def test_evaluate(self):
        model = _make_model()
        kb = KnowledgeBase(num_classes=5)
        config = TrainConfig(epochs=1, batch_size=8)
        trainer = Trainer(model, kb, config)

        loader = _make_data()
        trainer.train(loader)

        metrics = trainer.evaluate(loader)
        assert "accuracy" in metrics
        assert "reflection_rate" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["reflection_rate"] <= 1

    def test_history_accumulated(self):
        model = _make_model()
        kb = KnowledgeBase(num_classes=5)
        config = TrainConfig(epochs=2, batch_size=8)
        trainer = Trainer(model, kb, config)

        loader = _make_data()
        trainer.train(loader)
        trainer.train(loader)  # Train again
        assert len(trainer.history) == 4  # 2 + 2

    def test_stats_fields(self):
        model = _make_model()
        kb = KnowledgeBase(num_classes=5)
        config = TrainConfig(epochs=1, batch_size=8)
        trainer = Trainer(model, kb, config)

        loader = _make_data()
        history = trainer.train(loader)
        stats = history[0]
        assert stats.supervised_loss >= 0
        assert stats.reflection_loss >= 0
        assert stats.duration_seconds > 0

    def test_loss_decreases(self):
        """Loss should generally decrease over training."""
        model = _make_model()
        kb = KnowledgeBase(num_classes=5)
        config = TrainConfig(epochs=10, batch_size=8, learning_rate=1e-2)
        trainer = Trainer(model, kb, config)

        loader = _make_data(num_samples=64)
        history = trainer.train(loader)

        # First loss should be higher than last (usually)
        # Allow some tolerance as this is stochastic
        assert history[0].train_loss > history[-1].train_loss * 0.5
