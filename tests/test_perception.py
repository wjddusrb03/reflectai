"""Tests for reflectai.perception — neural network modules."""

import numpy as np
import pytest
import torch

from reflectai.perception import (
    CNNBody, MLPBody, MultiPositionPerception, OutputHead, PerceptionModule,
)


class TestMLPBody:
    def test_output_shape(self):
        body = MLPBody(input_dim=784, hidden_dim=128)
        x = torch.randn(4, 784)
        out = body(x)
        assert out.shape == (4, 128)

    def test_output_dim(self):
        body = MLPBody(input_dim=100, hidden_dim=64)
        assert body.output_dim == 64

    def test_single_sample(self):
        body = MLPBody(input_dim=50, hidden_dim=32)
        body.eval()  # BatchNorm requires eval mode for single samples
        x = torch.randn(1, 50)
        out = body(x)
        assert out.shape == (1, 32)

    def test_gradient_flows(self):
        body = MLPBody(input_dim=20, hidden_dim=16)
        x = torch.randn(2, 20, requires_grad=True)
        out = body(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestCNNBody:
    def test_output_shape(self):
        body = CNNBody(in_channels=1, hidden_dim=128)
        x = torch.randn(4, 1, 28, 28)
        out = body(x)
        assert out.shape == (4, 128)

    def test_output_dim(self):
        body = CNNBody(in_channels=1, hidden_dim=64)
        assert body.output_dim == 64

    def test_rgb_input(self):
        body = CNNBody(in_channels=3, hidden_dim=128)
        x = torch.randn(2, 3, 28, 28)
        out = body(x)
        assert out.shape == (2, 128)


class TestOutputHead:
    def test_output_shape(self):
        head = OutputHead(hidden_dim=128, num_classes=10)
        features = torch.randn(4, 128)
        logits = head(features)
        assert logits.shape == (4, 10)

    def test_different_classes(self):
        head = OutputHead(hidden_dim=64, num_classes=5)
        features = torch.randn(2, 64)
        logits = head(features)
        assert logits.shape == (2, 5)


class TestPerceptionModule:
    def setup_method(self):
        self.body = MLPBody(input_dim=784, hidden_dim=128)
        self.model = PerceptionModule(self.body, num_classes=10)

    def test_forward_shapes(self):
        x = torch.randn(4, 784)
        logits, features = self.model(x)
        assert logits.shape == (4, 10)
        assert features.shape == (4, 128)

    def test_predict(self):
        x = torch.randn(4, 784)
        labels, probs = self.model.predict(x)
        assert labels.shape == (4,)
        assert probs.shape == (4, 10)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_predict_values_valid(self):
        x = torch.randn(2, 784)
        labels, probs = self.model.predict(x)
        assert (labels >= 0).all() and (labels < 10).all()
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_num_classes(self):
        assert self.model.num_classes == 10

    def test_with_cnn_body(self):
        body = CNNBody(in_channels=1, hidden_dim=64)
        model = PerceptionModule(body, num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        logits, features = model(x)
        assert logits.shape == (2, 10)
        assert features.shape == (2, 64)


class TestMultiPositionPerception:
    def setup_method(self):
        self.body = MLPBody(input_dim=784, hidden_dim=128)
        self.model = MultiPositionPerception(
            self.body, num_positions=81, num_classes=10
        )

    def test_forward_shapes(self):
        x = torch.randn(2, 81, 784)
        logits, features = self.model(x)
        assert logits.shape == (2, 81, 10)
        assert features.shape == (2, 81, 128)

    def test_predict(self):
        x = torch.randn(2, 81, 784)
        labels, probs = self.model.predict(x)
        assert labels.shape == (2, 81)
        assert probs.shape == (2, 81, 10)

    def test_shared_weights(self):
        """Body should use the same weights for all positions."""
        self.model.eval()  # Eval mode for consistent BatchNorm behavior
        x = torch.randn(1, 4, 784)
        # Forward with same input at different positions
        x[:, 0] = x[:, 1]  # Same input at pos 0 and 1
        logits, _ = self.model(x)
        # Same input should give same output
        torch.testing.assert_close(logits[:, 0], logits[:, 1])

    def test_small_positions(self):
        body = MLPBody(input_dim=20, hidden_dim=16)
        model = MultiPositionPerception(body, num_positions=4, num_classes=5)
        x = torch.randn(3, 4, 20)
        logits, features = model(x)
        assert logits.shape == (3, 4, 5)
        assert features.shape == (3, 4, 16)
