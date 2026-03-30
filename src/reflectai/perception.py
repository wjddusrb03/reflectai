"""Neural perception module for ReflectAI.

Implements the Body + Output Head architecture from ABL-Refl:
- BodyBlock: shared feature extractor (CNN for images, MLP for vectors)
- OutputHead: classification head producing softmax probabilities
- PerceptionModule: combined body + head with forward pass
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
# Body Blocks (shared feature extractors)
# ══════════════════════════════════════════════════════════════════

class MLPBody(nn.Module):
    """MLP-based body block for flat/vector inputs.

    Architecture: Input → Linear → ReLU → Linear → ReLU → features
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, input_dim] → [B, hidden_dim]"""
        return self.net(x)


class CNNBody(nn.Module):
    """CNN-based body block for image inputs (e.g., MNIST digits).

    Architecture: Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten
    Designed for 28x28 grayscale images (MNIST).
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # For 28x28 input: after two 2x pooling → 7x7 → 64*7*7 = 3136
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → [B, hidden_dim]"""
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


# ══════════════════════════════════════════════════════════════════
# Output Head
# ══════════════════════════════════════════════════════════════════

class OutputHead(nn.Module):
    """Classification head producing per-class probabilities.

    Takes features from body block and outputs softmax distribution.
    """

    def __init__(self, hidden_dim: int, num_classes: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, hidden_dim] → logits: [B, num_classes]"""
        return self.head(features)


# ══════════════════════════════════════════════════════════════════
# Perception Module (Body + Output Head)
# ══════════════════════════════════════════════════════════════════

class PerceptionModule(nn.Module):
    """Combined perception: Body block + Output head.

    This is the standard neural component. The reflection head
    (in reflection.py) branches off from the body's features.

    Usage:
        model = PerceptionModule(body=MLPBody(784), num_classes=10)
        logits, features = model(x)  # features for reflection head
    """

    def __init__(self, body: nn.Module, num_classes: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.body = body
        self.output_head = OutputHead(body.output_dim, num_classes, dropout)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both logits and shared features.

        Args:
            x: Input tensor (shape depends on body type)

        Returns:
            logits: [B, num_classes] classification logits
            features: [B, hidden_dim] shared features for reflection head
        """
        features = self.body(x)
        logits = self.output_head(features)
        return logits, features

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get predicted labels and probabilities.

        Returns:
            labels: [B] predicted class indices
            probs: [B, num_classes] softmax probabilities
        """
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            labels = torch.argmax(probs, dim=-1)
        return labels, probs


# ══════════════════════════════════════════════════════════════════
# Multi-position Perception (for puzzles like Sudoku)
# ══════════════════════════════════════════════════════════════════

class MultiPositionPerception(nn.Module):
    """Perception for multiple positions sharing the same body.

    For Sudoku: 81 digit images → shared CNN body → 81 independent
    classification heads (or shared head applied position-wise).

    Args:
        body: Shared feature extractor
        num_positions: Number of positions (e.g., 81 for Sudoku)
        num_classes: Classes per position (e.g., 10 for digits 0-9)
    """

    def __init__(self, body: nn.Module, num_positions: int,
                 num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.body = body
        self.num_positions = num_positions
        self.num_classes = num_classes
        # Shared output head applied to each position
        self.output_head = OutputHead(body.output_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process all positions with shared body.

        Args:
            x: [B, N, ...] where N=num_positions, ... depends on body

        Returns:
            logits: [B, N, num_classes]
            features: [B, N, hidden_dim]
        """
        B, N = x.shape[0], x.shape[1]
        # Flatten batch and positions
        rest_shape = x.shape[2:]
        x_flat = x.reshape(B * N, *rest_shape)

        features = self.body(x_flat)        # [B*N, hidden_dim]
        logits = self.output_head(features)  # [B*N, num_classes]

        hidden_dim = features.shape[-1]
        features = features.reshape(B, N, hidden_dim)
        logits = logits.reshape(B, N, self.num_classes)

        return logits, features

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get predicted labels and probabilities for all positions."""
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            labels = torch.argmax(probs, dim=-1)
        return labels, probs
