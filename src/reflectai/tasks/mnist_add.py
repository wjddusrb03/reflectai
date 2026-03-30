"""MNIST Addition task for ReflectAI.

Task: Given images of two MNIST digits, predict the sum.
The reflection head detects when digit predictions are inconsistent
with the known sum constraint.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..knowledge import build_addition_kb
from ..models import KnowledgeBase, Prediction


def generate_addition_samples(num_samples: int = 100,
                              num_digits: int = 2,
                              seed: int | None = None) -> list[dict]:
    """Generate synthetic addition problems.

    Each sample has: digits (the actual values), target_sum,
    and synthetic "image" features (random vectors simulating
    neural network embeddings).

    Args:
        num_samples: Number of problems to generate
        num_digits: Number of digits to add (default 2)
        seed: Random seed

    Returns:
        List of dicts with 'digits', 'target_sum', 'features'
    """
    rng = np.random.RandomState(seed)
    samples = []

    for _ in range(num_samples):
        digits = rng.randint(0, 10, size=num_digits)
        target_sum = int(digits.sum())

        # Simulate neural features (28*28 = 784 dim per digit)
        features = rng.randn(num_digits, 784).astype(np.float32)

        samples.append({
            "digits": digits,
            "target_sum": target_sum,
            "features": features,
        })

    return samples


def simulate_addition_predictions(digits: np.ndarray,
                                  error_rate: float = 0.2,
                                  seed: int | None = None) -> Prediction:
    """Simulate noisy neural predictions for digit addition.

    Args:
        digits: [N] true digit values
        error_rate: Probability of each digit being misclassified
        seed: Random seed

    Returns:
        Prediction with possibly incorrect labels
    """
    rng = np.random.RandomState(seed)
    N = len(digits)

    labels = digits.copy()
    probs = np.full((N, 10), 0.01)

    for i in range(N):
        if rng.random() < error_rate:
            # Misclassify
            wrong = [v for v in range(10) if v != digits[i]]
            labels[i] = rng.choice(wrong)
            probs[i] = rng.dirichlet(np.ones(10) * 2)
        else:
            probs[i, labels[i]] = 0.9
            remaining = 0.1 / 9
            probs[i] = remaining
            probs[i, labels[i]] = 0.9

    confidence = probs.max(axis=1)
    return Prediction(labels=labels, probabilities=probs, confidence=confidence)


def evaluate_addition(prediction: Prediction,
                      true_digits: np.ndarray,
                      corrected_labels: np.ndarray,
                      target_sum: int) -> dict:
    """Evaluate addition task performance.

    Args:
        prediction: Original neural predictions
        true_digits: Ground truth digit values
        corrected_labels: Labels after reflection + reasoning
        target_sum: Expected sum

    Returns:
        Evaluation metrics
    """
    pred_sum = int(prediction.labels.sum())
    corr_sum = int(corrected_labels.sum())

    return {
        "true_digits": true_digits.tolist(),
        "predicted_digits": prediction.labels.tolist(),
        "corrected_digits": corrected_labels.tolist(),
        "target_sum": target_sum,
        "predicted_sum": pred_sum,
        "corrected_sum": corr_sum,
        "prediction_correct": pred_sum == target_sum,
        "correction_correct": corr_sum == target_sum,
        "digit_accuracy_before": float((prediction.labels == true_digits).mean()),
        "digit_accuracy_after": float((corrected_labels == true_digits).mean()),
    }
