"""ReflectAI pipeline — orchestrates perception → reflection → reasoning.

The pipeline connects the three stages:
1. Perception: Neural network produces predictions + confidence
2. Reflection: Binary error detector flags suspect positions
3. Reasoning: Abductive solver corrects only flagged positions

This module provides both a high-level Pipeline class and utility
functions for step-by-step usage.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .models import (
    Correction, KnowledgeBase, Prediction, Reflection,
    SolveResult, SolverType,
)
from .reasoner import AbductiveSolver, BacktrackSolver, create_solver
from .trainer import ReflectAIModel


class Pipeline:
    """Full ReflectAI inference pipeline.

    Usage:
        pipeline = Pipeline(model, kb, solver_type="backtrack")
        result = pipeline.solve(input_tensor)
        print(result.summary)
    """

    def __init__(self, model: ReflectAIModel, kb: KnowledgeBase,
                 solver_type: str = "backtrack",
                 reflection_threshold: float | None = None,
                 solver_timeout_ms: float = 1000.0,
                 device: str = "cpu"):
        self.model = model
        self.kb = kb
        self.solver = create_solver(solver_type)
        self.solver_timeout_ms = solver_timeout_ms
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        if reflection_threshold is not None:
            self.model.reflection_head.threshold = reflection_threshold

    def solve(self, x: torch.Tensor) -> SolveResult:
        """Run the full pipeline on input.

        Args:
            x: Input tensor (single sample, no batch dimension needed)

        Returns:
            SolveResult with prediction, reflection, correction, and timing
        """
        total_start = time.perf_counter()

        # Add batch dimension if needed
        if x.dim() == 1 or (x.dim() >= 2 and x.shape[0] != 1):
            x = x.unsqueeze(0)
        x = x.to(self.device)

        # Stage 1: Perception
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, features, ref_scores = self.model(x)
        perception_time = (time.perf_counter() - t0) * 1000

        # Convert to numpy
        if logits.dim() == 3:
            # Multi-position: [1, N, C]
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_labels = logits[0].argmax(dim=-1).cpu().numpy()
            scores_np = ref_scores[0].cpu().numpy()
        else:
            # Single: [1, C]
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_labels = logits[0].argmax(dim=-1).cpu().numpy()
            if pred_labels.ndim == 0:
                pred_labels = np.array([pred_labels.item()])
                probs = probs.reshape(1, -1)
            scores_np = ref_scores[0].cpu().numpy()
            if scores_np.ndim == 0:
                scores_np = np.array([scores_np.item()])

        prediction = Prediction(
            labels=pred_labels,
            probabilities=probs,
            confidence=probs.max(axis=-1) if probs.ndim > 1 else np.array([probs.max()]),
        )

        # Stage 2: Reflection
        t1 = time.perf_counter()
        threshold = self.model.reflection_head.threshold
        flags = (scores_np >= threshold).astype(int)
        reflection = Reflection(
            flags=flags,
            scores=scores_np,
            threshold=threshold,
        )
        reflection_time = (time.perf_counter() - t1) * 1000

        # Stage 3: Abductive Reasoning
        t2 = time.perf_counter()
        correction = self.solver.solve(
            prediction, reflection, self.kb,
            timeout_ms=self.solver_timeout_ms,
        )
        reasoning_time = (time.perf_counter() - t2) * 1000

        total_time = (time.perf_counter() - total_start) * 1000

        return SolveResult(
            prediction=prediction,
            reflection=reflection,
            correction=correction,
            knowledge_base=self.kb,
            perception_time_ms=perception_time,
            reflection_time_ms=reflection_time,
            reasoning_time_ms=reasoning_time,
            total_time_ms=total_time,
        )

    def solve_batch(self, x: torch.Tensor) -> list[SolveResult]:
        """Solve a batch of inputs.

        Args:
            x: [B, ...] batch of inputs

        Returns:
            List of SolveResult, one per sample
        """
        results = []
        for i in range(x.shape[0]):
            results.append(self.solve(x[i]))
        return results


# ══════════════════════════════════════════════════════════════════
# Utility: solve with raw numpy (no model needed)
# ══════════════════════════════════════════════════════════════════

def solve_from_predictions(labels: np.ndarray,
                           probabilities: np.ndarray,
                           reflection_scores: np.ndarray,
                           kb: KnowledgeBase,
                           threshold: float = 0.5,
                           solver_type: str = "backtrack",
                           timeout_ms: float = 1000.0) -> SolveResult:
    """Run reflection + reasoning from pre-computed predictions.

    Useful for testing or when perception is done externally.

    Args:
        labels: [N] predicted labels
        probabilities: [N, C] probability distributions
        reflection_scores: [N] reflection scores (0-1)
        kb: Knowledge base
        threshold: Reflection threshold
        solver_type: Solver backend
        timeout_ms: Solver timeout

    Returns:
        SolveResult
    """
    prediction = Prediction(
        labels=labels,
        probabilities=probabilities,
        confidence=probabilities.max(axis=-1) if probabilities.ndim > 1
        else np.array([probabilities.max()]),
    )

    flags = (reflection_scores >= threshold).astype(int)
    reflection = Reflection(flags=flags, scores=reflection_scores, threshold=threshold)

    solver = create_solver(solver_type)
    correction = solver.solve(prediction, reflection, kb, timeout_ms=timeout_ms)

    return SolveResult(
        prediction=prediction,
        reflection=reflection,
        correction=correction,
        knowledge_base=kb,
    )
