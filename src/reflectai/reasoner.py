"""Abductive reasoning module for ReflectAI.

Implements constraint-based correction of flagged predictions:
1. BacktrackSolver: Pure Python backtracking (always available)
2. Z3Solver: Z3 SMT solver (optional, pip install reflectai[z3])
3. SATSolver: PySAT-based solver (optional, pip install reflectai[sat])

The key insight: only flagged positions (from reflection head) are
searched — unflagged positions are fixed. This gives massive speedup
over searching all positions.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .models import Constraint, Correction, KnowledgeBase, Prediction, Reflection


# ══════════════════════════════════════════════════════════════════
# Abstract Solver
# ══════════════════════════════════════════════════════════════════

class AbductiveSolver(ABC):
    """Base class for constraint solvers."""

    @abstractmethod
    def solve(self, prediction: Prediction, reflection: Reflection,
              kb: KnowledgeBase, timeout_ms: float = 1000.0) -> Correction:
        """Find corrected labels that satisfy constraints.

        Only positions flagged by the reflection head are modified.
        Unflagged positions retain their predicted values.

        Args:
            prediction: Neural network predictions
            reflection: Reflection flags indicating suspect positions
            kb: Knowledge base with constraints
            timeout_ms: Solver timeout in milliseconds

        Returns:
            Correction with corrected labels
        """
        ...


# ══════════════════════════════════════════════════════════════════
# Backtracking Solver (always available, no dependencies)
# ══════════════════════════════════════════════════════════════════

class BacktrackSolver(AbductiveSolver):
    """Pure Python backtracking solver with constraint propagation.

    Uses the probability distribution from the neural network to
    order candidate values (most likely first), making the search
    much more efficient than blind enumeration.
    """

    def __init__(self, max_iterations: int = 100_000):
        self.max_iterations = max_iterations

    def solve(self, prediction: Prediction, reflection: Reflection,
              kb: KnowledgeBase, timeout_ms: float = 1000.0) -> Correction:
        start = time.perf_counter()

        labels = prediction.labels.copy()
        flagged = reflection.flagged_indices
        N = len(labels)

        if len(flagged) == 0:
            # Nothing flagged — return predictions as-is
            elapsed = (time.perf_counter() - start) * 1000
            return Correction(
                labels=labels,
                changed_mask=np.zeros(N, dtype=bool),
                consistent=True,
                solver_time_ms=elapsed,
            )

        # Order candidates by probability (most likely first)
        num_classes = kb.num_classes
        candidates = {}
        for idx in flagged:
            if prediction.probabilities.ndim > 1:
                probs = prediction.probabilities[idx]
                order = np.argsort(-probs)
            else:
                order = np.arange(num_classes)
            candidates[idx] = order

        # Backtracking search
        timeout_s = timeout_ms / 1000.0
        solution = self._backtrack(
            labels, flagged, candidates, kb, start, timeout_s
        )

        elapsed = (time.perf_counter() - start) * 1000

        if solution is not None:
            changed = solution != prediction.labels
            consistent, _ = kb.check_consistency(solution)
            return Correction(
                labels=solution,
                changed_mask=changed,
                consistent=consistent,
                solver_time_ms=elapsed,
            )
        else:
            # Timeout or no solution — return original predictions
            return Correction(
                labels=prediction.labels.copy(),
                changed_mask=np.zeros(N, dtype=bool),
                consistent=False,
                solver_time_ms=elapsed,
            )

    def _backtrack(self, labels: np.ndarray, flagged: np.ndarray,
                   candidates: dict, kb: KnowledgeBase,
                   start_time: float, timeout_s: float,
                   depth: int = 0) -> np.ndarray | None:
        """Recursive backtracking with early constraint checking."""
        if time.perf_counter() - start_time > timeout_s:
            return None

        if depth == len(flagged):
            # All flagged positions assigned — check full consistency
            consistent, _ = kb.check_consistency(labels)
            if consistent:
                return labels.copy()
            return None

        idx = flagged[depth]
        for value in candidates[idx]:
            labels[idx] = value

            # Early pruning: check constraints involving this position
            if self._partial_consistent(labels, idx, kb):
                result = self._backtrack(
                    labels, flagged, candidates, kb,
                    start_time, timeout_s, depth + 1
                )
                if result is not None:
                    return result

        # Restore original (will be overwritten by caller anyway)
        return None

    def _partial_consistent(self, labels: np.ndarray, idx: int,
                            kb: KnowledgeBase) -> bool:
        """Check only constraints involving the given position."""
        for c in kb.constraints:
            if idx in c.indices:
                values = labels[c.indices]
                if not kb._check_rule(values, c.rule, c.params):
                    return False
        return True


# ══════════════════════════════════════════════════════════════════
# Z3 Solver (optional)
# ══════════════════════════════════════════════════════════════════

class Z3Solver(AbductiveSolver):
    """Z3 SMT solver for constraint satisfaction.

    More powerful than backtracking for complex constraint sets.
    Requires: pip install z3-solver
    """

    def __init__(self):
        try:
            import z3
            self._z3 = z3
        except ImportError:
            raise ImportError(
                "Z3 solver requires z3-solver package. "
                "Install with: pip install reflectai[z3]"
            )

    def solve(self, prediction: Prediction, reflection: Reflection,
              kb: KnowledgeBase, timeout_ms: float = 1000.0) -> Correction:
        z3 = self._z3
        start = time.perf_counter()

        N = len(prediction.labels)
        flagged_set = set(reflection.flagged_indices.tolist())

        # Create Z3 variables only for flagged positions
        solver = z3.Solver()
        solver.set("timeout", int(timeout_ms))

        variables = {}
        for i in range(N):
            if i in flagged_set:
                v = z3.Int(f"x_{i}")
                # Domain constraint: value in [0, num_classes)
                solver.add(v >= 0, v < kb.num_classes)
                variables[i] = v
                # Soft preference: prefer more probable values
                if prediction.probabilities.ndim > 1:
                    best = int(np.argmax(prediction.probabilities[i]))
                    # Add as soft constraint via optimization later
            else:
                variables[i] = int(prediction.labels[i])

        # Add knowledge base constraints
        for c in kb.constraints:
            self._add_constraint(solver, c, variables, z3)

        elapsed = (time.perf_counter() - start) * 1000

        if solver.check() == z3.sat:
            model = solver.model()
            labels = prediction.labels.copy()
            for i in flagged_set:
                val = model.evaluate(variables[i])
                labels[i] = val.as_long()
            changed = labels != prediction.labels
            elapsed = (time.perf_counter() - start) * 1000
            consistent, _ = kb.check_consistency(labels)
            return Correction(
                labels=labels,
                changed_mask=changed,
                consistent=consistent,
                solver_time_ms=elapsed,
            )
        else:
            elapsed = (time.perf_counter() - start) * 1000
            return Correction(
                labels=prediction.labels.copy(),
                changed_mask=np.zeros(N, dtype=bool),
                consistent=False,
                solver_time_ms=elapsed,
            )

    def _add_constraint(self, solver, c: Constraint, variables: dict,
                        z3) -> None:
        """Add a single constraint to the Z3 solver."""
        vars_list = [
            variables[i] if isinstance(variables[i], z3.ArithRef)
            else z3.IntVal(int(variables[i]))
            for i in c.indices
        ]

        if c.rule == "all_distinct":
            solver.add(z3.Distinct(*vars_list))
        elif c.rule == "sum_equals":
            target = c.params.get("target", 0)
            solver.add(z3.Sum(vars_list) == target)
        elif c.rule == "in_range":
            lo = c.params.get("min", 1)
            hi = c.params.get("max", 9)
            for v in vars_list:
                solver.add(z3.And(v >= lo, v <= hi))


# ══════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════

def create_solver(solver_type: str = "backtrack",
                  **kwargs) -> AbductiveSolver:
    """Factory function to create a solver.

    Args:
        solver_type: "backtrack", "z3", or "sat"
        **kwargs: Solver-specific parameters

    Returns:
        AbductiveSolver instance
    """
    if solver_type == "backtrack":
        return BacktrackSolver(**kwargs)
    elif solver_type == "z3":
        return Z3Solver(**kwargs)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}. "
                         f"Available: backtrack, z3")
