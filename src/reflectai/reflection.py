"""Reflection head — the core innovation of ABL-Refl.

The reflection head is a binary classifier that learns to predict which
neural outputs are likely inconsistent with domain knowledge. It shares
the body features with the output head but has its own parameters.

Key properties:
- Trained with L_reflection_size to control flagging rate (target ~20%)
- Binary output: 1 = flagged (suspect), 0 = trusted
- Only flagged positions are sent to the abductive reasoner
- This targeted approach gives 10,000-15,000x speedup over brute-force
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectionHead(nn.Module):
    """Binary error detector branching from shared body features.

    Architecture:
        body features [B, hidden_dim]
        → Linear → ReLU → Linear → Sigmoid
        → reflection scores [B] (0.0 to 1.0)
        → threshold → binary flags [B]

    The reflection head can also take the output logits as additional
    input, allowing it to detect inconsistencies based on prediction
    confidence patterns.
    """

    def __init__(self, hidden_dim: int, num_classes: int = 10,
                 use_logits: bool = True, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.use_logits = use_logits

        # Input: features (+ optionally logits)
        input_dim = hidden_dim + (num_classes if use_logits else 0)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor,
                logits: torch.Tensor | None = None) -> torch.Tensor:
        """Compute continuous reflection scores.

        Args:
            features: [B, hidden_dim] or [B, N, hidden_dim] shared body features
            logits: [B, num_classes] or [B, N, num_classes] output head logits (optional)

        Returns:
            scores: [B] or [B, N] reflection scores in (0, 1)
        """
        if self.use_logits and logits is not None:
            x = torch.cat([features, logits], dim=-1)
        else:
            x = features

        scores = torch.sigmoid(self.net(x)).squeeze(-1)
        return scores

    def reflect(self, features: torch.Tensor,
                logits: torch.Tensor | None = None,
                threshold: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get both continuous scores and binary flags.

        Args:
            features: Body features
            logits: Output logits (optional)
            threshold: Override default threshold

        Returns:
            flags: [B] or [B, N] binary flags (1=flagged, 0=trusted)
            scores: [B] or [B, N] continuous scores
        """
        t = threshold if threshold is not None else self.threshold
        scores = self.forward(features, logits)
        flags = (scores >= t).float()
        return flags, scores


class MultiPositionReflectionHead(nn.Module):
    """Reflection head for multi-position inputs (e.g., Sudoku).

    Takes features from all positions AND inter-position context
    to detect inconsistencies. Uses a small attention mechanism to
    capture cross-position relationships.

    For Sudoku: considers all 81 positions to detect constraint violations.
    """

    def __init__(self, hidden_dim: int, num_classes: int = 10,
                 num_positions: int = 81, use_logits: bool = True,
                 use_context: bool = True, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.use_logits = use_logits
        self.use_context = use_context
        self.num_positions = num_positions

        per_pos_dim = hidden_dim + (num_classes if use_logits else 0)

        if use_context:
            # Simple cross-position attention for context
            self.context_proj = nn.Linear(per_pos_dim, hidden_dim // 2)
            self.context_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim // 2,
                num_heads=2,
                batch_first=True,
            )
            final_dim = per_pos_dim + hidden_dim // 2
        else:
            final_dim = per_pos_dim

        self.head = nn.Sequential(
            nn.Linear(final_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor,
                logits: torch.Tensor | None = None) -> torch.Tensor:
        """Compute reflection scores with cross-position context.

        Args:
            features: [B, N, hidden_dim]
            logits: [B, N, num_classes] (optional)

        Returns:
            scores: [B, N] reflection scores
        """
        if self.use_logits and logits is not None:
            x = torch.cat([features, logits], dim=-1)  # [B, N, D]
        else:
            x = features

        if self.use_context:
            ctx = self.context_proj(x)  # [B, N, hidden_dim//2]
            ctx, _ = self.context_attn(ctx, ctx, ctx)  # self-attention
            x = torch.cat([x, ctx], dim=-1)  # [B, N, D + hidden_dim//2]

        scores = torch.sigmoid(self.head(x)).squeeze(-1)  # [B, N]
        return scores

    def reflect(self, features: torch.Tensor,
                logits: torch.Tensor | None = None,
                threshold: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get flags and scores."""
        t = threshold if threshold is not None else self.threshold
        scores = self.forward(features, logits)
        flags = (scores >= t).float()
        return flags, scores


# ══════════════════════════════════════════════════════════════════
# Loss Functions for Reflection
# ══════════════════════════════════════════════════════════════════

def reflection_size_loss(scores: torch.Tensor,
                         target_rate: float = 0.2) -> torch.Tensor:
    """L_reflection_size: penalize deviation from target flagging rate.

    From the paper: encourages ~20% flagging rate (C=0.8).
    Prevents the reflection head from flagging everything or nothing.

    L = (mean(scores) - target_rate)^2

    Args:
        scores: [B] or [B, N] reflection scores
        target_rate: Target fraction of flagged positions

    Returns:
        Scalar loss
    """
    mean_score = scores.mean()
    return (mean_score - target_rate) ** 2


def consistency_reward_loss(scores: torch.Tensor,
                            flags: torch.Tensor,
                            consistency_improved: torch.Tensor) -> torch.Tensor:
    """L_consistency: REINFORCE-based reward for improved consistency.

    When abductive correction of flagged positions improves constraint
    satisfaction, reinforce the reflection pattern. This is the key
    signal that trains the reflection head to identify actual errors.

    Args:
        scores: [B, N] or [B] reflection scores (used as policy)
        flags: [B, N] or [B] binary flags (actions taken)
        consistency_improved: [B] float reward (1.0 if improved, 0 otherwise)

    Returns:
        Scalar policy gradient loss
    """
    # REINFORCE: -reward * log_prob(action)
    eps = 1e-8
    log_prob_flag = torch.log(scores + eps) * flags
    log_prob_trust = torch.log(1 - scores + eps) * (1 - flags)
    log_prob = log_prob_flag + log_prob_trust

    if log_prob.dim() > 1:
        log_prob = log_prob.sum(dim=-1)  # Sum over positions

    # Weight by reward
    if consistency_improved.dim() == 0:
        consistency_improved = consistency_improved.unsqueeze(0)

    loss = -(consistency_improved * log_prob).mean()
    return loss
