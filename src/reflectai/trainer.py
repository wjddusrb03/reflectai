"""Training loop for ReflectAI.

Implements the three-loss training from ABL-Refl:
1. L_supervised: Cross-entropy on labeled data
2. L_consistency: REINFORCE reward when correction improves consistency
3. L_reflection_size: Regularizer to control flagging rate (~20%)

Total loss = L_supervised + λ_c * L_consistency + λ_r * L_reflection_size
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import (
    Correction, KnowledgeBase, Prediction, Reflection,
    SolverType, TrainConfig, TrainStats,
)
from .perception import PerceptionModule, MultiPositionPerception
from .reasoner import AbductiveSolver, BacktrackSolver, create_solver
from .reflection import (
    ReflectionHead, MultiPositionReflectionHead,
    consistency_reward_loss, reflection_size_loss,
)


class ReflectAIModel(nn.Module):
    """Complete ReflectAI model: Perception + Reflection Head.

    This combines the perception module (body + output head) with
    the reflection head into a single trainable model.
    """

    def __init__(self, perception: PerceptionModule | MultiPositionPerception,
                 reflection_head: ReflectionHead | MultiPositionReflectionHead):
        super().__init__()
        self.perception = perception
        self.reflection_head = reflection_head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Returns:
            logits: Classification logits
            features: Shared body features
            reflection_scores: Reflection scores (0-1)
        """
        logits, features = self.perception(x)
        reflection_scores = self.reflection_head(features, logits)
        return logits, features, reflection_scores


class Trainer:
    """Training loop implementing ABL-Refl's three-loss objective.

    Usage:
        config = TrainConfig(epochs=50, learning_rate=1e-3)
        trainer = Trainer(model, kb, config)
        stats = trainer.train(train_loader)
    """

    def __init__(self, model: ReflectAIModel, kb: KnowledgeBase,
                 config: TrainConfig | None = None):
        self.model = model
        self.kb = kb
        self.config = config or TrainConfig()

        # Solver for generating consistency rewards
        self.solver = create_solver(self.config.solver.value)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        # History
        self.history: list[TrainStats] = []

    def train(self, train_loader: DataLoader,
              callback: Callable[[TrainStats], None] | None = None) -> list[TrainStats]:
        """Run full training loop.

        Args:
            train_loader: DataLoader yielding (inputs, labels) batches
            callback: Optional function called after each epoch with stats

        Returns:
            List of TrainStats, one per epoch
        """
        self.model.train()
        history = []

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            stats = self._train_epoch(train_loader, epoch)
            stats.duration_seconds = time.time() - epoch_start

            history.append(stats)
            self.history.append(stats)

            if callback:
                callback(stats)

        return history

    def _train_epoch(self, loader: DataLoader, epoch: int) -> TrainStats:
        """Train for one epoch."""
        total_loss = 0.0
        total_sup_loss = 0.0
        total_con_loss = 0.0
        total_ref_loss = 0.0
        correct = 0
        total = 0
        flag_counts = 0
        flag_total = 0
        consistent_count = 0
        batch_count = 0

        for batch in loader:
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)

            # Forward pass
            logits, features, ref_scores = self.model(inputs)

            # 1. Supervised loss (cross-entropy)
            if logits.dim() == 3:
                # Multi-position: [B, N, C] → [B*N, C]
                B, N, C = logits.shape
                sup_loss = F.cross_entropy(
                    logits.reshape(-1, C), labels.reshape(-1)
                )
            else:
                sup_loss = F.cross_entropy(logits, labels)

            # 2. Reflection size loss
            ref_loss = reflection_size_loss(
                ref_scores, self.config.reflection_target_rate
            )

            # 3. Consistency loss (REINFORCE)
            con_loss = self._compute_consistency_loss(
                logits, ref_scores, labels
            )

            # Total loss
            loss = (sup_loss
                    + self.config.lambda_consistency * con_loss
                    + self.config.lambda_reflection_size * ref_loss)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Track stats
            total_loss += loss.item()
            total_sup_loss += sup_loss.item()
            total_con_loss += con_loss.item()
            total_ref_loss += ref_loss.item()
            batch_count += 1

            # Accuracy
            with torch.no_grad():
                if logits.dim() == 3:
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.numel()
                    flag_counts += (ref_scores >= self.model.reflection_head.threshold).sum().item()
                    flag_total += ref_scores.numel()
                else:
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    flag_counts += (ref_scores >= self.model.reflection_head.threshold).sum().item()
                    flag_total += ref_scores.numel()

        n = max(batch_count, 1)
        return TrainStats(
            epoch=epoch,
            train_loss=total_loss / n,
            supervised_loss=total_sup_loss / n,
            consistency_loss=total_con_loss / n,
            reflection_loss=total_ref_loss / n,
            accuracy=correct / max(total, 1),
            reflection_rate=flag_counts / max(flag_total, 1),
            consistency_rate=consistent_count / max(batch_count, 1),
        )

    def _compute_consistency_loss(self, logits: torch.Tensor,
                                  ref_scores: torch.Tensor,
                                  true_labels: torch.Tensor) -> torch.Tensor:
        """Compute REINFORCE-based consistency loss.

        Uses the solver to check if correcting flagged positions
        improves constraint satisfaction.
        """
        with torch.no_grad():
            threshold = self.model.reflection_head.threshold

            if logits.dim() == 3:
                # Multi-position
                B, N, C = logits.shape
                probs = F.softmax(logits, dim=-1)
                pred_labels = logits.argmax(dim=-1)  # [B, N]
                flags = (ref_scores >= threshold).float()  # [B, N]

                rewards = []
                for b in range(B):
                    pred_np = pred_labels[b].cpu().numpy()
                    prob_np = probs[b].cpu().numpy()
                    flag_np = flags[b].cpu().numpy().astype(int)

                    prediction = Prediction(
                        labels=pred_np,
                        probabilities=prob_np,
                        confidence=prob_np.max(axis=-1),
                    )
                    reflection = Reflection(
                        flags=flag_np,
                        scores=ref_scores[b].cpu().numpy(),
                        threshold=threshold,
                    )

                    # Check pre-correction consistency
                    pre_ok, _ = self.kb.check_consistency(pred_np)

                    if reflection.num_flagged > 0:
                        correction = self.solver.solve(
                            prediction, reflection, self.kb,
                            timeout_ms=self.config.solver_timeout_ms,
                        )
                        post_ok, _ = self.kb.check_consistency(correction.labels)
                        reward = 1.0 if (post_ok and not pre_ok) else 0.0
                    else:
                        reward = 0.0

                    rewards.append(reward)

                rewards_t = torch.tensor(rewards, device=logits.device, dtype=torch.float32)
            else:
                # Single position — consistency doesn't really apply
                rewards_t = torch.zeros(logits.size(0), device=logits.device)

        # Policy gradient
        flags = (ref_scores >= threshold).float()
        return consistency_reward_loss(ref_scores, flags, rewards_t)

    def evaluate(self, test_loader: DataLoader) -> dict:
        """Evaluate model on test data.

        Returns dict with accuracy, reflection_rate, consistency_rate.
        """
        self.model.eval()
        correct = 0
        total = 0
        flag_counts = 0
        flag_total = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                logits, features, ref_scores = self.model(inputs)

                if logits.dim() == 3:
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.numel()
                else:
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                flag_counts += (ref_scores >= self.model.reflection_head.threshold).sum().item()
                flag_total += ref_scores.numel()

        self.model.train()
        return {
            "accuracy": correct / max(total, 1),
            "reflection_rate": flag_counts / max(flag_total, 1),
        }
