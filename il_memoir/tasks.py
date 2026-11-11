"""Goal geometry and the evaluation reward r_g.  EXPERT / ORACLE TESTS / EVALUATION ONLY.

Learners never import this module (METHOD_MAP section 1): the reward exists
to construct the finite expert and to evaluate policies.

Definitions (configs/environment.yaml, ``task`` block)
------------------------------------------------------
* Goal centre ``g = ((gx + 0.5) / size, (gy + 0.5) / size)`` for goal cell ``(gx, gy)``.
* Reward ``r_g(x) = exp(-|x - g|^2 / (2 w^2))`` with ``w = 0.5 / size`` (half a
  cell); ``0 < r_g <= 1``.  In the finite maze ``x`` is the cell centre.
* Timing: the reward is read at the next state, ``J_H = sum_{t<H} gamma^t r_g(s_{t+1})``
  (NOTATION section 9, OCC-4), exactly as in ``oracle.policy_evaluation`` and
  ``oracle.backward_induction``.
* Goal region ``{x : |x - g| <= rho}`` with ``rho = 0.5 / size``.  In the finite
  maze cell centres are ``1/size`` apart, so the region contains the goal cell only.
* Success (rule ``hit``): some ``s_t`` with ``t in 1..H`` lies in the goal region.
  The rule ``final`` (``s_H`` in the region) is available as a diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import EnvironmentConfig

Cell = Tuple[int, int]


@dataclass(frozen=True)
class GoalTask:
    name: str
    goal_cell: Cell
    grid_size: int
    width: float
    radius: float
    success_rule: str = "hit"

    def __post_init__(self) -> None:
        if not (np.isfinite(self.width) and np.isfinite(self.radius)):
            raise ValueError("width and radius must be finite")
        if self.width <= 0 or self.radius <= 0:
            raise ValueError("width and radius must be positive")
        if self.success_rule not in ("hit", "final"):
            raise ValueError("success_rule must be 'hit' or 'final'")

    @property
    def goal_pos(self) -> np.ndarray:
        return (np.asarray(self.goal_cell, dtype=np.float64) + 0.5) / self.grid_size

    # ------------------------------------------------------------ pointwise
    def reward(self, pos: np.ndarray) -> np.ndarray:
        """r_g at positions of shape ``(..., 2)``; values in (0, 1]."""
        diff = np.asarray(pos, dtype=np.float64) - self.goal_pos
        return np.exp(-np.sum(diff * diff, axis=-1) / (2.0 * self.width**2))

    def in_goal(self, pos: np.ndarray) -> np.ndarray:
        diff = np.asarray(pos, dtype=np.float64) - self.goal_pos
        return np.sqrt(np.sum(diff * diff, axis=-1)) <= self.radius

    def finite_reward(self, maze) -> np.ndarray:
        """Reward vector over the finite states of ``maze`` (a GridMaze): r_g at the cell centres."""
        return self.reward(maze.positions(np.arange(maze.n_states)))

    # ----------------------------------------------------- trajectory level
    @staticmethod
    def _check_positions(positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 3 or positions.shape[-1] != 2 or positions.shape[1] < 2:
            raise ValueError(f"expected positions of shape (n, H+1, 2), got {positions.shape}")
        return positions

    def discounted_returns(self, positions: np.ndarray, gamma: float) -> np.ndarray:
        """``sum_{t=0}^{H-1} gamma^t r_g(s_{t+1})`` for each trajectory (next-state timing)."""
        positions = self._check_positions(positions)
        rewards = self.reward(positions[:, 1:])
        discounts = gamma ** np.arange(rewards.shape[1], dtype=np.float64)
        return rewards @ discounts

    def hitting_times(self, positions: np.ndarray) -> np.ndarray:
        """First ``t in 1..H`` with ``s_t`` in the goal region, or ``-1`` if none."""
        positions = self._check_positions(positions)
        inside = self.in_goal(positions[:, 1:])
        first = np.argmax(inside, axis=1) + 1
        return np.where(inside.any(axis=1), first, -1).astype(np.int64)

    def success(self, positions: np.ndarray) -> np.ndarray:
        """Per-trajectory success under the configured rule (``hit`` or ``final``)."""
        positions = self._check_positions(positions)
        if self.success_rule == "hit":
            return self.in_goal(positions[:, 1:]).any(axis=1)
        return self.in_goal(positions[:, -1])


def make_task(cfg: EnvironmentConfig, name: str) -> GoalTask:
    return GoalTask(
        name=name,
        goal_cell=cfg.goal_cell(name),
        grid_size=cfg.grid_size,
        width=cfg.reward_width,
        radius=cfg.goal_radius,
        success_rule=cfg.success_rule,
    )


def make_tasks(cfg: EnvironmentConfig) -> Dict[str, GoalTask]:
    return {name: make_task(cfg, name) for name in cfg.goal_names}
