"""Continuous maze: positions in [0,1]^2, five discrete actions, segment collision checks.

Geometry
--------
The four-rooms lattice of :mod:`il_memoir.env_tabular` rescaled to the unit
square: cell ``(x, y)`` is the square ``[x L, (x+1) L] x [y L, (y+1) L]`` with
``L = 1/size``.  Blocked cells are **closed** wall squares ``W_c``.  The free
space is the open set

    S = (0, 1)^2 \\ union_c W_c,

of area ``1 - (#blocked) L^2`` (40/49 for four rooms).  Doorways are one cell
(``L = 1/7``, i.e. 14 noise standard deviations) wide.

Kernel (U-5, decided in Prompt 3A; theory section 7, ENV-2)
-----------------------------------------------------------
For ``s`` in ``S`` and action ``a`` with unit displacement ``delta_a``:

1. *Deterministic part.*  ``m = s + step * delta_a``.  If the closed segment
   ``[s, m]`` lies in ``S`` the base point is ``b = m``; otherwise the move is
   blocked and ``b = s`` (stay).  The stay action has ``m = s``.
2. *Noise.*  ``s' = b + xi`` with ``xi ~ N(0, sigma^2 I_2)``, redrawn until the
   closed segment ``[b, s']`` lies in ``S``.  So
   ``P(dx | s, a) = phi_sigma(x - b) 1[[b, x] in S] dx / alpha(b)``, where
   ``alpha(b) >= (Phi(L / (2 sigma)) - 1/2)^2`` (about 1/4).

There is no clipping and no "undo" atom: ``P(. | s, a)`` has a bounded Lebesgue
density on ``S`` (ENV-2).  Both legs ``s -> b`` and ``b -> s'`` avoid every
closed wall square, so no move crosses a wall (no tunnelling), whatever the
step length.  The number of noise draws is capped at ``max_noise_redraws + 1``;
if every draw fails, ``step`` raises ``RuntimeError`` instead of returning a
point.  The output of rejection sampling is independent of the number of draws,
so every completed step has exactly the law above (no atom at ``b``); a step
fails with probability at most ``(1 - alpha_min)^(cap + 1) < 1e-125``.
Visibility is decided by the slab test below: exact in real arithmetic; a
segment within rounding distance of a wall boundary can be misclassified, an
event of negligible probability that the proofs do not cover.

Data access
-----------
:class:`ContinuousMaze` is learner-facing: ``reset``, ``step``, ``rollout`` and
``sample_free_uniform`` return positions only, never a reward.  Observations
are the raw coordinates ``(x, y)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .common import Trajectories, as_policy, sample_actions, PolicyLike
from .config import EnvironmentConfig
from .env_tabular import GridLayout, make_layout

Cell = Tuple[int, int]


def segments_hit_boxes(p0: np.ndarray, p1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """True where the closed segment ``[p0_i, p1_i]`` meets at least one closed box.

    Liang-Barsky slab test.  ``p0``, ``p1`` have shape ``(n, 2)``; ``boxes`` has
    shape ``(W, 4)`` with rows ``[x0, y0, x1, y1]``.  A segment that only
    touches a box boundary counts as meeting it (boxes are closed).
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(-1, 2)
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 2)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    n, w = p0.shape[0], boxes.shape[0]
    if n == 0 or w == 0:
        return np.zeros(n, dtype=bool)
    d = p1 - p0
    t_lo = np.zeros((n, w))
    t_hi = np.ones((n, w))
    for k in (0, 1):
        lo = boxes[None, :, k]
        hi = boxes[None, :, k + 2]
        origin = p0[:, k][:, None]
        dk = d[:, k][:, None]
        parallel = dk == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (lo - origin) / dk
            tb = (hi - origin) / dk
        enter = np.where(parallel, -np.inf, np.minimum(ta, tb))
        leave = np.where(parallel, np.inf, np.maximum(ta, tb))
        outside_slab = parallel & ((origin < lo) | (origin > hi))
        t_lo = np.maximum(t_lo, enter)
        t_hi = np.minimum(t_hi, np.where(outside_slab, -np.inf, leave))
    return np.any(t_lo <= t_hi, axis=1)


@dataclass(frozen=True)
class StepInfo:
    """Diagnostics of one vectorized step: tests and validation only, never learners."""

    base: np.ndarray      # (n, 2) base points b after the deterministic part
    blocked: np.ndarray   # (n,) the deterministic move was blocked (b = s)
    redraws: np.ndarray   # (n,) number of rejected noise draws


class ContinuousMaze:
    """Learner-facing continuous maze.  Positions are float64 arrays of shape ``(n, 2)``."""

    def __init__(
        self,
        layout: GridLayout,
        action_deltas: Sequence[Cell],
        step_length: float,
        noise_std: float,
        start_cell: Cell,
        max_noise_redraws: int = 1000,
        action_names: Optional[Sequence[str]] = None,
    ):
        if not (np.isfinite(step_length) and np.isfinite(noise_std)):
            raise ValueError("step_length and noise_std must be finite")
        if step_length <= 0 or noise_std < 0 or max_noise_redraws < 1:
            raise ValueError("step_length > 0, noise_std >= 0 and max_noise_redraws >= 1 required")
        if not layout.is_free(start_cell):
            raise ValueError(f"start cell {start_cell} is not free in layout {layout.name!r}")
        self.layout = layout
        self.size = layout.size
        self.cell_length = 1.0 / layout.size
        self.step_length = float(step_length)
        self.noise_std = float(noise_std)
        self.max_noise_redraws = int(max_noise_redraws)
        self.action_deltas = tuple((int(dx), int(dy)) for dx, dy in action_deltas)
        self.action_names = tuple(action_names) if action_names is not None else tuple(
            f"a{i}" for i in range(len(self.action_deltas))
        )
        L = self.cell_length
        walls = np.array(
            [[x * L, y * L, (x + 1) * L, (y + 1) * L] for (x, y) in sorted(layout.blocked)], dtype=np.float64
        ).reshape(-1, 4)
        walls.setflags(write=False)
        self.walls = walls
        self.free_area = 1.0 - len(layout.blocked) * L * L
        start = (np.asarray(start_cell, dtype=np.float64) + 0.5) * L
        start.setflags(write=False)
        self.start_pos = start
        disp = np.asarray(self.action_deltas, dtype=np.float64) * self.step_length
        disp.setflags(write=False)
        self._displacements = disp

    @property
    def n_actions(self) -> int:
        return len(self.action_deltas)

    @property
    def obs_dim(self) -> int:
        return 2

    # --------------------------------------------------------------- geometry
    def is_free(self, pos: np.ndarray) -> np.ndarray:
        """Membership in the open free space ``S`` for positions of shape ``(..., 2)``."""
        pos = np.asarray(pos, dtype=np.float64)
        flat = pos.reshape(-1, 2)
        inside = np.all((flat > 0.0) & (flat < 1.0), axis=1)
        if self.walls.size:
            x = flat[:, 0][:, None]
            y = flat[:, 1][:, None]
            w = self.walls
            in_wall = (x >= w[None, :, 0]) & (x <= w[None, :, 2]) & (y >= w[None, :, 1]) & (y <= w[None, :, 3])
            inside &= ~in_wall.any(axis=1)
        return inside.reshape(pos.shape[:-1])

    def segment_clear(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """True where the closed segment ``[p0, p1]`` lies in ``S``.

        The open unit square is convex, so the segment stays in it iff both
        end points do; it avoids the closed walls iff the slab test finds no
        intersection.
        """
        p0 = np.asarray(p0, dtype=np.float64).reshape(-1, 2)
        p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 2)
        inside = np.all((p0 > 0.0) & (p0 < 1.0) & (p1 > 0.0) & (p1 < 1.0), axis=1)
        return inside & ~segments_hit_boxes(p0, p1, self.walls)

    def cell_of(self, pos: np.ndarray) -> np.ndarray:
        """Lattice cell ``floor(size * pos)``, clipped to the grid, shape ``(..., 2)``."""
        cells = np.floor(np.asarray(pos, dtype=np.float64) * self.size).astype(np.int64)
        return np.clip(cells, 0, self.size - 1)

    # --------------------------------------------------------------- dynamics
    def reset(self, n: int = 1) -> np.ndarray:
        """``n`` copies of the start position (the centre of the start cell)."""
        return np.tile(self.start_pos, (int(n), 1))

    def step_with_info(
        self, pos: np.ndarray, actions: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, StepInfo]:
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
        a = np.asarray(actions, dtype=np.int64).reshape(-1)
        if pos.shape[0] != a.shape[0]:
            raise ValueError("positions and actions must have the same length")
        if a.size and (a.min() < 0 or a.max() >= self.n_actions):
            raise ValueError("action index out of range")
        if not np.all(self.is_free(pos)):
            raise ValueError("step called from a position outside the free space")
        n = pos.shape[0]
        disp = self._displacements[a]
        target = pos + disp
        moving = np.any(disp != 0.0, axis=1)
        clear = np.ones(n, dtype=bool)
        if moving.any():
            clear[moving] = self.segment_clear(pos[moving], target[moving])
        base = np.where(clear[:, None], target, pos)
        blocked = moving & ~clear
        redraws = np.zeros(n, dtype=np.int64)
        if self.noise_std == 0.0:
            return base.copy(), StepInfo(base, blocked, redraws)
        nxt = np.empty_like(base)
        pending = np.arange(n)
        attempt = 0
        while True:
            cand = base[pending] + self.noise_std * rng.standard_normal((pending.size, 2))
            ok = self.segment_clear(base[pending], cand)
            nxt[pending[ok]] = cand[ok]
            pending = pending[~ok]
            if pending.size == 0:
                break
            if attempt == self.max_noise_redraws:
                raise RuntimeError(
                    f"noise redraw cap reached: {pending.size} point(s) rejected {attempt + 1} draws in a row "
                    f"(first base point {base[pending[0]].tolist()}); the step is not returned so that every "
                    "completed step has the exact kernel law"
                )
            attempt += 1
            redraws[pending] += 1
        return nxt, StepInfo(base, blocked, redraws)

    def step(self, pos: np.ndarray, actions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample next positions for a batch of (position, action) pairs.  Returns positions only."""
        nxt, _ = self.step_with_info(pos, actions, rng)
        return nxt

    def rollout(
        self,
        policy: PolicyLike,
        n: int,
        horizon: int,
        rng: np.random.Generator,
        start: Optional[np.ndarray] = None,
        record_info: bool = False,
    ):
        """``n`` trajectories of ``horizon`` transitions under a stationary policy.

        ``policy`` maps positions ``(n, 2)`` to action probabilities ``(n, A)``.
        Returns :class:`Trajectories`, or ``(Trajectories, dict)`` with stacked
        :class:`StepInfo` fields when ``record_info`` is true (diagnostics for
        tests and expert validation; learners must not request them).
        """
        pol = as_policy(policy)
        n, horizon = int(n), int(horizon)
        states = np.empty((n, horizon + 1, 2), dtype=np.float64)
        actions = np.empty((n, horizon), dtype=np.int64)
        states[:, 0] = self.reset(n) if start is None else np.asarray(start, dtype=np.float64).reshape(n, 2)
        info = None
        if record_info:
            info = {
                "base": np.empty((n, horizon, 2)),
                "blocked": np.empty((n, horizon), dtype=bool),
                "redraws": np.empty((n, horizon), dtype=np.int64),
            }
        for t in range(horizon):
            probs = pol(states[:, t])
            if probs.shape != (n, self.n_actions):
                raise ValueError(f"policy returned shape {probs.shape}, expected {(n, self.n_actions)}")
            actions[:, t] = sample_actions(probs, rng)
            nxt, step_info = self.step_with_info(states[:, t], actions[:, t], rng)
            states[:, t + 1] = nxt
            if info is not None:
                info["base"][:, t] = step_info.base
                info["blocked"][:, t] = step_info.blocked
                info["redraws"][:, t] = step_info.redraws
        traj = Trajectories(states, actions)
        return (traj, info) if record_info else traj

    def sample_free_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` positions uniform on ``S`` (rejection from the unit square)."""
        out = np.empty((int(n), 2))
        filled = 0
        while filled < n:
            cand = rng.random((max(2 * (n - filled), 16), 2))
            cand = cand[self.is_free(cand)]
            take = min(cand.shape[0], n - filled)
            out[filled:filled + take] = cand[:take]
            filled += take
        return out


def make_continuous_maze(
    cfg: EnvironmentConfig,
    layout: Optional[str] = None,
    noise_std: Optional[float] = None,
    step_cells: Optional[float] = None,
) -> ContinuousMaze:
    """Continuous maze from the configuration (layout, noise and step overridable for tests)."""
    lay = make_layout(cfg, cfg.continuous_layout if layout is None else layout)
    return ContinuousMaze(
        lay,
        cfg.action_deltas,
        (cfg.step_cells if step_cells is None else step_cells) / cfg.grid_size,
        cfg.noise_std if noise_std is None else noise_std,
        cfg.start,
        cfg.max_noise_redraws,
        cfg.action_names,
    )
