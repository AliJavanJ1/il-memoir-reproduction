"""Occupancy and density estimators shared by learner and expert data (theory section 7, ENV-4).

Discount weights (NOTATION section 4, spec section 5)
-----------------------------------------------------
For ``n`` trajectories of ``H`` transitions, the next state ``s^i_{t+1}`` gets
weight ``w_{i,t} = gamma^t / (n Z_H)``: every trajectory has total weight
``1/n`` and the weights sum to one.  The weighted empirical measure is unbiased
for ``d+_{pi,H}`` (ENV-4(a)).  The same weights are used for learner rollouts
and for expert demonstrations (``q_hat_E``).  ER_FB's uniform ``rho_e`` is a
different law and is not built here.

Free-space histogram (fallback and diagnostics)
-----------------------------------------------
``bins x bins`` squares on [0,1]^2 (half-open ``[k/b, (k+1)/b)``).  Bins whose
intersection with the free space ``S`` has zero area are dropped, so every
retained cell has positive free area (U-13).  Masses are discount-weighted
counts; the density of a cell is its mass divided by its free area; the
smoothing reference is ``upsilon_b = |B_b cap S| / |S|`` (masses with masses).

Weighted Gaussian KDE (first continuous candidate)
--------------------------------------------------
``p_hat(x) = sum_i w_i phi_h(x - x_i) 1_S(x) / m_i`` with
``m_i = int_S phi_h(y - x_i) dy``: each kernel is truncated to the free space
and renormalized ("cut and normalize").  Hence ``int_S p_hat = sum_i w_i = 1``
exactly and each trajectory keeps total mass ``1/n``.  ``m_i`` is computed in
closed form with Gaussian CDFs because ``S`` is the unit square minus
axis-aligned wall squares with disjoint interiors.  Truncation is Euclidean: a
kernel ignores walls between its centre and a point, so part of its mass can
lie on free points its centre cannot see (behind a wall or around a doorway
corner).  ``audit/prompt3A/kde_visibility.py`` measures this share for
``h = 0.05``: median 0 and mean 1.1% over a 70x70 grid of centres, 7.3% at the
95th percentile, 16% at a doorway corner and up to about 31% for centres within
about 0.05 L of a wall corner at a doorway entrance.  This is an approximation
made by the estimator (relevant when the pilot chooses ``h``), not a property
of the kernel ``P``.  Smoothing mixes densities with the reference density
``1/|S|`` on ``S``.  A KDE never proves absolute continuity of anything (spec
section 3).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.special import ndtr

from .common import Trajectories

TWO_PI = 2.0 * np.pi


# =============================================================================
# discount weights and finite estimators
# =============================================================================
def discount_weights(n_traj: int, horizon: int, gamma: float) -> np.ndarray:
    """``w[i, t] = gamma^t / (n_traj * Z_H)`` for ``t = 0..H-1`` (weights of ``s^i_{t+1}``)."""
    n_traj, horizon = int(n_traj), int(horizon)
    if n_traj < 1 or horizon < 1 or not 0.0 < gamma < 1.0:
        raise ValueError("n_traj >= 1, horizon >= 1 and 0 < gamma < 1 required")
    g = gamma ** np.arange(horizon, dtype=np.float64)
    return np.tile(g / (g.sum() * n_traj), (n_traj, 1))


def next_state_samples(traj: Trajectories, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Flattened next states ``s_{t+1}`` and their discount weights (equal trajectory weight)."""
    w = discount_weights(traj.n_traj, traj.horizon, gamma)
    nxt = traj.next_states
    points = nxt.reshape((-1,) + nxt.shape[2:])
    return points, w.reshape(-1)


def finite_occupancy(next_states: np.ndarray, n_states: int, gamma: float) -> np.ndarray:
    """Discount-weighted counts of ``s_{t+1}``: unbiased estimate of ``d+_{pi,H}`` (``next_states`` is ``(n, H)``)."""
    next_states = np.asarray(next_states, dtype=np.int64)
    if next_states.ndim != 2:
        raise ValueError("next_states must have shape (n, H)")
    w = discount_weights(next_states.shape[0], next_states.shape[1], gamma)
    return np.bincount(next_states.ravel(), weights=w.ravel(), minlength=n_states)


def finite_state_action_occupancy(
    states: np.ndarray, actions: np.ndarray, n_states: int, n_actions: int, gamma: float
) -> np.ndarray:
    """Estimate of the pre-action ``d^SA_{pi,H}``: weight ``gamma^t / (n Z_H)`` on ``(s_t, a_t)``, ``t < H``."""
    states = np.asarray(states, dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)
    if states.ndim != 2 or actions.shape != (states.shape[0], states.shape[1] - 1):
        raise ValueError("states must be (n, H+1) and actions (n, H)")
    w = discount_weights(actions.shape[0], actions.shape[1], gamma)
    flat = states[:, :-1] * n_actions + actions
    return np.bincount(flat.ravel(), weights=w.ravel(), minlength=n_states * n_actions).reshape(n_states, n_actions)


def uniform_reference(n_cells: int) -> np.ndarray:
    """Uniform reference law on ``n_cells`` finite cells (finite-maze smoothing)."""
    return np.full(int(n_cells), 1.0 / int(n_cells))


# =============================================================================
# free-space geometry
# =============================================================================
def _phi_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``Phi(b) - Phi(a)`` for ``a <= b``, computed in the tail where it is accurate."""
    upper = a > 0
    return np.where(upper, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))


def _rect_intersection(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Intersection of boxes ``r`` (k,4) and ``s`` (w,4) as (k, w, 4); empty boxes have x1 <= x0 or y1 <= y0."""
    lo = np.maximum(r[:, None, :2], s[None, :, :2])
    hi = np.minimum(r[:, None, 2:], s[None, :, 2:])
    return np.concatenate([lo, hi], axis=-1)


class FreeSpace:
    """``S = (0,1)^2`` minus closed axis-aligned wall squares with disjoint interiors."""

    def __init__(self, walls: np.ndarray):
        walls = np.asarray(walls, dtype=np.float64).reshape(-1, 4)
        if not np.all(np.isfinite(walls)):
            raise ValueError("wall coordinates must be finite")
        if np.any(walls[:, :2] < 0) or np.any(walls[:, 2:] > 1) or np.any(walls[:, 2:] <= walls[:, :2]):
            raise ValueError("walls must be non-degenerate boxes inside the unit square")
        if walls.shape[0] > 1:
            inter = _rect_intersection(walls, walls)
            overlap = np.clip(inter[..., 2] - inter[..., 0], 0, None) * np.clip(inter[..., 3] - inter[..., 1], 0, None)
            np.fill_diagonal(overlap, 0.0)
            if np.any(overlap > 1e-15):
                raise ValueError("wall interiors must be disjoint")
        walls.setflags(write=False)
        self.walls = walls
        self.area = float(1.0 - np.sum((walls[:, 2] - walls[:, 0]) * (walls[:, 3] - walls[:, 1])))

    @classmethod
    def from_env(cls, env) -> "FreeSpace":
        """Free space of a :class:`~il_memoir.env_continuous.ContinuousMaze` (uses its ``walls``)."""
        return cls(env.walls)

    @classmethod
    def from_layout(cls, layout) -> "FreeSpace":
        L = 1.0 / layout.size
        walls = [[x * L, y * L, (x + 1) * L, (y + 1) * L] for (x, y) in sorted(layout.blocked)]
        return cls(np.asarray(walls, dtype=np.float64).reshape(-1, 4))

    def contains(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float64)
        flat = pos.reshape(-1, 2)
        inside = np.all((flat > 0.0) & (flat < 1.0), axis=1)
        if self.walls.size:
            x, y = flat[:, 0][:, None], flat[:, 1][:, None]
            w = self.walls
            in_wall = (x >= w[None, :, 0]) & (x <= w[None, :, 2]) & (y >= w[None, :, 1]) & (y <= w[None, :, 3])
            inside &= ~in_wall.any(axis=1)
        return inside.reshape(pos.shape[:-1])

    def rect_free_area(self, rects: np.ndarray) -> np.ndarray:
        """``|R cap S|`` for boxes ``R`` of shape (k, 4): ``|R cap [0,1]^2| - sum_w |R cap W_w|``."""
        rects = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
        unit = np.array([[0.0, 0.0, 1.0, 1.0]])
        inter = _rect_intersection(rects, unit)[:, 0]
        area = np.clip(inter[:, 2] - inter[:, 0], 0, None) * np.clip(inter[:, 3] - inter[:, 1], 0, None)
        if self.walls.size:
            iw = _rect_intersection(rects, self.walls)
            area = area - np.sum(np.clip(iw[..., 2] - iw[..., 0], 0, None) * np.clip(iw[..., 3] - iw[..., 1], 0, None), axis=1)
        return np.clip(area, 0.0, None)

    @staticmethod
    def gaussian_rect_mass(mu: np.ndarray, h: float, rects: np.ndarray) -> np.ndarray:
        """Mass of ``N(mu_i, h^2 I)`` in box ``R_k``: array (m, k); empty boxes give 0."""
        mu = np.asarray(mu, dtype=np.float64).reshape(-1, 2)
        rects = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
        x0 = np.minimum(rects[None, :, 0], rects[None, :, 2])
        x1 = rects[None, :, 2]
        y0 = np.minimum(rects[None, :, 1], rects[None, :, 3])
        y1 = rects[None, :, 3]
        mx, my = mu[:, 0][:, None], mu[:, 1][:, None]
        gx = _phi_diff((x0 - mx) / h, (x1 - mx) / h)
        gy = _phi_diff((y0 - my) / h, (y1 - my) / h)
        empty = (rects[:, 2] <= rects[:, 0]) | (rects[:, 3] <= rects[:, 1])
        return np.where(empty[None, :], 0.0, gx * gy)

    def gaussian_free_mass(self, mu: np.ndarray, h: float) -> np.ndarray:
        """``m_i = int_S phi_h(y - mu_i) dy = G_[0,1]^2(mu_i) - sum_w G_{W_w}(mu_i)`` (exact up to Phi)."""
        unit = np.array([[0.0, 0.0, 1.0, 1.0]])
        mass = self.gaussian_rect_mass(mu, h, unit)[:, 0]
        if self.walls.size:
            mass = mass - self.gaussian_rect_mass(mu, h, self.walls).sum(axis=1)
        return mass

    def grid(self, resolution: int) -> Tuple[np.ndarray, float]:
        """Midpoints of the ``resolution^2`` grid squares whose midpoint is free, and the square area.

        If ``resolution`` is a multiple of the lattice size, every grid square is
        entirely free or entirely inside a wall, so the free squares tile ``S``.
        """
        c = (np.arange(int(resolution)) + 0.5) / resolution
        xx, yy = np.meshgrid(c, c, indexing="ij")
        pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        return pts[self.contains(pts)], 1.0 / resolution**2


# =============================================================================
# histogram
# =============================================================================
class FreeSpaceHistogram:
    """Histogram on the bins of positive free area.  Cells are ordered as ``np.argwhere`` of ``[ix, iy]``."""

    def __init__(self, space: FreeSpace, bins: int = 20, area_tol: float = 1e-14):
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError(f"bins must be a positive integer, got {bins!r}")
        if not np.isfinite(area_tol) or area_tol < 0:
            raise ValueError("area_tol must be finite and non-negative")
        self.space = space
        self.bins = int(bins)
        edges = np.linspace(0.0, 1.0, self.bins + 1)
        ix, iy = np.meshgrid(np.arange(self.bins), np.arange(self.bins), indexing="ij")
        all_rects = np.stack([edges[ix], edges[iy], edges[ix + 1], edges[iy + 1]], axis=-1).reshape(-1, 4)
        area = space.rect_free_area(all_rects).reshape(self.bins, self.bins)
        valid = area > area_tol
        self.edges = edges
        self.free_area_grid = area
        self.valid = valid
        self.cells = np.argwhere(valid)
        index = -np.ones((self.bins, self.bins), dtype=np.int64)
        index[valid] = np.arange(self.cells.shape[0])
        self.index = index
        self.free_area = area[valid]
        self.rects = all_rects.reshape(self.bins, self.bins, 4)[valid]
        self.reference = self.free_area / self.free_area.sum()

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    def cell_ids(self, pos: np.ndarray) -> np.ndarray:
        """Histogram cell of each position; raises for positions outside ``S`` or in a dropped bin."""
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
        if not np.all(self.space.contains(pos)):
            raise ValueError("histogram positions must lie in the free space")
        b = np.clip(np.floor(pos * self.bins).astype(np.int64), 0, self.bins - 1)
        ids = self.index[b[:, 0], b[:, 1]]
        if np.any(ids < 0):
            raise ValueError("a free position fell in a bin of zero free area")
        return ids

    def masses(self, pos: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Weighted counts per cell (equal weights if ``weights`` is None)."""
        ids = self.cell_ids(pos)
        if weights is None:
            weights = np.full(ids.shape[0], 1.0 / ids.shape[0])
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape != ids.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weights must be finite and non-negative, one per position")
        return np.bincount(ids, weights=weights, minlength=self.n_cells)

    def densities(self, masses: np.ndarray) -> np.ndarray:
        """Piecewise-constant density on each cell: mass / free area."""
        return np.asarray(masses, dtype=np.float64) / self.free_area

    def smooth(self, masses: np.ndarray, eps: float) -> np.ndarray:
        """``(1 - eps) masses + eps upsilon`` with the free-area reference law."""
        if not 0.0 <= eps <= 1.0:
            raise ValueError("eps must lie in [0, 1]")
        masses = np.asarray(masses, dtype=np.float64)
        if masses.shape != self.reference.shape or not np.all(np.isfinite(masses)):
            raise ValueError("masses must be finite, one per histogram cell")
        return (1.0 - eps) * masses + eps * self.reference


# =============================================================================
# weighted kernel density estimate
# =============================================================================
class WeightedKDE:
    """Cut-and-normalize weighted Gaussian KDE on the free space ``S``."""

    def __init__(self, space: FreeSpace, points: np.ndarray, weights: np.ndarray, bandwidth: float):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        # finiteness first: every comparison with NaN is false, so NaN would pass the range checks below
        if isinstance(bandwidth, (bool, np.bool_)) or not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError(f"bandwidth must be a finite positive number, got {bandwidth!r}")
        if weights.shape[0] != points.shape[0] or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weights must be finite and non-negative, one per point")
        if abs(weights.sum() - 1.0) > 1e-9:
            raise ValueError("KDE weights must sum to one")
        if not np.all(np.isfinite(points)) or not np.all(space.contains(points)):
            raise ValueError("KDE points must be finite and lie in the free space")
        self.space = space
        self.points = points
        self.weights = weights
        self.bandwidth = float(bandwidth)
        self.norm = space.gaussian_free_mass(points, self.bandwidth)
        if np.any(self.norm <= 0):
            raise ValueError("a kernel has no mass in the free space")
        self._coef = weights / (self.norm * TWO_PI * self.bandwidth**2)

    def density(self, x: np.ndarray, chunk_elems: int = 4_000_000) -> np.ndarray:
        """``p_hat(x)`` for finite positions of shape (k, 2); zero outside ``S``."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
        if not np.all(np.isfinite(x)):
            raise ValueError("KDE query positions must be finite")
        out = np.zeros(x.shape[0])
        inside = self.space.contains(x)
        idx = np.nonzero(inside)[0]
        step = max(1, chunk_elems // max(1, self.points.shape[0]))
        inv = 1.0 / (2.0 * self.bandwidth**2)
        for lo in range(0, idx.size, step):
            sel = idx[lo:lo + step]
            diff = x[sel, None, :] - self.points[None, :, :]
            out[sel] = np.exp(-np.sum(diff * diff, axis=-1) * inv) @ self._coef
        return out

    __call__ = density

    def smoothed_density(self, x: np.ndarray, eps: float) -> np.ndarray:
        """``(1 - eps) p_hat(x) + eps / |S|`` on ``S`` (zero outside): density-with-density smoothing."""
        if not 0.0 <= eps <= 1.0:
            raise ValueError("eps must lie in [0, 1]")
        x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
        inside = self.space.contains(x)
        return np.where(inside, (1.0 - eps) * self.density(x) + eps / self.space.area, 0.0)

    def cell_masses(self, hist: FreeSpaceHistogram) -> np.ndarray:
        """Exact KDE mass of every histogram cell ``B_b cap S`` (sums to one up to Phi rounding)."""
        if hist.space is not self.space and not np.array_equal(hist.space.walls, self.space.walls):
            raise ValueError("histogram and KDE must use the same free space")
        h = self.bandwidth
        scale = self.weights / self.norm
        bin_mass = self.space.gaussian_rect_mass(self.points, h, hist.rects)       # (m, K)
        if self.space.walls.size:
            inter = _rect_intersection(hist.rects, self.space.walls)               # (K, W, 4)
            nonempty = (inter[..., 2] > inter[..., 0]) & (inter[..., 3] > inter[..., 1])
            ks, ws = np.nonzero(nonempty)
            if ks.size:
                wall_mass = self.space.gaussian_rect_mass(self.points, h, inter[ks, ws])  # (m, pairs)
                np.subtract.at(bin_mass.T, ks, wall_mass.T)
        return scale @ bin_mass
