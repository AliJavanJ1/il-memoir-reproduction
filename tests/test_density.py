"""Density utilities: discount weights, finite counts, free-space histogram and the cut-and-normalize KDE."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from il_memoir.common import Trajectories, make_rng
from il_memoir.density import (
    FreeSpace,
    FreeSpaceHistogram,
    WeightedKDE,
    discount_weights,
    finite_occupancy,
    finite_state_action_occupancy,
    next_state_samples,
)

L = 1.0 / 7.0
GAMMA = 0.95
GRID = 7 * 80          # quadrature grid aligned with the walls (multiple of 7) and the bins (multiple of 20)


def test_discount_weights_signs_and_normalization():
    w = discount_weights(4, 100, GAMMA)
    z_h = (1 - GAMMA**100) / (1 - GAMMA)
    assert w.shape == (4, 100) and np.all(w > 0)
    assert np.isclose(w.sum(), 1.0, atol=1e-14)
    assert np.allclose(w.sum(axis=1), 0.25, atol=1e-15)                 # equal trajectory weight
    assert np.allclose(w[:, 1:] / w[:, :-1], GAMMA)                      # gamma^t, decreasing in t
    assert np.isclose(w[0, 0], 1 / (4 * z_h))
    with pytest.raises(ValueError):
        discount_weights(0, 10, GAMMA)


def test_finite_estimators_by_hand():
    nxt = np.array([[1, 1, 2], [0, 2, 2]])
    est = finite_occupancy(nxt, 3, 0.5)
    z = 1 + 0.5 + 0.25
    expected = np.array([1, 1 + 0.5, 0.25 + 0.5 + 0.25]) / (2 * z)
    assert np.allclose(est, expected) and np.isclose(est.sum(), 1.0)
    states = np.array([[0, 1, 1, 2], [2, 0, 2, 2]])
    actions = np.array([[1, 0, 1], [0, 0, 1]])
    sa = finite_state_action_occupancy(states, actions, 3, 2, 0.5)
    exp_sa = np.zeros((3, 2))
    for i in range(2):
        for t in range(3):
            exp_sa[states[i, t], actions[i, t]] += 0.5**t / (2 * z)
    assert np.allclose(sa, exp_sa) and np.isclose(sa.sum(), 1.0)


@pytest.fixture(scope="module")
def space(cont_env):
    return FreeSpace.from_env(cont_env)


@pytest.fixture(scope="module")
def hist(space):
    return FreeSpaceHistogram(space, bins=20)


def test_histogram_free_areas_against_subpixel_count(space, hist, four_rooms):
    """Exact free areas against an independent 140x140 sub-pixel count (140 = lcm(7, 20): no pixel straddles a wall or bin edge)."""
    assert np.isclose(space.area, 40 / 49)
    assert np.allclose(FreeSpace.from_layout(four_rooms.layout).walls, space.walls)
    n = 140
    c = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(c, c, indexing="ij")
    free = space.contains(np.stack([xx, yy], axis=-1))
    counts = free.reshape(20, 7, 20, 7).sum(axis=(1, 3)) / n**2
    assert np.allclose(hist.free_area_grid, counts, atol=1e-12)
    assert hist.n_cells == int(np.sum(counts > 0)) == 356                # 44 bins lie inside walls
    assert np.isclose(hist.free_area.sum(), space.area)
    assert np.all(hist.free_area > 0)                                   # U-13: every retained cell has free area
    assert np.isclose(hist.free_area.min(), (3 / 20 - 1 / 7) / 20)      # thinnest sliver: (0.15 - 1/7) x 0.05
    assert np.isclose(hist.reference.sum(), 1.0) and hist.reference.min() > 0


def test_histogram_masses_densities_and_smoothing(cont_env, hist):
    rng = make_rng(20261005, "hist")
    pol = lambda pos: np.full((pos.shape[0], 5), 0.2)  # noqa: E731
    traj = cont_env.rollout(pol, 50, 100, rng)
    pts, w = next_state_samples(traj, GAMMA)
    m = hist.masses(pts, w)
    assert np.isclose(m.sum(), 1.0) and np.all(m >= 0)
    assert np.isclose(np.sum(hist.densities(m) * hist.free_area), 1.0)
    ms = hist.smooth(m, 0.01)
    assert np.isclose(ms.sum(), 1.0) and ms.min() >= 0.01 * hist.reference.min() - 1e-16
    # uniform free-space samples never land in a dropped bin and match the free-area reference law
    uni = cont_env.sample_free_uniform(100_000, rng)
    counts = hist.masses(uni)
    assert np.allclose(counts, hist.reference, atol=5 * np.sqrt(hist.reference / 100_000) + 1e-4)
    with pytest.raises(ValueError):
        hist.cell_ids(np.array([[3.5 * L, 0.5 * L]]))


@pytest.fixture(scope="module")
def kde(cont_env, space):
    rng = make_rng(20261005, "kde")
    interior = cont_env.sample_free_uniform(150, rng)
    near_walls = np.array(
        [[3 * L - 1e-3, 3 * L - 1e-3], [1e-3, 1e-3], [3 * L - 1e-3, 2.5 * L], [4 * L + 1e-3, 1.5 * L], [5.5 * L, 1 - 1e-3]]
    )
    pts = np.vstack([interior, near_walls])
    w = rng.dirichlet(np.ones(pts.shape[0]))
    return WeightedKDE(space, pts, w, 0.05)


@pytest.fixture(scope="module")
def kde_grid(kde, space):
    grid, area = space.grid(GRID)
    return grid, area, kde.density(grid)


def test_kde_integrates_to_one_on_free_space(kde, kde_grid):
    grid, area, dens = kde_grid
    assert abs(dens.sum() * area - 1.0) < 1e-4
    assert abs(kde.smoothed_density(grid, 0.01).sum() * area - 1.0) < 1e-4
    outside = np.array([[3.5 * L, 3.5 * L], [0.5, 0.5], [-0.1, 0.5], [0.5, 1.0]])
    assert np.all(kde.density(outside) == 0.0)


def test_kde_cell_masses_exact_and_consistent(kde, kde_grid, hist):
    cm = kde.cell_masses(hist)
    assert abs(cm.sum() - 1.0) < 1e-10                 # exact Gaussian-CDF masses of the truncated kernels
    assert np.all(cm >= -1e-15)
    grid, area, dens = kde_grid
    quad = np.bincount(hist.cell_ids(grid), weights=dens * area, minlength=hist.n_cells)
    assert np.max(np.abs(quad - cm)) < 1e-4


def test_kde_normalizers_against_quadrature_and_geometry(space):
    pts = np.array(
        [
            [1.5 * L, 1.5 * L],                 # room centre: almost no truncation
            [3 * L - 1e-9, 3 * L - 1e-9],       # inside corner of a room: about a quadrant
            [1e-9, 1.5 * L],                    # long straight outer boundary: half plane
            [3 * L + 1e-9, 1.5 * L],            # doorway (3,1), next to the room boundary line
            [5.5 * L, 1 - 1e-9],                # top boundary
        ]
    )
    h = 0.05
    m = space.gaussian_free_mass(pts, h)
    grid, area = space.grid(GRID)
    d2 = ((grid[None, :, :] - pts[:, None, :]) ** 2).sum(axis=-1)
    quad = np.exp(-d2 / (2 * h * h)).sum(axis=1) * area / (2 * np.pi * h * h)
    assert np.allclose(m, quad, atol=5e-4)              # closed form vs independent quadrature
    assert m[0] > 0.9999 and abs(m[1] - 0.25) < 5e-3 and abs(m[2] - 0.5) < 1e-3
    rng = make_rng(3, "norm")
    sample = rng.uniform(0, 1, (4000, 2))
    sample = sample[space.contains(sample)]
    lower = (ndtr(L / (2 * h)) - 0.5) ** 2              # ENV-4(b): quadrant inside the point's own cell
    assert np.all(space.gaussian_free_mass(sample, h) >= lower - 1e-12)


def test_kde_keeps_equal_trajectory_weight(space):
    """Two stationary trajectories in different rooms: each keeps total KDE mass 1/2 in its own room."""
    a = np.tile([1.5 * L, 1.5 * L], (101, 1))
    b = np.tile([5.5 * L, 5.5 * L], (101, 1))
    traj = Trajectories(np.stack([a, b]), np.zeros((2, 100), dtype=np.int64))
    pts, w = next_state_samples(traj, GAMMA)
    k = WeightedKDE(space, pts, w, 0.05)
    hist = FreeSpaceHistogram(space, bins=20)
    cm = k.cell_masses(hist)
    centres = (hist.cells + 0.5) / 20
    lower_left = np.all(centres < 3 * L, axis=1)
    assert abs(cm[lower_left].sum() - 0.5) < 1e-5


def test_kde_recovers_uniform_density(cont_env, space):
    rng = make_rng(4, "uniform_kde")
    pts = cont_env.sample_free_uniform(50_000, rng)
    k = WeightedKDE(space, pts, np.full(pts.shape[0], 1 / pts.shape[0]), 0.05)
    probe = np.array([[1.5 * L, 1.5 * L], [5.5 * L, 5.5 * L], [1.5 * L, 5.5 * L], [5.5 * L, 1.5 * L]])
    assert np.allclose(k.density(probe), 1 / space.area, rtol=0.1)


def test_finite_smoothed_objectives_and_signals_from_real_rollouts(cfg, cont_env, tasks, space, hist):
    """Expert demonstrations against learner rollouts: smoothed objectives and plug-in signals are finite."""
    from il_memoir import divergences as dv
    from il_memoir.expert import generate_demonstrations, route_controller

    task = tasks["goal_5_5"]
    ctrl = route_controller(cont_env.layout, task.goal_cell, cfg.action_deltas)
    demos = generate_demonstrations(cont_env, ctrl, 4, cfg.horizon, make_rng(20261005, "e2e", "demos"))
    pol = lambda pos: np.full((pos.shape[0], 5), 0.2)  # noqa: E731
    learner = cont_env.rollout(pol, 10, cfg.horizon, make_rng(20261005, "e2e", "learner"))
    eps = cfg.smoothing_eps
    q_pts, q_w = next_state_samples(demos.trajectories, cfg.gamma)
    d_pts, d_w = next_state_samples(learner, cfg.gamma)
    q_mass, d_mass = hist.masses(q_pts, q_w), hist.masses(d_pts, d_w)
    assert np.isinf(dv.divergence(d_mass, q_mass, "kl_dq"))     # the learner visits cells the expert never visits
    q_kde = WeightedKDE(space, q_pts, q_w, cfg.kde_bandwidth)
    d_kde = WeightedKDE(space, d_pts, d_w, cfg.kde_bandwidth)
    grid, area = space.grid(7 * 20)
    d_at, q_at = d_kde.density(d_pts), q_kde.density(d_pts)
    d_grid, q_grid = d_kde.smoothed_density(grid, eps), q_kde.smoothed_density(grid, eps)
    for name in ("js", "kl_dq", "kl_qd"):
        value = dv.smoothed_divergence(d_mass, q_mass, name, eps, hist.reference)
        assert np.isfinite(value) and value > 0
        signal = dv.smoothed_signal(d_mass, q_mass, name, eps, hist.reference)
        assert np.all(np.isfinite(signal))
        kde_signal = dv.smoothed_signal(d_at, q_at, name, eps, 1.0 / space.area)
        assert np.all(np.isfinite(kde_signal))
        dens_value = dv.divergence_from_densities(d_grid, q_grid, area, name)
        assert np.isfinite(dens_value) and dens_value > 0
    assert dv.smoothed_divergence(d_mass, q_mass, "js", eps, hist.reference) <= np.log(2)


def test_kde_input_validation(space):
    with pytest.raises(ValueError):
        WeightedKDE(space, np.array([[0.2, 0.2]]), np.array([0.5]), 0.05)            # weights do not sum to 1
    with pytest.raises(ValueError):
        WeightedKDE(space, np.array([[3.5 * L, 0.5 * L]]), np.array([1.0]), 0.05)     # point inside a wall
    with pytest.raises(ValueError):
        WeightedKDE(space, np.array([[0.2, 0.2]]), np.array([1.0]), 0.0)


TWO_FREE_POINTS = np.array([[0.2, 0.2], [0.1, 0.1]])
NONFINITE_INPUTS = {
    # regression (external 3A review): NaN passes every range check, because each comparison with NaN is false
    "KDE bandwidth NaN": lambda s, h: WeightedKDE(s, TWO_FREE_POINTS, np.array([0.5, 0.5]), np.nan),
    "KDE bandwidth inf": lambda s, h: WeightedKDE(s, TWO_FREE_POINTS, np.array([0.5, 0.5]), np.inf),
    "KDE weight NaN": lambda s, h: WeightedKDE(s, TWO_FREE_POINTS, np.array([np.nan, 1.0]), 0.05),
    "KDE point NaN": lambda s, h: WeightedKDE(s, np.array([[np.nan, 0.2]]), np.array([1.0]), 0.05),
    "KDE query NaN": lambda s, h: WeightedKDE(s, TWO_FREE_POINTS, np.array([0.5, 0.5]), 0.05).density(
        np.array([[np.nan, 0.5]])
    ),
    "wall NaN": lambda s, h: FreeSpace(np.array([[0.1, 0.1, np.nan, 0.2]])),
    "histogram bins 0": lambda s, h: FreeSpaceHistogram(s, bins=0),
    "histogram bins 20.5": lambda s, h: FreeSpaceHistogram(s, bins=20.5),
    "histogram weight NaN": lambda s, h: h.masses(TWO_FREE_POINTS, np.array([np.nan, 1.0])),
    "smoothed masses NaN": lambda s, h: h.smooth(np.full(h.n_cells, np.nan), 0.01),
    "smoothing eps NaN": lambda s, h: h.smooth(h.reference, np.nan),
}


@pytest.mark.parametrize("case", sorted(NONFINITE_INPUTS))
def test_density_rejects_nonfinite_inputs(space, hist, case):
    with pytest.raises(ValueError):
        NONFINITE_INPUTS[case](space, hist)
