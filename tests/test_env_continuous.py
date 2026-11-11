"""Continuous maze: geometry, segment collisions, tunnelling, boundaries, noise and acceptance."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from il_memoir.common import Trajectories, make_rng
from il_memoir.env_continuous import make_continuous_maze, segments_hit_boxes

L = 1.0 / 7.0
STAY, RIGHT, UP, LEFT, DOWN = range(5)


def centre(cell):
    return (np.asarray(cell, dtype=np.float64) + 0.5) * L


def test_walls_free_area_and_cell_centres(cont_env, four_rooms):
    assert cont_env.walls.shape == (9, 4)
    assert np.isclose(cont_env.free_area, 40 / 49)
    expected = {tuple(np.round([x * L, y * L, (x + 1) * L, (y + 1) * L], 12)) for x, y in four_rooms.layout.blocked}
    assert {tuple(np.round(w, 12)) for w in cont_env.walls} == expected
    for x in range(7):
        for y in range(7):
            assert bool(cont_env.is_free(centre((x, y)))) == four_rooms.layout.is_free((x, y))
    assert np.allclose(cont_env.start_pos, centre((1, 1)))
    # walls are closed: their boundary is not free; the open unit square excludes its boundary
    assert not cont_env.is_free(np.array([3 * L, 2.5 * L]))
    assert not cont_env.is_free(np.array([0.0, 0.5])) and not cont_env.is_free(np.array([0.5, 1.0]))


def test_segment_box_cases():
    box = np.array([[0.4, 0.4, 0.6, 0.6]])
    cases = [
        ((0.3, 0.5), (0.7, 0.5), True),     # crosses
        ((0.3, 0.3), (0.4, 0.4), True),     # touches the closed corner
        ((0.3, 0.3), (0.39, 0.39), False),  # stops short
        ((0.3, 0.7), (0.7, 0.7), False),    # parallel, outside the slab
        ((0.4, 0.3), (0.4, 0.35), False),   # inside the x-slab, outside in y
        ((0.5, 0.5), (0.5, 0.5), True),     # zero length inside
        ((0.2, 0.2), (0.2, 0.2), False),    # zero length outside
        ((0.4, 0.2), (0.4, 0.8), True),     # along an edge
        ((0.65, 0.3), (0.35, 0.7), True),   # oblique through the box
        ((0.62, 0.3), (0.9, 0.5), False),   # oblique, misses
    ]
    p0 = np.array([c[0] for c in cases])
    p1 = np.array([c[1] for c in cases])
    got = segments_hit_boxes(p0, p1, box)
    assert list(got) == [c[2] for c in cases]
    assert not segments_hit_boxes(p0, p1, np.zeros((0, 4))).any()


@pytest.mark.parametrize(
    "cell, action, expected",
    [
        ((2, 1), RIGHT, (3, 1)),
        ((3, 1), RIGHT, (4, 1)),
        ((2, 2), RIGHT, (2, 2)),   # wall (3,2): blocked, stays
        ((1, 2), UP, (1, 3)),
        ((1, 3), UP, (1, 4)),
        ((0, 2), UP, (0, 2)),      # wall (0,3)
        ((0, 0), LEFT, (0, 0)),    # boundary
        ((0, 0), DOWN, (0, 0)),
        ((6, 6), UP, (6, 6)),
        ((5, 5), STAY, (5, 5)),
    ],
)
def test_deterministic_moves_match_finite_maze(cfg, cell, action, expected):
    env = make_continuous_maze(cfg, noise_std=0.0)
    nxt, info = env.step_with_info(centre(cell)[None, :], np.array([action]), make_rng(0, "det"))
    assert np.allclose(nxt[0], centre(expected), atol=1e-15)
    assert bool(info.blocked[0]) == (expected == cell and action != STAY)


def test_segment_check_prevents_tunnelling_with_long_steps(cfg):
    """A 2-cell step from (2,2) would end in the free cell (4,2) behind the wall (3,2): it must be blocked."""
    env = make_continuous_maze(cfg, noise_std=0.0, step_cells=2.0)
    start = centre((2, 2))[None, :]
    nxt, info = env.step_with_info(start, np.array([RIGHT]), make_rng(0, "long"))
    assert env.is_free(centre((4, 2)))            # an end-point-only check would accept the move
    assert info.blocked[0] and np.allclose(nxt[0], start[0])
    # through the doorway row the same long step is allowed: (1,1) -> (3,1)
    nxt, info = env.step_with_info(centre((1, 1))[None, :], np.array([RIGHT]), make_rng(0, "long2"))
    assert not info.blocked[0] and np.allclose(nxt[0], centre((3, 1)))


@pytest.mark.parametrize(
    "axis, side, action",
    [(0, -1, RIGHT), (0, +1, LEFT), (1, -1, UP), (1, +1, DOWN)],
)
def test_noisy_moves_never_cross_walls(cont_env, axis, side, action):
    """Starts within 0.03 of a wall face, pushed into the wall, with noise: nobody reaches the other side.

    Starts are kept within the middle tenth of a wall row (at least 6.4 noise standard deviations
    from a doorway row): next to a doorway corner, noise can legitimately carry the agent around the
    corner into the doorway (see the 5/8 acceptance case below), which is not tunnelling.
    """
    rng = make_rng(20261005, "tunnel", axis, "low" if side < 0 else "high")
    n = 20_000
    wall_lo, wall_hi = 3 * L, 4 * L
    depth = rng.uniform(1e-9, 0.03, n)
    face = wall_lo - depth if side < 0 else wall_hi + depth
    rows = rng.choice([0, 2, 4, 6], size=n)          # rows/columns where the wall is solid
    along = (rows + rng.uniform(0.45, 0.55, n)) * L
    pos = np.empty((n, 2))
    pos[:, axis] = face
    pos[:, 1 - axis] = along
    assert cont_env.is_free(pos).all()
    nxt, info = cont_env.step_with_info(pos, np.full(n, action), rng)
    assert info.blocked.all()
    if side < 0:
        assert np.all(nxt[:, axis] < wall_lo)
    else:
        assert np.all(nxt[:, axis] > wall_hi)
    assert cont_env.is_free(nxt).all()
    # U-5: a blocked move becomes "stay" and the noise is still applied (no undo atom at s):
    # the component along the wall is an untruncated N(0, 0.01^2) step.
    assert np.all(np.any(nxt != pos, axis=1))
    along_step = nxt[:, 1 - axis] - pos[:, 1 - axis]
    assert abs(along_step.mean()) <= 5 * 0.01 / np.sqrt(n)
    assert abs(along_step.std() - 0.01) <= 5e-4
    # the normal component is the Gaussian truncated at the wall face: the step towards the wall stays below the depth
    towards_wall = (nxt[:, axis] - pos[:, axis]) * (-side)
    assert np.all(towards_wall < depth)
    assert np.mean(towards_wall < 0) > 0.5           # truncation removes part of the half towards the wall


def test_boundary_moves_blocked_and_states_stay_inside(cont_env):
    rng = make_rng(1, "boundary")
    n = 5000
    pos = np.column_stack([rng.uniform(1e-6, 0.02, n), rng.uniform(0.05, 0.95, n) * L])  # left strip of cell (0,0)
    nxt, info = cont_env.step_with_info(pos, np.full(n, LEFT), rng)
    assert info.blocked.all() and np.all(nxt[:, 0] > 0.0) and cont_env.is_free(nxt).all()
    corners = np.array([[1e-6, 1e-6], [1 - 1e-6, 1e-6], [1e-6, 1 - 1e-6], [1 - 1e-6, 1 - 1e-6]])
    corners = np.repeat(corners, 2000, axis=0)
    nxt = cont_env.step(corners, np.zeros(corners.shape[0], dtype=np.int64), rng)
    assert np.all((nxt > 0) & (nxt < 1))


def test_noise_statistics_away_from_walls(cont_env):
    """Far from walls the step is the exact displacement plus N(0, 0.01^2 I) (no redraws)."""
    rng = make_rng(2, "noise")
    n = 40_000
    start = np.tile(centre((5, 5)), (n, 1))
    for action, disp in [(STAY, (0.0, 0.0)), (DOWN, (0.0, -L))]:
        nxt, info = cont_env.step_with_info(start, np.full(n, action), rng)
        delta = nxt - start - np.array(disp)
        assert info.redraws.sum() == 0 and not info.blocked.any()
        assert np.all(np.abs(delta.mean(axis=0)) <= 5 * 0.01 / np.sqrt(n))
        assert np.all(np.abs(delta.std(axis=0) - 0.01) <= 0.0003)
        assert abs(np.corrcoef(delta.T)[0, 1]) < 0.03


@pytest.mark.parametrize(
    "point, expected_acceptance",
    [
        ((3 * L - 1e-7, 3 * L - 1e-7), 0.25),   # inside corner of the lower-left room (quadrant)
        ((1e-7, 1e-7), 0.25),                   # corner of the unit square
        ((3 * L - 1e-7, 2.5 * L), 0.5),         # straight wall face
        # doorway (3,1) at the convex corner of wall (3,0): the upper half-plane is visible (1/2), the
        # wall quadrant is not, and only the half of the lower-left quadrant whose segments pass left of
        # the corner is visible (1/8): 5/8.  An end-point-only check would give 3/4.
        ((3 * L + 1e-7, 1 * L + 1e-7), 0.625),
    ],
)
def test_noise_acceptance_and_lower_bound(cont_env, point, expected_acceptance):
    """Acceptance alpha(b) of the noise redraw (ENV-2): matches the local geometry and its lower bound."""
    rng = make_rng(3, "accept", int(expected_acceptance * 100))
    n = 40_000
    base = np.tile(point, (n, 1))
    assert cont_env.is_free(base).all()
    nxt, info = cont_env.step_with_info(base, np.zeros(n, dtype=np.int64), rng)
    acceptance = n / (n + info.redraws.sum())
    assert abs(acceptance - expected_acceptance) < 0.01
    lower = (ndtr(L / (2 * 0.01)) - 0.5) ** 2
    assert acceptance >= lower - 0.01
    assert cont_env.is_free(nxt).all()


def test_redraw_cap_raises_instead_of_returning_an_atom(cfg):
    """With a cap of 2 draws at an inside corner (acceptance 1/4), some step fails: it must raise, not return b."""
    from il_memoir.env_continuous import ContinuousMaze
    from il_memoir.env_tabular import make_layout

    env = ContinuousMaze(make_layout(cfg, "four_rooms"), cfg.action_deltas, L, 0.01, cfg.start, max_noise_redraws=1)
    base = np.tile([3 * L - 1e-7, 3 * L - 1e-7], (200, 1))
    with pytest.raises(RuntimeError, match="cap"):
        env.step(base, np.zeros(200, dtype=np.int64), make_rng(6, "cap"))


@pytest.mark.parametrize("step, noise", [(np.nan, 0.01), (np.inf, 0.01), (L, np.nan), (L, np.inf)])
def test_kernel_parameters_must_be_finite(cfg, cont_env, step, noise):
    from il_memoir.env_continuous import ContinuousMaze

    with pytest.raises(ValueError, match="finite"):
        ContinuousMaze(cont_env.layout, cfg.action_deltas, step, noise, cfg.start, 1000)


def test_reset_rollout_interface_and_reproducibility(cont_env):
    names = {n.lower() for n in dir(cont_env)}
    assert not any("reward" in n for n in names)
    assert np.allclose(cont_env.reset(3), np.tile(centre((1, 1)), (3, 1)))
    pol = lambda pos: np.full((pos.shape[0], 5), 0.2)  # noqa: E731
    a = cont_env.rollout(pol, 5, 30, make_rng(4, "roll"))
    b = cont_env.rollout(pol, 5, 30, make_rng(4, "roll"))
    assert isinstance(a, Trajectories) and a.states.shape == (5, 31, 2) and a.actions.shape == (5, 30)
    assert np.array_equal(a.states, b.states) and np.array_equal(a.actions, b.actions)
    assert cont_env.is_free(a.states).all()
    with pytest.raises(ValueError):
        cont_env.step(np.array([[3.5 * L, 0.5 * L]]), np.array([0]), make_rng(4, "bad"))


def test_uniform_free_sampler(cont_env):
    rng = make_rng(5, "uniform")
    pts = cont_env.sample_free_uniform(40_000, rng)
    assert cont_env.is_free(pts).all()
    cells = cont_env.cell_of(pts)
    lower_left_room = np.all(cells <= 2, axis=1).mean()
    door = np.all(cells == [3, 1], axis=1).mean()
    assert abs(lower_left_room - 9 / 40) < 0.01 and abs(door - 1 / 40) < 0.004
