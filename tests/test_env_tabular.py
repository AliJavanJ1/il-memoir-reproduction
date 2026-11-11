"""Finite mazes: walls and doorways, exact P against the simulator, known policies, learner interface."""
from __future__ import annotations

import numpy as np
import pytest

from il_memoir.common import Trajectories, make_rng
from il_memoir.env_tabular import (
    GridMaze,
    four_rooms_rule_blocked,
    initial_distribution,
    make_grid_maze,
    transition_matrix,
)
from il_memoir import oracle

STAY, RIGHT, UP, LEFT, DOWN = range(5)
DOORWAYS = [(3, 1), (3, 5), (1, 3), (5, 3)]


def det_step(maze: GridMaze, cell, action) -> tuple:
    """One step of the deterministic (slip 0) simulator through its public interface."""
    nxt = maze.step(np.array([maze.index(cell)]), np.array([action]), make_rng(0, "det"))
    return maze.cell(int(nxt[0]))


def test_layout_counts_walls_and_doorways(four_rooms, open_maze):
    assert four_rooms.n_states == 40 and open_maze.n_states == 49
    blocked = four_rooms.layout.blocked
    assert blocked == four_rooms_rule_blocked()
    assert blocked == {(3, 0), (3, 2), (3, 3), (3, 4), (3, 6), (0, 3), (2, 3), (4, 3), (6, 3)}
    for door in DOORWAYS:
        assert four_rooms.layout.is_free(door)
    # state order: by row y, then column x
    cells = four_rooms.cells
    assert list(cells) == sorted(cells, key=lambda c: (c[1], c[0]))
    assert four_rooms.cell(four_rooms.start_state) == (1, 1)


@pytest.mark.parametrize(
    "cell, action, expected",
    [
        ((2, 1), RIGHT, (3, 1)),   # into the lower doorway
        ((3, 1), RIGHT, (4, 1)),   # through it
        ((4, 1), LEFT, (3, 1)),
        ((2, 2), RIGHT, (2, 2)),   # wall (3,2)
        ((2, 0), RIGHT, (2, 0)),   # wall (3,0)
        ((1, 2), UP, (1, 3)),      # into the left doorway
        ((1, 3), UP, (1, 4)),
        ((0, 2), UP, (0, 2)),      # wall (0,3)
        ((2, 2), UP, (2, 2)),      # wall (2,3)
        ((5, 4), DOWN, (5, 3)),    # right doorway from above
        ((4, 4), DOWN, (4, 4)),    # wall (4,3)
        ((3, 5), UP, (3, 5)),      # wall (3,6) above the upper doorway
        ((3, 5), DOWN, (3, 5)),    # wall (3,4) below it
        ((0, 0), LEFT, (0, 0)),    # grid boundary
        ((0, 0), DOWN, (0, 0)),
        ((6, 6), RIGHT, (6, 6)),
        ((6, 6), UP, (6, 6)),
        ((4, 5), STAY, (4, 5)),
    ],
)
def test_four_rooms_moves(cfg, cell, action, expected):
    maze = make_grid_maze(cfg, "four_rooms", slip_prob=0.0)
    assert det_step(maze, cell, action) == expected


def test_open_layout_has_no_internal_walls(cfg):
    maze = make_grid_maze(cfg, "open", slip_prob=0.0)
    assert det_step(maze, (2, 2), RIGHT) == (3, 2)
    assert det_step(maze, (3, 2), UP) == (3, 3)
    assert det_step(maze, (0, 3), DOWN) == (0, 2)


def _distances_by_relaxation(maze: GridMaze, goal) -> np.ndarray:
    """Shortest move counts by Bellman-Ford relaxation over the deterministic simulator (independent of BFS)."""
    n = maze.n_states
    dist = np.full(n, np.inf)
    dist[maze.index(goal)] = 0
    succ = np.array([[maze.index(det_step(maze, c, a)) for a in range(maze.n_actions)] for c in maze.cells])
    for _ in range(n):
        dist = np.minimum(dist, 1 + dist[succ].min(axis=1))
        dist[maze.index(goal)] = 0
    return dist


def _count_shortest_paths(maze: GridMaze, start, goal, dist) -> int:
    succ = {c: {det_step(maze, c, a) for a in range(1, maze.n_actions)} - {c} for c in maze.cells}
    memo = {}

    def count(c):
        if c == goal:
            return 1
        if c not in memo:
            memo[c] = sum(count(nb) for nb in succ[c] if dist[maze.index(nb)] == dist[maze.index(c)] - 1)
        return memo[c]

    return count(start)


@pytest.mark.parametrize("layout", ["four_rooms", "open"])
def test_shortest_paths_match_source_audit(cfg, layout):
    """SOURCE_AUDIT section 10: goals at 4, 8, 4 moves; 4 (four rooms) or 70 (open) shortest paths to (5,5)."""
    maze = make_grid_maze(cfg, layout, slip_prob=0.0)
    for goal, d_expected in [((5, 1), 4), ((5, 5), 8), ((1, 5), 4)]:
        relax = _distances_by_relaxation(maze, goal)
        bfs = maze.layout.bfs_distances(goal, cfg.action_deltas)
        assert relax[maze.index((1, 1))] == d_expected
        for c in maze.cells:
            assert bfs[c] == relax[maze.index(c)]
    relax = _distances_by_relaxation(maze, (5, 5))
    n_paths = _count_shortest_paths(maze, (1, 1), (5, 5), relax)
    assert n_paths == (4 if layout == "four_rooms" else 70)


@pytest.mark.parametrize("layout", ["four_rooms", "open"])
def test_transition_matrix_is_stochastic_and_matches_slip_rule(cfg, layout):
    maze = make_grid_maze(cfg, layout)
    P = transition_matrix(maze)
    assert P.shape == (maze.n_states, 5, maze.n_states)
    assert np.all(P >= 0) and np.allclose(P.sum(axis=2), 1.0, atol=1e-14)
    det = transition_matrix(make_grid_maze(cfg, layout, slip_prob=0.0))
    assert np.all((det == 0) | (det == 1)) and np.all(det.sum(axis=2) == 1)
    assert np.allclose(P, 0.9 * det + 0.1 * det.mean(axis=1, keepdims=True), atol=1e-15)


def test_stay_rows_by_hand(four_rooms, four_rooms_P):
    """Hand values: interior start (1,1) has no blocked move; corner (0,0) has three staying actions."""
    s = four_rooms.index((1, 1))
    row = four_rooms_P[s, STAY]
    assert np.isclose(row[s], 0.9 + 0.1 / 5)
    for nb in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        assert np.isclose(row[four_rooms.index(nb)], 0.1 / 5)
    c = four_rooms.index((0, 0))
    row = four_rooms_P[c, STAY]
    assert np.isclose(row[c], 0.9 + 0.1 * 3 / 5)
    assert np.isclose(row[four_rooms.index((1, 0))], 0.02) and np.isclose(row[four_rooms.index((0, 1))], 0.02)
    # an action into a wall behaves like stay at the doorway cell (3,1) going up
    d = four_rooms.index((3, 1))
    assert np.allclose(four_rooms_P[d, UP], four_rooms_P[d, STAY])


def test_simulator_frequencies_match_P(four_rooms, four_rooms_P):
    """Every (s, a): 4000 simulator draws against the exact row (5 standard errors)."""
    rng = make_rng(20261005, "sim_vs_P")
    n_rep = 4000
    S, A = four_rooms.n_states, four_rooms.n_actions
    s = np.repeat(np.arange(S), A * n_rep)
    a = np.tile(np.repeat(np.arange(A), n_rep), S)
    nxt = four_rooms.step(s, a, rng)
    counts = np.zeros((S, A, S))
    np.add.at(counts, (s, a, nxt), 1.0)
    freq = counts / n_rep
    se = np.sqrt(four_rooms_P * (1 - four_rooms_P) / n_rep)
    assert np.all(np.abs(freq - four_rooms_P) <= 5 * se + 1e-12)
    # impossible transitions never occur
    assert np.all(counts[four_rooms_P == 0] == 0)


def test_known_policy_always_right(cfg):
    """Deterministic dynamics: always-right from (1,1) visits (2,1),(3,1),(4,1),(5,1) then stays at (6,1)."""
    maze = make_grid_maze(cfg, "four_rooms", slip_prob=0.0)
    P = transition_matrix(maze)
    pi = np.zeros((maze.n_states, 5))
    pi[:, RIGHT] = 1.0
    traj = maze.rollout(pi, 3, 10, make_rng(1, "right"))
    path = [maze.cell(s) for s in traj.states[0]]
    assert path == [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)] + [(6, 1)] * 6
    gamma, H = cfg.gamma, cfg.horizon
    d = oracle.occupancy_next_state(P, pi, initial_distribution(maze), gamma, H)
    z_h = oracle.discount_normalizer(gamma, H)
    expected = {(2, 1): 1.0, (3, 1): gamma, (4, 1): gamma**2, (5, 1): gamma**3}
    for c, w in expected.items():
        assert np.isclose(d[maze.index(c)], w / z_h, atol=1e-15)
    assert np.isclose(d[maze.index((6, 1))], sum(gamma**t for t in range(4, H)) / z_h)


def test_learner_interface_has_no_reward_or_transition_matrix(four_rooms):
    names = {n.lower() for n in dir(four_rooms)}
    assert not any("reward" in n for n in names)
    assert "p" not in names and "transition_matrix" not in names
    out = four_rooms.step(np.array([0, 1]), np.array([1, 2]), make_rng(3, "iface"))
    assert isinstance(out, np.ndarray) and out.shape == (2,) and out.dtype == np.int64
    traj = four_rooms.rollout(np.full((40, 5), 0.2), 2, 5, make_rng(3, "iface2"))
    assert isinstance(traj, Trajectories)
    assert not any("reward" in f for f in vars(traj))


def test_rollouts_reproducible_and_observations(four_rooms):
    pi = np.full((40, 5), 0.2)
    a = four_rooms.rollout(pi, 4, 20, make_rng(5, "repro"))
    b = four_rooms.rollout(pi, 4, 20, make_rng(5, "repro"))
    assert np.array_equal(a.states, b.states) and np.array_equal(a.actions, b.actions)
    oh = four_rooms.one_hot(a.states)
    assert oh.shape == (4, 21, 40) and np.all(oh.sum(axis=-1) == 1)
    assert np.allclose(four_rooms.positions(four_rooms.start_state), [1.5 / 7, 1.5 / 7])
    assert np.all(a.states[:, 0] == four_rooms.start_state)
