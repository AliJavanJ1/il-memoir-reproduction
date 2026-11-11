"""Experts: optimality certificates and ties of the finite oracle expert; validity of the route controller."""
from __future__ import annotations

import numpy as np
import pytest

from il_memoir import oracle
from il_memoir.common import make_rng
from il_memoir.env_tabular import initial_distribution, make_grid_maze, transition_matrix
from il_memoir.expert import (
    expert_occupancy,
    finite_expert,
    generate_demonstrations,
    route_controller,
    validate_route_controller,
)

from _helpers import assert_within_se, per_trajectory_occupancy, random_policy

STAY, RIGHT, UP = 0, 1, 2
L = 1.0 / 7.0
BFS_FROM_START = {"goal_5_1": 4, "goal_5_5": 8, "goal_1_5": 4}


@pytest.mark.parametrize("layout", ["four_rooms", "open"])
def test_finite_expert_certified_and_dominant(cfg, tasks, layout):
    maze = make_grid_maze(cfg, layout)
    P = transition_matrix(maze)
    rng = make_rng(20261005, "expert", layout)
    for name, task in tasks.items():
        ex = finite_expert(maze, task, cfg.gamma, cfg.horizon, cfg.value_iteration_tol, cfg.tie_tol)
        assert np.all((ex.pi == 0) | (ex.pi == 1)) and np.all(ex.pi.sum(axis=1) == 1)
        s = ex.summary()
        assert s["vi_residual"] <= cfg.value_iteration_tol
        assert s["pi_residual"] <= cfg.tie_tol
        assert -1e-10 <= s["H_step_gap_start"] <= s["H_step_gap_bound"]
        assert np.max(np.abs(ex.V_pi - ex.V_star)) <= s["value_iterate_error_bound"] + s["policy_gap_bound"]
        assert ex.actions[maze.index(task.goal_cell)] == STAY
        r = task.finite_reward(maze)
        for _ in range(10):
            _, V = oracle.policy_evaluation(P, random_policy(rng, maze.n_states, 5, alpha=0.3), r, cfg.gamma)
            assert np.all(ex.V_pi >= V - 1e-10)


def test_finite_expert_follows_shortest_paths_and_ties(cfg, tasks):
    """Without slip, the expert computed for the slip-0.1 kernel reaches each goal in the BFS distance and stays.

    The two routes to (5,5) are mirror images, so Q(right) = Q(up) at (1,1): the first maximizer (right) is chosen.
    """
    maze = make_grid_maze(cfg, "four_rooms")
    det = make_grid_maze(cfg, "four_rooms", slip_prob=0.0)
    for name, task in tasks.items():
        ex = finite_expert(maze, task, cfg.gamma, cfg.horizon, cfg.value_iteration_tol, cfg.tie_tol)
        traj = det.rollout(ex.pi, 1, 20, make_rng(0, "det_expert"))
        cells = [det.cell(s) for s in traj.states[0]]
        hit = cells.index(task.goal_cell)
        assert hit == BFS_FROM_START[name]
        assert all(c == task.goal_cell for c in cells[hit:])
        again = finite_expert(maze, task, cfg.gamma, cfg.horizon, cfg.value_iteration_tol, cfg.tie_tol)
        assert np.array_equal(again.actions, ex.actions)
    ex = finite_expert(maze, tasks["goal_5_5"], cfg.gamma, cfg.horizon, cfg.value_iteration_tol, cfg.tie_tol)
    P = transition_matrix(maze)
    Q, _ = oracle.policy_evaluation(P, ex.pi, tasks["goal_5_5"].finite_reward(maze), cfg.gamma)
    s0 = maze.start_state
    assert abs(Q[s0, RIGHT] - Q[s0, UP]) < 1e-12
    assert ex.actions[s0] == RIGHT


def test_finite_expert_demonstrations_match_oracle_target(cfg, tasks, four_rooms):
    """Discount-weighted expert counts from 20 000 rollouts against q_E = d+_{pi*,H} (5 standard errors)."""
    task = tasks["goal_5_5"]
    ex = finite_expert(four_rooms, task, cfg.gamma, cfg.horizon, cfg.value_iteration_tol, cfg.tie_tol)
    n = 20_000
    demos = generate_demonstrations(four_rooms, ex.pi, n, cfg.horizon, make_rng(20261005, "expert", "qE"))
    Y = per_trajectory_occupancy(demos.trajectories.next_states, four_rooms.n_states, cfg.gamma)
    q_E = expert_occupancy(four_rooms, ex)
    assert_within_se(Y.mean(axis=0), q_E, Y.std(axis=0, ddof=1) / np.sqrt(n))
    assert q_E[four_rooms.index(task.goal_cell)] > 0.5


def test_route_controller_is_a_geometric_state_feedback_policy(cfg, cont_env, tasks):
    for task in tasks.values():
        ctrl = route_controller(cont_env.layout, task.goal_cell, cfg.action_deltas)
        dist = ctrl.bfs_distance
        for (x, y) in cont_env.layout.free_cells:
            a = ctrl.cell_actions[x, y]
            if (x, y) == task.goal_cell:
                assert a == STAY
                continue
            dx, dy = cfg.action_deltas[a]
            nb = (x + dx, y + dy)
            assert cont_env.layout.is_free(nb) and dist[nb] == dist[x, y] - 1
        rng = make_rng(1, "ctrl", task.name)
        pts = cont_env.sample_free_uniform(2000, rng)
        acts = ctrl.actions(pts)
        # depends on the current cell only: a second point in the same cell gets the same action
        jitter = (cont_env.cell_of(pts) + rng.uniform(0.01, 0.99, pts.shape)) / 7
        assert np.array_equal(ctrl.actions(jitter), acts)
        # no hidden state: order and repetition do not matter
        perm = rng.permutation(pts.shape[0])
        assert np.array_equal(ctrl.actions(pts[perm]), acts[perm])
        assert np.array_equal(ctrl.actions(pts), acts)
        assert np.allclose(ctrl(pts).sum(axis=1), 1.0)


@pytest.mark.parametrize("goal", ["goal_5_1", "goal_5_5", "goal_1_5"])
def test_route_controller_valid_with_real_rollouts(cfg, cont_env, tasks, goal):
    task = tasks[goal]
    ctrl = route_controller(cont_env.layout, task.goal_cell, cfg.action_deltas)
    stats = validate_route_controller(
        cont_env, ctrl, task, 400, cfg.horizon, cfg.gamma, make_rng(20261005, "validate", goal)
    )
    assert stats["valid"], stats["reasons"]
    assert stats["all_states_free"] and stats["bad_legs_dense_check"] == 0
    assert stats["blocked_planned_moves"] == 0
    assert stats["bfs_distance_start"] == BFS_FROM_START[goal]
    assert stats["hitting_time_mean"] <= BFS_FROM_START[goal] + 1.0
    assert stats["post_hit_in_goal_cell_frac"] >= 0.8          # the goal is held despite noise
    assert stats["success_rate"] >= 0.99


class _AlwaysRight:
    """A position-only controller that walks into walls: its planned moves get blocked."""

    def __call__(self, pos):
        out = np.zeros((pos.shape[0], 5))
        out[:, RIGHT] = 1.0
        return out

    def actions(self, pos):
        return np.full(np.asarray(pos).reshape(-1, 2).shape[0], RIGHT)


class _Counter:
    """Not a stationary policy: the action depends on an internal call counter."""

    def __init__(self, base):
        self.base, self.calls = base, 0

    def __call__(self, pos):
        self.calls += 1
        out = self.base(pos)
        if self.calls % 7 == 0:
            out = np.roll(out, 1, axis=1)
        return out

    def actions(self, pos):
        return self.base.actions(pos)


def test_validator_rejects_bad_controllers_and_kernels(cfg, cont_env, tasks):
    """Negative controls: each validity criterion of validate_route_controller can fail."""
    from il_memoir.env_continuous import ContinuousMaze

    task = tasks["goal_5_5"]
    good = route_controller(cont_env.layout, task.goal_cell, cfg.action_deltas)
    stats = validate_route_controller(cont_env, _AlwaysRight(), task, 50, 30, cfg.gamma, make_rng(1, "neg", "right"))
    assert not stats["valid"] and stats["blocked_planned_moves"] > 0
    assert any("blocked" in r for r in stats["reasons"]) and any("success" in r for r in stats["reasons"])
    stats = validate_route_controller(cont_env, _Counter(good), task, 50, 30, cfg.gamma, make_rng(1, "neg", "counter"))
    assert not stats["valid"] and not stats["actions_recomputed_equal"]

    class EndPointOnly(ContinuousMaze):
        """A wrong kernel that accepts any free end point (no segment check), with large noise."""

        def segment_clear(self, p0, p1):
            return self.is_free(np.asarray(p1).reshape(-1, 2))

    bad_env = EndPointOnly(cont_env.layout, cfg.action_deltas, L, 0.1, cfg.start, 1000)
    stats = validate_route_controller(bad_env, good, task, 200, 30, cfg.gamma, make_rng(1, "neg", "endpoint"))
    assert not stats["valid"] and stats["bad_legs_dense_check"] > 0
    assert any("crosses a wall" in r for r in stats["reasons"])


def test_demonstrations_expose_states_to_learners(cfg, cont_env, tasks, four_rooms):
    task = tasks["goal_5_1"]
    ctrl = route_controller(cont_env.layout, task.goal_cell, cfg.action_deltas)
    a = generate_demonstrations(cont_env, ctrl, 4, cfg.horizon, make_rng(7, "demo"), {"expert": "route"})
    b = generate_demonstrations(cont_env, ctrl, 4, cfg.horizon, make_rng(7, "demo"), {"expert": "route"})
    assert a.trajectories.states.shape == (4, 101, 2) and a.trajectories.actions.shape == (4, 100)
    assert np.array_equal(a.learner_states(), b.learner_states())
    view = a.learner_states()
    view[0, 0] = -1.0
    assert a.trajectories.states[0, 0, 0] > 0           # a copy: learners cannot alter the stored data
    assert a.meta == {"expert": "route", "n_traj": 4, "horizon": 100}
    c = generate_demonstrations(cont_env, ctrl, 4, cfg.horizon, (7, ("demo", "goal_5_1")))
    assert c.meta == {"seed": 7, "stream": ["demo", "goal_5_1"], "n_traj": 4, "horizon": 100}
    assert np.array_equal(c.learner_states(), generate_demonstrations(
        cont_env, ctrl, 4, cfg.horizon, make_rng(7, "demo", "goal_5_1")).learner_states())
    ex = finite_expert(four_rooms, task, cfg.gamma, cfg.horizon, cfg.value_iteration_tol, cfg.tie_tol)
    d = generate_demonstrations(four_rooms, ex.pi, 4, cfg.horizon, make_rng(7, "demo_f"))
    assert d.learner_states().shape == (4, 101) and d.learner_states().dtype == np.int64


def test_demonstrations_rebuilt_from_their_metadata(cfg, cont_env, tasks):
    """Regression (external 3A review): a JSON record of the stream rebuilds the same demonstrations.

    The labels keep their types; the string "2" would select a different stream than the integer 2.
    """
    import json

    task = tasks["goal_5_5"]
    ctrl = route_controller(cont_env.layout, task.goal_cell, cfg.action_deltas)
    demos = generate_demonstrations(cont_env, ctrl, 3, 30, (11, ("demo", np.int64(2), "goal_5_5")))
    meta = json.loads(json.dumps(demos.meta))
    assert meta["stream"] == ["demo", 2, "goal_5_5"] and isinstance(meta["stream"][1], int)
    again = generate_demonstrations(cont_env, ctrl, 3, 30, (meta["seed"], tuple(meta["stream"])))
    assert np.array_equal(again.learner_states(), demos.learner_states())
    as_text = generate_demonstrations(cont_env, ctrl, 3, 30, (11, ("demo", "2", "goal_5_5")))
    assert not np.array_equal(as_text.learner_states(), demos.learner_states())
    for bad in [("demo", 2.0), ("demo", True), "demo"]:
        with pytest.raises(TypeError):
            generate_demonstrations(cont_env, ctrl, 3, 30, (11, bad))
