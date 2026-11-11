"""Evaluation task definitions pinned by hand values: reward, goal region, hitting times, success, returns."""
from __future__ import annotations

import numpy as np
import pytest

from il_memoir.tasks import GoalTask

L = 1.0 / 7.0


@pytest.fixture()
def task(tasks):
    return tasks["goal_5_5"]


def test_reward_values(task, four_rooms):
    g = task.goal_pos
    assert np.allclose(g, [5.5 * L, 5.5 * L])
    assert task.reward(g) == 1.0
    assert np.isclose(task.reward(g + [L, 0.0]), np.exp(-2.0))          # adjacent cell centre (ENV-1(c))
    assert np.isclose(task.reward(g + [L, L]), np.exp(-4.0))            # diagonal cell centre
    assert np.isclose(task.reward(g + [0.0, 0.5 * L]), np.exp(-0.5))    # one width w = L/2 away
    r = task.finite_reward(four_rooms)
    assert r.shape == (40,) and np.all(r > 0) and r.max() == 1.0
    assert r.argmax() == four_rooms.index((5, 5))
    others = np.delete(r, four_rooms.index((5, 5)))
    assert others.max() <= np.exp(-2.0) + 1e-15


def test_goal_region_boundary(task):
    g = task.goal_pos
    assert np.isclose(task.radius, 0.5 * L)
    assert task.in_goal(g + [0.5 * L * (1 - 1e-12), 0.0])
    assert task.in_goal(g + [0.0, -0.5 * L * (1 - 1e-12)])
    assert not task.in_goal(g + [0.5 * L * (1 + 1e-9), 0.0])
    assert not task.in_goal(g + [0.4 * L, 0.4 * L])                     # inside the cell, outside the disc


@pytest.mark.parametrize("width, radius", [(np.nan, 0.5 * L), (np.inf, 0.5 * L), (0.5 * L, np.nan), (0.5 * L, np.inf)])
def test_task_parameters_must_be_finite(task, width, radius):
    with pytest.raises(ValueError, match="finite"):
        GoalTask("bad", task.goal_cell, 7, width, radius)


def test_hitting_times_success_and_returns(task):
    g = task.goal_pos
    far = g - [4 * L, 4 * L]
    # trajectory A: s_0 far, enters the region at t = 3, leaves at t = 5
    a = np.array([far, far, far, g, g, far])
    # trajectory B: never enters
    b = np.array([far] * 6)
    # trajectory C: starts in the region (s_0 does not count), enters again only at the final step t = 5
    c = np.array([g, far, far, far, far, g])
    pos = np.stack([a, b, c])
    assert task.hitting_times(pos).tolist() == [3, -1, 5]
    assert task.success(pos).tolist() == [True, False, True]
    final = GoalTask("goal", task.goal_cell, 7, task.width, task.radius, success_rule="final")
    assert final.success(pos).tolist() == [False, False, True]
    gamma = 0.9
    r_far = task.reward(far)
    expected_a = sum(gamma**t * (1.0 if t + 1 in (3, 4) else r_far) for t in range(5))
    assert np.isclose(task.discounted_returns(pos, gamma)[0], expected_a)   # reward read at s_{t+1}
    expected_c = sum(gamma**t * (1.0 if t + 1 == 5 else r_far) for t in range(5))
    assert np.isclose(task.discounted_returns(pos, gamma)[2], expected_c)   # s_0 is never rewarded
    with pytest.raises(ValueError):
        task.discounted_returns(pos[:, :, 0], gamma)
