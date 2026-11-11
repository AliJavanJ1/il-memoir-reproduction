"""Expert construction: the finite oracle expert and the continuous route controller.

Finite expert (METHOD_MAP section 7, ``expert.finite_expert``)
--------------------------------------------------------------
The stationary deterministic optimal policy ``pi*`` for the evaluation reward
``r_g`` (next-state timing) and the exact kernel ``P``, from value iteration
refined by exact policy iteration.  Ties are broken by the first maximizer in
the configured action order (within ``tie_tol``).  The expert uses ``P`` and
``r_g``; this is the documented expert-construction exception of METHOD_MAP
section 1.  No exploration noise is added to the expert's actions; the only
randomness in its demonstrations is the environment's slip.

Continuous expert (``expert.route_controller``)
-----------------------------------------------
A constructed reference controller, not a proven optimal policy.  It is a
stationary state-feedback policy: the action is a deterministic function of the
current lattice cell ``floor(7 x)`` of the position ``x`` (no time index, no
history, no waypoint counter), so it lies in the learners' Markov policy class
(SOURCE_AUDIT A4, spec S3).  In each cell it plays the first action, in the
configured order, that moves to a free neighbouring cell with a smaller
breadth-first-search distance to the goal cell; in the goal cell it stays.
It uses only the maze geometry: no reward, no transition matrix.  It acts
through the same five actions and the same noisy kernel as the learners.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from . import oracle
from .common import Demonstrations, PolicyLike, make_rng, stream_labels
from .env_continuous import ContinuousMaze
from .env_tabular import GridLayout, GridMaze, initial_distribution, transition_matrix
from .tasks import GoalTask

Cell = Tuple[int, int]


# =============================================================================
# finite oracle expert
# =============================================================================
@dataclass(frozen=True)
class FiniteExpert:
    task_name: str
    layout_name: str
    pi: np.ndarray            # (S, A) deterministic, one-hot rows
    actions: np.ndarray       # (S,)
    V_star: np.ndarray        # value-iteration iterate (residual vi_residual)
    V_pi: np.ndarray          # exact value of pi
    vi_residual: float
    vi_iterations: int
    pi_residual: float
    V_H_star: np.ndarray      # H-step optimum (backward induction)
    V_pi_H: np.ndarray        # H-step value of the stationary expert
    start_state: int
    gamma: float
    horizon: int
    reward_max: float

    def summary(self) -> Dict[str, float]:
        """Scalars at the start state, with the residual certificates of theory Lemma 1.16."""
        s0 = self.start_state
        gap_h = float(self.V_H_star[s0] - self.V_pi_H[s0])
        return {
            "V_star_vi_start": float(self.V_star[s0]),
            "V_pi_start": float(self.V_pi[s0]),
            "vi_residual": self.vi_residual,
            "vi_iterations": float(self.vi_iterations),
            "value_iterate_error_bound": self.vi_residual / (1.0 - self.gamma),
            "pi_residual": self.pi_residual,
            "policy_gap_bound": self.pi_residual / (1.0 - self.gamma),
            "V_H_star_start": float(self.V_H_star[s0]),
            "V_pi_H_start": float(self.V_pi_H[s0]),
            "H_step_gap_start": gap_h,
            # Prop. 1.17(c) for r >= 0, plus eta_pi / (1 - gamma) because pi is certified, not exact
            "H_step_gap_bound": (self.gamma**self.horizon * self.reward_max + self.pi_residual) / (1.0 - self.gamma),
        }


def finite_expert(
    maze: GridMaze,
    task: GoalTask,
    gamma: float,
    horizon: int,
    vi_tol: float = 1e-12,
    tie_tol: float = 1e-10,
    layout_name: Optional[str] = None,
) -> FiniteExpert:
    """Oracle expert ``pi*`` for ``task`` in ``maze`` (expert construction may read ``P`` and ``r_g``).

    Raises ``RuntimeError`` unless value iteration converged and the refined
    policy is ``tie_tol``-greedy for its own ``Q`` (``eta_pi <= tie_tol``).
    """
    P = transition_matrix(maze)
    r = task.finite_reward(maze)
    vi = oracle.value_iteration(P, r, gamma, tol=vi_tol, tie_tol=tie_tol)
    if not (vi.converged and vi.refinement_converged and vi.pi_residual <= tie_tol + 1e-12):
        raise RuntimeError(
            f"finite expert not certified: VI residual {vi.residual:.3g} (tol {vi_tol:.3g}), "
            f"refinement converged {vi.refinement_converged}, eta_pi {vi.pi_residual:.3g} (tie_tol {tie_tol:.3g})"
        )
    bi = oracle.backward_induction(P, r, gamma, horizon)
    V_pi_H = oracle.h_step_value(P, vi.pi, r, gamma, horizon)
    return FiniteExpert(
        task_name=task.name,
        layout_name=layout_name or maze.layout.name,
        pi=vi.pi,
        actions=vi.actions,
        V_star=vi.V,
        V_pi=vi.V_pi,
        vi_residual=vi.residual,
        vi_iterations=vi.n_iter,
        pi_residual=vi.pi_residual,
        V_H_star=bi.V_H_star,
        V_pi_H=V_pi_H,
        start_state=maze.start_state,
        gamma=gamma,
        horizon=int(horizon),
        reward_max=float(np.max(np.abs(r))),
    )


def expert_occupancy(maze: GridMaze, expert: FiniteExpert) -> np.ndarray:
    """Population expert target ``q_E = d+_{pi*,H}`` from the start state (oracle; evaluation only)."""
    P = transition_matrix(maze)
    return oracle.occupancy_next_state(P, expert.pi, initial_distribution(maze), expert.gamma, expert.horizon)


# =============================================================================
# continuous route controller
# =============================================================================
class RouteController:
    """Stationary state-feedback reference controller (geometry only).

    ``probs(pos)`` returns one-hot action distributions; the object has no
    mutable state, so the action depends on the current position alone.
    """

    def __init__(self, layout: GridLayout, goal_cell: Cell, action_deltas: Sequence[Cell]):
        deltas = [tuple(int(v) for v in d) for d in action_deltas]
        if (0, 0) not in deltas:
            raise ValueError("the route controller needs a stay action")
        self.layout = layout
        self.size = layout.size
        self.goal_cell = tuple(goal_cell)
        self.n_actions = len(deltas)
        self.stay_action = deltas.index((0, 0))
        dist = layout.bfs_distances(self.goal_cell, deltas)
        table = -np.ones((self.size, self.size), dtype=np.int64)
        for (x, y) in layout.free_cells:
            if dist[x, y] < 0:
                raise ValueError(f"cell {(x, y)} cannot reach the goal {self.goal_cell}")
            if (x, y) == self.goal_cell:
                table[x, y] = self.stay_action
                continue
            for a, (dx, dy) in enumerate(deltas):
                nb = (x + dx, y + dy)
                if (dx, dy) != (0, 0) and layout.is_free(nb) and dist[nb] == dist[x, y] - 1:
                    table[x, y] = a
                    break
        dist.setflags(write=False)
        table.setflags(write=False)
        self.bfs_distance = dist
        self.cell_actions = table

    def actions(self, pos: np.ndarray) -> np.ndarray:
        """Deterministic action for each position of shape ``(n, 2)``."""
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
        cells = np.clip(np.floor(pos * self.size).astype(np.int64), 0, self.size - 1)
        acts = self.cell_actions[cells[:, 0], cells[:, 1]]
        if np.any(acts < 0):
            raise ValueError("route controller queried at a position inside a wall")
        return acts

    def probs(self, pos: np.ndarray) -> np.ndarray:
        acts = self.actions(pos)
        out = np.zeros((acts.shape[0], self.n_actions))
        out[np.arange(acts.shape[0]), acts] = 1.0
        return out

    __call__ = probs


def route_controller(layout: GridLayout, goal_cell: Cell, action_deltas: Sequence[Cell]) -> RouteController:
    """METHOD_MAP name for the continuous reference controller."""
    return RouteController(layout, goal_cell, action_deltas)


# =============================================================================
# demonstrations
# =============================================================================
def generate_demonstrations(
    env,
    policy: PolicyLike,
    n: int,
    horizon: int,
    rng: Union[np.random.Generator, Tuple[int, Sequence[Union[str, int]]]],
    meta: Optional[Dict[str, Any]] = None,
) -> Demonstrations:
    """Roll out ``n`` expert trajectories of ``horizon`` transitions from the environment's start law.

    ``rng`` is a generator, or a pair ``(seed, stream_labels)``; in the second
    case the stream is created with :func:`~il_memoir.common.make_rng` and the
    seed and labels are recorded in ``meta`` with their types (``str`` or
    ``int``; see :func:`~il_memoir.common.stream_labels`), so that
    ``make_rng(meta["seed"], *meta["stream"])`` rebuilds the same stream.  The
    returned :class:`~il_memoir.common.Demonstrations` keeps the expert actions
    for diagnostics; learners get ``learner_states()`` only.
    """
    info = dict(meta or {})
    if isinstance(rng, tuple):
        seed, labels = rng
        stream = stream_labels(labels)
        rng = make_rng(seed, *stream)
        info.update({"seed": int(seed), "stream": stream})
    traj = env.rollout(policy, n, horizon, rng)
    info.update({"n_traj": int(n), "horizon": int(horizon)})
    return Demonstrations(traj, info)


# =============================================================================
# validation of the continuous controller with real rollouts
# =============================================================================
def _legs_free_dense(env: ContinuousMaze, a: np.ndarray, b: np.ndarray, samples: int, chunk: int = 20000) -> int:
    """Independent check: count segments ``[a_i, b_i]`` with a sampled point outside the free space.

    Uses ``env.is_free`` at ``samples`` equally spaced points per segment (not
    the slab test of ``segment_clear``).
    """
    ts = np.linspace(0.0, 1.0, samples)
    bad = 0
    for lo in range(0, a.shape[0], chunk):
        aa, bb = a[lo:lo + chunk], b[lo:lo + chunk]
        pts = aa[:, None, :] + ts[None, :, None] * (bb - aa)[:, None, :]
        bad += int(np.sum(~env.is_free(pts).all(axis=1)))
    return bad


def validate_route_controller(
    env: ContinuousMaze,
    controller: RouteController,
    task: GoalTask,
    n: int,
    horizon: int,
    gamma: float,
    rng: np.random.Generator,
    leg_samples: int = 33,
    min_success: float = 0.99,
) -> Dict[str, Any]:
    """Real noisy rollouts of the controller and the validity criteria used by tests and reports.

    ``valid`` requires: every state in the free space; both legs of every
    transition free under an independent dense-sampling check; no blocked
    planned move; actions recomputed from the stored positions equal the
    stored actions (stationary state feedback); success rate (rule of
    ``task``) at least ``min_success``.  (A noise redraw cap would raise in
    ``env.step``, so a completed validation also certifies that it was never
    reached.)
    """
    traj, info = env.rollout(controller, n, horizon, rng, record_info=True)
    pos = traj.states
    starts = pos[:, :-1].reshape(-1, 2)
    bases = info["base"].reshape(-1, 2)
    nexts = pos[:, 1:].reshape(-1, 2)
    bad_legs = _legs_free_dense(env, starts, bases, leg_samples) + _legs_free_dense(env, bases, nexts, leg_samples)
    recomputed = controller.actions(starts).reshape(n, horizon)
    hits = task.hitting_times(pos)
    reached = hits > 0
    in_goal = task.in_goal(pos[:, 1:])
    goal_cell = np.asarray(task.goal_cell)
    in_goal_cell = np.all(env.cell_of(pos[:, 1:]) == goal_cell, axis=-1)
    after = np.arange(1, horizon + 1)[None, :] >= np.where(reached, hits, horizon + 1)[:, None]
    n_after = int(after.sum())
    start_cell = tuple(int(v) for v in env.cell_of(env.start_pos))
    dist_table = getattr(controller, "bfs_distance", None)
    bfs = int(dist_table[start_cell]) if dist_table is not None else None
    returns = task.discounted_returns(pos, gamma)
    success = task.success(pos)
    stats: Dict[str, Any] = {
        "n_traj": int(n),
        "horizon": int(horizon),
        "all_states_free": bool(env.is_free(pos).all()),
        "bad_legs_dense_check": int(bad_legs),
        "blocked_planned_moves": int(info["blocked"].sum()),
        "mean_redraws_per_step": float(info["redraws"].mean()),
        "max_redraws": int(info["redraws"].max()),
        "actions_recomputed_equal": bool(np.array_equal(recomputed, traj.actions)),
        "success_rate": float(success.mean()),
        "success_rule": task.success_rule,
        "hit_rate": float(reached.mean()),
        "final_in_goal_rate": float(task.in_goal(pos[:, -1]).mean()),
        "bfs_distance_start": bfs,
        "hitting_time_mean": float(hits[reached].mean()) if reached.any() else None,
        "hitting_time_max": int(hits[reached].max()) if reached.any() else None,
        "hitting_time_min": int(hits[reached].min()) if reached.any() else None,
        "post_hit_in_goal_region_frac": float(in_goal[after].mean()) if n_after else None,
        "post_hit_in_goal_cell_frac": float(in_goal_cell[after].mean()) if n_after else None,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std(ddof=1)) if n > 1 else 0.0,
        "final_distance_mean": float(np.linalg.norm(pos[:, -1] - task.goal_pos, axis=1).mean()),
    }
    reasons = []
    if not stats["all_states_free"]:
        reasons.append("a state lies outside the free space")
    if stats["bad_legs_dense_check"]:
        reasons.append("a transition leg crosses a wall (dense check)")
    if stats["blocked_planned_moves"]:
        reasons.append("a planned move was blocked")
    if not stats["actions_recomputed_equal"]:
        reasons.append("actions are not a function of the current position")
    if stats["success_rate"] < min_success:
        reasons.append(f"success rate {stats['success_rate']:.3f} below {min_success}")
    stats["valid"] = not reasons
    stats["reasons"] = reasons
    return stats
