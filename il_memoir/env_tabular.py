"""Finite 7x7 mazes: layouts, the learner-facing simulator and the oracle-only P export.

Geometry (PROJECT_SPEC section 5, configs/environment.yaml)
-----------------------------------------------------------
Cells are integer pairs ``(x, y)`` with ``0 <= x, y < size``; ``x`` is the
column (left to right) and ``y`` the row (bottom to top).  The *open* layout has
no blocked cell (49 free cells).  The *four-rooms* layout blocks
``x = 3`` except ``y in {1, 5}`` and ``y = 3`` except ``x in {1, 5}``; the
intersection ``(3, 3)`` is blocked, leaving 40 free cells and the four doorways
``(3, 1), (3, 5), (1, 3), (5, 3)``.  Free cells are indexed in the order of
``(y, x)`` (row by row, from the bottom row), which defines the state index.
The cell ``(x, y)`` has centre ``((x + 0.5) / size, (y + 0.5) / size)`` in the
unit square shared with the continuous maze.

Dynamics (U-10, decided in Prompt 3A)
-------------------------------------
Five actions in the fixed order of the config (stay, right, up, left, down).
A move into a blocked cell or out of the grid leaves the agent where it is.
With probability ``slip_prob`` (0.1) the executed action is drawn uniformly from
all five actions, independently of the chosen one; otherwise the chosen action
is executed.  Hence

    P(. | s, a) = (1 - slip) * delta_{next(s, a)} + (slip / |A|) * sum_b delta_{next(s, b)}.

The stay action has the form ``(1 - slip) * delta_s + slip * u_s``; with
``slip < 1/2`` theory Prop. 5.8(c) gives ``rank P~ = #S`` (logged by
``oracle.transition_rank``).

Data access (METHOD_MAP section 1)
----------------------------------
:class:`GridMaze` is the learner-facing simulator: ``reset``, ``step``,
``rollout``, ``one_hot`` and ``positions``.  It has no reward and exposes no
transition matrix.  :func:`transition_matrix` builds the exact ``P[s, a, s']``
for the oracle, the finite expert, evaluation and tests only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .common import Trajectories, as_policy, sample_actions, PolicyLike
from .config import EnvironmentConfig

Cell = Tuple[int, int]


def four_rooms_rule_blocked(size: int = 7, wall: int = 3, doors: Sequence[int] = (1, 5)) -> frozenset:
    """Blocked cells of the spec's rule: ``x = wall`` except ``y in doors`` and ``y = wall`` except ``x in doors``."""
    vertical = {(wall, y) for y in range(size) if y not in doors}
    horizontal = {(x, wall) for x in range(size) if x not in doors}
    return frozenset(vertical | horizontal)


@dataclass(frozen=True)
class GridLayout:
    """A square lattice with blocked cells.  Pure geometry: no dynamics, no reward."""

    name: str
    size: int
    blocked: frozenset = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        blocked = frozenset((int(x), int(y)) for x, y in self.blocked)
        for x, y in blocked:
            if not (0 <= x < self.size and 0 <= y < self.size):
                raise ValueError(f"blocked cell {(x, y)} outside the {self.size}x{self.size} grid")
        object.__setattr__(self, "blocked", blocked)

    def in_grid(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.size and 0 <= y < self.size

    def is_free(self, cell: Cell) -> bool:
        return self.in_grid(cell) and tuple(cell) not in self.blocked

    @property
    def free_cells(self) -> Tuple[Cell, ...]:
        """Free cells in state-index order: by row ``y``, then column ``x``."""
        return tuple((x, y) for y in range(self.size) for x in range(self.size) if (x, y) not in self.blocked)

    @property
    def n_free(self) -> int:
        return self.size * self.size - len(self.blocked)

    def bfs_distances(self, target: Cell, deltas: Iterable[Cell]) -> np.ndarray:
        """Shortest number of moves to ``target`` on the free-cell graph.

        The graph has an edge between free cells ``c`` and ``c + delta`` for each
        non-zero ``delta``.  Returns an int array ``dist[x, y]`` with ``-1`` for
        blocked or unreachable cells.  Geometry only: no slip, no reward.
        """
        if not self.is_free(target):
            raise ValueError(f"target {target} is not a free cell")
        moves = [tuple(d) for d in deltas if tuple(d) != (0, 0)]
        dist = -np.ones((self.size, self.size), dtype=np.int64)
        dist[target] = 0
        queue = deque([tuple(target)])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in moves:
                # undirected graph for symmetric move sets: predecessor of c is c - delta
                nb = (cx - dx, cy - dy)
                if self.is_free(nb) and dist[nb] < 0:
                    dist[nb] = dist[cx, cy] + 1
                    queue.append(nb)
        return dist

    def ascii(self, marks: Optional[Dict[Cell, str]] = None) -> str:
        """Map with the top row first: ``#`` blocked, ``.`` free, plus one-character marks."""
        marks = marks or {}
        rows = []
        for y in range(self.size - 1, -1, -1):
            row = []
            for x in range(self.size):
                if (x, y) in self.blocked:
                    row.append("#")
                else:
                    row.append(marks.get((x, y), "."))
            rows.append(f"y={y} " + " ".join(row))
        rows.append("    " + " ".join(f"{x}" for x in range(self.size)) + "  (x)")
        return "\n".join(rows)


def make_layout(cfg: EnvironmentConfig, name: str) -> GridLayout:
    """Layout ``name`` from the environment configuration."""
    return GridLayout(name=name, size=cfg.grid_size, blocked=frozenset(cfg.blocked_cells(name)))


class GridMaze:
    """Learner-facing finite simulator: reset/step/rollout only, no reward, no P.

    ``states`` are integer indices into :attr:`cells`.  All methods are
    vectorized over a batch of states.
    """

    def __init__(
        self,
        layout: GridLayout,
        action_deltas: Sequence[Cell],
        slip_prob: float,
        start_cell: Cell,
        action_names: Optional[Sequence[str]] = None,
    ):
        if not 0.0 <= slip_prob < 1.0:
            raise ValueError("slip_prob must lie in [0, 1)")
        if not layout.is_free(start_cell):
            raise ValueError(f"start cell {start_cell} is not free in layout {layout.name!r}")
        self.layout = layout
        self.size = layout.size
        self.slip_prob = float(slip_prob)
        self.action_deltas = tuple((int(dx), int(dy)) for dx, dy in action_deltas)
        self.action_names = tuple(action_names) if action_names is not None else tuple(
            f"a{i}" for i in range(len(self.action_deltas))
        )
        if len(self.action_names) != len(self.action_deltas):
            raise ValueError("action_names and action_deltas differ in length")
        self.cells: Tuple[Cell, ...] = layout.free_cells
        self._index = {c: i for i, c in enumerate(self.cells)}
        self.start_state = self._index[tuple(start_cell)]
        # Deterministic successor table (private): next cell index for each (s, a).
        nxt = np.empty((len(self.cells), len(self.action_deltas)), dtype=np.int64)
        for s, (x, y) in enumerate(self.cells):
            for a, (dx, dy) in enumerate(self.action_deltas):
                target = (x + dx, y + dy)
                nxt[s, a] = self._index[target] if layout.is_free(target) else s
        nxt.setflags(write=False)
        self._next_state = nxt
        centres = (np.asarray(self.cells, dtype=np.float64) + 0.5) / self.size
        centres.setflags(write=False)
        self._centres = centres

    # ------------------------------------------------------------------ sizes
    @property
    def n_states(self) -> int:
        return len(self.cells)

    @property
    def n_actions(self) -> int:
        return len(self.action_deltas)

    def index(self, cell: Cell) -> int:
        return self._index[tuple(cell)]

    def cell(self, state: int) -> Cell:
        return self.cells[int(state)]

    # ---------------------------------------------------------- observations
    def positions(self, states: np.ndarray) -> np.ndarray:
        """Cell centres in [0,1]^2 for an array of state indices (any shape)."""
        return self._centres[np.asarray(states, dtype=np.int64)]

    def one_hot(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.int64)
        out = np.zeros(states.shape + (self.n_states,), dtype=np.float64)
        np.put_along_axis(out, states[..., None], 1.0, axis=-1)
        return out

    # -------------------------------------------------------------- dynamics
    def reset(self, n: int = 1) -> np.ndarray:
        """``n`` copies of the start state (mu_0 is the Dirac mass at the start cell)."""
        return np.full(int(n), self.start_state, dtype=np.int64)

    def step(self, states: np.ndarray, actions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample next states for a batch of (state, action) pairs.  Returns states only."""
        s = np.asarray(states, dtype=np.int64).reshape(-1)
        a = np.asarray(actions, dtype=np.int64).reshape(-1)
        if s.shape != a.shape:
            raise ValueError("states and actions must have the same length")
        if s.size and (s.min() < 0 or s.max() >= self.n_states):
            raise ValueError("state index out of range")
        if a.size and (a.min() < 0 or a.max() >= self.n_actions):
            raise ValueError("action index out of range")
        if self.slip_prob > 0.0:
            slipped = rng.random(s.size) < self.slip_prob
            random_actions = rng.integers(0, self.n_actions, size=s.size)
            executed = np.where(slipped, random_actions, a)
        else:
            executed = a
        return self._next_state[s, executed]

    def rollout(
        self,
        policy: PolicyLike,
        n: int,
        horizon: int,
        rng: np.random.Generator,
        start_states: Optional[np.ndarray] = None,
    ) -> Trajectories:
        """``n`` independent trajectories of ``horizon`` transitions under a stationary policy.

        ``policy`` is an ``(S, A)`` matrix or a callable ``states -> (n, A)``
        probabilities.  The environment is continuing: reaching a goal changes
        nothing, and there is no reset within a trajectory.
        """
        pol = as_policy(policy)
        n, horizon = int(n), int(horizon)
        states = np.empty((n, horizon + 1), dtype=np.int64)
        actions = np.empty((n, horizon), dtype=np.int64)
        states[:, 0] = self.reset(n) if start_states is None else np.asarray(start_states, dtype=np.int64)
        for t in range(horizon):
            probs = pol(states[:, t])
            if probs.shape != (n, self.n_actions):
                raise ValueError(f"policy returned shape {probs.shape}, expected {(n, self.n_actions)}")
            actions[:, t] = sample_actions(probs, rng)
            states[:, t + 1] = self.step(states[:, t], actions[:, t], rng)
        return Trajectories(states, actions)


def make_grid_maze(cfg: EnvironmentConfig, layout: str, slip_prob: Optional[float] = None) -> GridMaze:
    """Finite maze for ``layout`` with the configured actions, start and slip (overridable for tests)."""
    return GridMaze(
        make_layout(cfg, layout),
        cfg.action_deltas,
        cfg.slip_prob if slip_prob is None else slip_prob,
        cfg.start,
        cfg.action_names,
    )


def transition_matrix(maze: GridMaze) -> np.ndarray:
    """Exact ``P[s, a, s']`` (float64).  ORACLE / EXPERT / EVALUATION / TESTS ONLY.

    Learners must never call this function (METHOD_MAP section 1).  It is built
    by enumerating the same deterministic successor table as :meth:`GridMaze.step`
    and the slip rule, so ``tests/test_env_tabular.py`` can compare it with the
    simulator's sampled frequencies.
    """
    n_s, n_a = maze.n_states, maze.n_actions
    det = np.zeros((n_s, n_a, n_s), dtype=np.float64)
    det[np.arange(n_s)[:, None], np.arange(n_a)[None, :], maze._next_state] = 1.0
    slip = maze.slip_prob
    return (1.0 - slip) * det + slip * det.mean(axis=1, keepdims=True)


def initial_distribution(maze: GridMaze) -> np.ndarray:
    """mu_0 as a probability vector: the Dirac mass at the start state."""
    mu0 = np.zeros(maze.n_states, dtype=np.float64)
    mu0[maze.start_state] = 1.0
    return mu0
