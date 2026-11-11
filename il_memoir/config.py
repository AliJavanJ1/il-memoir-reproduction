"""Load and validate the frozen environment configuration (``configs/environment.yaml``).

Unknown keys are rejected so that a typo cannot silently fall back to a
default.  Every number must be finite (NaN and infinities are rejected before
the range checks), and integer fields must be integers, not floats or booleans.
Geometry is stored in cell units; the continuous quantities in [0,1]^2 units
are derived properties (one cell has side ``1/grid_size``).
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_CONFIG = PROJECT_ROOT / "configs" / "environment.yaml"

Cell = Tuple[int, int]

_TOP_KEYS = {"schema", "gamma", "horizon", "grid", "actions", "finite", "continuous", "task", "density", "oracle"}
_SECTION_KEYS = {
    "grid": {"size", "start", "goals", "layouts"},
    "finite": {"slip_prob"},
    "continuous": {"layout", "step_cells", "noise_std", "max_noise_redraws"},
    "task": {"reward", "reward_width_cells", "goal_radius_cells", "success"},
    "density": {"kde_bandwidth", "histogram_bins", "smoothing_eps"},
    "oracle": {"value_iteration_tol", "tie_tol"},
}
SCHEMA = "il_memoir/environment-v1"


@dataclass(frozen=True)
class EnvironmentConfig:
    gamma: float
    horizon: int
    grid_size: int
    start: Cell
    goals: Tuple[Tuple[str, Cell], ...]
    layouts: Tuple[Tuple[str, Tuple[Cell, ...]], ...]
    action_names: Tuple[str, ...]
    action_deltas: Tuple[Cell, ...]
    slip_prob: float
    continuous_layout: str
    step_cells: float
    noise_std: float
    max_noise_redraws: int
    reward_kind: str
    reward_width_cells: float
    goal_radius_cells: float
    success_rule: str
    kde_bandwidth: float
    histogram_bins: int
    smoothing_eps: float
    value_iteration_tol: float
    tie_tol: float
    source: str = "<dict>"

    # ----------------------------------------------------------------- lookups
    @property
    def goal_names(self) -> Tuple[str, ...]:
        return tuple(name for name, _ in self.goals)

    @property
    def layout_names(self) -> Tuple[str, ...]:
        return tuple(name for name, _ in self.layouts)

    def goal_cell(self, name: str) -> Cell:
        for goal_name, cell in self.goals:
            if goal_name == name:
                return cell
        raise KeyError(f"unknown goal {name!r}; known: {self.goal_names}")

    def blocked_cells(self, layout: str) -> Tuple[Cell, ...]:
        for layout_name, cells in self.layouts:
            if layout_name == layout:
                return cells
        raise KeyError(f"unknown layout {layout!r}; known: {self.layout_names}")

    # ------------------------------------------------- [0,1]^2 derived lengths
    @property
    def cell_length(self) -> float:
        return 1.0 / self.grid_size

    @property
    def step_length(self) -> float:
        return self.step_cells / self.grid_size

    @property
    def reward_width(self) -> float:
        return self.reward_width_cells / self.grid_size

    @property
    def goal_radius(self) -> float:
        return self.goal_radius_cells / self.grid_size

    # -------------------------------------------------------------- provenance
    def to_dict(self) -> Dict[str, Any]:
        """JSON-compatible dictionary of every setting (the ``source`` path excluded)."""
        return {
            "schema": SCHEMA,
            "gamma": self.gamma,
            "horizon": self.horizon,
            "grid": {
                "size": self.grid_size,
                "start": list(self.start),
                "goals": {name: list(cell) for name, cell in self.goals},
                "layouts": {name: [list(c) for c in cells] for name, cells in self.layouts},
            },
            "actions": [
                {"name": n, "dx": d[0], "dy": d[1]} for n, d in zip(self.action_names, self.action_deltas)
            ],
            "finite": {"slip_prob": self.slip_prob},
            "continuous": {
                "layout": self.continuous_layout,
                "step_cells": self.step_cells,
                "noise_std": self.noise_std,
                "max_noise_redraws": self.max_noise_redraws,
            },
            "task": {
                "reward": self.reward_kind,
                "reward_width_cells": self.reward_width_cells,
                "goal_radius_cells": self.goal_radius_cells,
                "success": self.success_rule,
            },
            "density": {
                "kde_bandwidth": self.kde_bandwidth,
                "histogram_bins": self.histogram_bins,
                "smoothing_eps": self.smoothing_eps,
            },
            "oracle": {"value_iteration_tol": self.value_iteration_tol, "tie_tol": self.tie_tol},
        }

    def sha256(self) -> str:
        """Hash of the canonical JSON of ``to_dict()``; independent of YAML formatting."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def learner_dict(self) -> Dict[str, Any]:
        """The settings a learner may read: discount, horizon, geometry, actions and density estimator.

        Removed (METHOD_MAP section 1): the goal cells, the evaluation reward and
        success rule (``task``), the oracle tolerances, and the transition
        parameters (finite slip, continuous step, noise and redraw cap), which
        together with the layout define ``P``.  Learners interact with the
        simulator and never need them.  Runners pass this view, not the full
        configuration, to learners.
        """
        full = self.to_dict()
        grid = dict(full["grid"])
        grid.pop("goals")
        return {
            "schema": SCHEMA + "/learner",
            "gamma": full["gamma"],
            "horizon": full["horizon"],
            "grid": grid,
            "actions": full["actions"],
            "continuous": {"layout": full["continuous"]["layout"]},
            "density": full["density"],
        }


def _require_keys(name: str, section: Mapping[str, Any], allowed: set) -> None:
    if not isinstance(section, Mapping):
        raise ValueError(f"section {name!r} must be a mapping")
    unknown = set(section) - allowed
    missing = allowed - set(section)
    if unknown:
        raise ValueError(f"unknown key(s) in {name!r}: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing key(s) in {name!r}: {sorted(missing)}")


def _real(value: Any, what: str) -> float:
    """A finite real number (int or float, not bool or str).  NaN and infinities are rejected
    before any range check, because a comparison with NaN is always false."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{what} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return out


def _integer(value: Any, what: str) -> int:
    """An integer (not bool, not float: a float would otherwise be truncated silently)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    return int(value)


def _cell(value: Any, what: str, size: int) -> Cell:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{what} must be a pair [x, y], got {value!r}")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise ValueError(f"{what} must have integer coordinates, got {value!r}")
    x, y = int(value[0]), int(value[1])
    if not (0 <= x < size and 0 <= y < size):
        raise ValueError(f"{what} {value!r} lies outside the {size}x{size} grid")
    return (x, y)


def _check_connected(name: str, size: int, blocked: Tuple[Cell, ...], start: Cell, moves) -> None:
    """Every free cell must be reachable from the start cell (otherwise BFS experts are undefined)."""
    blocked_set = set(blocked)
    free = {(x, y) for x in range(size) for y in range(size)} - blocked_set
    seen = {start}
    frontier = [start]
    while frontier:
        cx, cy = frontier.pop()
        for dx, dy in moves:
            nb = (cx + dx, cy + dy)
            if nb in free and nb not in seen:
                seen.add(nb)
                frontier.append(nb)
    if seen != free:
        raise ValueError(f"layout {name!r} has free cells unreachable from the start: {sorted(free - seen)}")


def environment_config_from_dict(raw: Mapping[str, Any], source: str = "<dict>") -> EnvironmentConfig:
    """Validate a parsed YAML mapping and build an :class:`EnvironmentConfig`."""
    if not isinstance(raw, Mapping):
        raise ValueError("the environment configuration must be a mapping")
    _require_keys("<top level>", raw, _TOP_KEYS)
    if raw["schema"] != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}, got {raw['schema']!r}")
    for name, keys in _SECTION_KEYS.items():
        _require_keys(name, raw[name], keys)

    gamma = _real(raw["gamma"], "gamma")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie in (0, 1)")
    horizon = _integer(raw["horizon"], "horizon")
    if horizon < 1:
        raise ValueError("horizon must be a positive integer")

    grid = raw["grid"]
    size = _integer(grid["size"], "grid.size")
    if size < 2:
        raise ValueError("grid.size must be an integer >= 2")
    start = _cell(grid["start"], "grid.start", size)
    if not isinstance(grid["goals"], Mapping) or not grid["goals"]:
        raise ValueError("grid.goals must be a non-empty mapping")
    goals = tuple((str(k), _cell(v, f"goal {k}", size)) for k, v in grid["goals"].items())
    if len({cell for _, cell in goals}) != len(goals):
        raise ValueError("two goals share the same cell")
    if not isinstance(grid["layouts"], Mapping) or not grid["layouts"]:
        raise ValueError("grid.layouts must be a non-empty mapping")
    layouts = []
    for lname, cells in grid["layouts"].items():
        blocked = tuple(sorted({_cell(c, f"blocked cell of {lname}", size) for c in (cells or [])}))
        if len(blocked) != len(cells or []):
            raise ValueError(f"layout {lname!r} lists a blocked cell twice")
        for name, cell in (("start", start),) + goals:
            if cell in blocked:
                raise ValueError(f"{name} cell {cell} is blocked in layout {lname!r}")
        layouts.append((str(lname), blocked))

    actions = raw["actions"]
    if not isinstance(actions, list) or len(actions) < 2:
        raise ValueError("actions must be a list of at least two actions")
    names, deltas = [], []
    for a in actions:
        if not isinstance(a, Mapping) or set(a) != {"name", "dx", "dy"}:
            raise ValueError(f"each action needs exactly name, dx, dy; got {a!r}")
        dx, dy = _integer(a["dx"], f"action {a['name']!r} dx"), _integer(a["dy"], f"action {a['name']!r} dy")
        if abs(dx) + abs(dy) > 1:
            raise ValueError(f"action {a['name']!r} must move at most one cell along one axis")
        names.append(str(a["name"]))
        deltas.append((dx, dy))
    if len(set(names)) != len(names) or len(set(deltas)) != len(deltas):
        raise ValueError("action names and displacements must be unique")
    if (0, 0) not in deltas:
        raise ValueError("the action set must contain a stay action (dx = dy = 0)")
    for lname, blocked in layouts:
        _check_connected(lname, size, blocked, start, [d for d in deltas if d != (0, 0)])

    slip = _real(raw["finite"]["slip_prob"], "finite.slip_prob")
    if not 0.0 <= slip < 1.0:
        raise ValueError("finite.slip_prob must lie in [0, 1)")

    cont = raw["continuous"]
    cont_layout = str(cont["layout"])
    if cont_layout not in {n for n, _ in layouts}:
        raise ValueError(f"continuous.layout {cont_layout!r} is not a declared layout")
    step_cells = _real(cont["step_cells"], "continuous.step_cells")
    noise_std = _real(cont["noise_std"], "continuous.noise_std")
    max_redraws = _integer(cont["max_noise_redraws"], "continuous.max_noise_redraws")
    if step_cells <= 0 or noise_std < 0 or max_redraws < 1:
        raise ValueError("continuous: step_cells > 0, noise_std >= 0 and max_noise_redraws >= 1 required")

    task = raw["task"]
    if task["reward"] != "gaussian":
        raise ValueError("task.reward must be 'gaussian'")
    width = _real(task["reward_width_cells"], "task.reward_width_cells")
    radius = _real(task["goal_radius_cells"], "task.goal_radius_cells")
    if width <= 0 or radius <= 0:
        raise ValueError("task widths must be positive")
    if task["success"] not in ("hit", "final"):
        raise ValueError("task.success must be 'hit' or 'final'")

    dens = raw["density"]
    bandwidth = _real(dens["kde_bandwidth"], "density.kde_bandwidth")
    bins = _integer(dens["histogram_bins"], "density.histogram_bins")
    eps = _real(dens["smoothing_eps"], "density.smoothing_eps")
    if bandwidth <= 0 or bins < 1 or not 0.0 < eps <= 1.0:
        raise ValueError("density: kde_bandwidth > 0, histogram_bins >= 1, 0 < smoothing_eps <= 1 required")

    orc = raw["oracle"]
    vi_tol = _real(orc["value_iteration_tol"], "oracle.value_iteration_tol")
    tie_tol = _real(orc["tie_tol"], "oracle.tie_tol")
    if vi_tol <= 0 or tie_tol < 0:
        raise ValueError("oracle: value_iteration_tol > 0 and tie_tol >= 0 required")

    return EnvironmentConfig(
        gamma=gamma,
        horizon=horizon,
        grid_size=size,
        start=start,
        goals=goals,
        layouts=tuple(layouts),
        action_names=tuple(names),
        action_deltas=tuple(deltas),
        slip_prob=slip,
        continuous_layout=cont_layout,
        step_cells=step_cells,
        noise_std=noise_std,
        max_noise_redraws=max_redraws,
        reward_kind="gaussian",
        reward_width_cells=width,
        goal_radius_cells=radius,
        success_rule=str(task["success"]),
        kde_bandwidth=bandwidth,
        histogram_bins=bins,
        smoothing_eps=eps,
        value_iteration_tol=vi_tol,
        tie_tol=tie_tol,
        source=source,
    )


def load_environment_config(path: Optional[Union[str, Path]] = None) -> EnvironmentConfig:
    """Read and validate an environment YAML file (default ``configs/environment.yaml``)."""
    cfg_path = Path(path) if path is not None else DEFAULT_ENV_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return environment_config_from_dict(raw, source=str(cfg_path))
