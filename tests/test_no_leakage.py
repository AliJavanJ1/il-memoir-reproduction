"""Data-access rules of METHOD_MAP section 1, enforced statically on the learner modules.

Learners (``il_memoir/fpg.py``, ``fb.py`` and ``fb_il.py``, written in Prompt 3B)
must not reach the transition matrix, the kernel parameters, the evaluation
reward, the goals, the oracle, the expert's actions or the simulator's
privileged diagnostics.  The checker parses each learner module and rejects
forbidden imports and forbidden names or attributes.  It is validated here on
synthetic sources, so it is known to work before the learner modules exist; the
per-module test is skipped, with a reason, while a module does not exist yet.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEARNER_MODULES = ("fpg", "fb", "fb_il")

FORBIDDEN_MODULES = {"oracle", "tasks", "expert", "evaluation"}
FORBIDDEN_NAMES = {
    # finite P and its ingredients (env_tabular exports P for the oracle only)
    "transition_matrix", "initial_distribution", "_next_state", "slip_prob",
    # continuous kernel parameters and privileged per-step diagnostics
    "noise_std", "step_length", "max_noise_redraws", "step_with_info", "record_info", "StepInfo",
    # evaluation reward, goals and success (il_memoir.tasks, full configuration)
    "GoalTask", "make_task", "make_tasks", "finite_reward", "goal_cell", "goal_pos", "goals", "goal_names",
    "reward_width", "reward_width_cells", "goal_radius", "goal_radius_cells", "success_rule", "reward_kind",
    # expert actions: learners receive Demonstrations.learner_states() arrays, never the object
    "Demonstrations", "trajectories", "generate_demonstrations",
    # the full configuration object (learners receive EnvironmentConfig.learner_dict())
    "load_environment_config", "environment_config_from_dict",
}


def violations(source: str) -> list:
    """Forbidden imports, names and attributes in a Python source (a list of messages)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "il_memoir" and len(parts) > 1 and parts[1] in FORBIDDEN_MODULES:
                    found.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            parts = module.split(".") if module else []
            in_package = node.level > 0 or (parts and parts[0] == "il_memoir")
            sub = parts[1:] if parts and parts[0] == "il_memoir" else parts
            if in_package and sub and sub[0] in FORBIDDEN_MODULES:
                found.append(f"line {node.lineno}: from {'.' * node.level}{module} import ...")
            if in_package and not sub:
                for alias in node.names:
                    if alias.name in FORBIDDEN_MODULES:
                        found.append(f"line {node.lineno}: from {'.' * node.level}{module} import {alias.name}")
            for alias in node.names:
                if alias.name in FORBIDDEN_NAMES:
                    found.append(f"line {node.lineno}: imports forbidden name {alias.name}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            found.append(f"line {node.lineno}: name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            found.append(f"line {node.lineno}: attribute .{node.attr}")
        elif isinstance(node, ast.keyword) and node.arg in FORBIDDEN_NAMES:
            found.append(f"line {getattr(node, 'lineno', '?')}: keyword {node.arg}=")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr", "setattr"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in FORBIDDEN_NAMES
        ):
            found.append(f"line {node.lineno}: {node.func.id}(..., {node.args[1].value!r})")
    return found


CLEAN = '''
import numpy as np
from .common import make_rng, sample_actions, Trajectories
from .density import discount_weights, WeightedKDE
from . import divergences as dv

def collect(env, policy, n, horizon, rng):
    traj = env.rollout(policy, n, horizon, rng)
    return traj.states, traj.actions          # the learner's own actions are allowed

def target(demo_states, gamma):
    return discount_weights(demo_states.shape[0], demo_states.shape[1] - 1, gamma)
'''

DIRTY = {
    "imports the oracle": "from il_memoir import oracle\n",
    "imports the oracle relatively": "from . import oracle\n",
    "imports the oracle module path": "import il_memoir.oracle as o\n",
    "imports from tasks": "from .tasks import GoalTask\n",
    "imports from expert": "from il_memoir.expert import finite_expert\n",
    "uses P": "from .env_tabular import transition_matrix\nP = transition_matrix(maze)\n",
    "reads the slip": "p = maze.slip_prob\n",
    "reads the private table": "t = maze._next_state\n",
    "reads the noise level": "s = env.noise_std\n",
    "uses step diagnostics": "nxt, info = env.step_with_info(x, a, rng)\n",
    "requests diagnostics": "traj, info = env.rollout(pol, 4, 10, rng, record_info=True)\n",
    "reads goals": "g = cfg.goal_cell('goal_5_5')\n",
    "reads the reward width": "w = cfg.reward_width\n",
    "reads expert actions": "a = demos.trajectories.actions\n",
    "loads the full config": "from il_memoir.config import load_environment_config\n",
    "reads the slip by name": "p = getattr(maze, 'slip_prob')\n",
}


def test_checker_accepts_clean_learner_code():
    assert violations(CLEAN) == []


@pytest.mark.parametrize("label", sorted(DIRTY))
def test_checker_flags_each_forbidden_access(label):
    assert violations(DIRTY[label]), label


@pytest.mark.parametrize("module", LEARNER_MODULES)
def test_learner_modules_respect_data_access_rules(module):
    path = ROOT / "il_memoir" / f"{module}.py"
    if not path.exists():
        pytest.skip(f"learner module il_memoir/{module}.py is written in Prompt 3B")
    found = violations(path.read_text(encoding="utf-8"))
    assert not found, f"{path.name}: " + "; ".join(found)


def test_shared_learner_safe_modules_do_not_import_privileged_modules():
    """Modules that learners import (common, density, divergences, config, nets) must not pull in the oracle, tasks or expert."""
    for name in ("common", "density", "divergences", "config", "nets"):
        source = (ROOT / "il_memoir" / f"{name}.py").read_text(encoding="utf-8")
        imports = [v for v in violations(source) if "import" in v]
        assert not imports, f"{name}.py: {imports}"


# ---------------------------------------------------------------------------
# The wall-geometry exception (METHOD_MAP section 1): learners may read the
# wall cells of the learner view only to build the free space for density
# estimation.  Statically, the learner-view key "layouts" may appear only inside
# fpg.free_space_from_view, and no learner module may read a simulator's
# ``walls`` or a layout's ``blocked`` cells directly.
# ---------------------------------------------------------------------------
GEOMETRY_KEYS = {"layouts"}
GEOMETRY_ATTRS = {"walls", "blocked", "blocked_cells"}
ALLOWED_GEOMETRY_FUNCTIONS = {("fpg", "free_space_from_view")}


def geometry_uses(module: str, source: str) -> list:
    """Uses of the wall geometry outside the allowed density helper (a list of messages)."""
    tree = ast.parse(source)
    found = []

    def visit(node, func):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node.name
        if isinstance(node, ast.Constant) and node.value in GEOMETRY_KEYS and (module, func) not in ALLOWED_GEOMETRY_FUNCTIONS:
            found.append(f"line {node.lineno}: key {node.value!r} in {func or '<module>'}")
        if isinstance(node, ast.Attribute) and node.attr in GEOMETRY_ATTRS:
            found.append(f"line {node.lineno}: attribute .{node.attr} in {func or '<module>'}")
        for child in ast.iter_child_nodes(node):
            visit(child, func)

    visit(tree, None)
    return found


def test_geometry_checker_flags_uses_outside_the_density_helper():
    assert geometry_uses("fpg", "def free_space_from_view(view):\n    return view['grid']['layouts']\n") == []
    assert geometry_uses("fpg", "def plan(view):\n    return view['grid']['layouts']\n")
    assert geometry_uses("fb", "def free_space_from_view(view):\n    return view['grid']['layouts']\n")
    assert geometry_uses("fb", "def f(env):\n    return env.walls\n")


@pytest.mark.parametrize("module", LEARNER_MODULES)
def test_learners_use_wall_geometry_only_for_density_estimation(module):
    path = ROOT / "il_memoir" / f"{module}.py"
    if not path.exists():
        pytest.skip(f"learner module il_memoir/{module}.py is written in Prompt 3B")
    found = geometry_uses(module, path.read_text(encoding="utf-8"))
    assert not found, f"{path.name}: " + "; ".join(found)


# ---------------------------------------------------------------------------
# Prompt 3C: the training executors (il_memoir/train.py) are on the learner side of the data-access
# rules.  They may build simulators and hand learners the learner view, the demonstration states and
# the task-blind replay; they must not import the oracle, the evaluation reward, the expert, the
# evaluation or the data-generation module, nor read expert actions or held-out references.
# ---------------------------------------------------------------------------
TRAIN_FORBIDDEN_MODULES = FORBIDDEN_MODULES | {"prepare"}
TRAIN_FORBIDDEN_NAMES = {"load_demonstrations", "load_reference_states", "finite_expert", "route_controller",
                         "expert_occupancy", "FiniteOracle", "mc_metrics"}


def test_training_executors_respect_the_data_access_rules():
    source = (ROOT / "il_memoir" / "train.py").read_text(encoding="utf-8")
    found = violations(source)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            mod = (node.module or "").split(".")[0]
            if mod in TRAIN_FORBIDDEN_MODULES or any(a.name in TRAIN_FORBIDDEN_MODULES for a in node.names if not mod):
                found.append(f"line {node.lineno}: imports {mod or [a.name for a in node.names]}")
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name in TRAIN_FORBIDDEN_NAMES:
                found.append(f"line {node.lineno}: {name}")
    assert not found, "train.py: " + "; ".join(found)
    assert "load_demo_states" in source and "learner_dict()" in source


def test_the_learner_side_data_loader_returns_states_only(smoke_run):
    from il_memoir import datasets as ds

    man = ds.load_manifest(smoke_run["data"], smoke_run["suite"].data_id)
    key = next(k for k in man["items"] if k.startswith("demos/"))
    states = ds.load_demo_states(smoke_run["data"], smoke_run["suite"].data_id, key, man["items"][key]["content_sha256"])
    assert isinstance(states, np.ndarray) and states.ndim in (2, 3) and states.shape[0] == 4
    _, actions, _ = ds.load_demonstrations(smoke_run["data"], smoke_run["suite"].data_id, key)
    assert actions.shape == (states.shape[0], states.shape[1] - 1)     # the actions exist but are not returned


def test_analysis_and_audit_only_read_numbers_from_saved_runs():
    """No random number generator or simulator in the result-producing code: every number comes from run files."""
    for name in ("analysis", "audit"):
        tree = ast.parse((ROOT / "il_memoir" / f"{name}.py").read_text(encoding="utf-8"))
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] + [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                bad += [m for m in mods if m in ("random", "numpy.random", "common", "env_tabular", "env_continuous",
                                                 "oracle", "evaluation", "train", "fpg", "fb", "fb_il")]
            if isinstance(node, ast.Attribute) and node.attr in ("random", "default_rng", "make_rng", "rollout"):
                bad.append(node.attr)
        assert not bad, f"{name}.py: {bad}"
