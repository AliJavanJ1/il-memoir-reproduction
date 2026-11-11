"""Generation of the frozen data (``scripts/prepare_data.py``; also called by the runner for missing items).

Every item is a deterministic function of its specification
(``experiment.demos_item``, ``replay_item``, ``reference_item``): the
environment configuration, the seed and the stream labels.  Preparing an item
therefore always regenerates it in memory and compares its *content* hash with
the stored file, if any:

* no stored item: the file is written and entered in ``data/<id>/manifest.json``;
* same content: nothing is written ("verified");
* different specification or content: :class:`DataConflict` is raised and nothing
  is overwritten.  A changed data definition needs a new ``data.id`` (INTERFACE:
  "never overwrite different data without a distinct ID").

Demonstrations: the finite oracle expert (certified by ``expert.finite_expert``)
or the continuous route controller, ``N_E`` trajectories of ``H`` transitions
from the start state.  The expert actions are stored for diagnostics; learners
receive the states only (``datasets.load_demo_states``).

Replay: ``fb.collect_replay`` (uniform actions, uniform free-space resets,
segments of ``segment_length``); coverage of every state-action pair (finite)
or of every free lattice cell with every action (continuous) is checked
before the item is accepted (spec section 6: "Check coverage before training").

Held-out reference (continuous only; the finite evaluation is exact): route
controller rollouts with the reference seed of the suite kind and the stream
``("heldout_reference", env, layout, goal)``, never given to a learner.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from . import datasets as ds
from .common import make_rng
from .env_continuous import make_continuous_maze
from .env_tabular import make_grid_maze, make_layout
from .experiment import DataItem, SuiteConfig, data_items
from .expert import finite_expert, generate_demonstrations, route_controller
from .fb import ReplayBuffer, ReplayConfig, collect_replay, replay_coverage
from .logging import code_scope, replace_file, sha256_array, sha256_file, utc_now
from .tasks import make_task


class DataConflict(RuntimeError):
    pass


def make_env(cfg, env: str, layout: str):
    """The simulator of a condition: ``GridMaze`` (finite) or ``ContinuousMaze`` (continuous)."""
    if env == "finite":
        return make_grid_maze(cfg, layout)
    if env == "continuous":
        return make_continuous_maze(cfg, layout)
    raise ValueError(f"unknown environment {env!r}")


def _expert_policy(cfg, env: str, layout: str, task: str):
    """Expert policy and a summary of how it was certified (finite) or built (continuous)."""
    t = make_task(cfg, task)
    if env == "finite":
        maze = make_grid_maze(cfg, layout)
        ex = finite_expert(maze, t, cfg.gamma, cfg.horizon, cfg.value_iteration_tol, cfg.tie_tol, layout_name=layout)
        summ = {k: float(v) for k, v in ex.summary().items()}
        return maze, ex.pi, {"expert": "finite_oracle_optimal", "certificate": summ,
                             "expert_actions_sha256": sha256_array(ex.actions)}
    maze = make_continuous_maze(cfg, layout)
    ctrl = route_controller(make_layout(cfg, layout), cfg.goal_cell(task), cfg.action_deltas)
    return maze, ctrl, {"expert": "route_controller", "expert_actions_sha256": sha256_array(ctrl.cell_actions)}


def generate_demos(suite: SuiteConfig, item: DataItem):
    s = item.spec
    cfg = suite.env_cfg
    maze, policy, info = _expert_policy(cfg, s["env"], s["layout"], s["task"])
    demos = generate_demonstrations(maze, policy, s["n_trajectories"], s["horizon"], (s["seed"], tuple(s["stream"])))
    states = np.asarray(demos.trajectories.states)
    meta = {"kind": "demos", "spec_sha256": item.spec_sha256, "seed": demos.meta["seed"], "stream": demos.meta["stream"],
            **info}
    if s["env"] == "continuous":
        meta["all_states_free"] = bool(maze.is_free(states).all())
        if not meta["all_states_free"]:
            raise DataConflict(f"{item.key}: a demonstration state lies outside the free space")
    return {"states": states, "actions": np.asarray(demos.trajectories.actions)}, meta


def generate_reference(suite: SuiteConfig, item: DataItem):
    s = item.spec
    cfg = suite.env_cfg
    maze, policy, info = _expert_policy(cfg, s["env"], s["layout"], s["task"])
    rng = make_rng(s["seed"], *s["stream"])
    traj = maze.rollout(policy, s["n_trajectories"], s["horizon"], rng)
    states = np.asarray(traj.states)
    if not bool(maze.is_free(states).all()):
        raise DataConflict(f"{item.key}: a reference state lies outside the free space")
    meta = {"kind": "reference", "spec_sha256": item.spec_sha256, "seed": int(s["seed"]), "stream": list(s["stream"]),
            "purpose": "held-out evaluation target; never a learner input", **info}
    return {"states": states}, meta


def generate_replay(suite: SuiteConfig, item: DataItem) -> ReplayBuffer:
    s = item.spec
    cfg = suite.env_cfg
    env = make_env(cfg, s["env"], s["layout"])
    buf = collect_replay(env, ReplayConfig(s["n_transitions"], s["segment_length"]), make_rng(s["seed"], *s["stream"]))
    cov = replay_coverage(buf, env.n_actions, cfg.grid_size)
    if s["env"] == "finite":
        ok = cov["states_visited"] == env.n_states and cov["min_state_action_count"] > 0
        summary = {"states_visited": cov["states_visited"], "n_states": env.n_states,
                   "min_state_action_count": cov["min_state_action_count"],
                   "min_next_state_count": cov["min_next_state_count"]}
    else:
        n_free = make_layout(cfg, s["layout"]).n_free
        ok = cov["cells_visited"] == n_free and cov["min_cell_action_count"] > 0
        summary = {"cells_visited": cov["cells_visited"], "n_free_cells": n_free,
                   "min_cell_action_count": cov["min_cell_action_count"]}
    summary["action_counts"] = [int(v) for v in cov["action_counts"]]
    summary["covers_all"] = bool(ok)
    if not ok:
        raise DataConflict(f"{item.key}: the task-blind replay does not cover every room/state and action: {summary}")
    buf.meta.update({"spec_sha256": item.spec_sha256, "seed": int(s["seed"]), "stream": list(s["stream"]),
                     "coverage": summary})
    return buf


def prepare_items(
    suite: SuiteConfig,
    data_root: Union[str, Path],
    keys: Optional[Sequence[str]] = None,
    log: Callable[[str], None] = print,
) -> List[Dict[str, Any]]:
    """Generate, verify or write every data item of ``suite`` (or only ``keys``).  Raises on any conflict."""
    items = data_items(suite)
    if keys is not None:
        missing = set(keys) - set(items)
        if missing:
            raise KeyError(f"not data items of suite {suite.name}: {sorted(missing)}")
        items = {k: items[k] for k in keys}
    data_id = suite.data_id
    manifest = ds.load_manifest(data_root, data_id)
    manifest.setdefault("suites", [])
    if suite.name not in manifest["suites"]:
        manifest["suites"].append(suite.name)
    code = code_scope("data")
    report = []
    for key, item in items.items():
        t0 = time.perf_counter()
        path = ds.item_path(data_root, data_id, key)
        entry = manifest["items"].get(key)
        if entry is not None and entry.get("spec_sha256") != item.spec_sha256:
            raise DataConflict(
                f"data item {key!r} in {ds.data_dir(data_root, data_id)} was generated from another specification "
                f"(stored spec {entry.get('spec_sha256')}, current {item.spec_sha256}); refusing to overwrite. "
                "Set a new data.id in the suite configuration.")
        if item.kind == "replay":
            buf = generate_replay(suite, item)
            content = buf.sha256()
            extra = {"coverage": buf.meta["coverage"], "n_transitions": len(buf), "n_resets": int(buf.meta["n_resets"])}
            def write(buf=buf, path=path):
                tmp = path.with_name(path.name + ".tmp.npz")              # atomic: a partial file is never left
                buf.save(tmp)
                replace_file(tmp, path)
        else:
            arrays, meta = generate_demos(suite, item) if item.kind == "demos" else generate_reference(suite, item)
            content = ds.arrays_sha256(arrays, meta)
            extra = {"shape": list(arrays["states"].shape)}
            if item.kind == "demos":
                extra["expert"] = meta["expert"]
                extra["expert_actions_sha256"] = meta["expert_actions_sha256"]
                if "certificate" in meta:
                    extra["certificate"] = meta["certificate"]
            def write(arrays=arrays, meta=meta, path=path):
                ds.save_arrays(path, arrays, meta)
        if entry is not None and path.exists():
            if entry.get("content_sha256") != content:
                raise DataConflict(
                    f"data item {key!r}: regenerated content {content} differs from the stored {entry.get('content_sha256')}; "
                    "refusing to overwrite (code or dependencies changed?). Set a new data.id.")
            if sha256_file(path) != entry.get("file_sha256"):
                raise DataConflict(f"data file {path} was modified after preparation; refusing to use or overwrite it")
            status = "verified"
        else:
            if path.exists():
                raise DataConflict(f"{path} exists but is not in the manifest; move it away or choose a new data.id")
            if entry is not None and entry.get("content_sha256") != content:
                raise DataConflict(
                    f"data item {key!r}: its file {path} is missing and the regenerated content {content} differs from "
                    f"the recorded {entry.get('content_sha256')}; refusing to replace recorded data with different data "
                    "(code or dependencies changed?). Restore the file, or set a new data.id.")
            path.parent.mkdir(parents=True, exist_ok=True)
            write()
            if item.kind == "replay":
                stored = ReplayBuffer.load(path)
                if stored.sha256() != content:
                    raise DataConflict(f"{path}: written replay does not reload with the same content hash")
            restored = entry is not None                  # same content as recorded: the missing file is rewritten
            new_entry = {"kind": item.kind, "spec": item.spec, "spec_sha256": item.spec_sha256, "content_sha256": content,
                         "file_sha256": sha256_file(path), "bytes": path.stat().st_size, "path": f"{key}.npz",
                         "created_utc": entry.get("created_utc", utc_now()) if restored else utc_now(),
                         "data_code_sha256": code["sha256"], **extra}
            if restored:
                new_entry["restored_utc"] = utc_now()
            entry = new_entry
            manifest["items"][key] = entry
            ds.save_manifest(data_root, data_id, manifest)
            status = "restored" if restored else "written"
        rec = {"key": key, "kind": item.kind, "status": status, "content_sha256": content,
               "seconds": time.perf_counter() - t0}
        if status in ("written", "restored"):
            manifest["items"][key]["generation_seconds"] = rec["seconds"]
            ds.save_manifest(data_root, data_id, manifest)
        report.append(rec)
        log(f"  {status:8s} {key}  ({rec['seconds']:.2f} s)")
    ds.save_manifest(data_root, data_id, manifest)
    return report


def missing_items(suite: SuiteConfig, data_root: Union[str, Path]) -> List[str]:
    """Items of the suite that are not yet in the manifest (or whose file is missing)."""
    man = ds.load_manifest(data_root, suite.data_id)
    out = []
    for key in data_items(suite):
        if key not in man["items"] or not ds.item_path(data_root, suite.data_id, key).exists():
            out.append(key)
    return out
