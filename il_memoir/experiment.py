"""Suite configurations, the frozen seed manifest and the deterministic job graph.

A *suite* (``configs/core.yaml``, ``compact_ablations.yaml``, ``reduced.yaml``,
``pilot.yaml``, ``smoke.yaml``) declares conditions (environment, layout,
goals), seeds, budgets, method settings and evaluation settings.  This module
validates it strictly (unknown keys are errors, as in ``config.py``), resolves
its seeds from ``configs/seeds.yaml`` and expands it into jobs:

* ``fb_pretrain``  task-blind FB pretraining, one per (environment, layout, seed, FB variant),
                   shared by every goal of the layout (spec section 7: cached per layout/seed);
* ``adapt``        ER_FB and DM_FB latents for one goal on a frozen FB model (no simulator data);
* ``fpg``          f-PG for one goal, divergence and f-PG variant;
* ``eval``         held-out evaluation of every checkpoint of one ``fpg`` or ``adapt`` job;
* ``eval_expert``  the same evaluation of the expert (reference row and noise floor).

Each job has a deterministic identifier ``<type>-<env>-<layout>[-<goal>][-<divergence>][-<variant>]-s<seed>-<h>``
where ``h`` is the first 10 hex digits of the SHA-256 of its *identity*: every
setting that can change its outputs (learner configuration, budgets,
checkpoint schedule, simulator hash, input data specifications, parent job
identifiers, stream labels).  Runner settings that only change how often a
resume checkpoint is written are not part of the identity.  Two suites that
declare the same job (for example the JS f-PG run of the core and of the
compact ablations) obtain the same identifier, so the job runs once and is
reused (spec section 7: "Reuse base runs rather than retraining them").

Random streams (``common.make_rng(seed, *labels)``; recorded per job):

==================  =====================================================
demonstrations      ``("demos", env, layout, goal)``
FB replay           ``("replay", env, layout)``
held-out reference  ``("heldout_reference", env, layout, goal)`` with the reference seed of the suite kind
f-PG                ``("fpg", env, layout, goal)`` + ``"policy_init"`` / ``"rollouts"`` (FPGLearner)
FB pretraining      ``("fb", env, layout)`` + ``"init"`` / ``"minibatches"`` (FBPretrainer); ``"diagnostics"``
adaptation          ``("adapt", env, layout, goal, "fallback")`` (used only if ``z_ER = 0``)
evaluation          ``("evaluation", env, layout, goal)``
==================  =====================================================

The f-PG and FB streams do not contain the divergence or the variant: ablation
variants of the same goal and seed share their initialization and first batch
(common random numbers, a paired comparison).  Evaluation streams are shared by
all methods of a goal and seed for the same reason.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from .config import EnvironmentConfig, load_environment_config
from .divergences import GENERATORS
from .fb import FBConfig, ReplayConfig
from .fb_il import DMConfig
from .fpg import FPGConfig
from .logging import PROJECT_ROOT, sha256_json

SUITE_SCHEMA = "il_memoir/suite-v1"
SEEDS_SCHEMA = "il_memoir/seeds-v1"
KINDS = ("final", "pilot", "smoke")
ENVS = ("finite", "continuous")
JOB_TYPES = ("fb_pretrain", "adapt", "fpg", "eval", "eval_expert")
DEFAULT_SEEDS = PROJECT_ROOT / "configs" / "seeds.yaml"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

FPG_CONTINUOUS_ONLY = frozenset({"density", "kde_bandwidth", "histogram_bins", "hidden", "n_hidden_layers",
                                 "activation", "out_init_scale"})


class ConfigError(ValueError):
    pass


# =============================================================================
# strict mapping helpers
# =============================================================================
def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _keys(section: Mapping[str, Any], what: str, required: Sequence[str], optional: Sequence[str] = ()) -> None:
    allowed = set(required) | set(optional)
    unknown = set(section) - allowed
    missing = set(required) - set(section)
    if unknown:
        raise ConfigError(f"unknown key(s) in {what}: {sorted(unknown)} (allowed: {sorted(allowed)})")
    if missing:
        raise ConfigError(f"missing key(s) in {what}: {sorted(missing)}")


def _int(value: Any, what: str, low: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ConfigError(f"{what} must be an integer >= {low}, got {value!r}")
    return int(value)


def _real(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{what} must be a number, got {value!r}")
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        raise ConfigError(f"{what} must be finite, got {value!r}")
    return v


def _name(value: Any, what: str) -> str:
    if not isinstance(value, str) or not NAME_RE.match(value):
        raise ConfigError(f"{what} must match [a-z0-9][a-z0-9_]*, got {value!r}")
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursive merge for configuration inheritance: mappings merge, lists and ``variants`` replace."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping) and key != "variants":
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_path(path: Union[str, Path], relative_to: Optional[Path] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if relative_to is not None and (relative_to / p).exists():
        return (relative_to / p).resolve()
    return (PROJECT_ROOT / p).resolve()


def project_relative(path: Union[str, Path]) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


# =============================================================================
# seed manifest
# =============================================================================
@dataclass(frozen=True)
class SeedManifest:
    final_seeds: Tuple[int, ...]
    pilot_seed: int
    smoke_seed: int
    reference_seeds: Dict[str, int]
    path: str
    sha256: str

    def seeds_for(self, kind: str) -> List[int]:
        if kind == "final":
            return list(self.final_seeds)
        if kind == "pilot":
            return [self.pilot_seed]
        if kind == "smoke":
            return [self.smoke_seed]
        raise ConfigError(f"unknown suite kind {kind!r}")

    def reference_seed(self, kind: str) -> int:
        return int(self.reference_seeds[kind])

    def to_dict(self) -> Dict[str, Any]:
        return {"final_seeds": list(self.final_seeds), "pilot_seed": self.pilot_seed, "smoke_seed": self.smoke_seed,
                "reference_seeds": dict(self.reference_seeds), "path": self.path, "sha256": self.sha256}


def load_seed_manifest(path: Union[str, Path, None] = None) -> SeedManifest:
    """Read ``configs/seeds.yaml``.  Final, pilot, smoke and reference seeds must all be distinct."""
    p = Path(path) if path is not None else DEFAULT_SEEDS
    with open(p, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw = _mapping(raw, "seeds.yaml")
    _keys(raw, "seeds.yaml", ["schema", "final_seeds", "pilot_seed", "smoke_seed", "reference_seeds"])
    if raw["schema"] != SEEDS_SCHEMA:
        raise ConfigError(f"seeds.yaml schema must be {SEEDS_SCHEMA!r}")
    finals = raw["final_seeds"]
    if not isinstance(finals, list) or not finals:
        raise ConfigError("final_seeds must be a non-empty list")
    finals = tuple(_int(s, "final seed") for s in finals)
    pilot = _int(raw["pilot_seed"], "pilot_seed")
    smoke = _int(raw["smoke_seed"], "smoke_seed")
    refs = _mapping(raw["reference_seeds"], "reference_seeds")
    _keys(refs, "reference_seeds", list(KINDS))
    refs = {k: _int(v, f"reference seed {k}") for k, v in refs.items()}
    every = list(finals) + [pilot, smoke] + list(refs.values())
    if len(set(every)) != len(every):
        raise ConfigError("final, pilot, smoke and reference seeds must all be distinct")
    return SeedManifest(finals, pilot, smoke, refs, project_relative(p), sha256_json(raw))


# =============================================================================
# suite configuration
# =============================================================================
@dataclass(frozen=True)
class Condition:
    env: str
    layout: str
    tasks: Tuple[str, ...]


@dataclass
class SuiteConfig:
    name: str
    kind: str
    description: str
    path: str
    raw: Dict[str, Any]                 # fully resolved (inheritance applied), canonical
    sha256: str
    env_cfg: EnvironmentConfig
    env_config_path: str
    seed_manifest: SeedManifest
    seeds: List[int]
    reference_seed: int
    conditions: List[Condition]

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw["data"]

    @property
    def methods(self) -> Dict[str, Any]:
        return self.raw["methods"]

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def runner(self) -> Dict[str, Any]:
        return self.raw["runner"]

    @property
    def pilot(self) -> Optional[Dict[str, Any]]:
        return self.raw.get("pilot")

    @property
    def data_id(self) -> str:
        return str(self.raw["data"]["id"])


TOP_REQUIRED = ["schema", "suite", "kind", "description", "environment_config", "seeds", "data", "grid", "methods",
                "evaluation", "runner"]
TOP_OPTIONAL = ["base", "pilot"]


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return dict(_mapping(raw, str(path)))


def _resolve_inheritance(path: Path, depth: int = 0) -> Dict[str, Any]:
    if depth > 5:
        raise ConfigError("configuration inheritance (base:) is nested more than 5 levels")
    raw = _read_yaml(path)
    base = raw.pop("base", None)
    if base is None:
        return raw
    base_path = _resolve_path(base, path.parent)
    parent = _resolve_inheritance(base_path, depth + 1)
    if "pilot" not in raw:
        parent.pop("pilot", None)          # a pilot section is never inherited
    return deep_merge(parent, raw)


def load_suite_config(path: Union[str, Path], seeds_path: Union[str, Path, None] = None) -> SuiteConfig:
    """Load, inherit (``base:``), validate and resolve a suite configuration."""
    p = _resolve_path(path)
    raw = _resolve_inheritance(p)
    _keys(raw, f"suite {p.name}", TOP_REQUIRED, TOP_OPTIONAL)
    if raw["schema"] != SUITE_SCHEMA:
        raise ConfigError(f"suite schema must be {SUITE_SCHEMA!r}, got {raw['schema']!r}")
    name = _name(raw["suite"], "suite")
    kind = raw["kind"]
    if kind not in KINDS:
        raise ConfigError(f"kind must be one of {KINDS}, got {kind!r}")
    if raw["seeds"] != kind:
        raise ConfigError(f"suite {name!r} of kind {kind!r} must declare seeds: {kind} (frozen seed manifest)")
    if kind != "pilot" and "pilot" in raw:
        raise ConfigError("only a pilot suite may have a 'pilot' section")
    if kind == "pilot" and "pilot" not in raw:
        raise ConfigError("a pilot suite needs a 'pilot' section (timing workloads and ETA suites)")
    seed_manifest = load_seed_manifest(seeds_path)
    env_path = _resolve_path(raw["environment_config"], p.parent)
    env_cfg = load_environment_config(env_path)
    raw = dict(raw)
    raw["environment_config"] = project_relative(env_path)
    conditions = _validate_grid(raw["grid"], env_cfg)
    _validate_data(raw["data"])
    _validate_methods(raw["methods"])
    _validate_evaluation(raw["evaluation"])
    _validate_runner(raw["runner"])
    if kind == "pilot":
        _validate_pilot(raw["pilot"])
    suite = SuiteConfig(
        name=name, kind=kind, description=str(raw["description"]).strip(), path=project_relative(p),
        raw=raw, sha256=sha256_json(raw), env_cfg=env_cfg, env_config_path=project_relative(env_path),
        seed_manifest=seed_manifest, seeds=seed_manifest.seeds_for(kind),
        reference_seed=seed_manifest.reference_seed(kind), conditions=conditions,
    )
    expand_jobs(suite)          # validates every learner configuration and detects duplicate variants
    return suite


def _validate_grid(grid: Any, env_cfg: EnvironmentConfig) -> List[Condition]:
    if not isinstance(grid, list) or not grid:
        raise ConfigError("grid must be a non-empty list of conditions")
    out, seen = [], set()
    for k, row in enumerate(grid):
        row = _mapping(row, f"grid[{k}]")
        _keys(row, f"grid[{k}]", ["env", "layout", "tasks"])
        env, layout = row["env"], row["layout"]
        if env not in ENVS:
            raise ConfigError(f"grid[{k}].env must be one of {ENVS}")
        if layout not in env_cfg.layout_names:
            raise ConfigError(f"grid[{k}].layout {layout!r} is not a layout of the environment configuration")
        if env == "continuous" and layout != env_cfg.continuous_layout:
            raise ConfigError(f"the continuous maze is defined for layout {env_cfg.continuous_layout!r} only")
        if (env, layout) in seen:
            raise ConfigError(f"grid lists ({env}, {layout}) twice; list all its goals in one row")
        seen.add((env, layout))
        tasks = row["tasks"]
        if not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks):
            raise ConfigError(f"grid[{k}].tasks must be a non-empty list without duplicates")
        for t in tasks:
            if t not in env_cfg.goal_names:
                raise ConfigError(f"unknown goal {t!r}; known: {env_cfg.goal_names}")
        out.append(Condition(env, layout, tuple(tasks)))
    return out


def _validate_data(data: Any) -> None:
    data = _mapping(data, "data")
    _keys(data, "data", ["id", "n_demos", "replay", "reference"])
    _name(data["id"].replace("-", "_") if isinstance(data["id"], str) else data["id"], "data.id")
    _int(data["n_demos"], "data.n_demos", 1)
    replay = _mapping(data["replay"], "data.replay")
    _keys(replay, "data.replay", [], list(ENVS))
    for env, spec in replay.items():
        spec = _mapping(spec, f"data.replay.{env}")
        _keys(spec, f"data.replay.{env}", ["n_transitions", "segment_length"])
        ReplayConfig(int(spec["n_transitions"]), int(spec["segment_length"]))
    ref = _mapping(data["reference"], "data.reference")
    _keys(ref, "data.reference", [], ["continuous"])
    if "continuous" in ref:
        spec = _mapping(ref["continuous"], "data.reference.continuous")
        _keys(spec, "data.reference.continuous", ["n_trajectories"])
        _int(spec["n_trajectories"], "data.reference.continuous.n_trajectories", 2)


def _validate_variants(variants: Any, what: str) -> None:
    variants = _mapping(variants, what)
    if not variants:
        raise ConfigError(f"{what} must declare at least one variant (e.g. base: {{}})")
    for vname, spec in variants.items():
        _name(vname, f"{what} name")
        spec = _mapping(spec if spec is not None else {}, f"{what}.{vname}")
        _keys(spec, f"{what}.{vname}", [], ["envs", "all", "finite", "continuous"])
        if "envs" in spec and (not isinstance(spec["envs"], list) or not set(spec["envs"]) <= set(ENVS)):
            raise ConfigError(f"{what}.{vname}.envs must be a list of {ENVS}")


def _validate_methods(methods: Any) -> None:
    methods = _mapping(methods, "methods")
    _keys(methods, "methods", ["fpg", "fb", "adaptation"])
    fpg = _mapping(methods["fpg"], "methods.fpg")
    _keys(fpg, "methods.fpg", ["divergences", "variants"], list(ENVS))
    divs = fpg["divergences"]
    if not isinstance(divs, list) or not divs or not set(divs) <= set(GENERATORS) or len(set(divs)) != len(divs):
        raise ConfigError(f"methods.fpg.divergences must be a list of distinct names from {GENERATORS}")
    for env in ENVS:
        if env in fpg:
            block = _mapping(fpg[env], f"methods.fpg.{env}")
            _keys(block, f"methods.fpg.{env}", ["settings", "budget_transitions", "eval_every_iterations"])
    _validate_variants(fpg["variants"], "methods.fpg.variants")
    fb = _mapping(methods["fb"], "methods.fb")
    _keys(fb, "methods.fb", ["variants"], list(ENVS))
    for env in ENVS:
        if env in fb:
            block = _mapping(fb[env], f"methods.fb.{env}")
            _keys(block, f"methods.fb.{env}", ["settings", "snapshot_every", "log_every_updates"])
    _validate_variants(fb["variants"], "methods.fb.variants")
    ad = _mapping(methods["adaptation"], "methods.adaptation")
    _keys(ad, "methods.adaptation", ["dm", "dm_snapshot_steps", "fb_snapshots"])
    DMConfig.from_dict(dict(_mapping(ad["dm"], "methods.adaptation.dm")))
    if ad["fb_snapshots"] not in ("all", "final"):
        raise ConfigError("methods.adaptation.fb_snapshots must be 'all' or 'final'")


def _validate_evaluation(ev: Any) -> None:
    ev = _mapping(ev, "evaluation")
    _keys(ev, "evaluation", ["device", "finite", "continuous", "convergence"])
    if ev["device"] not in ("cpu", "run"):
        raise ConfigError("evaluation.device must be 'cpu' or 'run' (the suite device)")
    fin = _mapping(ev["finite"], "evaluation.finite")
    _keys(fin, "evaluation.finite", ["mc_rollouts"])
    _int(fin["mc_rollouts"], "evaluation.finite.mc_rollouts", 2)
    con = _mapping(ev["continuous"], "evaluation.continuous")
    _keys(con, "evaluation.continuous", ["rollouts"])
    _int(con["rollouts"], "evaluation.continuous.rollouts", 2)
    conv = _mapping(ev["convergence"], "evaluation.convergence")
    _keys(conv, "evaluation.convergence", ["js_max_nats", "success_min"])
    js = _real(conv["js_max_nats"], "js_max_nats")
    sm = _real(conv["success_min"], "success_min")
    if not (0.0 < js <= 0.6931471805599453 and 0.0 <= sm <= 1.0):
        raise ConfigError("convergence: 0 < js_max_nats <= log 2 (nats) and 0 <= success_min <= 1")


def _validate_runner(r: Any) -> None:
    r = _mapping(r, "runner")
    _keys(r, "runner", ["cpu_threads", "checkpoint_every_updates", "checkpoint_every_seconds"])
    _int(r["cpu_threads"], "runner.cpu_threads", 1)
    _int(r["checkpoint_every_updates"], "runner.checkpoint_every_updates", 1)
    _real(r["checkpoint_every_seconds"], "runner.checkpoint_every_seconds")


def _validate_pilot(p: Any) -> None:
    p = _mapping(p, "pilot")
    _keys(p, "pilot", ["timing", "eta_suites", "overhead"])
    timing = _mapping(p["timing"], "pilot.timing")
    _keys(timing, "pilot.timing", list(ENVS), ["repeats"])
    for env in ENVS:
        t = _mapping(timing[env], f"pilot.timing.{env}")
        _keys(t, f"pilot.timing.{env}", ["replay_transitions", "fb_updates", "fpg_iterations", "dm_steps", "eval_rollouts"])
        for k, v in t.items():
            _int(v, f"pilot.timing.{env}.{k}", 1)
    if not isinstance(p["eta_suites"], list) or not p["eta_suites"]:
        raise ConfigError("pilot.eta_suites must list the suite configurations to extrapolate")
    if _real(p["overhead"], "pilot.overhead") < 0.5:
        raise ConfigError("pilot.overhead must be at least 0.5 (INTERFACE: at least 50% overhead)")


# =============================================================================
# resolved learner settings
# =============================================================================
def _variant_block(variant: Mapping[str, Any], env: str) -> Dict[str, Any]:
    """Overrides of one variant for ``env``: ``all`` first, then the env-specific block."""
    out: Dict[str, Any] = {}
    if variant.get("all"):
        out = deep_merge(out, variant["all"])
    if variant.get(env):
        out = deep_merge(out, variant[env])
    return out


def _variant_applies(variant: Mapping[str, Any], env: str) -> bool:
    return env in variant.get("envs", ENVS)


def fpg_settings(suite: SuiteConfig, env: str, variant: str, divergence: str) -> Tuple[FPGConfig, Dict[str, Any]]:
    block = suite.methods["fpg"].get(env)
    if block is None:
        raise ConfigError(f"methods.fpg.{env} is required for a grid row with env {env}")
    vspec = suite.methods["fpg"]["variants"][variant] or {}
    block = deep_merge(block, _variant_block(vspec, env))
    _keys(block, f"methods.fpg.{env} (variant {variant})", ["settings", "budget_transitions", "eval_every_iterations"])
    settings = dict(_mapping(block["settings"], f"methods.fpg.{env}.settings"))
    if "divergence" in settings:
        raise ConfigError("the f-PG divergence is set by methods.fpg.divergences, not in settings")
    if env == "finite":
        bad = set(settings) & FPG_CONTINUOUS_ONLY
        if bad:
            raise ConfigError(f"continuous-only f-PG settings for a finite job: {sorted(bad)}")
    settings["divergence"] = divergence
    cfg = FPGConfig.from_dict(settings)
    budget = _int(block["budget_transitions"], "budget_transitions", 1)
    every = _int(block["eval_every_iterations"], "eval_every_iterations", 1)
    per_iter = cfg.n_traj * suite.env_cfg.horizon
    if budget % per_iter:
        raise ConfigError(f"f-PG budget {budget} is not a multiple of n_traj * H = {per_iter} transitions")
    n_iter = budget // per_iter
    checkpoints = sorted(set(list(range(0, n_iter + 1, every)) + [n_iter]))
    return cfg, {"budget_transitions": budget, "n_iterations": n_iter, "eval_every_iterations": every,
                 "policy_checkpoints": checkpoints}


def fb_settings(suite: SuiteConfig, env: str, variant: str) -> Tuple[FBConfig, Dict[str, Any]]:
    block = suite.methods["fb"].get(env)
    if block is None:
        raise ConfigError(f"methods.fb.{env} is required for a grid row with env {env}")
    vspec = suite.methods["fb"]["variants"][variant] or {}
    block = deep_merge(block, _variant_block(vspec, env))
    _keys(block, f"methods.fb.{env} (variant {variant})", ["settings", "snapshot_every", "log_every_updates"])
    cfg = FBConfig.from_dict(dict(_mapping(block["settings"], f"methods.fb.{env}.settings")))
    every = _int(block["snapshot_every"], "snapshot_every", 1)
    log_every = _int(block["log_every_updates"], "log_every_updates", 1)
    snaps = sorted(set(list(range(every, cfg.n_updates + 1, every)) + [cfg.n_updates]))
    return cfg, {"snapshot_updates": snaps, "log_every_updates": log_every}


def dm_settings(suite: SuiteConfig) -> Tuple[DMConfig, Dict[str, Any]]:
    ad = suite.methods["adaptation"]
    cfg = DMConfig.from_dict(dict(ad["dm"]))
    steps = ad["dm_snapshot_steps"]
    if not isinstance(steps, list) or not steps:
        raise ConfigError("dm_snapshot_steps must be a non-empty list")
    steps = sorted(set(_int(s, "dm snapshot step") for s in steps) | {cfg.steps, 0})
    if steps[-1] > cfg.steps:
        raise ConfigError("dm_snapshot_steps must not exceed methods.adaptation.dm.steps")
    return cfg, {"dm_snapshot_steps": steps, "fb_snapshots": ad["fb_snapshots"]}


def evaluation_settings(suite: SuiteConfig, env: str) -> Dict[str, Any]:
    ev = suite.evaluation
    if env == "finite":
        return {"device": ev["device"], "exact": "oracle", "mc_rollouts": int(ev["finite"]["mc_rollouts"])}
    return {"device": ev["device"], "rollouts": int(ev["continuous"]["rollouts"]),
            "occupancy_estimator": {"kind": "free_space_histogram", "bins": int(suite.env_cfg.histogram_bins),
                                    "weights": "gamma^t / (n Z_H) on s_{t+1}", "smoothing": None}}


# =============================================================================
# environment hashes
# =============================================================================
def sim_sha256(cfg: EnvironmentConfig) -> str:
    """Everything that can change a learner's data: the learner view plus the transition parameters."""
    return sha256_json({
        "learner_view": cfg.learner_dict(),
        "finite_slip_prob": cfg.slip_prob,
        "continuous": {"step_cells": cfg.step_cells, "noise_std": cfg.noise_std, "max_noise_redraws": cfg.max_noise_redraws},
    })


# =============================================================================
# data items
# =============================================================================
@dataclass(frozen=True)
class DataItem:
    key: str
    kind: str                # demos | replay | reference
    spec: Dict[str, Any]
    spec_sha256: str

    @property
    def relpath(self) -> str:
        return self.key + ".npz"


def demos_item(suite: SuiteConfig, env: str, layout: str, task: str, seed: int) -> DataItem:
    spec = {"kind": "demos", "env": env, "layout": layout, "task": task, "seed": int(seed),
            "n_trajectories": int(suite.data["n_demos"]), "horizon": int(suite.env_cfg.horizon),
            "env_sha256": suite.env_cfg.sha256(), "stream": ["demos", env, layout, task],
            "expert": "finite_oracle_optimal" if env == "finite" else "route_controller"}
    return DataItem(f"demos/{env}/{layout}/{task}/s{seed}", "demos", spec, sha256_json(spec))


def replay_item(suite: SuiteConfig, env: str, layout: str, seed: int) -> DataItem:
    rep = suite.data["replay"].get(env)
    if rep is None:
        raise ConfigError(f"data.replay.{env} is required for a grid row with env {env}")
    spec = {"kind": "replay", "env": env, "layout": layout, "seed": int(seed),
            "n_transitions": int(rep["n_transitions"]), "segment_length": int(rep["segment_length"]),
            "sim_sha256": sim_sha256(suite.env_cfg), "stream": ["replay", env, layout],
            "behaviour": "uniform random actions; segments from uniform free-space resets (fb.collect_replay)"}
    return DataItem(f"replay/{env}/{layout}/s{seed}", "replay", spec, sha256_json(spec))


def reference_item(suite: SuiteConfig, env: str, layout: str, task: str) -> Optional[DataItem]:
    if env != "continuous":
        return None          # finite evaluation is exact (oracle): no reference sample
    ref = suite.data["reference"].get("continuous")
    if ref is None:
        raise ConfigError("data.reference.continuous is required for continuous conditions")
    rseed = int(suite.reference_seed)
    spec = {"kind": "reference", "env": env, "layout": layout, "task": task, "seed": rseed,
            "n_trajectories": int(ref["n_trajectories"]), "horizon": int(suite.env_cfg.horizon),
            "env_sha256": suite.env_cfg.sha256(), "stream": ["heldout_reference", env, layout, task],
            "expert": "route_controller"}
    return DataItem(f"reference/{env}/{layout}/{task}/r{rseed}", "reference", spec, sha256_json(spec))


def _item_ref(suite: SuiteConfig, item: Optional[DataItem]) -> Optional[Dict[str, str]]:
    if item is None:
        return None
    return {"data_id": suite.data_id, "key": item.key, "spec_sha256": item.spec_sha256}


# =============================================================================
# jobs
# =============================================================================
@dataclass(frozen=True)
class JobSpec:
    job_id: str
    type: str
    identity: Dict[str, Any]
    identity_sha256: str
    labels: Dict[str, Any]
    deps: Tuple[str, ...] = ()
    data_inputs: Dict[str, str] = field(default_factory=dict)   # role -> data item key

    def to_dict(self) -> Dict[str, Any]:
        return {"job_id": self.job_id, "type": self.type, "identity": self.identity,
                "identity_sha256": self.identity_sha256, "labels": self.labels, "deps": list(self.deps),
                "data_inputs": dict(self.data_inputs)}


def _make_job(prefix: Sequence[str], jtype: str, identity: Dict[str, Any], labels: Dict[str, Any],
              deps: Sequence[str] = (), data_inputs: Optional[Dict[str, str]] = None) -> JobSpec:
    h = sha256_json(identity)
    job_id = "-".join(str(p) for p in prefix) + "-" + h[:10]
    return JobSpec(job_id, jtype, identity, h, labels, tuple(deps), dict(data_inputs or {}))


def _vtag(variant: str) -> List[str]:
    return [] if variant == "base" else [variant]


def expand_jobs(suite: SuiteConfig) -> List[JobSpec]:
    """All jobs of a suite, in execution order (dependencies first), without duplicates.

    Order: for each seed, for each grid row: the FB pretraining runs; then for each goal the
    expert evaluation, each FB variant's adaptation and its evaluation, and each f-PG run and its
    evaluation.  Complete seeds therefore finish first if a suite is stopped early.
    """
    cfg = suite.env_cfg
    sim = sim_sha256(cfg)
    env_hash = cfg.sha256()
    dm_cfg, dm_extra = dm_settings(suite)
    jobs: List[JobSpec] = []
    by_identity: Dict[str, str] = {}

    def add(job: JobSpec) -> JobSpec:
        prev = by_identity.get(job.identity_sha256)
        if prev is not None and prev != job.job_id:
            raise ConfigError(f"jobs {prev} and {job.job_id} resolve to identical settings; remove the duplicate variant")
        if prev is None:
            by_identity[job.identity_sha256] = job.job_id
            jobs.append(job)
        return job

    fpg_vars = suite.methods["fpg"]["variants"]
    fb_vars = suite.methods["fb"]["variants"]
    for seed in suite.seeds:
        for cond in suite.conditions:
            env, layout = cond.env, cond.layout
            fb_jobs: Dict[str, JobSpec] = {}
            rep = replay_item(suite, env, layout, seed)
            for vname, vspec in fb_vars.items():
                if not _variant_applies(vspec or {}, env):
                    continue
                fbc, extra = fb_settings(suite, env, vname)
                identity = {"type": "fb_pretrain", "env": env, "layout": layout, "seed": int(seed), "sim_sha256": sim,
                            "replay": _item_ref(suite, rep), "fb": fbc.to_dict(), **extra,
                            "stream": ["fb", env, layout]}
                labels = {"type": "fb_pretrain", "env": env, "layout": layout, "seed": int(seed), "method": "fb",
                          "fb_variant": vname}
                fb_jobs[vname] = add(_make_job(["fb", env, layout] + _vtag(vname) + [f"s{seed}"], "fb_pretrain",
                                               identity, labels, (), {"replay": rep.key}))
            for task in cond.tasks:
                demos = demos_item(suite, env, layout, task, seed)
                ref = reference_item(suite, env, layout, task)
                ev = evaluation_settings(suite, env)
                eval_stream = ["evaluation", env, layout, task]
                data_eval = {"reference": ref.key} if ref is not None else {}
                identity = {"type": "eval_expert", "env": env, "layout": layout, "task": task, "seed": int(seed),
                            "env_sha256": env_hash, "evaluation": ev, "reference": _item_ref(suite, ref),
                            "demos": _item_ref(suite, demos), "stream": eval_stream}
                labels = {"type": "eval_expert", "env": env, "layout": layout, "task": task, "seed": int(seed),
                          "method": "expert"}
                add(_make_job(["evalexp", env, layout, task, f"s{seed}"], "eval_expert", identity, labels, (),
                              dict(data_eval, demos=demos.key)))
                for vname, fbj in fb_jobs.items():
                    identity = {"type": "adapt", "env": env, "layout": layout, "task": task, "seed": int(seed),
                                "fb_job": fbj.job_id, "demos": _item_ref(suite, demos), "er": True,
                                "dm": dm_cfg.to_dict(), **dm_extra, "stream": ["adapt", env, layout, task, "fallback"]}
                    labels = {"type": "adapt", "env": env, "layout": layout, "task": task, "seed": int(seed),
                              "method": "fb_il", "fb_variant": vname}
                    adj = add(_make_job(["adapt", env, layout, task] + _vtag(vname) + [f"s{seed}"], "adapt",
                                        identity, labels, (fbj.job_id,), {"demos": demos.key}))
                    identity = {"type": "eval", "target": adj.job_id, "target_type": "adapt", "env": env,
                                "layout": layout, "task": task, "seed": int(seed), "env_sha256": env_hash,
                                "evaluation": ev, "reference": _item_ref(suite, ref), "stream": eval_stream}
                    labels = dict(labels, type="eval", target_type="adapt")
                    add(_make_job(["eval", "adapt", env, layout, task] + _vtag(vname) + [f"s{seed}"], "eval",
                                  identity, labels, (adj.job_id,), data_eval))
                for div in suite.methods["fpg"]["divergences"]:
                    for vname, vspec in fpg_vars.items():
                        if not _variant_applies(vspec or {}, env):
                            continue
                        fcfg, extra = fpg_settings(suite, env, vname, div)
                        identity = {"type": "fpg", "env": env, "layout": layout, "task": task, "seed": int(seed),
                                    "divergence": div, "sim_sha256": sim, "demos": _item_ref(suite, demos),
                                    "fpg": fcfg.to_dict(), **extra, "stream": ["fpg", env, layout, task]}
                        labels = {"type": "fpg", "env": env, "layout": layout, "task": task, "seed": int(seed),
                                  "method": "fpg", "divergence": div, "fpg_variant": vname}
                        fj = add(_make_job(["fpg", env, layout, task, div] + _vtag(vname) + [f"s{seed}"], "fpg",
                                           identity, labels, (), {"demos": demos.key}))
                        identity = {"type": "eval", "target": fj.job_id, "target_type": "fpg", "env": env,
                                    "layout": layout, "task": task, "seed": int(seed), "env_sha256": env_hash,
                                    "evaluation": ev, "reference": _item_ref(suite, ref), "stream": eval_stream}
                        labels = dict(labels, type="eval", target_type="fpg")
                        add(_make_job(["eval", "fpg", env, layout, task, div] + _vtag(vname) + [f"s{seed}"], "eval",
                                      identity, labels, (fj.job_id,), data_eval))
    return jobs


def data_items(suite: SuiteConfig) -> Dict[str, DataItem]:
    """Every data item the suite's jobs read, keyed by item key (demos, replay, held-out reference)."""
    items: Dict[str, DataItem] = {}
    for seed in suite.seeds:
        for cond in suite.conditions:
            rep = replay_item(suite, cond.env, cond.layout, seed)
            items[rep.key] = rep
            for task in cond.tasks:
                d = demos_item(suite, cond.env, cond.layout, task, seed)
                items[d.key] = d
                ref = reference_item(suite, cond.env, cond.layout, task)
                if ref is not None:
                    items[ref.key] = ref
    return items


def seed_manifest_record(suite: SuiteConfig, jobs: Sequence[JobSpec]) -> Dict[str, Any]:
    """The frozen seed record of a suite: seeds, reference seed and every job's seed and stream labels."""
    return {
        "suite": suite.name,
        "kind": suite.kind,
        "seed_manifest": suite.seed_manifest.to_dict(),
        "seeds": list(suite.seeds),
        "reference_seed": suite.reference_seed,
        "jobs": {j.job_id: {"seed": j.identity.get("seed"), "stream": j.identity.get("stream")} for j in jobs},
        "data_streams": {k: {"seed": v.spec["seed"], "stream": v.spec["stream"]} for k, v in data_items(suite).items()},
    }
