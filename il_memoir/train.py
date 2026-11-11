"""Training executors: f-PG, task-blind FB pretraining and ER/DM adaptation jobs (METHOD_MAP sections 3-6).

The runner (:mod:`il_memoir.runner`) calls one executor per job with a
:class:`JobContext`.  An executor builds the simulator and the learner from the
job's identity, feeds the learner only what METHOD_MAP section 1 allows (the
learner view ``EnvironmentConfig.learner_dict()``, the simulator, the
demonstration **states**, the task-blind replay), writes its checkpoints and
outputs inside the job directory, and returns the job result.  It never imports
the oracle, the evaluation reward, the expert or the evaluation module
(``tests/test_no_leakage.py`` checks this statically).

Checkpoints and resume (INTERFACE.md)
-------------------------------------
* f-PG and FB pretraining save a resume checkpoint (``checkpoints/resume.pt``:
  learner state with optimizer and random-generator states, the number of log
  records written, the output records and the accumulated phase timers)
  after every declared output checkpoint, every
  ``runner.checkpoint_every_updates`` updates and every
  ``runner.checkpoint_every_seconds`` seconds.  Resuming restores that state,
  truncates the metric log to the checkpointed number of records and continues;
  on the CPU the result equals an uninterrupted run bit for bit
  (``tests/test_runner.py``).
* Ctrl+C (``logging.STOP``) is checked after every update: the job writes a
  resume checkpoint and raises :class:`~il_memoir.logging.JobInterrupted`.
* An adaptation job (ER and at most 300 DM steps per FB snapshot, a few
  seconds) has no partial state worth keeping: an interrupted one restarts.

Outputs
-------
* f-PG: ``outputs/policies/iter_XXXXXX.pt`` (``fpg.policy_record``) at the
  declared iterations, including 0 (the initial policy) and the last;
* FB: ``outputs/models/fb_uXXXXXX.pt`` (the frozen online copy with
  ``G_hat_B``, ``FBPretrainer.model``) at the declared update counts, with
  ``fb.diagnostics`` for each;
* adaptation: ``outputs/latents.json`` with, for each FB snapshot, ``z_ER``
  (raw and projected, with its norm), the DM iterates at the declared steps,
  the full DM log and the U-2 diagnostics (policy entropy and score spread / tau
  at the DM start states).
Each training output is recorded with its SHA-256 so that evaluation can verify
what it reads; ``outputs/probe.npz`` keeps the final policy probabilities on
the demonstration states for the reload check of ``scripts/smoke.py``.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import torch

from . import fpg as fpg_mod
from .common import make_rng
from .config import EnvironmentConfig
from .datasets import load_demo_states, replay_path
from .env_continuous import make_continuous_maze
from .env_tabular import make_grid_maze, make_layout
from .fb import FBConfig, FBModel, FBPretrainer, ReplayBuffer, diagnostics as fb_diagnostics, policy_scores, replay_coverage
from .fb_il import DMConfig, dm_fit, dm_targets, er_fb_z
from .fpg import FPGConfig, FPGLearner
from .hardware import PhaseTimer, memory_record, sync
from .logging import (
    JobContext,
    JobInterrupted,
    JsonlLog,
    StopController,
    load_torch,
    relpath,
    save_torch_atomic,
    sha256_array,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from .nets import ObsSpec

RESUME = "resume.pt"


def make_env(cfg: EnvironmentConfig, env: str, layout: str):
    if env == "finite":
        return make_grid_maze(cfg, layout)
    if env == "continuous":
        return make_continuous_maze(cfg, layout)
    raise ValueError(f"unknown environment {env!r}")


def _demo_states(ctx: JobContext) -> np.ndarray:
    entry = ctx.data_entries["demos"]
    return load_demo_states(ctx.data_root, ctx.data_id, ctx.job.data_inputs["demos"], entry["content_sha256"])


class _Resume:
    """Resume checkpoint of one job: written atomically, refused if it belongs to another job or code."""

    def __init__(self, ctx: JobContext, code_sha256: str):
        self.ctx = ctx
        self.path = ctx.job_dir / "checkpoints" / RESUME
        self.code_sha256 = code_sha256
        self.last_save = time.perf_counter()

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        rec = load_torch(self.path)
        if rec.get("job_id") != self.ctx.job.job_id or rec.get("identity_sha256") != self.ctx.job.identity_sha256:
            raise RuntimeError(f"{self.path} belongs to another job")
        if rec.get("code_sha256") != self.code_sha256:
            raise RuntimeError(f"{self.path} was written by other code; the runner must invalidate this job")
        return rec

    def save(self, payload: Dict[str, Any], timer: PhaseTimer) -> None:
        t0 = time.perf_counter()
        rec = {"job_id": self.ctx.job.job_id, "identity_sha256": self.ctx.job.identity_sha256,
               "code_sha256": self.code_sha256, "saved_utc": utc_now(), **payload}
        rec["timer"] = timer.as_dict()
        save_torch_atomic(rec, self.path)
        timer.add("checkpoint", time.perf_counter() - t0)
        self.last_save = time.perf_counter()

    def due(self, n_done: int) -> bool:
        every = int(self.ctx.runner.get("checkpoint_every_updates", 1000))
        seconds = float(self.ctx.runner.get("checkpoint_every_seconds", 120.0))
        return n_done % every == 0 or (time.perf_counter() - self.last_save) >= seconds


def _out(ctx: JobContext, path: Path) -> Dict[str, Any]:
    return {"path": path.relative_to(ctx.job_dir).as_posix(), "sha256": sha256_file(path)}


# =============================================================================
# f-PG
# =============================================================================
def run_fpg(ctx: JobContext, code_sha256: str) -> Dict[str, Any]:
    ident = ctx.identity
    cfg = ctx.env_cfg
    env = make_env(cfg, ident["env"], ident["layout"])
    view = cfg.learner_dict()                       # METHOD_MAP section 1: the learner view only
    demo_states = _demo_states(ctx)                 # states only
    fcfg = FPGConfig.from_dict(ident["fpg"])
    learner = FPGLearner(env, view, demo_states, fcfg, seed=int(ident["seed"]), stream=tuple(ident["stream"]),
                         device=ctx.device)
    log = JsonlLog(ctx.job_dir / "log.jsonl")
    resume = _Resume(ctx, code_sha256)
    timer = PhaseTimer(ctx.device)
    outputs: List[Dict[str, Any]] = []
    rec = resume.load()
    discarded = 0
    if rec is not None:
        learner.load_state_dict(rec["learner"])
        discarded = log.truncate(int(rec["log_records"]))
        outputs = list(rec["outputs"])
        timer = PhaseTimer(ctx.device, rec["timer"])
        ctx.log(f"    resumed f-PG at iteration {learner.iteration} ({discarded} log records after the checkpoint discarded)")
    elif log.path.exists():
        log.truncate(0)
    n_iter = int(ident["n_iterations"])
    checkpoints = set(int(k) for k in ident["policy_checkpoints"])
    pol_dir = ctx.job_dir / "outputs" / "policies"

    def save_policy() -> None:
        it = learner.iteration
        path = pol_dir / f"iter_{it:06d}.pt"
        t0 = time.perf_counter()
        save_torch_atomic(fpg_mod.policy_record(learner.policy), path)
        timer.add("policy_checkpoint", time.perf_counter() - t0)
        outputs[:] = [o for o in outputs if o["iteration"] != it]
        outputs.append({"iteration": it, "env_steps": int(learner.env_steps), "n_updates": int(learner.n_updates),
                        **_out(ctx, path)})

    def payload() -> Dict[str, Any]:
        return {"learner": learner.state_dict(), "log_records": log.durable_count(), "outputs": list(outputs)}

    if learner.iteration == 0 and not any(o["iteration"] == 0 for o in outputs):
        save_policy()
        resume.save(payload(), timer)
    while learner.iteration < n_iter:
        with timer.phase("train"):
            m = learner.step(env)
        log.append(m)
        if learner.iteration in checkpoints:
            save_policy()
            resume.save(payload(), timer)
        elif resume.due(learner.iteration):
            resume.save(payload(), timer)
        if ctx.stop.safe_point() and learner.iteration < n_iter:
            resume.save(payload(), timer)
            raise JobInterrupted(f"f-PG stopped at iteration {learner.iteration} of {n_iter}")
    if learner.iteration not in {o["iteration"] for o in outputs}:
        save_policy()
    probe_states = demo_states.reshape((-1,) + demo_states.shape[2:])
    probs = learner.policy_callable()(probe_states)
    (ctx.job_dir / "outputs").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ctx.job_dir / "outputs" / "probe.npz", states=probe_states, probs=probs)
    records = log.records()
    inner = {k: float(sum(r[k] for r in records)) for k in ("time_collect_s", "time_signal_s", "time_update_s")}
    return {
        "outputs": {"policies": sorted(outputs, key=lambda o: o["iteration"]),
                    "probe": _out(ctx, ctx.job_dir / "outputs" / "probe.npz"), "log": "log.jsonl"},
        "costs": {
            "phase": "f-PG per-task learning",
            "env_steps_train": int(learner.env_steps),
            "iterations": int(learner.iteration),
            "updates": int(learner.n_updates),
            "trajectories_per_iteration": int(fcfg.n_traj),
            "demonstrations": {"n_trajectories": int(demo_states.shape[0]), "transitions": int(demo_states.shape[0] * (demo_states.shape[1] - 1))},
            "wall_s": timer.as_dict(),
            "inner_s": inner,
            "discarded_log_records_on_resume": int(discarded),
        },
        "summary": {"resolved": learner.resolved, "final_training_metrics": records[-1] if records else None},
    }


# =============================================================================
# FB pretraining
# =============================================================================
def _check_coverage(cov: Dict[str, Any], env, cfg: EnvironmentConfig, kind: str, layout: str) -> Dict[str, Any]:
    if kind == "finite":
        ok = cov["states_visited"] == env.n_states and cov["min_state_action_count"] > 0
        out = {"states_visited": int(cov["states_visited"]), "n_states": int(env.n_states),
               "min_state_action_count": int(cov["min_state_action_count"])}
    else:
        n_free = make_layout(cfg, layout).n_free
        ok = cov["cells_visited"] == n_free and cov["min_cell_action_count"] > 0
        out = {"cells_visited": int(cov["cells_visited"]), "n_free_cells": int(n_free),
               "min_cell_action_count": int(cov["min_cell_action_count"])}
    out["covers_all"] = bool(ok)
    if not ok:
        raise RuntimeError(f"replay coverage incomplete, FB pretraining refused: {out}")
    return out


def run_fb_pretrain(ctx: JobContext, code_sha256: str) -> Dict[str, Any]:
    ident = ctx.identity
    cfg = ctx.env_cfg
    env = make_env(cfg, ident["env"], ident["layout"])
    spec = ObsSpec.from_env(env)
    entry = ctx.data_entries["replay"]
    timer = PhaseTimer(ctx.device)
    with timer.phase("load"):
        buf = ReplayBuffer.load(replay_path(ctx.data_root, ctx.data_id, ctx.job.data_inputs["replay"]))
    if buf.sha256() != entry["content_sha256"]:
        raise RuntimeError("replay buffer content differs from the data manifest")
    coverage = _check_coverage(replay_coverage(buf, env.n_actions, cfg.grid_size), env, cfg, ident["env"], ident["layout"])
    fcfg = FBConfig.from_dict(ident["fb"])
    trainer = FBPretrainer(spec, env.n_actions, cfg.gamma, fcfg, seed=int(ident["seed"]), stream=tuple(ident["stream"]),
                           device=ctx.device)
    trainer.attach_buffer(buf)
    log = JsonlLog(ctx.job_dir / "log.jsonl")
    resume = _Resume(ctx, code_sha256)
    snapshots: List[Dict[str, Any]] = []
    rec = resume.load()
    discarded = 0
    if rec is not None:
        trainer.load_state_dict(rec["trainer"])
        discarded = log.truncate(int(rec["log_records"]))
        snapshots = list(rec["outputs"])
        prior = dict(rec["timer"])
        prior["load"] = prior.get("load", 0.0) + timer.totals.get("load", 0.0)
        timer = PhaseTimer(ctx.device, prior)
        ctx.log(f"    resumed FB pretraining at update {trainer.n_updates} ({discarded} log records discarded)")
    elif log.path.exists():
        log.truncate(0)
    snap_at = set(int(u) for u in ident["snapshot_updates"])
    log_every = int(ident["log_every_updates"])
    model_dir = ctx.job_dir / "outputs" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    def payload() -> Dict[str, Any]:
        return {"trainer": trainer.state_dict(), "log_records": log.durable_count(), "outputs": list(snapshots)}

    while trainer.n_updates < fcfg.n_updates:
        with timer.phase("train"):
            m = trainer.update()
        k = trainer.n_updates
        if k % log_every == 0 or k == fcfg.n_updates:
            log.append(m)
        if k in snap_at:
            with timer.phase("snapshot"):
                model = trainer.model()
                path = model_dir / f"fb_u{k:06d}.pt"
                model.save(path)
                diag = fb_diagnostics(model, buf, make_rng(int(ident["seed"]), *ident["stream"], "diagnostics", k))
            snapshots[:] = [s for s in snapshots if s["update"] != k]
            snapshots.append({"update": int(k), "diagnostics": diag, **_out(ctx, path)})
            resume.save(payload(), timer)
        elif resume.due(k):
            resume.save(payload(), timer)
        if ctx.stop.safe_point() and trainer.n_updates < fcfg.n_updates:
            resume.save(payload(), timer)
            raise JobInterrupted(f"FB pretraining stopped at update {trainer.n_updates} of {fcfg.n_updates}")
    final = max(snapshots, key=lambda s: s["update"])
    model = FBModel.load(ctx.job_dir / final["path"], ctx.device)
    probe = buf.obs[: min(len(buf), 512)]
    z_probe = np.asarray(make_rng(int(ident["seed"]), *ident["stream"], "probe").standard_normal(fcfg.d))
    z_probe = fcfg.radius * z_probe / np.linalg.norm(z_probe)
    probs = model.numpy_policy(z_probe)(probe)
    np.savez_compressed(ctx.job_dir / "outputs" / "probe.npz", states=probe, z=z_probe, probs=probs)
    data_meta = ctx.data_entries["replay"]
    return {
        "outputs": {"snapshots": sorted(snapshots, key=lambda s: s["update"]),
                    "probe": _out(ctx, ctx.job_dir / "outputs" / "probe.npz"), "log": "log.jsonl"},
        "costs": {
            "phase": "task-blind FB pretraining (shared by every goal of the layout and seed)",
            "replay_transitions": int(len(buf)),
            "replay_resets": int(buf.meta.get("n_resets", 0)),
            "replay_generation_s": data_meta.get("generation_seconds"),
            "updates": int(trainer.n_updates),
            "batch_size": int(fcfg.batch_size),
            "target_batch_size": int(fcfg.target_batch_size),
            "wall_s": timer.as_dict(),
            "discarded_log_records_on_resume": int(discarded),
        },
        "summary": {"coverage": coverage, "final_diagnostics": final["diagnostics"],
                    "buffer_sha256": buf.sha256(), "final_training_metrics": log.records()[-1] if log.count() else None},
    }


# =============================================================================
# ER_FB and DM_FB
# =============================================================================
@torch.no_grad()
def _policy_diagnostics(model: FBModel, start_obs: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    """U-2 readings at the DM start states: mean policy entropy (nats) and mean score spread / tau."""
    zt = model.latent(z)
    F = model.forward_features(start_obs, zt)
    scores = policy_scores(F, zt)
    probs = torch.softmax(scores / model.tau, dim=-1)
    ent = float(-(probs * torch.log(probs.clamp_min(1e-30))).sum(-1).mean())
    spread = float(((scores.max(-1).values - scores.min(-1).values) / model.tau).mean())
    return {"policy_entropy_nats": ent, "score_spread_over_tau": spread,
            "max_action_prob_mean": float(probs.max(-1).values.mean())}


def run_adapt(ctx: JobContext, code_sha256: str) -> Dict[str, Any]:
    ident = ctx.identity
    fb_id = ident["fb_job"]
    parent = ctx.parents[fb_id]
    snaps = parent["result"]["outputs"]["snapshots"]
    use = snaps if ident["fb_snapshots"] == "all" else [max(snaps, key=lambda s: s["update"])]
    demo_states = _demo_states(ctx)                     # states only
    dcfg = DMConfig.from_dict(ident["dm"])
    steps = [int(s) for s in ident["dm_snapshot_steps"]]
    timer = PhaseTimer(ctx.device)
    records = []
    probe_states = demo_states.reshape((-1,) + demo_states.shape[2:])
    probe: Dict[str, np.ndarray] = {"states": probe_states}
    final_update = max(s["update"] for s in use)
    for snap in sorted(use, key=lambda s: s["update"]):
        if ctx.stop.requested():
            raise JobInterrupted("adaptation stopped before FB snapshot %d (restarts from scratch)" % snap["update"])
        path = Path(parent["dir"]) / snap["path"]
        if sha256_file(path) != snap["sha256"]:
            raise RuntimeError(f"FB snapshot {path} changed after pretraining")
        with timer.phase("load"):
            model = FBModel.load(path, ctx.device)
        fallback = make_rng(int(ident["seed"]), *ident["stream"], int(snap["update"]))
        sync(ctx.device)
        t0 = time.perf_counter()
        z_raw_timed = er_fb_z(model, demo_states)          # ER_FB's whole cost: one pass of B over the demos
        sync(ctx.device)
        t_er = time.perf_counter() - t0
        timer.add("er", t_er)
        t0 = time.perf_counter()
        dm = dm_fit(model, demo_states, dcfg, z0=None, fallback_rng=fallback, snapshot_steps=steps)
        sync(ctx.device)
        t_dm = time.perf_counter() - t0
        timer.add("dm", t_dm)
        er = dm.er                                         # the same z_ER, computed inside dm_fit (its warm start)
        if not np.allclose(er.z_raw, z_raw_timed, rtol=1e-6, atol=1e-9):
            raise RuntimeError("z_ER inside DM differs from the timed ER computation")
        data = dm_targets(model, demo_states, dcfg.K, dcfg.include_t0)
        start_obs = data.start_obs
        diag_er = _policy_diagnostics(model, start_obs, er.z)
        diag_dm = _policy_diagnostics(model, start_obs, dm.z)
        records.append({
            "fb_update": int(snap["update"]),
            "fb_model": {"path": snap["path"], "sha256": snap["sha256"]},
            "er": {"z": er.z.tolist(), "z_raw": er.z_raw.tolist(), "norm_raw": float(er.norm_raw),
                   "fallback": bool(er.fallback), "seconds": float(t_er), "u2_at_dm_start_states": diag_er},
            "dm": {"z": dm.z.tolist(), "snapshots": {str(k): v.tolist() for k, v in sorted(dm.snapshots.items())},
                   "log": dm.log, "steps": int(dcfg.steps), "seconds": float(t_dm), "n_start_states": int(len(start_obs)),
                   "u2_at_dm_start_states": diag_dm},
        })
        if snap["update"] == final_update:
            probe["er_probs"] = model.numpy_policy(er.z)(probe_states)
            probe["dm_probs"] = model.numpy_policy(dm.z)(probe_states)
            probe["er_z"] = er.z
            probe["dm_z"] = dm.z
    out_dir = ctx.job_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "latents.json", {"job_id": ctx.job.job_id, "fb_job": fb_id, "records": records})
    np.savez_compressed(out_dir / "probe.npz", **probe)
    return {
        "outputs": {"latents": _out(ctx, out_dir / "latents.json"), "probe": _out(ctx, out_dir / "probe.npz"),
                    "fb_job": fb_id, "fb_snapshots_used": [r["fb_update"] for r in records]},
        "costs": {
            "phase": "latent adaptation on the frozen FB model (no simulator data)",
            "env_steps": 0,
            "dm_steps_per_snapshot": int(dcfg.steps),
            "n_fb_snapshots": len(records),
            "demonstrations": {"n_trajectories": int(demo_states.shape[0]), "transitions": int(demo_states.shape[0] * (demo_states.shape[1] - 1))},
            "wall_s": timer.as_dict(),
            "final_snapshot_s": {"er": float(records[-1]["er"]["seconds"]), "dm": float(records[-1]["dm"]["seconds"])},
        },
        "summary": {"final": {"fb_update": records[-1]["fb_update"], "er_norm_raw": records[-1]["er"]["norm_raw"],
                              "dm_final_loss": records[-1]["dm"]["log"][-1]["loss"],
                              "dm_initial_loss": records[-1]["dm"]["log"][0]["loss"],
                              "u2_er": records[-1]["er"]["u2_at_dm_start_states"],
                              "u2_dm": records[-1]["dm"]["u2_at_dm_start_states"]}},
    }


EXECUTORS = {"fpg": run_fpg, "fb_pretrain": run_fb_pretrain, "adapt": run_adapt}
