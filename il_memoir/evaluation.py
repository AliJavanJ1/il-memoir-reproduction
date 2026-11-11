"""Held-out evaluation, common to every method (METHOD_MAP section 7).  ORACLE CALLS HAPPEN ONLY HERE.

Evaluation reads checkpointed policies and never updates them: an f-PG policy
record (``fpg.policy_from_record`` -> ``fpg.policy_callable`` /
``fpg.tabular_matrix``) or a frozen FB model with a latent
(``FBModel.numpy_policy(z)`` / ``FBModel.policy_matrix(z)``).  These are the
callables used during training and adaptation, so the evaluated probabilities
are the learner's (CLAUDE.md, 3B check).  Rows are float64 and renormalized by
``common.check_probability_rows`` before any oracle call.  Every file read is
checked against the SHA-256 recorded by the training job.

Finite mazes (exact, oracle; ``H = 100``, next-state reward timing)
-------------------------------------------------------------------
* ``js_nats`` = JS(d+_{pi,H}, q_E) with ``q_E = d+_{pi*,H}`` of the certified
  expert (OCC-3, OCC-5; DIV-2); ``kl_dq_nats``, ``kl_qd_nats`` for the ablation
  directions (``None`` with a flag when infinite);
* ``js_sa_nats`` = JS(d^SA_{pi,H}, q^SA_E): the state-action diagnostic (FPG-6);
  state matching does not identify expert actions;
* ``return`` = V^pi_H(s_0) = sum_{t<H} gamma^t E r_g(s_{t+1}) (OCC-4) and
  ``gap`` = V*_H(s_0) - V^pi_H(s_0) >= 0 against the H-step optimum (OCC-6);
* ``success`` = P(s_t in goal for some t in 1..H) by the taboo-kernel
  recursion :func:`hit_probability`; ``final_in_goal`` = P(s_H in goal);
* FB model error at the evaluated latent (FB-6, ``oracle.fb_error_diagnostic``),
  for ER_FB and DM_FB at the final FB snapshot;
* plus the Monte Carlo metrics below from a separate evaluation stream (a
  cross-check of the exact values and the goal-holding diagnostics).

Continuous maze (Monte Carlo, Heuristic: codes S1, S2)
------------------------------------------------------
``rollouts`` fresh trajectories from the start state with the evaluation
stream ``("evaluation", env, layout, goal)`` (the same stream for every
method and checkpoint of a goal and seed: common random numbers).  One common
held-out density estimator for all methods: the masses of the 20 x 20
free-space histogram (``density.FreeSpaceHistogram``) with weights
``gamma^t/(n Z_H)`` on ``s_{t+1}``, for the policy's rollouts and for the
held-out expert reference rollouts (``data/<id>/reference``), compared by the
plug-in ``JS`` in nats without smoothing.  The plug-in JS of two samples is
biased upwards; the expert's own row (``eval_expert``: the controller's fresh
rollouts against the same reference) gives the noise floor.

Evaluation transitions are counted (``env_steps_eval``) and never given to a learner.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import divergences as dv
from . import oracle
from .common import check_probability_rows, make_rng
from .datasets import load_demonstrations, load_reference_states, replay_path
from .density import FreeSpace, FreeSpaceHistogram, discount_weights, finite_occupancy
from .env_continuous import make_continuous_maze
from .env_tabular import initial_distribution, make_grid_maze, make_layout, transition_matrix
from .expert import expert_occupancy, finite_expert, route_controller
from .fb import FBModel, ReplayBuffer
from .fpg import policy_callable, policy_from_record, tabular_matrix
from .logging import JobContext, JobInterrupted, JsonlLog, load_torch, read_json, sha256_array, sha256_file
from .tasks import GoalTask, make_task

LOG2 = float(np.log(2.0))


def make_env(cfg, env: str, layout: str):
    if env == "finite":
        return make_grid_maze(cfg, layout)
    if env == "continuous":
        return make_continuous_maze(cfg, layout)
    raise ValueError(f"unknown environment {env!r}")


# =============================================================================
# exact finite quantities
# =============================================================================
def hit_probability(P: np.ndarray, pi: np.ndarray, mu0: np.ndarray, goal: np.ndarray, H: int) -> Tuple[float, np.ndarray]:
    """``P(s_t in goal for some t in 1..H)`` and the law of the first hitting time ``T`` on ``1..H``.

    Taboo recursion: ``v_0 = mu0`` (``s_0`` is not counted), and for ``t = 1..H``
    ``u_t = v_{t-1} P_pi``, ``P(T = t) = u_t(goal)``, ``v_t = u_t`` restricted to
    the non-goal states, so ``v_t(x) = P(s_t = x, T > t)``.  ``goal`` is a boolean mask.
    """
    P_pi = oracle.policy_kernel(P, pi)
    goal = np.asarray(goal, dtype=bool)
    v = np.asarray(mu0, dtype=np.float64).copy()
    first = np.zeros(int(H) + 1)
    for t in range(1, int(H) + 1):
        u = v @ P_pi
        first[t] = float(u[goal].sum())
        v = np.where(goal, 0.0, u)
    return float(first.sum()), first


def _kl_or_none(p: np.ndarray, q: np.ndarray, name: str) -> Optional[float]:
    val = dv.divergence(p, q, name)
    return None if not np.isfinite(val) else float(val)


class FiniteOracle:
    """Exact evaluation of a finite task: ``P``, ``mu0``, ``r_g``, the certified expert and ``q_E``."""

    def __init__(self, cfg, layout: str, task_name: str):
        self.cfg = cfg
        self.maze = make_grid_maze(cfg, layout)
        self.task: GoalTask = make_task(cfg, task_name)
        self.P = transition_matrix(self.maze)
        self.mu0 = initial_distribution(self.maze)
        self.r = self.task.finite_reward(self.maze)
        self.gamma, self.H = float(cfg.gamma), int(cfg.horizon)
        self.expert = finite_expert(self.maze, self.task, self.gamma, self.H, cfg.value_iteration_tol, cfg.tie_tol,
                                    layout_name=layout)
        self.q_E = expert_occupancy(self.maze, self.expert)
        self.q_E_sa = oracle.occupancy_state_action(self.P, self.expert.pi, self.mu0, self.gamma, self.H)
        self.s0 = self.maze.start_state
        self.goal_mask = self.task.in_goal(self.maze.positions(np.arange(self.maze.n_states)))
        self.V_H_star = float(self.expert.V_H_star[self.s0])

    def exact_metrics(self, pi: np.ndarray) -> Dict[str, Any]:
        pi = check_probability_rows(pi)
        d = oracle.occupancy_next_state(self.P, pi, self.mu0, self.gamma, self.H)
        d_sa = oracle.occupancy_state_action(self.P, pi, self.mu0, self.gamma, self.H)
        J_H = float(oracle.h_step_value(self.P, pi, self.r, self.gamma, self.H)[self.s0])
        z_h = oracle.discount_normalizer(self.gamma, self.H)
        hit, first = hit_probability(self.P, pi, self.mu0, self.goal_mask, self.H)
        final = float(oracle.state_marginals(self.P, pi, self.mu0, self.H)[self.H][self.goal_mask].sum())
        kl_dq = _kl_or_none(d, self.q_E, "kl_dq")
        kl_qd = _kl_or_none(d, self.q_E, "kl_qd")
        t = np.arange(first.size)
        return {
            "js_nats": dv.js(d, self.q_E),
            "kl_dq_nats": kl_dq, "kl_dq_infinite": kl_dq is None,
            "kl_qd_nats": kl_qd, "kl_qd_infinite": kl_qd is None,
            "js_sa_nats": dv.js(d_sa.ravel(), self.q_E_sa.ravel()),
            "return": J_H,
            "return_from_occupancy": float(z_h * d @ self.r),
            "gap": self.V_H_star - J_H,
            "V_H_star": self.V_H_star,
            "success": hit,
            "final_in_goal": final,
            "hit_time_mean_given_hit": float((t * first).sum() / hit) if hit > 0 else None,
            "goal_occupancy_mass": float(d[self.goal_mask].sum()),
        }


def fb_model_error(model: FBModel, z: np.ndarray, nu_hat: np.ndarray, orc: FiniteOracle) -> Dict[str, Any]:
    """FB-6 diagnostic at ``z`` (Thm 5.17; ``oracle.fb_error_diagnostic``), with ``r = r_g``.  Evaluation only."""
    states = np.arange(orc.maze.n_states)
    with torch.no_grad():
        F = model.forward_features(states, z).to("cpu", torch.float64).numpy()
        B = model.backward_features(states).to("cpu", torch.float64).numpy()
    out = oracle.fb_error_diagnostic(F, B, nu_hat, orc.P, np.asarray(z, dtype=np.float64), model.tau, orc.gamma, r=orc.r)
    err = out["eval_error"]
    return {"sup_tv": out["sup_tv"], "sup_l2_dual": out["sup_l2_dual"] if np.isfinite(out["sup_l2_dual"]) else None,
            "delta_tv": out["delta_tv"],
            "delta_l2_dual": out["delta_l2_dual"] if np.isfinite(out["delta_l2_dual"]) else None,
            "q_error_max": float(err.max()), "q_error_mean": float(err.mean()),
            "q_error_at_start_max": float(err[orc.s0].max())}


# =============================================================================
# Monte Carlo metrics (both mazes)
# =============================================================================
def _mean_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-300, 1.0)
    return max(0.0, float(-(probs * np.log(p)).sum(axis=1).mean()))


def mc_metrics(env, policy: Callable[[np.ndarray], np.ndarray], task: GoalTask, gamma: float, H: int, n: int,
               rng: np.random.Generator, hist: Optional[FreeSpaceHistogram] = None,
               ref_masses: Optional[np.ndarray] = None, q_E: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Monte Carlo metrics of ``n`` fresh rollouts.  ``hist`` and ``ref_masses`` give the continuous JS;
    ``q_E`` (finite) gives the plug-in JS of the visit counts against the exact expert occupancy."""
    traj = env.rollout(policy, n, H, rng)
    states = np.asarray(traj.states)
    finite = states.ndim == 2
    pos = env.positions(states) if finite else states
    returns = task.discounted_returns(pos, gamma)
    hits = task.hitting_times(pos)
    reached = hits > 0
    in_goal = task.in_goal(pos[:, 1:])
    after = np.arange(1, H + 1)[None, :] >= np.where(reached, hits, H + 1)[:, None]
    goal_cell = np.asarray(task.goal_cell)
    cells = np.asarray(env.cells, dtype=np.int64)[states[:, 1:]] if finite else env.cell_of(pos[:, 1:])
    in_cell = np.all(cells == goal_cell, axis=-1)
    visited = states[:, :-1].reshape((-1,) + states.shape[2:])
    probs = policy(visited)
    succ = float(reached.mean())
    out: Dict[str, Any] = {
        "mc_n": int(n),
        "mc_return_mean": float(returns.mean()),
        "mc_return_se": float(returns.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
        "mc_success": succ,
        "mc_success_se": float(np.sqrt(succ * (1 - succ) / n)),
        "mc_final_in_goal": float(task.in_goal(pos[:, -1]).mean()),
        "mc_hit_time_mean": float(hits[reached].mean()) if reached.any() else None,
        "mc_post_hit_goal_region_frac": float(in_goal[after].mean()) if after.any() else None,
        "mc_post_hit_goal_cell_frac": float(in_cell[after].mean()) if after.any() else None,
        "policy_entropy_visited_nats": _mean_entropy(probs),
    }
    if finite and q_E is not None:
        counts = finite_occupancy(states[:, 1:], env.n_states, gamma)
        out["mc_js_counts_nats"] = dv.js(counts, q_E)
    if not finite:
        if not bool(env.is_free(states).all()):
            raise RuntimeError("an evaluation rollout left the free space")
        if hist is not None and ref_masses is not None:
            w = discount_weights(n, H, gamma).reshape(-1)
            masses = hist.masses(states[:, 1:].reshape(-1, 2), w)
            out["js_nats"] = dv.js(masses, ref_masses)
            out["kl_dq_nats_smoothed"] = dv.smoothed_divergence(masses, ref_masses, "kl_dq", 0.01, hist.reference)
            out["kl_qd_nats_smoothed"] = dv.smoothed_divergence(masses, ref_masses, "kl_qd", 0.01, hist.reference)
            out["success"] = succ
            out["return"] = out["mc_return_mean"]
            out["final_in_goal"] = out["mc_final_in_goal"]
    return out


class ContinuousReference:
    """The common held-out estimator: free-space histogram masses of the held-out expert reference rollouts."""

    def __init__(self, cfg, layout: str, reference_states: np.ndarray):
        self.env = make_continuous_maze(cfg, layout)
        self.space = FreeSpace.from_env(self.env)
        self.hist = FreeSpaceHistogram(self.space, int(cfg.histogram_bins))
        ref = np.asarray(reference_states, dtype=np.float64)
        n, H = ref.shape[0], ref.shape[1] - 1
        w = discount_weights(n, H, cfg.gamma).reshape(-1)
        self.masses = self.hist.masses(ref[:, 1:].reshape(-1, 2), w)
        self.n_reference = int(n)


# =============================================================================
# checkpoints of a training job
# =============================================================================
def _verify(path: Path, sha: str) -> Path:
    actual = sha256_file(path)
    if actual != sha:
        raise RuntimeError(f"{path}: SHA-256 {actual} differs from the recorded {sha}; refusing to evaluate")
    return path


def fpg_checkpoints(parent_dir: Path, result: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for o in result["outputs"]["policies"]:
        out.append({"key": f"fpg-iter{o['iteration']:06d}", "method": "fpg", "kind": "policy",
                    "path": parent_dir / o["path"], "sha256": o["sha256"],
                    "coords": {"iteration": o["iteration"], "env_steps": o["env_steps"], "n_updates": o["n_updates"]}})
    return out


def adapt_checkpoints(parent_dir: Path, result: Dict[str, Any], fb_dir: Path) -> List[Dict[str, Any]]:
    """ER and final DM latents at every FB snapshot; every declared DM step at the final snapshot."""
    lat = result["outputs"]["latents"]
    latents = read_json(_verify(parent_dir / lat["path"], lat["sha256"]))
    recs = sorted(latents["records"], key=lambda r: r["fb_update"])
    final_u = recs[-1]["fb_update"]
    out = []
    for r in recs:
        model = {"path": fb_dir / r["fb_model"]["path"], "sha256": r["fb_model"]["sha256"]}
        base = {"fb_update": r["fb_update"], "final_fb_snapshot": r["fb_update"] == final_u}
        out.append({"key": f"er-u{r['fb_update']:06d}", "method": "er", "kind": "latent", "model": model,
                    "z": r["er"]["z"], "coords": dict(base, dm_step=None, er_norm_raw=r["er"]["norm_raw"])})
        steps = sorted(int(k) for k in r["dm"]["snapshots"])
        wanted = steps if r["fb_update"] == final_u else [steps[-1]]
        for k in wanted:
            out.append({"key": f"dm-u{r['fb_update']:06d}-k{k:04d}", "method": "dm", "kind": "latent", "model": model,
                        "z": r["dm"]["snapshots"][str(k)],
                        "coords": dict(base, dm_step=k, final_dm_step=k == steps[-1],
                                       dm_loss=r["dm"]["log"][k]["loss"])})
    return out


# =============================================================================
# evaluation jobs
# =============================================================================
def _eval_device(ctx: JobContext) -> str:
    return "cpu" if ctx.identity["evaluation"]["device"] == "cpu" else ctx.device


def _nu_hat_finite(ctx: JobContext, fb_job_dir: Path, n_states: int) -> np.ndarray:
    job = read_json(fb_job_dir / "job.json")
    key = job["data_inputs"]["replay"]
    buf = ReplayBuffer.load(replay_path(ctx.data_root, ctx.data_id, key))
    counts = np.bincount(np.asarray(buf.next_obs, dtype=np.int64), minlength=n_states).astype(np.float64)
    return counts / counts.sum()


class _Evaluator:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        ident = ctx.identity
        cfg = ctx.env_cfg
        self.env_kind, self.layout = ident["env"], ident["layout"]
        self.task = make_task(cfg, ident["task"])
        self.gamma, self.H = float(cfg.gamma), int(cfg.horizon)
        self.env = make_env(cfg, self.env_kind, self.layout)
        self.n_mc = int(ident["evaluation"]["mc_rollouts" if self.env_kind == "finite" else "rollouts"])
        self.oracle: Optional[FiniteOracle] = None
        self.reference: Optional[ContinuousReference] = None
        if self.env_kind == "finite":
            self.oracle = FiniteOracle(cfg, self.layout, ident["task"])
        else:
            entry = ctx.data_entries["reference"]
            ref_states, _ = load_reference_states(ctx.data_root, ctx.data_id, ctx.job.data_inputs["reference"],
                                                  entry["content_sha256"])
            self.reference = ContinuousReference(cfg, self.layout, ref_states)

    def rng(self) -> np.random.Generator:
        ident = self.ctx.identity
        return make_rng(int(ident["seed"]), *ident["stream"])     # a fresh copy per checkpoint (common random numbers)

    def metrics(self, policy: Callable[[np.ndarray], np.ndarray], matrix: Optional[np.ndarray]) -> Dict[str, Any]:
        if self.env_kind == "finite":
            out = self.oracle.exact_metrics(matrix)
            out.update(mc_metrics(self.env, policy, self.task, self.gamma, self.H, self.n_mc, self.rng(),
                                  q_E=self.oracle.q_E))
            return out
        return mc_metrics(self.env, policy, self.task, self.gamma, self.H, self.n_mc, self.rng(),
                          hist=self.reference.hist, ref_masses=self.reference.masses)


def run_eval(ctx: JobContext, code_sha256: str) -> Dict[str, Any]:
    """Evaluate every checkpoint of one f-PG or adaptation job; resumable per checkpoint."""
    ident = ctx.identity
    target = ident["target"]
    parent = ctx.parents[target]
    parent_dir = Path(parent["dir"])
    device = _eval_device(ctx)
    ev = _Evaluator(ctx)
    fb_dir = None
    if ident["target_type"] == "fpg":
        checkpoints = fpg_checkpoints(parent_dir, parent["result"])
    else:
        fb_id = parent["result"]["outputs"]["fb_job"]
        fb_dir = ctx.job_dir_of(fb_id)
        checkpoints = adapt_checkpoints(parent_dir, parent["result"], fb_dir)
    log = JsonlLog(ctx.job_dir / "outputs" / "evaluations.jsonl")
    done = {r["checkpoint"]: r for r in log.records()}
    nu_hat = None
    models: Dict[str, FBModel] = {}
    t_start = time.perf_counter()
    env_steps = sum(int(r["env_steps_eval"]) for r in done.values())
    for ck in checkpoints:
        if ck["key"] in done:
            continue
        if ctx.stop.requested():
            raise JobInterrupted(f"evaluation stopped before {ck['key']} ({len(done)} checkpoints kept)")
        t0 = time.perf_counter()
        row: Dict[str, Any] = {"checkpoint": ck["key"], "method": ck["method"], "coords": ck["coords"]}
        if ck["kind"] == "policy":
            rec = load_torch(_verify(ck["path"], ck["sha256"]))
            pol = policy_from_record(rec, device)
            call = policy_callable(pol)
            matrix = tabular_matrix(pol) if ev.env_kind == "finite" else None
            row["source"] = {"path": str(Path(ck["path"]).relative_to(parent_dir).as_posix()), "sha256": ck["sha256"]}
        else:
            mkey = ck["model"]["sha256"]
            if mkey not in models:
                models[mkey] = FBModel.load(_verify(ck["model"]["path"], mkey), device)
            model = models[mkey]
            z = np.asarray(ck["z"], dtype=np.float64)
            call = model.numpy_policy(z)
            matrix = model.policy_matrix(z) if ev.env_kind == "finite" else None
            row["source"] = {"fb_model_sha256": mkey, "z": ck["z"]}
        m = ev.metrics(call, matrix)
        if ck["kind"] == "latent" and ev.env_kind == "finite" and ck["coords"].get("final_fb_snapshot") \
                and (ck["method"] == "er" or ck["coords"].get("final_dm_step")):
            if nu_hat is None:
                nu_hat = _nu_hat_finite(ctx, fb_dir, ev.oracle.maze.n_states)
            m["fb_model_error"] = fb_model_error(model, np.asarray(ck["z"]), nu_hat, ev.oracle)
        row["metrics"] = m
        row["env_steps_eval"] = int(ev.n_mc * ev.H)
        row["seconds"] = time.perf_counter() - t0
        row["rng"] = {"seed": int(ident["seed"]), "stream": list(ident["stream"])}
        log.append(row)
        done[ck["key"]] = row
        env_steps += row["env_steps_eval"]
    rows = [done[ck["key"]] for ck in checkpoints]
    return {
        "outputs": {"evaluations": {"path": "outputs/evaluations.jsonl", "sha256": sha256_file(log.path)},
                    "n_checkpoints": len(rows)},
        "costs": {"phase": "held-out evaluation (never training data)", "env_steps_eval": int(env_steps),
                  "n_checkpoints": len(rows), "device": device,
                  "wall_s": {"evaluate": float(sum(r["seconds"] for r in rows))},     # every row, over all attempts
                  "this_attempt_wall_s": time.perf_counter() - t_start,               # overlaps wall_s: not added
                  "oracle": ev.env_kind == "finite"},
        "summary": {"last": {r["method"]: r["metrics"] for r in rows}},
    }


def run_eval_expert(ctx: JobContext, code_sha256: str) -> Dict[str, Any]:
    """The expert under the same evaluation: the reference row and, in the continuous maze, the JS noise floor."""
    ident = ctx.identity
    cfg = ctx.env_cfg
    t0 = time.perf_counter()
    ev = _Evaluator(ctx)
    states, actions, meta = load_demonstrations(ctx.data_root, ctx.data_id, ctx.job.data_inputs["demos"],
                                                ctx.data_entries["demos"]["content_sha256"])
    n_e = states.shape[0]
    if ev.env_kind == "finite":
        orc = ev.oracle
        if meta.get("expert_actions_sha256") is not None:
            if sha256_array(orc.expert.actions) != meta["expert_actions_sha256"]:
                raise RuntimeError("the evaluation's expert differs from the expert that generated the demonstrations")
        pi = orc.expert.pi
        call = lambda s: pi[np.asarray(s, dtype=np.int64)]                       # noqa: E731
        m = ev.metrics(call, pi)
        target = finite_occupancy(states[:, 1:], orc.maze.n_states, ev.gamma)
        m["demo_target_js_nats"] = dv.js(target, orc.q_E)
        m["expert_certificate"] = orc.expert.summary()
    else:
        ctrl = route_controller(make_layout(cfg, ev.layout), cfg.goal_cell(ident["task"]), cfg.action_deltas)
        m = ev.metrics(ctrl, None)
        w = discount_weights(n_e, states.shape[1] - 1, ev.gamma).reshape(-1)
        demo_masses = ev.reference.hist.masses(states[:, 1:].reshape(-1, 2), w)
        m["demo_target_js_nats"] = dv.js(demo_masses, ev.reference.masses)
        m["n_reference"] = ev.reference.n_reference
    row = {"checkpoint": "expert", "method": "expert", "coords": {}, "metrics": m,
           "env_steps_eval": int(ev.n_mc * ev.H), "seconds": time.perf_counter() - t0,
           "rng": {"seed": int(ident["seed"]), "stream": list(ident["stream"])}}
    log = JsonlLog(ctx.job_dir / "outputs" / "evaluations.jsonl")
    if log.path.exists():
        log.truncate(0)
    log.append(row)
    return {
        "outputs": {"evaluations": {"path": "outputs/evaluations.jsonl", "sha256": sha256_file(log.path)}, "n_checkpoints": 1},
        "costs": {"phase": "held-out evaluation of the expert", "env_steps_eval": row["env_steps_eval"], "n_checkpoints": 1,
                  "wall_s": {"evaluate": row["seconds"]}, "oracle": ev.env_kind == "finite",
                  "demonstrations": {"n_trajectories": int(n_e), "transitions": int(n_e * (states.shape[1] - 1))}},
        "summary": {"last": {"expert": m}},
    }


EXECUTORS = {"eval": run_eval, "eval_expert": run_eval_expert}
