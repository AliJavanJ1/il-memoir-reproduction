"""Pilot benchmark (``scripts/benchmark.py``; INTERFACE command 4, PROJECT_SPEC section 7, WORKFLOW section 6).

Three measured stages, then an extrapolation:

1. **Timing workloads** (``pilot.timing`` of ``configs/pilot.yaml``) on the CPU
   and, where CUDA is available (never inferred on a login node), on CUDA:
   replay collection, FB updates, the Gram matrix, DM latent steps, f-PG
   iterations (finite counts; continuous KDE and histogram), held-out
   evaluation and checkpoint saving/loading.  Every reading is
   synchronized.  These short runs check speed, not learning.
2. **Device choice**: the device with the smaller projected core-suite time
   from these per-unit measurements, unless ``--device cpu|cuda`` is explicit
   (recorded as such).  The choice is saved in ``runs/benchmark.json`` and is what
   ``--device auto`` uses afterwards.
3. **Representative learning pilot**: the ``pilot`` suite (seed 999, core
   budgets, the declared grid) through the real runner on the chosen device,
   with its analysis in ``runs/pilot_results/`` (labelled PILOT).
4. **ETA** of every suite of ``pilot.eta_suites`` from the per-unit costs
   measured in the learning pilot's own jobs (``result.json``), plus the
   declared overhead (at least 50%), with a low-high range from the spread of
   the measured per-unit times.  It is a linear extrapolation; thermal
   throttling, background load and other machines are not modelled.

``runs/benchmark.json`` and ``runs/PILOT.md`` record everything measured;
``complete`` is true only if every pilot job completed.  An existing
``benchmark.json`` from another computer is never overwritten (``--force``
overrides), and ``runs/environment_laptop.json`` is never touched.
"""
from __future__ import annotations

import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .common import make_rng
from .experiment import SuiteConfig, expand_jobs, fb_settings, fpg_settings, load_suite_config
from .hardware import cuda_available, hardware_record, scheduler_context, set_cpu_threads, sync
from .logging import JsonlLog, host_fingerprint, read_json, relpath, utc_now, write_json_atomic, write_text_atomic

PILOT_SCHEMA = "il_memoir/pilot-v1"


class PilotError(RuntimeError):
    pass


# =============================================================================
# 1. timing workloads
# =============================================================================
def _timed(device: str, fn: Callable[[], Any]) -> Tuple[float, Any]:
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    return time.perf_counter() - t0, out


def timing_workloads(suite: SuiteConfig, device: str, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Short synchronized workloads on ``device``; per-unit seconds for every job phase."""
    from . import fb, fb_il, fpg
    from .evaluation import ContinuousReference, FiniteOracle, mc_metrics
    from .expert import finite_expert, generate_demonstrations, route_controller
    from .env_tabular import make_layout
    from .fb import FBModel, FBPretrainer, ReplayConfig, collect_replay
    from .nets import ObsSpec
    from .prepare import make_env
    from .tasks import make_task

    cfg = suite.env_cfg
    timing = suite.pilot["timing"]
    seed = suite.seeds[0]
    out: Dict[str, Any] = {"device": device, "torch_threads": torch.get_num_threads()}
    for cond in suite.conditions:
        env_kind, layout, task = cond.env, cond.layout, cond.tasks[0]
        t = timing[env_kind]
        env = make_env(cfg, env_kind, layout)
        rec: Dict[str, Any] = {"layout": layout, "task": task}
        rng = make_rng(seed, "pilot_timing", env_kind, layout)
        n_rep = t["replay_transitions"] - t["replay_transitions"] % 50
        dt, buf = _timed("cpu", lambda: collect_replay(env, ReplayConfig(n_rep, 50), rng))
        rec["replay_s_per_transition"] = dt / n_rep
        # demonstrations with the real expert (timing of the data step)
        tk = make_task(cfg, task)
        if env_kind == "finite":
            ex = finite_expert(env, tk, cfg.gamma, cfg.horizon, cfg.value_iteration_tol, cfg.tie_tol)
            expert_policy = ex.pi
        else:
            expert_policy = route_controller(make_layout(cfg, layout), cfg.goal_cell(task), cfg.action_deltas)
        dt, demos = _timed("cpu", lambda: generate_demonstrations(env, expert_policy, 4, cfg.horizon,
                                                                   make_rng(seed, "pilot_timing", "demos")))
        rec["demos_s"] = dt
        states = demos.learner_states()
        # FB updates and Gram matrix
        fcfg, _ = fb_settings(suite, env_kind, "base")
        fcfg = fb.FBConfig.from_dict(dict(fcfg.to_dict(), n_updates=int(t["fb_updates"])))
        trainer = FBPretrainer(ObsSpec.from_env(env), env.n_actions, cfg.gamma, fcfg, seed, ("pilot_timing", env_kind), device)
        trainer.attach_buffer(buf)
        trainer.update()                                         # warm-up (CUDA context, kernels)
        dt, _ = _timed(device, lambda: [trainer.update() for _ in range(int(t["fb_updates"]))])
        rec["fb_s_per_update"] = dt / int(t["fb_updates"])
        dt, model = _timed(device, trainer.model)
        rec["fb_snapshot_s_per_replay_transition"] = dt / len(buf)
        # DM
        steps = min(300, int(t["dm_steps"]))
        dcfg = fb_il.DMConfig(steps=steps)
        dt, _ = _timed(device, lambda: fb_il.dm_fit(model, states, dcfg))
        rec["dm_s_per_step"] = dt / steps
        dt, _ = _timed(device, lambda: fb_il.er_fb_z(model, states))
        rec["er_s"] = dt
        # f-PG
        densities = ["counts"] if env_kind == "finite" else ["kde", "histogram"]
        for dens in densities:
            fcfg_p, _ = fpg_settings(suite, env_kind, "base", "js")
            raw = fcfg_p.to_dict()
            if env_kind == "continuous":
                raw["density"] = dens
            learner = fpg.FPGLearner(env, cfg.learner_dict(), states, fpg.FPGConfig.from_dict(raw), seed,
                                     ("pilot_timing", env_kind, dens), device)
            learner.step(env)                                    # warm-up
            k = int(t["fpg_iterations"])
            dt, _ = _timed(device, lambda: [learner.step(env) for _ in range(k)])
            rec[f"fpg_{dens}_s_per_iteration"] = dt / k
            rec[f"fpg_{dens}_n_traj"] = int(raw["n_traj"])
        # evaluation (always measured on the CPU: evaluation.device is cpu in the configurations)
        pol_cpu = fpg.policy_from_record(fpg.policy_record(learner.policy), "cpu")
        n_eval = int(t["eval_rollouts"])
        if env_kind == "finite":
            def one_eval():
                orc = FiniteOracle(cfg, layout, task)
                m = orc.exact_metrics(fpg.tabular_matrix(pol_cpu))
                m.update(mc_metrics(env, fpg.policy_callable(pol_cpu), tk, cfg.gamma, cfg.horizon, n_eval,
                                    make_rng(seed, "pilot_timing", "eval"), q_E=orc.q_E))
                return m
        else:
            ref_states = env.rollout(expert_policy, 200, cfg.horizon, make_rng(seed, "pilot_timing", "ref")).states
            reference = ContinuousReference(cfg, layout, ref_states)

            def one_eval():
                return mc_metrics(env, fpg.policy_callable(pol_cpu), tk, cfg.gamma, cfg.horizon, n_eval,
                                  make_rng(seed, "pilot_timing", "eval"), hist=reference.hist, ref_masses=reference.masses)
        dt, _ = _timed("cpu", one_eval)
        rec["eval_s_per_checkpoint"] = dt
        rec["eval_rollouts"] = n_eval
        # caching: save and reload a model and a resume checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.pt"
            dt_s, _ = _timed(device, lambda: model.save(p))
            dt_l, _ = _timed(device, lambda: FBModel.load(p, device))
            q = Path(tmp) / "r.pt"
            dt_r, _ = _timed(device, lambda: torch.save(trainer.state_dict(), q))
        rec["cache_model_save_s"], rec["cache_model_load_s"], rec["cache_resume_save_s"] = dt_s, dt_l, dt_r
        out[env_kind] = rec
        log(f"  timing {env_kind:10s} on {device}: FB {rec['fb_s_per_update'] * 1e3:.2f} ms/update, "
            f"DM {rec['dm_s_per_step'] * 1e3:.2f} ms/step, "
            + ", ".join(f"f-PG {d} {rec[f'fpg_{d}_s_per_iteration']:.3f} s/it" for d in densities)
            + f", eval {rec['eval_s_per_checkpoint']:.2f} s")
    return out


# =============================================================================
# per-unit costs and job-time prediction
# =============================================================================
UnitCosts = Dict[str, Dict[str, List[float]]]


def units_from_timing(t: Dict[str, Any]) -> UnitCosts:
    """Per-unit costs (lists, to keep the spread) from one device's timing workloads."""
    u: UnitCosts = defaultdict(lambda: defaultdict(list))
    for env_kind in ("finite", "continuous"):
        if env_kind not in t:
            continue
        r = t[env_kind]
        u[env_kind]["fb_update"].append(r["fb_s_per_update"])
        u[env_kind]["snapshot_per_transition"].append(r["fb_snapshot_s_per_replay_transition"])
        u[env_kind]["dm_step"].append(r["dm_s_per_step"])
        u[env_kind]["er"].append(r["er_s"])
        u[env_kind]["replay_per_transition"].append(r["replay_s_per_transition"])
        u[env_kind]["demos_item"].append(r["demos_s"])
        u[env_kind]["eval_per_rollout"].append(r["eval_s_per_checkpoint"] / r["eval_rollouts"])
        for dens in ("counts", "kde", "histogram"):
            key = f"fpg_{dens}_s_per_iteration"
            if key in r:
                u[env_kind][f"fpg_{dens}_per_trajectory"].append(r[key] / r[f"fpg_{dens}_n_traj"])
        u[env_kind]["checkpoint_save"].append(r["cache_resume_save_s"] + r["cache_model_save_s"])
    return u


def units_from_runs(runs_root: Path, manifest: Dict[str, Any], data_root: Path, device: Optional[str] = None,
                    skipped: Optional[List[str]] = None) -> UnitCosts:
    """Per-unit costs measured inside the completed jobs of a suite (the learning pilot).

    With ``device``, only jobs that ran on that device count (an interrupted
    pilot resumed on another device must not mix CPU and CUDA timings); the
    identifiers of the other jobs are appended to ``skipped``.  Data
    generation runs on the CPU in every case and always counts.
    """
    from . import datasets as ds

    u: UnitCosts = defaultdict(lambda: defaultdict(list))
    man_data = ds.load_manifest(data_root, manifest["data"]["id"])
    for key, e in man_data["items"].items():
        env_kind = e["spec"]["env"]
        if e.get("generation_seconds") is None:
            continue
        if e["kind"] == "replay":
            u[env_kind]["replay_per_transition"].append(e["generation_seconds"] / e["spec"]["n_transitions"])
        elif e["kind"] == "demos":
            u[env_kind]["demos_item"].append(e["generation_seconds"])
        elif e["kind"] == "reference":
            u[env_kind]["reference_per_trajectory"].append(e["generation_seconds"] / e["spec"]["n_trajectories"])
    for j in manifest["jobs"]:
        d = runs_root / "jobs" / j["job_id"]
        if not (d / "result.json").exists():
            continue
        res = read_json(d / "result.json")
        if device is not None and res.get("device") != device:
            if skipped is not None:
                skipped.append(j["job_id"])
            continue
        c, ident, env_kind = res["costs"], j["identity"], j["labels"]["env"]
        wall = c.get("wall_s", {})
        if j["type"] == "fpg":
            dens = "counts" if env_kind == "finite" else ident["fpg"]["density"]
            n_traj = ident["fpg"]["n_traj"]
            u[env_kind][f"fpg_{dens}_per_trajectory"].append(wall.get("train", 0.0) / (c["iterations"] * n_traj))
            u[env_kind]["policy_checkpoint"].append(wall.get("policy_checkpoint", 0.0) / len(res["outputs"]["policies"]))
        elif j["type"] == "fb_pretrain":
            u[env_kind]["fb_update"].append(wall.get("train", 0.0) / c["updates"])
            n_snap = len(res["outputs"]["snapshots"])
            u[env_kind]["snapshot_per_transition"].append(wall.get("snapshot", 0.0) / (n_snap * c["replay_transitions"]))
            u[env_kind]["checkpoint_per_update"].append(wall.get("checkpoint", 0.0) / c["updates"])
        elif j["type"] == "adapt":
            lat = read_json(d / res["outputs"]["latents"]["path"])
            for r in lat["records"]:
                u[env_kind]["dm_step"].append(r["dm"]["seconds"] / max(1, r["dm"]["steps"]))
                u[env_kind]["er"].append(r["er"]["seconds"])
        elif j["type"] in ("eval", "eval_expert"):
            n_mc = ident["evaluation"].get("mc_rollouts", ident["evaluation"].get("rollouts"))
            for row in JsonlLog(d / res["outputs"]["evaluations"]["path"]).records():
                u[env_kind]["eval_per_rollout"].append(row["seconds"] / n_mc)
    return u


def _stat(values: List[float], how: str) -> float:
    if not values:
        raise PilotError("a per-unit cost needed for the ETA was not measured")
    if how == "median":
        return float(statistics.median(values))
    if how == "max":
        return float(max(values))
    return float(min(values))


def predict_job_seconds(job, suite: SuiteConfig, units: UnitCosts, how: str = "median",
                        by_id: Optional[Dict[str, Any]] = None) -> float:
    """Predicted wall time of one job from per-unit costs (linear in budgets)."""
    if by_id is None:
        by_id = {j.job_id: j for j in expand_jobs(suite)}
    ident, env_kind = job.identity, job.labels["env"]
    u = units[env_kind]
    if job.type == "fpg":
        dens = "counts" if env_kind == "finite" else ident["fpg"]["density"]
        per_traj = _stat(u[f"fpg_{dens}_per_trajectory"], how)
        n_ck = len(ident["policy_checkpoints"])
        ck = _stat(u.get("policy_checkpoint") or u["checkpoint_save"], how)
        return ident["n_iterations"] * ident["fpg"]["n_traj"] * per_traj + n_ck * ck
    if job.type == "fb_pretrain":
        n_trans = int(suite.data["replay"][env_kind]["n_transitions"])
        upd = ident["fb"]["n_updates"]
        snap = _stat(u["snapshot_per_transition"], how) * n_trans * len(ident["snapshot_updates"])
        ck = _stat(u.get("checkpoint_per_update") or [0.0], how) * upd
        return upd * _stat(u["fb_update"], how) + snap + ck
    if job.type == "adapt":
        n_snap = 1 if ident["fb_snapshots"] == "final" else None
        if n_snap is None:
            n_snap = len(by_id[ident["fb_job"]].identity["snapshot_updates"])
        return n_snap * (ident["dm"]["steps"] * _stat(u["dm_step"], how) + _stat(u["er"], how))
    if job.type in ("eval", "eval_expert"):
        n_mc = ident["evaluation"].get("mc_rollouts", ident["evaluation"].get("rollouts"))
        if job.type == "eval_expert":
            n_ck = 1
        elif ident["target_type"] == "fpg":
            n_ck = len(by_id[ident["target"]].identity["policy_checkpoints"])
        else:
            tgt = by_id[ident["target"]]
            fbj = by_id[tgt.identity["fb_job"]]
            n_snap = 1 if tgt.identity["fb_snapshots"] == "final" else len(fbj.identity["snapshot_updates"])
            n_ck = 2 * n_snap + max(0, len(tgt.identity["dm_snapshot_steps"]) - 1)
        return n_ck * n_mc * _stat(u["eval_per_rollout"], how)
    raise PilotError(f"unknown job type {job.type}")


def predict_data_seconds(suite: SuiteConfig, units: UnitCosts, how: str = "median") -> float:
    from .experiment import data_items

    total = 0.0
    for item in data_items(suite).values():
        u = units[item.spec["env"]]
        if item.kind == "replay":
            total += item.spec["n_transitions"] * _stat(u["replay_per_transition"], how)
        elif item.kind == "demos":
            total += _stat(u["demos_item"], how)
        else:
            per = u.get("reference_per_trajectory") or [x / 4 for x in u["demos_item"]]
            total += item.spec["n_trajectories"] * _stat(per, how)
    return total


def eta(suite_paths: Sequence[Union[str, Path]], units: UnitCosts, overhead: float,
        exclude_ids: Sequence[str] = ()) -> Dict[str, Any]:
    """Per-suite predicted seconds: full suite and the part not already in earlier suites (shared jobs)."""
    out: Dict[str, Any] = {}
    seen = set(exclude_ids)
    for path in suite_paths:
        s = load_suite_config(path)
        jobs = expand_jobs(s)
        by_id = {j.job_id: j for j in jobs}
        rec: Dict[str, Any] = {"config": s.path, "config_sha256": s.sha256, "n_jobs": len(jobs)}
        for how in ("min", "median", "max"):
            full = sum(predict_job_seconds(j, s, units, how, by_id) for j in jobs) + predict_data_seconds(s, units, how)
            extra = sum(predict_job_seconds(j, s, units, how, by_id) for j in jobs if j.job_id not in seen)
            rec[f"measured_{how}_s"] = full
            rec[f"extra_after_previous_{how}_s"] = extra
        rec["n_extra_jobs"] = sum(1 for j in jobs if j.job_id not in seen)
        rec["by_type_median_s"] = {}
        for j in jobs:
            rec["by_type_median_s"][j.type] = rec["by_type_median_s"].get(j.type, 0.0) + predict_job_seconds(j, s, units, "median", by_id)
        rec["overhead"] = overhead
        rec["proposed_s"] = rec["measured_median_s"] * (1.0 + overhead)
        rec["proposed_range_s"] = [rec["measured_min_s"] * (1.0 + overhead), rec["measured_max_s"] * (1.0 + overhead)]
        out[s.name] = rec
        if s.name == "core":
            seen |= {j.job_id for j in jobs}
    return out


# =============================================================================
# the benchmark
# =============================================================================
def _hours(s: float) -> str:
    return f"{s / 3600:.2f} h ({s / 60:.1f} min)"


def run_benchmark(
    config: Union[str, Path],
    device_request: str,
    runs_root: Union[str, Path],
    data_root: Union[str, Path],
    force: bool = False,
    skip_learning: bool = False,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    from .analysis import AnalysisError, Collection, analyze
    from .runner import EXIT_OK, Runner

    t_start = time.perf_counter()
    suite = load_suite_config(config)
    if suite.kind != "pilot":
        raise PilotError("benchmark.py needs a pilot configuration (kind: pilot)")
    runs_root, data_root = Path(runs_root), Path(data_root)
    out_json = runs_root / "benchmark.json"
    here = host_fingerprint()
    if out_json.exists() and not force:
        old = read_json(out_json)
        if old.get("host", {}).get("hostname") != here["hostname"] or old.get("host", {}).get("system") != here["system"]:
            raise PilotError(f"{out_json} was recorded on another computer ({old.get('host')}); it is not overwritten. "
                             "Use --runs <other directory> for this machine, or --force")
    threads = set_cpu_threads(int(suite.runner["cpu_threads"]))
    sched = scheduler_context()
    candidates = ["cpu"]
    cuda_ok = cuda_available() and not sched["login_node"]
    if cuda_ok:
        candidates.append("cuda")
    log(f"pilot: candidate devices {candidates} (torch threads {threads}; scheduler {sched})")
    if device_request == "cuda" and not cuda_ok:
        log("WARNING: --device cuda was requested but CUDA is not available here; the CPU is used (recorded)")
    timings: Dict[str, Any] = {}
    for dev in candidates:
        log(f"timing workloads on {dev}:")
        timings[dev] = timing_workloads(suite, dev, log)
    # projected core time per device from the timing workloads
    core_path = next((p for p in suite.pilot["eta_suites"] if Path(p).stem == "core"), suite.pilot["eta_suites"][0])
    core = load_suite_config(core_path)
    projected = {}
    core_jobs = expand_jobs(core)
    core_by_id = {j.job_id: j for j in core_jobs}
    for dev in candidates:
        units = units_from_timing(timings[dev])
        projected[dev] = sum(predict_job_seconds(j, core, units, "median", core_by_id) for j in core_jobs
                             if j.type in ("fpg", "fb_pretrain", "adapt"))
    pilot_manifest = runs_root / "suites" / suite.name / "manifest.json"
    frozen = read_json(pilot_manifest).get("device", {}).get("device") if pilot_manifest.exists() else None
    if device_request in ("cpu", "cuda"):
        chosen = device_request if (device_request == "cpu" or cuda_ok) else "cpu"
        reason = f"explicit --device {device_request}" + (" (CUDA unavailable: CPU fallback)" if chosen != device_request else "")
    elif frozen is not None:
        # a resumed pilot keeps the device of its first launch; the timing workloads are re-measured and
        # recorded, but they do not switch the device of the remaining learning-pilot jobs
        chosen = frozen if (frozen == "cpu" or cuda_ok) else "cpu"
        reason = f"frozen by the learning pilot's first launch ({relpath(pilot_manifest)})"
        if chosen != frozen:
            reason += "; CUDA is unavailable now: CPU fallback"
            log("WARNING: the learning pilot was started on cuda but CUDA is not available now; the CPU is used "
                "(recorded; the ETA uses the jobs that ran on the CPU only)")
    else:
        chosen = min(projected, key=projected.get)
        reason = ("measured: smaller projected training time of the core suite from the timing workloads "
                  + ", ".join(f"{d} {projected[d]:.0f} s" for d in candidates))
    log(f"device choice: {chosen} ({reason})")
    record: Dict[str, Any] = {
        "schema": PILOT_SCHEMA, "complete": False, "recorded_utc": utc_now(), "host": here,
        "hardware": hardware_record(chosen), "pilot_config": suite.path, "pilot_config_sha256": suite.sha256,
        "pilot_seed": suite.seeds[0], "timing_workloads": timings,
        "projected_core_training_s_from_timing": projected,
        "device_choice": {"device": chosen, "reason": reason, "explicit": device_request != "auto",
                          "candidates": candidates},
    }
    write_json_atomic(out_json, record)            # incomplete record first: auto ignores it until complete
    learning: Dict[str, Any] = {"skipped": bool(skip_learning)}
    if not skip_learning:
        log(f"representative learning pilot ({suite.name}, seed {suite.seeds[0]}) on {chosen}:")
        runner = Runner(suite, runs_root, data_root, device=chosen, log=log, argv=["scripts/benchmark.py"])
        rc = runner.run(resume=True)
        status = read_json(runs_root / "suites" / suite.name / "status.json")
        manifest = read_json(runs_root / "suites" / suite.name / "manifest.json")
        learning.update({"exit_code": rc, "status_counts": status["counts"],
                         "manifest": relpath(runs_root / "suites" / suite.name / "manifest.json")})
        try:
            res = analyze(runs_root, runs_root / "pilot_results", kinds=("pilot",), names=[suite.name])
            learning["analysis"] = relpath(Path(res["out"]) / "RESULTS.md")
        except AnalysisError as exc:
            learning["analysis_error"] = str(exc)
        other_device: List[str] = []
        units = units_from_runs(runs_root, manifest, data_root, device=chosen, skipped=other_device)
        learning["per_unit_s"] = {env: {k: {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)}
                                        for k, v in d.items() if v} for env, d in units.items()}
        learning["units_device"] = chosen
        learning["jobs_on_other_device_excluded"] = other_device
        if other_device:
            log(f"WARNING: {len(other_device)} pilot job(s) ran on another device than {chosen}; their timings are "
                "excluded from the ETA")
        learning["memory"] = _memory_summary(runs_root, manifest)
        try:
            record["eta"] = eta(suite.pilot["eta_suites"], units, float(suite.pilot["overhead"]))
        except (PilotError, KeyError) as exc:
            record["eta_error"] = f"{type(exc).__name__}: {exc}"
        record["complete"] = rc == EXIT_OK
        coll = Collection(runs_root, ("pilot",), [suite.name])
        learning["summary"] = _learning_summary(coll)
    record["learning_pilot"] = learning
    record["wall_time_s"] = time.perf_counter() - t_start
    write_json_atomic(out_json, record)
    write_text_atomic(runs_root / "PILOT.md", pilot_markdown(record))
    return record


def _memory_summary(runs_root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    rss, vram = [], []
    for j in manifest["jobs"]:
        p = runs_root / "jobs" / j["job_id"] / "status.json"
        if not p.exists():
            continue
        for a in read_json(p).get("attempts", []):
            if a.get("process_peak_rss_mib") is not None:
                rss.append(a["process_peak_rss_mib"])
            if a.get("cuda_peak_allocated_mib") is not None:
                vram.append(a["cuda_peak_allocated_mib"])
    return {"process_peak_rss_mib": max(rss) if rss else None, "cuda_peak_allocated_mib": max(vram) if vram else None}


def _learning_summary(coll) -> List[Dict[str, Any]]:
    from .analysis import final_tables

    out = []
    for man in coll.suites:
        _, summary = final_tables(coll, man)
        for s in summary:
            out.append({"env": s["env"], "layout": s["layout"], "task": s["task"], "method": s["method_label"],
                        "n": s["n_completed"],
                        **{m: s[m]["mean"] for m in ("js_nats", "success", "return", "gap") if m in s}})
    return out


def pilot_markdown(rec: Dict[str, Any]) -> str:
    from .analysis import fmt

    hw = rec["hardware"]
    lines = ["# Pilot benchmark (measured; not a final result)", "",
             f"Recorded {rec['recorded_utc']} on `{rec['host']['hostname']}` ({hw['platform']}; Python {hw['python']}; "
             f"torch {hw['torch']}, CUDA build {hw['torch_cuda_build']}, CUDA available {hw['cuda_available']}"
             + (f"; GPU {hw['gpus'][0]['name']} {hw['gpus'][0]['vram_mib']:.0f} MiB" if hw.get("gpus") else "")
             + f"; {hw['logical_cpus']} logical CPUs; RAM {hw.get('ram_total_mib')} MiB). Pilot config "
             f"`{rec['pilot_config']}` (SHA-256 `{rec['pilot_config_sha256'][:16]}`), seed {rec['pilot_seed']}. "
             f"Total wall time {rec.get('wall_time_s', 0):.0f} s. `complete`: {rec['complete']}.", "",
             "## 1. Short timing workloads (synchronized; execution and speed only, not learning)", ""]
    devs = list(rec["timing_workloads"])
    keys = sorted({k for d in devs for env in ("finite", "continuous") if env in rec["timing_workloads"][d]
                   for k in rec["timing_workloads"][d][env] if isinstance(rec["timing_workloads"][d][env][k], float)})
    lines += ["| environment | quantity | " + " | ".join(devs) + " |", "|---|---|" + "---|" * len(devs)]
    for env in ("finite", "continuous"):
        for k in keys:
            vals = [rec["timing_workloads"][d].get(env, {}).get(k) for d in devs]
            if all(v is None for v in vals):
                continue
            lines.append(f"| {env} | {k} | " + " | ".join("n/a" if v is None else f"{v:.4g}" for v in vals) + " |")
    lines += ["", "## 2. Device choice", "",
              f"**{rec['device_choice']['device']}** - {rec['device_choice']['reason']}. `--device auto` uses this choice "
              "from now on (on this computer); a suite freezes its device in its manifest at the first launch.", ""]
    lp = rec.get("learning_pilot", {})
    lines += ["## 3. Representative learning pilot", ""]
    if lp.get("skipped"):
        lines += ["Skipped (`--skip-learning`): no learning evidence was collected; the record is not complete.", ""]
    else:
        lines += [f"Runner exit code {lp.get('exit_code')}; job statuses {lp.get('status_counts')}. Full analysis: "
                  f"`{lp.get('analysis', lp.get('analysis_error'))}` (curves, per-seed values, costs, FB diagnostics). "
                  f"Peak process RSS {lp.get('memory', {}).get('process_peak_rss_mib')} MiB; peak CUDA allocation "
                  f"{lp.get('memory', {}).get('cuda_peak_allocated_mib')} MiB.", "",
                  "Last-checkpoint held-out metrics of the pilot (seed 999; one seed is not evidence of stability):", "",
                  "| env | layout | goal | method | JS (nats) | success | return | gap |", "|---|---|---|---|---|---|---|---|"]
        for s in lp.get("summary", []):
            lines.append(f"| {s['env']} | {s['layout']} | {s['task']} | {s['method']} | {fmt(s.get('js_nats'))} | "
                         f"{fmt(s.get('success'))} | {fmt(s.get('return'))} | {fmt(s.get('gap'))} |")
        lines.append("")
    lines += ["## 4. Extrapolated time of the suites", ""]
    if "eta" in rec:
        excluded = lp.get("jobs_on_other_device_excluded") or []
        lines += ["Linear extrapolation of the per-unit times measured in the learning pilot's own jobs (median; range "
                  "from the fastest and slowest measured unit), plus the declared overhead. Shared jobs are counted once: "
                  "the 'extra' column of a suite excludes jobs of the core suite. Not modelled: thermal throttling, "
                  "background load, other machines, changed budgets. Uncertainty is at least the stated range. "
                  f"Per-unit times come from jobs that ran on {lp.get('units_device', 'the chosen device')}"
                  + (f"; {len(excluded)} job(s) that ran on another device are excluded." if excluded else "."), "",
                  "| suite | jobs | measured (median) | proposed with overhead | proposed range | extra jobs | extra after core (median) |",
                  "|---|---|---|---|---|---|---|"]
        for name, e in rec["eta"].items():
            lines.append(f"| {name} | {e['n_jobs']} | {_hours(e['measured_median_s'])} | {_hours(e['proposed_s'])} "
                         f"(+{int(100 * e['overhead'])}%) | {_hours(e['proposed_range_s'][0])} to {_hours(e['proposed_range_s'][1])} | "
                         f"{e['n_extra_jobs']} | {_hours(e['extra_after_previous_median_s'])} |")
        lines.append("")
    elif "eta_error" in rec:
        lines += [f"Not computed: {rec['eta_error']}", ""]
    lines += ["## What this does not show", "",
              "- The timing workloads measure speed; a short loss value or one pilot seed is not evidence that the "
              "budgets suffice or that a method converges.",
              "- Prompt 4 reads this file, `runs/benchmark.json` and the pilot analysis to choose and freeze the settings "
              "(learning rates, tau and the latent radius, KDE bandwidth or histogram, budgets) before any final run.", ""]
    return "\n".join(lines) + "\n"
