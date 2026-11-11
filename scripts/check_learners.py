"""Learner execution check on one device (Prompt 3B).  Short runs only: no study result is produced.

Usage, from the project root:

    python scripts/check_learners.py --device cpu  --out audit/prompt3B/learner_check_cpu
    python scripts/check_learners.py --device cuda --out audit/prompt3B/learner_check_laptop_cuda

On the requested device (``cuda`` must be available, otherwise the script
stops; ``auto`` picks CUDA when available and says so) it runs, with real
simulator data and real demonstrations:

* f-PG on the finite four-rooms maze (counts) and on the continuous maze (KDE
  and histogram signal): a few iterations, save/reload, and an interrupted run
  resumed from its checkpoint compared with an uninterrupted run;
* task-blind FB pretraining on both mazes: replay collection and coverage, a
  few hundred updates, the Gram matrix and diagnostics, save/reload, resume;
* ER_FB and DM_FB (300 latent steps) on both pretrained models, with the
  freeze check (every network parameter and ``G_hat_B`` unchanged);
* on CUDA: the same FB update, f-PG update and DM gradient computed on the CPU
  and on CUDA, compared to float32 precision.

These runs check execution, finiteness, determinism and device agreement; the
timings are short-workload measurements, not the pilot of Prompt 4, and the
losses are not evidence of learning.  Writes ``learner_check.json`` and
``learner_check.md`` to ``--out``; exit status 0 iff every check passes.
Only ASCII is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # launched as `python scripts/...` from the project root
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from il_memoir import fb, fb_il, fpg  # noqa: E402
from il_memoir.common import make_rng  # noqa: E402
from il_memoir.config import load_environment_config  # noqa: E402
from il_memoir.env_continuous import make_continuous_maze  # noqa: E402
from il_memoir.env_tabular import make_grid_maze, make_layout  # noqa: E402
from il_memoir.expert import finite_expert, generate_demonstrations, route_controller  # noqa: E402
from il_memoir.nets import ObsSpec, params_fingerprint, resolve_device  # noqa: E402
from il_memoir.tasks import make_task  # noqa: E402


def file_hashes(paths):
    """SHA-256 of each file with line endings normalized to LF (CRLF checkouts hash the same)."""
    out = {}
    for p in sorted(paths):
        data = p.read_bytes().replace(b"\r\n", b"\n")
        out[str(p.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(data).hexdigest()
    combined = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in out.items()).encode()).hexdigest()
    return out, combined


def environment_record(device):
    rec = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
    }
    if device.type == "cuda":
        rec["gpu_name"] = torch.cuda.get_device_name(device)
        rec["cudnn"] = torch.backends.cudnn.version()
    return rec


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def roundtrip(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=True)


def max_param_diff(m1, m2):
    return max(float((p1.detach().cpu().double() - p2.detach().cpu().double()).abs().max())
               for p1, p2 in zip(m1.parameters(), m2.parameters()))


def finite_dict(d):
    return all(np.isfinite(v) for v in d.values() if isinstance(v, (int, float)) and not isinstance(v, bool))


# ----------------------------------------------------------------------------- f-PG
def check_fpg(env, view, demos, cfg, device, n_iter, seed):
    out, checks = {}, {}
    learner = fpg.FPGLearner(env, view, demos, cfg, seed=seed, device=device)
    logs, times = [], []
    for _ in range(n_iter):
        sync(device)
        t0 = time.perf_counter()
        logs.append(learner.step(env))
        sync(device)
        times.append(time.perf_counter() - t0)
    out["last_metrics"] = logs[-1]
    out["seconds_per_iteration"] = float(np.mean(times))
    checks["metrics_finite"] = all(finite_dict({k: v for k, v in m.items() if k != "grad_norm"}) for m in logs)
    checks["env_steps_counted"] = learner.env_steps == n_iter * cfg.n_traj * view["horizon"]
    # resume: 2 + (n_iter - 2) iterations from a checkpoint equal n_iter iterations
    half = fpg.FPGLearner(env, view, demos, cfg, seed=seed, device=device)
    for _ in range(2):
        half.step(env)
    resumed = fpg.FPGLearner(env, view, demos, cfg, seed=seed, device=device)
    resumed.load_state_dict(roundtrip(half.state_dict()))
    for _ in range(n_iter - 2):
        resumed.step(env)
    diff = max_param_diff(learner.policy, resumed.policy)
    out["resume_max_param_diff"] = diff
    checks["resume_identical"] = diff == 0.0 if device.type == "cpu" else diff <= 1e-5
    back = fpg.policy_from_record(roundtrip(fpg.policy_record(learner.policy)), device)
    probe = demos[:, :5].reshape((-1,) + demos.shape[2:])
    checks["reload_identical"] = bool(np.array_equal(fpg.policy_callable(back)(probe), learner.policy_callable()(probe)))
    return out, checks


# ----------------------------------------------------------------------------- FB, ER, DM
def check_fb(env, view, demos, device, n_transitions, n_updates, seed):
    out, checks = {}, {}
    spec = ObsSpec.from_env(env)
    t0 = time.perf_counter()
    buf = fb.collect_replay(env, fb.ReplayConfig(n_transitions, 50), make_rng(seed, "check", "replay", spec.kind))
    out["replay_seconds"] = time.perf_counter() - t0
    cov = fb.replay_coverage(buf, env.n_actions, view["grid"]["size"])
    out["coverage"] = {k: v for k, v in cov.items() if not isinstance(v, np.ndarray)}
    out["coverage"]["action_counts"] = [int(v) for v in cov["action_counts"]]
    if spec.finite:
        checks["replay_covers_all_state_actions"] = cov["min_state_action_count"] > 0 and cov["states_visited"] == spec.n_states
    else:
        checks["replay_covers_all_lattice_cells_and_actions"] = cov["cells_visited"] == 40 and cov["min_cell_action_count"] > 0
    cfg = fb.FBConfig(n_updates=n_updates)
    trainer = fb.FBPretrainer(spec, env.n_actions, view["gamma"], cfg, seed=seed, device=device)
    trainer.attach_buffer(buf)
    sync(device)
    t0 = time.perf_counter()
    res = trainer.train(n_updates, log_every=max(1, n_updates // 5))
    sync(device)
    out["seconds_per_update"] = (time.perf_counter() - t0) / n_updates
    out["logs"] = res["logs"]
    checks["losses_finite"] = all(finite_dict(m) for m in res["logs"])
    half = fb.FBPretrainer(spec, env.n_actions, view["gamma"], cfg, seed=seed, device=device)
    half.attach_buffer(buf)
    half.train(n_updates // 2)
    resumed = fb.FBPretrainer(spec, env.n_actions, view["gamma"], cfg, seed=seed, device=device)
    resumed.attach_buffer(buf)
    resumed.load_state_dict(roundtrip(half.state_dict()))
    resumed.train(n_updates - n_updates // 2)
    diff = max(max_param_diff(trainer.F, resumed.F), max_param_diff(trainer.B, resumed.B))
    out["resume_max_param_diff"] = diff
    checks["resume_identical"] = diff == 0.0 if device.type == "cpu" else diff <= 1e-4
    model = trainer.model()
    out["diagnostics"] = fb.diagnostics(model, buf, make_rng(seed, "check", "diag", spec.kind))
    checks["gram_finite_symmetric_psd"] = bool(
        np.all(np.isfinite(model.gram64)) and np.allclose(model.gram64, model.gram64.T)
        and np.linalg.eigvalsh(model.gram64).min() > -1e-8
    )
    back = fb.FBModel.from_record(roundtrip(model.record()), device)
    z = fb.sample_latents(1, cfg.d, cfg.radius, make_rng(seed, "check", "z"))[0]
    checks["model_reload_identical"] = bool(np.array_equal(back.numpy_policy(z)(buf.obs[:256]), model.numpy_policy(z)(buf.obs[:256])))
    # ER and DM with the frozen online copy
    before = params_fingerprint(model)
    er = fb_il.er_fb(model, demos)
    out["er_norm_before_projection"] = er.norm_raw
    sync(device)
    t0 = time.perf_counter()
    dm = fb_il.dm_fit(model, demos, fb_il.DMConfig())
    sync(device)
    out["dm_seconds"] = time.perf_counter() - t0
    out["dm_first"] = dm.log[0]
    out["dm_last"] = dm.log[-1]
    checks["er_on_sphere"] = abs(np.linalg.norm(er.z) - cfg.radius) < 1e-6
    checks["dm_logs_finite"] = all(finite_dict(e) for e in dm.log) and len(dm.log) == 301
    checks["dm_model_frozen"] = bool(torch.equal(before, params_fingerprint(model))) and all(
        p.grad is None and not p.requires_grad for p in model.parameters()
    )
    checks["dm_z_on_sphere"] = abs(np.linalg.norm(dm.z) - cfg.radius) < 1e-3
    probs = model.numpy_policy(dm.z)(demos[:, 0])
    checks["dm_policy_rows_valid"] = bool(np.allclose(probs.sum(axis=1), 1.0, atol=1e-12) and (probs >= 0).all())
    return out, checks, buf, model


def check_cpu_cuda_agreement(env, view, demos, buf, model_cpu, seed):
    """The same first FB update, the same f-PG update and the same DM gradient on the CPU and on CUDA."""
    out, checks = {}, {}
    spec = ObsSpec.from_env(env)
    cfg = fb.FBConfig()
    runs = {}
    for dev in ("cpu", "cuda"):
        tr = fb.FBPretrainer(spec, env.n_actions, view["gamma"], cfg, seed=seed, device=dev)
        tr.attach_buffer(buf)
        runs[dev] = (tr.update(), tr)
    rel = abs(runs["cpu"][0]["loss"] - runs["cuda"][0]["loss"]) / max(1.0, abs(runs["cpu"][0]["loss"]))
    pdiff = max_param_diff(runs["cpu"][1].F, runs["cuda"][1].F)
    out["fb_first_update_loss_rel_diff"] = rel
    out["fb_first_update_param_max_diff"] = pdiff
    checks["fb_update_agrees"] = rel <= 1e-4 and pdiff <= 1e-5
    fcfg = fpg.FPGConfig(n_traj=10)
    learners = {dev: fpg.FPGLearner(env, view, demos, fcfg, seed=seed, device=dev) for dev in ("cpu", "cuda")}
    for learner in learners.values():
        learner.step(env)
    pd = max_param_diff(learners["cpu"].policy, learners["cuda"].policy)
    out["fpg_first_update_param_max_diff"] = pd
    checks["fpg_update_agrees"] = pd <= 1e-5
    data_cpu = fb_il.dm_targets(model_cpu, demos, 60)
    z = fb_il.er_fb(model_cpu, demos).z
    g_cpu = fb_il.dm_grad(model_cpu, z, data_cpu)
    model_gpu = fb.FBModel.from_record(model_cpu.record(), "cuda")
    g_gpu = fb_il.dm_grad(model_gpu, z, fb_il.dm_targets(model_gpu, demos, 60))
    gdiff = float(np.abs(g_cpu - g_gpu).max() / max(1.0, np.abs(g_cpu).max()))
    out["dm_grad_rel_max_diff"] = gdiff
    checks["dm_grad_agrees"] = gdiff <= 1e-3
    return out, checks


def write_markdown(report, path):
    meta = report["meta"]
    lines = [
        "# Learner execution check (Prompt 3B)",
        "",
        f"- Recorded (UTC): {meta['recorded_utc']}; wall time {meta['wall_time_s']:.1f} s; exit status {0 if report['all_checks_pass'] else 1}",
        f"- Command: `{meta['command']}`",
        f"- Device requested `{meta['device_requested']}`, used `{meta['environment']['device']}`"
        + (f" ({meta['environment'].get('gpu_name')})" if meta["environment"].get("gpu_name") else ""),
        f"- Python {meta['environment']['python']}, torch {meta['environment']['torch']} (CUDA build {meta['environment']['torch_cuda_build']}), "
        f"NumPy {meta['environment']['numpy']}; {meta['environment']['platform']}",
        f"- il_memoir code sha256: `{meta['code_sha256']}`; seed {meta['seed']}",
        "- These are execution checks on short workloads, not study results and not the Prompt 4 pilot.",
        "",
        "| Section | Check | Result |",
        "|---|---|---|",
    ]
    for section, checks in report["checks"].items():
        for name, ok in checks.items():
            lines.append(f"| {section} | {name} | {'pass' if ok else 'FAIL'} |")
    lines += ["", "## Measured short-workload timings (seconds)", ""]
    for key, value in report["timings"].items():
        lines.append(f"- {key}: {value:.4g}")
    if report.get("errors"):
        lines += ["", "## Errors", ""] + [f"- {e}" for e in report["errors"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--out", default=str(ROOT / "audit" / "prompt3B" / "learner_check"))
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--fpg-iterations", type=int, default=4)
    parser.add_argument("--fb-updates", type=int, default=400)
    parser.add_argument("--replay", type=int, default=10_000)
    args = parser.parse_args(argv)
    if args.fpg_iterations < 3:
        parser.error("--fpg-iterations must be at least 3 (the resume check splits the run)")
    t_start = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but torch.cuda.is_available() is False; no check was run.")
        return 2
    device = resolve_device(args.device)
    if args.device == "auto":
        print(f"--device auto resolved to {device}")
    torch.zeros(4, device=device).sum().item()          # the device actually executes a kernel
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_environment_config()
    view = cfg.learner_dict()
    maze = make_grid_maze(cfg, "four_rooms")
    cont = make_continuous_maze(cfg)
    expert = finite_expert(maze, make_task(cfg, "goal_5_5"), cfg.gamma, cfg.horizon, cfg.value_iteration_tol, cfg.tie_tol)
    demos_f = generate_demonstrations(maze, expert.pi, 4, cfg.horizon, (args.seed, ("check", "demos", "finite"))).learner_states()
    ctrl = route_controller(make_layout(cfg, cfg.continuous_layout), cfg.goal_cell("goal_5_5"), cfg.action_deltas)
    demos_c = generate_demonstrations(cont, ctrl, 4, cfg.horizon, (args.seed, ("check", "demos", "continuous"))).learner_states()
    report = {"checks": {}, "timings": {}, "details": {}, "errors": []}

    def run(section, fn):
        try:
            return fn()
        except Exception as exc:                   # recorded, never hidden
            report["errors"].append(f"{section}: {type(exc).__name__}: {exc}")
            report["checks"][section] = {"completed": False}
            traceback.print_exc()
            return None

    res = run("fpg_finite", lambda: check_fpg(maze, view, demos_f, fpg.FPGConfig(), device, args.fpg_iterations, args.seed))
    if res:
        report["details"]["fpg_finite"], report["checks"]["fpg_finite"] = res
        report["timings"]["fpg_finite_seconds_per_iteration (20 traj x 100 steps)"] = res[0]["seconds_per_iteration"]
    for dens in ("kde", "histogram"):
        res = run(f"fpg_continuous_{dens}", lambda: check_fpg(
            cont, view, demos_c, fpg.FPGConfig(density=dens, lr=1e-3), device, args.fpg_iterations, args.seed))
        if res:
            report["details"][f"fpg_continuous_{dens}"], report["checks"][f"fpg_continuous_{dens}"] = res
            report["timings"][f"fpg_continuous_{dens}_seconds_per_iteration (20 traj x 100 steps)"] = res[0]["seconds_per_iteration"]
    models = {}
    for kind, env, demos in (("finite", maze, demos_f), ("continuous", cont, demos_c)):
        res = run(f"fb_{kind}", lambda: check_fb(env, view, demos, device, args.replay, args.fb_updates, args.seed))
        if res:
            details, checks, buf, model = res
            report["details"][f"fb_{kind}"], report["checks"][f"fb_{kind}"] = details, checks
            report["timings"][f"fb_{kind}_seconds_per_update (batch 128)"] = details["seconds_per_update"]
            report["timings"][f"dm_{kind}_seconds_for_300_steps"] = details["dm_seconds"]
            models[kind] = (env, demos, buf, model)
    if device.type == "cuda":
        for kind, (env, demos, buf, model) in models.items():
            cpu_model = fb.FBModel.from_record(model.record(), "cpu")
            res = run(f"cpu_vs_cuda_{kind}", lambda: check_cpu_cuda_agreement(env, view, demos, buf, cpu_model, args.seed))
            if res:
                report["details"][f"cpu_vs_cuda_{kind}"], report["checks"][f"cpu_vs_cuda_{kind}"] = res
    hashes, combined = file_hashes(list((ROOT / "il_memoir").glob("*.py")))
    all_ok = not report["errors"] and all(all(c.values()) for c in report["checks"].values())
    report["all_checks_pass"] = bool(all_ok)
    report["meta"] = {
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_s": time.perf_counter() - t_start,
        "command": "python " + " ".join([Path(sys.argv[0]).as_posix()] + (argv if argv is not None else sys.argv[1:])),
        "device_requested": args.device,
        "environment": environment_record(device),
        "seed": args.seed,
        "code_sha256": combined,
        "code_files": hashes,
        "settings": {"fpg_iterations": args.fpg_iterations, "fb_updates": args.fb_updates, "replay": args.replay},
    }
    (out_dir / "learner_check.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float), encoding="utf-8")
    write_markdown(report, out_dir / "learner_check.md")
    for section, checks in report["checks"].items():
        for name, ok in checks.items():
            print(f"{'pass' if ok else 'FAIL'}  {section}: {name}")
    for key, value in report["timings"].items():
        print(f"time  {key}: {value:.4g}")
    print("ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
