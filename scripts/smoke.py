"""Smoke test of the complete data-to-results path in BOTH environments (INTERFACE command 3).  TEST FIXTURE.

Usage, from the project root:

    python scripts/smoke.py --device auto

In a fresh directory ``runs/smoke/<UTC time>/`` (never ``runs/jobs`` or ``data/``),
with the tiny budgets of ``configs/smoke.yaml`` and the smoke seed:

1. prepares the data (demonstrations, task-blind replay with coverage check,
   held-out reference) and runs every job type through the real runner in the
   finite four-rooms maze and the continuous maze: f-PG, FB pretraining,
   ER_FB and DM_FB, held-out evaluation of every checkpoint, expert evaluation;
2. reloads every saved policy, FB snapshot and latent and compares the action
   probabilities with those recorded in-process (bitwise on the CPU);
3. checks that evaluation left every training output unchanged;
4. runs the same suite again in a second directory while requesting a stop
   (the Ctrl+C path) every few updates, resumes it until it completes, and
   compares every final output with the uninterrupted run (bitwise on the CPU);
5. analyses the uninterrupted run into ``<dir>/results`` (labelled TEST FIXTURE)
   and audits it.

Writes ``smoke_report.json`` and ``smoke_report.md``; exit status 0 iff every
check passes.  ``--device auto`` uses the pilot's saved choice if it exists,
else CUDA if available, else the CPU (the message says which).  The numbers
produced are execution checks, not results.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # launched as `python scripts/...` from the project root
    sys.path.insert(0, str(ROOT))

from il_memoir.analysis import analyze  # noqa: E402
from il_memoir.audit import audit  # noqa: E402
from il_memoir.checks import compare_runs, output_hashes_intact, reload_checks  # noqa: E402
from il_memoir.experiment import load_suite_config  # noqa: E402
from il_memoir.hardware import hardware_record, read_pilot_choice, resolve_run_device  # noqa: E402
from il_memoir.logging import StopController, read_json, utc_now, utc_stamp, write_json_atomic, write_text_atomic  # noqa: E402
from il_memoir.runner import EXIT_INTERRUPTED, EXIT_OK, Runner  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.yaml"))
    parser.add_argument("--out", default=str(ROOT / "runs" / "smoke"), help="parent directory of the smoke run")
    parser.add_argument("--pilot", default=str(ROOT / "runs" / "benchmark.json"))
    parser.add_argument("--stop-every", type=int, default=7, help="safe points between requested stops in the resume check")
    args = parser.parse_args(argv)
    t_start = time.perf_counter()
    suite = load_suite_config(args.config)
    if suite.kind != "smoke":
        print("smoke.py runs smoke suites only (kind: smoke)")
        return 2
    decision = resolve_run_device(args.device, pilot=read_pilot_choice(args.pilot))
    print(decision.message)
    device = decision.device
    base = Path(args.out) / utc_stamp()
    base.mkdir(parents=True, exist_ok=False)
    write_text_atomic(base / "SMOKE_TEST_FIXTURE.txt",
                      "TEST FIXTURE: smoke run with tiny budgets (scripts/smoke.py). Not a scientific result; safe to delete.\n")
    data_root = base / "data"
    checks = {}
    details = {"device_decision": decision.to_dict(), "hardware": hardware_record(device)}
    errors = []
    atol = 0.0 if device == "cpu" else 1e-5

    def step(name, fn):
        try:
            return fn()
        except Exception as exc:                       # recorded and reported, never hidden
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            checks[name] = False
            return None

    # 1. the complete run
    run_a = base / "runs"
    code = step("suite_run", lambda: Runner(suite, run_a, data_root, device=device, stop=StopController(),
                                             argv=["scripts/smoke.py"]).run(resume=True))
    checks["suite_run_completed"] = code == EXIT_OK
    manifest = read_json(run_a / "suites" / suite.name / "manifest.json") if (run_a / "suites" / suite.name / "manifest.json").exists() else None
    if manifest is not None:
        status = read_json(run_a / "suites" / suite.name / "status.json")
        details["status_counts"] = status["counts"]
        types = {j["type"] for j in manifest["jobs"]}
        envs = {j["labels"]["env"] for j in manifest["jobs"]}
        checks["every_job_type_planned"] = types == {"fb_pretrain", "adapt", "fpg", "eval", "eval_expert"}
        checks["both_environments_planned"] = envs == {"finite", "continuous"}
        checks["every_job_completed"] = status["counts"] == {"completed": len(manifest["jobs"])}
        # 2.-3. reload and integrity
        rel = step("reload", lambda: reload_checks(run_a, manifest, atol=atol)) or {}
        checks.update(rel)
        checks["reload_checked_every_training_job"] = len(rel) == sum(1 for j in manifest["jobs"] if j["type"] in ("fpg", "fb_pretrain", "adapt"))
        checks.update(step("outputs_intact", lambda: output_hashes_intact(run_a, manifest)) or {})
    # 4. interrupted and resumed run
    run_b = base / "runs_interrupted"

    def interrupted_run():
        launches, interruptions = 0, 0
        while True:
            stop = StopController()
            stop.stop_after = args.stop_every
            launches += 1
            rc = Runner(suite, run_b, data_root, device=device, stop=stop, argv=["scripts/smoke.py"]).run(resume=True)
            if rc == EXIT_INTERRUPTED:
                interruptions += 1
                if launches > 200:
                    raise RuntimeError("the interrupted run did not finish in 200 launches")
                continue
            return rc, launches, interruptions

    res = step("interrupted_run", interrupted_run)
    if res is not None and manifest is not None:
        rc, launches, interruptions = res
        details["resume"] = {"launches": launches, "interruptions": interruptions, "stop_every_safe_points": args.stop_every}
        checks["interrupted_run_completed"] = rc == EXIT_OK
        checks["interruptions_happened"] = interruptions >= 2
        attempts = {}
        for j in manifest["jobs"]:
            st = read_json(run_b / "jobs" / j["job_id"] / "status.json")
            attempts[j["job_id"]] = len(st["attempts"])
        details["resume"]["attempts_per_job"] = attempts
        checks["some_job_resumed_from_checkpoint"] = any(v > 1 for v in attempts.values())
        cmp = step("compare", lambda: compare_runs(run_a, run_b, manifest, atol=atol))
        if cmp:
            checks.update(cmp["checks"])
            details["resume"]["max_abs_diff"] = cmp["max_abs_diff"]
    # 5. analysis and audit of the fixture
    results = base / "results"
    out = step("analysis", lambda: analyze(run_a, results, kinds=("smoke",)))
    if out is not None:
        text = (results / "RESULTS.md").read_text(encoding="utf-8")
        checks["results_labelled_test_fixture"] = text.startswith("> **TEST FIXTURE")
        rep = audit(run_a, results, kinds=("smoke",))
        details["audit"] = rep.to_dict()
        checks["audit_passes"] = not rep.errors
    all_ok = bool(checks) and all(checks.values()) and not errors
    report = {"test_fixture": True, "recorded_utc": utc_now(), "wall_time_s": time.perf_counter() - t_start,
              "directory": str(base), "device": device, "all_checks_pass": all_ok, "checks": checks,
              "errors": errors, "details": details}
    write_json_atomic(base / "smoke_report.json", report)
    lines = ["# Smoke report (TEST FIXTURE - not a result)", "",
             f"- Recorded (UTC): {report['recorded_utc']}; wall time {report['wall_time_s']:.1f} s; device {device} "
             f"({decision.message}); exit status {0 if all_ok else 1}",
             f"- Directory: `{base}`", "", "| Check | Result |", "|---|---|"]
    lines += [f"| {k} | {'pass' if v else 'FAIL'} |" for k, v in checks.items()]
    if errors:
        lines += ["", "## Errors", ""] + [f"- {e}" for e in errors]
    write_text_atomic(base / "smoke_report.md", "\n".join(lines) + "\n")
    for k, v in checks.items():
        print(f"{'pass' if v else 'FAIL'}  {k}")
    print(f"smoke directory: {base}")
    print("ALL SMOKE CHECKS PASS" if all_ok else "SOME SMOKE CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
