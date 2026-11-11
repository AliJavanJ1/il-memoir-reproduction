"""Run a suite sequentially with cached, resumable jobs (INTERFACE commands 5-7).

Usage, from the project root:

    python scripts/run_suite.py --config configs/core.yaml --device auto --resume
    python scripts/run_suite.py --config configs/compact_ablations.yaml --device auto --resume
    python scripts/run_suite.py --config configs/reduced.yaml --device auto --resume
    python scripts/run_suite.py --config configs/core.yaml --status        # statuses only, nothing launched

The runner writes ``runs/suites/<suite>/manifest.json`` before launching
anything, prepares missing data, skips completed jobs whose code, settings,
data and dependencies are unchanged, resumes interrupted jobs from their
checkpoint (optimizer and random-generator states restored), records failures
without retrying them (``--retry-failed`` reruns them) and never mixes outputs
of changed code: stale jobs stop the launch unless ``--invalidate-changed``
moves them to ``runs/invalidated/``.  Ctrl+C makes the running job save a
checkpoint and stop (no further job starts); run the same command again to
resume.  A seed removed from ``configs/seeds.yaml`` after the launch is always
refused; other seed or configuration changes need ``--allow-config-change``
and are recorded as manifest revisions.  Every job outcome is appended to
``runs/suites/<suite>/outcomes.jsonl``; a deleted job directory is reported.

``--device auto`` uses the device frozen in the suite manifest, else the
pilot's saved choice (``runs/benchmark.json``); before the pilot exists a
final suite needs an explicit ``--device cpu`` or ``--device cuda``.

Exit status: 0 all jobs completed; 1 some job failed or is blocked;
2 configuration, device or data error; 3 stale jobs (nothing launched);
130 interrupted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # launched as `python scripts/...` from the project root
    sys.path.insert(0, str(ROOT))

from il_memoir import datasets as ds  # noqa: E402
from il_memoir.experiment import ConfigError, load_suite_config  # noqa: E402
from il_memoir.hardware import DeviceError  # noqa: E402
from il_memoir.logging import LockError  # noqa: E402
from il_memoir.prepare import DataConflict  # noqa: E402
from il_memoir.runner import EXIT_CONFIG, Runner, RunnerError  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="suite configuration (configs/core.yaml, ...)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", action="store_true", help="continue a suite that was launched before")
    parser.add_argument("--runs", default=str(ROOT / "runs"), help="runs directory (default: runs/)")
    parser.add_argument("--data-root", default=str(ROOT / "data"), help="data directory (default: data/)")
    parser.add_argument("--pilot", default=None, help="pilot record with the device choice (default: <runs>/benchmark.json)")
    parser.add_argument("--status", action="store_true", help="print the job statuses and exit without launching")
    parser.add_argument("--invalidate-changed", action="store_true",
                        help="move stale jobs (code, settings, data or dependency changed) to runs/invalidated/ and rerun them")
    parser.add_argument("--allow-config-change", action="store_true",
                        help="accept a configuration that differs from the frozen manifest (recorded as a revision)")
    parser.add_argument("--retry-failed", action="store_true", help="rerun failed jobs (their earlier attempts are kept)")
    args = parser.parse_args(argv)
    try:
        suite = load_suite_config(args.config)
    except (ConfigError, ValueError, OSError) as exc:
        print(f"configuration error: {exc}")
        return EXIT_CONFIG
    runner = Runner(suite, runs_root=args.runs, data_root=args.data_root, device=args.device, pilot_path=args.pilot,
                    allow_config_change=args.allow_config_change, invalidate_changed=args.invalidate_changed,
                    retry_failed=args.retry_failed, argv=[Path(sys.argv[0]).as_posix()] + (argv if argv is not None else sys.argv[1:]))
    if args.status:
        rep = runner.status_report()
        for job_id, (st, reason) in rep["jobs"].items():
            print(f"{st:12s} {job_id}" + (f"  ({reason})" if reason else ""))
        print("counts: " + ", ".join(f"{v} {k}" for k, v in sorted(rep["counts"].items())))
        return 0
    print(f"suite {suite.name} ({suite.kind}): {len(runner.jobs)} jobs; config {suite.path} sha256 {suite.sha256[:12]}")
    try:
        return runner.run(resume=args.resume)
    except (RunnerError, DeviceError, LockError, DataConflict, ds.DataError) as exc:
        print(f"ERROR: {exc}")
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
