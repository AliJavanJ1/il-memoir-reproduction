"""Pilot benchmark: CPU/GPU timing, device choice, representative learning pilot and ETA (INTERFACE command 4).

Usage, from the project root (run it natively on the laptop):

    python scripts/benchmark.py --config configs/pilot.yaml --device auto

1. Times short synchronized workloads on the CPU and, if available, on CUDA:
   replay collection, FB updates and Gram matrix, DM steps, f-PG iterations
   (finite counts, continuous KDE and histogram), evaluation, checkpointing.
2. Chooses the device with the smaller projected core training time (or uses
   an explicit ``--device cpu|cuda``, recorded as explicit).
3. Runs the representative learning pilot (suite ``pilot``: seed 999, core
   budgets, declared grid) with the real runner on that device; its jobs stay
   in ``runs/jobs/`` and its analysis is written to ``runs/pilot_results/``.
   ``--resume`` semantics are automatic: completed pilot jobs are reused.
4. Extrapolates the time of core, compact ablations and reduced suites from
   the per-unit costs measured in the pilot's jobs, adds the declared overhead
   (at least 50%) and states the range.

Writes ``runs/benchmark.json`` (the saved device choice used by ``--device
auto``; ``complete`` only when every pilot job completed) and ``runs/PILOT.md``.
A ``benchmark.json`` recorded on another computer is not overwritten unless
``--force``; ``runs/environment_laptop.json`` is never written.  Exit status 0
iff the pilot completed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # launched as `python scripts/...` from the project root
    sys.path.insert(0, str(ROOT))

from il_memoir.experiment import ConfigError  # noqa: E402
from il_memoir.hardware import DeviceError  # noqa: E402
from il_memoir.logging import LockError  # noqa: E402
from il_memoir.pilot import PilotError, run_benchmark  # noqa: E402
from il_memoir.runner import RunnerError  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="pilot configuration (configs/pilot.yaml)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="auto: measure CPU and CUDA and choose; cpu/cuda: explicit choice (recorded)")
    parser.add_argument("--runs", default=str(ROOT / "runs"), help="runs directory (default: runs/)")
    parser.add_argument("--data-root", default=str(ROOT / "data"), help="data directory (default: data/)")
    parser.add_argument("--force", action="store_true", help="overwrite a benchmark.json recorded on another computer")
    parser.add_argument("--skip-learning", action="store_true",
                        help="timing and device choice only (the record stays incomplete: --device auto ignores it)")
    args = parser.parse_args(argv)
    try:
        rec = run_benchmark(args.config, args.device, args.runs, args.data_root, force=args.force,
                            skip_learning=args.skip_learning)
    except (PilotError, ConfigError, DeviceError, RunnerError, LockError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"device choice: {rec['device_choice']['device']} ({rec['device_choice']['reason']})")
    for name, e in rec.get("eta", {}).items():
        print(f"ETA {name}: measured {e['measured_median_s'] / 60:.1f} min; proposed with +{int(100 * e['overhead'])}% "
              f"overhead {e['proposed_s'] / 60:.1f} min (range {e['proposed_range_s'][0] / 60:.1f} to "
              f"{e['proposed_range_s'][1] / 60:.1f} min)")
    print(f"wrote {Path(args.runs) / 'benchmark.json'} and {Path(args.runs) / 'PILOT.md'}; complete = {rec['complete']}")
    return 0 if rec["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
