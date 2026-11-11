"""Audit provenance, seeds and metric integrity of saved runs and generated results (INTERFACE command 9).

Usage, from the project root:

    python scripts/audit_results.py --runs runs --results results

Checks (il_memoir/audit.py): seeds and reference seeds against
configs/seeds.yaml and the suite manifests (no seed of an earlier revision
dropped); each suite configuration file against its manifest hash; job
identities and output hashes; deleted job directories and lost failed
attempts against the outcome journals; one current code version per job type;
metric ranges and interaction counts; provenance.json hashes of every table,
figure and RESULTS.md; an independent recomputation of the per-run table, the
summary means and standard deviations, the cost table and the threshold
table; no smoke or pilot run in results/.  Writes
``<results>/audit_report.json`` (when the results directory exists) and exits
with status 1 if any integrity error is found, 0 otherwise.  Warnings (jobs not
completed or not run yet, mixed devices, revisions that removed planned jobs)
do not change the exit status but are printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # launched as `python scripts/...` from the project root
    sys.path.insert(0, str(ROOT))

from il_memoir.audit import audit, write_report  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default=str(ROOT / "runs"), help="runs directory (default: runs/)")
    parser.add_argument("--results", default=str(ROOT / "results"), help="results directory (default: results/)")
    parser.add_argument("--include-kind", action="append", default=[], choices=["smoke", "pilot"],
                        help="audit smoke or pilot suites instead of the final ones")
    parser.add_argument("--runs-only", action="store_true", help="audit the runs without a results directory")
    args = parser.parse_args(argv)
    kinds = list(args.include_kind) or ["final"]
    results = None if args.runs_only else Path(args.results)
    rep = audit(args.runs, results, kinds=kinds)
    if results is not None and results.exists():
        write_report(rep, results / "audit_report.json")
    for w in rep.warnings:
        print("warning: " + w)
    for e in rep.errors:
        print("ERROR:   " + e)
    print(f"{sum(rep.checks.values())} checks in {len(rep.checks)} groups; {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
    print("AUDIT PASSED" if not rep.errors else "AUDIT FAILED")
    return 0 if not rep.errors else 1


if __name__ == "__main__":
    sys.exit(main())
