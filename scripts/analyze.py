"""Generate results/RESULTS.md, tables, figures and provenance.json from real saved runs (INTERFACE command 8).

Usage, from the project root:

    python scripts/analyze.py --runs runs --out results

Reads the manifests of the final suites in ``runs/suites/`` (core,
compact_ablations, reduced; pilot and smoke suites are excluded), their job
statuses, results and evaluation rows.  It lists planned, completed, failed
and missing runs, reports every seed with mean and sample standard deviation
over completed seeds only (missing seeds are never invented), the last
checkpoint of every run, operational convergence, per-phase costs and FB
diagnostics, and writes ``provenance.json`` mapping every table and figure to
job identifiers and hashes.  If no learner evaluation is completed, nothing is
written and the exit status is 1.  ``--include-kind smoke`` analyses smoke
test fixtures into another directory (never ``results/``), labelled as such.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # launched as `python scripts/...` from the project root
    sys.path.insert(0, str(ROOT))

from il_memoir.analysis import AnalysisError, analyze  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default=str(ROOT / "runs"), help="runs directory (default: runs/)")
    parser.add_argument("--out", default=str(ROOT / "results"), help="output directory (default: results/)")
    parser.add_argument("--suites", default=None, help="comma-separated suite names (default: every final suite)")
    parser.add_argument("--include-kind", action="append", default=[], choices=["smoke", "pilot"],
                        help="also analyse smoke or pilot suites (test fixtures; never into results/)")
    parser.add_argument("--no-figures", action="store_true", help="tables and RESULTS.md only")
    args = parser.parse_args(argv)
    kinds = ["final"] + list(args.include_kind) if not args.include_kind else list(args.include_kind)
    names = [s.strip() for s in args.suites.split(",")] if args.suites else None
    try:
        out = analyze(args.runs, args.out, kinds=kinds, names=names, figures=not args.no_figures)
    except AnalysisError as exc:
        print(f"no results written: {exc}")
        return 1
    print(f"suites analysed: {', '.join(out['suites'])}; {out['n_eval_rows']} evaluation rows")
    print(f"wrote {Path(out['out']) / 'RESULTS.md'}, tables/, figures/ and provenance.json")
    print("next: python scripts/audit_results.py --runs " + args.runs + " --results " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
