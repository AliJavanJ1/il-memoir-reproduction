"""Compile the report and slides with one available LaTeX engine and save the logs (INTERFACE command 10).

Usage, from the project root:

    python scripts/build.py --target all          # report/report.tex and slides/slides.tex
    python scripts/build.py --target report
    python scripts/build.py --target theory       # theory/theory.tex (the proof compendium)

The engine is the first available of latexmk (with pdflatex), pdflatex and
tectonic, the same for every target of the call (``--engine`` forces one).
Intermediate files go to a temporary directory.  Each target writes its PDF,
``build.log`` (the complete engine output) and ``build.json`` (engine, version,
pages, errors, undefined references and citations, box warnings, hashes) next
to its main file.  A target counts as built only if the engine exits with 0,
the PDF exists and the log has no error and no undefined reference or
citation; otherwise the previous PDF is kept and the exit status is 1.  If no
engine is installed, nothing is compiled, the script says so and exits with 2:
use ``python scripts/package.py --out deliverables`` and compile the Overleaf ZIP.
The report and slides are written in Prompts 5 and 6; until then ``all``
reports them as missing.  Note: rebuilding ``theory`` with another TeX
distribution can shift page numbers that METHOD_MAP.md cites.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # launched as `python scripts/...` from the project root
    sys.path.insert(0, str(ROOT))

from il_memoir.documents import ALL, TARGETS, BuildError, build_targets  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--target", required=True, choices=["all"] + sorted(TARGETS))
    parser.add_argument("--engine", default=None, choices=["latexmk", "pdflatex", "tectonic"],
                        help="force one engine (default: the first available)")
    parser.add_argument("--out-root", default=None,
                        help="write <out-root>/<target>/ instead of next to the sources (checks)")
    args = parser.parse_args(argv)
    targets = list(ALL) if args.target == "all" else [args.target]
    try:
        code, _ = build_targets(targets, ROOT, args.engine, Path(args.out_root) if args.out_root else None)
    except BuildError as exc:
        print(f"ERROR: {exc}")
        return 2
    print("BUILD SUCCEEDED" if code == 0 else ("NO LATEX ENGINE: NOTHING COMPILED" if code == 2 else "BUILD FAILED"))
    return code


if __name__ == "__main__":
    sys.exit(main())
