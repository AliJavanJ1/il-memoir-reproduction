"""Write the deliverable bundles with SHA-256 checksums (INTERFACE command 11).

Usage, from the project root:

    python scripts/package.py --out deliverables
    python scripts/package.py --out deliverables --verify --include-large

Writes to ``--out``: the reproduction bundle (code, tests, configurations with
the seed manifest, dependency and hardware records, documents, compact run
records, data manifests and demonstrations, results, report and slides),
the audit-review bundle for the independent review, one self-contained
Overleaf ZIP per existing LaTeX document (theory, report, slides; main ``.tex``
at the archive root), optionally the large artifacts (replay buffers,
references, model files), ``CHECKSUMS.sha256`` and ``MANIFEST.json``.  Excludes
``.venv`` and other virtual environments, ``.git``, caches, smoke runs, resume
checkpoints, LaTeX intermediates and anything that looks like a secret.
Regenerates ``CHECKSUMS.sha256`` at the project root (``--no-root-checksums``
skips it).  ``--verify`` test-compiles each Overleaf ZIP in a temporary copy
(if a LaTeX engine is installed; never claimed otherwise; the log is kept in
``<bundle>-verification/``), runs the smoke check inside an extracted copy of
the reproduction bundle and the audit inside an extracted copy of the
audit-review bundle.  Exit status 0 iff every bundle was written and, with
``--verify``, no check failed (``packaging.verification_failures``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # launched as `python scripts/...` from the project root
    sys.path.insert(0, str(ROOT))

from il_memoir.documents import BuildError  # noqa: E402
from il_memoir.packaging import package, verification_failures  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="output directory, e.g. deliverables")
    parser.add_argument("--include-large", action="store_true", help="also bundle the replay buffers and held-out references")
    parser.add_argument("--verify", action="store_true",
                        help="test-compile the Overleaf ZIPs and run the smoke check in an extracted reproduction bundle")
    parser.add_argument("--no-root-checksums", action="store_true", help="do not regenerate CHECKSUMS.sha256 at the project root")
    args = parser.parse_args(argv)
    try:
        man = package(Path(args.out), ROOT, include_large=args.include_large, verify=args.verify,
                      update_root_checksums=not args.no_root_checksums)
    except BuildError as exc:
        print(f"ERROR: {exc}")
        return 1
    failures = verification_failures(man) if args.verify else []
    for name, b in man["bundles"].items():
        print(f"{name}: " + (b["skipped"] if "skipped" in b else f"{len(b['files'])} files, sha256 {b.get('sha256', '')[:16]}"))
    print(f"wrote {Path(args.out) / 'CHECKSUMS.sha256'} and {Path(args.out) / 'MANIFEST.json'}")
    for f in failures:
        print("VERIFICATION FAILED: " + f)
    print("PACKAGING OK" if not failures else "PACKAGING CHECK FAILED")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
