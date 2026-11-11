"""Deliverable bundles with checksums (``scripts/package.py``; INTERFACE "Build and packaging"; bookkeeping only).

Bundles written to ``--out`` (default ``deliverables/``):

* ``IL-Memoir-reproduction.zip`` (enclosing folder ``IL-Memoir-Starter/``): code,
  tests and fixtures, configurations with the seed manifest, dependency
  records (``requirements-*.txt``, ``runs/environment_laptop.json``), the
  contract and method documents, the theory sources and PDF, the audit logs,
  compact run records (manifests, seed manifests, outcome journals, job,
  status and result records, metric logs, evaluations, latents, and the small
  output files whose SHA-256 the audit verifies: f-PG policy checkpoints, FB
  snapshots and probe arrays; never a resume checkpoint),
  data manifests with the demonstrations, generated results, report and slides
  sources and PDFs if they exist, and ``REPRODUCE.md`` with the minimal
  clean-environment smoke command;
* ``IL-Memoir-audit-review.zip``: the same records for an independent reviewer
  (Prompt 7), without data arrays; the five source PDFs are attached
  separately and listed with their SHA-256 in ``SOURCES.sha256``;
* ``<doc>-overleaf.zip`` for each existing LaTeX document (theory, report,
  slides): main ``.tex`` at the archive root, every dependency at its relative
  path (``documents.latex_bundle``);
* ``IL-Memoir-large-artifacts.zip`` only with ``--include-large``: replay
  buffers and held-out references;
* ``CHECKSUMS.sha256`` (SHA-256 of every bundle) and ``MANIFEST.json`` (every file
  of every bundle with its SHA-256, the excluded files and why).

Never bundled: ``.venv`` and any virtual environment, ``.git``, caches
(``__pycache__``, ``.pytest_cache``), ``deliverables/`` itself, smoke runs, resume
checkpoints, LaTeX intermediate files, and anything that looks like a secret
(``.env``, keys, ``.netrc``, credentials, files containing a private-key block).
``CHECKSUMS.sha256`` at the project root is regenerated for the files of the
reproduction bundle and the source PDFs.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .documents import TARGETS, BuildError, find_engine, latex_bundle, verify_bundle
from .logging import PROJECT_ROOT, replace_file, sha256_file, utc_now, write_json_atomic, write_text_atomic

ZIP_DATE = (2026, 1, 1, 0, 0, 0)

ALWAYS_EXCLUDED_DIRS = {".venv", "venv", "env", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints",
                        "deliverables", "node_modules", ".idea", ".vscode"}
SECRET_PATTERNS = [".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*", ".netrc", ".pypirc",
                   "*credential*", "*secret*", "*token*", "*.kdbx"]
LATEX_INTERMEDIATE = ["*.aux", "*.bbl", "*.blg", "*.fdb_latexmk", "*.fls", "*.out", "*.toc", "*.nav", "*.snm", "*.vrb",
                      "*.synctex.gz", "*.lof", "*.lot", "*.run.xml", "*.bcf"]
PRIVATE_KEY = re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")
CONTRACT_DOCS = ["README.md", "CLAUDE.md", "PROJECT_SPEC.md", "INTERFACE.md", "WORKFLOW.md", "PROMPTS.md", "STATE.md",
                 "METHOD_MAP.md", "NOTATION.md", "THEORY_INDEX.md", "SOURCE_AUDIT.md", "RUN_PLAN.md"]
ROOT_FILES = ["pytest.ini", "requirements-base.txt", "requirements-laptop.txt", ".gitignore", "CHECKSUMS.sha256"]
RUN_RECORD_GLOBS = ["runs/environment_laptop.json", "runs/benchmark.json", "runs/PILOT.md",
                    "runs/suites/*/*.json", "runs/suites/*/*.jsonl", "runs/suites/*/hardware/*.json",
                    "runs/jobs/*/job.json", "runs/jobs/*/status.json", "runs/jobs/*/result.json", "runs/jobs/*/log.jsonl",
                    "runs/jobs/*/outputs/*.json", "runs/jobs/*/outputs/*.jsonl",
                    # recorded outputs the audit checks by hash (small: policies ~2-25 KB, FB snapshots ~75 KB)
                    "runs/jobs/*/outputs/models/*.pt", "runs/jobs/*/outputs/policies/*.pt", "runs/jobs/*/outputs/probe.npz",
                    "runs/invalidated/*/invalidated.json", "runs/invalidated/*/job.json", "runs/invalidated/*/status.json",
                    "runs/invalidated/*/result.json", "runs/pilot_results/**/*"]
LARGE_GLOBS = ["data/*/replay/**/*.npz", "data/*/reference/**/*.npz"]


def _match_any(name: str, patterns: Sequence[str]) -> bool:
    low = name.lower()
    return any(fnmatch.fnmatch(low, p.lower()) for p in patterns)


def excluded_reason(rel: str, path: Path, root: Path = PROJECT_ROOT) -> Optional[str]:
    """Why a file is never bundled, or ``None``."""
    parts = rel.split("/")
    for i, p in enumerate(parts[:-1]):
        if p in ALWAYS_EXCLUDED_DIRS or p.startswith(".venv"):
            return f"directory {p!r} (virtual environment, VCS, cache or output)"
        if (root.joinpath(*parts[:i + 1]) / "pyvenv.cfg").exists():
            return f"directory {p!r} is a Python virtual environment"
    name = parts[-1]
    if name.endswith((".pyc", ".pyo")):
        return "compiled Python cache"
    if _match_any(name, SECRET_PATTERNS):
        return "looks like a secret or credential file"
    if _match_any(name, LATEX_INTERMEDIATE):
        return "LaTeX intermediate file"
    if parts[:2] == ["runs", "smoke"]:
        return "smoke run (test fixture; reproducible with scripts/smoke.py)"
    if "checkpoints" in parts:
        return "resume checkpoint (unnecessary after completion)"
    if name.endswith(".tmp") or ".tmp" in name:
        return "temporary file"
    try:
        if path.stat().st_size < 5_000_000 and PRIVATE_KEY.search(path.read_bytes()):
            return "contains a private-key block"
    except OSError:
        return "unreadable"
    return None


def _glob(root: Path, patterns: Iterable[str]) -> List[Path]:
    out = set()
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file():
                out.add(p)
    return sorted(out)


def collect(root: Path, patterns: Iterable[str]) -> Tuple[List[Tuple[Path, str]], Dict[str, str]]:
    """``([(path, relative name)], {excluded relative name: reason})`` for the given glob patterns."""
    keep, excluded = [], {}
    for p in _glob(root, patterns):
        rel = p.relative_to(root).as_posix()
        why = excluded_reason(rel, p, root)
        if why:
            excluded[rel] = why
        else:
            keep.append((p, rel))
    return keep, excluded


def _zip(zip_path: Path, entries: Sequence[Tuple[str, bytes]]) -> Dict[str, str]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_name(zip_path.name + ".partial")
    listing = {}
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arc, data in sorted(entries):
            info = zipfile.ZipInfo(arc, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
            listing[arc] = hashlib.sha256(data).hexdigest()
    replace_file(tmp, zip_path)
    return listing


def source_patterns() -> List[str]:
    return (["il_memoir/*.py", "scripts/*.py", "tests/*.py", "tests/fixtures/*", "configs/*.yaml",
             "theory/*.tex", "theory/*.bib", "theory/theory.pdf", "theory/checks/*.py", "theory/checks/*.txt",
             "audit/**/*.py", "audit/**/*.txt", "audit/**/*.json", "audit/**/*.md", "audit/**/*.png",
             "report/**/*", "slides/**/*", "results/**/*"]
            + CONTRACT_DOCS + ROOT_FILES)


def reproduce_md(engine: Optional[Dict[str, str]]) -> str:
    return "\n".join([
        "# Reproducing this bundle", "",
        f"Bundle written by `scripts/package.py` on {utc_now()}. In Windows PowerShell, from the folder that contains "
        "this ZIP, extract it and run (Python 3.11 and uv, as in WORKFLOW.md section 3):", "",
        "```powershell",
        'Expand-Archive IL-Memoir-reproduction.zip -DestinationPath "$env:TEMP\\il-memoir-check" -Force',
        'Set-Location "$env:TEMP\\il-memoir-check\\IL-Memoir-Starter"',
        "uv venv --python 3.11 .venv",
        "uv pip install --python .venv\\Scripts\\python.exe torch --torch-backend=auto",
        "uv pip install --python .venv\\Scripts\\python.exe -r requirements-base.txt",
        ".\\.venv\\Scripts\\python.exe -m pytest -q",
        ".\\.venv\\Scripts\\python.exe scripts\\smoke.py --device cpu",
        "```", "",
        "The last two commands are the minimal clean-environment check: the test suite and the smoke run of the complete "
        "data-to-results path in both environments (a few seconds to a minute; test fixtures only). "
        "`requirements-laptop.txt` records the exact versions of the original laptop environment. "
        "The full experiment commands are in README.md. Run records include the policy checkpoints, FB snapshots and "
        "probe arrays that `scripts/audit_results.py` verifies; replay buffers and held-out references are in "
        "`IL-Memoir-large-artifacts.zip` when it was built, and `scripts/prepare_data.py` regenerates them "
        "deterministically and checks them against `data/<id>/manifest.json`.", ""])


def regenerate_root_checksums(root: Path, files: Sequence[Tuple[Path, str]]) -> Path:
    """``CHECKSUMS.sha256`` at the project root (``sha256sum`` format, raw bytes) for the bundled files and sources."""
    rels = {rel: p for p, rel in files if rel != "CHECKSUMS.sha256"}
    for p in sorted((root / "sources").glob("*.pdf")):
        rels[p.relative_to(root).as_posix()] = p
    lines = [f"{sha256_file(p)}  {rel}" for rel, p in sorted(rels.items())]
    out = root / "CHECKSUMS.sha256"
    write_text_atomic(out, "\n".join(lines) + "\n")
    return out


def package(out_dir: Path, root: Path = PROJECT_ROOT, include_large: bool = False, verify: bool = False,
            update_root_checksums: bool = True, log: Callable[[str], None] = print) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    if out_dir.resolve() == root.resolve():
        raise BuildError("--out must not be the project root")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"schema": "il_memoir/deliverables-v1", "created_utc": utc_now(), "bundles": {}}
    engine = find_engine()
    # ---------------------------------------------------------------- reproduction bundle
    src, excl_src = collect(root, source_patterns())
    runs, excl_runs = collect(root, RUN_RECORD_GLOBS)
    data_meta, excl_data = collect(root, ["data/*/manifest.json", "data/*/demos/**/*.npz"])
    if update_root_checksums:
        regenerate_root_checksums(root, src)
        src, excl_src = collect(root, source_patterns())
    files = src + [f for f in runs + data_meta if f not in src]
    entries = [(f"IL-Memoir-Starter/{rel}", p.read_bytes()) for p, rel in files]
    entries.append(("IL-Memoir-Starter/REPRODUCE.md", reproduce_md(engine).encode("utf-8")))
    listing = _zip(out_dir / "IL-Memoir-reproduction.zip", entries)
    manifest["bundles"]["IL-Memoir-reproduction.zip"] = {"files": listing, "excluded": {**excl_src, **excl_runs, **excl_data}}
    log(f"reproduction bundle: {len(listing)} files")
    # ---------------------------------------------------------------- audit-review bundle
    audit_files = [f for f in files if not f[1].startswith("data/") or f[1].endswith("manifest.json")]
    src_sums = "\n".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}" for p in sorted((root / "sources").glob("*.pdf")))
    note = ("# Audit-review bundle\n\nFor the independent review (PROMPTS.md, Prompt 7). Attach the five source PDFs "
            "separately; their SHA-256 are in SOURCES.sha256. Contents: contract and method documents, theory, report and "
            "slides if written, code and tests, configurations and seed manifest, run manifests and records, results and "
            "provenance, audit and build logs.\n")
    entries = [(rel, p.read_bytes()) for p, rel in audit_files]
    entries += [("SOURCES.sha256", (src_sums + "\n").encode("utf-8")), ("AUDIT_BUNDLE_README.md", note.encode("utf-8"))]
    listing = _zip(out_dir / "IL-Memoir-audit-review.zip", entries)
    manifest["bundles"]["IL-Memoir-audit-review.zip"] = {"files": listing}
    log(f"audit-review bundle: {len(listing)} files")
    # ---------------------------------------------------------------- Overleaf bundles
    for name, rel in TARGETS.items():
        main = root / rel
        if not main.exists():
            manifest["bundles"][f"{name}-overleaf.zip"] = {"skipped": f"{rel.as_posix()} does not exist yet"}
            log(f"{name}: no {rel.as_posix()} yet; no Overleaf bundle")
            continue
        rec = latex_bundle(main, out_dir / f"{name}-overleaf.zip")
        entry = {"files": rec["files"], "main": rec["main"]}
        if verify:
            entry["verification"] = verify_bundle(out_dir / f"{name}-overleaf.zip", engine)
            log(f"{name}: Overleaf bundle test-compiled: {entry['verification']}")
        manifest["bundles"][f"{name}-overleaf.zip"] = entry
        log(f"{name}: Overleaf bundle with {rec['n_files']} files (main {rec['main']} at the root)")
    # ---------------------------------------------------------------- large artifacts
    if include_large:
        large, excl_large = collect(root, LARGE_GLOBS)
        listing = _zip(out_dir / "IL-Memoir-large-artifacts.zip", [(f"IL-Memoir-Starter/{rel}", p.read_bytes()) for p, rel in large])
        manifest["bundles"]["IL-Memoir-large-artifacts.zip"] = {"files": listing, "excluded": excl_large}
        log(f"large artifacts: {len(listing)} files")
    # ---------------------------------------------------------------- checksums and verification
    sums = []
    for z in sorted(out_dir.glob("*.zip")):
        if z.name in manifest["bundles"] and "files" in manifest["bundles"][z.name]:
            h = sha256_file(z)
            manifest["bundles"][z.name]["sha256"] = h
            manifest["bundles"][z.name]["bytes"] = z.stat().st_size
            sums.append(f"{h}  {z.name}")
    write_text_atomic(out_dir / "CHECKSUMS.sha256", "\n".join(sums) + "\n")
    if verify:
        manifest["reproduction_check"] = verify_reproduction(out_dir / "IL-Memoir-reproduction.zip", log)
        manifest["audit_check"] = verify_audit_bundle(out_dir / "IL-Memoir-audit-review.zip", log)
    write_json_atomic(out_dir / "MANIFEST.json", manifest)
    return manifest


def verification_failures(manifest: Dict[str, Any]) -> List[str]:
    """Every failed check recorded by ``package(..., verify=True)``; empty when all passed.

    Counted as failures: an Overleaf bundle whose test compilation failed (a missing LaTeX engine is reported,
    not counted), a missing or failed smoke check of the reproduction bundle, and a missing or failed audit of
    any suite kind inside the audit-review bundle.  An audit that was skipped because the bundle holds no suite
    record yet is not a failure.  ``scripts/package.py --verify`` exits with status 1 if this list is not empty.
    """
    out: List[str] = []
    for name, bundle in sorted(manifest.get("bundles", {}).items()):
        v = bundle.get("verification")
        if v is None or v.get("compiled") or "no LaTeX engine" in str(v.get("message", "")):
            continue
        why = "; ".join(v.get("failure_reasons") or []) or v.get("message") or "not compiled"
        out.append(f"{name}: the test compilation failed ({why}; log {v.get('log')})")
    rc = manifest.get("reproduction_check")
    if not isinstance(rc, dict):
        out.append("IL-Memoir-reproduction.zip: the smoke check did not run")
    elif not rc.get("passed"):
        out.append(f"IL-Memoir-reproduction.zip: the smoke check failed (exit status {rc.get('exit_code')})")
    ac = manifest.get("audit_check")
    if not isinstance(ac, dict) or not ac:
        out.append("IL-Memoir-audit-review.zip: the audit did not run")
    elif "skipped" not in ac:
        for kind, r in sorted(ac.items()):
            if not (isinstance(r, dict) and r.get("passed")):
                code = r.get("exit_code") if isinstance(r, dict) else None
                out.append(f"IL-Memoir-audit-review.zip: the audit of the {kind} suites failed (exit status {code})")
    return out


def verify_audit_bundle(zip_path: Path, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Extract the audit-review bundle and run ``scripts/audit_results.py`` inside it, once per suite kind it holds."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        proj = Path(tmp)
        kinds = sorted({json.loads(p.read_text(encoding="utf-8")).get("kind")
                        for p in (proj / "runs" / "suites").glob("*/manifest.json")} - {None})
        if not kinds:
            log("audit-review bundle: no suite records yet; audit not run")
            return {"skipped": "no suite records in the bundle"}
        out: Dict[str, Any] = {}
        for kind in kinds:
            cmd = [sys.executable, "scripts/audit_results.py", "--runs", "runs"]
            if kind == "final" and (proj / "results" / "provenance.json").exists():
                cmd += ["--results", "results"]
            else:
                cmd += ["--runs-only"]
            if kind != "final":
                cmd += ["--include-kind", kind]
            proc = subprocess.run(cmd, cwd=str(proj), capture_output=True, text=True, errors="replace", timeout=1800)
            out[kind] = {"command": " ".join(cmd[1:]), "exit_code": proc.returncode, "passed": proc.returncode == 0,
                         "tail": proc.stdout.splitlines()[-4:]}
            log(f"audit inside an extracted copy of the audit-review bundle ({kind} suites): "
                f"{'pass' if proc.returncode == 0 else 'FAIL'}")
        return out


def verify_reproduction(zip_path: Path, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Extract the reproduction bundle and run its smoke check with the current interpreter (not a clean venv)."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        proj = Path(tmp) / "IL-Memoir-Starter"
        cmd = [sys.executable, "scripts/smoke.py", "--device", "cpu"]
        proc = subprocess.run(cmd, cwd=str(proj), capture_output=True, text=True, timeout=1800)
        ok = proc.returncode == 0 and "ALL SMOKE CHECKS PASS" in proc.stdout
        log(f"reproduction bundle smoke check in a temporary copy: {'pass' if ok else 'FAIL'}")
        return {"command": " ".join(cmd[1:]), "interpreter": sys.executable, "exit_code": proc.returncode,
                "passed": bool(ok), "tail": proc.stdout.splitlines()[-5:]}
