"""LaTeX build and self-contained Overleaf bundles (``scripts/build.py``, ``scripts/package.py``; bookkeeping only).

Build.  One supported engine is used for every target of a call, chosen in
this order: ``latexmk`` driving ``pdflatex``, ``pdflatex`` (three passes, with
``bibtex`` when the document has a bibliography), ``tectonic``.  The engine
runs in place in a temporary *replica* of the document: the document's folder
without build products, plus every dependency outside it (``../results/...``)
at the same relative position.  No output-directory option is used, so
LaTeX, latexmk and BibTeX see the layout of an ordinary build on every
platform, and latexmk runs with ``-norc`` (no system, user or project rc file
changes the build).  The source folder stays clean.  The PDF, the complete
engine output with the TeX and BibTeX logs (``build.log``) and ``build.json``
(engine and version, every command with its exit status, pages, errors,
undefined references and citations, overfull boxes, the reasons for a failure,
SHA-256 of the PDF and of every source file) are written next to the main
``.tex`` (or to ``out_root``).  A build counts as successful only if the engine
exits with status 0, the PDF exists and the log has no error and no undefined
reference or citation; a failed build never replaces an existing PDF.  If no
engine is available the call fails with a message, and nothing claims a
compilation.

Overleaf bundles.  :func:`latex_bundle` follows ``\\input``, ``\\include``,
``\\includegraphics``, ``\\bibliography``, ``\\addbibresource``,
``\\lstinputlisting``, ``\\includepdf`` and local ``.sty``/``.cls`` files from the
main file, puts the main ``.tex`` at the archive root (no enclosing directory),
maps a dependency outside the main folder (``../results/figures/x.png``) to the
same path without the leading ``../`` (``results/figures/x.png``) and rewrites
that argument in the copied sources.  A missing dependency stops the bundle.
:func:`verify_bundle` test-compiles a bundle in a temporary extraction and
keeps its ``build.log`` and ``build.json`` in a lasting folder (by default
``<bundle>-verification/`` next to the ZIP); its result carries the engine's
exit status and the reasons for a failure.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .logging import PROJECT_ROOT, replace_file, sha256_file, utc_now, write_json_atomic

TARGETS = {"report": Path("report") / "report.tex", "slides": Path("slides") / "slides.tex",
           "theory": Path("theory") / "theory.tex"}
ALL = ("report", "slides")
GRAPHIC_EXT = ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps")
ZIP_DATE = (2026, 1, 1, 0, 0, 0)        # fixed timestamp: identical inputs give identical archives


class BuildError(RuntimeError):
    pass


# =============================================================================
# engines
# =============================================================================
def _probe(cmd: List[str]) -> Tuple[int, str]:
    """Exit status and first output line of ``cmd`` (``(-1, message)`` if it cannot be started)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, f"{type(exc).__name__}: {exc}"
    text = (out.stdout or out.stderr or "").strip()
    return out.returncode, text.splitlines()[0] if text else ""


def _version(cmd: List[str]) -> str:
    return _probe(cmd)[1]


def find_engine(prefer: Optional[str] = None, log=None) -> Optional[Dict[str, str]]:
    """The first engine that runs (or ``prefer``): latexmk+pdflatex, pdflatex, tectonic.

    An engine is used only if ``<engine> --version`` exits with status 0: on
    Windows, MiKTeX installs ``latexmk.exe`` even without the Perl it needs, and
    such a latexmk is skipped in favour of pdflatex (with a message) instead
    of failing every build.
    """
    order = ["latexmk", "pdflatex", "tectonic"]
    if prefer:
        if prefer not in order:
            raise BuildError(f"unsupported engine {prefer!r}; supported: {order}")
        order = [prefer]
    for name in order:
        path = shutil.which(name)
        if not path:
            continue
        if name == "latexmk" and not shutil.which("pdflatex"):
            continue
        code, version = _probe([path, "--version"])
        if code != 0:
            msg = f"{name} at {path} is on PATH but does not run ({version or f'exit {code}'})"
            if name == "latexmk":
                msg += "; on Windows MiKTeX's latexmk needs Perl"
            if prefer:
                raise BuildError(msg)
            if log is not None:
                log(msg + "; trying the next engine")
            continue
        rec = {"name": name, "path": path, "version": version}
        if name == "latexmk":
            rec["pdflatex_version"] = _version([shutil.which("pdflatex"), "--version"])
        return rec
    return None


def _has_bibliography(tex: str) -> bool:
    return bool(re.search(r"\\(bibliography|addbibresource)\s*\{", tex))


def _run(cmd: List[str], cwd: Path, log_parts: List[str], timeout: int, env: Optional[Dict[str, str]] = None,
         commands: Optional[List[Dict[str, Any]]] = None) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, errors="replace", timeout=timeout, env=env)
    log_parts.append(f"$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}\n[exit {proc.returncode}]\n")
    if commands is not None:
        commands.append({"command": " ".join(cmd), "exit_code": proc.returncode})
    return proc.returncode


# build products never copied into the replica (a stale .fdb_latexmk or PDF could make latexmk skip the build)
_BUILD_SUFFIXES = (".aux", ".log", ".fls", ".fdb_latexmk", ".out", ".toc", ".bbl", ".blg", ".nav", ".snm", ".vrb",
                   ".lof", ".lot", ".bcf", ".xdv", ".dvi", ".idx", ".ilg", ".ind", ".bak")
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "_minted", ".ipynb_checkpoints", "node_modules"}


def _is_build_product(path: Path, stem: str) -> bool:
    name = path.name.lower()
    return (path.suffix.lower() in _BUILD_SUFFIXES or name.endswith((".synctex.gz", ".run.xml", "_build.log"))
            or name in ("build.log", "build.json", f"{stem.lower()}.pdf"))


def replicate_document(main_tex: Path, dest: Path) -> Path:
    """Copy the document folder (without build products) and its dependencies outside it into ``dest``.

    Every file keeps its position relative to the main file, so ``../results/figures/x`` still resolves; returns
    the path of the copied main file.  Dependencies come from :func:`latex_dependencies`.
    """
    main_tex = Path(main_tex).resolve()
    src = main_tex.parent
    deps, _, _ = latex_dependencies(main_tex)
    outside: List[Path] = []
    depth = 0
    for dep in deps:
        try:
            rel = Path(os.path.relpath(dep, src))
        except ValueError as exc:                          # another drive (Windows)
            raise BuildError(f"{dep}: a LaTeX dependency on another drive than {main_tex} is not supported") from exc
        ups = 0
        for part in rel.parts:
            if part != "..":
                break
            ups += 1
        if ups:
            outside.append(dep)
            depth = max(depth, ups)
    anchor = src
    for _ in range(depth):
        anchor = anchor.parent
    main_dir = Path(dest) / src.relative_to(anchor)
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if not path.is_file() or any(part in _SKIP_DIRS for part in rel.parts[:-1]) or _is_build_product(path, main_tex.stem):
            continue
        target = main_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for dep in outside:
        target = Path(dest) / dep.relative_to(anchor)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dep, target)
    return main_dir / main_tex.name


def _tail(text: str, n: int = 40) -> str:
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join(lines[-n:])


def unwrap_log(text: str, width: int = 79) -> str:
    """Undo TeX's hard line breaks: a log line of exactly ``max_print_line`` (79) characters continues on the next.

    Without this, a long output path splits ``Output written on <path> (N pages`` or a warning with a long label
    over two lines, and the page count or the label is missed.
    """
    out, buf = [], ""
    for line in text.splitlines():
        buf += line
        if len(line) == width:
            continue
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return "\n".join(out)


def parse_log(text: str) -> Dict[str, Any]:
    """Errors, undefined references and citations, rerun requests, box warnings and pages from a LaTeX log."""
    text = unwrap_log(text)
    errors = [l for l in text.splitlines() if l.startswith("!") or re.match(r"^\S+\.tex:\d+: ", l)]
    undef_ref = sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", text)))
    undef_cit = sorted(set(re.findall(r"Citation `([^']+)' on page \d+ undefined", text)))
    pages = re.findall(r"Output written on .*?\((\d+) pages?", text)
    return {
        "errors": errors[:50],
        "n_errors": len(errors),
        "undefined_references": undef_ref,
        "undefined_citations": undef_cit,
        "there_were_undefined": "There were undefined references" in text or "There were undefined citations" in text,
        "rerun_requested": bool(re.search(r"Rerun to get (cross-references|citations) right", text)),
        "overfull_boxes": len(re.findall(r"^Overfull \\[hv]box", text, flags=re.M)),
        "underfull_boxes": len(re.findall(r"^Underfull \\[hv]box", text, flags=re.M)),
        "pages": int(pages[-1]) if pages else None,
    }


def compile_document(main_tex: Path, engine: Dict[str, str], out_pdf: Path, out_dir: Path, timeout: int = 600) -> Dict[str, Any]:
    """Compile ``main_tex`` with ``engine`` in a temporary replica; write the PDF, ``build.log`` and ``build.json``.

    The record (also in ``build.json``) has the engine's exit status, every command run with its exit status, the
    parsed log and ``failure_reasons`` (empty on success); a failed build also carries ``log_tail``, the last lines
    of the engine output.
    """
    main_tex = Path(main_tex).resolve()
    src_dir = main_tex.parent
    stem = main_tex.stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_parts: List[str] = []
    commands: List[Dict[str, Any]] = []
    tmp = Path(tempfile.mkdtemp(prefix=f"build_{stem}_"))
    tex_env = dict(os.environ, max_print_line="10000")    # TeX Live: no hard log wrapping (unwrap_log covers the rest)
    try:
        work_main = replicate_document(main_tex, tmp)
        work = work_main.parent
        tex = work_main.read_text(encoding="utf-8", errors="replace")
        common = ["-interaction=nonstopmode", "-halt-on-error", "-file-line-error"]
        if engine["name"] == "latexmk":
            code = _run([engine["path"], "-norc", "-pdf", *common, work_main.name], work, log_parts, timeout,
                        env=tex_env, commands=commands)
        elif engine["name"] == "pdflatex":
            code = 0
            for k in range(3):
                code = _run([engine["path"], *common, work_main.name], work, log_parts, timeout, env=tex_env,
                            commands=commands)
                if code != 0:
                    break
                if k == 0 and _has_bibliography(tex):
                    bibtex = shutil.which("bibtex")
                    if not bibtex:
                        log_parts.append("bibtex not found on PATH: citations stay undefined\n")
                        continue
                    bib_code = _run([bibtex, stem], work, log_parts, timeout, env=tex_env, commands=commands)
                    if bib_code >= 2:                     # 1 = warnings only; 2 = errors; 3 = fatal
                        code = bib_code
                        break
        else:
            code = _run([engine["path"], "--keep-logs", work_main.name], work, log_parts, timeout, env=tex_env,
                        commands=commands)
        tex_log = work / f"{stem}.log"
        log_text = tex_log.read_text(encoding="utf-8", errors="replace") if tex_log.exists() else ""
        blg = work / f"{stem}.blg"
        parsed = parse_log(log_text)
        pdf = work / f"{stem}.pdf"
        reasons: List[str] = []
        if code != 0:
            reasons.append(f"engine exit status {code}")
        if not pdf.exists():
            reasons.append("no PDF was produced")
        if not tex_log.exists():
            reasons.append(f"no TeX log {tex_log.name} was written")
        if parsed["n_errors"]:
            reasons.append(f"{parsed['n_errors']} LaTeX error(s)")
        if parsed["undefined_references"]:
            reasons.append("undefined references: " + ", ".join(parsed["undefined_references"][:10]))
        if parsed["undefined_citations"]:
            reasons.append("undefined citations: " + ", ".join(parsed["undefined_citations"][:10]))
        if parsed["there_were_undefined"] and not (parsed["undefined_references"] or parsed["undefined_citations"]):
            reasons.append("the TeX log reports undefined references or citations")
        ok = not reasons
        engine_output = "".join(log_parts)
        full_log = engine_output + "\n===== " + tex_log.name + " =====\n" + log_text
        if blg.exists():
            full_log += "\n===== " + blg.name + " =====\n" + blg.read_text(encoding="utf-8", errors="replace")
        (out_dir / "build.log").write_text(full_log, encoding="utf-8")
        rec: Dict[str, Any] = {"utc": utc_now(), "main": main_tex.name, "source_dir": str(src_dir), "engine": engine,
                               "build_dir": "temporary replica of the document tree (removed after the build)",
                               "commands": commands, "exit_code": code, "success": bool(ok), "failure_reasons": reasons,
                               "log": str(out_dir / "build.log"), **parsed}
        if ok:
            out_pdf = Path(out_pdf)
            out_pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pdf, out_pdf)
            rec["pdf"] = out_pdf.name
            rec["pdf_sha256"] = sha256_file(out_pdf)
        else:
            rec["pdf"] = None
            rec["message"] = ("compilation failed (" + "; ".join(reasons) + "); no PDF was written (see build.log)")
            rec["log_tail"] = _tail(engine_output)
        rec["sources"] = {str(p.relative_to(src_dir)).replace(os.sep, "/"): sha256_file(p) for p in sorted(src_dir.rglob("*"))
                          if p.is_file() and p.suffix in (".tex", ".bib", ".sty", ".cls")}
        write_json_atomic(out_dir / "build.json", rec)
        return rec
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build_targets(targets: Sequence[str], root: Path = PROJECT_ROOT, engine_name: Optional[str] = None,
                  out_root: Optional[Path] = None, log=print) -> Tuple[int, Dict[str, Any]]:
    """Build the targets with one engine.  Exit status 0 iff every requested target compiled cleanly."""
    engine = find_engine(engine_name, log=log)
    results: Dict[str, Any] = {"engine": engine, "targets": {}}
    if engine is None:
        log("No supported LaTeX engine (latexmk with pdflatex, pdflatex or tectonic) was found on PATH; nothing was "
            "compiled. Run `python scripts/package.py --out deliverables` and compile the Overleaf ZIP of each document "
            "(main .tex at the archive root) on Overleaf, then download its PDF and log into report/ or slides/.")
        return 2, results
    log(f"engine: {engine['name']} ({engine['version']})")
    status = 0
    for name in targets:
        main = root / TARGETS[name]
        if not main.exists():
            log(f"{name}: {TARGETS[name].as_posix()} does not exist (the report is written in Prompt 5, the slides in "
                "Prompt 6); not built")
            results["targets"][name] = {"success": False, "message": "source missing"}
            status = 1
            continue
        out_dir = (out_root / name) if out_root is not None else main.parent
        out_pdf = out_dir / f"{main.stem}.pdf"
        log(f"{name}: compiling {TARGETS[name].as_posix()} ...")
        try:
            rec = compile_document(main, engine, out_pdf, out_dir)
        except subprocess.TimeoutExpired:
            rec = {"success": False, "message": "the engine timed out"}
        results["targets"][name] = rec
        if rec.get("success"):
            log(f"{name}: compiled, {rec.get('pages')} pages, {rec.get('overfull_boxes')} overfull boxes -> {out_pdf}")
        else:
            status = 1
            log(f"{name}: FAILED ({rec.get('message')}); errors: {rec.get('errors', [])[:3]}; undefined: "
                f"{rec.get('undefined_references', [])[:5]} {rec.get('undefined_citations', [])[:5]}")
    return status, results


# =============================================================================
# Overleaf bundles
# =============================================================================
_PATTERNS = [
    ("tex", re.compile(r"\\(?:input|include|subfile)\s*\{([^}]+)\}")),
    ("graphic", re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")),
    ("bib", re.compile(r"\\bibliography\s*\{([^}]+)\}")),
    ("res", re.compile(r"\\addbibresource\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")),
    ("listing", re.compile(r"\\lstinputlisting\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")),
    ("pdf", re.compile(r"\\includepdf\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")),
    ("sty", re.compile(r"\\(?:usepackage|RequirePackage)\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")),
    ("cls", re.compile(r"\\documentclass\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")),
]
_GRAPHICSPATH = re.compile(r"\\graphicspath\s*\{((?:\{[^}]*\})+)\}")


def _strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _bundle_name(rel: str) -> str:
    """``../results/figures/x.png`` -> ``results/figures/x.png``; paths inside the main folder are unchanged."""
    parts = [p for p in PurePosixPath(rel.replace("\\", "/")).parts if p not in (".",)]
    while parts and parts[0] == "..":
        parts = parts[1:]
    return "/".join(parts)


def _resolve(base: Path, arg: str, kind: str, graphic_dirs: Sequence[str]) -> Optional[Path]:
    arg = arg.strip()
    cands: List[Path] = []
    if kind == "tex":
        cands = [base / arg, base / (arg + ".tex")]
    elif kind in ("graphic", "pdf"):
        for d in [""] + list(graphic_dirs):
            for ext in GRAPHIC_EXT:
                cands.append(base / d / (arg + ext))
    elif kind == "bib":
        cands = [base / (arg if arg.endswith(".bib") else arg + ".bib")]
    elif kind in ("res", "listing"):
        cands = [base / arg]
    elif kind == "sty":
        cands = [base / (arg + ".sty")]
    elif kind == "cls":
        cands = [base / (arg + ".cls")]
    for c in cands:
        if c.is_file():
            return c.resolve()
    return None


def latex_dependencies(main_tex: Path) -> Tuple[Dict[Path, str], Dict[Path, List[Tuple[str, str]]], List[str]]:
    """Files needed by ``main_tex``: ``{absolute path: bundle path}``, rewrites per ``.tex`` file, missing references."""
    main_tex = Path(main_tex).resolve()
    base = main_tex.parent
    files: Dict[Path, str] = {main_tex: main_tex.name}
    rewrites: Dict[Path, List[Tuple[str, str]]] = {}
    missing: List[str] = []
    stack = [main_tex]
    seen = set()
    while stack:
        tex_file = stack.pop()
        if tex_file in seen:
            continue
        seen.add(tex_file)
        text = _strip_comments(tex_file.read_text(encoding="utf-8", errors="replace"))
        gdirs: List[str] = []
        for m in _GRAPHICSPATH.finditer(text):
            dirs = re.findall(r"\{([^}]*)\}", m.group(1))
            gdirs += dirs
            moved = []
            for d in dirs:                                # parent-relative folders move inside the bundle root
                nd = d.replace("\\", "/")
                while nd.startswith("../"):
                    nd = nd[3:]
                moved.append(nd)
            if moved != dirs:
                rewrites.setdefault(tex_file, []).append(
                    (m.group(0), "\\graphicspath{" + "".join("{" + nd + "}" for nd in moved) + "}"))
        for kind, pat in _PATTERNS:
            for m in pat.finditer(text):
                for arg in m.group(1).split(",") if kind in ("bib", "sty") else [m.group(1)]:
                    arg = arg.strip()
                    if not arg:
                        continue
                    path = _resolve(base, arg, kind, gdirs)
                    if path is None:
                        if kind not in ("sty", "cls"):          # system packages and classes are not local files
                            missing.append(f"{tex_file.name}: {kind} {arg!r}")
                        continue
                    rel = os.path.relpath(path, base).replace(os.sep, "/")
                    target = _bundle_name(rel)
                    files[path] = target
                    if rel != target:
                        new_arg = arg.replace("\\", "/")
                        while new_arg.startswith("../"):
                            new_arg = new_arg[3:]
                        rewrites.setdefault(tex_file, []).append((arg, new_arg))
                    if path.suffix == ".tex" or kind == "tex":
                        stack.append(path)
    return files, rewrites, missing


def _zip_write(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    info = zipfile.ZipInfo(arcname, date_time=ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data)


def latex_bundle(main_tex: Path, zip_path: Path, extra: Sequence[Path] = ()) -> Dict[str, Any]:
    """Self-contained Overleaf ZIP: the main ``.tex`` at the archive root, every dependency at its relative path."""
    files, rewrites, missing = latex_dependencies(main_tex)
    if missing:
        raise BuildError(f"{main_tex}: missing LaTeX dependencies: {missing}")
    for p in extra:
        files[Path(p).resolve()] = Path(p).name
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    listing = {}
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        for path, arc in sorted(files.items(), key=lambda kv: kv[1]):
            data = path.read_bytes()
            for old, new in rewrites.get(path, []):
                data = data.replace(old.encode("utf-8"), new.encode("utf-8"))
            _zip_write(zf, arc, data)
            listing[arc] = hashlib.sha256(data).hexdigest()
        manifest = {"main": Path(main_tex).name, "engine_hint": "pdfLaTeX", "files": listing,
                    "rewritten": {files[p]: r for p, r in rewrites.items()},
                    "note": "Upload this ZIP to Overleaf (New project > Upload project), select the main file and the "
                            "pdfLaTeX compiler, compile, then download the PDF and the log."}
        _zip_write(zf, "BUNDLE_MANIFEST.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    replace_file(tmp, zip_path)
    return {"zip": zip_path, "sha256": sha256_file(zip_path), "main": Path(main_tex).name, "n_files": len(listing),
            "files": listing}


def verify_bundle(zip_path: Path, engine: Optional[Dict[str, str]] = None,
                  log_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Extract a bundle into a temporary directory and compile its main file there (a check of the bundle only).

    ``build.log`` and ``build.json`` of the test compilation are kept in ``log_dir`` (default
    ``<bundle stem>-verification/`` next to the ZIP), and the result carries the engine's exit status and the
    reasons for a failure, so a failed verification can be diagnosed afterwards.
    """
    zip_path = Path(zip_path)
    engine = engine or find_engine()
    if engine is None:
        return {"compiled": False, "message": "no LaTeX engine available; the bundle was not test-compiled"}
    log_dir = Path(log_dir) if log_dir is not None else zip_path.with_name(zip_path.stem + "-verification")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
            main = json.loads(zf.read("BUNDLE_MANIFEST.json"))["main"]
        rec = compile_document(Path(tmp) / main, engine, Path(tmp) / "_pdf" / (Path(main).stem + ".pdf"), log_dir)
    out = {"compiled": bool(rec.get("success")), "exit_code": rec.get("exit_code"), "pages": rec.get("pages"),
           "engine": engine["name"], "failure_reasons": rec.get("failure_reasons", []),
           "errors": rec.get("errors", [])[:5], "undefined_references": rec.get("undefined_references", []),
           "undefined_citations": rec.get("undefined_citations", []), "log": str(log_dir / "build.log")}
    if not out["compiled"]:
        out["log_tail"] = rec.get("log_tail", "")
    return out
