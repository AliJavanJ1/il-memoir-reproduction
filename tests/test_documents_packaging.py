"""Document build and packaging (INTERFACE "Build and packaging", Prompt 3C).

Overleaf bundles keep the main file at the archive root with every dependency
at a valid relative path; the build never claims a compilation it did not do;
the deliverables exclude virtual environments, caches, secrets, smoke runs and
resume checkpoints, and their checksums verify.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from il_memoir import documents
from il_memoir.documents import BuildError, build_targets, find_engine, latex_bundle, latex_dependencies
from il_memoir.packaging import package

ROOT = Path(__file__).resolve().parents[1]
QUIET = lambda m: None          # noqa: E731
PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
       b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa7\x35\x81\x84\x00\x00\x00\x00IEND\xaeB`\x82")


def _doc_tree(tmp_path: Path) -> Path:
    (tmp_path / "report" / "sections").mkdir(parents=True)
    (tmp_path / "results" / "figures").mkdir(parents=True)
    (tmp_path / "results" / "figures" / "fig.png").write_bytes(PNG)
    (tmp_path / "report" / "sections" / "intro.tex").write_text("Intro text, see Figure~\\ref{fig:a}.\n", encoding="utf-8")
    (tmp_path / "report" / "refs.bib").write_text("@misc{a, title={A}, author={B}, year={2026}}\n", encoding="utf-8")
    (tmp_path / "report" / "report.tex").write_text(
        "\\documentclass{article}\n\\usepackage{graphicx}\n\\begin{document}\n\\input{sections/intro}\n"
        "% \\input{commented/out}\n\\begin{figure}\\includegraphics[width=2cm]{../results/figures/fig}"
        "\\caption{x}\\label{fig:a}\\end{figure}\n\\cite{a}\n\\bibliographystyle{plain}\n\\bibliography{refs}\n\\end{document}\n",
        encoding="utf-8")
    return tmp_path / "report" / "report.tex"


def test_overleaf_bundle_puts_the_main_file_at_the_root_and_rewrites_parent_paths(tmp_path):
    main = _doc_tree(tmp_path)
    files, rewrites, missing = latex_dependencies(main)
    assert not missing
    rec = latex_bundle(main, tmp_path / "out" / "report-overleaf.zip")
    with zipfile.ZipFile(rec["zip"]) as zf:
        names = set(zf.namelist())
        assert names == {"report.tex", "sections/intro.tex", "results/figures/fig.png", "refs.bib", "BUNDLE_MANIFEST.json"}
        text = zf.read("report.tex").decode("utf-8")
        assert "{results/figures/fig}" in text and "../" not in text
        assert json.loads(zf.read("BUNDLE_MANIFEST.json"))["main"] == "report.tex"
    again = latex_bundle(main, tmp_path / "out2" / "report-overleaf.zip")
    assert again["sha256"] == rec["sha256"]            # deterministic archive


def test_a_missing_dependency_stops_the_bundle(tmp_path):
    main = tmp_path / "doc.tex"
    main.write_text("\\documentclass{article}\\begin{document}\\input{nothere}\\end{document}\n", encoding="utf-8")
    with pytest.raises(BuildError):
        latex_bundle(main, tmp_path / "x.zip")


def test_build_fails_clearly_without_an_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(documents.shutil, "which", lambda name: None)
    lines = []
    code, rec = build_targets(["theory"], ROOT, log=lines.append)
    assert code == 2 and rec["engine"] is None and "nothing was compiled" in lines[0]


def test_build_reports_missing_sources(tmp_path):
    if find_engine() is None:
        pytest.skip("no LaTeX engine on this machine")
    lines = []
    code, rec = build_targets(["report", "slides"], tmp_path, log=lines.append)
    assert code == 1 and not rec["targets"]["report"]["success"] and "does not exist" in "".join(lines)


@pytest.mark.skipif(find_engine() is None, reason="no LaTeX engine on this machine")
def test_compile_succeeds_only_without_errors_or_undefined_references(tmp_path):
    good = tmp_path / "good" / "good.tex"
    good.parent.mkdir()
    good.write_text("\\documentclass{article}\\begin{document}\\section{A}\\label{s}See \\ref{s}.\\end{document}\n", encoding="utf-8")
    eng = find_engine()
    rec = documents.compile_document(good, eng, tmp_path / "good" / "good.pdf", tmp_path / "good")
    assert rec["success"] and rec["pages"] == 1 and (tmp_path / "good" / "good.pdf").exists()
    assert (tmp_path / "good" / "build.log").exists() and json.loads((tmp_path / "good" / "build.json").read_text())["success"]
    assert not list((tmp_path / "good").glob("*.aux"))         # intermediates stay in the temporary directory
    bad = tmp_path / "bad" / "bad.tex"
    bad.parent.mkdir()
    bad.write_text("\\documentclass{article}\\begin{document}See \\ref{nowhere}.\\end{document}\n", encoding="utf-8")
    rec = documents.compile_document(bad, eng, tmp_path / "bad" / "bad.pdf", tmp_path / "bad")
    assert not rec["success"] and "nowhere" in rec["undefined_references"] and not (tmp_path / "bad" / "bad.pdf").exists()


def test_packaging_excludes_environments_caches_secrets_and_checkpoints(tmp_path):
    root = tmp_path / "proj"
    files = {
        "il_memoir/a.py": "x = 1\n", "scripts/b.py": "print(1)\n", "tests/test_x.py": "def test(): pass\n",
        "configs/c.yaml": "a: 1\n", "README.md": "# r\n", "STATE.md": "# s\n", "requirements-base.txt": "numpy\n",
        "theory/theory.tex": "\\documentclass{article}\\begin{document}\\input{part}\\end{document}\n",
        "theory/part.tex": "Text.\n", "runs/jobs/j1/job.json": "{}\n", "runs/jobs/j1/result.json": "{}\n",
        "runs/environment_laptop.json": "{}\n", "data/d1/manifest.json": "{}\n",
        # must be excluded:
        ".venv/pyvenv.cfg": "home = x\n", ".venv/Lib/site.py": "x\n", "il_memoir/__pycache__/a.cpython-311.pyc": "x",
        ".env": "TOKEN=1\n", "configs/id_rsa": "key\n",
        "scripts/notes.py": "-----BEGIN " + "RSA PRIVATE KEY-----\nabc\n",      # split: this test file itself is bundled
        "runs/smoke/20261005T000000Z/x.json": "{}\n", "runs/jobs/j1/checkpoints/resume.pt": "x",
        "theory/theory.aux": "x", "myenv/pyvenv.cfg": "x\n", "myenv/lib/thing.py": "x\n", "sources/paper.pdf": "%PDF-1.4\n",
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    out = tmp_path / "deliverables"
    man = package(out, root, include_large=False, verify=False, update_root_checksums=True, log=QUIET)
    banned = (".venv/", "__pycache__", ".env", "id_rsa", "notes.py", "runs/smoke", "checkpoints/", ".aux", "myenv/")
    for z in out.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                assert not any(b in name for b in banned), (z.name, name)
    with zipfile.ZipFile(out / "IL-Memoir-reproduction.zip") as zf:
        names = set(zf.namelist())
        assert "IL-Memoir-Starter/REPRODUCE.md" in names and "IL-Memoir-Starter/il_memoir/a.py" in names
        assert "IL-Memoir-Starter/runs/jobs/j1/result.json" in names and "IL-Memoir-Starter/CHECKSUMS.sha256" in names
        assert not any(n.endswith(".pdf") and "sources/" in n for n in names)
    with zipfile.ZipFile(out / "theory-overleaf.zip") as zf:
        assert {"theory.tex", "part.tex"} <= set(zf.namelist())
    for line in (out / "CHECKSUMS.sha256").read_text().splitlines():
        h, name = line.split("  ")
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == h
    root_sums = (root / "CHECKSUMS.sha256").read_text().splitlines()
    assert any(l.endswith("  sources/paper.pdf") for l in root_sums) and not any(".venv" in l for l in root_sums)
    excluded = man["bundles"]["IL-Memoir-reproduction.zip"]["excluded"]
    assert excluded.get("scripts/notes.py") == "contains a private-key block"
    assert man["bundles"]["report-overleaf.zip"]["skipped"].endswith("does not exist yet")


def test_overleaf_bundle_rewrites_a_parent_relative_graphicspath(tmp_path):
    main = _doc_tree(tmp_path)
    text = main.read_text(encoding="utf-8").replace(
        "\\includegraphics[width=2cm]{../results/figures/fig}",
        "\\graphicspath{{../results/figures/}}\\includegraphics[width=2cm]{fig}")
    main.write_text(text, encoding="utf-8")
    rec = latex_bundle(main, tmp_path / "out" / "report-overleaf.zip")
    with zipfile.ZipFile(rec["zip"]) as zf:
        bundled = zf.read("report.tex").decode("utf-8")
        assert "\\graphicspath{{results/figures/}}" in bundled and "../" not in bundled
        assert "results/figures/fig.png" in zf.namelist()
    if find_engine() is not None:
        log_dir = tmp_path / "verification"
        check = documents.verify_bundle(rec["zip"], log_dir=log_dir)
        if not check["compiled"]:                     # shown under "Captured stdout" when the test fails
            print(json.dumps(check, indent=1, default=str))
            print((log_dir / "build.log").read_text(encoding="utf-8", errors="replace")[-8000:])
        assert check["compiled"], f"exit status {check['exit_code']}; {check['failure_reasons']}; log {check['log']}"
        assert check["exit_code"] == 0 and (log_dir / "build.log").exists() and (log_dir / "build.json").exists()


@pytest.mark.skipif(not (shutil.which("pdflatex") and shutil.which("bibtex")), reason="needs pdflatex and bibtex")
def test_the_pdflatex_engine_runs_bibtex_inside_the_build_directory(tmp_path):
    main = _doc_tree(tmp_path)
    rec = documents.compile_document(main, find_engine("pdflatex"), tmp_path / "report" / "report.pdf", tmp_path / "report")
    assert rec["success"] and not rec["undefined_citations"], (rec, (tmp_path / "report" / "build.log").read_text()[-3000:])


def test_an_engine_that_does_not_run_is_skipped_with_a_message(monkeypatch):
    paths = {"latexmk": "/x/latexmk", "pdflatex": "/x/pdflatex"}
    monkeypatch.setattr(documents.shutil, "which", lambda name: paths.get(name))

    def probe(cmd):
        if "latexmk" in cmd[0]:
            return 1, "MiKTeX could not find the script engine 'perl.exe'"
        return 0, "pdfTeX 3.141592653-2.6-1.40.25 (MiKTeX 24.1)"

    monkeypatch.setattr(documents, "_probe", probe)
    lines = []
    engine = documents.find_engine(log=lines.append)
    assert engine["name"] == "pdflatex" and "needs Perl" in lines[0]
    with pytest.raises(BuildError, match="does not run"):
        documents.find_engine("latexmk")


def test_record_bundles_hold_every_output_the_audit_verifies(runner_run):
    from il_memoir.audit import _walk_outputs
    from il_memoir.packaging import RUN_RECORD_GLOBS, collect
    from il_memoir.logging import read_json

    keep, _ = collect(runner_run["base"], RUN_RECORD_GLOBS)
    names = {rel for _, rel in keep}
    n = 0
    for j in runner_run["manifest"]["jobs"]:
        found = []
        _walk_outputs(read_json(runner_run["runs"] / "jobs" / j["job_id"] / "result.json")["outputs"], found)
        for o in found:
            assert f"runs/jobs/{j['job_id']}/{o['path']}" in names, o["path"]
            n += 1
    assert n > 10 and not any("/checkpoints/" in name for name in names)
    assert any(name.endswith("/outcomes.jsonl") for name in names)


def test_the_secret_scan_keeps_every_project_source_file():
    from il_memoir.packaging import collect, source_patterns

    keep, excluded = collect(ROOT, source_patterns())
    names = {rel for _, rel in keep}
    assert "tests/test_documents_packaging.py" in names       # its fixture text must not look like a key
    assert not [rel for rel in excluded if rel.startswith(("il_memoir/", "scripts/", "tests/", "configs/"))], excluded


def _mini_project(runner_run, dest):
    for d in ("il_memoir", "scripts", "configs", "tests"):
        shutil.copytree(ROOT / d, dest / d, ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("pytest.ini", "requirements-base.txt"):
        shutil.copy(ROOT / f, dest / f)
    shutil.copytree(runner_run["runs"], dest / "runs")
    return dest


def test_the_audit_passes_inside_an_extracted_audit_review_bundle(runner_run, tmp_path):
    from il_memoir.packaging import verify_audit_bundle

    proj = _mini_project(runner_run, tmp_path / "proj")
    package(tmp_path / "out", proj, include_large=False, verify=False, update_root_checksums=False, log=QUIET)
    rec = verify_audit_bundle(tmp_path / "out" / "IL-Memoir-audit-review.zip", log=QUIET)
    assert set(rec) == {"smoke"} and rec["smoke"]["passed"], rec


def test_a_failed_audit_inside_the_bundle_fails_package_verify(runner_run, tmp_path, monkeypatch):
    from il_memoir import packaging
    from il_memoir.packaging import verification_failures

    proj = _mini_project(runner_run, tmp_path / "proj")
    ev = next(j["job_id"] for j in runner_run["manifest"]["jobs"] if j["type"] == "eval")
    path = proj / "runs" / "jobs" / ev / "outputs" / "evaluations.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace('"js_nats": ', '"js_nats": 9.0, "_was": '), encoding="utf-8")
    # the smoke check of the reproduction bundle is not what this test is about
    monkeypatch.setattr(packaging, "verify_reproduction", lambda zip_path, log=print: {"passed": True, "exit_code": 0})
    man = package(tmp_path / "out", proj, include_large=False, verify=True, update_root_checksums=False, log=QUIET)
    assert man["audit_check"]["smoke"]["passed"] is False and man["audit_check"]["smoke"]["exit_code"] == 1
    assert verification_failures(man) == ["IL-Memoir-audit-review.zip: the audit of the smoke suites failed (exit status 1)"]


def _package_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_package_script", ROOT / "scripts" / "package.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verified_manifest(audit_check):
    files = {"files": {"x": "0" * 64}, "sha256": "0" * 64}
    return {"bundles": {"IL-Memoir-reproduction.zip": dict(files), "IL-Memoir-audit-review.zip": dict(files),
                        "theory-overleaf.zip": dict(files, verification={"compiled": True, "exit_code": 0})},
            "reproduction_check": {"passed": True, "exit_code": 0}, "audit_check": audit_check}


def test_package_verify_exits_nonzero_when_any_recorded_check_failed(tmp_path, monkeypatch, capsys):
    from il_memoir.packaging import verification_failures

    failed = {"final": {"passed": False, "exit_code": 1}, "pilot": {"passed": True, "exit_code": 0}}
    passed = {"final": {"passed": True, "exit_code": 0}}
    assert verification_failures(_verified_manifest(passed)) == []
    assert verification_failures(_verified_manifest({"skipped": "no suite records in the bundle"})) == []
    assert verification_failures(_verified_manifest(None)) == ["IL-Memoir-audit-review.zip: the audit did not run"]
    assert verification_failures(_verified_manifest(failed)) == [
        "IL-Memoir-audit-review.zip: the audit of the final suites failed (exit status 1)"]
    broken = _verified_manifest(passed)
    broken["bundles"]["theory-overleaf.zip"]["verification"] = {"compiled": False, "exit_code": 12,
                                                                "failure_reasons": ["engine exit status 12"], "log": "x.log"}
    broken["reproduction_check"] = {"passed": False, "exit_code": 1}
    assert len(verification_failures(broken)) == 2
    mod = _package_script()
    for man, code in ((_verified_manifest(failed), 1), (broken, 1), (_verified_manifest(passed), 0)):
        monkeypatch.setattr(mod, "package", lambda *a, _m=man, **k: _m)
        assert mod.main(["--out", str(tmp_path / "out"), "--verify", "--no-root-checksums"]) == code
    out = capsys.readouterr().out
    assert "VERIFICATION FAILED: IL-Memoir-audit-review.zip: the audit of the final suites failed" in out
    assert "VERIFICATION FAILED: theory-overleaf.zip: the test compilation failed (engine exit status 12" in out
    assert out.count("PACKAGING CHECK FAILED") == 2 and out.count("PACKAGING OK") == 1


def test_log_parsing_survives_tex_line_wrapping():
    """TeX breaks log lines at 79 characters; a long build path split the page count on the Cowork VM."""
    out = "Output written on /sessions/a-long-session-name/tmp/build_report_abcdefgh/report.pdf (60 pages, 1234567 bytes)."
    warn = ("LaTeX Warning: Reference `sec:a-label-long-enough-to-cross-the-seventy-nine-character-limit' on page 3 "
            "undefined on input line 12.")

    def wrap(line):
        return "\n".join(line[i:i + 79] for i in range(0, len(line), 79))

    rec = documents.parse_log("This is pdfTeX\n" + wrap(warn) + "\n" + wrap(out) + "\n")
    assert rec["pages"] == 60
    assert rec["undefined_references"] == ["sec:a-label-long-enough-to-cross-the-seventy-nine-character-limit"]


def test_a_failed_verification_keeps_its_log_exit_status_and_reasons(tmp_path):
    if find_engine() is None:
        pytest.skip("no LaTeX engine on this machine")
    main = tmp_path / "doc" / "doc.tex"
    main.parent.mkdir()
    main.write_text("\\documentclass{article}\\begin{document}See \\ref{nowhere} and \\undefinedmacro.\\end{document}\n",
                    encoding="utf-8")
    rec = latex_bundle(main, tmp_path / "out" / "doc-overleaf.zip")
    check = documents.verify_bundle(rec["zip"])
    log_dir = tmp_path / "out" / "doc-overleaf-verification"          # the default: next to the ZIP
    assert not check["compiled"] and check["exit_code"] != 0 and check["log"] == str(log_dir / "build.log")
    assert any("exit status" in r for r in check["failure_reasons"]) and any("error" in r for r in check["failure_reasons"])
    assert "undefinedmacro" in (log_dir / "build.log").read_text(encoding="utf-8", errors="replace")
    assert check["log_tail"] and not json.loads((log_dir / "build.json").read_text(encoding="utf-8"))["success"]


def test_the_build_compiles_a_replica_and_ignores_stale_build_products(tmp_path):
    main = _doc_tree(tmp_path)
    src = main.parent
    (src / "report.fdb_latexmk").write_text("# Fdb version 4\n", encoding="utf-8")       # stale editor build
    (src / "report.aux").write_text("\\relax\n", encoding="utf-8")
    (src / "report.pdf").write_bytes(b"%PDF-1.4 stale\n")
    work = documents.replicate_document(main, tmp_path / "replica")
    names = {p.relative_to(tmp_path / "replica").as_posix() for p in (tmp_path / "replica").rglob("*") if p.is_file()}
    assert names == {"report/report.tex", "report/refs.bib", "report/sections/intro.tex", "results/figures/fig.png"}
    assert work == tmp_path / "replica" / "report" / "report.tex"
    if find_engine() is None:
        return
    rec = documents.compile_document(main, find_engine(), tmp_path / "pdf" / "report.pdf", tmp_path / "log")
    assert rec["success"] and rec["exit_code"] == 0 and not rec["failure_reasons"], rec.get("log_tail")
    assert rec["pages"] == 1 and (tmp_path / "pdf" / "report.pdf").stat().st_size > 1000      # a real build, not the stale file
    assert sorted(p.name for p in src.iterdir()) == ["refs.bib", "report.aux", "report.fdb_latexmk", "report.pdf",
                                                     "report.tex", "sections"]            # the source folder is untouched
