"""Every INTERFACE script starts from the project root without editing sys.path and prints its help (Prompt 3C)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["prepare_data", "smoke", "benchmark", "run_suite", "analyze", "audit_results", "build", "package",
           "check_environments", "check_learners", "diagnose"]


def test_every_script_runs_from_the_project_root_with_help():
    procs = {s: subprocess.Popen([sys.executable, f"scripts/{s}.py", "--help"], cwd=str(ROOT),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for s in SCRIPTS}
    for name, proc in procs.items():
        out, err = proc.communicate(timeout=300)
        assert proc.returncode == 0, f"{name}: {err[-1500:]}"
        assert "usage" in out.lower(), name


def test_interface_commands_are_documented_in_the_readme():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for cmd in ["-m pytest -q", r"scripts\prepare_data.py --config configs\core.yaml", r"scripts\smoke.py --device auto",
                r"scripts\benchmark.py --config configs\pilot.yaml --device auto",
                r"scripts\run_suite.py --config configs\core.yaml --device auto --resume",
                r"scripts\run_suite.py --config configs\compact_ablations.yaml --device auto --resume",
                r"scripts\run_suite.py --config configs\reduced.yaml --device auto --resume",
                r"scripts\analyze.py --runs runs --out results", r"scripts\audit_results.py --runs runs --results results",
                r"scripts\build.py --target all", r"scripts\package.py --out deliverables",
                r"Expand-Archive deliverables\IL-Memoir-reproduction.zip"]:
        assert cmd in text, cmd
