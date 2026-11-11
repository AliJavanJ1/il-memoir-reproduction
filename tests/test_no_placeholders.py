"""INTERFACE.md: no TODO / NotImplemented path or stub in the scientific package and scripts."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"\b(TODO|FIXME|XXX|NotImplemented(Error)?)\b")


def _sources():
    files = sorted((ROOT / "il_memoir").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    assert files, "no source files found"
    return files


def test_no_todo_markers():
    offenders = []
    for path in _sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if PATTERN.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_no_stub_functions():
    """A function whose body is only a docstring, ``pass`` or ``...`` is a stub."""
    stubs = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [b for b in node.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str))]
                if all(isinstance(b, ast.Pass) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and b.value.value is Ellipsis) for b in body):
                    stubs.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.name}")
    assert not stubs, "\n".join(stubs)
