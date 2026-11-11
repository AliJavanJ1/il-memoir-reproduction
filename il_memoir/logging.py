"""Run identity, provenance, checkpoints and real metric records (INTERFACE.md, ``il_memoir/logging.py``).

Bookkeeping only.  Nothing in this module changes a number that a learner, the
oracle or the evaluation computes; it records what was run and makes the
records safe to write and to resume.  (The module is ``il_memoir.logging``;
inside the package the standard library is still ``import logging``.)

Contents
--------
* canonical JSON and SHA-256 hashes of configurations, files and arrays;
* **code scopes**: the SHA-256 of the package modules that can change the
  output of each job type (the transitive closure of the package imports of a
  job type's anchor modules, computed from the source with ``ast``), so that a
  code change invalidates exactly the affected runs (INTERFACE: "If code
  changes, invalidate affected runs");
* dependency, git and host records;
* atomic JSON and torch writes (write to a temporary file, then ``os.replace``);
* an append-only JSON-lines metric log that can be truncated to the last
  checkpoint on resume;
* job status records (``planned``, ``running``, ``interrupted``, ``failed``,
  ``completed``) with one entry per attempt;
* the Ctrl+C controller: the first Ctrl+C asks the running job to checkpoint
  and stop at its next safe point; a second one aborts immediately;
* a lock file against two runners on the same suite or job.

Units (``UNITS``): seconds, MiB, nats (natural logarithm), environment
transitions, optimizer updates.  Returns are discounted sums with the
next-state timing of ``il_memoir.tasks`` (gamma and H from the environment
configuration).
"""
from __future__ import annotations

import ast
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Union

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
PACKAGE = "il_memoir"

# job status values (status.json); "blocked" and "stale" are computed by the runner, never stored
PLANNED, RUNNING, INTERRUPTED, FAILED, COMPLETED = "planned", "running", "interrupted", "failed", "completed"
STATUSES = (PLANNED, RUNNING, INTERRUPTED, FAILED, COMPLETED)

UNITS = {
    "time": "seconds (s); wall-clock, with torch.cuda.synchronize() before each reading on CUDA",
    "memory": "MiB (2**20 bytes)",
    "divergence": "nats (natural logarithm); JS(d, q) with half weights lies in [0, log 2]",
    "return": "discounted sum sum_{t<H} gamma^t r_g(s_{t+1}) (next-state timing), gamma and H from configs/environment.yaml",
    "probability": "fraction in [0, 1]",
    "env_steps": "environment transitions (one call of the simulator kernel per trajectory and time step)",
    "updates": "optimizer steps (f-PG policy updates, FB minibatch updates, DM latent steps)",
}

# Modules that only record or schedule work.  They are excluded from code scopes:
# editing them cannot change a learner update, a datum or a metric.
BOOKKEEPING = frozenset({"logging", "hardware", "runner", "analysis", "audit", "pilot", "documents", "packaging",
                         "checks", "experiment", "__init__"})

# Anchor modules of each job type.  A scope is the transitive closure of the package imports of
# its anchors, except that ``train`` (the shared executor file) contributes its own source only:
# its imports of the other learners are covered by each type's learner anchor.
SCOPE_ANCHORS: Dict[str, Sequence[str]] = {
    "data": ("prepare",),
    "fpg": ("fpg", "datasets", "train"),
    "fb_pretrain": ("fb", "datasets", "train"),
    "adapt": ("fb_il", "datasets", "train"),
    "eval": ("evaluation", "datasets"),
    "eval_expert": ("evaluation", "datasets"),
}
FILE_ONLY_ANCHORS = frozenset({"train"})


# =============================================================================
# time, JSON and hashing
# =============================================================================
def utc_now() -> str:
    """UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    """UTC time for file names: ``YYYYMMDDTHHMMSSZ``."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def to_jsonable(obj: Any) -> Any:
    """Plain JSON types: NumPy scalars and arrays, paths, tuples, sets and dataclasses are converted."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(to_jsonable(v) for v in obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def canonical_json(obj: Any) -> str:
    """Key-sorted compact JSON (no NaN or infinity): the input of every configuration hash."""
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Union[str, Path], text: bool = False) -> str:
    """SHA-256 of a file's bytes; ``text=True`` normalizes CRLF to LF first (Windows checkouts hash the same)."""
    p = Path(path)
    if text:
        return sha256_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
    h = hashlib.sha256()
    with open(p, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """SHA-256 of an array's dtype, shape and C-order bytes."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def finite_or_none(x: Any) -> Optional[float]:
    """A float, or ``None`` for a missing value; NaN and infinities raise (real metrics must be finite)."""
    if x is None:
        return None
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"metric value {x!r} is not finite")
    return v


def relpath(path: Union[str, Path], root: Union[str, Path] = PROJECT_ROOT) -> str:
    """``path`` relative to ``root`` with forward slashes, or the absolute POSIX path if outside it."""
    p = Path(path).resolve()
    try:
        return p.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return p.as_posix()


# =============================================================================
# atomic writes
# =============================================================================
def _replace(tmp: Path, dest: Path, attempts: int = 20) -> None:
    """``os.replace`` with retries: on Windows a reader holding the target open makes it fail briefly."""
    for k in range(attempts):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(0.05 * (k + 1))


replace_file = _replace          # public name for modules that write their own temporary file


def write_text_atomic(path: Union[str, Path], text: str) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    _replace(tmp, dest)


def write_json_atomic(path: Union[str, Path], obj: Any, indent: int = 2) -> None:
    write_text_atomic(path, json.dumps(to_jsonable(obj), indent=indent, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_torch_atomic(obj: Any, path: Union[str, Path]) -> None:
    """``torch.save`` through a temporary file, so an interruption never leaves a truncated checkpoint."""
    import torch

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as handle:
        torch.save(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())
    _replace(tmp, dest)


def load_torch(path: Union[str, Path]) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


# =============================================================================
# JSON-lines metric log
# =============================================================================
class JsonlLog:
    """Append-only JSON-lines log.  ``truncate(n)`` keeps the first ``n`` records (resume from a checkpoint).

    A crash or power loss during an append can leave a last line without its
    newline.  ``records()`` ignores such a torn last line when it is not valid
    JSON, and ``append()`` first rewrites the file without it, so a resumed job
    continues instead of failing on its own log.  A malformed line anywhere
    else is corruption and raises.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _unterminated(self) -> bool:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return False
        if size == 0:
            return False
        with open(self.path, "rb") as handle:
            handle.seek(size - 1)
            return handle.read(1) != b"\n"

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(to_jsonable(record), sort_keys=True, allow_nan=False)
        if self._unterminated():
            self.truncate(len(self.records()))        # drop a torn last line (or terminate a complete one)
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()

    def records(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8", newline="") as handle:
            text = handle.read()
        lines = text.split("\n")
        out = []
        for k, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if k == len(lines) - 1 and not text.endswith("\n"):
                    break                             # torn last append: the record was never completed
                raise ValueError(f"{self.path}: line {k + 1} is not valid JSON ({exc})") from exc
        return out

    def count(self) -> int:
        return len(self.records())

    def durable_count(self) -> int:
        """Record count after forcing the file to disk; a resume checkpoint stores this count."""
        if self.path.exists():
            with open(self.path, "ab") as handle:
                os.fsync(handle.fileno())
        return self.count()

    def truncate(self, n_records: int) -> int:
        """Keep the first ``n_records`` records; returns the number of records discarded."""
        recs = self.records()
        if n_records > len(recs):
            raise ValueError(f"log {self.path} has {len(recs)} records, fewer than the checkpointed {n_records}")
        write_text_atomic(self.path, "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in recs[:n_records]))
        return len(recs) - n_records


# =============================================================================
# code scopes (static import closure of the package)
# =============================================================================
def package_modules() -> List[str]:
    """Names of the modules of the package (``il_memoir/*.py``)."""
    return sorted(p.stem for p in PACKAGE_DIR.glob("*.py"))


def module_imports(name: str) -> Set[str]:
    """Package modules imported anywhere in ``il_memoir/<name>.py`` (module-level or inside functions)."""
    path = PACKAGE_DIR / f"{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    known = set(package_modules())
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == PACKAGE and len(parts) > 1 and parts[1] in known:
                    found.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            parts = module.split(".") if module else []
            if node.level > 0:
                if parts and parts[0] in known:
                    found.add(parts[0])
                elif not parts:
                    found.update(a.name for a in node.names if a.name in known)
            elif parts and parts[0] == PACKAGE:
                if len(parts) > 1 and parts[1] in known:
                    found.add(parts[1])
                elif len(parts) == 1:
                    found.update(a.name for a in node.names if a.name in known)
    found.discard(name)
    return found


def import_closure(anchors: Iterable[str], exclude: Iterable[str] = BOOKKEEPING,
                   file_only: Iterable[str] = FILE_ONLY_ANCHORS) -> List[str]:
    """Anchors plus every package module they import transitively, minus ``exclude`` (bookkeeping)."""
    excluded = set(exclude)
    file_only = set(file_only)
    seen: Set[str] = set()
    stack = list(anchors)
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        if not (PACKAGE_DIR / f"{mod}.py").exists():
            raise FileNotFoundError(f"package module {mod!r} does not exist")
        seen.add(mod)
        if mod in file_only:
            continue
        stack.extend(m for m in module_imports(mod) if m not in excluded)
    return sorted(m for m in seen if m not in excluded or m in file_only)


def code_hash(modules: Sequence[str]) -> Dict[str, Any]:
    """Per-file SHA-256 (LF-normalized) of ``il_memoir/<m>.py`` for the given modules, and their combination."""
    files = {f"{PACKAGE}/{m}.py": sha256_file(PACKAGE_DIR / f"{m}.py", text=True) for m in sorted(set(modules))}
    combined = sha256_text("".join(f"{k}:{v}\n" for k, v in sorted(files.items())))
    return {"modules": sorted(set(modules)), "files": files, "sha256": combined}


def code_scope(job_type: str) -> Dict[str, Any]:
    """Code record of a job type: the modules whose source can change its outputs, with their hashes."""
    if job_type not in SCOPE_ANCHORS:
        raise KeyError(f"unknown job type {job_type!r}; known: {sorted(SCOPE_ANCHORS)}")
    rec = code_hash(import_closure(SCOPE_ANCHORS[job_type]))
    rec["scope"] = job_type
    return rec


def package_code_hash() -> Dict[str, Any]:
    """Hash of every module of the package (provenance of a whole launch)."""
    return code_hash(package_modules())


# =============================================================================
# dependency, git and host records
# =============================================================================
NUMERIC_DEPENDENCIES = ("python", "numpy", "scipy", "torch", "torch_cuda_build")


def dependency_record() -> Dict[str, Any]:
    """Versions of Python, NumPy, SciPy, torch (with its CUDA build), PyYAML and Matplotlib.

    ``sha256`` hashes only the versions that can change a job's numbers
    (``NUMERIC_DEPENDENCIES``); the runner's cache validation compares it.
    PyYAML and Matplotlib are recorded but not hashed: a configuration that
    parses differently already changes the job identity, and jobs draw no
    figures, so a plotting upgrade does not make finished jobs stale.
    """
    rec: Dict[str, Any] = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import scipy

        rec["scipy"] = scipy.__version__
    except ImportError:
        rec["scipy"] = None
    try:
        import torch

        rec["torch"] = torch.__version__
        rec["torch_cuda_build"] = torch.version.cuda
    except ImportError:
        rec["torch"] = None
        rec["torch_cuda_build"] = None
    try:
        import yaml

        rec["pyyaml"] = yaml.__version__
    except ImportError:
        rec["pyyaml"] = None
    try:
        import matplotlib

        rec["matplotlib"] = matplotlib.__version__
    except ImportError:
        rec["matplotlib"] = None
    rec["sha256"] = sha256_json({k: rec[k] for k in NUMERIC_DEPENDENCIES})
    return rec


def git_record(root: Union[str, Path] = PROJECT_ROOT) -> Optional[Dict[str, Any]]:
    """Commit and dirty flag if ``git`` is installed and ``root`` is a repository; ``None`` otherwise."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, timeout=10)
        if head.returncode != 0:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=str(root), capture_output=True, text=True, timeout=20)
        return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None}
    except (OSError, subprocess.SubprocessError):
        return None


def host_fingerprint() -> Dict[str, str]:
    """Host identity used to keep hardware records and device choices apart (never a secret)."""
    return {"hostname": socket.gethostname(), "system": platform.system(), "machine": platform.machine()}


# =============================================================================
# job status records
# =============================================================================
def read_status(job_dir: Union[str, Path]) -> Dict[str, Any]:
    path = Path(job_dir) / "status.json"
    if not path.exists():
        return {"status": PLANNED, "attempts": []}
    return read_json(path)


OUTCOME_JOURNAL = "outcomes.jsonl"
TERMINAL_EVENTS = (COMPLETED, FAILED, INTERRUPTED, "aborted")


def read_outcome_journal(runs_root: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Job outcome events of every suite launch (``runs/suites/*/outcomes.jsonl``), per job, oldest first.

    The journal is append-only and lives outside the job directories, so a job
    directory that is deleted (instead of being invalidated by the runner)
    leaves a trace: its recorded outcomes no longer match ``runs/jobs``.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted((Path(runs_root) / "suites").glob("*/" + OUTCOME_JOURNAL)):
        for event in JsonlLog(path).records():
            event = dict(event, journal=path.parent.name)
            out.setdefault(event["job_id"], []).append(event)
    for events in out.values():
        events.sort(key=lambda e: e["utc"])            # stable: ties keep the order of the file
    return out


def current_outcomes(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The events after the job's last invalidation: the history of the directory now in ``runs/jobs``."""
    last = max((k for k, e in enumerate(events) if e["event"] == "invalidated"), default=-1)
    return list(events[last + 1:])


def write_status(job_dir: Union[str, Path], status: str, **fields: Any) -> Dict[str, Any]:
    """Set the status of a job; ``attempt`` (a dict) updates the current attempt or opens a new one."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    rec = read_status(job_dir)
    rec["status"] = status
    rec["updated_utc"] = utc_now()
    attempt = fields.pop("attempt", None)
    new_attempt = fields.pop("new_attempt", False)
    if attempt is not None:
        attempts = rec.setdefault("attempts", [])
        if new_attempt or not attempts:
            attempts.append(dict(attempt))
        else:
            attempts[-1].update(attempt)
    rec.update(fields)
    write_json_atomic(Path(job_dir) / "status.json", rec)
    return rec


# =============================================================================
# Ctrl+C: checkpoint at the next safe point
# =============================================================================
class JobInterrupted(Exception):
    """Raised by a job after it saved a resume checkpoint because a stop was requested."""


class StopController:
    """Process-wide stop flag.  ``install()`` routes Ctrl+C (SIGINT) to it while a suite runs.

    First Ctrl+C: the running job finishes its current update, writes a resume
    checkpoint and stops (status ``interrupted``); no further job starts.
    Second Ctrl+C: ``KeyboardInterrupt`` is raised immediately; the job keeps
    its last periodic checkpoint and is resumed from it next time.
    Tests call :meth:`request` or set ``stop_after`` to trigger the same path.
    """

    def __init__(self) -> None:
        self._flag = threading.Event()
        self._previous: Optional[Callable] = None
        self._presses = 0
        self.stop_after: Optional[int] = None     # test hook: request a stop after this many safe points
        self._safe_points = 0

    def request(self, reason: str = "requested") -> None:
        self.reason = reason
        self._flag.set()

    def requested(self) -> bool:
        return self._flag.is_set()

    def reset(self) -> None:
        self._flag.clear()
        self._presses = 0
        self._safe_points = 0
        self.stop_after = None

    def safe_point(self) -> bool:
        """Called by jobs after each completed update; True when the job must checkpoint and stop."""
        self._safe_points += 1
        if self.stop_after is not None and self._safe_points >= self.stop_after:
            self.request("stop_after test hook")
        return self.requested()

    def _handler(self, signum, frame) -> None:          # pragma: no cover - exercised through a subprocess test
        self._presses += 1
        if self._presses == 1:
            self.request("Ctrl+C")
            print("\nCtrl+C received: the current job will save a checkpoint and stop at its next safe point. "
                  "Press Ctrl+C again to abort immediately.", flush=True)
        else:
            raise KeyboardInterrupt

    @contextmanager
    def installed(self) -> Iterator["StopController"]:
        """Install the SIGINT handler (main thread only); restore the previous handler afterwards."""
        if threading.current_thread() is not threading.main_thread():
            yield self
            return
        self._previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)
        try:
            yield self
        finally:
            signal.signal(signal.SIGINT, self._previous)


STOP = StopController()


# =============================================================================
# job context (passed by the runner to the executors)
# =============================================================================
@dataclasses.dataclass
class JobContext:
    """Everything an executor may use.  ``env_cfg`` is the full environment configuration: training executors
    pass learners ``env_cfg.learner_dict()`` only and use the full object to build simulators."""

    job: Any                                   # experiment.JobSpec
    job_dir: Path
    data_root: Path
    data_id: str
    env_cfg: Any                               # config.EnvironmentConfig
    device: str
    runner: Mapping[str, Any]
    stop: StopController
    data_entries: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)   # role -> manifest entry
    parents: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)        # job_id -> {"dir", "result"}
    job_dir_of: Optional[Callable[[str], Path]] = None                                   # any job id -> its directory
    log: Callable[[str], None] = print

    @property
    def identity(self) -> Dict[str, Any]:
        return self.job.identity


# =============================================================================
# lock files
# =============================================================================
def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except ImportError:                                  # pragma: no cover - psutil is a base requirement
        if os.name == "nt":
            return True
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class LockError(RuntimeError):
    pass


@contextmanager
def file_lock(path: Union[str, Path], what: str) -> Iterator[Path]:
    """Exclusive lock file with host and PID.  A lock left by a dead process on this host is taken over."""
    lock = Path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    me = {"pid": os.getpid(), "host": socket.gethostname(), "utc": utc_now(), "what": what}
    if lock.exists():
        try:
            other = read_json(lock)
        except (OSError, ValueError):
            other = {}
        same_host = other.get("host") == me["host"]
        if same_host and isinstance(other.get("pid"), int) and _pid_alive(int(other["pid"])) and other["pid"] != me["pid"]:
            raise LockError(f"{what} is locked by process {other['pid']} on this computer since {other.get('utc')} "
                            f"({lock}); do not run two copies at once")
        if not same_host and other:
            raise LockError(f"{what} is locked by host {other.get('host')!r} ({lock}); remove the lock file only if "
                            "that host is no longer running it")
    write_json_atomic(lock, me)
    try:
        yield lock
    finally:
        try:
            if lock.exists() and read_json(lock).get("pid") == me["pid"]:
                lock.unlink()
        except (OSError, ValueError):
            pass
