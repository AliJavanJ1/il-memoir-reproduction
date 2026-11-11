"""Sequential, cached, resumable suite runner (``scripts/run_suite.py``; INTERFACE.md).

Layout of ``--runs`` (default ``runs/``)::

    runs/jobs/<job_id>/            one directory per job, shared by every suite that declares the job
        job.json                   identity (complete resolved settings), labels, inputs and their hashes,
                                   code scope with per-file SHA-256, dependency versions
        status.json                planned / running / interrupted / failed / completed, one entry per attempt
                                   (UTC start/end, host, device, synchronized wall time, memory, error)
        log.jsonl                  real training metrics (f-PG iterations, FB updates)
        checkpoints/resume.pt      latest resume checkpoint (optimizer and random-generator states)
        outputs/...                policies, FB snapshots, latents, evaluations (with SHA-256 in result.json)
        result.json                outputs, costs per phase and summary; written last (the completion marker)
    runs/suites/<suite>/
        manifest.json              written before any job is launched: frozen configuration and its hash,
                                   seeds, data, device, every planned job, code hashes, revisions
        seed_manifest.json         every job's seed and stream labels
        status.json                current status of every planned job
        launches.jsonl             one record per launch (command, device decision, code, dependencies)
        outcomes.jsonl             append-only journal: every job outcome (completed, failed, interrupted,
                                   aborted) and invalidation of this suite's launches; the audit compares
                                   it with runs/jobs, so a deleted job directory cannot go unnoticed
        hardware/<host>-<utc>.json hardware record of each launch (runs/environment_laptop.json is never touched)
    runs/invalidated/<job_id>__<utc>/   job directories invalidated by a code, configuration, data or
                                        dependency change (moved, never deleted)

Scheduling is sequential, in the dependency order of ``experiment.expand_jobs``.

Cache validation.  A job directory is reused only if its identity, the SHA-256
of its code scope (``logging.code_scope``), the dependency versions, the
content hashes of its data inputs and the result hashes of its parent jobs
equal the current ones, and (when completed) every output file still has its
recorded hash.  Otherwise the job is *stale*.  The runner then launches
nothing and lists the stale jobs with the reason, unless
``--invalidate-changed`` is given: the stale directories are moved to
``runs/invalidated/`` and the jobs rerun.  Outputs of different code are never
mixed silently.

Statuses.  ``completed`` jobs are skipped (cached); ``interrupted`` and stale
``running`` jobs (a process that died) resume from their checkpoint; ``failed``
jobs keep their error and are not retried unless ``--retry-failed``; jobs whose
dependencies are not completed are ``blocked``.  Every planned seed stays in
the manifest and in the status table: a resumed suite whose seed manifest
lost a seed is refused (even with ``--allow-config-change``), and any other
seed change needs ``--allow-config-change`` and is recorded as a revision.
A stop request (Ctrl+C) also prevents any further job from starting.

Device.  The first launch freezes the device in the manifest
(``hardware.resolve_run_device``); ``--device auto`` on a resumed suite uses it.
An explicit ``--device`` overrides it for the jobs launched now; each job
records the device it ran on.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import datasets as ds
from .experiment import JobSpec, SuiteConfig, data_items, expand_jobs, seed_manifest_record, sim_sha256
from .hardware import DeviceDecision, hardware_record, memory_record, read_pilot_choice, reset_cuda_peak, resolve_run_device, set_cpu_threads
from .logging import (
    COMPLETED,
    FAILED,
    INTERRUPTED,
    PLANNED,
    RUNNING,
    STOP,
    JobContext,
    JobInterrupted,
    OUTCOME_JOURNAL,
    TERMINAL_EVENTS,
    JsonlLog,
    LockError,
    StopController,
    code_scope,
    current_outcomes,
    dependency_record,
    file_lock,
    git_record,
    host_fingerprint,
    package_code_hash,
    read_json,
    read_outcome_journal,
    read_status,
    relpath,
    sha256_file,
    utc_now,
    utc_stamp,
    write_json_atomic,
    write_status,
)

MANIFEST_SCHEMA = "il_memoir/suite-manifest-v1"
JOB_SCHEMA = "il_memoir/job-v1"
EXIT_OK, EXIT_FAILED, EXIT_CONFIG, EXIT_STALE, EXIT_INTERRUPTED = 0, 1, 2, 3, 130


class RunnerError(RuntimeError):
    pass


def _executors() -> Dict[str, Callable]:
    from . import evaluation, train

    return {**train.EXECUTORS, **evaluation.EXECUTORS}


class Runner:
    """Plan, validate, launch and record one suite."""

    def __init__(
        self,
        suite: SuiteConfig,
        runs_root: Union[str, Path] = "runs",
        data_root: Union[str, Path] = "data",
        device: str = "auto",
        pilot_path: Union[str, Path, None] = None,
        allow_config_change: bool = False,
        invalidate_changed: bool = False,
        retry_failed: bool = False,
        stop: StopController = STOP,
        log: Callable[[str], None] = print,
        argv: Optional[Sequence[str]] = None,
    ):
        self.suite = suite
        self.runs_root = Path(runs_root)
        self.data_root = Path(data_root)
        self.device_request = device
        self.pilot_path = Path(pilot_path) if pilot_path is not None else self.runs_root / "benchmark.json"
        self.allow_config_change = allow_config_change
        self.invalidate_changed = invalidate_changed
        self.retry_failed = retry_failed
        self.stop = stop
        self.log = log
        self.argv = list(argv) if argv is not None else list(sys.argv)
        self.jobs: List[JobSpec] = expand_jobs(suite)
        self.by_id = {j.job_id: j for j in self.jobs}
        self.suite_dir = self.runs_root / "suites" / suite.name
        self._code_cache: Dict[str, Dict[str, Any]] = {}
        self._deps = dependency_record()
        self.decision: Optional[DeviceDecision] = None

    # ------------------------------------------------------------------ paths
    def job_dir(self, job_id: str) -> Path:
        return self.runs_root / "jobs" / job_id

    @property
    def manifest_path(self) -> Path:
        return self.suite_dir / "manifest.json"

    def code(self, job_type: str) -> Dict[str, Any]:
        if job_type not in self._code_cache:
            self._code_cache[job_type] = code_scope(job_type)
        return self._code_cache[job_type]

    # ------------------------------------------------------------------- data
    def ensure_data(self) -> Dict[str, Dict[str, Any]]:
        """Verify every data item (file hash and specification); prepare the missing ones."""
        from .prepare import missing_items, prepare_items

        missing = missing_items(self.suite, self.data_root)
        if missing:
            self.log(f"preparing {len(missing)} missing data item(s) in {self.data_root / self.suite.data_id}")
            prepare_items(self.suite, self.data_root, missing, log=self.log)
        entries = {}
        for key, item in data_items(self.suite).items():
            entries[key] = ds.verify_item(self.data_root, self.suite.data_id, key, item.spec_sha256)
        return entries

    # ------------------------------------------------------------- validation
    def _input_hashes(self, job: JobSpec, entries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        data = {role: entries[key]["content_sha256"] for role, key in sorted(job.data_inputs.items())}
        parents = {}
        for dep in job.deps:
            res = self.job_dir(dep) / "result.json"
            parents[dep] = sha256_file(res) if res.exists() else None
        return {"data": data, "parents": parents}

    def job_state(self, job: JobSpec, entries: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
        """``(status, reason)`` with status in planned, running, interrupted, failed, completed, stale."""
        d = self.job_dir(job.job_id)
        if not (d / "job.json").exists():
            return PLANNED, ""
        rec = read_json(d / "job.json")
        status = read_status(d)["status"]
        reasons = []
        if rec.get("identity_sha256") != job.identity_sha256:
            reasons.append("identity hash differs (the directory belongs to other settings)")
        code_now = self.code(job.type)
        if rec.get("code", {}).get("sha256") != code_now["sha256"]:
            old = rec.get("code", {}).get("files", {})
            changed = sorted(k for k in set(old) | set(code_now["files"]) if old.get(k) != code_now["files"].get(k))
            reasons.append("code changed: " + ", ".join(changed))
        if rec.get("dependencies", {}).get("sha256") != self._deps["sha256"]:
            reasons.append("dependency versions changed")
        inputs = rec.get("input_hashes", {})
        now = self._input_hashes(job, entries)
        if inputs.get("data") != now["data"]:
            reasons.append("data input content changed")
        if status != PLANNED and inputs.get("parents") != now["parents"]:
            reasons.append("a parent job's result changed (it was rerun)")
        if status == COMPLETED and not reasons:
            bad = self._verify_outputs(d)
            if bad:
                reasons.append("output files changed after completion: " + ", ".join(bad))
        if reasons:
            return "stale", "; ".join(reasons)
        return status, ""

    @staticmethod
    def _verify_outputs(job_dir: Path) -> List[str]:
        res_path = job_dir / "result.json"
        if not res_path.exists():
            return ["result.json missing"]
        res = read_json(res_path)
        bad = []

        def walk(obj):
            if isinstance(obj, dict):
                if "path" in obj and "sha256" in obj and isinstance(obj["path"], str):
                    p = job_dir / obj["path"]
                    if not p.exists() or sha256_file(p) != obj["sha256"]:
                        bad.append(obj["path"])
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)

        walk(res.get("outputs", {}))
        return bad

    def states(self, entries: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
        """Status of every planned job; descendants of a stale job are stale too (they read its outputs)."""
        states = {j.job_id: self.job_state(j, entries) for j in self.jobs}
        for j in self.jobs:                                   # dependency order: parents are decided first
            if states[j.job_id][0] in (PLANNED, "stale"):
                continue
            bad = [dep for dep in j.deps if states.get(dep, (PLANNED, ""))[0] == "stale"]
            if bad:
                states[j.job_id] = ("stale", "a parent job is stale: " + ", ".join(bad))
        return states

    def _archive(self, job_id: str, reason: str) -> Path:
        dest = self.runs_root / "invalidated" / f"{job_id}__{utc_stamp()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.job_dir(job_id)), str(dest))
        write_json_atomic(dest / "invalidated.json", {"job_id": job_id, "utc": utc_now(), "reason": reason,
                                                      "suite": self.suite.name})
        self._journal(job_id, "invalidated", archive=relpath(dest), reason=reason)
        self.log(f"  invalidated {job_id} -> {relpath(dest)} ({reason})")
        return dest

    def _journal(self, job_id: str, event: str, **fields: Any) -> None:
        rec = {"utc": utc_now(), "job_id": job_id, "event": event, **fields}
        if event in TERMINAL_EVENTS:
            attempts = read_status(self.job_dir(job_id)).get("attempts") or [{}]
            rec["attempt_start_utc"] = attempts[-1].get("start_utc")
            rec["n_attempts"] = len(attempts)
        JsonlLog(self.suite_dir / OUTCOME_JOURNAL).append(rec)

    def missing_directories(self) -> Dict[str, str]:
        """Planned jobs whose directory is gone although an outcome of it was recorded (deleted, not invalidated)."""
        journal = read_outcome_journal(self.runs_root)
        out = {}
        for j in self.jobs:
            if (self.job_dir(j.job_id) / "job.json").exists():
                continue
            done = [e for e in current_outcomes(journal.get(j.job_id, [])) if e["event"] in TERMINAL_EVENTS]
            if done:
                out[j.job_id] = f"{done[-1]['event']} at {done[-1]['utc']} (suite {done[-1]['journal']})"
        return out

    def _seed_change(self, previous: Dict[str, Any]) -> Optional[str]:
        """``None`` if the suite has the seeds it was launched with; refuses a removed seed outright."""
        old = [int(s) for s in previous.get("seeds", [])]
        new = [int(s) for s in self.suite.seeds]
        removed = sorted(set(old) - set(new))
        if removed:
            raise RunnerError(
                f"seed(s) {removed} of suite {self.suite.name!r} are no longer in the seed manifest "
                f"{self.suite.seed_manifest.path} (launched with {old}, now {new}). Every planned seed is preserved: "
                "restore the seed manifest. This is refused even with --allow-config-change.")
        parts = []
        if old != new:
            parts.append(f"seeds {old} -> {new}")
        if previous.get("reference_seed") != self.suite.reference_seed:
            parts.append(f"reference seed {previous.get('reference_seed')} -> {self.suite.reference_seed}")
        return "; ".join(parts) or None

    # --------------------------------------------------------------- manifest
    def _manifest(self, previous: Optional[Dict[str, Any]], entries: Dict[str, Dict[str, Any]],
                  seed_change: Optional[str] = None) -> Dict[str, Any]:
        suite = self.suite
        revisions = list(previous.get("revisions", [])) if previous else []
        config_changed = bool(previous) and previous.get("config_sha256") != suite.sha256
        if not previous or config_changed or seed_change:
            reasons = ["first launch"] if not previous else []
            if config_changed:
                reasons.append("configuration changed (--allow-config-change)")
            if seed_change:
                reasons.append(f"seed manifest changed: {seed_change} (--allow-config-change)")
            old_ids = {j["job_id"] for j in previous.get("jobs", [])} if previous else set()
            new_ids = {j.job_id for j in self.jobs}
            revisions.append({"utc": utc_now(), "config_sha256": suite.sha256, "reason": "; ".join(reasons),
                              "seeds": list(suite.seeds), "reference_seed": suite.reference_seed,
                              "seed_manifest_sha256": suite.seed_manifest.sha256,
                              "n_jobs": len(new_ids), "added_jobs": len(new_ids - old_ids) if previous else len(new_ids),
                              "removed_jobs": sorted(old_ids - new_ids)})
        device = previous.get("device") if previous else None
        if device is None:
            device = {"device": self.decision.device, "decision": self.decision.to_dict(), "frozen_utc": utc_now()}
        return {
            "schema": MANIFEST_SCHEMA,
            "suite": suite.name,
            "kind": suite.kind,
            "test_fixture": suite.kind == "smoke",
            "description": suite.description,
            "created_utc": previous.get("created_utc") if previous else utc_now(),
            "updated_utc": utc_now(),
            "config_path": suite.path,
            "config_sha256": suite.sha256,
            "config": suite.raw,
            "environment_config": {"path": suite.env_config_path, "sha256": suite.env_cfg.sha256(),
                                   "sim_sha256": sim_sha256(suite.env_cfg)},
            "seeds": list(suite.seeds),
            "reference_seed": suite.reference_seed,
            "seed_manifest": suite.seed_manifest.to_dict(),
            "data": {"root": relpath(self.data_root), "id": suite.data_id,
                     "items": {k: {"spec_sha256": e["spec_sha256"], "content_sha256": e["content_sha256"]}
                               for k, e in sorted(entries.items())}},
            "device": device,
            "jobs": [j.to_dict() for j in self.jobs],
            "n_jobs": {t: sum(1 for j in self.jobs if j.type == t) for t in sorted({j.type for j in self.jobs})},
            "code": {"package": package_code_hash(), "scopes": {t: self.code(t)["sha256"] for t in sorted({j.type for j in self.jobs})}},
            "dependencies": self._deps,
            "git": git_record(),
            "revisions": revisions,
        }

    def write_status_table(self, states: Dict[str, Tuple[str, str]], blocked: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        blocked = blocked or {}
        rows = {}
        for j in self.jobs:
            st, reason = states.get(j.job_id, (PLANNED, ""))
            if j.job_id in blocked and st != COMPLETED:
                st, reason = "blocked", blocked[j.job_id]
            rows[j.job_id] = {"status": st, "reason": reason, "type": j.type, "labels": j.labels}
        counts: Dict[str, int] = {}
        for r in rows.values():
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        rec = {"suite": self.suite.name, "updated_utc": utc_now(), "counts": counts, "jobs": rows}
        write_json_atomic(self.suite_dir / "status.json", rec)
        return rec

    # ------------------------------------------------------------------ jobs
    def _write_job_record(self, job: JobSpec, entries: Dict[str, Dict[str, Any]]) -> None:
        d = self.job_dir(job.job_id)
        d.mkdir(parents=True, exist_ok=True)
        if (d / "job.json").exists():
            return
        rec = {"schema": JOB_SCHEMA, **job.to_dict(), "created_utc": utc_now(), "code": self.code(job.type),
               "dependencies": self._deps, "input_hashes": self._input_hashes(job, entries),
               "data": {"root": relpath(self.data_root), "id": self.suite.data_id}}
        write_json_atomic(d / "job.json", rec)

    def run_job(self, job: JobSpec, entries: Dict[str, Dict[str, Any]], device: str) -> str:
        """Run one job under its own lock; record the attempt (start, end, device, memory, outcome, error)."""
        d = self.job_dir(job.job_id)
        d.mkdir(parents=True, exist_ok=True)
        with file_lock(d / ".lock", f"job {job.job_id}"):          # before any record: another process may run it
            return self._run_locked(job, entries, device, d)

    def _run_locked(self, job: JobSpec, entries: Dict[str, Dict[str, Any]], device: str, d: Path) -> str:
        import torch

        self._write_job_record(job, entries)
        parents = {dep: {"dir": self.job_dir(dep), "result": read_json(self.job_dir(dep) / "result.json")} for dep in job.deps}
        ctx = JobContext(job=job, job_dir=d, data_root=self.data_root, data_id=self.suite.data_id,
                         env_cfg=self.suite.env_cfg, device=device, runner=self.suite.runner, stop=self.stop,
                         data_entries={role: entries[key] for role, key in job.data_inputs.items()},
                         parents=parents, job_dir_of=self.job_dir, log=self.log)
        attempt = {"start_utc": utc_now(), "host": host_fingerprint(), "pid": os.getpid(),
                   "device": device, "device_decision": self.decision.to_dict() if self.decision else None,
                   "suite": self.suite.name, "torch_threads": torch.get_num_threads()}
        write_status(d, RUNNING, attempt=attempt, new_attempt=True)
        reset_cuda_peak(device)
        t0 = time.perf_counter()
        code = self.code(job.type)
        try:
            result = _executors()[job.type](ctx, code["sha256"])
        except JobInterrupted as exc:
            write_status(d, INTERRUPTED, attempt={"end_utc": utc_now(), "wall_s": time.perf_counter() - t0,
                                                  "outcome": INTERRUPTED, "message": str(exc), **memory_record(device)})
            raise
        except KeyboardInterrupt:
            write_status(d, INTERRUPTED, attempt={"end_utc": utc_now(), "wall_s": time.perf_counter() - t0,
                                                  "outcome": "aborted", "message": "second Ctrl+C: aborted without a final "
                                                  "checkpoint; resumes from the last periodic checkpoint"})
            raise
        except Exception as exc:                        # recorded, never hidden
            write_status(d, FAILED, attempt={"end_utc": utc_now(), "wall_s": time.perf_counter() - t0, "outcome": FAILED,
                                             "error": f"{type(exc).__name__}: {exc}",
                                             "traceback": traceback.format_exc(limit=20), **memory_record(device)})
            self.log(f"    FAILED: {type(exc).__name__}: {exc}")
            return FAILED
        wall = time.perf_counter() - t0
        status = read_status(d)
        result = {"schema": "il_memoir/result-v1", "job_id": job.job_id, "type": job.type, "labels": job.labels,
                  "completed_utc": utc_now(), "device": device, "code_sha256": code["sha256"],
                  "attempts": len(status.get("attempts", [])), "this_attempt_wall_s": wall,
                  "memory": memory_record(device), **result}
        write_json_atomic(d / "result.json", result)
        write_status(d, COMPLETED, attempt={"end_utc": utc_now(), "wall_s": wall, "outcome": COMPLETED,
                                            **memory_record(device)})
        return COMPLETED

    # -------------------------------------------------------------------- run
    def resolve_device(self, previous: Optional[Dict[str, Any]]) -> DeviceDecision:
        frozen = previous.get("device", {}).get("device") if previous else None
        pilot = read_pilot_choice(self.pilot_path)
        decision = resolve_run_device(self.device_request, frozen=frozen, pilot=pilot,
                                      require_pilot_for_auto=(self.suite.kind == "final" and frozen is None))
        if frozen is not None and decision.device != frozen:
            decision.message += f" (explicit override of the frozen device {frozen}; recorded per run)"
        return decision

    def status_report(self) -> Dict[str, Any]:
        """Statuses without launching or preparing anything (``--status``)."""
        entries = {}
        man = ds.load_manifest(self.data_root, self.suite.data_id)
        for key, item in data_items(self.suite).items():
            e = man["items"].get(key)
            if e is not None:
                entries[key] = e
        states = {}
        for j in self.jobs:
            if all(k in entries for k in j.data_inputs.values()):
                states[j.job_id] = self.job_state(j, entries)
            else:
                states[j.job_id] = (PLANNED, "data not prepared")
        counts: Dict[str, int] = {}
        for st, _ in states.values():
            counts[st] = counts.get(st, 0) + 1
        return {"suite": self.suite.name, "counts": counts, "jobs": states}

    def run(self, resume: bool = False) -> int:
        suite = self.suite
        self.suite_dir.mkdir(parents=True, exist_ok=True)
        with file_lock(self.suite_dir / ".lock", f"suite {suite.name}"):
            previous = read_json(self.manifest_path) if self.manifest_path.exists() else None
            if previous is not None and not resume:
                raise RunnerError(f"suite {suite.name!r} was already launched ({self.manifest_path}); add --resume to continue")
            if previous is not None and previous.get("config_sha256") != suite.sha256 and not self.allow_config_change:
                raise RunnerError(
                    f"the configuration of suite {suite.name!r} changed since its manifest was frozen "
                    f"({previous.get('config_sha256')[:12]} -> {suite.sha256[:12]}). Restore the frozen file, or pass "
                    "--allow-config-change: jobs whose settings changed get new identifiers and run, and the change is "
                    "recorded as a manifest revision.")
            seed_change = self._seed_change(previous) if previous is not None else None
            if seed_change and not self.allow_config_change:
                raise RunnerError(
                    f"the seeds of suite {suite.name!r} changed since its manifest was frozen ({seed_change}). Restore "
                    f"{suite.seed_manifest.path}, or pass --allow-config-change to add the new seeds' jobs; the change is "
                    "recorded as a manifest revision.")
            self.decision = self.resolve_device(previous)
            self.log(self.decision.message)
            threads = set_cpu_threads(int(suite.runner["cpu_threads"]))
            entries = self.ensure_data()
            states = self.states(entries)
            stale = {jid: reason for jid, (st, reason) in states.items() if st == "stale"}
            if stale and not self.invalidate_changed:
                self.write_status_table(states)
                self.log(f"{len(stale)} job(s) are stale; nothing was launched:")
                for jid, reason in stale.items():
                    self.log(f"  {jid}: {reason}")
                self.log("Rerun with --invalidate-changed to move them to runs/invalidated/ and recompute them.")
                return EXIT_STALE
            for jid, reason in stale.items():
                self._archive(jid, reason)
                states[jid] = (PLANNED, "")
            missing_dirs = self.missing_directories()
            for jid, what in missing_dirs.items():
                self.log(f"WARNING: the directory of job {jid} is missing although its outcome was recorded ({what}). "
                         "Job directories are invalidated by the runner, never deleted; the audit reports this. "
                         "The job is run again.")
            manifest = self._manifest(previous, entries, seed_change)
            write_json_atomic(self.manifest_path, manifest)
            write_json_atomic(self.suite_dir / "seed_manifest.json", seed_manifest_record(suite, self.jobs))
            hw = hardware_record(self.decision.device)
            stamp = utc_stamp()
            write_json_atomic(self.suite_dir / "hardware" / f"{host_fingerprint()['hostname']}-{stamp}.json", hw)
            JsonlLog(self.suite_dir / "launches.jsonl").append({
                "utc": utc_now(), "argv": self.argv, "device": self.decision.to_dict(), "torch_threads": threads,
                "code_sha256": manifest["code"]["package"]["sha256"], "dependencies": self._deps, "git": manifest["git"],
                "counts_before": {s: sum(1 for st, _ in states.values() if st == s) for s in {st for st, _ in states.values()}},
                "invalidated": sorted(stale),
                "missing_directories": missing_dirs,
            })
            device = self.decision.device
            blocked: Dict[str, str] = {}
            outcome = EXIT_OK
            self.write_status_table(states)
            n = len(self.jobs)
            with self.stop.installed():
                for k, job in enumerate(self.jobs, 1):
                    st, _ = states[job.job_id]
                    tag = f"[{k}/{n}] {job.job_id}"
                    if self.stop.requested():
                        self.write_status_table(states, blocked)
                        self.log(f"stop requested: no further job is started ({n - k + 1} not visited). "
                                 "Resume with the same command and --resume.")
                        return EXIT_INTERRUPTED
                    if st == COMPLETED:
                        now, reason = self.job_state(job, entries)      # a parent rerun in this launch changes its inputs
                        if now == COMPLETED:
                            self.log(f"{tag}: completed (cached)")
                            continue
                        self._archive(job.job_id, reason or "parent rerun in this launch")
                        st = PLANNED
                    if st == FAILED and not self.retry_failed:
                        self.log(f"{tag}: failed earlier (kept; --retry-failed reruns it)")
                        outcome = EXIT_FAILED
                        continue
                    missing = [dep for dep in job.deps if states[dep][0] != COMPLETED]
                    if missing:
                        blocked[job.job_id] = "dependency not completed: " + ", ".join(missing)
                        self.log(f"{tag}: blocked ({blocked[job.job_id]})")
                        outcome = EXIT_FAILED
                        continue
                    self.log(f"{tag}: {'resuming' if st in (INTERRUPTED, RUNNING) else 'running'} on {device}")
                    try:
                        new = self.run_job(job, entries, device)
                    except JobInterrupted as exc:
                        states[job.job_id] = (INTERRUPTED, str(exc))
                        self._journal(job.job_id, INTERRUPTED, message=str(exc))
                        self.write_status_table(states, blocked)
                        self.log(f"{tag}: interrupted ({exc}). Resume with the same command and --resume.")
                        return EXIT_INTERRUPTED
                    except KeyboardInterrupt:
                        states[job.job_id] = (INTERRUPTED, "aborted")
                        self._journal(job.job_id, "aborted")
                        self.write_status_table(states, blocked)
                        self.log(f"{tag}: aborted. Resume with the same command and --resume.")
                        return EXIT_INTERRUPTED
                    self._journal(job.job_id, new)
                    states[job.job_id] = (new, "")
                    if new == FAILED:
                        outcome = EXIT_FAILED
                    self.write_status_table(states, blocked)
            table = self.write_status_table(states, blocked)
            self.log("suite " + suite.name + ": " + ", ".join(f"{v} {k}" for k, v in sorted(table["counts"].items())))
            return outcome
