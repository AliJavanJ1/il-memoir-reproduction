"""Runner mechanics (INTERFACE.md, Prompt 3C): manifest first, caching, resume, stale code, failures, locks, devices.

All suites here are TEST FIXTURES with tiny budgets (``tests/fixtures/runner_tiny.yaml``:
the finite four-rooms maze, every job type) run in temporary directories.
"""
from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from il_memoir import runner as runner_mod
from il_memoir import train
from il_memoir.checks import compare_runs
from il_memoir.experiment import load_suite_config
from il_memoir.hardware import DeviceError, resolve_run_device
from il_memoir.logging import JsonlLog, LockError, StopController, file_lock, read_json, read_status, write_json_atomic
from il_memoir.runner import EXIT_FAILED, EXIT_INTERRUPTED, EXIT_OK, EXIT_STALE, Runner, RunnerError

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "tests" / "fixtures" / "runner_tiny.yaml"
QUIET = lambda m: None          # noqa: E731


def _copy(run, dest):
    shutil.copytree(run["runs"], dest / "runs")
    return dest / "runs"


def _attempts(runs, manifest):
    return {j["job_id"]: len(read_json(runs / "jobs" / j["job_id"] / "status.json")["attempts"]) for j in manifest["jobs"]}


def test_manifest_is_written_before_the_first_job(tmp_path, monkeypatch):
    seen = []
    original = runner_mod.Runner.run_job

    def spy(self, job, entries, device):
        seen.append((self.manifest_path.exists(), (self.suite_dir / "seed_manifest.json").exists()))
        return original(self, job, entries, device)

    monkeypatch.setattr(runner_mod.Runner, "run_job", spy)
    suite = load_suite_config(CFG)
    assert Runner(suite, tmp_path / "runs", tmp_path / "data", device="cpu", log=QUIET).run(resume=True) == EXIT_OK
    assert seen and all(a and b for a, b in seen)
    man = read_json(tmp_path / "runs" / "suites" / suite.name / "manifest.json")
    assert man["device"]["device"] == "cpu" and man["seeds"] == suite.seeds and man["test_fixture"] is True
    assert len(man["jobs"]) == 6 and set(man["code"]["scopes"]) == {"fb_pretrain", "adapt", "fpg", "eval", "eval_expert"}


def test_completed_jobs_are_reused_not_rerun(runner_run, tmp_path):
    runs = _copy(runner_run, tmp_path)
    before = _attempts(runs, runner_run["manifest"])
    suite = runner_run["suite"]
    assert Runner(suite, runs, runner_run["data"], device="auto", log=QUIET).run(resume=True) == EXIT_OK
    assert _attempts(runs, runner_run["manifest"]) == before == {k: 1 for k in before}
    with pytest.raises(RunnerError):                       # a launched suite needs --resume
        Runner(suite, runs, runner_run["data"], device="cpu", log=QUIET).run(resume=False)


def test_interrupted_jobs_resume_to_the_uninterrupted_result(runner_run, tmp_path):
    suite = runner_run["suite"]
    runs = tmp_path / "runs"
    codes = []
    for _ in range(100):
        stop = StopController()
        stop.stop_after = 5
        code = Runner(suite, runs, runner_run["data"], device="cpu", stop=stop, log=QUIET).run(resume=True)
        codes.append(code)
        if code != EXIT_INTERRUPTED:
            break
        statuses = [read_status(runs / "jobs" / j["job_id"])["status"] for j in runner_run["manifest"]["jobs"]]
        interrupted = [j for j, st in zip(runner_run["manifest"]["jobs"], statuses) if st == "interrupted"]
        # either one job stopped at a safe point, or the stop came with a job's last update: that job completed
        # and no further job started (the jobs run in manifest order, so only a prefix is completed)
        assert len(interrupted) <= 1
        done = [st == "completed" for st in statuses]
        assert done == sorted(done, reverse=True)
        if interrupted:
            assert statuses.index("interrupted") == sum(done)
            if interrupted[0]["type"] in ("fpg", "fb_pretrain"):
                assert (runs / "jobs" / interrupted[0]["job_id"] / "checkpoints" / "resume.pt").exists()
        else:
            assert all(st in ("completed", "planned") for st in statuses)
    assert codes[-1] == EXIT_OK and codes.count(EXIT_INTERRUPTED) >= 3
    att = _attempts(runs, runner_run["manifest"])
    assert max(att.values()) > 1
    cmp = compare_runs(runner_run["runs"], runs, runner_run["manifest"], atol=0.0)
    assert all(cmp["checks"].values()), {k: v for k, v in cmp["max_abs_diff"].items() if v}


def test_changed_code_stops_the_launch_until_jobs_are_invalidated(runner_run, tmp_path, monkeypatch):
    runs = _copy(runner_run, tmp_path)
    suite = runner_run["suite"]
    before = _attempts(runs, runner_run["manifest"])
    original = runner_mod.Runner.code

    def changed(self, job_type):
        rec = dict(original(self, job_type))
        if job_type == "fpg":
            rec["sha256"] = "0" * 64
            rec["files"] = dict(rec["files"], **{"il_memoir/fpg.py": "1" * 64})
        return rec

    monkeypatch.setattr(runner_mod.Runner, "code", changed)
    r = Runner(suite, runs, runner_run["data"], device="cpu", log=QUIET)
    assert r.run(resume=True) == EXIT_STALE
    assert _attempts(runs, runner_run["manifest"]) == before            # nothing was launched
    status = read_json(runs / "suites" / suite.name / "status.json")["jobs"]
    stale = {k for k, v in status.items() if v["status"] == "stale"}
    fpg_ids = {j["job_id"] for j in runner_run["manifest"]["jobs"] if j["type"] == "fpg"}
    fpg_eval = {j["job_id"] for j in runner_run["manifest"]["jobs"] if j["deps"] and set(j["deps"]) & fpg_ids}
    assert stale == fpg_ids | fpg_eval
    assert "il_memoir/fpg.py" in next(v["reason"] for k, v in status.items() if k in fpg_ids)
    r = Runner(suite, runs, runner_run["data"], device="cpu", invalidate_changed=True, log=QUIET)
    assert r.run(resume=True) == EXIT_OK
    archived = sorted(p.name.split("__")[0] for p in (runs / "invalidated").iterdir())
    assert set(archived) == stale
    assert all((p / "invalidated.json").exists() and (p / "result.json").exists() for p in (runs / "invalidated").iterdir())
    after = _attempts(runs, runner_run["manifest"])
    assert all(after[k] == 1 for k in after)                            # rerun jobs start a new record
    fb = [j["job_id"] for j in runner_run["manifest"]["jobs"] if j["type"] == "fb_pretrain"]
    assert all(read_json(runs / "jobs" / k / "status.json")["attempts"][0] == read_json(runner_run["runs"] / "jobs" / k / "status.json")["attempts"][0] for k in fb)


def test_failed_jobs_are_recorded_block_dependents_and_are_retried_only_on_request(tmp_path, monkeypatch):
    suite = load_suite_config(CFG)
    real = train.EXECUTORS["fb_pretrain"]

    def broken(ctx, code):
        raise RuntimeError("injected failure (test fixture)")

    monkeypatch.setitem(train.EXECUTORS, "fb_pretrain", broken)
    runs, data = tmp_path / "runs", tmp_path / "data"
    assert Runner(suite, runs, data, device="cpu", log=QUIET).run(resume=True) == EXIT_FAILED
    table = read_json(runs / "suites" / suite.name / "status.json")["jobs"]
    by_type = {}
    for jid, rec in table.items():
        by_type.setdefault(rec["type"], set()).add(rec["status"])
    assert by_type["fb_pretrain"] == {"failed"} and by_type["adapt"] == {"blocked"}
    assert by_type["fpg"] == {"completed"} and by_type["eval_expert"] == {"completed"}
    fb_id = next(j for j, r in table.items() if r["type"] == "fb_pretrain")
    st = read_json(runs / "jobs" / fb_id / "status.json")
    assert "injected failure" in st["attempts"][-1]["error"] and "Traceback" in st["attempts"][-1]["traceback"]
    monkeypatch.setitem(train.EXECUTORS, "fb_pretrain", real)
    assert Runner(suite, runs, data, device="cpu", log=QUIET).run(resume=True) == EXIT_FAILED      # not retried
    assert len(read_json(runs / "jobs" / fb_id / "status.json")["attempts"]) == 1
    assert Runner(suite, runs, data, device="cpu", retry_failed=True, log=QUIET).run(resume=True) == EXIT_OK
    st = read_json(runs / "jobs" / fb_id / "status.json")
    assert [a["outcome"] for a in st["attempts"]] == ["failed", "completed"]


def test_a_changed_configuration_is_refused_unless_allowed(runner_run, tmp_path):
    runs = _copy(runner_run, tmp_path)
    raw = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    raw["base"] = str(ROOT / "configs" / "smoke.yaml")
    raw["description"] = "TEST FIXTURE: changed description"
    p = tmp_path / "runner_tiny.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    suite = load_suite_config(p)
    with pytest.raises(RunnerError):
        Runner(suite, runs, runner_run["data"], device="cpu", log=QUIET).run(resume=True)
    assert Runner(suite, runs, runner_run["data"], device="cpu", allow_config_change=True, log=QUIET).run(resume=True) == EXIT_OK
    man = read_json(runs / "suites" / suite.name / "manifest.json")
    assert len(man["revisions"]) == 2 and man["config_sha256"] == suite.sha256


def test_lock_refuses_a_second_live_runner_and_takes_over_a_dead_one(tmp_path):
    lock = tmp_path / ".lock"
    write_json_atomic(lock, {"pid": os.getppid(), "host": socket.gethostname(), "utc": "x", "what": "other"})
    with pytest.raises(LockError):
        with file_lock(lock, "suite test"):
            pass
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    write_json_atomic(lock, {"pid": proc.pid, "host": socket.gethostname(), "utc": "x", "what": "dead"})
    with file_lock(lock, "suite test"):
        assert read_json(lock)["pid"] == os.getpid()
    assert not lock.exists()


def test_device_policy():
    none = {"login_node": False, "in_allocation": False}
    assert resolve_run_device("cpu", cuda_ok=True, sched=none).device == "cpu"
    d = resolve_run_device("cuda", cuda_ok=False, sched=none)
    assert d.device == "cpu" and d.fallback and "WARNING" in d.message
    assert resolve_run_device("cuda", cuda_ok=True, sched=none).device == "cuda"
    with pytest.raises(DeviceError):                          # frozen CUDA suite, no CUDA now: never silent
        resolve_run_device("auto", frozen="cuda", cuda_ok=False, sched=none)
    assert resolve_run_device("auto", frozen="cpu", cuda_ok=True, sched=none).device == "cpu"
    pilot = {"device": "cuda", "same_host": True, "path": "runs/benchmark.json"}
    assert resolve_run_device("auto", pilot=pilot, cuda_ok=True, sched=none).source == "pilot"
    other = dict(pilot, same_host=False, host={"hostname": "cluster"})
    with pytest.raises(DeviceError):
        resolve_run_device("auto", pilot=other, cuda_ok=True, sched=none, require_pilot_for_auto=True)
    with pytest.raises(DeviceError):
        resolve_run_device("auto", cuda_ok=True, sched=none, require_pilot_for_auto=True)
    login = {"login_node": True, "in_allocation": False}
    assert resolve_run_device("auto", cuda_ok=True, sched=login).device == "cpu"
    with pytest.raises(DeviceError):
        resolve_run_device("cuda", cuda_ok=True, sched=login)
    alloc = {"login_node": False, "in_allocation": True}
    assert resolve_run_device("auto", cuda_ok=True, sched=alloc).device == "cuda"


def test_a_final_suite_needs_the_pilot_choice_or_an_explicit_device(tmp_path):
    suite = load_suite_config(ROOT / "configs" / "core.yaml")
    r = Runner(suite, tmp_path / "runs", tmp_path / "data", device="auto", log=QUIET)
    with pytest.raises(DeviceError):
        r.resolve_device(None)


@pytest.mark.skipif(os.name == "nt", reason="sends a real SIGINT to a child process (POSIX); Windows uses the same handler")
def test_ctrl_c_saves_a_checkpoint_and_the_same_command_resumes(tmp_path):
    raw = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    raw["base"] = str(ROOT / "configs" / "smoke.yaml")
    raw["suite"] = "ctrl_c_fixture"
    raw["data"] = {"id": "ctrl-c-fixture"}
    raw["methods"] = {"fb": {"finite": {"settings": {"n_updates": 1500}, "snapshot_every": 750}}}
    cfg = tmp_path / "ctrl_c.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "scripts" / "run_suite.py"), "--config", str(cfg), "--device", "cpu", "--resume",
           "--runs", str(tmp_path / "runs"), "--data-root", str(tmp_path / "data")]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    fb_dir = None
    t0 = time.time()
    for line in proc.stdout:
        if "fb-finite" in line and "running on" in line:
            fb_dir = tmp_path / "runs" / "jobs" / line.split("] ")[1].split(":")[0]
            break
        assert time.time() - t0 < 120
    assert fb_dir is not None
    while not (fb_dir / "log.jsonl").exists() and time.time() - t0 < 120:
        time.sleep(0.05)
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == EXIT_INTERRUPTED, out
    st = read_json(fb_dir / "status.json")
    assert st["status"] == "interrupted" and (fb_dir / "checkpoints" / "resume.pt").exists()
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    assert res.returncode == EXIT_OK, res.stdout[-2000:]
    st = read_json(fb_dir / "status.json")
    assert st["status"] == "completed" and [a["outcome"] for a in st["attempts"]] == ["interrupted", "completed"]
    assert "resumed FB pretraining at update" in res.stdout


# ----------------------------------------------------------------------------- seeds, deleted runs, stop, logs
def _seeds_file(tmp_path, **changes):
    raw = yaml.safe_load((ROOT / "configs" / "seeds.yaml").read_text(encoding="utf-8"))
    for key, value in changes.items():
        if key.startswith("reference_"):
            raw["reference_seeds"][key[len("reference_"):]] = value
        else:
            raw[key] = value
    p = tmp_path / "seeds.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def _copy_all(run, dest):
    shutil.copytree(run["runs"], dest / "runs")
    shutil.copytree(run["data"], dest / "data")
    return dest / "runs", dest / "data"


def test_a_removed_seed_is_refused_even_with_allow_config_change(runner_run, tmp_path):
    runs, data = _copy_all(runner_run, tmp_path)
    suite = load_suite_config(CFG, seeds_path=_seeds_file(tmp_path, smoke_seed=4244))
    assert suite.seeds == [4244]
    before = read_json(runs / "suites" / suite.name / "manifest.json")
    with pytest.raises(RunnerError, match="no longer in the seed manifest"):
        Runner(suite, runs, data, device="cpu", allow_config_change=True, log=QUIET).run(resume=True)
    assert read_json(runs / "suites" / suite.name / "manifest.json") == before     # nothing rewritten
    from il_memoir.audit import audit
    rep = audit(runs, None, kinds=("smoke",), seeds_path=tmp_path / "seeds.yaml")
    assert any("[suite_seeds]" in e for e in rep.errors)                           # and the audit says so


def test_a_changed_reference_seed_needs_allow_config_change_and_is_recorded(runner_run, tmp_path):
    from il_memoir.audit import audit

    runs, data = _copy_all(runner_run, tmp_path)
    seeds = _seeds_file(tmp_path, reference_smoke=4245)
    suite = load_suite_config(CFG, seeds_path=seeds)
    with pytest.raises(RunnerError, match="seeds of suite"):
        Runner(suite, runs, data, device="cpu", log=QUIET).run(resume=True)
    assert Runner(suite, runs, data, device="cpu", allow_config_change=True, log=QUIET).run(resume=True) == EXIT_OK
    man = read_json(runs / "suites" / suite.name / "manifest.json")
    assert len(man["revisions"]) == 2 and "reference seed 4243 -> 4245" in man["revisions"][-1]["reason"]
    assert man["revisions"][-1]["seeds"] == [4242] and man["revisions"][-1]["removed_jobs"] == []
    assert not [e for e in audit(runs, None, kinds=("smoke",), seeds_path=seeds).errors if "seed" in e]
    assert any("[reference_seed]" in e for e in audit(runs, None, kinds=("smoke",)).errors)   # against configs/seeds.yaml


def test_a_stop_request_prevents_any_further_job_start(runner_run, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(runner_run["data"], data)
    stop = StopController()
    stop.request("test fixture")
    suite = runner_run["suite"]
    assert Runner(suite, tmp_path / "runs", data, device="cpu", stop=stop, log=QUIET).run(resume=True) == EXIT_INTERRUPTED
    assert (tmp_path / "runs" / "suites" / suite.name / "manifest.json").exists()
    assert not (tmp_path / "runs" / "jobs").exists() or not any((tmp_path / "runs" / "jobs").iterdir())


def test_the_outcome_journal_records_every_job_outcome(runner_run):
    from il_memoir.logging import read_outcome_journal

    journal = read_outcome_journal(runner_run["runs"])
    for j in runner_run["manifest"]["jobs"]:
        events = journal[j["job_id"]]
        st = read_json(runner_run["runs"] / "jobs" / j["job_id"] / "status.json")
        assert [e["event"] for e in events] == ["completed"]
        assert events[0]["attempt_start_utc"] == st["attempts"][0]["start_utc"] and events[0]["n_attempts"] == 1


def test_a_deleted_completed_job_directory_is_reported_and_then_rerun(runner_run, tmp_path):
    from il_memoir.analysis import Collection
    from il_memoir.audit import audit

    runs, data = _copy_all(runner_run, tmp_path)
    suite = runner_run["suite"]
    fpg = next(j["job_id"] for j in runner_run["manifest"]["jobs"] if j["type"] == "fpg")
    shutil.rmtree(runs / "jobs" / fpg)
    r = Runner(suite, runs, data, device="cpu", log=QUIET)
    assert set(r.missing_directories()) == {fpg}
    assert Collection(runs, ("smoke",)).jobs[fpg]["status"] == "missing"
    assert any("[job_directory_deleted]" in e and fpg in e for e in audit(runs, None, kinds=("smoke",)).errors)
    lines = []
    assert Runner(suite, runs, data, device="cpu", invalidate_changed=True, log=lines.append).run(resume=True) == EXIT_OK
    assert any(l.startswith("WARNING: the directory of job " + fpg) for l in lines)
    rep = audit(runs, None, kinds=("smoke",))
    assert not rep.errors, rep.errors
    assert any("[attempt_history_complete]" in w and fpg in w for w in rep.warnings)
    launch = JsonlLog(runs / "suites" / suite.name / "launches.jsonl").records()[-1]
    assert fpg in launch["missing_directories"]


def test_a_deleted_failed_run_stays_an_audit_error_after_the_rerun(runner_run, tmp_path, monkeypatch):
    from il_memoir.audit import audit

    data = tmp_path / "data"
    shutil.copytree(runner_run["data"], data)
    runs = tmp_path / "runs"
    suite = runner_run["suite"]
    real = train.EXECUTORS["fpg"]

    def broken(ctx, code):
        raise RuntimeError("injected failure (test fixture)")

    monkeypatch.setitem(train.EXECUTORS, "fpg", broken)
    assert Runner(suite, runs, data, device="cpu", log=QUIET).run(resume=True) == EXIT_FAILED
    fpg = next(j["job_id"] for j in runner_run["manifest"]["jobs"] if j["type"] == "fpg")
    assert not audit(runs, None, kinds=("smoke",)).errors                   # a recorded failure is not an error
    shutil.rmtree(runs / "jobs" / fpg)                                       # ... deleting it is
    monkeypatch.setitem(train.EXECUTORS, "fpg", real)
    assert Runner(suite, runs, data, device="cpu", log=QUIET).run(resume=True) == EXIT_OK
    errors = audit(runs, None, kinds=("smoke",)).errors
    assert any("[failed_attempt_kept]" in e and fpg in e for e in errors), errors


def test_a_torn_last_log_line_is_dropped_and_corruption_elsewhere_raises(tmp_path):
    log = JsonlLog(tmp_path / "log.jsonl")
    log.append({"k": 1})
    log.append({"k": 2})
    with open(log.path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"k": 3, "partial')                     # a crash during an append
    assert [r["k"] for r in log.records()] == [1, 2]
    log.append({"k": 4})
    assert [r["k"] for r in log.records()] == [1, 2, 4]
    assert log.path.read_text(encoding="utf-8").endswith("\n") and log.durable_count() == 3
    log.path.write_text('{"k": 1}\nnot json\n{"k": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        log.records()


def test_only_versions_that_change_numbers_invalidate_jobs(monkeypatch):
    import matplotlib
    import numpy

    from il_memoir.logging import dependency_record

    base = dependency_record()
    monkeypatch.setattr(matplotlib, "__version__", "99.0.0")
    monkeypatch.setattr(yaml, "__version__", "99.0")
    rec = dependency_record()
    assert rec["matplotlib"] == "99.0.0" and rec["sha256"] == base["sha256"]
    monkeypatch.setattr(numpy, "__version__", "0.0.1")
    assert dependency_record()["sha256"] != base["sha256"]


def test_a_configuration_file_edited_after_the_launch_is_an_audit_error(runner_run, tmp_path):
    from il_memoir.audit import audit

    runs, data = _copy_all(runner_run, tmp_path)
    raw = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    raw["base"] = str(ROOT / "configs" / "smoke.yaml")
    raw["suite"] = "edited_fixture"
    cfg = tmp_path / "edited.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert Runner(load_suite_config(cfg), runs, data, device="cpu", log=QUIET).run(resume=True) == EXIT_OK
    assert not [e for e in audit(runs, None, kinds=("smoke",)).errors if "[config_file]" in e]
    raw["description"] = "TEST FIXTURE: edited after the launch"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert any("[config_file]" in e and "edited_fixture" in e for e in audit(runs, None, kinds=("smoke",)).errors)
