"""The full data-to-results path through the INTERFACE scripts, on clearly labelled TEST FIXTURES (Prompt 3C).

prepare_data -> run_suite -> analyze -> audit_results on ``configs/smoke.yaml`` in
temporary directories; the audit must pass on intact runs and fail on tampered
ones; missing runs are listed, never invented; data are never overwritten by
different data; withheld expert actions and the evaluation reward cannot change
what a learner computes; the pilot benchmark writes its records without
touching the laptop hardware record.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from il_memoir import datasets as ds
from il_memoir import train
from il_memoir.analysis import AnalysisError, Collection, analyze, final_tables
from il_memoir.audit import audit
from il_memoir.config import environment_config_from_dict, load_environment_config
from il_memoir.experiment import expand_jobs, load_suite_config
from il_memoir.hardware import read_pilot_choice
from il_memoir.logging import JobContext, JsonlLog, StopController, load_torch, read_json, sha256_file, write_json_atomic
from il_memoir.prepare import DataConflict, prepare_items
from il_memoir.runner import EXIT_FAILED, EXIT_OK, Runner

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"
QUIET = lambda m: None          # noqa: E731


def _script(name):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_data_to_results_through_the_scripts(tmp_path, capsys):
    runs, data, results = tmp_path / "runs", tmp_path / "data", tmp_path / "results"
    assert _script("prepare_data").main(["--config", str(SMOKE), "--data-root", str(data)]) == 0
    assert _script("prepare_data").main(["--config", str(SMOKE), "--data-root", str(data), "--check"]) == 0
    assert _script("run_suite").main(["--config", str(SMOKE), "--device", "cpu", "--resume", "--runs", str(runs),
                                      "--data-root", str(data)]) == 0
    assert _script("run_suite").main(["--config", str(SMOKE), "--status", "--runs", str(runs), "--data-root", str(data)]) == 0
    assert "12 completed" in capsys.readouterr().out
    assert _script("analyze").main(["--runs", str(runs), "--out", str(results)]) == 1          # no final suite: nothing written
    assert not (results / "RESULTS.md").exists()
    assert _script("analyze").main(["--runs", str(runs), "--out", str(results), "--include-kind", "smoke"]) == 0
    text = (results / "RESULTS.md").read_text(encoding="utf-8")
    assert text.startswith("> **TEST FIXTURE")
    assert _script("audit_results").main(["--runs", str(runs), "--results", str(results), "--include-kind", "smoke"]) == 0
    prov = read_json(results / "provenance.json")
    man = read_json(runs / "suites" / "smoke" / "manifest.json")
    evals = {j["job_id"] for j in man["jobs"] if j["type"] in ("eval", "eval_expert")}
    assert evals <= set(prov["jobs"]) and prov["test_fixture"] is True
    for rel, t in prov["tables"].items():
        assert sha256_file(results / rel) == t["sha256"]
    rep = read_json(results / "audit_report.json")
    assert rep["n_errors"] == 0 and sum(rep["checks_run"].values()) > 500


def test_analysis_never_writes_fixtures_into_the_project_results(smoke_run):
    with pytest.raises(AnalysisError):
        analyze(smoke_run["runs"], ROOT / "results", kinds=("smoke",))
    assert not (ROOT / "results" / "RESULTS.md").exists() or "TEST FIXTURE" not in (ROOT / "results" / "RESULTS.md").read_text(encoding="utf-8")


def test_audit_fails_on_tampered_metrics_and_stale_results(smoke_run, tmp_path):
    runs = tmp_path / "runs"
    shutil.copytree(smoke_run["runs"], runs)
    results = tmp_path / "results"
    analyze(runs, results, kinds=("smoke",), figures=False)
    assert not audit(runs, results, kinds=("smoke",)).errors
    ev_job = next(j for j in smoke_run["manifest"]["jobs"] if j["type"] == "eval" and j["labels"]["env"] == "finite")
    path = runs / "jobs" / ev_job["job_id"] / "outputs" / "evaluations.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["metrics"]["js_nats"] = 0.01
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    errors = audit(runs, results, kinds=("smoke",)).errors
    assert any("[output_hash]" in e for e in errors)
    assert any("[final_row_values]" in e for e in errors)
    # an impossible value is caught even when the hashes are made consistent again
    res_path = runs / "jobs" / ev_job["job_id"] / "result.json"
    res = read_json(res_path)
    rows[-1]["metrics"]["success"] = 1.5
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    res["outputs"]["evaluations"]["sha256"] = sha256_file(path)
    write_json_atomic(res_path, res)
    errors = audit(runs, None, kinds=("smoke",)).errors
    assert any("[probability_range]" in e for e in errors)


def test_missing_runs_are_listed_and_never_invented(tmp_path, monkeypatch):
    def broken(ctx, code):
        raise RuntimeError("injected f-PG failure (test fixture)")

    monkeypatch.setitem(train.EXECUTORS, "fpg", broken)
    suite = load_suite_config(SMOKE)
    runs, data = tmp_path / "runs", tmp_path / "data"
    assert Runner(suite, runs, data, device="cpu", log=QUIET).run(resume=True) == EXIT_FAILED
    out = analyze(runs, tmp_path / "results", kinds=("smoke",), figures=False)
    coll = Collection(runs, ("smoke",))
    rows, summary = final_tables(coll, coll.suites[0])
    fpg = [s for s in summary if s["method_label"].startswith("f-PG")]
    assert fpg and all(s["n_completed"] == 0 and s["n_planned"] == 1 for s in fpg)
    assert all("injected f-PG failure" in " ".join(s["seeds_missing"].values()) for s in fpg)
    assert not any(r["method"] == "fpg" for r in rows)
    text = (Path(out["out"]) / "RESULTS.md").read_text(encoding="utf-8")
    assert "Jobs not completed" in text and "injected f-PG failure" in text
    assert not audit(runs, tmp_path / "results", kinds=("smoke",)).errors


def test_prepared_data_are_deterministic_and_never_overwritten_by_different_data(tmp_path):
    suite = load_suite_config(SMOKE)
    a = prepare_items(suite, tmp_path / "a", log=QUIET)
    b = prepare_items(suite, tmp_path / "b", log=QUIET)
    assert [r["content_sha256"] for r in a] == [r["content_sha256"] for r in b]
    again = prepare_items(suite, tmp_path / "a", log=QUIET)
    assert {r["status"] for r in again} == {"verified"}
    man = ds.load_manifest(tmp_path / "a", suite.data_id)
    key = next(k for k in man["items"] if k.startswith("demos/"))
    man["items"][key]["content_sha256"] = "0" * 64
    ds.save_manifest(tmp_path / "a", suite.data_id, man)
    with pytest.raises(DataConflict):
        prepare_items(suite, tmp_path / "a", log=QUIET)
    raw_suite = load_suite_config(SMOKE)
    raw_suite.raw["data"]["n_demos"] = 5           # a different specification under the same data id
    with pytest.raises(DataConflict):
        prepare_items(raw_suite, tmp_path / "b", log=QUIET)


def _final_training_outputs(runs, manifest):
    out = {}
    for j in manifest["jobs"]:
        d = runs / "jobs" / j["job_id"]
        res = read_json(d / "result.json")
        if j["type"] == "fpg":
            last = max(res["outputs"]["policies"], key=lambda o: o["iteration"])
            out[j["job_id"]] = load_torch(d / last["path"])["state_dict"]
        elif j["type"] == "fb_pretrain":
            last = max(res["outputs"]["snapshots"], key=lambda s: s["update"])
            rec = load_torch(d / last["path"])
            out[j["job_id"]] = {**rec["F"], **{"B." + k: v for k, v in rec["B"].items()}}
        elif j["type"] == "adapt":
            lat = read_json(d / res["outputs"]["latents"]["path"])
            out[j["job_id"]] = [(r["er"]["z"], r["dm"]["z"]) for r in lat["records"]]
    return out


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], dict):
            assert all(torch.equal(a[k][n], b[k][n]) for n in a[k]), k
        else:
            assert a[k] == b[k], k


def test_withheld_expert_actions_cannot_change_any_learner(smoke_run, tmp_path):
    """Scrambling the stored expert actions (states unchanged) leaves every training output bit-identical."""
    data_b = tmp_path / "data"
    shutil.copytree(smoke_run["data"], data_b)
    suite = smoke_run["suite"]
    man = ds.load_manifest(data_b, suite.data_id)
    rng = np.random.default_rng(0)
    for key, entry in man["items"].items():
        if entry["kind"] != "demos":
            continue
        arrays, meta = ds.load_arrays(ds.item_path(data_b, suite.data_id, key))
        arrays["actions"] = rng.permutation(arrays["actions"].ravel()).reshape(arrays["actions"].shape)
        arrays["actions"] = (arrays["actions"] + 1) % 5
        path = ds.item_path(data_b, suite.data_id, key)
        path.unlink()
        entry["content_sha256"] = ds.save_arrays(path, arrays, meta)
        entry["file_sha256"] = sha256_file(path)
    ds.save_manifest(data_b, suite.data_id, man)
    runs_b = tmp_path / "runs"
    assert Runner(suite, runs_b, data_b, device="cpu", log=QUIET).run(resume=True) == EXIT_OK
    _same(_final_training_outputs(smoke_run["runs"], smoke_run["manifest"]),
          _final_training_outputs(runs_b, smoke_run["manifest"]))


def test_evaluation_reward_settings_cannot_change_any_learner(smoke_run, tmp_path):
    """The training executors with another reward width, goal radius and success rule compute the same outputs."""
    cfg = load_environment_config()
    raw = cfg.to_dict()
    raw["task"] = {"reward": "gaussian", "reward_width_cells": 1.7, "goal_radius_cells": 0.9, "success": "final"}
    other = environment_config_from_dict(raw)
    assert other.sha256() != cfg.sha256() and other.learner_dict() == cfg.learner_dict()
    suite, runs, man = smoke_run["suite"], smoke_run["runs"], smoke_run["manifest"]
    for j in man["jobs"]:
        if j["type"] not in ("fpg", "fb_pretrain"):
            continue
        job = next(x for x in expand_jobs(suite) if x.job_id == j["job_id"])
        entries = {role: ds.manifest_entry(smoke_run["data"], suite.data_id, key) for role, key in job.data_inputs.items()}
        d = tmp_path / j["job_id"]
        ctx = JobContext(job=job, job_dir=d, data_root=smoke_run["data"], data_id=suite.data_id, env_cfg=other,
                         device="cpu", runner=suite.runner, stop=StopController(), data_entries=entries, log=QUIET)
        res = train.EXECUTORS[j["type"]](ctx, "test")
        ref = read_json(runs / "jobs" / j["job_id"] / "result.json")
        if j["type"] == "fpg":
            got = load_torch(d / res["outputs"]["policies"][-1]["path"])["state_dict"]
            want = load_torch(runs / "jobs" / j["job_id"] / ref["outputs"]["policies"][-1]["path"])["state_dict"]
        else:
            got = load_torch(d / res["outputs"]["snapshots"][-1]["path"])["F"]
            want = load_torch(runs / "jobs" / j["job_id"] / ref["outputs"]["snapshots"][-1]["path"])["F"]
        assert all(torch.equal(got[k], want[k]) for k in got)


def test_pilot_benchmark_fixture_writes_its_records_and_respects_the_laptop_record(tmp_path):
    from il_memoir.pilot import PilotError, run_benchmark

    runs, data = tmp_path / "runs", tmp_path / "data"
    runs.mkdir()
    sentinel = runs / "environment_laptop.json"
    sentinel.write_text('{"sentinel": true}\n', encoding="utf-8")
    rec = run_benchmark(ROOT / "tests" / "fixtures" / "pilot_tiny.yaml", "auto", runs, data, log=QUIET)
    assert rec["complete"] and rec["device_choice"]["device"] in ("cpu", "cuda")
    assert sentinel.read_text(encoding="utf-8") == '{"sentinel": true}\n'
    saved = read_json(runs / "benchmark.json")
    assert saved["complete"] and set(saved["eta"]) == {"core", "compact_ablations", "reduced"}
    for e in saved["eta"].values():
        assert e["overhead"] >= 0.5 and e["proposed_s"] == pytest.approx(e["measured_median_s"] * (1 + e["overhead"]))
        assert e["proposed_range_s"][0] <= e["proposed_s"] <= e["proposed_range_s"][1]
    assert saved["eta"]["compact_ablations"]["n_extra_jobs"] == 21 and saved["eta"]["reduced"]["n_extra_jobs"] == 0
    assert (runs / "PILOT.md").exists() and (runs / "pilot_results" / "RESULTS.md").read_text(encoding="utf-8").startswith("> **PILOT")
    choice = read_pilot_choice(runs / "benchmark.json")
    assert choice["same_host"] and choice["device"] == saved["device_choice"]["device"]
    # a rerun (e.g. after an interruption) re-measures the timing workloads but keeps the pilot's device
    again = run_benchmark(ROOT / "tests" / "fixtures" / "pilot_tiny.yaml", "auto", runs, data, log=QUIET)
    assert again["device_choice"]["reason"].startswith("frozen by the learning pilot's first launch")
    assert again["device_choice"]["device"] == saved["device_choice"]["device"]
    assert again["learning_pilot"]["units_device"] == saved["device_choice"]["device"]
    assert again["learning_pilot"]["jobs_on_other_device_excluded"] == []
    saved = read_json(runs / "benchmark.json")
    saved["host"]["hostname"] = "another-computer"
    write_json_atomic(runs / "benchmark.json", saved)
    assert not read_pilot_choice(runs / "benchmark.json")["same_host"]
    with pytest.raises(PilotError):
        run_benchmark(ROOT / "tests" / "fixtures" / "pilot_tiny.yaml", "auto", runs, data, log=QUIET)


def test_a_missing_data_file_is_restored_only_with_identical_content(tmp_path):
    suite = load_suite_config(SMOKE)
    root = tmp_path / "a"
    prepare_items(suite, root, log=QUIET)
    man = ds.load_manifest(root, suite.data_id)
    key = next(k for k in man["items"] if k.startswith("demos/"))
    path = ds.item_path(root, suite.data_id, key)
    original = dict(man["items"][key])
    path.unlink()
    rep = prepare_items(suite, root, keys=[key], log=QUIET)
    assert rep[0]["status"] == "restored"
    entry = ds.load_manifest(root, suite.data_id)["items"][key]
    assert entry["content_sha256"] == original["content_sha256"] and entry["created_utc"] == original["created_utc"]
    assert "restored_utc" in entry
    path.unlink()
    man = ds.load_manifest(root, suite.data_id)
    man["items"][key]["content_sha256"] = "0" * 64              # recorded data that the code no longer produces
    ds.save_manifest(root, suite.data_id, man)
    with pytest.raises(DataConflict, match="missing"):
        prepare_items(suite, root, keys=[key], log=QUIET)
    assert not path.exists() and ds.load_manifest(root, suite.data_id)["items"][key]["content_sha256"] == "0" * 64


def _rewrite_table(results, rel, edit):
    from il_memoir.analysis import read_csv

    path = results / rel
    rows = read_csv(path)
    fields = list(rows[0])
    edit(rows)
    import csv

    with open(path, "w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    prov = read_json(results / "provenance.json")
    prov["tables"][rel]["sha256"] = sha256_file(path)          # hashes made consistent: only recomputation can tell
    write_json_atomic(results / "provenance.json", prov)


def test_costs_count_every_phase_once_and_the_audit_recomputes_cost_and_threshold_tables(smoke_run, tmp_path):
    from il_memoir.analysis import read_csv

    runs = tmp_path / "runs"
    shutil.copytree(smoke_run["runs"], runs)
    results = tmp_path / "results"
    analyze(runs, results, kinds=("smoke",), figures=False)
    rows = read_csv(results / "tables" / "costs_smoke.csv")
    assert rows
    for row in rows:
        ev = read_json(runs / "jobs" / row["eval_job"] / "result.json")["costs"]
        assert set(ev["wall_s"]) == {"evaluate"}                   # one phase: the attempt total is kept apart
        assert float(row["eval_wall_s"]) == pytest.approx(ev["wall_s"]["evaluate"], rel=1e-12)
    assert not audit(runs, results, kinds=("smoke",)).errors

    def double_fpg_time(rs):
        r = next(r for r in rs if r["method_label"].startswith("f-PG"))
        r["fpg_wall_s"] = repr(2 * float(r["fpg_wall_s"]))

    _rewrite_table(results, "tables/costs_smoke.csv", double_fpg_time)
    errors = audit(runs, results, kinds=("smoke",)).errors
    assert any("[cost_values]" in e and "fpg_wall_s" in e for e in errors) and not any("[table_hash]" in e for e in errors)

    def flip_threshold(rs):
        rs[0]["reached_at_last"] = str(rs[0]["reached_at_last"] != "True")

    _rewrite_table(results, "tables/convergence_smoke.csv", flip_threshold)
    assert any("[threshold_values]" in e for e in audit(runs, results, kinds=("smoke",)).errors)


def test_pilot_unit_costs_use_only_jobs_of_the_chosen_device(runner_run):
    from il_memoir.pilot import units_from_runs

    jobs = {j["job_id"] for j in runner_run["manifest"]["jobs"]}
    skipped = []
    u = units_from_runs(runner_run["runs"], runner_run["manifest"], runner_run["data"], device="cuda", skipped=skipped)
    assert set(skipped) == jobs                                   # every fixture job ran on the CPU
    assert set(u["finite"]) <= {"replay_per_transition", "demos_item", "reference_per_trajectory"}
    skipped = []
    u = units_from_runs(runner_run["runs"], runner_run["manifest"], runner_run["data"], device="cpu", skipped=skipped)
    assert not skipped and {"fb_update", "dm_step", "eval_per_rollout"} <= set(u["finite"])
