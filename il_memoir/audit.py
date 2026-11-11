"""Provenance, seed and metric integrity audit of saved runs and generated results (``scripts/audit_results.py``).

Errors (non-zero exit status) mean that a number in ``results/`` cannot be
traced to an intact saved run, that a planned seed is missing from a
manifest, that a metric is impossible, or that outputs of different code or
configurations are mixed.  Warnings report facts a reader must know (jobs not
completed, mixed devices) without invalidating the numbers.

Checks
------
1. **Seeds and configuration.** Each suite's seeds and reference seed equal
   the frozen seed manifest (``configs/seeds.yaml``) for its kind, and no seed
   of an earlier manifest revision was dropped; every (condition, method) of
   the manifest is planned for every seed; ``seed_manifest.json`` lists every
   job; the suite configuration file still hashes to the manifest's
   ``config_sha256``; a revision that removed planned jobs is reported.
2. **Identity and history.** For every job directory: ``job.json`` equals the
   manifest's job, the identity hashes to the recorded SHA-256 and to the
   identifier suffix.  A planned job without a directory is an error when the
   outcome journal (``runs/suites/*/outcomes.jsonl``) recorded an outcome for
   it (the directory was deleted instead of invalidated), a warning otherwise
   (never run); a recorded failed attempt missing from the job's attempt
   history is an error (a failed run must never disappear).
3. **Outputs.** Completed jobs have ``result.json``; every output file recorded
   in it has its SHA-256; every evaluation row names a checkpoint whose
   recorded hash matches the training job's output.
4. **Code.** Completed jobs of one type share one code-scope hash per suite,
   and it equals the current code (otherwise the results cannot be reproduced
   with the files in the folder: rerun the affected jobs).
5. **Metrics and counts.** JS in [0, log 2], probabilities in [0, 1], returns in
   [0, Z_H], gaps >= 0, entropies in [0, log |A|], finite values; f-PG
   transitions = iterations x trajectories x H = the budget; FB updates and DM
   steps as configured; evaluation transitions = checkpoints x rollouts x H.
6. **Results.** ``provenance.json`` hashes match every table, figure and
   ``RESULTS.md``; every job it cites still has the recorded result hash; every
   completed evaluation is cited (no stale results); the per-run table is
   recomputed independently from the raw evaluation rows, the summary means
   and sample standard deviations from that table, the cost table from the
   ``result.json`` files (wall-time phases must be the known disjoint ones)
   and the threshold table from the raw evaluation rows; no test fixture or
   pilot run appears in final results.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .analysis import ALL_METRICS, method_label, read_csv
from .config import load_environment_config
from .experiment import load_seed_manifest, load_suite_config
from .logging import (TERMINAL_EVENTS, JsonlLog, PROJECT_ROOT, code_scope, current_outcomes, read_json, read_outcome_journal,
                      read_status, sha256_file, sha256_json, utc_now, write_json_atomic)

LOG2 = math.log(2.0)
TOL = 1e-9


class AuditReport:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.checks: Dict[str, int] = defaultdict(int)

    def check(self, name: str, ok: bool, message: str, warn: bool = False) -> bool:
        self.checks[name] += 1
        if not ok:
            (self.warnings if warn else self.errors).append(f"[{name}] {message}")
        return ok

    def to_dict(self) -> Dict[str, Any]:
        return {"utc": utc_now(), "n_errors": len(self.errors), "n_warnings": len(self.warnings),
                "errors": self.errors, "warnings": self.warnings, "checks_run": dict(self.checks)}


def _walk_outputs(obj: Any, found: List[Dict[str, str]]) -> None:
    if isinstance(obj, dict):
        if isinstance(obj.get("path"), str) and isinstance(obj.get("sha256"), str):
            found.append(obj)
        for v in obj.values():
            _walk_outputs(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _walk_outputs(v, found)


def _in(x: Optional[float], lo: float, hi: float) -> bool:
    return x is None or (isinstance(x, (int, float)) and math.isfinite(x) and lo - TOL <= x <= hi + TOL)


def audit_metrics(rep: AuditReport, job_id: str, row: Dict[str, Any], z_h: float, n_actions: int) -> None:
    m = row["metrics"]
    for k, v in m.items():
        if isinstance(v, float):
            rep.check("metric_finite", math.isfinite(v), f"{job_id} {row['checkpoint']}: {k} = {v} is not finite")
    rep.check("js_range", _in(m.get("js_nats"), 0.0, LOG2), f"{job_id} {row['checkpoint']}: js_nats {m.get('js_nats')} outside [0, log 2]")
    if "js_sa_nats" in m:
        rep.check("js_range", _in(m.get("js_sa_nats"), 0.0, LOG2), f"{job_id}: js_sa_nats outside [0, log 2]")
    for k in ("success", "final_in_goal", "mc_success", "mc_final_in_goal", "mc_post_hit_goal_region_frac"):
        rep.check("probability_range", _in(m.get(k), 0.0, 1.0), f"{job_id} {row['checkpoint']}: {k} = {m.get(k)} outside [0, 1]")
    for k in ("return", "mc_return_mean"):
        rep.check("return_range", _in(m.get(k), 0.0, z_h), f"{job_id} {row['checkpoint']}: {k} = {m.get(k)} outside [0, Z_H = {z_h:.4f}]")
    if m.get("gap") is not None:
        rep.check("gap_nonnegative", m["gap"] >= -1e-8, f"{job_id} {row['checkpoint']}: negative gap {m['gap']}")
    rep.check("entropy_range", _in(m.get("policy_entropy_visited_nats"), 0.0, math.log(n_actions)),
              f"{job_id}: policy entropy outside [0, log |A|]")
    for k in ("kl_dq_nats", "kl_qd_nats", "kl_dq_nats_smoothed", "kl_qd_nats_smoothed"):
        if m.get(k) is not None:
            rep.check("kl_nonnegative", m[k] >= -1e-9, f"{job_id}: {k} negative")


def audit_suite_config(rep: AuditReport, man: Dict[str, Any], seeds, seeds_path: Optional[Path]) -> None:
    """Seeds, reference seed, revisions and configuration file of one suite against the frozen files."""
    name, kind = man["suite"], man["kind"]
    expected = seeds.seeds_for(kind)
    rep.check("suite_seeds", list(man["seeds"]) == expected,
              f"suite {name}: seeds {man['seeds']} differ from the frozen seed manifest {expected} for kind {kind}")
    rep.check("reference_seed", man.get("reference_seed") == seeds.reference_seed(kind),
              f"suite {name}: reference seed {man.get('reference_seed')} differs from the seed manifest "
              f"({seeds.reference_seed(kind)})")
    rec_sm = man.get("seed_manifest", {}).get("sha256")
    rep.check("seed_manifest_file", rec_sm == seeds.sha256,
              f"suite {name}: {seeds.path} changed after the suite's last launch (recorded {str(rec_sm)[:12]}, now "
              f"{seeds.sha256[:12]})", warn=list(man["seeds"]) == expected and man.get("reference_seed") == seeds.reference_seed(kind))
    ever = set()
    for r in man.get("revisions", []):
        ever |= {int(x) for x in r.get("seeds", [])}
    dropped = sorted(ever - {int(x) for x in man["seeds"]})
    rep.check("seeds_never_dropped", not dropped, f"suite {name}: seed(s) {dropped} of an earlier revision are no longer planned")
    for r in man.get("revisions", [])[1:]:
        removed = r.get("removed_jobs") or []
        rep.check("revision_removed_jobs", not removed,
                  f"suite {name}: revision of {r.get('utc')} ({r.get('reason')}) removed {len(removed)} planned job(s) "
                  f"from the plan: {', '.join(removed[:6])}{' ...' if len(removed) > 6 else ''}", warn=True)
    path = PROJECT_ROOT / man["config_path"]
    try:
        now = load_suite_config(path, seeds_path).sha256
    except Exception as exc:                       # noqa: BLE001 - any failure to load the file is reported
        rep.check("config_file", False, f"suite {name}: configuration {man['config_path']} cannot be read ({exc})")
        return
    rep.check("config_file", now == man["config_sha256"],
              f"suite {name}: {man['config_path']} changed after the suite was launched (manifest "
              f"{man['config_sha256'][:12]}, file {now[:12]}); the results no longer match the configuration file")


def audit_history(rep: AuditReport, runs_root: Path, man: Dict[str, Any], journal: Dict[str, List[Dict[str, Any]]]) -> None:
    """Planned jobs without a directory, and recorded outcomes missing from the job directories."""
    for j in man["jobs"]:
        jid = j["job_id"]
        d = runs_root / "jobs" / jid
        events = [e for e in current_outcomes(journal.get(jid, [])) if e["event"] in TERMINAL_EVENTS]
        seed = j["labels"].get("seed")
        if not (d / "job.json").exists():
            if events:
                rep.check("job_directory_deleted", False,
                          f"{jid} (seed {seed}): the directory is missing although its outcome '{events[-1]['event']}' was "
                          f"recorded at {events[-1]['utc']}; job directories are invalidated by the runner, never deleted")
            else:
                rep.check("job_not_run", False, f"{jid} (seed {seed}, {j['type']}): planned, not run yet", warn=True)
            continue
        attempts = read_status(d).get("attempts", [])
        n_failed = sum(1 for e in events if e["event"] == "failed")
        failed_kept = sum(1 for a in attempts if a.get("outcome") == "failed")
        rep.check("failed_attempt_kept", n_failed <= failed_kept,
                  f"{jid} (seed {seed}): {n_failed} failed attempt(s) were recorded but the job's history keeps "
                  f"{failed_kept}; a failed run was deleted")
        rep.check("attempt_history_complete", len(events) <= len(attempts),
                  f"{jid}: {len(events)} recorded outcome(s) but {len(attempts)} attempt(s) in the job directory "
                  "(the directory was replaced)", warn=True)


def audit_runs(rep: AuditReport, runs_root: Path, manifests: List[Dict[str, Any]], seeds_path: Optional[Path] = None) -> None:
    seeds = load_seed_manifest(seeds_path)
    journal = read_outcome_journal(runs_root)
    code_now: Dict[str, str] = {}
    for man in manifests:
        name = man["suite"]
        expected = seeds.seeds_for(man["kind"])
        audit_suite_config(rep, man, seeds, seeds_path)
        audit_history(rep, runs_root, man, journal)
        cells: Dict[tuple, set] = defaultdict(set)
        for j in man["jobs"]:
            lab = j["labels"]
            rep.check("job_seed_planned", int(lab["seed"]) in expected, f"{j['job_id']}: seed {lab['seed']} is not a suite seed")
            key = (j["type"], lab.get("env"), lab.get("layout"), lab.get("task"), lab.get("divergence"),
                   lab.get("fpg_variant"), lab.get("fb_variant"))
            cells[key].add(int(lab["seed"]))
        for key, got in cells.items():
            rep.check("all_seeds_planned", sorted(got) == sorted(expected),
                      f"suite {name}: {key} is planned for seeds {sorted(got)}, not for every seed {expected}")
        sm_path = runs_root / "suites" / name / "seed_manifest.json"
        if rep.check("seed_manifest_exists", sm_path.exists(), f"suite {name}: {sm_path} missing"):
            sm = read_json(sm_path)
            rep.check("seed_manifest_jobs", set(sm["jobs"]) == {j["job_id"] for j in man["jobs"]},
                      f"suite {name}: seed_manifest.json does not list exactly the planned jobs")
        cfg = load_environment_config(PROJECT_ROOT / man["environment_config"]["path"])
        rep.check("environment_config_unchanged", cfg.sha256() == man["environment_config"]["sha256"],
                  f"suite {name}: {man['environment_config']['path']} changed after the suite was launched")
        z_h = (1 - cfg.gamma ** cfg.horizon) / (1 - cfg.gamma)
        n_actions = len(cfg.action_names)
        horizon = cfg.horizon
        type_code: Dict[str, set] = defaultdict(set)
        type_deps: Dict[str, set] = defaultdict(set)
        devices = set()
        outputs_of: Dict[str, Dict[str, str]] = {}
        for j in man["jobs"]:
            jid = j["job_id"]
            d = runs_root / "jobs" / jid
            if not (d / "job.json").exists():
                continue
            rec = read_json(d / "job.json")
            rep.check("identity_matches_manifest", rec.get("identity") == j["identity"], f"{jid}: job.json identity differs from the manifest")
            h = sha256_json(rec.get("identity"))
            rep.check("identity_hash", h == rec.get("identity_sha256") and jid.endswith(h[:10]),
                      f"{jid}: identity hash {h[:10]} does not match the record or the identifier")
            status = read_status(d)["status"]
            if status != "completed":
                rep.check("job_completed", False, f"{jid}: {status} (excluded from results)", warn=True)
                continue
            res_path = d / "result.json"
            if not rep.check("result_exists", res_path.exists(), f"{jid}: completed without result.json"):
                continue
            res = read_json(res_path)
            found: List[Dict[str, str]] = []
            _walk_outputs(res.get("outputs", {}), found)
            for o in found:
                p = d / o["path"]
                ok = p.exists() and sha256_file(p) == o["sha256"]
                rep.check("output_hash", ok, f"{jid}: output {o['path']} missing or changed after completion")
                outputs_of.setdefault(jid, {})[o["path"]] = o["sha256"]
            rep.check("result_code_matches_job", res.get("code_sha256") == rec["code"]["sha256"],
                      f"{jid}: result code hash differs from job.json")
            type_code[j["type"]].add(rec["code"]["sha256"])
            type_deps[j["type"]].add(rec["dependencies"]["sha256"])
            devices.add(res.get("device"))
            if j["type"] not in code_now:
                code_now[j["type"]] = code_scope(j["type"])["sha256"]
            rep.check("code_is_current", rec["code"]["sha256"] == code_now[j["type"]],
                      f"{jid}: produced by code that differs from the current files (scope {j['type']}); rerun it "
                      "(run_suite.py --invalidate-changed)")
            ident = j["identity"]
            c = res.get("costs", {})
            if j["type"] == "fpg":
                n_traj = int(ident["fpg"]["n_traj"])
                rep.check("fpg_counts", c.get("env_steps_train") == ident["budget_transitions"]
                          == c.get("iterations") * n_traj * horizon,
                          f"{jid}: f-PG transitions {c.get('env_steps_train')} != budget {ident['budget_transitions']} "
                          f"or iterations x n_traj x H")
                epochs = max(1, int(ident["fpg"].get("ppo_epochs", 0)))
                rep.check("fpg_counts", c.get("updates") == c.get("iterations") * epochs, f"{jid}: f-PG update count")
            elif j["type"] == "fb_pretrain":
                rep.check("fb_counts", c.get("updates") == ident["fb"]["n_updates"], f"{jid}: FB updates {c.get('updates')}")
            elif j["type"] == "adapt":
                lat = read_json(d / res["outputs"]["latents"]["path"])
                for r in lat["records"]:
                    rep.check("dm_counts", len(r["dm"]["log"]) == ident["dm"]["steps"] + 1, f"{jid}: DM log length")
            elif j["type"] in ("eval", "eval_expert"):
                rows = JsonlLog(d / res["outputs"]["evaluations"]["path"]).records()
                n_mc = ident["evaluation"].get("mc_rollouts", ident["evaluation"].get("rollouts"))
                rep.check("eval_counts", c.get("env_steps_eval") == len(rows) * n_mc * horizon,
                          f"{jid}: evaluation transitions {c.get('env_steps_eval')} != rows x rollouts x H")
                for row in rows:
                    audit_metrics(rep, jid, row, z_h, n_actions)
                    src = row.get("source", {})
                    if j["type"] == "eval" and "path" in src:
                        tgt = ident["target"]
                        rep.check("eval_source_hash", outputs_of.get(tgt, {}).get(src["path"]) == src["sha256"],
                                  f"{jid} {row['checkpoint']}: evaluated checkpoint hash is not the training output's")
        for t, codes in type_code.items():
            rep.check("code_not_mixed", len(codes) == 1, f"suite {name}: completed {t} jobs come from {len(codes)} code versions")
        for t, deps in type_deps.items():
            rep.check("dependencies_not_mixed", len(deps) == 1,
                      f"suite {name}: completed {t} jobs used {len(deps)} dependency sets", warn=True)
        rep.check("device_not_mixed", len(devices - {None}) <= 1,
                  f"suite {name}: completed jobs ran on several devices {sorted(d for d in devices if d)} (METHOD_MAP K2)", warn=True)


def _final_rows_recomputed(runs_root: Path, man: Dict[str, Any]) -> Dict[tuple, Dict[str, Any]]:
    """Independent recomputation of the last-checkpoint rows from the raw evaluation files."""
    out = {}
    for j in man["jobs"]:
        if j["type"] not in ("eval", "eval_expert"):
            continue
        d = runs_root / "jobs" / j["job_id"]
        if read_status(d)["status"] != "completed":
            continue
        rows = JsonlLog(d / "outputs" / "evaluations.jsonl").records()
        fpg_rows = [r for r in rows if r["method"] == "fpg"]
        last_it = max((r["coords"]["iteration"] for r in fpg_rows), default=None)
        for r in rows:
            c = r.get("coords", {})
            if r["method"] == "fpg":
                final = c["iteration"] == last_it
            elif r["method"] == "er":
                final = bool(c.get("final_fb_snapshot"))
            elif r["method"] == "dm":
                final = bool(c.get("final_fb_snapshot")) and bool(c.get("final_dm_step"))
            else:
                final = True
            if final:
                out[(j["job_id"], r["checkpoint"])] = {"metrics": r["metrics"], "label": method_label(j["labels"], r["method"]),
                                                       "labels": j["labels"]}
    return out


def _num(s: str) -> Optional[float]:
    return None if s in ("", None) else float(s)


def _close(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))


# disjoint wall-time phases recorded by each job type (train.py, evaluation.py); a total adds exactly these
WALL_PHASES = {"fpg": {"train", "checkpoint", "policy_checkpoint"}, "fb_pretrain": {"load", "train", "snapshot", "checkpoint"},
               "adapt": {"load", "er", "dm"}, "eval": {"evaluate"}, "eval_expert": {"evaluate"}}


def _phase_total(rep: AuditReport, jid: str, job_type: str, wall: Dict[str, Any]) -> float:
    unknown = sorted(set(wall) - WALL_PHASES[job_type])
    rep.check("wall_phases_known", not unknown,
              f"{jid}: wall-time entries {unknown} are not disjoint phases of a {job_type} job (they would be double-counted)")
    return float(sum(float(v) for k, v in wall.items() if k in WALL_PHASES[job_type]))


def audit_costs(rep: AuditReport, runs_root: Path, results_dir: Path, man: Dict[str, Any]) -> None:
    """Independent recomputation of ``costs_<suite>.csv`` from the ``result.json`` files."""
    path = results_dir / "tables" / f"costs_{man['suite']}.csv"
    if not rep.check("cost_table_exists", path.exists(), f"{path} missing"):
        return
    cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def result(jid: Optional[str]) -> Optional[Dict[str, Any]]:
        if jid is None:
            return None
        if jid not in cache:
            d = runs_root / "jobs" / jid
            cache[jid] = read_json(d / "result.json") if read_status(d)["status"] == "completed" and (d / "result.json").exists() else None
        return cache[jid]

    by_id = {j["job_id"]: j for j in man["jobs"]}
    eval_for = {j["identity"]["target"]: j["job_id"] for j in man["jobs"] if j["type"] == "eval"}
    goals: Dict[str, set] = defaultdict(set)
    for j in man["jobs"]:
        if j["type"] == "adapt":
            goals[j["identity"]["fb_job"]].add(j["labels"]["task"])
    expected = set()
    for j in man["jobs"]:
        if j["type"] == "fpg" and result(j["job_id"]) is not None:
            expected.add((j["job_id"], method_label(j["labels"], "fpg")))
        elif j["type"] == "adapt" and result(j["job_id"]) is not None and result(j["identity"]["fb_job"]) is not None:
            expected |= {(j["job_id"], method_label(j["labels"], m)) for m in ("dm", "er")}
    seen = set()
    for row in read_csv(path):
        key = (row["job_id"], row["method_label"])
        seen.add(key)
        if not rep.check("cost_row_traceable", key in expected, f"{path.name}: row {key} is not a completed run of the suite"):
            continue
        j = by_id[row["job_id"]]
        c = result(j["job_id"])["costs"]
        ev_id = eval_for.get(j["job_id"])
        ev = result(ev_id)
        want: Dict[str, Any] = {"expert_demo_transitions": c["demonstrations"]["transitions"],
                                "eval_env_steps": ev["costs"]["env_steps_eval"] if ev else None,
                                "eval_wall_s": _phase_total(rep, ev_id, "eval", ev["costs"]["wall_s"]) if ev else None}
        if j["type"] == "fpg":
            w = _phase_total(rep, j["job_id"], "fpg", c["wall_s"])
            steps = c["env_steps_train"]
            want.update(fpg_env_steps=steps, fpg_updates=c["updates"], fpg_wall_s=w, n_tasks_sharing=1,
                        one_task_env_steps=steps, one_task_wall_s=w, amortized_env_steps=steps, amortized_wall_s=w)
        else:
            fb_id = j["identity"]["fb_job"]
            fc = result(fb_id)["costs"]
            pre = _phase_total(rep, fb_id, "fb_pretrain", fc["wall_s"]) + float(fc.get("replay_generation_s") or 0.0)
            _phase_total(rep, j["job_id"], "adapt", c["wall_s"])
            method = "dm" if row["method_label"].startswith("DM_FB") else "er"
            a = float(c["final_snapshot_s"][method] or 0.0)
            n = len(goals[fb_id])
            steps = fc["replay_transitions"]
            want.update(fb_pretrain_transitions=steps, fb_pretrain_resets=fc["replay_resets"], fb_pretrain_updates=fc["updates"],
                        fb_pretrain_wall_s=pre, adapt_wall_s=a, adapt_env_steps=0, n_tasks_sharing=n,
                        dm_steps=c["dm_steps_per_snapshot"] if method == "dm" else 0,
                        one_task_env_steps=steps, one_task_wall_s=pre + a, amortized_env_steps=steps / n,
                        amortized_wall_s=pre / n + a)
        for k, v in want.items():
            rep.check("cost_values", _close(_num(row.get(k, "")), None if v is None else float(v)),
                      f"{path.name} {key}: {k} = {row.get(k)!r} but the run records give {v}")
    rep.check("cost_rows_complete", seen == expected,
              f"{path.name}: {len(expected - seen)} cost row(s) missing, {len(seen - expected)} extra")


def audit_thresholds(rep: AuditReport, runs_root: Path, results_dir: Path, man: Dict[str, Any]) -> None:
    """Independent recomputation of ``convergence_<suite>.csv`` (first checkpoint meeting the thresholds)."""
    path = results_dir / "tables" / f"convergence_{man['suite']}.csv"
    if not rep.check("threshold_table_exists", path.exists(), f"{path} missing"):
        return
    conv = man["config"]["evaluation"]["convergence"]
    js_max, s_min = float(conv["js_max_nats"]), float(conv["success_min"])
    curves: Dict[tuple, List[tuple]] = defaultdict(list)
    for j in man["jobs"]:
        d = runs_root / "jobs" / j["job_id"]
        if j["type"] != "eval" or read_status(d)["status"] != "completed":
            continue
        for r in JsonlLog(d / "outputs" / "evaluations.jsonl").records():
            c = r.get("coords", {})
            if r["method"] == "fpg":
                x = c["env_steps"]
            elif r["method"] == "er" or (r["method"] == "dm" and c.get("final_dm_step")):
                x = c["fb_update"]
            else:
                continue
            curves[(j["job_id"], method_label(j["labels"], r["method"]))].append((x, r["metrics"]))
    truth = {}
    for key, pts in curves.items():
        pts.sort(key=lambda p: p[0])
        hit = [x for x, m in pts if m["js_nats"] <= js_max and m["success"] >= s_min]
        last = pts[-1][1]
        truth[key] = (pts[-1][0], hit[0] if hit else None, bool(last["js_nats"] <= js_max and last["success"] >= s_min),
                      last["js_nats"], last["success"])
    seen = set()
    for row in read_csv(path):
        key = (row["eval_job"], row["method_label"])
        seen.add(key)
        if not rep.check("threshold_row_traceable", key in truth, f"{path.name}: row {key} has no evaluation curve"):
            continue
        budget, first, at_last, js_last, s_last = truth[key]
        ok = (_close(_num(row["budget"]), budget) and _close(_num(row["first_reached"]), first)
              and row["reached_at_last"] == str(at_last) and _close(_num(row["js_last"]), js_last)
              and _close(_num(row["success_last"]), s_last))
        rep.check("threshold_values", ok, f"{path.name} {key}: differs from the evaluation rows "
                  f"(budget {budget}, first reached {first}, reached at last {at_last})")
    rep.check("threshold_rows_complete", seen == set(truth),
              f"{path.name}: {len(set(truth) - seen)} row(s) missing, {len(seen - set(truth))} extra")


def audit_results(rep: AuditReport, runs_root: Path, results_dir: Path, manifests: List[Dict[str, Any]]) -> None:
    prov_path = results_dir / "provenance.json"
    if not rep.check("provenance_exists", prov_path.exists(), f"{prov_path} missing (run scripts/analyze.py)"):
        return
    prov = read_json(prov_path)
    is_project_results = results_dir.resolve() == (PROJECT_ROOT / "results").resolve()
    if is_project_results:
        rep.check("no_fixture_in_results", not prov.get("test_fixture") and prov.get("kinds") == ["final"],
                  "results/ was generated from smoke or pilot runs")
    rep.check("results_md_hash", (results_dir / "RESULTS.md").exists()
              and sha256_file(results_dir / "RESULTS.md") == prov["results_md"]["sha256"], "RESULTS.md changed after analysis")
    for rel, t in prov["tables"].items():
        p = results_dir / rel
        rep.check("table_hash", p.exists() and sha256_file(p) == t["sha256"], f"{rel} missing or changed after analysis")
    for f in prov["figures"]:
        p = results_dir / f["file"]
        rep.check("figure_hash", p.exists() and sha256_file(p) == f["sha256"], f"{f['file']} missing or changed after analysis")
    for jid, j in prov["jobs"].items():
        p = runs_root / "jobs" / jid / "result.json"
        rep.check("cited_job_intact", p.exists() and sha256_file(p) == j["result_sha256"],
                  f"{jid}: cited in provenance.json but its result.json is missing or changed (results are stale; rerun analyze.py)")
    names = {m["suite"] for m in manifests}
    for name in names:
        rep.check("suite_in_provenance", name in prov["suites"], f"suite {name} has runs but is not in provenance.json (rerun analyze.py)")
    for man in manifests:
        if man["suite"] not in prov["suites"]:
            continue
        rep.check("manifest_hash", prov["suites"][man["suite"]]["manifest_sha256"] == sha256_file(man["_path"]),
                  f"suite {man['suite']}: manifest changed after analysis (rerun analyze.py)")
        if is_project_results:
            rep.check("final_only", man["kind"] == "final", f"suite {man['suite']} of kind {man['kind']} in results/")
        for j in man["jobs"]:
            if j["type"] in ("eval", "eval_expert") and read_status(runs_root / "jobs" / j["job_id"])["status"] == "completed":
                rep.check("completed_eval_cited", j["job_id"] in prov["jobs"],
                          f"{j['job_id']} completed after the analysis (rerun analyze.py)")
        fr_path = results_dir / "tables" / f"final_runs_{man['suite']}.csv"
        if not rep.check("final_table_exists", fr_path.exists(), f"{fr_path} missing"):
            continue
        table = read_csv(fr_path)
        truth = _final_rows_recomputed(runs_root, man)
        seen = set()
        for row in table:
            key = (row["eval_job"], row["checkpoint"])
            seen.add(key)
            if not rep.check("final_row_traceable", key in truth, f"{fr_path.name}: row {key} has no final evaluation row"):
                continue
            t = truth[key]
            rep.check("final_row_label", row["method_label"] == t["label"], f"{key}: method label differs")
            for mk in ALL_METRICS:
                a, b = _num(row.get(mk, "")), t["metrics"].get(mk)
                b = None if isinstance(b, bool) else b
                ok = (a is None and b is None) or (a is not None and b is not None and abs(a - float(b)) <= 1e-12 * max(1.0, abs(float(b))))
                rep.check("final_row_values", ok, f"{key}: {mk} in the table ({a}) differs from the run file ({b})")
        rep.check("final_rows_complete", seen == set(truth),
                  f"{fr_path.name}: {len(set(truth) - seen)} final evaluation row(s) missing, {len(seen - set(truth))} extra")
        sm_path = results_dir / "tables" / f"final_summary_{man['suite']}.csv"
        if rep.check("summary_table_exists", sm_path.exists(), f"{sm_path} missing"):
            groups: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
            for row in table:
                groups[(row["env"], row["layout"], row["task"], row["method_label"])].append(row)
            for srow in read_csv(sm_path):
                key = (srow["env"], srow["layout"], srow["task"], srow["method_label"])
                g = groups.get(key, [])
                rep.check("summary_n", int(srow["n_completed"]) == len(g), f"{sm_path.name} {key}: n_completed differs")
                for mk in ("js_nats", "success", "return"):
                    if f"{mk}_mean" not in srow or srow[f"{mk}_mean"] == "":
                        continue
                    vals = [float(r[mk]) for r in g if r.get(mk, "") != ""]
                    m = statistics.fmean(vals) if vals else None
                    sd = statistics.stdev(vals) if len(vals) >= 2 else None
                    rep.check("summary_mean", m is not None and abs(m - float(srow[f"{mk}_mean"])) <= 1e-12 * max(1, abs(m)),
                              f"{sm_path.name} {key}: {mk} mean differs from the per-run table")
                    sd_t = _num(srow.get(f"{mk}_sd", ""))
                    rep.check("summary_sd", (sd is None and sd_t is None) or (sd is not None and sd_t is not None
                              and abs(sd - sd_t) <= 1e-12 * max(1, abs(sd))), f"{sm_path.name} {key}: {mk} SD differs")
        audit_costs(rep, runs_root, results_dir, man)
        audit_thresholds(rep, runs_root, results_dir, man)


def audit(runs_root: Union[str, Path], results_dir: Union[str, Path, None], kinds: Sequence[str] = ("final",),
          seeds_path: Optional[Path] = None) -> AuditReport:
    runs_root = Path(runs_root)
    rep = AuditReport()
    manifests = []
    for path in sorted((runs_root / "suites").glob("*/manifest.json")):
        man = read_json(path)
        if man.get("kind") in kinds:
            man["_path"] = path
            manifests.append(man)
    if not rep.check("suites_found", bool(manifests), f"no suite of kind {list(kinds)} under {runs_root / 'suites'}"):
        return rep
    audit_runs(rep, runs_root, manifests, seeds_path)
    if results_dir is not None:
        audit_results(rep, runs_root, Path(results_dir), manifests)
    return rep


def write_report(rep: AuditReport, path: Union[str, Path]) -> None:
    write_json_atomic(path, rep.to_dict())
