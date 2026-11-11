"""Analysis of real saved runs: tables, figures, ``RESULTS.md`` and ``provenance.json`` (``scripts/analyze.py``).

Only saved job records are read: suite manifests (the planned jobs and seeds),
job statuses, ``result.json`` files (costs) and evaluation rows
(``outputs/evaluations.jsonl``).  Nothing is imputed:

* every planned job and seed is listed with its status (completed, failed,
  interrupted, running, blocked, never started, or ``missing``: its directory
  was deleted although the outcome journal recorded an outcome) and its error;
* means and sample standard deviations (``ddof = 1``; ``n/a`` for ``n < 2``)
  are over completed seeds only, with ``n`` and the missing seeds shown next
  to them; individual seed values are always printed;
* tables report the **last** checkpoint of every run (full budget), next to the
  first checkpoint that meets the operational thresholds of the suite
  configuration ("not reached" otherwise); full curves are in
  ``tables/curves_long.csv`` and the figures;
* costs keep the phases apart: expert data, task-blind FB pretraining (shared
  by the goals of a layout), per-task adaptation, f-PG learning and
  evaluation; FB totals are given for one task and amortized over the goals
  that share the pretrained model.

Suites of kind ``final`` are analysed by default.  A ``smoke`` suite is a test
fixture: its RESULTS.md says so in its first line, and it cannot be written
into the project's ``results/`` directory.  Every table row and figure is
traced to job identifiers, hashes and configurations in ``provenance.json``.
Units: ``logging.UNITS`` (nats, seconds, transitions).
"""
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .logging import (PROJECT_ROOT, TERMINAL_EVENTS, UNITS, JsonlLog, current_outcomes, read_json, read_outcome_journal,
                      read_status, relpath, replace_file, sha256_file, utc_now, write_json_atomic, write_text_atomic)

METRICS_FINITE = ["js_nats", "success", "return", "gap", "js_sa_nats", "final_in_goal", "mc_success",
                  "policy_entropy_visited_nats"]
METRICS_CONTINUOUS = ["js_nats", "success", "return", "final_in_goal", "mc_post_hit_goal_region_frac",
                      "policy_entropy_visited_nats"]
ALL_METRICS = ["js_nats", "success", "return", "gap", "js_sa_nats", "final_in_goal", "mc_success", "mc_return_mean",
               "mc_post_hit_goal_region_frac", "policy_entropy_visited_nats", "kl_dq_nats", "kl_qd_nats",
               "kl_dq_nats_smoothed", "kl_qd_nats_smoothed", "demo_target_js_nats"]
FIXTURE_BANNER = "TEST FIXTURE - smoke run with tiny budgets; these numbers are NOT scientific results."
PILOT_BANNER = ("PILOT (seed 999) - for choosing and freezing settings in Prompt 4 only; these numbers are NOT final "
                "results and do not use the final seeds.")

# reference palette (dataviz skill references/palette.md): categorical slots 1-3 validated all-pairs (light);
# neutral ink for the expert reference; recessive chrome
METHOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


class AnalysisError(RuntimeError):
    pass


# =============================================================================
# loading
# =============================================================================
def discover_suites(runs_root: Union[str, Path], kinds: Sequence[str] = ("final",),
                    names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    root = Path(runs_root) / "suites"
    out = []
    for path in sorted(root.glob("*/manifest.json")):
        man = read_json(path)
        if man.get("kind") not in kinds:
            continue
        if names and man.get("suite") not in names:
            continue
        man["_path"] = path
        man["_sha256"] = sha256_file(path)
        out.append(man)
    return out


def _job_record(runs_root: Path, job_id: str, journal: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    d = runs_root / "jobs" / job_id
    rec: Dict[str, Any] = {"job_id": job_id, "dir": d, "exists": (d / "job.json").exists()}
    st = read_status(d) if d.exists() else {"status": "planned", "attempts": []}
    rec["status"] = st.get("status", "planned")
    last = (st.get("attempts") or [{}])[-1]
    rec["error"] = last.get("error")
    if not rec["exists"]:
        if journal is None:
            journal = read_outcome_journal(runs_root)
        done = [e for e in current_outcomes(journal.get(job_id, [])) if e["event"] in TERMINAL_EVENTS]
        if done:
            rec["status"] = "missing"
            rec["error"] = (f"job directory missing although its outcome '{done[-1]['event']}' was recorded at "
                            f"{done[-1]['utc']}")
    rec["device"] = last.get("device")
    rec["result"] = read_json(d / "result.json") if (d / "result.json").exists() and rec["status"] == "completed" else None
    rec["job"] = read_json(d / "job.json") if rec["exists"] else None
    return rec


def method_label(labels: Dict[str, Any], method: str) -> str:
    """Display name: ``f-PG[js]``, ``f-PG[kl_dq]/lr_low``, ``DM_FB``, ``ER_FB/no_orth``, ``expert``."""
    if method == "fpg":
        lab = f"f-PG[{labels.get('divergence')}]"
        v = labels.get("fpg_variant", "base")
        return lab if v == "base" else f"{lab}/{v}"
    if method in ("er", "dm"):
        lab = "ER_FB" if method == "er" else "DM_FB"
        v = labels.get("fb_variant", "base")
        return lab if v == "base" else f"{lab}/{v}"
    if method == "expert":
        return "expert"
    raise ValueError(method)


def _method_order(label: str) -> Tuple[int, str]:
    order = ["f-PG", "DM_FB", "ER_FB", "expert"]
    for k, prefix in enumerate(order):
        if label.startswith(prefix):
            return (k, label)
    return (len(order), label)


class Collection:
    """Everything the tables need, for one or several suites."""

    def __init__(self, runs_root: Union[str, Path], kinds: Sequence[str] = ("final",),
                 names: Optional[Sequence[str]] = None):
        self.runs_root = Path(runs_root)
        self.suites = discover_suites(self.runs_root, kinds, names)
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.journal = read_outcome_journal(self.runs_root)
        for man in self.suites:
            for j in man["jobs"]:
                if j["job_id"] not in self.jobs:
                    rec = _job_record(self.runs_root, j["job_id"], self.journal)
                    rec["spec"] = j
                    self.jobs[j["job_id"]] = rec
        self.rows = self._eval_rows()

    def suite_jobs(self, man: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [self.jobs[j["job_id"]] for j in man["jobs"]]

    def _eval_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for man in self.suites:
            for j in man["jobs"]:
                if j["type"] not in ("eval", "eval_expert"):
                    continue
                rec = self.jobs[j["job_id"]]
                if rec["status"] != "completed" or rec["result"] is None:
                    continue
                ev = rec["result"]["outputs"]["evaluations"]
                path = rec["dir"] / ev["path"]
                for r in JsonlLog(path).records():
                    labels = j["labels"]
                    coords = r.get("coords", {})
                    method = r["method"]
                    row = {
                        "suite": man["suite"], "kind": man["kind"], "env": labels["env"], "layout": labels["layout"],
                        "task": labels["task"], "seed": int(labels["seed"]), "method": method,
                        "method_label": method_label(labels, method), "divergence": labels.get("divergence"),
                        "fpg_variant": labels.get("fpg_variant"), "fb_variant": labels.get("fb_variant"),
                        "checkpoint": r["checkpoint"], "eval_job": j["job_id"],
                        "target_job": j["identity"].get("target"),
                        "iteration": coords.get("iteration"), "env_steps": coords.get("env_steps"),
                        "fb_update": coords.get("fb_update"), "dm_step": coords.get("dm_step"),
                        "final_fb_snapshot": coords.get("final_fb_snapshot"), "final_dm_step": coords.get("final_dm_step"),
                        "metrics": r["metrics"],
                    }
                    rows.append(row)
        self._mark_final(rows)
        return rows

    @staticmethod
    def _mark_final(rows: List[Dict[str, Any]]) -> None:
        last_iter: Dict[str, int] = {}
        for r in rows:
            if r["method"] == "fpg":
                last_iter[r["eval_job"]] = max(last_iter.get(r["eval_job"], -1), int(r["iteration"]))
        for r in rows:
            if r["method"] == "fpg":
                r["final"] = int(r["iteration"]) == last_iter[r["eval_job"]]
            elif r["method"] == "er":
                r["final"] = bool(r["final_fb_snapshot"])
            elif r["method"] == "dm":
                r["final"] = bool(r["final_fb_snapshot"]) and bool(r["final_dm_step"])
            else:
                r["final"] = True


# =============================================================================
# statistics
# =============================================================================
def mean_sd(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    sd = statistics.stdev(vals) if len(vals) >= 2 else None
    return m, sd


def fmt(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}"


def fmt_ms(m: Optional[float], sd: Optional[float], digits: int = 3) -> str:
    if m is None:
        return "n/a"
    return f"{fmt(m, digits)} +/- {fmt(sd, digits)}" if sd is not None else f"{fmt(m, digits)} (n=1)"


# =============================================================================
# tables
# =============================================================================
def planned_seeds(coll: Collection, man: Dict[str, Any]) -> Dict[Tuple, Dict[int, Dict[str, Any]]]:
    """``(env, layout, task, method_label) -> {seed: eval job record}`` from the manifest (all planned seeds)."""
    out: Dict[Tuple, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for j in man["jobs"]:
        lab = j["labels"]
        if j["type"] == "eval_expert":
            labels = ["expert"]
        elif j["type"] == "eval" and lab.get("target_type") == "fpg":
            labels = [method_label(lab, "fpg")]
        elif j["type"] == "eval" and lab.get("target_type") == "adapt":
            labels = [method_label(lab, "dm"), method_label(lab, "er")]
        else:
            continue
        for ml in labels:
            out[(lab["env"], lab["layout"], lab["task"], ml)][int(lab["seed"])] = coll.jobs[j["job_id"]]
    return out


def _why_missing(coll: Collection, rec: Dict[str, Any]) -> str:
    """Status of an evaluation job and, if it never ran, of the training job it waits for."""
    if rec["status"] == "completed":
        return "completed"
    msg = rec["status"]
    if rec.get("error"):
        msg += f" ({rec['error']})"
    for dep in rec["spec"].get("deps", []):
        d = coll.jobs.get(dep) or _job_record(coll.runs_root, dep, coll.journal)
        if d["status"] != "completed":
            msg += f"; training job {dep}: {d['status']}" + (f" ({d['error']})" if d.get("error") else "")
    return msg


def final_tables(coll: Collection, man: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Per-run final rows and their per-condition summaries for one suite."""
    rows = [r for r in coll.rows if r["suite"] == man["suite"] and r["final"]]
    plan = planned_seeds(coll, man)
    by_key: Dict[Tuple, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_key[(r["env"], r["layout"], r["task"], r["method_label"])][r["seed"]] = r
    summary = []
    for key in sorted(plan, key=lambda k: (k[0], k[1], k[2], _method_order(k[3]))):
        env, layout, task, ml = key
        planned = sorted(plan[key])
        got = by_key.get(key, {})
        rec = {"suite": man["suite"], "env": env, "layout": layout, "task": task, "method_label": ml,
               "n_planned": len(planned), "n_completed": len(got), "seeds_completed": sorted(got),
               "seeds_missing": {s: _why_missing(coll, plan[key][s]) for s in planned if s not in got},
               "eval_jobs": sorted({got[s]["eval_job"] for s in got}),
               "target_jobs": sorted({got[s]["target_job"] for s in got if got[s]["target_job"]})}
        metrics = METRICS_FINITE if env == "finite" else METRICS_CONTINUOUS
        for m in set(metrics) | {"demo_target_js_nats"}:
            vals = [got[s]["metrics"].get(m) for s in sorted(got)]
            if all(v is None for v in vals):
                continue
            mean, sd = mean_sd(vals)
            rec[m] = {"mean": mean, "sd": sd, "values": {s: got[s]["metrics"].get(m) for s in sorted(got)}}
        summary.append(rec)
    return rows, summary


def convergence_table(coll: Collection, man: Dict[str, Any]) -> List[Dict[str, Any]]:
    """First checkpoint meeting ``js <= js_max`` and ``success >= success_min``, and whether the last one does."""
    conv = man["config"]["evaluation"]["convergence"]
    js_max, s_min = float(conv["js_max_nats"]), float(conv["success_min"])
    curves: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in coll.rows:
        if r["suite"] != man["suite"] or r["method"] == "expert":
            continue
        if r["method"] == "fpg":
            axis, x = "learner env steps", r["env_steps"]
        elif r["method"] == "er" or (r["method"] == "dm" and r["final_dm_step"]):
            axis, x = "FB pretraining updates", r["fb_update"]
        else:
            continue
        curves[(r["env"], r["layout"], r["task"], r["method_label"], r["seed"], axis)].append({"x": x, "row": r})
    out = []
    for key in sorted(curves, key=lambda k: (k[0], k[1], k[2], _method_order(k[3]), k[4])):
        pts = sorted(curves[key], key=lambda p: p["x"])
        ok = [p for p in pts if p["row"]["metrics"]["js_nats"] <= js_max and p["row"]["metrics"]["success"] >= s_min]
        last = pts[-1]["row"]["metrics"]
        out.append({"suite": man["suite"], "env": key[0], "layout": key[1], "task": key[2], "method_label": key[3],
                    "seed": key[4], "axis": key[5], "budget": pts[-1]["x"],
                    "first_reached": ok[0]["x"] if ok else None,
                    "reached_at_last": bool(last["js_nats"] <= js_max and last["success"] >= s_min),
                    "js_last": last["js_nats"], "success_last": last["success"],
                    "js_max_nats": js_max, "success_min": s_min, "eval_job": pts[-1]["row"]["eval_job"]})
    return out


def _sum_wall(costs: Dict[str, Any]) -> float:
    return float(sum(v for v in (costs.get("wall_s") or {}).values() if isinstance(v, (int, float))))


def cost_table(coll: Collection, man: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-run costs by phase, for every completed f-PG and adaptation run of the suite."""
    by_id = {j["job_id"]: j for j in man["jobs"]}
    tasks_per_fb: Dict[str, set] = defaultdict(set)
    for j in man["jobs"]:
        if j["type"] == "adapt":
            tasks_per_fb[j["identity"]["fb_job"]].add(j["labels"]["task"])
    eval_of: Dict[str, Dict[str, Any]] = {}
    for j in man["jobs"]:
        if j["type"] == "eval":
            eval_of[j["identity"]["target"]] = coll.jobs[j["job_id"]]
    rows = []
    for j in man["jobs"]:
        rec = coll.jobs[j["job_id"]]
        if j["type"] not in ("fpg", "adapt") or rec["result"] is None:
            continue
        lab = j["labels"]
        res = rec["result"]
        c = res["costs"]
        ev = eval_of.get(j["job_id"])
        ev_c = ev["result"]["costs"] if ev and ev["result"] else None
        base = {"suite": man["suite"], "env": lab["env"], "layout": lab["layout"], "task": lab["task"],
                "seed": int(lab["seed"]), "job_id": j["job_id"], "device": res.get("device"),
                "expert_demo_transitions": c["demonstrations"]["transitions"],
                "eval_env_steps": ev_c["env_steps_eval"] if ev_c else None,
                "eval_wall_s": _sum_wall(ev_c) if ev_c else None, "eval_job": ev["job_id"] if ev else None}
        if j["type"] == "fpg":
            rows.append(dict(base, method_label=method_label(lab, "fpg"), fpg_env_steps=c["env_steps_train"],
                             fpg_updates=c["updates"], fpg_wall_s=_sum_wall(c),
                             one_task_env_steps=c["env_steps_train"], one_task_wall_s=_sum_wall(c),
                             amortized_env_steps=c["env_steps_train"], amortized_wall_s=_sum_wall(c), n_tasks_sharing=1))
            continue
        fb_id = j["identity"]["fb_job"]
        fb = coll.jobs.get(fb_id) or _job_record(coll.runs_root, fb_id, coll.journal)
        if fb["result"] is None:
            continue
        fc = fb["result"]["costs"]
        n_share = max(1, len(tasks_per_fb[fb_id]))
        pre_wall = _sum_wall(fc) + float(fc.get("replay_generation_s") or 0.0)
        pre_steps = int(fc["replay_transitions"])
        fin = c["final_snapshot_s"]
        for method, key in (("dm", "dm"), ("er", "er")):
            adapt_s = float(fin[key] or 0.0)
            rows.append(dict(base, method_label=method_label(lab, method), fb_job=fb_id,
                             fb_pretrain_transitions=pre_steps, fb_pretrain_resets=int(fc["replay_resets"]),
                             fb_pretrain_updates=int(fc["updates"]), fb_pretrain_wall_s=pre_wall,
                             adapt_wall_s=adapt_s, dm_steps=int(c["dm_steps_per_snapshot"]) if method == "dm" else 0,
                             adapt_env_steps=0, n_tasks_sharing=n_share,
                             one_task_env_steps=pre_steps, one_task_wall_s=pre_wall + adapt_s,
                             amortized_env_steps=pre_steps / n_share, amortized_wall_s=pre_wall / n_share + adapt_s))
    return rows


def fb_diagnostics_table(coll: Collection, man: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for j in man["jobs"]:
        rec = coll.jobs[j["job_id"]]
        if j["type"] != "adapt" or rec["result"] is None:
            continue
        fb = coll.jobs.get(j["identity"]["fb_job"])
        diag = fb["result"]["summary"]["final_diagnostics"] if fb and fb["result"] else {}
        s = rec["result"]["summary"]["final"]
        lab = j["labels"]
        fbme = {}
        for r in coll.rows:
            if r["suite"] == man["suite"] and r["target_job"] == j["job_id"] and r["final"] and "fb_model_error" in r["metrics"]:
                fbme[r["method"]] = r["metrics"]["fb_model_error"]
        rows.append({"suite": man["suite"], "env": lab["env"], "layout": lab["layout"], "task": lab["task"],
                     "seed": int(lab["seed"]), "fb_variant": lab.get("fb_variant"), "adapt_job": j["job_id"],
                     "fb_job": j["identity"]["fb_job"], "fb_update": s["fb_update"],
                     "gram_lambda_min": diag.get("gram_lambda_min"), "gram_lambda_max": diag.get("gram_lambda_max"),
                     "gram_dev_fro": diag.get("gram_dev_fro"), "F_dot_meanB_mean": diag.get("F_dot_meanB_mean"),
                     "one_over_one_minus_gamma": diag.get("one_over_one_minus_gamma"),
                     "er_norm_raw": s["er_norm_raw"], "dm_initial_loss": s["dm_initial_loss"], "dm_final_loss": s["dm_final_loss"],
                     "er_entropy_nats": s["u2_er"]["policy_entropy_nats"], "er_score_spread_over_tau": s["u2_er"]["score_spread_over_tau"],
                     "dm_entropy_nats": s["u2_dm"]["policy_entropy_nats"], "dm_score_spread_over_tau": s["u2_dm"]["score_spread_over_tau"],
                     "er_fb6_sup_tv": fbme.get("er", {}).get("sup_tv"), "dm_fb6_sup_tv": fbme.get("dm", {}).get("sup_tv"),
                     "er_fb6_q_error_max": fbme.get("er", {}).get("q_error_max"),
                     "dm_fb6_q_error_max": fbme.get("dm", {}).get("q_error_max")})
    return rows


def status_rows(coll: Collection, man: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for j in man["jobs"]:
        rec = coll.jobs[j["job_id"]]
        status = rec["status"]
        if status == "planned":
            blocked = [d for d in j.get("deps", []) if coll.jobs.get(d, {}).get("status") != "completed"]
            if blocked:
                status = "blocked"
        lab = j["labels"]
        out.append({"suite": man["suite"], "job_id": j["job_id"], "type": j["type"], "env": lab.get("env"),
                    "layout": lab.get("layout"), "task": lab.get("task", ""), "seed": lab.get("seed"),
                    "method": lab.get("method"), "variant": lab.get("fpg_variant") or lab.get("fb_variant") or "",
                    "divergence": lab.get("divergence") or "", "status": status, "error": rec.get("error") or "",
                    "device": rec.get("device") or ""})
    return out


# =============================================================================
# writing
# =============================================================================
def _flat(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        f: Dict[str, Any] = {}
        for k, v in r.items():
            if k == "metrics":
                for mk in ALL_METRICS:
                    f[mk] = v.get(mk)
            elif isinstance(v, dict) and set(v) >= {"mean", "sd"}:
                f[f"{k}_mean"] = v["mean"]
                f[f"{k}_sd"] = v["sd"]
                f[f"{k}_values"] = ";".join(f"s{s}={'' if x is None else repr(float(x))}" for s, x in v["values"].items())
            elif isinstance(v, (dict, list)):
                f[k] = repr(v) if k != "seeds_missing" else ";".join(f"s{s}: {why}" for s, why in v.items())
            else:
                f[k] = v
        out.append(f)
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> str:
    rows = _flat(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (repr(r[k]) if isinstance(r[k], float) else r[k])) for k in fields})
    replace_file(tmp, path)
    return sha256_file(path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# =============================================================================
# figures
# =============================================================================
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def _colors(labels: Sequence[str]) -> Dict[str, str]:
    """Fixed slot per method label (slot order by method family, never by rank)."""
    ordered = sorted(set(l for l in labels if l != "expert"), key=_method_order)
    return {lab: METHOD_COLORS[k % len(METHOD_COLORS)] for k, lab in enumerate(ordered)}


def curve_figures(coll: Collection, man: Dict[str, Any], out_dir: Path, fixture: bool) -> List[Dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    rows = [r for r in coll.rows if r["suite"] == man["suite"]]
    conds = sorted({(r["env"], r["layout"], r["task"]) for r in rows})
    for env, layout, task in conds:
        rs = [r for r in rows if (r["env"], r["layout"], r["task"]) == (env, layout, task)]
        colors = _colors([r["method_label"] for r in rs])
        expert = [r for r in rs if r["method"] == "expert"]
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        panels = [("f-PG", lambda r: r["method"] == "fpg", "env_steps", "learner transitions"),
                  ("ER_FB and DM_FB over FB pretraining",
                   lambda r: r["method"] == "er" or (r["method"] == "dm" and r["final_dm_step"]), "fb_update",
                   "FB pretraining updates (task-blind)"),
                  ("DM_FB latent optimization (final FB snapshot)",
                   lambda r: r["method"] == "dm" and r["final_fb_snapshot"], "dm_step", "DM latent steps")]
        jobs_used = set()
        for col, (title, sel, xkey, xlab) in enumerate(panels):
            for row_i, (metric, ylab) in enumerate((("js_nats", "held-out JS (nats)"), ("success", "success rate"))):
                ax = axes[row_i, col]
                _style(ax)
                sub = [r for r in rs if sel(r)]
                for lab in sorted({r["method_label"] for r in sub}, key=_method_order):
                    by_seed: Dict[int, List] = defaultdict(list)
                    for r in sub:
                        if r["method_label"] == lab:
                            by_seed[r["seed"]].append((r[xkey], r["metrics"][metric]))
                            jobs_used.add(r["eval_job"])
                    xs_all: Dict[Any, List[float]] = defaultdict(list)
                    for seed, pts in sorted(by_seed.items()):
                        pts.sort()
                        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colors[lab], linewidth=1.0, alpha=0.35)
                        for x, y in pts:
                            xs_all[x].append(y)
                    xs = sorted(xs_all)
                    ax.plot(xs, [sum(xs_all[x]) / len(xs_all[x]) for x in xs], color=colors[lab], linewidth=2.0,
                            marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.0,
                            label=f"{lab} (mean of {len(by_seed)} seed{'s' if len(by_seed) != 1 else ''})")
                if expert:
                    vals = [r["metrics"][metric] for r in expert if r["metrics"].get(metric) is not None]
                    if vals:
                        ax.axhline(sum(vals) / len(vals), color=INK2, linewidth=1.0, linestyle=(0, (4, 3)),
                                   label="expert (held-out evaluation)")
                if row_i == 0:
                    ax.set_title(title, fontsize=9, color=INK)
                else:
                    ax.set_xlabel(xlab, fontsize=8, color=INK2)
                ax.set_ylabel(ylab, fontsize=8, color=INK2)
                if metric == "success":
                    ax.set_ylim(-0.03, 1.03)
                if sub or expert:
                    ax.legend(fontsize=6.5, frameon=False, labelcolor=INK2, loc="best")
        head = f"{man['suite']}: {env} {layout}, {task}"
        if man["kind"] == "smoke":
            head = "TEST FIXTURE (smoke) - " + head
        elif man["kind"] == "pilot":
            head = "PILOT (seed 999, not a final result) - " + head
        fig.suptitle(head + "  (thin: individual seeds; thick: mean; last checkpoint = full budget)", fontsize=9, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = out_dir / f"curves_{man['suite']}_{env}_{layout}_{task}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        figs.append({"file": path, "jobs": sorted(jobs_used), "condition": [env, layout, task]})
    return figs


def final_figure(summary: List[Dict[str, Any]], man: Dict[str, Any], out_dir: Path, metric: str, ylab: str,
                 fixture: bool) -> Optional[Dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [s for s in summary if metric in s and s["n_completed"] > 0]
    if not rows:
        return None
    conds = sorted({(s["env"], s["layout"], s["task"]) for s in rows})
    labels = sorted({s["method_label"] for s in rows}, key=_method_order)
    colors = _colors(labels)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(conds) + 2), 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    width = 0.8 / max(1, len(labels))
    jobs = set()
    labelled = set()
    for k, lab in enumerate(labels):
        for c_i, cond in enumerate(conds):
            match = [s for s in rows if (s["env"], s["layout"], s["task"]) == cond and s["method_label"] == lab]
            if not match:
                continue
            s = match[0]
            x0 = c_i - 0.4 + width * (k + 0.5)
            vals = [v for v in s[metric]["values"].values() if v is not None]
            color = INK2 if lab == "expert" else colors[lab]
            ax.scatter([x0] * len(vals), vals, s=14, color=color, alpha=0.55, linewidths=0, zorder=3)
            ax.plot([x0 - width * 0.35, x0 + width * 0.35], [s[metric]["mean"]] * 2, color=color, linewidth=2.0, zorder=4,
                    label=None if lab in labelled else lab)
            labelled.add(lab)
            jobs.update(s["eval_jobs"])
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([f"{e}\n{l}\n{t}" for e, l, t in conds], fontsize=7, color=INK2)
    ax.set_ylabel(ylab, fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2, loc="best")
    title = f"{man['suite']}: {ylab} at the last checkpoint (dots: seeds; bar: mean over completed seeds)"
    prefix = {"smoke": "TEST FIXTURE (smoke) - ", "pilot": "PILOT (seed 999) - "}.get(man["kind"], "")
    ax.set_title(prefix + title, fontsize=8.5, color=INK)
    fig.tight_layout()
    path = out_dir / f"final_{metric}_{man['suite']}.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return {"file": path, "jobs": sorted(jobs), "metric": metric}


# =============================================================================
# RESULTS.md
# =============================================================================
def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return out


def _seed_values(rec: Dict[str, Any], metric: str) -> str:
    if metric not in rec:
        return "n/a"
    return ", ".join(f"s{s}: {fmt(v)}" for s, v in rec[metric]["values"].items())


def results_markdown(coll: Collection, per_suite: Dict[str, Dict[str, Any]], out_dir: Path, fixture: bool) -> str:
    lines: List[str] = []
    kinds = {m["kind"] for m in coll.suites}
    if "smoke" in kinds:
        lines += [f"> **{FIXTURE_BANNER}**", ""]
    elif "pilot" in kinds:
        lines += [f"> **{PILOT_BANNER}**", ""]
    lines += ["# Results", "",
              f"Generated by `scripts/analyze.py` on {utc_now()} from `{relpath(coll.runs_root)}`. "
              "Do not edit by hand: rerun the analysis. Every number below is read from saved run files; "
              f"`provenance.json` maps each table row and figure to job identifiers and hashes, and "
              "`scripts/audit_results.py` checks them.", "",
              "Units: divergences in nats (natural logarithm; JS with half weights, in [0, log 2]); returns are "
              "discounted sums sum_{t<H} gamma^t r_g(s_{t+1}) with gamma and H of `configs/environment.yaml`; "
              "success = hit of the goal region at some t in 1..H; times in seconds; interactions in environment "
              "transitions. Means and sample standard deviations are over completed seeds only (n shown); "
              "individual seeds are listed; nothing is imputed.", ""]
    for name, info in per_suite.items():
        man = info["manifest"]
        lines += [f"## Suite `{name}` ({man['kind']})", "", man["description"], "",
                  f"Configuration `{man['config_path']}` (SHA-256 `{man['config_sha256'][:16]}`), seeds {man['seeds']}, "
                  f"device frozen at launch: {man['device']['device']}.", ""]
        revs = man.get("revisions", [])
        if len(revs) > 1:
            lines += [f"The manifest was revised after its first launch ({len(revs) - 1} revision(s)); jobs removed from "
                      "the plan by a revision are listed and are not part of the tables below:", ""]
            for r in revs[1:]:
                removed = r.get("removed_jobs") or []
                lines.append(f"- {r['utc']}: {r['reason']}" + (f"; removed {len(removed)} planned job(s): "
                             + ", ".join(f"`{x}`" for x in removed) if removed else ""))
            lines.append("")
        counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for r in info["status"]:
            counts[(r["type"], r["status"])] += 1
        types = sorted({r["type"] for r in info["status"]})
        sts = ["completed", "failed", "interrupted", "running", "blocked", "planned", "missing"]
        lines += ["### Run status", ""]
        lines += _md_table(["job type"] + sts + ["total"],
                           [[t] + [counts.get((t, s), 0) for s in sts] + [sum(counts.get((t, s), 0) for s in sts)] for t in types])
        bad = [r for r in info["status"] if r["status"] != "completed"]
        if bad:
            lines += ["", "Jobs not completed (they are excluded from the statistics below, never imputed):", ""]
            for r in bad:
                lines.append(f"- `{r['job_id']}`: {r['status']}" + (f" - {r['error']}" if r["error"] else ""))
        lines += ["", "### Held-out metrics at the last checkpoint", ""]
        for env in ("finite", "continuous"):
            sub = [s for s in info["summary"] if s["env"] == env]
            if not sub:
                continue
            metrics = METRICS_FINITE[:5] if env == "finite" else METRICS_CONTINUOUS[:4]
            if env == "finite":
                lines.append("Finite mazes: exact oracle evaluation (JS of d+_{pi,H} against q_E = d+_{pi*,H}; exact "
                             "return, gap to V*_H and hit probability; js_sa = state-action diagnostic).")
            else:
                lines.append("Continuous maze: Monte Carlo evaluation; JS between free-space histogram masses of the "
                             "policy's rollouts and of held-out expert reference rollouts (plug-in, biased upward; "
                             "the expert row gives the noise floor).")
            lines.append("")
            header = ["layout", "goal", "method", "n/planned"] + [f"{m} mean +/- SD" for m in metrics]
            body = []
            for s in sub:
                body.append([s["layout"], s["task"], s["method_label"], f"{s['n_completed']}/{s['n_planned']}"]
                            + [fmt_ms(s[m]["mean"], s[m]["sd"]) if m in s else "n/a" for m in metrics])
            lines += _md_table(header, body)
            lines += ["", "Individual seeds (JS, success):", ""]
            for s in sub:
                miss = "; ".join(f"s{k}: {v}" for k, v in s["seeds_missing"].items())
                lines.append(f"- {s['layout']} {s['task']} {s['method_label']}: JS [{_seed_values(s, 'js_nats')}]; "
                             f"success [{_seed_values(s, 'success')}]" + (f"; missing {miss}" if miss else ""))
            lines.append("")
        conv = info["convergence"]
        if conv:
            c0 = conv[0]
            lines += ["### Operational convergence", "",
                      f"Thresholds of the suite configuration: JS <= {c0['js_max_nats']} nats and success >= "
                      f"{c0['success_min']} (study choices, not universal standards). 'not reached' means not within "
                      "the full budget.", ""]
            body = [[c["env"], c["layout"], c["task"], c["method_label"], c["seed"], c["axis"],
                     fmt(c["first_reached"], 0) if c["first_reached"] is not None else "not reached",
                     "yes" if c["reached_at_last"] else "no", fmt(c["budget"], 0)] for c in conv]
            lines += _md_table(["env", "layout", "goal", "method", "seed", "axis", "first reached", "met at last", "budget"], body)
            lines.append("")
        costs = info["costs"]
        if costs:
            lines += ["### Cost accounting", "",
                      "Phases are kept apart. Expert data: the N_E demonstrations (same for every method). "
                      "FB pretraining: task-blind replay transitions, FB updates and their wall time, shared by the goals "
                      "of a layout and seed; one-task total = pretraining + adaptation; amortized = pretraining / "
                      "(number of goals sharing it) + adaptation. f-PG: its own learner transitions and wall time. "
                      "Adaptation collects no transition. Evaluation is reported separately and never counts as training "
                      "data; one evaluation job covers both ER_FB and DM_FB checkpoints of an adaptation run, so they "
                      "share its evaluation cost. Mean +/- SD over seeds.", ""]
            groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
            for c in costs:
                groups[(c["env"], c["layout"], c["task"], c["method_label"])].append(c)
            body = []
            for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], _method_order(k[3]))):
                g = groups[key]

                def ms(field: str, digits: int = 1) -> str:
                    m, sd = mean_sd([x.get(field) for x in g])
                    return fmt_ms(m, sd, digits)

                body.append([key[0], key[1], key[2], key[3], len(g), ms("expert_demo_transitions", 0),
                             ms("one_task_env_steps", 0), ms("one_task_wall_s"), ms("amortized_env_steps", 0),
                             ms("amortized_wall_s"), ms("eval_env_steps", 0), ms("eval_wall_s")])
            lines += _md_table(["env", "layout", "goal", "method", "n", "expert transitions", "one-task transitions",
                                "one-task wall s", "amortized transitions", "amortized wall s", "evaluation transitions",
                                "evaluation wall s"], body)
            lines.append("")
        fbd = info["fb_diagnostics"]
        if fbd:
            lines += ["### FB diagnostics (final snapshot; U-2 and FB-6)", "",
                      "Gram matrix of the replay next states (uncentered), ||z_ER|| before projection, policy entropy "
                      "(nats) and score spread / tau at the DM start states; FB-6: sup TV of the model's successor-measure "
                      "error and the largest |Q - F^T z_r| (finite mazes, evaluation only).", ""]
            body = [[r["env"], r["layout"], r["task"], r["fb_variant"], r["seed"], fmt(r["gram_lambda_min"], 4), fmt(r["gram_dev_fro"]),
                     fmt(r["er_norm_raw"]), fmt(r["er_entropy_nats"]), fmt(r["dm_entropy_nats"]),
                     fmt(r["dm_score_spread_over_tau"], 1), fmt(r["dm_fb6_sup_tv"]), fmt(r["dm_fb6_q_error_max"])]
                    for r in fbd]
            lines += _md_table(["env", "layout", "goal", "FB variant", "seed", "lambda_min(G)", "||G-I||_F", "||z_ER||",
                                "ER entropy", "DM entropy", "DM spread/tau", "DM FB-6 sup TV", "DM FB-6 max Q error"], body)
            lines.append("")
        if info["figures"]:
            lines += ["### Figures", ""]
            for f in info["figures"]:
                rel = Path(f["file"]).relative_to(out_dir).as_posix()
                lines.append(f"- [`{rel}`]({rel})")
            lines.append("")
    lines += ["## Notes on what these numbers do and do not show", "",
              "- State occupancy matching does not identify the expert's action choices; the finite js_sa column is a "
              "diagnostic, and return and success are evaluation-only quantities that no learner optimizes.",
              "- f-PG optimizes a smoothed plug-in objective against the 4-demonstration target; its gradient estimate is "
              "biased (theory FPG-5(b), FPG-9). DM_FB minimizes the model's estimate of its objective; nothing certifies "
              "the true objective (FBIL-2(e)). No convergence or expert-recovery guarantee applies to these runs.",
              "- Three seeds do not support broad superiority claims; read the individual seeds.",
              "- Continuous JS values are plug-in estimates from finite samples (biased upward); compare them with the "
              "expert row of the same goal.", ""]
    return "\n".join(lines) + "\n"


# =============================================================================
# entry point
# =============================================================================
def analyze(runs_root: Union[str, Path], out_dir: Union[str, Path], kinds: Sequence[str] = ("final",),
            names: Optional[Sequence[str]] = None, figures: bool = True) -> Dict[str, Any]:
    out = Path(out_dir)
    fixture = any(k != "final" for k in kinds)
    if fixture and out.resolve() == (PROJECT_ROOT / "results").resolve():
        raise AnalysisError("smoke or pilot runs are never written into the project's results/ directory")
    coll = Collection(runs_root, kinds, names)
    if not coll.suites:
        raise AnalysisError(f"no suite of kind {list(kinds)} in {Path(runs_root) / 'suites'}")
    n_completed = sum(1 for r in coll.jobs.values() if r["status"] == "completed" and r["spec"]["type"] in ("eval", "eval_expert"))
    learner_rows = [r for r in coll.rows if r["method"] != "expert"]
    if n_completed == 0 or not learner_rows:
        raise AnalysisError("no completed evaluation of a learner in the selected suites; RESULTS.md is not written "
                            "(it is generated from real runs only)")
    tables_dir = out / "tables"
    fig_dir = out / "figures"
    per_suite: Dict[str, Dict[str, Any]] = {}
    prov_tables: Dict[str, Any] = {}
    prov_figs: List[Dict[str, Any]] = []
    for man in coll.suites:
        name = man["suite"]
        rows, summary = final_tables(coll, man)
        info = {"manifest": man, "status": status_rows(coll, man), "final_rows": rows, "summary": summary,
                "convergence": convergence_table(coll, man), "costs": cost_table(coll, man),
                "fb_diagnostics": fb_diagnostics_table(coll, man), "figures": []}
        curves = [r for r in coll.rows if r["suite"] == name]
        files = {
            f"status_{name}.csv": info["status"],
            f"final_runs_{name}.csv": rows,
            f"final_summary_{name}.csv": summary,
            f"convergence_{name}.csv": info["convergence"],
            f"costs_{name}.csv": info["costs"],
            f"fb_diagnostics_{name}.csv": info["fb_diagnostics"],
            f"curves_long_{name}.csv": curves,
        }
        for fname, trows in files.items():
            sha = write_csv(tables_dir / fname, trows)
            jobs = sorted({r.get("eval_job") or r.get("job_id") or r.get("adapt_job") for r in trows
                           if (r.get("eval_job") or r.get("job_id") or r.get("adapt_job"))})
            prov_tables[f"tables/{fname}"] = {"sha256": sha, "suite": name, "n_rows": len(trows), "jobs": jobs}
        if figures:
            figs = curve_figures(coll, man, fig_dir, fixture)
            for metric, ylab in (("js_nats", "held-out JS (nats)"), ("success", "success rate")):
                f = final_figure(summary, man, fig_dir, metric, ylab, fixture)
                if f:
                    figs.append(f)
            for f in figs:
                prov_figs.append({"file": Path(f["file"]).relative_to(out).as_posix(), "sha256": sha256_file(f["file"]),
                                  "suite": name, "jobs": f["jobs"]})
            info["figures"] = figs
        per_suite[name] = info
    md = results_markdown(coll, per_suite, out, fixture)
    write_text_atomic(out / "RESULTS.md", md)
    jobs_prov = {}
    for jid, rec in coll.jobs.items():
        if rec["status"] != "completed" or rec["result"] is None:
            continue
        job = rec["job"]
        jobs_prov[jid] = {"type": rec["spec"]["type"], "result_sha256": sha256_file(rec["dir"] / "result.json"),
                          "identity_sha256": job["identity_sha256"], "code_sha256": job["code"]["sha256"],
                          "dependencies_sha256": job["dependencies"]["sha256"], "input_hashes": job["input_hashes"],
                          "device": rec["result"].get("device")}
    prov = {"schema": "il_memoir/provenance-v1", "generated_utc": utc_now(), "runs_root": relpath(coll.runs_root),
            "kinds": list(kinds), "test_fixture": fixture, "units": UNITS,
            "suites": {m["suite"]: {"manifest": relpath(m["_path"]), "manifest_sha256": m["_sha256"],
                                    "config_sha256": m["config_sha256"], "kind": m["kind"], "seeds": m["seeds"]}
                       for m in coll.suites},
            "tables": prov_tables, "figures": prov_figs, "jobs": jobs_prov,
            "results_md": {"file": "RESULTS.md", "sha256": sha256_file(out / "RESULTS.md")}}
    write_json_atomic(out / "provenance.json", prov)
    return {"out": out, "suites": list(per_suite), "n_eval_rows": len(coll.rows), "provenance": prov}
