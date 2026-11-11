#!/usr/bin/env python3
"""Independent verification of every number the report uses (Prompt 5).

Reads ONLY saved run files (runs/suites/*, runs/jobs/*) and recomputes, without importing
il_memoir: the final held-out metrics of every run, the per-condition means and sample
standard deviations, the operational-threshold table, the cost table, the FB diagnostics and
the derived counts quoted in the report.  It then

  1. compares each recomputed table with the code-generated tables in results/tables/
     (relative tolerance 1e-9), and
  2. writes the LaTeX tables of the report (report/generated/*.tex) and a JSON file of
     derived numbers (report/verify/numbers.json) from the recomputed values, and
  3. checks every claim registered in report/verify/claims.py against the recomputed values at
     the printed precision, and checks that the TeX sources print each registered literal
     through the macro \\vn{<claim id>}{<literal>} (claims.bind()); the numeric tokens of the
     sources outside \\vn are listed in report/verify/claims_coverage.txt for review by hand.

Usage (project root):  python3 report/verify/verify_results.py
Exit status 0 only if every comparison and every claim check passes.
Standard library only (no NumPy, no torch), so it runs anywhere Python 3.8+ runs.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
RESULTS = ROOT / "results" / "tables"
PILOT = RUNS / "pilot_results" / "tables"
OUT_GEN = ROOT / "report" / "generated"
OUT_VER = ROOT / "report" / "verify"
SUITES = ("core", "compact_ablations")
JS_MAX, SUCC_MIN = 0.10, 0.80

LOG: list = []
N_FAIL = 0


def log(msg: str) -> None:
    LOG.append(msg)


def check(ok: bool, what: str) -> None:
    global N_FAIL
    if not ok:
        N_FAIL += 1
        LOG.append("MISMATCH " + what)


def close(a, b, rel=1e-9, absol=1e-12) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= max(absol, rel * max(abs(float(a)), abs(float(b))))


def read_json(p: Path):
    with open(p, "r", encoding="utf-8") as h:
        return json.load(h)


def read_jsonl(p: Path):
    with open(p, "r", encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def read_csv(p: Path):
    with open(p, "r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def fnum(s):
    return None if s in ("", None) else float(s)


# ---------------------------------------------------------------------------------------------
# 1. load suites and jobs
# ---------------------------------------------------------------------------------------------
def method_label(labels, method):
    if method == "fpg":
        lab = f"f-PG[{labels.get('divergence')}]"
        v = labels.get("fpg_variant", "base")
        return lab if v == "base" else f"{lab}/{v}"
    if method in ("er", "dm"):
        lab = "ER_FB" if method == "er" else "DM_FB"
        v = labels.get("fb_variant", "base")
        return lab if v == "base" else f"{lab}/{v}"
    return "expert"


manifests = {s: read_json(RUNS / "suites" / s / "manifest.json") for s in SUITES}
jobs = {}
for s, man in manifests.items():
    for j in man["jobs"]:
        jid = j["job_id"]
        if jid in jobs:
            jobs[jid]["suites"].append(s)
            continue
        d = RUNS / "jobs" / jid
        rec = {"spec": j, "dir": d, "suites": [s], "status": read_json(d / "status.json"),
               "result": read_json(d / "result.json"), "job": read_json(d / "job.json")}
        if j["type"] in ("eval", "eval_expert"):
            rec["evals"] = read_jsonl(d / "outputs" / "evaluations.jsonl") if (d / "outputs" / "evaluations.jsonl").exists() else None
        jobs[jid] = rec

# status and attempts
status_counts = {}
for s, man in manifests.items():
    c = defaultdict(int)
    att = defaultdict(int)
    for j in man["jobs"]:
        st = jobs[j["job_id"]]["status"]
        c[(j["type"], st["status"])] += 1
        att[len(st["attempts"])] += 1
        outcomes = [a["outcome"] for a in st["attempts"]]
        check(outcomes == ["completed"], f"{j['job_id']}: attempts {outcomes}")
        check(jobs[j["job_id"]]["result"]["code_sha256"] == jobs[j["job_id"]]["job"]["code"]["sha256"],
              f"{j['job_id']}: result code hash differs from job code hash")
    status_counts[s] = {"by_type": {f"{k[0]}:{k[1]}": v for k, v in sorted(c.items())},
                        "attempts_hist": dict(att), "n_jobs": len(man["jobs"])}
    journal = read_jsonl(RUNS / "suites" / s / "outcomes.jsonl")
    status_counts[s]["journal_events"] = dict(defaultdict(int, {e: sum(1 for r in journal if r["event"] == e) for e in {r["event"] for r in journal}}))
    status_counts[s]["first_start"] = min(r["attempt_start_utc"] for r in journal)
    status_counts[s]["last_end"] = max(r["utc"] for r in journal)
    launches = read_jsonl(RUNS / "suites" / s / "launches.jsonl")
    status_counts[s]["launches"] = [{"utc": l["utc"], "git": l["git"], "code_sha256": l["code_sha256"],
                                     "device": l["device"]["device"], "torch_threads": l["torch_threads"],
                                     "counts_before": l["counts_before"]} for l in launches]
log(f"suites: { {s: status_counts[s]['n_jobs'] for s in SUITES} }; unique jobs {len(jobs)}")

# ---------------------------------------------------------------------------------------------
# 2. evaluation rows (all checkpoints) and final rows
# ---------------------------------------------------------------------------------------------
rows = []
for jid, rec in jobs.items():
    j = rec["spec"]
    if j["type"] not in ("eval", "eval_expert"):
        continue
    lab = j["labels"]
    for r in rec["evals"]:
        co = r.get("coords", {})
        rows.append({"suites": rec["suites"], "env": lab["env"], "layout": lab["layout"], "task": lab["task"],
                     "seed": int(lab["seed"]), "method": r["method"], "label": method_label(lab, r["method"]),
                     "eval_job": jid, "target_job": j["identity"].get("target"), "checkpoint": r["checkpoint"],
                     "iteration": co.get("iteration"), "env_steps": co.get("env_steps"),
                     "fb_update": co.get("fb_update"), "dm_step": co.get("dm_step"),
                     "final_fb_snapshot": co.get("final_fb_snapshot"), "final_dm_step": co.get("final_dm_step"),
                     "dm_loss": co.get("dm_loss"), "m": r["metrics"], "env_steps_eval": r.get("env_steps_eval")})
last_iter = defaultdict(lambda: -1)
for r in rows:
    if r["method"] == "fpg":
        last_iter[r["eval_job"]] = max(last_iter[r["eval_job"]], int(r["iteration"]))
for r in rows:
    if r["method"] == "fpg":
        r["final"] = int(r["iteration"]) == last_iter[r["eval_job"]]
    elif r["method"] == "er":
        r["final"] = bool(r["final_fb_snapshot"])
    elif r["method"] == "dm":
        r["final"] = bool(r["final_fb_snapshot"]) and bool(r["final_dm_step"])
    else:
        r["final"] = True
final = [r for r in rows if r["final"]]

# budgets reached at the final checkpoint
for r in final:
    if r["method"] == "fpg":
        bud = jobs[r["target_job"]]["spec"]["identity"]["budget_transitions"]
        check(r["env_steps"] == bud, f"{r['eval_job']}: final f-PG checkpoint at {r['env_steps']} != budget {bud}")
    if r["method"] in ("er", "dm"):
        fbj = jobs[r["target_job"]]["spec"]["identity"]["fb_job"]
        n_up = jobs[fbj]["spec"]["identity"]["fb"]["n_updates"]
        check(r["fb_update"] == n_up, f"{r['eval_job']}: final FB snapshot {r['fb_update']} != {n_up}")
        if r["method"] == "dm":
            check(r["dm_step"] == 300, f"{r['eval_job']}: final DM step {r['dm_step']}")

METRICS = ["js_nats", "success", "return", "gap", "js_sa_nats", "final_in_goal", "mc_success",
           "policy_entropy_visited_nats", "mc_post_hit_goal_region_frac", "demo_target_js_nats"]


def mean_sd(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None, None
    return sum(vals) / len(vals), (statistics.stdev(vals) if len(vals) >= 2 else None)


summary = {}   # (suite, env, layout, task, label) -> metric -> {mean, sd, values{seed: v}}
for s in SUITES:
    keys = defaultdict(dict)
    for r in final:
        if s in r["suites"]:
            keys[(r["env"], r["layout"], r["task"], r["label"])][r["seed"]] = r
    for k, by_seed in keys.items():
        rec = {"n": len(by_seed), "seeds": sorted(by_seed), "eval_jobs": sorted({x["eval_job"] for x in by_seed.values()})}
        for m in METRICS:
            vals = {sd: by_seed[sd]["m"].get(m) for sd in sorted(by_seed)}
            if all(v is None for v in vals.values()):
                continue
            mu, sdv = mean_sd(list(vals.values()))
            rec[m] = {"mean": mu, "sd": sdv, "values": vals}
        summary[(s,) + k] = rec

# compare with results/tables/final_summary_<suite>.csv and final_runs_<suite>.csv
for s in SUITES:
    tab = read_csv(RESULTS / f"final_summary_{s}.csv")
    seen = 0
    for t in tab:
        key = (s, t["env"], t["layout"], t["task"], t["method_label"])
        check(key in summary, f"summary row missing in recomputation: {key}")
        if key not in summary:
            continue
        rec = summary[key]
        check(int(t["n_completed"]) == rec["n"] == 3, f"{key}: n {t['n_completed']} vs {rec['n']}")
        for m in METRICS:
            if f"{m}_mean" in t and t[f"{m}_mean"] != "":
                check(m in rec, f"{key}: metric {m} absent in recomputation")
                if m in rec:
                    check(close(fnum(t[f"{m}_mean"]), rec[m]["mean"]), f"{key} {m} mean {t[m+'_mean']} vs {rec[m]['mean']}")
                    check(close(fnum(t[f"{m}_sd"]), rec[m]["sd"]), f"{key} {m} sd {t[m+'_sd']} vs {rec[m]['sd']}")
                    seen += 1
    log(f"final_summary_{s}.csv: {len(tab)} rows, {seen} metric means/SDs compared")
    tab = read_csv(RESULTS / f"final_runs_{s}.csv")
    n = 0
    for t in tab:
        match = [r for r in final if s in r["suites"] and r["eval_job"] == t["eval_job"] and r["checkpoint"] == t["checkpoint"]]
        check(len(match) == 1, f"final_runs row {t['eval_job']} {t['checkpoint']} not matched")
        if match:
            for m in ("js_nats", "success", "return", "gap", "js_sa_nats", "final_in_goal"):
                if t.get(m, "") != "":
                    check(close(fnum(t[m]), match[0]["m"].get(m)), f"final_runs {t['eval_job']} {m}")
                    n += 1
    log(f"final_runs_{s}.csv: {len(tab)} rows, {n} values compared")

# ---------------------------------------------------------------------------------------------
# 3. operational convergence
# ---------------------------------------------------------------------------------------------
conv = {}
for s in SUITES:
    curves = defaultdict(list)
    for r in rows:
        if s not in r["suites"] or r["method"] == "expert":
            continue
        if r["method"] == "fpg":
            axis, x = "learner env steps", r["env_steps"]
        elif r["method"] == "er" or (r["method"] == "dm" and r["final_dm_step"]):
            axis, x = "FB pretraining updates", r["fb_update"]
        else:
            continue
        curves[(r["env"], r["layout"], r["task"], r["label"], r["seed"])].append((x, r))
    for k, pts in curves.items():
        pts.sort(key=lambda p: p[0])
        ok = [x for x, r in pts if r["m"]["js_nats"] <= JS_MAX and r["m"]["success"] >= SUCC_MIN]
        last = pts[-1][1]
        conv[(s,) + k] = {"first": ok[0] if ok else None, "n_met": len(ok), "n_points": len(pts),
                          "met_last": last["m"]["js_nats"] <= JS_MAX and last["m"]["success"] >= SUCC_MIN,
                          "budget": pts[-1][0]}
    tab = read_csv(RESULTS / f"convergence_{s}.csv")
    for t in tab:
        key = (s, t["env"], t["layout"], t["task"], t["method_label"], int(t["seed"]))
        check(key in conv, f"convergence row missing: {key}")
        if key in conv:
            fr = None if t["first_reached"] == "" else int(float(t["first_reached"]))
            check(fr == conv[key]["first"], f"{key}: first reached {fr} vs {conv[key]['first']}")
            check((t["reached_at_last"] == "True") == conv[key]["met_last"], f"{key}: met at last")
    log(f"convergence_{s}.csv: {len(tab)} rows compared")

# ---------------------------------------------------------------------------------------------
# 4. costs
# ---------------------------------------------------------------------------------------------
def wall_sum(c):
    return sum(float(v) for v in c["wall_s"].values())


costs = {}
for s in SUITES:
    man = manifests[s]
    tasks_per_fb = defaultdict(set)
    for j in man["jobs"]:
        if j["type"] == "adapt":
            tasks_per_fb[j["identity"]["fb_job"]].add(j["labels"]["task"])
    eval_of = {j["identity"]["target"]: jobs[j["job_id"]] for j in man["jobs"] if j["type"] == "eval"}
    out = []
    for j in man["jobs"]:
        if j["type"] not in ("fpg", "adapt"):
            continue
        res = jobs[j["job_id"]]["result"]
        c = res["costs"]
        lab = j["labels"]
        ev = eval_of[j["job_id"]]["result"]["costs"]
        base = {"env": lab["env"], "layout": lab["layout"], "task": lab["task"], "seed": int(lab["seed"]),
                "job": j["job_id"], "demo": c["demonstrations"]["transitions"],
                "eval_steps": ev["env_steps_eval"], "eval_wall": wall_sum(ev)}
        if j["type"] == "fpg":
            out.append(dict(base, label=method_label(lab, "fpg"), train_steps=c["env_steps_train"], updates=c["updates"],
                            wall=wall_sum(c), one_steps=c["env_steps_train"], one_wall=wall_sum(c),
                            am_steps=c["env_steps_train"], am_wall=wall_sum(c), n_share=1))
            continue
        fbj = j["identity"]["fb_job"]
        fc = jobs[fbj]["result"]["costs"]
        pre_wall = wall_sum(fc) + float(fc.get("replay_generation_s") or 0.0)
        n_share = max(1, len(tasks_per_fb[fbj]))
        for meth in ("dm", "er"):
            a = float(c["final_snapshot_s"][meth])
            out.append(dict(base, label=method_label(lab, meth), fb_job=fbj, fb_steps=fc["replay_transitions"],
                            fb_resets=fc["replay_resets"], fb_updates=fc["updates"], fb_wall=pre_wall,
                            fb_train_wall=float(fc["wall_s"]["train"]), adapt_wall=a,
                            adapt_job_wall_total=sum(float(v) for v in c["wall_s"].values()),
                            dm_steps=c["dm_steps_per_snapshot"] if meth == "dm" else 0, adapt_steps=c["env_steps"],
                            one_steps=fc["replay_transitions"], one_wall=pre_wall + a,
                            am_steps=fc["replay_transitions"] / n_share, am_wall=pre_wall / n_share + a, n_share=n_share))
    costs[s] = out
    tab = read_csv(RESULTS / f"costs_{s}.csv")
    check(len(tab) == len(out), f"costs_{s}: {len(tab)} rows vs {len(out)}")
    for t in tab:
        m = [o for o in out if o["job"] == t["job_id"] and o["label"] == t["method_label"]]
        check(len(m) == 1, f"cost row {t['job_id']} {t['method_label']} unmatched")
        if m:
            o = m[0]
            for a, b in (("one_task_env_steps", "one_steps"), ("one_task_wall_s", "one_wall"),
                         ("amortized_env_steps", "am_steps"), ("amortized_wall_s", "am_wall"),
                         ("eval_env_steps", "eval_steps"), ("eval_wall_s", "eval_wall"),
                         ("expert_demo_transitions", "demo")):
                check(close(fnum(t[a]), o[b]), f"cost {t['job_id']} {t['method_label']} {a}: {t[a]} vs {o[b]}")
    log(f"costs_{s}.csv: {len(tab)} rows compared")

# ---------------------------------------------------------------------------------------------
# 5. FB diagnostics (final snapshot) and FB-6 model errors
# ---------------------------------------------------------------------------------------------
fbdiag = {}
for s in SUITES:
    out = []
    for j in manifests[s]["jobs"]:
        if j["type"] != "adapt":
            continue
        res = jobs[j["job_id"]]["result"]
        fbj = j["identity"]["fb_job"]
        diag = jobs[fbj]["result"]["summary"]["final_diagnostics"]
        fin = res["summary"]["final"]
        lab = j["labels"]
        fbme = {}
        for r in final:
            if r["target_job"] == j["job_id"] and r["method"] in ("er", "dm") and "fb_model_error" in r["m"]:
                fbme[r["method"]] = r["m"]["fb_model_error"]
        out.append({"env": lab["env"], "layout": lab["layout"], "task": lab["task"], "seed": int(lab["seed"]),
                    "variant": lab.get("fb_variant"), "fb_job": fbj, "adapt_job": j["job_id"],
                    "lmin": diag["gram_lambda_min"], "lmax": diag["gram_lambda_max"], "dev": diag["gram_dev_fro"],
                    "FmeanB": diag["F_dot_meanB_mean"], "FmeanB_sd": diag["F_dot_meanB_std"],
                    "er_norm": fin["er_norm_raw"], "dm_l0": fin["dm_initial_loss"], "dm_l1": fin["dm_final_loss"],
                    "er_ent": fin["u2_er"]["policy_entropy_nats"], "dm_ent": fin["u2_dm"]["policy_entropy_nats"],
                    "er_spread": fin["u2_er"]["score_spread_over_tau"], "dm_spread": fin["u2_dm"]["score_spread_over_tau"],
                    "er_tv": fbme.get("er", {}).get("sup_tv"), "dm_tv": fbme.get("dm", {}).get("sup_tv"),
                    "er_q": fbme.get("er", {}).get("q_error_max"), "dm_q": fbme.get("dm", {}).get("q_error_max"),
                    "er_qmean": fbme.get("er", {}).get("q_error_mean"), "dm_qmean": fbme.get("dm", {}).get("q_error_mean"),
                    "coverage": jobs[fbj]["result"]["summary"]["coverage"]})
    fbdiag[s] = out
    tab = read_csv(RESULTS / f"fb_diagnostics_{s}.csv")
    for t in tab:
        m = [o for o in out if o["adapt_job"] == t["adapt_job"]]
        check(len(m) == 1, f"fb diag row {t['adapt_job']} unmatched")
        if m:
            for a, b in (("gram_lambda_min", "lmin"), ("gram_dev_fro", "dev"), ("er_norm_raw", "er_norm"),
                         ("dm_entropy_nats", "dm_ent"), ("dm_fb6_sup_tv", "dm_tv"), ("dm_fb6_q_error_max", "dm_q")):
                check(close(fnum(t[a]), m[0][b]), f"fbdiag {t['adapt_job']} {a}")
    log(f"fb_diagnostics_{s}.csv: {len(tab)} rows compared")

# ---------------------------------------------------------------------------------------------
# 6. derived numbers quoted in the report
# ---------------------------------------------------------------------------------------------
N = {}


def rng(vals):
    vals = [v for v in vals if v is not None]
    return [min(vals), max(vals)] if vals else None


core_final = [r for r in final if "core" in r["suites"]]
abl_final = [r for r in final if "compact_ablations" in r["suites"]]


def finals(env=None, label=None, layout=None, suite_rows=None, task=None):
    src = core_final if suite_rows is None else suite_rows
    return [r for r in src if (env is None or r["env"] == env) and (label is None or r["label"] == label)
            and (layout is None or r["layout"] == layout) and (task is None or r["task"] == task)]


for env in ("finite", "continuous"):
    for lab in ("f-PG[js]", "ER_FB", "DM_FB", "expert"):
        fr = finals(env, lab)
        N[f"{env}.{lab}.n_runs"] = len(fr)
        N[f"{env}.{lab}.js_range"] = rng([r["m"]["js_nats"] for r in fr])
        N[f"{env}.{lab}.success_range"] = rng([r["m"]["success"] for r in fr])
        N[f"{env}.{lab}.n_met_last"] = sum(1 for r in fr if r["m"]["js_nats"] <= JS_MAX and r["m"]["success"] >= SUCC_MIN)
        N[f"{env}.{lab}.n_success_ge_0.8"] = sum(1 for r in fr if r["m"]["success"] >= SUCC_MIN)
        N[f"{env}.{lab}.n_js_le_0.1"] = sum(1 for r in fr if r["m"]["js_nats"] <= JS_MAX)
        if env == "finite":
            N[f"{env}.{lab}.gap_range"] = rng([r["m"]["gap"] for r in fr])
            N[f"{env}.{lab}.js_sa_range"] = rng([r["m"]["js_sa_nats"] for r in fr])
        else:
            N[f"{env}.{lab}.final_in_goal_range"] = rng([r["m"].get("final_in_goal") for r in fr])
        cv = [v for k, v in conv.items() if k[0] == "core" and k[1] == env and k[4] == lab]
        N[f"{env}.{lab}.conv_first"] = sorted([v["first"] for v in cv if v["first"] is not None])
        N[f"{env}.{lab}.conv_n_ever"] = sum(1 for v in cv if v["first"] is not None)
        N[f"{env}.{lab}.conv_n_last"] = sum(1 for v in cv if v["met_last"])
        N[f"{env}.{lab}.conv_n"] = len(cv)

# further final-checkpoint ranges quoted in the text (core): staying in the goal, entropy, returns
for env in ("finite", "continuous"):
    for lab in ("f-PG[js]", "ER_FB", "DM_FB", "expert"):
        fr = finals(env, lab)
        N[f"{env}.{lab}.fig_range"] = rng([r["m"].get("final_in_goal") for r in fr])
        N[f"{env}.{lab}.entropy_range"] = rng([r["m"].get("policy_entropy_visited_nats") for r in fr])
        N[f"{env}.{lab}.post_hit_range"] = rng([r["m"].get("mc_post_hit_goal_region_frac") for r in fr])
        N[f"{env}.{lab}.ret_range"] = rng([r["m"].get("return") for r in fr])
        N[f"{env}.{lab}.n_fig_ge_0.5"] = sum(1 for r in fr if (r["m"].get("final_in_goal") or 0.0) >= 0.5)

# compact ablations: mean over the 3 seeds of further metrics (four rooms, goal (5,5))
for lab in ("f-PG[js]", "f-PG[kl_dq]", "f-PG[kl_qd]", "ER_FB", "ER_FB/no_orth", "DM_FB", "DM_FB/no_orth"):
    fr = finals(label=lab, suite_rows=abl_final)
    for met in ("return", "gap", "policy_entropy_visited_nats", "final_in_goal", "js_sa_nats"):
        vals = [r["m"].get(met) for r in fr if r["m"].get(met) is not None]
        N[f"abl.{lab}.mean.{met}"] = sum(vals) / len(vals) if vals else None

# DM versus ER at the final snapshot, per (task, seed)
for env in ("finite", "continuous"):
    pairs = defaultdict(dict)
    for r in finals(env):
        if r["method"] in ("er", "dm"):
            pairs[(r["layout"], r["task"], r["seed"])][r["method"]] = r["m"]
    N[f"{env}.dm_vs_er.n_pairs"] = len(pairs)
    N[f"{env}.dm_vs_er.n_dm_js_higher"] = sum(1 for p in pairs.values() if p["dm"]["js_nats"] > p["er"]["js_nats"])
    N[f"{env}.dm_vs_er.n_dm_success_lower"] = sum(1 for p in pairs.values() if p["dm"]["success"] < p["er"]["success"])

# DM latent path at the final FB snapshot: model objective versus held-out JS
dm_path = {}
for r in rows:
    if r["method"] == "dm" and r["final_fb_snapshot"] and "core" in r["suites"]:
        dm_path.setdefault((r["env"], r["layout"], r["task"], r["seed"]), {})[r["dm_step"]] = (r["dm_loss"], r["m"]["js_nats"], r["m"]["success"])
n_loss_down = n_js_up = n_js_down = 0
for k, p in dm_path.items():
    l0, j0, s0 = p[0]
    l1, j1, s1 = p[300]
    n_loss_down += l1 < l0
    n_js_up += j1 > j0
    n_js_down += j1 < j0
N["dm_path.n"] = len(dm_path)
N["dm_path.n_model_loss_decreased"] = n_loss_down
N["dm_path.n_js_increased"] = n_js_up
N["dm_path.n_js_decreased"] = n_js_down
for env in ("finite", "continuous"):
    ks = [k for k in dm_path if k[0] == env]
    N[f"dm_path.{env}.n"] = len(ks)
    N[f"dm_path.{env}.n_js_increased"] = sum(1 for k in ks if dm_path[k][300][1] > dm_path[k][0][1])
    N[f"dm_path.{env}.n_success_decreased"] = sum(1 for k in ks if dm_path[k][300][2] < dm_path[k][0][2])
# DM model objective: fraction of total decrease achieved by step 100 (adapt logs, final snapshot)
frac100, zdev = [], []
for j in manifests["core"]["jobs"]:
    if j["type"] != "adapt":
        continue
    L = read_json(jobs[j["job_id"]]["dir"] / "outputs" / "latents.json")
    rec = [x for x in L["records"] if x["fb_update"] == max(y["fb_update"] for y in L["records"])][0]
    lg = rec["dm"]["log"]
    l0, l100, l300 = lg[0]["loss"], lg[100]["loss"], lg[300]["loss"]
    if l0 != l300:
        frac100.append((l0 - l100) / (l0 - l300))
    check(all(abs(x["z_norm"] - 4.0) < 1e-5 for x in lg), f"{j['job_id']}: DM iterate off the sphere")
    zdev.append(max(abs(x["z_norm"] - 4.0) for x in lg))
N["dm_path.frac_decrease_by_100"] = rng(frac100)
N["dm_path.max_z_norm_dev"] = max(zdev)
N["dm_path.n_frac100_ge_0.95"] = sum(1 for f in frac100 if f >= 0.95)
N["dm_path.n_runs_frac"] = len(frac100)

# ER and DM across FB snapshots: instability of the FB-IL outcome over pretraining
for env in ("finite", "continuous"):
    swings = []
    for k, v in conv.items():
        if k[0] == "core" and k[1] == env and k[4] in ("ER_FB", "DM_FB"):
            pass
    js_by = defaultdict(list)
    for r in rows:
        if "core" in r["suites"] and r["env"] == env and (r["method"] == "er" or (r["method"] == "dm" and r["final_dm_step"])):
            js_by[(r["layout"], r["task"], r["seed"], r["method"])].append((r["fb_update"], r["m"]["js_nats"], r["m"]["success"]))
    N[f"{env}.fbil.max_js_swing_between_snapshots"] = max(
        max(abs(a[1] - b[1]) for a, b in zip(sorted(v)[:-1], sorted(v)[1:])) for v in js_by.values())
    N[f"{env}.fbil.n_curves"] = len(js_by)
    N[f"{env}.fbil.n_curves_success_ge_0.8_somewhere_not_last"] = sum(
        1 for v in js_by.values() if any(x[2] >= SUCC_MIN for x in sorted(v)[:-1]) and sorted(v)[-1][2] < SUCC_MIN)

# FB diagnostics ranges
for env in ("finite", "continuous"):
    d = [x for x in fbdiag["core"] if x["env"] == env]
    N[f"{env}.fb.lmin_range"] = rng([x["lmin"] for x in d])
    N[f"{env}.fb.lmax_range"] = rng([x["lmax"] for x in d])
    N[f"{env}.fb.dev_range"] = rng([x["dev"] for x in d])
    N[f"{env}.fb.FmeanB_range"] = rng([x["FmeanB"] for x in d])
    N[f"{env}.fb.er_norm_range"] = rng([x["er_norm"] for x in d])
    N[f"{env}.fb.er_ent_range"] = rng([x["er_ent"] for x in d])
    N[f"{env}.fb.dm_ent_range"] = rng([x["dm_ent"] for x in d])
    N[f"{env}.fb.dm_spread_range"] = rng([x["dm_spread"] for x in d])
    if env == "finite":
        N[f"{env}.fb.tv_range"] = rng([x["er_tv"] for x in d] + [x["dm_tv"] for x in d])
        N[f"{env}.fb.q_range"] = rng([x["er_q"] for x in d] + [x["dm_q"] for x in d])
        N[f"{env}.fb.er_q_range"] = rng([x["er_q"] for x in d])
        N[f"{env}.fb.dm_q_range"] = rng([x["dm_q"] for x in d])
    N[f"{env}.fb.min_state_action_count"] = rng([x["coverage"].get("min_state_action_count") or x["coverage"].get("min_cell_action_count") for x in d])
    N[f"{env}.fb.coverage_all"] = all(x["coverage"].get("covers_all") for x in d)
# continuous lambda_min exact values (they print as 0.0000 in RESULTS.md)
N["continuous.fb.lmin_values"] = sorted({round(x["lmin"], 12) for x in fbdiag["core"] if x["env"] == "continuous"})
N["finite.fb.lmin_values"] = sorted({round(x["lmin"], 6) for x in fbdiag["core"] if x["env"] == "finite"})

# experts
for env in ("finite", "continuous"):
    ex = finals(env, "expert")
    N[f"{env}.expert.js_range"] = rng([r["m"]["js_nats"] for r in ex])
    N[f"{env}.expert.success_range"] = rng([r["m"]["success"] for r in ex])
    N[f"{env}.expert.demo_target_js_range"] = rng([r["m"].get("demo_target_js_nats") for r in ex])
    N[f"{env}.expert.return_range"] = rng([r["m"]["return"] for r in ex])
    if env == "continuous":
        N[f"{env}.expert.final_in_goal_range"] = rng([r["m"].get("final_in_goal") for r in ex])
        N[f"{env}.expert.post_hit_region_range"] = rng([r["m"].get("mc_post_hit_goal_region_frac") for r in ex])
VH = {}
for r in finals("finite", "expert"):
    VH[(r["layout"], r["task"])] = r["m"]["return"]
    cert = jobs[r["eval_job"]]["result"]["summary"]["last"]["expert"]["expert_certificate"]
    check(cert["H_step_gap_start"] == 0.0, f"{r['eval_job']}: expert H-step gap {cert['H_step_gap_start']}")
    check(cert["pi_residual"] < 1e-12 and cert["vi_residual"] < 1e-11, f"{r['eval_job']}: expert residuals")
N["finite.expert.V_H_star"] = {f"{k[0]}/{k[1]}": v for k, v in sorted(VH.items())}
N["finite.expert.max_pi_residual"] = max(jobs[r["eval_job"]]["result"]["summary"]["last"]["expert"]["expert_certificate"]["pi_residual"] for r in finals("finite", "expert"))
N["finite.expert.max_vi_residual"] = max(jobs[r["eval_job"]]["result"]["summary"]["last"]["expert"]["expert_certificate"]["vi_residual"] for r in finals("finite", "expert"))
N["finite.expert.goal_mass"] = {f"{r['layout']}/{r['task']}": r["m"]["goal_occupancy_mass"] for r in finals("finite", "expert") if r["seed"] == 0}

# ablations
for lab in ("f-PG[js]", "f-PG[kl_dq]", "f-PG[kl_qd]", "ER_FB", "DM_FB", "ER_FB/no_orth", "DM_FB/no_orth"):
    fr = finals(label=lab, suite_rows=abl_final)
    N[f"abl.{lab}.js_values"] = [r["m"]["js_nats"] for r in sorted(fr, key=lambda r: r["seed"])]
    N[f"abl.{lab}.success_values"] = [r["m"]["success"] for r in sorted(fr, key=lambda r: r["seed"])]
    cv = [v for k, v in conv.items() if k[0] == "compact_ablations" and k[4] == lab]
    N[f"abl.{lab}.conv_first"] = [v["first"] for v in sorted(cv, key=lambda v: 0)]
    N[f"abl.{lab}.conv_n_last"] = sum(1 for v in cv if v["met_last"])
ab = [x for x in fbdiag["compact_ablations"]]
for var in ("base", "no_orth"):
    d = [x for x in ab if x["variant"] == var]
    N[f"abl.fb.{var}.lmin_range"] = rng([x["lmin"] for x in d])
    N[f"abl.fb.{var}.lmax_range"] = rng([x["lmax"] for x in d])
    N[f"abl.fb.{var}.dev_range"] = rng([x["dev"] for x in d])
    N[f"abl.fb.{var}.er_norm_range"] = rng([x["er_norm"] for x in d])
    N[f"abl.fb.{var}.tv_range"] = rng([x["er_tv"] for x in d] + [x["dm_tv"] for x in d])
    N[f"abl.fb.{var}.q_range"] = rng([x["er_q"] for x in d] + [x["dm_q"] for x in d])
    N[f"abl.fb.{var}.FmeanB_range"] = rng([x["FmeanB"] for x in d])
    N[f"abl.fb.{var}.ent_range"] = rng([x["er_ent"] for x in d] + [x["dm_ent"] for x in d])

# costs: aggregates per environment and method (core)
def cagg(env, lab, field, suite="core"):
    vals = [o[field] for o in costs[suite] if o["env"] == env and o["label"] == lab]
    return mean_sd(vals)


for env in ("finite", "continuous"):
    for lab in ("f-PG[js]", "ER_FB", "DM_FB"):
        for f in ("one_steps", "one_wall", "am_steps", "am_wall", "eval_steps", "eval_wall"):
            N[f"cost.{env}.{lab}.{f}"] = cagg(env, lab, f)
    N[f"cost.{env}.fb_wall"] = mean_sd([o["fb_wall"] for o in costs["core"] if o["env"] == env and o["label"] == "DM_FB"])
    N[f"cost.{env}.fb_train_wall"] = mean_sd([o["fb_train_wall"] for o in costs["core"] if o["env"] == env and o["label"] == "DM_FB"])
    N[f"cost.{env}.dm_adapt_wall"] = mean_sd([o["adapt_wall"] for o in costs["core"] if o["env"] == env and o["label"] == "DM_FB"])
    N[f"cost.{env}.er_adapt_wall"] = mean_sd([o["adapt_wall"] for o in costs["core"] if o["env"] == env and o["label"] == "ER_FB"])
    N[f"cost.{env}.adapt_job_wall_total"] = mean_sd([o["adapt_job_wall_total"] for o in costs["core"] if o["env"] == env and o["label"] == "DM_FB"])
    N[f"cost.{env}.fb_updates"] = sorted({o["fb_updates"] for o in costs["core"] if o["env"] == env and "fb_updates" in o})
    N[f"cost.{env}.fb_steps"] = sorted({o["fb_steps"] for o in costs["core"] if o["env"] == env and "fb_steps" in o})
    N[f"cost.{env}.fb_resets"] = sorted({o["fb_resets"] for o in costs["core"] if o["env"] == env and "fb_resets" in o})
    N[f"cost.{env}.fpg_updates"] = sorted({o["updates"] for o in costs["core"] if o["env"] == env and o["label"] == "f-PG[js]"})

# total job wall time per suite and per type
for s in SUITES:
    tot = defaultdict(float)
    for j in manifests[s]["jobs"]:
        st = jobs[j["job_id"]]["status"]
        if s == "compact_ablations" and "core" in jobs[j["job_id"]]["suites"]:
            continue
        tot[j["type"]] += sum(float(a["wall_s"]) for a in st["attempts"])
    N[f"wall.{s}.by_type"] = dict(tot)
    N[f"wall.{s}.total"] = sum(tot.values())
    N[f"wall.{s}.first_start"] = status_counts[s]["first_start"]
    N[f"wall.{s}.last_end"] = status_counts[s]["last_end"]
N["peak_rss_mib"] = max(a.get("process_peak_rss_mib") or 0 for rec in jobs.values() for a in rec["status"]["attempts"])

# training data totals (core)
N["core.total_fpg_transitions"] = sum(o["train_steps"] for o in costs["core"] if o["label"].startswith("f-PG"))
N["core.total_fb_replay_transitions"] = sum(jobs[j["job_id"]]["result"]["costs"]["replay_transitions"] for j in manifests["core"]["jobs"] if j["type"] == "fb_pretrain")
N["core.total_fb_updates"] = sum(jobs[j["job_id"]]["result"]["costs"]["updates"] for j in manifests["core"]["jobs"] if j["type"] == "fb_pretrain")
N["core.total_eval_transitions"] = sum(jobs[j["job_id"]]["result"]["costs"]["env_steps_eval"] for j in manifests["core"]["jobs"] if j["type"] in ("eval", "eval_expert"))
N["abl.extra_eval_transitions"] = sum(jobs[j["job_id"]]["result"]["costs"]["env_steps_eval"] for j in manifests["compact_ablations"]["jobs"] if j["type"] in ("eval", "eval_expert") and "core" not in jobs[j["job_id"]]["suites"])
N["status"] = status_counts

# f-PG learning curve facts: JS at the start and at the last checkpoint (finite, continuous)
for env in ("finite", "continuous"):
    init = [r["m"]["js_nats"] for r in rows if "core" in r["suites"] and r["env"] == env and r["method"] == "fpg" and r["iteration"] == 0]
    N[f"{env}.fpg.js_at_iter0_range"] = rng(init)
    ent = [r["m"]["policy_entropy_visited_nats"] for r in finals(env, "f-PG[js]")]
    N[f"{env}.fpg.final_entropy_range"] = rng(ent)
# continuous f-PG: last-interval decrease of JS (mean over 9 runs), per 10,000 transitions
dec = []
for r in finals("continuous", "f-PG[js]"):
    prev = [x for x in rows if x["eval_job"] == r["eval_job"] and x["iteration"] == r["iteration"] - 5]
    dec.append(prev[0]["m"]["js_nats"] - r["m"]["js_nats"])
N["continuous.fpg.last_interval_js_decrease"] = rng(dec)
N["continuous.fpg.last_interval_js_decrease_mean"] = sum(dec) / len(dec)

# pilot (tuning-scope) numbers, read from the pilot's code-generated tables
pil = read_csv(PILOT / "final_runs_pilot.csv")
P = {}
for t in pil:
    P[f"{t['env']}.{t['method_label']}"] = {"js": float(t["js_nats"]), "success": float(t["success"])}
N["pilot"] = P
pd = read_csv(PILOT / "fb_diagnostics_pilot.csv")
N["pilot.fb"] = {f"{t['env']}.{t['fb_variant']}": {"lmin": float(t["gram_lambda_min"]), "dev": float(t["gram_dev_fro"]),
                                                   "dm_l0": float(t["dm_initial_loss"]), "dm_l1": float(t["dm_final_loss"]),
                                                   "tv": [fnum(t["er_fb6_sup_tv"]), fnum(t["dm_fb6_sup_tv"])],
                                                   "q": [fnum(t["er_fb6_q_error_max"]), fnum(t["dm_fb6_q_error_max"])]} for t in pd}

# mathematical constants used in the text
g, H = 0.95, 100
N["const.gamma_H"] = g ** H
N["const.Z_H"] = (1 - g ** H) / (1 - g)
N["const.gamma_H_over_1mg"] = g ** H / (1 - g)
N["const.gamma_60"] = g ** 60
N["const.gamma_60_over_1mg"] = g ** 60 / (1 - g)
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
astar = (Phi((1 / 7) / (2 * 0.01)) - 0.5) ** 2
N["const.alpha_star"] = astar
N["const.p_max"] = 1 / (2 * math.pi * 0.01 ** 2 * astar)
N["const.n_min"] = astar / (5 * 50 * (40 / 49))
N["const.kde_m_star"] = (Phi((1 / 7) / (2 * 0.05)) - 0.5) ** 2
N["const.tau_logA_over_1mg"] = 1.0 * math.log(5) / (1 - g)
N["const.kappa_H"] = sum((t + 1) * g ** t for t in range(H)) / sum(g ** t for t in range(H))

# ---------------------------------------------------------------------------------------------
# 6b. closeout accounting (October 2026; AUDIT.md M4, M6, L1-L4).  The comparisons with results/tables above keep
#     the definitions of scripts/analyze.py, whose outputs are hashed by results/provenance.json and stay unchanged.
#     The quantities below are derived from the same raw run files with the definitions stated here, and each
#     definition is checked; report/verify/costs_closeout.json records them (section 8).
# ---------------------------------------------------------------------------------------------
CLOSEOUT = {"conventions": {
    "per_task_phase_time": "sum of the phase timers in a job's result.json (costs.wall_s); for FB pretraining plus the "
                           "recorded replay_generation_s of its buffer, which the data-preparation step generated before "
                           "and outside the timed jobs; ER_FB/DM_FB adaptation: the final-snapshot timer "
                           "(costs.final_snapshot_s); used by tab_costs and tab_cost_components",
    "whole_job_wall_time": "attempt wall time in a job's status.json (start to end of the job: set-up, phases, checkpoints "
                           "and output writing); excludes replay generation; used for the suite totals",
    "fb_pretraining_sd": "mean and sample SD over the distinct FB pretraining jobs (each model counted once), not over "
                         "the goal rows that share a model",
    "adaptation_evaluation": "one evaluation job per adaptation job evaluates both ER_FB and DM_FB checkpoints; its "
                             "transitions and evaluate-phase time are a joint ER+DM cost, shown once for the pair and "
                             "counted once in totals",
    "unique_model_amortization": "pretraining divided by the number of distinct (layout, goal) tasks that adapt the same "
                                 "FB job in any final suite (core and compact ablations); results/tables/costs_*.csv "
                                 "(scripts/analyze.py) divide by the goals of the same suite only",
    "threshold_any_checkpoint": "a finite ER_FB/DM_FB curve meets the thresholds at some evaluated checkpoint if ER_FB at "
                                "any FB snapshot, DM_FB after its 300 latent steps at any snapshot, or a DM_FB "
                                "intermediate iterate (latent steps 50-250) at the final snapshot meets them; the "
                                "convergence rule of the protocol (section 3) is unchanged"}}

# L4: FB pretraining time per distinct model (the cost rows repeat each model once per goal)
for env in ("finite", "continuous"):
    per_model = {}
    for o in costs["core"]:
        if o["env"] == env and o["label"] == "DM_FB":
            if o["fb_job"] in per_model:
                check(close(per_model[o["fb_job"]], o["fb_wall"]), f"{o['fb_job']}: pretraining time differs between its goal rows")
            per_model[o["fb_job"]] = o["fb_wall"]
    N[f"cost.{env}.fb_wall_over_goal_rows"] = N[f"cost.{env}.fb_wall"]      # the earlier, row-weighted SD (kept for the record)
    N[f"cost.{env}.fb_wall"] = mean_sd(list(per_model.values()))             # mean +/- SD over distinct models
    N[f"cost.{env}.n_fb_models"] = len(per_model)
check(N["cost.finite.n_fb_models"] == 6 and N["cost.continuous.n_fb_models"] == 3, "core FB model counts")
CLOSEOUT["fb_pretraining_per_model"] = {env: {"n_models": N[f"cost.{env}.n_fb_models"], "mean_s": N[f"cost.{env}.fb_wall"][0],
                                              "sd_s": N[f"cost.{env}.fb_wall"][1],
                                              "sd_s_over_goal_rows_as_printed_before": N[f"cost.{env}.fb_wall_over_goal_rows"][1]}
                                        for env in ("finite", "continuous")}

# L1: the adaptation evaluation job is shared by ER_FB and DM_FB
for env in ("finite", "continuous"):
    nck = {"er": set(), "dm": set()}
    ntr = {"er": set(), "dm": set()}
    for j in manifests["core"]["jobs"]:
        if j["type"] != "eval" or j["identity"].get("target_type") != "adapt" or j["labels"]["env"] != env:
            continue
        ev = jobs[j["job_id"]]["evals"]
        check(all(r["method"] in ("er", "dm") for r in ev), f"{j['job_id']}: evaluation records other than ER_FB/DM_FB")
        for m in ("er", "dm"):
            nck[m].add(sum(1 for r in ev if r["method"] == m))
            ntr[m].add(sum(int(r["env_steps_eval"]) for r in ev if r["method"] == m))
        check(sum(int(r["env_steps_eval"]) for r in ev) == int(jobs[j["job_id"]]["result"]["costs"]["env_steps_eval"]),
              f"{j['job_id']}: ER_FB and DM_FB evaluation transitions do not add up to the job's")
    N[f"cost.{env}.adapt_eval.checkpoints"] = {m: sorted(v) for m, v in nck.items()}
    N[f"cost.{env}.adapt_eval.transitions"] = {m: sorted(v) for m, v in ntr.items()}
for o in costs["core"]:
    if o["label"] in ("ER_FB", "DM_FB"):
        twin = [p for p in costs["core"] if p["job"] == o["job"] and p["label"] != o["label"]]
        check(len(twin) == 1 and twin[0]["eval_steps"] == o["eval_steps"] and close(twin[0]["eval_wall"], o["eval_wall"]),
              f"{o['job']}: ER_FB and DM_FB rows do not share one evaluation job")
CLOSEOUT["adaptation_evaluation"] = {env: {"checkpoints_per_job": N[f"cost.{env}.adapt_eval.checkpoints"],
                                           "transitions_per_job": N[f"cost.{env}.adapt_eval.transitions"]}
                                     for env in ("finite", "continuous")}

# L2: per-type totals under both conventions, and the replay generation that only the per-task convention includes
for s in SUITES:
    ph = defaultdict(float)
    for j in manifests[s]["jobs"]:
        if s == "compact_ablations" and "core" in jobs[j["job_id"]]["suites"]:
            continue
        ph[j["type"]] += sum(float(v) for v in jobs[j["job_id"]]["result"]["costs"].get("wall_s", {}).values())
    N[f"phase.{s}.by_type"] = dict(ph)
N["replay_generation.core.total_s"] = sum(float(jobs[j["job_id"]]["result"]["costs"].get("replay_generation_s") or 0.0)
                                          for j in manifests["core"]["jobs"] if j["type"] == "fb_pretrain")
n_adapt_core = sum(1 for j in manifests["core"]["jobs"] if j["type"] == "adapt")
N["wall.core.adapt_per_job"] = N["wall.core.by_type"]["adapt"] / n_adapt_core
for t, w in N["wall.core.by_type"].items():
    check(N["phase.core.by_type"][t] <= w + 1e-9, f"core {t}: phase timers exceed the whole-job wall time")
CLOSEOUT["totals_core_s"] = {"whole_job_by_type": N["wall.core.by_type"], "phase_timers_by_type": N["phase.core.by_type"],
                             "replay_generation_total": N["replay_generation.core.total_s"],
                             "adapt_whole_job_mean_per_job": N["wall.core.adapt_per_job"]}

# L3: amortization over the distinct tasks that use a model in any final suite
tasks_all = defaultdict(set)
for s in SUITES:
    for j in manifests[s]["jobs"]:
        if j["type"] == "adapt":
            tasks_all[j["identity"]["fb_job"]].add((j["labels"]["layout"], j["labels"]["task"]))
um_rows, n_core_differ = [], 0
for s in SUITES:
    for o in costs[s]:
        if "fb_job" not in o:
            continue
        n_all = len(tasks_all[o["fb_job"]])
        am_s, am_w = o["fb_steps"] / n_all, o["fb_wall"] / n_all + o["adapt_wall"]
        differs = n_all != o["n_share"]
        n_core_differ += differs and s == "core"
        um_rows.append({"suite": s, "adapt_job": o["job"], "method_label": o["label"], "fb_job": o["fb_job"],
                        "n_goals_sharing_in_suite": o["n_share"], "n_goals_sharing_all_final_suites": n_all,
                        "amortized_env_steps_suite_convention": o["am_steps"], "amortized_wall_s_suite_convention": o["am_wall"],
                        "amortized_env_steps_unique_model": am_s, "amortized_wall_s_unique_model": am_w,
                        "differs": differs})
check(n_core_differ == 0, "core suite: the per-suite and unique-model amortizations differ")
N["cost.abl.rows_reused_model"] = sorted(f"{r['adapt_job']}|{r['method_label']}" for r in um_rows if r["differs"])
CLOSEOUT["unique_model_amortization_rows"] = um_rows

# M4: threshold crossings at any evaluated checkpoint (finite FB imitation, core)
met = lambda m: m["js_nats"] <= JS_MAX and m["success"] >= SUCC_MIN
crossed = []
n_never_any = 0
for k, v in conv.items():
    if k[0] != "core" or k[1] != "finite" or k[4] not in ("ER_FB", "DM_FB"):
        continue
    hit = v["first"] is not None
    if not hit and k[4] == "DM_FB":
        mid = [r for r in rows if "core" in r["suites"] and r["env"] == "finite" and r["method"] == "dm"
               and r["final_fb_snapshot"] and r["layout"] == k[2] and r["task"] == k[3] and r["seed"] == k[5]
               and r["dm_step"] not in (0, 300)]
        hits = sorted(r["dm_step"] for r in mid if met(r["m"]))
        last = [r for r in rows if "core" in r["suites"] and r["env"] == "finite" and r["method"] == "dm"
                and r["final_fb_snapshot"] and r["final_dm_step"] and r["layout"] == k[2] and r["task"] == k[3]
                and r["seed"] == k[5]]
        if hits:
            check(len(last) == 1 and not met(last[0]["m"]), f"{k}: crossing run meets the thresholds at its final step")
            crossed.append(f"{k[2]}/{k[3]}/{k[5]}@{hits[0]}")
            hit = True
    n_never_any += not hit
N["finite.fbil.dm_mid_cross"] = sorted(crossed)
N["finite.fbil.n_never_any_checkpoint"] = n_never_any
CLOSEOUT["thresholds_any_checkpoint"] = {"dm_runs_crossing_only_at_intermediate_steps": N["finite.fbil.dm_mid_cross"],
                                         "n_curves_never_meeting_at_any_evaluated_checkpoint": n_never_any,
                                         "n_curves_not_reached_under_protocol_rule":
                                             sum(1 for k, v in conv.items() if k[0] == "core" and k[1] == "finite"
                                                 and k[4] in ("ER_FB", "DM_FB") and v["first"] is None)}

# ---------------------------------------------------------------------------------------------
# 7. claim registry: the numbers printed in the report, recomputed and bound to the TeX sources
# ---------------------------------------------------------------------------------------------
sys.path.insert(0, str(OUT_VER))
n_claims = 0
try:
    import claims  # type: ignore
    for cid, printed, value, digits in claims.registry(N, summary, conv, costs, fbdiag, rows, final):
        n_claims += 1
        if value is None:
            check(False, f"claim {cid}: no recomputed value")
            continue
        if isinstance(printed, str):
            check(printed == value, f"claim {cid}: printed {printed!r} recomputed {value!r}")
        else:
            check(round(float(value), digits) == round(float(printed), digits),
                  f"claim {cid}: printed {printed} recomputed {round(float(value), digits + 3)}")
    log(f"registered claims: {len(claims.registered())} claim ids, {n_claims} comparisons with recomputed values")
    bind_checks, bstat = claims.bind()
    n_bad = 0
    for ok, what in bind_checks:
        check(ok, what)
        n_bad += not ok
    log(f"TeX binding: {bstat['occurrences']} \\vn occurrences in {bstat['files']} files name {bstat['ids_used']} claim ids; "
        f"{bstat['bound']} claims are registered as printed (each must occur), {bstat['unbound']} as not tied to the text; "
        f"{n_bad} binding errors")
    (OUT_VER / "claims_coverage.txt").write_text(claims.coverage(), encoding="utf-8")
except ModuleNotFoundError:
    log("claims.py not present yet: no prose claim checked")

# ---------------------------------------------------------------------------------------------
# 8. outputs
# ---------------------------------------------------------------------------------------------
OUT_VER.mkdir(parents=True, exist_ok=True)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


with open(OUT_VER / "numbers.json", "w", encoding="utf-8") as h:
    json.dump(jsonable(N), h, indent=1, sort_keys=True)
with open(OUT_VER / "summary_recomputed.json", "w", encoding="utf-8") as h:
    json.dump({"|".join(map(str, k)): v for k, v in summary.items()}, h, indent=1, sort_keys=True)
with open(OUT_VER / "convergence_recomputed.json", "w", encoding="utf-8") as h:
    json.dump({"|".join(map(str, k)): v for k, v in conv.items()}, h, indent=1, sort_keys=True)
with open(OUT_VER / "costs_closeout.json", "w", encoding="utf-8") as h:
    json.dump(jsonable(CLOSEOUT), h, indent=1, sort_keys=True)

try:
    import make_tables  # type: ignore
    written = make_tables.write_all(OUT_GEN, N, summary, conv, costs, fbdiag, rows, final, manifests, jobs)
    log(f"LaTeX tables written: {', '.join(written)}")
except ModuleNotFoundError:
    log("make_tables.py not present yet: no LaTeX table written")

log(f"RESULT: {'PASS' if N_FAIL == 0 else 'FAIL'} ({N_FAIL} mismatches)")
text = "\n".join(LOG) + "\n"
with open(OUT_VER / "verification_log.txt", "w", encoding="utf-8") as h:
    h.write(text)
print(text)
sys.exit(0 if N_FAIL == 0 else 1)
