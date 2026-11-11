"""Registry of the numbers printed in the report: each is recomputed from saved files and bound to the TeX.

verify_results.py imports this module, calls ``registry(N, summary, conv, costs, fbdiag, rows, final)``
and then ``bind()`` and ``coverage()``.

registry() registers every claim with the literal that the report prints and returns the entries
``(claim id, printed value, recomputed value, decimals)`` that verify_results.py compares:

* a number is compared with the recomputed value rounded to the printed number of decimals (a range
  "a to b" is two claims ``id.lo`` and ``id.hi``; a percentage is compared in percent; a literal
  ``m\\times10^{k}`` is compared through its mantissa; ``num_all`` gives one literal for several
  recomputed values, each of which must round to it; ``lt`` states a strict upper bound);
* a string entry (hashes, times, versions, counts printed as words, yes/no facts) must be equal.

The TeX prints a registered claim as ``\\vn{<claim id>}{<literal>}`` (preamble.tex: the macro
typesets its second argument).  bind() reads sections/*.tex and appendix/*.tex, without comments
and verbatim blocks, and checks that every \\vn occurrence names a registered claim and prints its
registered literal (TeX spacing and grouping braces aside), and that every claim registered with
``bound=True`` occurs at least once.  A literal printed through \\vn therefore equals, at its printed
precision, the value recomputed from the saved files.

Claims registered with ``bound=False`` are checked against their sources but are not tied to an
occurrence in the text: configuration settings (quoted many times, also in formulas and tables),
identities of particular runs, file hashes and yes/no facts.  coverage() lists the numeric tokens
of the TeX sources outside \\vn for review by hand (report/verify/claims_coverage.txt).

Sources: the recomputed numbers N and evaluation rows of verify_results.py (raw run files), the
suite manifests and job records, the pilot tables, the provenance audit, the test log, the laptop
environment record, the benchmark, the environment and KDE check reports, and closed-form constants
recomputed here.  Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
REPORT = ROOT / "report"
TEX_DIRS = ("sections", "appendix")

_REG: dict = {}   # claim id -> {"text": printed literal, "bound": bool}
WORDS = {"none": 0, "one": 1, "a single": 1, "two": 2, "three": 3, "four": 4, "nine": 9,
         "a third": Fraction(1, 3)}
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December")


def _rd(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _csv(p):
    import csv
    with open(p, encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def _norm(text) -> str:
    """A literal as typeset: math shifts, grouping braces, ties and thin or explicit spaces removed."""
    t = str(text)
    for a in ("\\,", "\\!", "\\ ", "~", "$", "{", "}", " "):
        t = t.replace(a, "")
    return t


def _parse(text):
    """(mantissa, power of ten, decimals) of a printed number: 20{,}000, -125.06, 9.7\\times10^{-13}, 10^{-5}."""
    t = _norm(text)
    m = re.fullmatch(r"(-?[0-9.,]+)\\times10\^(-?\d+)", t)
    if m:
        mant, e = m.group(1), int(m.group(2))
    else:
        m = re.fullmatch(r"10\^(-?\d+)", t)
        mant, e = ("1", int(m.group(1))) if m else (t, 0)
    mant = mant.replace(",", "")
    return float(mant), e, (len(mant.split(".")[1]) if "." in mant else 0)


def _phi_upper(x: float) -> float:
    """P(N(0,1) > x)."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _env_scalars() -> dict:
    """key: value scalars of configs/environment.yaml (simple line parser; no YAML library)."""
    out = {}
    for line in (ROOT / "configs" / "environment.yaml").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].rstrip()
        m = re.match(r"^\s*([A-Za-z_]+):\s*([-+0-9.eE]+)\s*$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def _utc(ts: str, n: int = 8) -> str:
    """hh:mm:ss (n=8) or hh:mm (n=5) of an ISO timestamp in UTC; any other zone fails the comparison."""
    return ts[11:11 + n] if ts.endswith("Z") else "not UTC: " + ts


def _long_date(ts: str) -> str:
    y, m, d = ts[:10].split("-")
    return f"{int(d)} {MONTHS[int(m) - 1]} {y}"


def registered() -> dict:
    return dict(_REG)


def registry(N, summary, conv, costs, fbdiag, rows, final):
    _REG.clear()
    C = []

    def reg(cid, text, bound):
        if cid in _REG:
            raise ValueError(f"claim id registered twice: {cid}")
        _REG[cid] = {"text": str(text), "bound": bool(bound)}

    def num(cid, text, value, scale=1.0, bound=True):
        """A number printed as ``text``; ``value`` is recomputed (times ``scale``, e.g. 100 for percent)."""
        reg(cid, text, bound)
        mant, e, dec = _parse(text)
        C.append((cid, mant, None if value is None else float(value) * scale / 10.0 ** e, dec))

    def num_all(cid, text, values, scale=1.0, bound=True):
        """One printed literal for several recomputed values, each of which must round to it."""
        reg(cid, text, bound)
        mant, e, dec = _parse(text)
        values = list(values or [])
        if not values:
            C.append((cid, mant, None, dec))
        for i, v in enumerate(values):
            C.append((f"{cid}[{i}]", mant, None if v is None else float(v) * scale / 10.0 ** e, dec))

    def rng(cid, lo, hi, pair, scale=1.0, bound=True):
        num(cid + ".lo", lo, None if pair is None else pair[0], scale, bound)
        num(cid + ".hi", hi, None if pair is None else pair[1], scale, bound)

    def lt(cid, text, value, scale=1.0, bound=True):
        """A strict upper bound printed as ``text``."""
        reg(cid, text, bound)
        mant, e, _ = _parse(text)
        C.append((cid, "True", None if value is None else str(float(value) * scale < mant * 10.0 ** e), 0))

    def word(cid, text, values, bound=True):
        """A count printed as a word ("none", "a single", "a third", ...): exact equality."""
        reg(cid, text, bound)
        vals = list(values) if isinstance(values, (list, tuple)) else [values]
        for i, v in enumerate(vals):
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            C.append((cid if len(vals) == 1 else f"{cid}[{i}]", str(WORDS[text]), None if v is None else str(v), 0))

    def fact(cid, printed, value, bound=False):
        """A string fact, equal as strings; with bound=True ``printed`` is also the TeX literal."""
        reg(cid, printed, bound)
        C.append((cid, str(printed), None if value is None else str(value), 0))

    F, Cn = "finite", "continuous"
    fpg, er, dm = "f-PG[js]", "ER_FB", "DM_FB"
    man = {s: _rd(RUNS / "suites" / s / "manifest.json") for s in ("core", "compact_ablations")}
    cfg = man["core"]["config"]
    env = _env_scalars()
    gamma, H = env["gamma"], int(env["horizon"])
    ZH = (1 - gamma ** H) / (1 - gamma)
    fb_f, fb_c = cfg["methods"]["fb"]["finite"], cfg["methods"]["fb"]["continuous"]
    fpg_f, fpg_c = cfg["methods"]["fpg"]["finite"], cfg["methods"]["fpg"]["continuous"]
    ad = cfg["methods"]["adaptation"]
    rep_f, rep_c = cfg["data"]["replay"]["finite"], cfg["data"]["replay"]["continuous"]
    K = ad["dm"]["K"]

    # ------------------------------------------------------------------ environment check report
    rep = (ROOT / "audit" / "prompt3A" / "env_check_device" / "env_report.md").read_text(encoding="utf-8")
    fact("env.report_config", man["core"]["environment_config"]["sha256"],
         re.search(r"Config sha256: `([0-9a-f]+)`", rep).group(1))
    lay = {}
    for blk in re.split(r"^### ", rep, flags=re.M)[1:]:
        m = re.match(r"(\w+) \((\d+) free cells\)", blk)
        if m:
            lay[m.group(1)] = (int(m.group(2)), int(re.search(r"rank P~ = (\d+)", blk).group(1)),
                               float(re.search(r"smallest singular value ([0-9.]+)", blk).group(1)),
                               int(re.search(r"D = (\d+)", blk).group(1)))
    num("env.cells.open", "49", lay["open"][0])
    num("env.cells.4r", "40", lay["four_rooms"][0])
    num("env.walls", "9", lay["open"][0] - lay["four_rooms"][0])
    num("env.rank.open", "49", lay["open"][1])
    num("env.rank.4r", "40", lay["four_rooms"][1])
    num_all("env.sv", "2.01", [v[2] for v in lay.values()])
    num_all("env.D", "10", [v[3] for v in lay.values()])
    hm = re.search(r"(\d+) trajectories of H = \d+: (\d+) of (\d+) cells visited; minimum count \d+; "
                   r"(\d+) bins of zero free area dropped", rep)
    nb = int(env["histogram_bins"])
    num("env.unif_traj", "2000", int(hm.group(1)))
    num("env.hist_cells", "356", int(hm.group(3)))
    num("env.hist_dropped", "44", int(hm.group(4)))
    num("env.hist_total", "400", nb * nb)
    fact("env.hist_partition", "True", int(hm.group(3)) + int(hm.group(4)) == nb * nb)
    cap = int(re.search(r"redraw cap \((\d+) draws\)", rep).group(1))
    num("env.kcap", "1000", cap - 1)
    num("env.cap", "1001", cap)
    # shortest paths from the start S to the goals G on the ASCII maps of the report (breadth-first search)
    dist = {}
    for name, body in re.findall(r"^### (\w+) \(\d+ free cells\)\n+```\n(.*?)```", rep, flags=re.M | re.S):
        grid = {}
        for row in body.splitlines():
            mrow = re.match(r"y=(\d+) (.*)", row)
            if mrow:
                for x, ch in enumerate(mrow.group(2).split()):
                    grid[(x, int(mrow.group(1)))] = ch
        start = next(c for c, ch in grid.items() if ch == "S")
        d, frontier = {start: 0}, [start]
        while frontier:
            nxt = []
            for (x, y) in frontier:
                for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if grid.get(c, "#") != "#" and c not in d:
                        d[c] = d[(x, y)] + 1
                        nxt.append(c)
            frontier = nxt
        dist[name] = {c: d.get(c) for c, ch in grid.items() if ch == "G"}
    num_all("env.dist.side", "4", [dist[l].get(g) for l in ("open", "four_rooms") for g in ((5, 1), (1, 5))])
    num_all("env.dist.far", "8", [dist[l].get((5, 5)) for l in ("open", "four_rooms")])

    # ------------------------------------------------------------------ closed-form constants
    num("const.ZH", "19.88", ZH)
    num("const.gH", "5.92\\times10^{-3}", gamma ** H)
    num("const.gH.006", "0.006", gamma ** H)
    num("const.gH_over", "0.118", gamma ** H / (1 - gamma))
    num("const.gH_over4", "0.1184", gamma ** H / (1 - gamma))
    lt("const.gH_over_lt", "0.119", gamma ** H / (1 - gamma))
    num("const.mass", "20", 1 / (1 - gamma))
    num("const.taulog5", "32.2", fb_f["settings"]["tau"] * math.log(5) / (1 - gamma))
    num("const.log5", "1.61", math.log(5))
    num("const.log2", "0.693", math.log(2))
    num("const.gK_over", "0.92", gamma ** K / (1 - gamma))
    num("const.gK_pct", "4.6", gamma ** K, 100)
    num("const.n_starts", "41", H - K + 1)
    # dwell example (s02, appendix F): absorbing goal reached at step 4
    num("const.dwell.qE", "0.8565", (gamma ** 3 - gamma ** H) / (1 - gamma ** H))
    num("const.dwell.rho", "0.97", (H - 3) / H)
    # continuous kernel constants (appendix G)
    L, sigma, h = 1 / 7, env["noise_std"], env["kde_bandwidth"]
    alpha_star = (0.5 - _phi_upper(L / (2 * sigma))) ** 2
    seg_c = rep_c["segment_length"]
    num("const.L2sigma", "7.14", L / (2 * sigma))
    num("const.alpha_gap", "4.6\\times10^{-13}", 0.25 - alpha_star)
    num("const.cap_log10", "-125.06", cap * math.log10(1 - alpha_star))
    lt("const.cap_lt", "10^{-125}", 10 ** (cap * math.log10(1 - alpha_star)))
    num("const.n_min_den", "204.1", 5 * seg_c * 40 / 49)
    num("const.n_min", "1.2\\times10^{-3}", alpha_star / (5 * seg_c * 40 / 49))
    num("const.p_max", "6366", N["const.p_max"])
    num("const.Lh", "2.9", L / h)
    num("const.kde_tail_half_pct", "0.4", _phi_upper(L / h) / 0.5, 100)
    num("const.kde_tail_max_pct", "1.2", _phi_upper(L / h) / N["const.kde_m_star"], 100)
    num("const.kde_m_star", "0.179", N["const.kde_m_star"])
    # entropy example (appendix C)
    num("const.ent.state", "0.637", math.log(3) - 2 / 3 * math.log(2))
    num("const.ent.action", "1.040", -(2 * 0.25 * math.log(0.25) + 0.5 * math.log(0.5)))

    # ------------------------------------------------------------------ KDE visibility measurement
    kde = (ROOT / "audit" / "prompt3A" / "kde_visibility_device.txt").read_text(encoding="utf-8")
    num("kde.median", "0", float(re.search(r"median ([0-9.]+)", kde).group(1)), 100)
    num("kde.mean_pct", "1.1", float(re.search(r"mean ([0-9.]+)", kde).group(1)), 100)
    num("kde.p95_pct", "7.3", float(re.search(r"95th percentile ([0-9.]+)", kde).group(1)), 100)
    mx = re.search(r"refined search.*max ([0-9.]+) at \(([0-9.]+) L, ([0-9.]+) L\)", kde)
    num("kde.max_pct", "31.5", float(mx.group(1)), 100)
    x0, y0 = float(mx.group(2)), float(mx.group(3))
    num("kde.max_dist", "0.05", math.hypot(x0 - round(x0), y0 - round(y0)))
    num("kde.face_pct", "8", float(re.search(r"middle of the left face.*?: ([0-9.]+)", kde).group(1)), 100)
    num("kde.doorway_pct", "16", float(re.search(r"doorway \(3,1\).*?: ([0-9.]+)", kde).group(1)), 100)
    g = re.search(r"on a (\d+)x(\d+) grid", kde)
    fact("kde.grid", "70\\times70", f"{g.group(1)}\\times{g.group(2)}", bound=True)

    # ------------------------------------------------------------------ environment figures (appendix copies)
    lap = (ROOT / "audit" / "prompt3A" / "env_check_laptop" / "env_report.md").read_text(encoding="utf-8")
    fact("fig.env.seed", "20261005", re.search(r"seed (\d+)", lap).group(1), bound=True)
    fact("fig.env.config_same", man["core"]["environment_config"]["sha256"],
         re.search(r"Config sha256: `([0-9a-f]+)`", lap).group(1))
    for a, b in (("env_finite_expert_maps.png", "finite_expert_maps.png"),
                 ("env_continuous_expert_rollouts.png", "continuous_expert_rollouts.png")):
        fact(f"fig.env.copy.{a}", hashlib.sha256((ROOT / "audit/prompt3A/env_check_laptop" / b).read_bytes()).hexdigest(),
             hashlib.sha256((REPORT / "figures" / a).read_bytes()).hexdigest())

    # ------------------------------------------------------------------ settings (facts, not tied to the text)
    fact("set.gamma", "0.95", gamma)
    fact("set.H", "100", H)
    fact("set.slip", "0.1", env["slip_prob"])
    fact("set.noise", "0.01", env["noise_std"])
    fact("set.h", "0.05", env["kde_bandwidth"])
    fact("set.eps", "0.01", env["smoothing_eps"])
    fact("set.bins", "20", nb)
    fact("set.w_cells", "0.5", env["reward_width_cells"])
    fact("set.goal_radius_cells", "0.5", env["goal_radius_cells"])
    fact("set.seeds", "[0, 1, 2]", man["core"]["seeds"])
    fact("set.n_demos", "4", cfg["data"]["n_demos"])
    fact("set.n_reference", "2000", cfg["data"]["reference"]["continuous"]["n_trajectories"])
    fact("set.replay", "50000/100000", f"{rep_f['n_transitions']}/{rep_c['n_transitions']}")
    fact("set.segment", "50/50", f"{rep_f['segment_length']}/{rep_c['segment_length']}")
    fact("set.eval_rollouts", "200/500", f"{cfg['evaluation']['finite']['mc_rollouts']}/{cfg['evaluation']['continuous']['rollouts']}")
    fact("set.thresholds", "0.1/0.8", f"{cfg['evaluation']['convergence']['js_max_nats']}/{cfg['evaluation']['convergence']['success_min']}")
    fact("set.threads", "4", cfg["runner"]["cpu_threads"])
    fact("set.fpg.lr", "0.3/0.003", f"{fpg_f['settings']['lr']}/{fpg_c['settings']['lr']}")
    fact("set.fpg.ntraj", "20/20", f"{fpg_f['settings']['n_traj']}/{fpg_c['settings']['n_traj']}")
    fact("set.fpg.folds", "2/2", f"{fpg_f['settings']['n_folds']}/{fpg_c['settings']['n_folds']}")
    fact("set.fpg.budget", "50000/100000", f"{fpg_f['budget_transitions']}/{fpg_c['budget_transitions']}")
    fact("set.fpg.eval_every", "5/5", f"{fpg_f['eval_every_iterations']}/{fpg_c['eval_every_iterations']}")
    fact("set.fpg.ppo_entropy", "0/0.0", f"{fpg_f['settings']['ppo_epochs']}/{fpg_f['settings']['entropy_coef']}")
    for k, v in (("d", "16"), ("hidden", "64"), ("n_hidden_layers", "2"), ("batch_size", "128"),
                 ("target_batch_size", "128"), ("lr", "0.0003"), ("orth_coef", "1.0"), ("target_momentum", "0.99"),
                 ("tau", "1.0"), ("activation", "silu"), ("first_layer_norm", "True")):
        fact(f"set.fb.{k}", f"{v}/{v}", f"{fb_f['settings'][k]}/{fb_c['settings'][k]}")
    fact("set.fb.updates", "10000/20000", f"{fb_f['settings']['n_updates']}/{fb_c['settings']['n_updates']}")
    fact("set.fb.snapshots", "2500/5000", f"{fb_f['snapshot_every']}/{fb_c['snapshot_every']}")
    jfb = next(j for j in man["core"]["jobs"] if j["type"] == "fb_pretrain")
    idfb = _rd(RUNS / "jobs" / jfb["job_id"] / "job.json")["identity"]["fb"]
    fact("set.fb.radius", "4", int(math.isqrt(idfb["d"])) if idfb["latent_radius"] is None else idfb["latent_radius"])
    fact("set.dm", "60/True/0.05/300/True", f"{ad['dm']['K']}/{ad['dm']['include_t0']}/{ad['dm']['lr']}/{ad['dm']['steps']}/{ad['dm']['project']}")
    fact("set.dm.steps_logged", "[0, 50, 100, 150, 200, 250, 300]", ad["dm_snapshot_steps"])
    j0 = next(j for j in man["core"]["jobs"] if j["type"] == "fpg" and j["labels"]["env"] == Cn)
    ident = _rd(RUNS / "jobs" / j0["job_id"] / "job.json")["identity"]["fpg"]
    fact("set.fpg.mlp", "tanh/64/2/0.01", f"{ident['activation']}/{ident['hidden']}/{ident['n_hidden_layers']}/{ident['out_init_scale']}")
    fact("set.fpg.baseline", "time", ident["baseline"])
    pseeds = sorted({t["seed"] for t in _csv(RUNS / "pilot_results" / "tables" / "final_runs_pilot.csv")})
    fact("pilot.seed", "999", "/".join(pseeds), bound=True)
    # settings quoted as quantities in the results and cost statements (bound)
    num("budget.fpg.c", "100,000", fpg_c["budget_transitions"])
    num("replay.f", "50,000", rep_f["n_transitions"])
    num("replay.c", "100,000", rep_c["n_transitions"])
    num_all("budget.same.f", "50,000", [fpg_f["budget_transitions"], rep_f["n_transitions"]])
    num_all("budget.same.c", "100,000", [fpg_c["budget_transitions"], rep_c["n_transitions"]])
    num_all("replay.segment", "50", [rep_f["segment_length"], rep_c["segment_length"]])
    num("fb.updates.c", "20,000", fb_c["settings"]["n_updates"])
    word("fb.n_snapshots", "four", [b["settings"]["n_updates"] // b["snapshot_every"]
                                    if b["settings"]["n_updates"] % b["snapshot_every"] == 0 else None for b in (fb_f, fb_c)])
    num("fpg.interval.c", "10,000", fpg_c["eval_every_iterations"] * fpg_c["settings"]["n_traj"] * H)
    # closeout (AUDIT.md L8): evaluation interval of finite f-PG and the finite FB update budget
    num("fpg.interval.f", "10,000", fpg_f["eval_every_iterations"] * fpg_f["settings"]["n_traj"] * H)
    num("fb.updates.f", "10,000", fb_f["settings"]["n_updates"])
    # evaluation sizes recorded in the rows
    mcn = lambda e: sorted({r["m"].get("mc_n") for r in rows if r["env"] == e and r["m"].get("mc_n") is not None})
    fact("rows.mc_n", "[200]/[500]", f"{mcn(F)}/{mcn(Cn)}")
    fact("rows.mc_n_everywhere", "True", all(r["m"].get("mc_n") is not None for r in rows))
    fact("rows.n_reference", "2000", sorted({r["m"].get("n_reference") for r in rows if r["env"] == Cn and r["method"] == "expert"})[0])
    fact("rows.fpg_checkpoints.finite", "[0, 10000, 20000, 30000, 40000, 50000]",
         sorted({r["env_steps"] for r in rows if r["env"] == F and r["method"] == "fpg"}))
    fact("rows.dm_steps", "[0, 50, 100, 150, 200, 250, 300]",
         sorted({r["dm_step"] for r in rows if r["method"] == "dm" and r["final_fb_snapshot"]}))

    # ------------------------------------------------------------------ run counts
    nr = lambda e, lab: N[f"{e}.{lab}.n_runs"]
    num_all("n.runs.finite", "18", [nr(F, fpg), nr(F, er), nr(F, dm)])
    num_all("n.runs.cont", "9", [nr(Cn, fpg), nr(Cn, er), nr(Cn, dm)])
    word("n.runs.cont.word", "nine", [nr(Cn, fpg), nr(Cn, er), nr(Cn, dm)])
    num_all("n.fpg.per_maze", "9", [sum(1 for k in conv if k[0] == "core" and k[1] == e and k[2] == lo and k[4] == fpg)
                                    for e, lo in ((F, "open"), (F, "four_rooms"), (Cn, "four_rooms"))])

    # ------------------------------------------------------------------ abstract (s00)
    fc = N[f"finite.{fpg}.conv_first"]
    num("abs.fpg.n", "18", N[f"finite.{fpg}.conv_n_ever"])
    num("abs.fpg.first.lo", "20,000", min(fc))
    num("abs.fpg.first.hi", "30,000", max(fc))
    num("abs.er.last", "2", N[f"finite.{er}.conv_n_last"])
    word("abs.dm.last", "none", N[f"finite.{dm}.conv_n_last"])
    num("abs.dm.loss_down", "27", N["dm_path.n_model_loss_decreased"])
    num("abs.dm.n", "27", N["dm_path.n"])
    fact("abs.dm.loss_all", "True", N["dm_path.n_model_loss_decreased"] == N["dm_path.n"])
    num("abs.dm.js_up", "20", N["dm_path.n_js_increased"])
    num("abs.dm.js_up_finite", "16", N["dm_path.finite.n_js_increased"])
    num("abs.dm.n_finite", "18", N["dm_path.finite.n"])
    num("abs.dm.n_cont", "9", N["dm_path.continuous.n"])
    num("abs.cont.none", "0", sum(N[f"continuous.{m}.conv_n_ever"] for m in (fpg, er, dm)), bound=False)
    rng("abs.cont.fpg.js", "0.24", "0.41", N[f"continuous.{fpg}.js_range"])
    rng("abs.cont.fpg.succ", "0.92", "0.99", N[f"continuous.{fpg}.success_range"])

    # ------------------------------------------------------------------ s06 (FB)
    rng("s06.tv", "36.2", "79.7", N["finite.fb.tv_range"])
    rng("s06.q", "2.69", "15.22", N["finite.fb.q_range"])

    # ------------------------------------------------------------------ s09 (hyperparameters)
    P = N["pilot"]
    num("s09.hist.js", "0.488", P["continuous.f-PG[js]/histogram"]["js"])
    num("s09.kde.js", "0.486", P["continuous.f-PG[js]"]["js"])
    pfb = {(t["env"], t["fb_variant"]): t for t in _csv(RUNS / "pilot_results" / "tables" / "fb_diagnostics_pilot.csv")}
    num("s09.pilot.spread.0.1", "110", float(pfb[(F, "base")]["dm_score_spread_over_tau"]))
    num("s09.pilot.spread.0.3", "35", float(pfb[(F, "tau_0p3")]["dm_score_spread_over_tau"]))
    num("s09.pilot.spread.1", "11", float(pfb[(F, "tau_1")]["dm_score_spread_over_tau"]))
    num("s09.pilot.ent.0.1", "0.05", float(pfb[(F, "base")]["dm_entropy_nats"]))
    num("s09.pilot.ent.0.3", "0.06", float(pfb[(F, "tau_0p3")]["dm_entropy_nats"]))
    num("s09.pilot.ent.1", "0.62", float(pfb[(F, "tau_1")]["dm_entropy_nats"]))
    erf = {t: P[f"finite.ER_FB{s}"] for t, s in (("0.1", ""), ("0.3", "/tau_0p3"), ("1", "/tau_1"))}
    erc = {t: P[f"continuous.ER_FB{s}"] for t, s in (("0.1", ""), ("0.3", "/tau_0p3"), ("1", "/tau_1"))}
    dmf = {t: P[f"finite.DM_FB{s}"] for t, s in (("0.1", ""), ("0.3", "/tau_0p3"), ("1", "/tau_1"))}
    fact("s09.pilot.er_finite_best_tau", "1/1", f"{min(erf, key=lambda t: erf[t]['js'])}/{max(erf, key=lambda t: erf[t]['success'])}")
    fact("s09.pilot.er_cont_best_success_tau1", "1", max(erc, key=lambda t: erc[t]["success"]))
    fact("s09.pilot.dm_finite_best_js_tau0.3", "0.3", min(dmf, key=lambda t: dmf[t]["js"]))
    # the pilot selection rules of s10.4: f-PG lowest held-out JS among the three learning rates (both mazes);
    # FB temperature with the best mean JS and mean success over ER_FB and DM_FB in both mazes
    lr_best = [min(("_lr_low", "", "_lr_high"), key=lambda v: P[f"{e}.f-PG[js]{v.replace('_', '/', 1) if v else ''}"]["js"]) for e in (F, Cn)]
    fact("s10.pilot.lr_rule", "['_lr_high', '_lr_high']", lr_best)
    tau_js = {t: sum(P[f"{e}.{m}{s}"]["js"] for e in (F, Cn) for m in ("ER_FB", "DM_FB")) / 4
              for t, s in (("0.1", ""), ("0.3", "/tau_0p3"), ("1", "/tau_1"))}
    tau_su = {t: sum(P[f"{e}.{m}{s}"]["success"] for e in (F, Cn) for m in ("ER_FB", "DM_FB")) / 4
              for t, s in (("0.1", ""), ("0.3", "/tau_0p3"), ("1", "/tau_1"))}
    fact("s10.pilot.tau_rule", "1/1", f"{min(tau_js, key=tau_js.get)}/{max(tau_su, key=tau_su.get)}")
    rng("s09.erz.base", "2.6", "2.9", N["abl.fb.base.er_norm_range"])
    rng("s09.erz.noorth", "13.8", "17.2", N["abl.fb.no_orth.er_norm_range"])
    num("s09.swing", "0.53", N["finite.fbil.max_js_swing_between_snapshots"])
    num("s09.lost", "12", N["finite.fbil.n_curves_success_ge_0.8_somewhere_not_last"])
    num("s09.curves", "36", N["finite.fbil.n_curves"])
    num("s09.frac100", "97", math.floor(100 * N["dm_path.frac_decrease_by_100"][0]))
    ak = N["abl.f-PG[kl_qd].conv_first"]
    num("s09.klqd.first.lo", "40,000", min(ak))
    num("s09.klqd.first.hi", "50,000", max(ak))
    aj = N["abl.f-PG[js].conv_first"] + N["abl.f-PG[kl_dq].conv_first"]
    num("s09.js_kldq.first.lo", "20,000", min(aj))
    num("s09.js_kldq.first.hi", "30,000", max(aj))

    # ------------------------------------------------------------------ s10 (protocol)
    num("s10.vi_res", "9.7\\times10^{-13}", N["finite.expert.max_vi_residual"])
    num("s10.pi_res", "1.1\\times10^{-14}", N["finite.expert.max_pi_residual"])
    fact("s10.expert_gap0", "0.0/0.0", "/".join(str(x) for x in N["finite.expert.gap_range"]))
    rng("s10.cont_expert_fig", "72", "76", N["continuous.expert.fig_range"], 100)
    fact("s10.cont_expert_success", "1.0/1.0", "/".join(str(x) for x in N["continuous.expert.success_range"]))
    num("s10.cover.finite", "137", N["finite.fb.min_state_action_count"][0])
    num("s10.cover.cont", "423", N["continuous.fb.min_state_action_count"][0])
    st = N["status"]
    bt = st["core"]["by_type"]
    nseed = len(man["core"]["seeds"])
    per = lambda c: c / nseed if c % nseed == 0 else None
    num("s10.core.jobs", "144", st["core"]["n_jobs"])
    num("s10.core.fpg", "27", bt["fpg:completed"])
    num("s10.core.fb", "9", bt["fb_pretrain:completed"])
    num("s10.core.adapt", "27", bt["adapt:completed"])
    # closeout (AUDIT.md L7): the finite FB-imitation runs are adaptations on a smaller number of pretrained models
    num("s10.core.fb.finite", "6", sum(1 for j in man["core"]["jobs"] if j["type"] == "fb_pretrain" and j["labels"]["env"] == F))
    num("s10.core.adapt.finite", "18", sum(1 for j in man["core"]["jobs"] if j["type"] == "adapt" and j["labels"]["env"] == F))
    num("s10.core.eval", "54", bt["eval:completed"], bound=False)
    num("s10.core.evalexp", "27", bt["eval_expert:completed"], bound=False)
    num("s10.perseed.fb", "3", per(bt["fb_pretrain:completed"]))
    num("s10.perseed.adapt", "9", per(bt["adapt:completed"]))
    num("s10.perseed.fpg", "9", per(bt["fpg:completed"]))
    num("s10.perseed.eval", "18", per(bt["eval:completed"]))
    num("s10.perseed.evalexp", "9", per(bt["eval_expert:completed"]))
    fact("s10.perseed.sum", "True", sum(bt[k] for k in ("fb_pretrain:completed", "adapt:completed", "fpg:completed",
                                                       "eval:completed", "eval_expert:completed")) == st["core"]["n_jobs"])
    n_new = st["compact_ablations"]["launches"][0]["counts_before"]["planned"]
    num("s10.abl.new", "21", n_new)
    abt = st["compact_ablations"]["by_type"]
    num("s10.abl.fpg_new", "6", abt["fpg:completed"] - 3)
    num("s10.abl.fb_new", "3", abt["fb_pretrain:completed"] - 3)
    num("s10.pilot.er_start", "0.179", P["finite.ER_FB/tau_1"]["js"])
    num("s10.pilot.dm_end", "0.384", P["finite.DM_FB/tau_1"]["js"])
    num("s10.pilot.l0", "-1456", N["pilot.fb"]["finite.tau_1"]["dm_l0"])
    num("s10.pilot.l1", "-1647", N["pilot.fb"]["finite.tau_1"]["dm_l1"])
    num_all("s10.pilot.lmin", "2\\times10^{-5}", [N["pilot.fb"][f"continuous.{v}"]["lmin"] for v in ("base", "tau_0p3", "tau_1")])
    num("s10.pilot.cont_er_dm_fail", "0", sum(1 for k, v in P.items() if k.startswith("continuous.") and ("ER_FB" in k or "DM_FB" in k)
                                              and (v["js"] <= 0.1 and v["success"] >= 0.8)), bound=False)
    rng("s10.floor", "0.0005", "0.0013", N["continuous.expert.js_range"])
    t0, t1 = N["wall.core.first_start"], N["wall.core.last_end"]
    a0, a1 = N["wall.compact_ablations.first_start"], N["wall.compact_ablations.last_end"]
    fact("time.core.date", "5 October 2026", _long_date(t0), bound=True)
    fact("time.core.date_iso", "2026-10-05", t0[:10], bound=True)
    fact("time.same_day", "True", len({t[:10] for t in (t0, t1, a0, a1)}) == 1)
    fact("time.core.start", "21:46:55", _utc(t0), bound=True)
    fact("time.core.end", "22:02:35", _utc(t1), bound=True)
    fact("time.abl.start", "22:20:00", _utc(a0), bound=True)
    fact("time.abl.end", "22:22:55", _utc(a1), bound=True)
    aud = _rd(ROOT / "results" / "audit_report.json")
    t_an = re.search(r"on (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) from", (ROOT / "results" / "RESULTS.md").read_text(encoding="utf-8")).group(1)
    fact("time.audit", "22:23", _utc(aud["utc"], 5) if _utc(aud["utc"], 5) == _utc(t_an, 5) else f"{aud['utc']} / {t_an}", bound=True)
    hist = {s: st[s]["attempts_hist"] for s in ("core", "compact_ablations")}
    fact("s10.first_attempt_only", "True", all(set(map(str, v)) == {"1"} for v in hist.values()))
    n_first = int(hist["core"].get("1", hist["core"].get(1, 0))) + n_new
    num("s10.jobs_first_attempt", "165", n_first)
    num("s10.peak_rss", "823", N["peak_rss_mib"])
    fact("s10.audit", "0/0", f"{aud['n_errors']}/{aud['n_warnings']}")
    envlap = _rd(RUNS / "environment_laptop.json")
    pk = envlap["packages"]
    fact("s10.python", "3.11.16", envlap["python"].split()[0], bound=True)
    fact("s10.torch", "2.14.1", str(pk.get("torch", "")).split("+")[0], bound=True)
    fact("s10.threads", "4", st["core"]["launches"][0]["torch_threads"], bound=True)
    fact("s10.device", "cpu/cpu", f"{st['core']['launches'][0]['device']}/{st['compact_ablations']['launches'][0]['device']}")

    # ------------------------------------------------------------------ s11 (finite results)
    V = N["finite.expert.V_H_star"]
    num("s11.expert_gap", "0", max(abs(x) for x in N["finite.expert.gap_range"]))
    num_all("s11.V.open.side", "15.51", [V["open/goal_5_1"], V["open/goal_1_5"]])
    num_all("s11.V.4r.side", "15.53", [V["four_rooms/goal_5_1"], V["four_rooms/goal_1_5"]])
    num("s11.V.open55", "12.43", V["open/goal_5_5"])
    num("s11.V.4r55", "12.36", V["four_rooms/goal_5_5"])
    gm = list(N["finite.expert.goal_mass"].values())
    rng("s11.goalmass", "0.61", "0.77", (min(gm), max(gm)))
    num_all("s11.expert_fig", "0.916", N["finite.expert.fig_range"])
    rng("s11.demo_target", "0.007", "0.022", N["finite.expert.demo_target_js_range"])
    rng("s11.fpg.js", "0.0011", "0.032", N[f"finite.{fpg}.js_range"])
    num("s11.fpg.succ_min", "0.998", N[f"finite.{fpg}.success_range"][0])
    rng("s11.fpg.gap", "0.15", "2.68", N[f"finite.{fpg}.gap_range"])
    rng("s11.fpg.fig", "0.853", "0.912", N[f"finite.{fpg}.fig_range"])
    rng("s11.fpg.ent", "0.03", "0.17", N[f"finite.{fpg}.entropy_range"])
    rng("s11.er.succ", "0.47", "1.00", N[f"finite.{er}.success_range"])
    num("s11.er.n08", "14", N[f"finite.{er}.n_success_ge_0.8"])
    rng("s11.er.fig", "0.001", "0.51", N[f"finite.{er}.fig_range"])
    word("s11.er.fig05", "a single", N[f"finite.{er}.n_fig_ge_0.5"])
    rng("s11.er.gap", "4.3", "14.1", N[f"finite.{er}.gap_range"])
    rng("s11.er.js", "0.079", "0.497", N[f"finite.{er}.js_range"])
    rng("s11.dm.js", "0.122", "0.645", N[f"finite.{dm}.js_range"])
    rng("s11.dm.succ", "0.0005", "0.975", N[f"finite.{dm}.success_range"])
    num("s11.dm.n08", "4", N[f"finite.{dm}.n_success_ge_0.8"])
    rng("s11.dm.gap", "6.7", "15.4", N[f"finite.{dm}.gap_range"])
    num("s11.pair.js", "16", N["finite.dm_vs_er.n_dm_js_higher"])
    num("s11.pair.succ", "17", N["finite.dm_vs_er.n_dm_success_lower"])
    num("s11.pair.n", "18", N["finite.dm_vs_er.n_pairs"])
    fact("s11.jssa_ge_js", "True", all(r["m"]["js_sa_nats"] >= r["m"]["js_nats"] - 1e-12 for r in final if r["env"] == F))
    rng("s11.fpg.jssa", "0.006", "0.111", N[f"finite.{fpg}.js_sa_range"])
    rng("s11.fpg.js0", "0.47", "0.52", N["finite.fpg.js_at_iter0_range"])
    num("s11.fpg.n20k", "17", sum(1 for x in fc if x == min(fc)))
    word("s11.fpg.n30k", "one", sum(1 for x in fc if x == 30000))
    num("s11.fpg.updates_at_20k", "10", min(fc) / (fpg_f["settings"]["n_traj"] * H))
    c30 = [k for k, v in conv.items() if k[0] == "core" and k[1] == F and k[4] == fpg and v["first"] == 30000]
    fact("s11.fpg.30k_run", "four_rooms/goal_5_5/0", "/".join(str(x) for x in (c30[0][2], c30[0][3], c30[0][5])) if len(c30) == 1 else c30)
    num("s11.fpg.last", "18", N[f"finite.{fpg}.conv_n_last"])
    num("s11.swing", "0.53", N["finite.fbil.max_js_swing_between_snapshots"])
    num("s11.lost", "12", N["finite.fbil.n_curves_success_ge_0.8_somewhere_not_last"])
    num("s11.curves", "36", N["finite.fbil.n_curves"])
    er_last = sorted("/".join(str(x) for x in (k[2], k[3], k[5])) for k, v in conv.items()
                     if k[0] == "core" and k[1] == F and k[4] == er and v["met_last"])
    fact("s11.er.last_runs", "['four_rooms/goal_1_5/0', 'open/goal_1_5/2']", er_last)
    er_early = [("/".join(str(x) for x in (k[2], k[3], k[5])), v["first"]) for k, v in conv.items()
                if k[0] == "core" and k[1] == F and k[4] == er and v["first"] is not None and not v["met_last"]]
    fact("s11.er.early", "[('open/goal_5_1/0', 2500)]", er_early)
    word("s11.er.early_n", "one", len(er_early))
    num("s11.er.early_first", "2,500", er_early[0][1] if len(er_early) == 1 else None)
    firsts = sorted(N[f"finite.{dm}.conv_first"])
    for i, t in enumerate(("2,500", "2,500", "5,000", "5,000", "7,500")):
        num(f"s11.dm.first{i}", t, firsts[i] if i < len(firsts) else None)
    num("s11.dm.ever", "5", N[f"finite.{dm}.conv_n_ever"])
    num("s11.notlast", "34", 36 - N[f"finite.{er}.conv_n_last"] - N[f"finite.{dm}.conv_n_last"])
    num("s11.earlyonly", "6", (N[f"finite.{er}.conv_n_ever"] - N[f"finite.{er}.conv_n_last"]) + (N[f"finite.{dm}.conv_n_ever"] - N[f"finite.{dm}.conv_n_last"]))
    num("s11.never", "28", (18 - N[f"finite.{er}.conv_n_ever"]) + (18 - N[f"finite.{dm}.conv_n_ever"]))
    # closeout (AUDIT.md M4): counting every evaluated checkpoint, including the intermediate DM_FB iterates
    num("s11.never_any", "25", N["finite.fbil.n_never_any_checkpoint"])
    word("s11.dm.mid_cross", "three", len(N["finite.fbil.dm_mid_cross"]))
    fact("s11.dm.mid_cross_runs", "['open/goal_1_5/1@50', 'open/goal_1_5/2@100', 'open/goal_5_1/0@50']",
         N["finite.fbil.dm_mid_cross"])
    num("s11.dmpath.loss", "27", N["dm_path.n_model_loss_decreased"])
    num("s11.dmpath.frac", "97", math.floor(100 * N["dm_path.frac_decrease_by_100"][0]))
    num("s11.dmpath.js_f", "16", N["dm_path.finite.n_js_increased"])
    num("s11.dmpath.js_c", "4", N["dm_path.continuous.n_js_increased"])
    num("s11.dmpath.succ_f", "17", N["dm_path.finite.n_success_decreased"])
    rng("s11.tv", "36.2", "79.7", N["finite.fb.tv_range"])
    rng("s11.q.er", "2.9", "8.9", N["finite.fb.er_q_range"])
    rng("s11.q.dm", "2.7", "15.2", N["finite.fb.dm_q_range"])
    rng("s11.lmin", "0.93", "0.96", N["finite.fb.lmin_range"])
    rng("s11.lmax", "2.26", "2.65", N["finite.fb.lmax_range"])
    rng("s11.mass", "18.2", "19.2", N["finite.fb.FmeanB_range"])
    num("s11.jobs", "165", n_first)
    lt("s11.sphere_tol", "10^{-5}", N["dm_path.max_z_norm_dev"])
    sd = lambda lab: [summary[k]["js_nats"]["sd"] for k in summary if k[0] == "core" and k[1] == F and k[4] == lab]
    num("s11.sd.fpg_max", "0.014", max(sd(fpg)))
    num("s11.sd.dm_max", "0.25", max(sd(dm)))
    fact("s11.sd.dm_n_ge_0.20", "3", sum(1 for x in sd(dm) if x >= 0.195))
    num("s11.sd.dm_third", "0.20", sorted(sd(dm))[-3])
    num("s11.sd.er_max", "0.21", max(sd(er)))
    for i, (j_t, s_t) in enumerate((("0.244", "0.95"), ("0.554", "0.031"), ("0.629", "0.0005"))):
        num(f"s11.dm55.js{i}", j_t, N["abl.DM_FB.js_values"][i])
        num(f"s11.dm55.succ{i}", s_t, N["abl.DM_FB.success_values"][i])
    fact("s11.dm55.same_as_core", "True", all(
        abs(summary[("core", F, "four_rooms", "goal_5_5", dm)]["js_nats"]["values"][s] - N["abl.DM_FB.js_values"][s]) < 1e-12
        for s in (0, 1, 2)))

    # ------------------------------------------------------------------ s12 (continuous results)
    rng("s12.floor", "0.0005", "0.0013", N["continuous.expert.js_range"])
    word("s12.orders", "two", int(math.floor(math.log10(min(N[f"continuous.{m}.js_range"][0] for m in (fpg, er, dm))
                                                        / N["continuous.expert.js_range"][1]))))
    rng("s12.fpg.js", "0.24", "0.41", N[f"continuous.{fpg}.js_range"])
    rng("s12.fpg.succ_pct", "92", "99", N[f"continuous.{fpg}.success_range"], 100)
    num("s12.fpg.n08", "9", N[f"continuous.{fpg}.n_success_ge_0.8"], bound=False)
    rng("s12.fpg.fig_pct", "6", "14", N[f"continuous.{fpg}.fig_range"], 100)
    rng("s12.exp.fig_pct", "72", "76", N["continuous.expert.fig_range"], 100)
    rng("s12.fpg.post_pct", "10", "18", N[f"continuous.{fpg}.post_hit_range"], 100)
    rng("s12.exp.post_pct", "77", "79", N["continuous.expert.post_hit_range"], 100)
    rng("s12.fpg.ent", "1.04", "1.47", N[f"continuous.{fpg}.entropy_range"])
    rng("s12.fpg.dec", "0.003", "0.039", N["continuous.fpg.last_interval_js_decrease"])
    num("s12.fpg.dec_mean", "0.025", N["continuous.fpg.last_interval_js_decrease_mean"])
    rng("s12.er.succ_pct", "12", "99", N[f"continuous.{er}.success_range"], 100)
    num("s12.er.n08", "4", N[f"continuous.{er}.n_success_ge_0.8"])
    num("s12.er.fig_max_pct", "0.4", N[f"continuous.{er}.fig_range"][1], 100)
    rng("s12.er.js", "0.48", "0.66", N[f"continuous.{er}.js_range"])
    rng("s12.dm.js", "0.35", "0.69", N[f"continuous.{dm}.js_range"])
    rng("s12.dm.succ", "0", "0.88", N[f"continuous.{dm}.success_range"])
    num("s12.dm.n08", "2", N[f"continuous.{dm}.n_success_ge_0.8"])
    num("s12.dm.max", "0.693", N[f"continuous.{dm}.js_range"][1])
    num("s12.dm.max_vs_log2_e4", "0", (math.log(2) - N[f"continuous.{dm}.js_range"][1]) * 1e4, bound=False)
    num("s12.pair.js", "4", N["continuous.dm_vs_er.n_dm_js_higher"])
    num("s12.pair.succ", "7", N["continuous.dm_vs_er.n_dm_success_lower"])
    num("s12.pair.n", "9", N["continuous.dm_vs_er.n_pairs"])
    num("s12.dmpath.js", "4", N["dm_path.continuous.n_js_increased"])
    rng("s12.fpg.js0", "0.57", "0.59", N["continuous.fpg.js_at_iter0_range"])
    # continuous f-PG learning curve (core): means over the runs at each checkpoint
    cfp = defaultdict(list)
    for r in rows:
        if r["env"] == Cn and r["method"] == "fpg" and "core" in r["suites"]:
            cfp[r["env_steps"]].append(r["m"])
    xs = sorted(cfp)
    msu = {x: sum(m["success"] for m in cfp[x]) / len(cfp[x]) for x in xs}
    mjs = {x: sum(m["js_nats"] for m in cfp[x]) / len(cfp[x]) for x in xs}
    fact("s12.fpg.curve_runs", "[9]", sorted({len(v) for v in cfp.values()}))
    num("s12.fpg.succ0", "0.29", msu.get(0))
    num("s12.fpg.x20k", "20,000", 20000 if 20000 in msu else None)
    num("s12.fpg.succ20k", "0.81", msu.get(20000))
    num("s12.fpg.succ_end", "0.97", msu.get(fpg_c["budget_transitions"]))
    fact("s12.fpg.js_mono", "True", all(mjs[a] > mjs[b] for a, b in zip(xs, xs[1:])))
    # continuous ER_FB and DM_FB (after its latent steps) at every FB snapshot: means over the runs
    snap = defaultdict(list)
    for r in rows:
        if r["env"] == Cn and "core" in r["suites"] and (r["method"] == "er" or (r["method"] == "dm" and r["final_dm_step"])):
            snap[(r["method"], r["fb_update"])].append(r["m"]["js_nats"])
    fact("s12.fbil.snap_runs", "[9]/8", f"{sorted({len(v) for v in snap.values()})}/{len(snap)}")
    sm = [sum(v) / len(v) for v in snap.values()]
    rng("s12.fbil.snapmean", "0.51", "0.56", (min(sm), max(sm)) if sm else None)
    num("s12.swing", "0.31", N["continuous.fbil.max_js_swing_between_snapshots"])
    num("s12.lost", "5", N["continuous.fbil.n_curves_success_ge_0.8_somewhere_not_last"])
    num("s12.curves", "18", N["continuous.fbil.n_curves"])
    num("s12.cover", "423", N["continuous.fb.min_state_action_count"][0])
    # closeout (AUDIT.md M5): coverage of the continuous replay buffers on the 356 cells of the evaluation histogram,
    # recomputed from the stored buffers (content hashes checked) by report/verify/replay_hist_coverage.py
    hc = _rd(REPORT / "verify" / "replay_hist_coverage.json")
    hb = {b["seed"]: b for b in hc["buffers"]}
    dman = _rd(ROOT / "data" / "final-v1" / "manifest.json")["items"]
    fact("s12.hcov.seeds", "[0, 1, 2]", sorted(hb))
    fact("s12.hcov.hash_checked", "True", all(b["content_hash_checked"] and b["content_sha256"] == dman[b["key"]]["content_sha256"]
                                             for b in hb.values()))
    fact("s12.hcov.cells", "356", "/".join(sorted({str(b["histogram"]["n_cells"]) for b in hb.values()})))
    fact("s12.hcov.lattice_as_recorded", "True",
         sorted(b["lattice"]["min_cell_action_count"] for b in hb.values())
         == sorted(x["coverage"]["min_cell_action_count"] for x in fbdiag["core"] if x["env"] == Cn and x["task"] == "goal_5_5"))
    fact("s12.hcov.low_threshold", "10", hc["low_threshold"], bound=True)
    for s in (0, 1, 2):
        num(f"s12.hcov.min.s{s}", ("0", "1", "2")[s], hb[s]["histogram"]["min_count"])
    lows = [b["histogram"]["n_pairs_below_low"] for b in hb.values()]
    rng("s12.hcov.low", "29", "34", (min(lows), max(lows)))
    num_all("s12.hcov.free_pct", "14", [b["histogram"]["free_area_fraction_of_low_bins"][e] for b in hb.values()
                                        for e in ("min", "max")], 100)
    rng("s12.lmin", "1.7\\times10^{-5}", "4.2\\times10^{-5}", N["continuous.fb.lmin_range"])
    rng("s12.lmax", "2.0", "2.2", N["continuous.fb.lmax_range"])
    num_all("s12.dev", "3.4", N["continuous.fb.dev_range"])

    # ------------------------------------------------------------------ s13 (ablations)
    A = lambda lab, met: N[f"abl.{lab}.mean.{met}"]
    rng("s13.kldq.js", "0.0085", "0.0169", (min(N["abl.f-PG[kl_dq].js_values"]), max(N["abl.f-PG[kl_dq].js_values"])))
    rng("s13.js.js", "0.0121", "0.0212", (min(N["abl.f-PG[js].js_values"]), max(N["abl.f-PG[js].js_values"])))
    num_all("s13.kldq.first30", "30,000", N["abl.f-PG[kl_dq].conv_first"])
    fact("s13.kldq.n", "3", len(N["abl.f-PG[kl_dq].conv_first"]))
    ja = N["abl.f-PG[js].conv_first"]
    num("s13.js.first.lo", "20,000", min(ja))
    num("s13.js.first.hi", "30,000", max(ja))
    fact("s13.all9", "9", N["abl.f-PG[js].conv_n_last"] + N["abl.f-PG[kl_dq].conv_n_last"] + N["abl.f-PG[kl_qd].conv_n_last"])
    num("s13.kldq.ret", "10.64", A("f-PG[kl_dq]", "return"))
    num("s13.js.ret", "10.92", A("f-PG[js]", "return"))
    rng("s13.klqd.js", "0.055", "0.071", (min(N["abl.f-PG[kl_qd].js_values"]), max(N["abl.f-PG[kl_qd].js_values"])))
    num("s13.klqd.ret", "7.99", A("f-PG[kl_qd]", "return"))
    num("s13.klqd.gap", "4.37", A("f-PG[kl_qd]", "gap"))
    num("s13.js.gap", "1.44", A("f-PG[js]", "gap"))
    num("s13.klqd.fig", "0.82", A("f-PG[kl_qd]", "final_in_goal"))
    num("s13.js.fig", "0.89", A("f-PG[js]", "final_in_goal"))
    num("s13.klqd.ent", "0.38", A("f-PG[kl_qd]", "policy_entropy_visited_nats"))
    num("s13.js.ent", "0.12", A("f-PG[js]", "policy_entropy_visited_nats"))
    num("s13.kldq.ent", "0.16", A("f-PG[kl_dq]", "policy_entropy_visited_nats"))
    num("s13.klqd.jssa", "0.146", A("f-PG[kl_qd]", "js_sa_nats"))
    num("s13.js.jssa", "0.048", A("f-PG[js]", "js_sa_nats"))
    word("s13.ratio", "three", round(A("f-PG[kl_qd]", "js_sa_nats") / A("f-PG[js]", "js_sa_nats")))
    rng("s13.noorth.lmin", "0.05", "0.19", N["abl.fb.no_orth.lmin_range"])
    rng("s13.noorth.lmax", "118", "146", N["abl.fb.no_orth.lmax_range"])
    rng("s13.base.lmin", "0.95", "0.96", N["abl.fb.base.lmin_range"])
    rng("s13.base.lmax", "2.3", "2.7", N["abl.fb.base.lmax_range"])
    rng("s13.base.erz", "2.6", "2.9", N["abl.fb.base.er_norm_range"])
    rng("s13.noorth.erz", "13.8", "17.2", N["abl.fb.no_orth.er_norm_range"])
    rng("s13.noorth.tv", "21.7", "44.8", N["abl.fb.no_orth.tv_range"])
    rng("s13.base.tv", "36.9", "53.9", N["abl.fb.base.tv_range"])
    rng("s13.noorth.q", "1.6", "6.2", N["abl.fb.no_orth.q_range"])
    rng("s13.base.q", "2.7", "12.3", N["abl.fb.base.q_range"])
    rng("s13.noorth.ent", "1.30", "1.44", N["abl.fb.no_orth.ent_range"])
    rng("s13.base.ent", "0.25", "0.72", N["abl.fb.base.ent_range"])
    fact("s13.none_met", "0", sum(N[f"abl.{l}.conv_n_last"] for l in ("ER_FB", "ER_FB/no_orth", "DM_FB", "DM_FB/no_orth")))
    rng("s13.noorth.er.succ", "0.95", "0.98", (min(N["abl.ER_FB/no_orth.success_values"]), max(N["abl.ER_FB/no_orth.success_values"])))
    rng("s13.base.er.succ", "0.65", "1.00", (min(N["abl.ER_FB.success_values"]), max(N["abl.ER_FB.success_values"])))
    num("s13.noorth.er.js", "0.358", sum(N["abl.ER_FB/no_orth.js_values"]) / 3)
    num("s13.base.er.js", "0.348", sum(N["abl.ER_FB.js_values"]) / 3)
    num("s13.noorth.er.fig", "0.13", A("ER_FB/no_orth", "final_in_goal"))
    num("s13.base.er.fig", "0.12", A("ER_FB", "final_in_goal"))
    num("s13.noorth.dm.js", "0.467", sum(N["abl.DM_FB/no_orth.js_values"]) / 3)
    num("s13.base.dm.js", "0.476", sum(N["abl.DM_FB.js_values"]) / 3)
    sdv = lambda v: (sum((x - sum(v) / 3) ** 2 for x in v) / 2) ** 0.5
    fact("s13.dm.spread_smaller", "True", sdv(N["abl.DM_FB/no_orth.js_values"]) < sdv(N["abl.DM_FB.js_values"]) / 2)

    # ------------------------------------------------------------------ s14 (costs)
    num("s14.demo_transitions", "400", cfg["data"]["n_demos"] * H)
    num("s14.fpg.wall.f", "0.38", N["cost.finite.f-PG[js].one_wall"][0])
    num("s14.fb.wall.f", "51.2", N["cost.finite.fb_wall"][0])
    num("s14.fpg.wall.c", "16.0", N["cost.continuous.f-PG[js].one_wall"][0])
    num("s14.fb.wall.c", "99.8", N["cost.continuous.fb_wall"][0])
    lt("s14.er.ms", "1", max(N["cost.finite.er_adapt_wall"][0], N["cost.continuous.er_adapt_wall"][0]), 1000)
    num_all("s14.dm.wall", "0.54", [N["cost.finite.dm_adapt_wall"][0], N["cost.continuous.dm_adapt_wall"][0]])
    num("s14.pct.lo", "40", 100 * min(fc) / fpg_f["budget_transitions"])
    num("s14.pct.hi", "60", 100 * max(fc) / fpg_f["budget_transitions"])
    num("s14.notlast", "34", 36 - N[f"finite.{er}.conv_n_last"] - N[f"finite.{dm}.conv_n_last"])
    num("s14.fpg.tr", "1,800,000", N["core.total_fpg_transitions"])
    num("s14.fb.tr", "600,000", N["core.total_fb_replay_transitions"])
    num("s14.eval.tr", "19,260,000", N["core.total_eval_transitions"])
    num("s14.abl.eval.tr", "1,560,000", N["abl.extra_eval_transitions"])
    W = N["wall.core.by_type"]
    num("s14.wall.total", "932.7", N["wall.core.total"])
    num("s14.wall.fb", "618.7", W["fb_pretrain"])
    num("s14.wall.fpg", "160.2", W["fpg"])
    num("s14.wall.eval", "90.2", W["eval"])
    num("s14.wall.adapt", "61.2", W["adapt"])
    num("s14.wall.evalexp", "2.3", W["eval_expert"])
    # closeout (AUDIT.md L2): whole-job wall time per adaptation job (status records), the convention of the totals
    # that the sentence reports (the earlier literal 2.2 was the mean of the phase timers, a different convention)
    num("s14.adapt_job", "2.3", N["wall.core.adapt_per_job"])
    # closeout (AUDIT.md M6): amortized FB per-task phase time relative to f-PG's (Table 14; means over the runs)
    ratio = lambda env: [N[f"cost.{env}.{lab}.am_wall"][0] / N[f"cost.{env}.{fpg}.one_wall"][0] for lab in (er, dm)]
    num("s14.ratio.f.lo", "45", min(ratio(F)))
    num("s14.ratio.f.hi", "46", max(ratio(F)))
    num_all("s14.ratio.c", "2.1", ratio(Cn))
    # closeout (AUDIT.md L1): checkpoints of the one evaluation job that ER_FB and DM_FB share per adaptation
    num_all("s14.eval.ckpt.er", "4", [c for env in (F, Cn) for c in N[f"cost.{env}.adapt_eval.checkpoints"]["er"]])
    num_all("s14.eval.ckpt.dm", "10", [c for env in (F, Cn) for c in N[f"cost.{env}.adapt_eval.checkpoints"]["dm"]])
    # closeout (AUDIT.md L4, L3): FB pretraining SD over the distinct models (printed in tab_cost_components, so not
    # tied to a \vn occurrence); the ablation cost rows of results/tables that amortize a reused core model over 1 goal
    num("s14.fb.wall_sd.f", "1.0", N["cost.finite.fb_wall"][1], bound=False)
    num("s14.fb.wall_sd.c", "0.5", N["cost.continuous.fb_wall"][1], bound=False)
    num("s14.abl.reused_rows", "6", len(N["cost.abl.rows_reused_model"]), bound=False)
    num("s14.abl.wall", "174.4", N["wall.compact_ablations.total"])
    num("s14.peak", "823", N["peak_rss_mib"])
    num("s14.amortized.f", "16,667", N["cost.finite.DM_FB.am_steps"][0])

    # ------------------------------------------------------------------ s15 (discussion)
    num("s15.tv.lo", "36", N["finite.fb.tv_range"][0])
    num("s15.tv.hi", "80", N["finite.fb.tv_range"][1])
    word("s15.third", "a third", Fraction(N["finite.fbil.n_curves_success_ge_0.8_somewhere_not_last"], N["finite.fbil.n_curves"]))

    # ------------------------------------------------------------------ appendix J (reproducibility)
    lc = st["core"]["launches"][0]
    la = st["compact_ablations"]["launches"][0]
    fact("aJ.commit", "db2e6d6", lc["git"]["commit"][:7] if lc["git"]["commit"] == la["git"]["commit"] else "differs", bound=True)
    fact("aJ.dirty", "True", lc["git"]["dirty"])
    fact("aJ.package", "d5fb4a98", lc["code_sha256"][:8] if lc["code_sha256"] == la["code_sha256"] else "differs", bound=True)
    plan = (ROOT / "RUN_PLAN.md").read_text(encoding="utf-8")
    fact("aJ.package_in_plan", "True", lc["code_sha256"] in plan)
    fact("aJ.cfg.core", "acabfe8d", man["core"]["config_sha256"][:8], bound=True)
    fact("aJ.cfg.abl", "45499164", man["compact_ablations"]["config_sha256"][:8], bound=True)
    fact("aJ.cfg_in_plan", "True", man["core"]["config_sha256"] in plan and man["compact_ablations"]["config_sha256"] in plan)
    fact("aJ.seedman", "619a8560", man["core"]["revisions"][0]["seed_manifest_sha256"][:8], bound=True)
    fact("aJ.refseed", "314159", man["core"]["reference_seed"], bound=True)
    num("aJ.abl.jobs", "39", st["compact_ablations"]["n_jobs"])
    num("aJ.data_items", "39", len(_rd(ROOT / "data" / "final-v1" / "manifest.json")["items"]))
    fact("aJ.platform", "Windows-10-10.0.26200", envlap["platform"].split("-SP")[0], bound=True)
    num("aJ.cpus", "12", envlap["logical_cpus"])
    num("aJ.ram", "15.35", envlap["ram_total_gib"])
    gpu = re.search(r"RTX (\d+)", envlap["devices"][0]["name"]) if envlap.get("devices") else None
    fact("aJ.gpu", "3060", gpu.group(1) if gpu else None, bound=True)
    fact("aJ.cuda_build", "13.2", envlap.get("torch_cuda_build"), bound=True)
    fact("aJ.numpy", "2.4.6", pk.get("numpy"), bound=True)
    fact("aJ.scipy", "1.17.1", pk.get("scipy"), bound=True)
    bench = re.findall(r"cpu (\d+) s, cuda (\d+) s", _rd(RUNS / "benchmark.json")["device_choice"]["reason"])[0]
    num("aJ.bench.cpu", "827", int(bench[0]))
    num("aJ.bench.cuda", "2163", int(bench[1]))
    num("aJ.audit.kinds", "62", len(aud["checks_run"]))
    num("aJ.audit.checks", "31,749", sum(aud["checks_run"].values()))
    pt = (ROOT / "audit" / "prompt4" / "pytest_laptop_frozen.txt").read_bytes().decode("utf-16")
    m = re.search(r"(\d+) passed, (\d+) skipped in ([0-9.]+)s", pt)
    num("aJ.pytest.passed", "366", int(m.group(1)))
    num("aJ.pytest.skipped", "1", int(m.group(2)))
    num("aJ.pytest.seconds", "155", float(m.group(3)))
    skips = re.findall(r"SKIPPED \[\d+\] (.*)", pt)
    fact("aJ.pytest.skip_posix", "True", len(skips) == 1 and "POSIX" in skips[0])
    return C


# ---------------------------------------------------------------------------------------------
# binding: every \vn{id}{literal} of the TeX sources against the registry
# ---------------------------------------------------------------------------------------------
_VN = "\\vn{"


def _tex_files():
    return sorted(p for d in TEX_DIRS for p in (REPORT / d).glob("*.tex"))


def _strip_comment(line: str) -> str:
    out, i = [], 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if line[i] == "%":
            break
        out.append(line[i])
        i += 1
    return "".join(out)


def _group(s: str, i: int):
    """Content of the brace group opening at s[i] == '{' and the index after it (None if unbalanced)."""
    if i >= len(s) or s[i] != "{":
        return None, i
    depth, j = 0, i
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return None, i


def _lines(path: Path):
    """(line number, text) of the typeset lines of a TeX file: comments removed, verbatim skipped."""
    verb = False
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if verb:
            verb = "\\end{verbatim}" not in raw
            continue
        if "\\begin{verbatim}" in raw:
            verb = "\\end{verbatim}" not in raw
            continue
        yield n, _strip_comment(raw)


def _occurrences():
    occ, bad = [], []
    for p in _tex_files():
        rel = p.relative_to(REPORT).as_posix()
        for n, line in _lines(p):
            i = line.find(_VN)
            while i >= 0:
                cid, j = _group(line, i + 3)
                lit, k = _group(line, j) if cid is not None else (None, j)
                if cid is None or lit is None:
                    bad.append((rel, n, line[i:i + 50]))
                    i = line.find(_VN, i + 4)
                    continue
                occ.append((rel, n, cid, lit))
                i = line.find(_VN, k)
    return occ, bad


def bind():
    """Checks of the \\vn occurrences against the registry (call after registry()); returns (checks, stats)."""
    occ, bad = _occurrences()
    checks = [(False, f"TeX {f}:{n}: malformed \\vn near {s!r}") for f, n, s in bad]
    seen = defaultdict(int)
    for f, n, cid, lit in occ:
        r = _REG.get(cid)
        if r is None:
            checks.append((False, f"TeX {f}:{n}: \\vn id {cid!r} is not a registered claim"))
            continue
        seen[cid] += 1
        checks.append((_norm(lit) == _norm(r["text"]),
                       f"TeX {f}:{n}: {cid} prints {lit!r} but its registered literal is {r['text']!r}"))
    for cid, r in _REG.items():
        if r["bound"] and not seen.get(cid):
            checks.append((False, f"claim {cid} ({r['text']!r}) is registered as printed but no \\vn{{{cid}}} occurs in the TeX"))
        if not r["bound"] and seen.get(cid):
            checks.append((False, f"claim {cid} is registered as not printed but occurs in the TeX through \\vn"))
    stats = {"occurrences": len(occ), "files": len({o[0] for o in occ}), "ids_used": len(seen),
             "registered": len(_REG), "bound": sum(1 for r in _REG.values() if r["bound"]),
             "unbound": sum(1 for r in _REG.values() if not r["bound"])}
    return checks, stats


# ---------------------------------------------------------------------------------------------
# coverage: numeric tokens of the TeX sources outside \vn, for review by hand
# ---------------------------------------------------------------------------------------------
_SKIP_CMD = re.compile(r"\\(?:label|cref|Cref|ref|eqref|pageref|audit|citep|citet|cite|includegraphics|input|claim|hyperref|"
                       r"nolinkurl|url|vspace|hspace|setlength|setcounter|addtocontents|begin|end)\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})*")
_SRC_REF = re.compile(r"(?:Eqs?\.|Theorems?|Thm|Lemmas?|Propositions?|Remarks?|Tables?|Algorithm|Sections?|App\.|Appendix|"
                      r"lines?|pp?\.|Def\.|Definition|Corollary|Figures?|Fig\.|Example|Assumption|\\S)~?\s*\(?[0-9A-Z][0-9A-Za-z.]*\)?"
                      r"(?:\s*(?:--|,|and)\s*\(?[0-9A-Z][0-9A-Za-z.]*\)?)*|\d+~pp\.|arXiv:\S+")
_TOKEN = re.compile(r"(?<![0-9.\\_^{\[:~#])-?\d[\d{},]*(?:\.\d+)?")
_MATH_ENVS = ("equation", "align", "gather", "multline", "displaymath")


def _strip_vn(line: str) -> str:
    out, i = [], 0
    while True:
        j = line.find(_VN, i)
        if j < 0:
            out.append(line[i:])
            return "".join(out)
        cid, k = _group(line, j + 3)
        lit, k2 = _group(line, k) if cid is not None else (None, k)
        if cid is None or lit is None:
            out.append(line[i:])
            return "".join(out)
        out.append(line[i:j] + " VN ")
        i = k2


def _math_mask(line: str, display: bool):
    """Per-character math flag of a line and the display state after it."""
    mask, inline, i = [], False, 0
    while i < len(line):
        two = line[i:i + 2]
        if two in ("\\[", "\\("):
            display = True if two == "\\[" else display
            inline = True if two == "\\(" else inline
            mask += [True, True]
            i += 2
            continue
        if two in ("\\]", "\\)"):
            display = False if two == "\\]" else display
            inline = False if two == "\\)" else inline
            mask += [True, True]
            i += 2
            continue
        if line[i] == "\\" and i + 1 < len(line):
            mask += [display or inline] * 2
            i += 2
            continue
        if line[i] == "$":
            inline = not inline
            mask.append(True)
            i += 1
            continue
        mask.append(display or inline)
        i += 1
    return mask, display


def coverage() -> str:
    reg_unbound = sorted(c for c, r in _REG.items() if not r["bound"])
    head = ["# Numeric tokens of the TeX sources (sections/, appendix/) outside \\vn, for review by hand.",
            "# Every token in text mode is listed; in math mode, only tokens with two or more digits, a decimal point",
            "# or a thousands separator.  Comments, verbatim blocks, labels, references, citations, figure files,",
            "# lengths and references to the sources (Eq., Theorem, p., ...) are skipped.  Generated tables are not",
            "# scanned: make_tables.py writes them from the recomputed values.",
            f"# Registered claims not tied to the text (bound=False): {len(reg_unbound)}: " + ", ".join(reg_unbound), ""]
    out, per_file = [], {}
    for p in _tex_files():
        rel = p.relative_to(REPORT).as_posix()
        display, n_tok = False, 0
        for n, line in _lines(p):
            for env_name in _MATH_ENVS:
                if re.search(r"\\begin\{" + env_name + r"\*?\}", line):
                    display = True
            s = _strip_vn(line)
            s = _SKIP_CMD.sub(lambda m: " " * len(m.group(0)), s)
            s = _SRC_REF.sub(lambda m: " " * len(m.group(0)), s)
            s = re.sub(r"\\[dt]?frac\d\d", lambda m: " " * len(m.group(0)), s)   # \frac12 and the like
            mask, display_after = _math_mask(s, display)
            for m in _TOKEN.finditer(s):
                j = m.start() - 1
                if j >= 0 and s[j].isalpha():
                    while j >= 0 and s[j].isalpha():
                        j -= 1
                    if j < 0 or s[j] != "\\":
                        continue   # part of a word or name (FB21, prompt3A); a number after a control word is kept
                tok = m.group(0).replace("{", "").replace("}", "").rstrip(",")
                if re.fullmatch(r"(19[5-9]\d|20[0-3]\d)", tok):
                    continue   # years
                is_math = mask[m.start()] if m.start() < len(mask) else False
                digits = sum(ch.isdigit() for ch in tok)
                if is_math and digits < 2 and "." not in tok and "," not in tok:
                    continue
                out.append(f"{rel}:{n}: {tok}{'  [math]' if is_math else ''}")
                n_tok += 1
            display = display_after
            for env_name in _MATH_ENVS:
                if re.search(r"\\end\{" + env_name + r"\*?\}", line):
                    display = False
        per_file[rel] = n_tok
    head.append("# tokens per file: " + ", ".join(f"{k} {v}" for k, v in per_file.items() if v))
    return "\n".join(head + [""] + out) + "\n"
