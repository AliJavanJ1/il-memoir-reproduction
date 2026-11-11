#!/usr/bin/env python3
"""Figures of the report, drawn from the saved run files only.

Reads runs/suites/{core,compact_ablations}/manifest.json and the evaluation records
runs/jobs/<eval job>/outputs/evaluations.jsonl (the same records that scripts/analyze.py and
report/verify/verify_results.py read).  Writes vector PDFs next to this script.

Usage (project root):  python3 report/figures/make_figures.py
Colours: the categorical palette of il_memoir/analysis.py (f-PG blue, DM_FB orange, ER_FB aqua).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
BLUE, ORANGE, AQUA, VIOLET, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
JS_MAX, SUCC_MIN = 0.10, 0.80

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_rows(suite):
    man = read_json(RUNS / "suites" / suite / "manifest.json")
    out = []
    for j in man["jobs"]:
        if j["type"] not in ("eval", "eval_expert"):
            continue
        lab = j["labels"]
        p = RUNS / "jobs" / j["job_id"] / "outputs" / "evaluations.jsonl"
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            co = r.get("coords", {})
            out.append({"env": lab["env"], "layout": lab["layout"], "task": lab["task"], "seed": int(lab["seed"]),
                        "method": r["method"], "div": lab.get("divergence"), "fbv": lab.get("fb_variant", "base"),
                        "x_fpg": co.get("env_steps"), "fb": co.get("fb_update"), "k": co.get("dm_step"),
                        "final_fb": co.get("final_fb_snapshot"), "final_k": co.get("final_dm_step"),
                        "loss": co.get("dm_loss"), "js": r["metrics"]["js_nats"], "succ": r["metrics"]["success"],
                        "job": j["job_id"]})
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5)


def curves(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    return by


def plot_runs(ax, groups, xk, yk, color, ls="-", label=None):
    xs_all = defaultdict(list)
    for g, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r[xk])
        ax.plot([r[xk] for r in rs], [r[yk] for r in rs], color=color, linewidth=0.7, alpha=0.35, linestyle=ls)
        for r in rs:
            xs_all[r[xk]].append(r[yk])
    xs = sorted(xs_all)
    ax.plot(xs, [sum(xs_all[x]) / len(xs_all[x]) for x in xs], color=color, linewidth=2.0, linestyle=ls,
            marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.8, label=label)


def threshold(ax, y, text):
    ax.axhline(y, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=6.5, color=INK2)


def fig_fpg(core):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4))
    for col, env in enumerate(("finite", "continuous")):
        rs = [r for r in core if r["method"] == "fpg" and r["env"] == env]
        for row, (yk, lab, thr, ttext) in enumerate((("js", "held-out JS (nats)", JS_MAX, "threshold 0.10"),
                                                     ("succ", "success", SUCC_MIN, "threshold 0.80"))):
            ax = axes[row, col]
            style(ax)
            if env == "finite":
                for lay, ls in (("open", "-"), ("four_rooms", "--")):
                    g = curves([r for r in rs if r["layout"] == lay], lambda r: (r["task"], r["seed"]))
                    plot_runs(ax, g, "x_fpg", yk, BLUE, ls, label=f"{'open' if lay == 'open' else 'four rooms'} (mean of 9 runs)")
            else:
                g = curves(rs, lambda r: (r["task"], r["seed"]))
                plot_runs(ax, g, "x_fpg", yk, BLUE, "-", label="four rooms (mean of 9 runs)")
            threshold(ax, thr, ttext)
            ax.set_ylabel(lab)
            if row == 0:
                ax.set_title(f"$f$-PG, {env} maze{'s' if env == 'finite' else ''}")
                ax.set_ylim(bottom=0)
            else:
                ax.set_xlabel("learner transitions")
                ax.set_ylim(-0.03, 1.03)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{int(v/1000)}k" if v else "0"))
            ax.legend(frameon=False, loc="upper right" if row == 0 else "lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_fpg_curves.pdf")
    plt.close(fig)


def fig_fbil(core):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.9))
    for col, env in enumerate(("finite", "continuous")):
        rs = [r for r in core if r["env"] == env and (r["method"] == "er" or (r["method"] == "dm" and r["final_k"]))]
        for row, (yk, lab, thr, ttext) in enumerate((("js", "held-out JS (nats)", JS_MAX, "threshold 0.10"),
                                                     ("succ", "success", SUCC_MIN, "threshold 0.80"))):
            ax = axes[row, col]
            style(ax)
            for meth, color, name in (("dm", ORANGE, "DM$_{\\mathrm{FB}}$"), ("er", AQUA, "ER$_{\\mathrm{FB}}$")):
                g = curves([r for r in rs if r["method"] == meth], lambda r: (r["layout"], r["task"], r["seed"]))
                n = len(g)
                plot_runs(ax, g, "fb", yk, color, "-", label=f"{name} (mean of {n} runs)")
            threshold(ax, thr, ttext)
            ax.set_ylabel(lab)
            if row == 0:
                ax.set_title(f"FB imitation over pretraining, {env} maze{'s' if env == 'finite' else ''}")
                ax.set_ylim(0, 0.72)
            else:
                ax.set_xlabel("FB pretraining updates (task-blind)")
                ax.set_ylim(-0.03, 1.03)
            snaps = sorted({r["fb"] for r in rs})
            ax.set_xticks(snaps)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, p: (f"{v/1000:.1f}k".replace(".0k", "k")) if v else "0"))
            if row == 1:
                ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fbil_snapshots.pdf")
    plt.close(fig)


def fig_dm_path(core):
    rs = [r for r in core if r["method"] == "dm" and r["final_fb"]]
    g = curves(rs, lambda r: (r["env"], r["layout"], r["task"], r["seed"]))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for ax in axes:
        style(ax)
    for key, pts in sorted(g.items()):
        pts = sorted(pts, key=lambda r: r["k"])
        l0, l1 = pts[0]["loss"], pts[-1]["loss"]
        ls = "-" if key[0] == "finite" else "--"
        axes[0].plot([p["k"] for p in pts], [(p["loss"] - l1) / (l0 - l1) for p in pts], color=ORANGE, linewidth=0.8,
                     alpha=0.55, linestyle=ls)
        axes[1].plot([p["k"] for p in pts], [p["js"] - pts[0]["js"] for p in pts], color=ORANGE, linewidth=0.8,
                     alpha=0.55, linestyle=ls)
    axes[0].set_title("model objective $\\hat\\ell$ (normalized)")
    axes[0].set_ylabel("$(\\hat\\ell_k-\\hat\\ell_{300})/(\\hat\\ell_0-\\hat\\ell_{300})$")
    axes[1].set_title("held-out JS minus its value at step 0 ($z_{\\mathrm{ER}}$)")
    axes[1].set_ylabel("change in JS (nats)")
    axes[1].axhline(0, color=INK2, linewidth=0.9)
    for ax in axes:
        ax.set_xlabel("DM$_{\\mathrm{FB}}$ latent steps (final FB snapshot)")
    axes[0].plot([], [], color=ORANGE, linestyle="-", label="finite (18 runs)")
    axes[0].plot([], [], color=ORANGE, linestyle="--", label="continuous (9 runs)")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dm_path.pdf")
    plt.close(fig)


def fig_ablation(abl):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    for ax in axes:
        style(ax)
    ax = axes[0]
    for div, color, ls, mk, name in (("js", BLUE, "-", "o", "JS"), ("kl_dq", VIOLET, "--", "s", "KL($d\\Vert q$)"),
                                     ("kl_qd", YELLOW, ":", "^", "KL($q\\Vert d$)")):
        rs = [r for r in abl if r["method"] == "fpg" and r["div"] == div]
        g = curves(rs, lambda r: r["seed"])
        xs_all = defaultdict(list)
        for sd, pts in g.items():
            pts = sorted(pts, key=lambda r: r["x_fpg"])
            ax.plot([p["x_fpg"] for p in pts], [p["js"] for p in pts], color=color, linewidth=0.7, alpha=0.35, linestyle=ls)
            for p in pts:
                xs_all[p["x_fpg"]].append(p["js"])
        xs = sorted(xs_all)
        ax.plot(xs, [sum(xs_all[x]) / len(xs_all[x]) for x in xs], color=color, linewidth=1.8, linestyle=ls, marker=mk,
                markersize=3.5, markeredgecolor="white", markeredgewidth=0.6, label=f"$f$-PG, {name}")
    threshold(ax, JS_MAX, "threshold 0.10")
    ax.set_title("$f$-PG: choice of $f$ (3 seeds)")
    ax.set_xlabel("learner transitions")
    ax.set_ylabel("held-out JS (nats)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{int(v/1000)}k" if v else "0"))
    ax.legend(frameon=False, loc="upper right")
    fin = [r for r in abl if r["method"] in ("er", "dm") and r["final_fb"] and (r["method"] == "er" or r["final_k"])]
    cats = [("er", "base", "ER\nbase"), ("er", "no_orth", "ER\nno orth."), ("dm", "base", "DM\nbase"), ("dm", "no_orth", "DM\nno orth.")]
    for ax, yk, lab, thr in ((axes[1], "js", "held-out JS (nats)", JS_MAX), (axes[2], "succ", "success", SUCC_MIN)):
        for i, (m, v, name) in enumerate(cats):
            vals = [r[yk] for r in fin if r["method"] == m and r["fbv"] == v]
            color = AQUA if m == "er" else ORANGE
            face = color if v == "base" else "white"
            offs = [(-0.12 + 0.12 * j) for j in range(len(vals))] if len(vals) == 3 else [0.0] * len(vals)
            ax.scatter([i + o for o in offs], vals, s=22, facecolor=face, edgecolor=color, linewidth=1.2, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [sum(vals) / len(vals)] * 2, color=color, linewidth=2.0, zorder=4)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([c[2] for c in cats], fontsize=6.5)
        ax.axhline(thr, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.set_ylabel(lab)
        ax.set_xlim(-0.6, len(cats) - 0.4)
    axes[1].set_title("FB orthonormality: JS")
    axes[2].set_title("FB orthonormality: success")
    axes[1].set_ylim(0, 0.72)
    axes[2].set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation.pdf")
    plt.close(fig)


def final_rows(rows):
    """Last checkpoint of every run: f-PG at its last iteration, ER_FB at the final FB snapshot,
    DM_FB at the final FB snapshot and final latent step, and the expert rows."""
    out = {}
    for r in rows:
        key = (r["env"], r["layout"], r["task"], r["seed"], r["method"])
        if r["method"] == "fpg":
            if key not in out or r["x_fpg"] > out[key]["x_fpg"]:
                out[key] = r
        elif r["method"] == "er":
            if r["final_fb"]:
                out[key] = r
        elif r["method"] == "dm":
            if r["final_fb"] and r["final_k"]:
                out[key] = r
        elif r["method"] == "expert":
            out[key] = r
    return out


def fig_final_seeds(core):
    fin = final_rows(core)
    cells = [("finite", "open", g) for g in ("goal_5_1", "goal_5_5", "goal_1_5")] + \
            [("finite", "four_rooms", g) for g in ("goal_5_1", "goal_5_5", "goal_1_5")] + \
            [("continuous", "four_rooms", g) for g in ("goal_5_1", "goal_5_5", "goal_1_5")]
    meths = [("fpg", BLUE, "$f$-PG (JS)"), ("dm", ORANGE, "DM$_{\\mathrm{FB}}$"), ("er", AQUA, "ER$_{\\mathrm{FB}}$"),
             ("expert", INK2, "expert")]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.6), sharex=True)
    for ax, yk, lab, thr in ((axes[0], "js", "held-out JS (nats)", JS_MAX), (axes[1], "succ", "success", SUCC_MIN)):
        style(ax)
        ax.grid(False, axis="x")
        for ci, (env, lay, g) in enumerate(cells):
            for mi, (m, color, name) in enumerate(meths):
                x0 = ci + (mi - 1.5) * 0.2
                vals = [fin[(env, lay, g, sd, m)][yk] for sd in (0, 1, 2) if (env, lay, g, sd, m) in fin]
                if not vals:
                    continue
                ax.scatter([x0 + (j - 1) * 0.05 for j in range(len(vals))], vals, s=10, color=color, alpha=0.85,
                           linewidths=0, zorder=3, label=name if ci == 0 and yk == "js" else None)
                ax.plot([x0 - 0.08, x0 + 0.08], [sum(vals) / len(vals)] * 2, color=color, linewidth=1.8, zorder=4)
        ax.axhline(thr, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
        for b in (2.5, 5.5):
            ax.axvline(b, color=AXIS, linewidth=0.8)
        ax.set_ylabel(lab)
    axes[0].set_ylim(-0.02, 0.72)
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].set_xticks(range(len(cells)))
    axes[1].set_xticklabels(["(5,1)", "(5,5)", "(1,5)"] * 3)
    for x, t in ((1, "finite open"), (4, "finite four rooms"), (7, "continuous four rooms")):
        axes[0].text(x, 0.735, t, ha="center", va="bottom", fontsize=7.5, color=INK)
    axes[1].set_xlim(-0.6, len(cells) - 0.4)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_final_seeds.pdf")
    plt.close(fig)


def main():
    core = load_rows("core")
    abl = load_rows("compact_ablations")
    fig_fpg(core)
    fig_fbil(core)
    fig_dm_path(core)
    fig_ablation(abl)
    fig_final_seeds(core)
    print("written:", ", ".join(p.name for p in sorted(OUT.glob("fig_*.pdf"))))


if __name__ == "__main__":
    main()
