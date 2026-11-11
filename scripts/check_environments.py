"""Environment, oracle and expert check (Prompt 3A).  No learning algorithm is run.

Usage, from the project root:

    python scripts/check_environments.py
    python scripts/check_environments.py --out audit/prompt3A/laptop --n-rollouts 1000

Writes ``env_report.json``, ``env_report.md`` and three figures to ``--out``:

* finite mazes: transition-matrix checks, rank of P~, oracle values with
  Bellman-residual certificates, the H-step optimum and the expert's H-step
  return, and Monte Carlo returns of the finite expert;
* continuous maze: real noisy rollouts of the route controller with the
  validity criteria of ``il_memoir.expert.validate_route_controller``, the
  four seeded demonstrations per goal, and an empirical histogram-coverage
  diagnostic under a uniform random policy (U-13; not a proof);
* density utilities on the demonstrations: discount weights, histogram masses
  and the cut-and-normalize KDE.

Exit status 0 iff every validity check passes.  Only ASCII is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # the script is launched as `python scripts/...` from the root
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from il_memoir import oracle  # noqa: E402
from il_memoir.common import make_rng  # noqa: E402
from il_memoir.config import load_environment_config  # noqa: E402
from il_memoir.density import FreeSpace, FreeSpaceHistogram, WeightedKDE, next_state_samples  # noqa: E402
from il_memoir.env_continuous import make_continuous_maze  # noqa: E402
from il_memoir.env_tabular import initial_distribution, make_grid_maze, transition_matrix  # noqa: E402
from il_memoir.expert import (  # noqa: E402
    expert_occupancy,
    finite_expert,
    generate_demonstrations,
    route_controller,
    validate_route_controller,
)
from il_memoir.tasks import make_tasks  # noqa: E402

# reference palette (dataviz skill): categorical slots 1-3 (validated all-pairs, light), ink and chrome
GOAL_COLORS = {"goal_5_1": "#2a78d6", "goal_5_5": "#eb6834", "goal_1_5": "#1baf7a"}
SURFACE, INK, INK2, MUTED, GRID, WALL = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#52514e"


def _f(value, spec):
    """Format a number, or 'n/a' for a statistic that is undefined (e.g. no trajectory reached the goal)."""
    return "n/a" if value is None else format(value, spec)


def file_hashes(paths):
    """SHA-256 of each file with line endings normalized to LF (CRLF checkouts hash the same)."""
    out = {}
    for p in sorted(paths):
        data = p.read_bytes().replace(b"\r\n", b"\n")
        out[str(p.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(data).hexdigest()
    combined = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in out.items()).encode()).hexdigest()
    return out, combined


def environment_record():
    import matplotlib
    import scipy
    import yaml

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pyyaml": yaml.__version__,
    }


# ----------------------------------------------------------------------------- finite
def check_finite(cfg, tasks, seed, n_rollouts):
    report, ok = {}, True
    for layout in cfg.layout_names:
        maze = make_grid_maze(cfg, layout)
        P = transition_matrix(maze)
        rank = oracle.transition_rank(P)
        row_err = float(np.max(np.abs(P.sum(axis=2) - 1.0)))
        dist = maze.layout.bfs_distances(cfg.start, cfg.action_deltas)
        far = np.argwhere(dist == dist.max())
        lay = {
            "n_states": maze.n_states,
            "ascii": maze.layout.ascii({cfg.start: "S", **{c: "G" for _, c in cfg.goals}}),
            "bfs_max_distance_from_start": int(dist.max()),
            "bfs_farthest_cells": [tuple(int(v) for v in c) for c in far],
            "P_max_row_sum_error": row_err,
            "P_min_entry": float(P.min()),
            "rank_P_tilde": rank.rank,
            "smallest_singular_value": rank.smallest_singular_value,
            "exact_rank16_FB_possible": rank.rank <= 16,
            "goals": {},
        }
        ok &= row_err < 1e-12 and P.min() >= 0
        for name, task in tasks.items():
            ex = finite_expert(maze, task, cfg.gamma, cfg.horizon, cfg.value_iteration_tol, cfg.tie_tol)
            summ = ex.summary()
            q_E = expert_occupancy(maze, ex)
            demos = generate_demonstrations(
                maze, ex.pi, n_rollouts, cfg.horizon, make_rng(seed, "check", "finite", layout, name)
            )
            pos = maze.positions(demos.trajectories.states)
            returns = task.discounted_returns(pos, cfg.gamma)
            se = float(returns.std(ddof=1) / np.sqrt(n_rollouts))
            g = {
                **summ,
                "q_E_goal_mass": float(q_E[maze.index(task.goal_cell)]),
                "mc_return_mean": float(returns.mean()),
                "mc_return_se": se,
                "mc_minus_oracle_in_se": float((returns.mean() - summ["V_pi_H_start"]) / se) if se > 0 else 0.0,
                "mc_success_rate": float(task.success(pos).mean()),
                "mc_final_in_goal_rate": float(task.in_goal(pos[:, -1]).mean()),
                "expert_action_at_start": cfg.action_names[int(ex.actions[maze.start_state])],
            }
            checks = {
                "vi_converged": summ["vi_residual"] <= cfg.value_iteration_tol,
                "pi_certified": summ["pi_residual"] <= cfg.tie_tol,
                "h_step_gap_within_bound": -1e-10 <= summ["H_step_gap_start"] <= summ["H_step_gap_bound"] + 1e-12,
                "mc_return_consistent": abs(g["mc_minus_oracle_in_se"]) <= 5.0,
            }
            g["checks"] = checks
            ok &= all(checks.values())
            lay["goals"][name] = g
        report[layout] = lay
    return report, ok


# ------------------------------------------------------------------------- continuous
def check_continuous(cfg, tasks, seed, n_rollouts, n_demos):
    env = make_continuous_maze(cfg)
    report = {
        "layout": cfg.continuous_layout,
        "free_area": env.free_area,
        "step_length": env.step_length,
        "noise_std": env.noise_std,
        "acceptance_lower_bound": float((ndtr(env.cell_length / (2 * env.noise_std)) - 0.5) ** 2),
        "goals": {},
    }
    ok = True
    demos_out = {}
    for name, task in tasks.items():
        ctrl = route_controller(env.layout, task.goal_cell, cfg.action_deltas)
        stats = validate_route_controller(
            env, ctrl, task, n_rollouts, cfg.horizon, cfg.gamma, make_rng(seed, "check", "continuous", name)
        )
        demos = generate_demonstrations(
            env, ctrl, n_demos, cfg.horizon, make_rng(seed, "check", "demos", name), {"expert": "route_controller"}
        )
        pos = demos.trajectories.states
        stats["demos"] = {
            "n": n_demos,
            "hitting_times": task.hitting_times(pos).tolist(),
            "returns": [float(v) for v in task.discounted_returns(pos, cfg.gamma)],
            "all_free": bool(env.is_free(pos).all()),
        }
        ok &= stats["valid"] and stats["demos"]["all_free"]
        report["goals"][name] = stats
        demos_out[name] = demos
    return env, report, ok, demos_out


def coverage_diagnostic(env, hist, seed, n_traj, horizon):
    """Histogram cells visited by a uniform random policy from the start (U-13 diagnostic, not a proof)."""
    pol = lambda pos: np.full((pos.shape[0], env.n_actions), 1.0 / env.n_actions)  # noqa: E731
    traj = env.rollout(pol, n_traj, horizon, make_rng(seed, "check", "coverage"))
    ids = hist.cell_ids(traj.next_states.reshape(-1, 2))
    counts = np.bincount(ids, minlength=hist.n_cells)
    unvisited = np.nonzero(counts == 0)[0]
    return {
        "n_traj": n_traj,
        "n_cells": hist.n_cells,
        "cells_visited": int(np.sum(counts > 0)),
        "min_count": int(counts.min()),
        "unvisited_cells": [
            {"bin": hist.cells[k].tolist(), "free_area": float(hist.free_area[k])} for k in unvisited
        ],
        "min_free_area": float(hist.free_area.min()),
        "dropped_bins": int(hist.bins**2 - hist.n_cells),
    }


def density_summary(cfg, hist, space, demos):
    out = {}
    for name, demo in demos.items():
        pts, w = next_state_samples(demo.trajectories, cfg.gamma)
        kde = WeightedKDE(space, pts, w, cfg.kde_bandwidth)
        masses = hist.masses(pts, w)
        cm = kde.cell_masses(hist)
        out[name] = {
            "histogram_mass_sum": float(masses.sum()),
            "kde_cell_mass_sum": float(cm.sum()),
            "kde_min_kernel_normalizer": float(kde.norm.min()),
            "smoothed_hist_min": float(hist.smooth(masses, cfg.smoothing_eps).min()),
        }
    return out


# ---------------------------------------------------------------------------- figures
def _draw_walls(ax, walls, extent=1.0):
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((0, 0), extent, extent, facecolor=SURFACE, edgecolor=MUTED, lw=1.0, zorder=0))
    for x0, y0, x1, y1 in walls:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=WALL, edgecolor="none", zorder=1))
    ax.set_xlim(-0.01, extent + 0.01)
    ax.set_ylim(-0.01, extent + 0.01)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure_finite(cfg, tasks, seed, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.4), facecolor=SURFACE)
    L = 1.0 / cfg.grid_size
    for i, layout in enumerate(cfg.layout_names):
        maze = make_grid_maze(cfg, layout)
        walls = np.array([[x * L, y * L, (x + 1) * L, (y + 1) * L] for x, y in sorted(maze.layout.blocked)]).reshape(-1, 4)
        for j, (name, task) in enumerate(tasks.items()):
            ax = axes[i, j]
            _draw_walls(ax, walls)
            for k in range(1, cfg.grid_size):
                ax.plot([k * L, k * L], [0, 1], color=GRID, lw=0.6, zorder=0.5)
                ax.plot([0, 1], [k * L, k * L], color=GRID, lw=0.6, zorder=0.5)
            ex = finite_expert(maze, task, cfg.gamma, cfg.horizon, cfg.value_iteration_tol, cfg.tie_tol)
            for s, cell in enumerate(maze.cells):
                cx, cy = (np.asarray(cell) + 0.5) * L
                dx, dy = cfg.action_deltas[int(ex.actions[s])]
                if (dx, dy) == (0, 0):
                    ax.plot(cx, cy, "o", ms=3, color=INK2, zorder=2)
                else:
                    ax.annotate("", xy=(cx + 0.32 * dx * L, cy + 0.32 * dy * L), xytext=(cx - 0.18 * dx * L, cy - 0.18 * dy * L),
                                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, mutation_scale=7), zorder=2)
            demo = generate_demonstrations(maze, ex.pi, 1, cfg.horizon, make_rng(seed, "figure", layout, name))
            path = maze.positions(demo.trajectories.states[0])
            jitter = make_rng(seed, "figure", "jitter").normal(0, 0.006, path.shape)
            ax.plot(path[:, 0] + jitter[:, 0], path[:, 1] + jitter[:, 1], color=GOAL_COLORS[name], lw=2, alpha=0.85,
                    solid_capstyle="round", zorder=3)
            sx, sy = (np.asarray(cfg.start) + 0.5) * L
            gx, gy = task.goal_pos
            ax.plot(sx, sy, "o", ms=9, color=INK, mec=SURFACE, mew=2, zorder=4)
            ax.plot(gx, gy, "s", ms=10, color=GOAL_COLORS[name], mec=SURFACE, mew=2, zorder=4)
            ax.set_title(f"{layout}, {name}: V*_H(s0) = {ex.V_H_star[maze.start_state]:.3f}", fontsize=9, color=INK)
    fig.suptitle("Finite mazes: oracle expert actions (arrows; dot = stay) and one demonstration with slip 0.1\n"
                 "black dot = start (1,1); square = goal cell", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def figure_continuous(cfg, env, tasks, seed, demos, out_path, n_faint=100):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), facecolor=SURFACE)
    for ax, (name, task) in zip(axes, tasks.items()):
        _draw_walls(ax, env.walls)
        ctrl = route_controller(env.layout, task.goal_cell, cfg.action_deltas)
        faint = env.rollout(ctrl, n_faint, cfg.horizon, make_rng(seed, "figure", "faint", name)).states
        for tr in faint:
            ax.plot(tr[:, 0], tr[:, 1], color=GOAL_COLORS[name], lw=0.6, alpha=0.10, zorder=2)
        for tr in demos[name].trajectories.states:
            ax.plot(tr[:, 0], tr[:, 1], color=GOAL_COLORS[name], lw=2, alpha=0.95, solid_capstyle="round", zorder=3)
        ax.add_patch(Circle(task.goal_pos, task.radius, facecolor="none", edgecolor=INK, lw=1.2, zorder=5))
        ax.plot(*env.start_pos, "o", ms=9, color=INK, mec=SURFACE, mew=2, zorder=4)
        ax.set_title(f"{name}: 4 demonstrations (thick), {n_faint} rollouts (faint)", fontsize=9, color=INK)
    fig.suptitle("Continuous four-rooms maze: route controller with step 1/7 and N(0, 0.01^2) noise; "
                 "circle = goal region (radius 1/14)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def figure_density(cfg, env, space, hist, demos, name, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    blues = LinearSegmentedColormap.from_list(
        "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    )
    pts, w = next_state_samples(demos[name].trajectories, cfg.gamma)
    kde = WeightedKDE(space, pts, w, cfg.kde_bandwidth)
    res = 7 * 40
    c = (np.arange(res) + 0.5) / res
    xx, yy = np.meshgrid(c, c, indexing="ij")
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    dens = kde.density(grid).reshape(res, res)
    dens = np.where(space.contains(grid).reshape(res, res), dens, np.nan)
    masses = hist.masses(pts, w)
    img = np.full((hist.bins, hist.bins), np.nan)
    img[hist.cells[:, 0], hist.cells[:, 1]] = masses

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), facecolor=SURFACE)
    for ax, data, title, label in [
        (axes[0], dens, f"weighted KDE, h = {cfg.kde_bandwidth}", "density on S"),
        (axes[1], img, f"{hist.bins}x{hist.bins} free-space histogram", "cell mass"),
    ]:
        _draw_walls(ax, env.walls)
        im = ax.imshow(data.T, origin="lower", extent=(0, 1, 0, 1), cmap=blues, zorder=1.5, interpolation="nearest")
        for x0, y0, x1, y1 in env.walls:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=WALL, edgecolor="none", zorder=2))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(label, fontsize=8, color=INK2)
        cb.ax.tick_params(labelsize=7, colors=INK2)
        ax.set_title(title, fontsize=9, color=INK)
    fig.suptitle(f"Expert next-state occupancy estimate, {name}: 4 demonstrations, weights gamma^t / (4 Z_H)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------- main
def write_markdown(report, path):
    lines = ["# Environment check (Prompt 3A)", ""]
    meta = report["meta"]
    lines += [
        f"- Recorded (UTC): {meta['recorded_utc']}; wall time {meta['wall_time_s']:.1f} s; exit status {0 if report['all_checks_pass'] else 1}",
        f"- Interpreter: Python {meta['environment']['python']} on {meta['environment']['platform']}; "
        f"NumPy {meta['environment']['numpy']}, SciPy {meta['environment']['scipy']}, Matplotlib {meta['environment']['matplotlib']}",
        f"- Command: `{meta['command']}`",
        f"- Config sha256: `{meta['config_sha256']}`; il_memoir code sha256: `{meta['code_sha256']}`; seed {meta['seed']}",
        "",
        "## Finite mazes",
        "",
    ]
    for layout, lay in report["finite"].items():
        lines += [f"### {layout} ({lay['n_states']} free cells)", "", "```", lay["ascii"], "```", ""]
        lines += [
            f"- P: max |row sum - 1| = {lay['P_max_row_sum_error']:.1e}; rank P~ = {lay['rank_P_tilde']} "
            f"(smallest singular value {lay['smallest_singular_value']:.3f}); exact rank-16 FB possible: {lay['exact_rank16_FB_possible']}",
            f"- Largest BFS distance from the start (1,1): D = {lay['bfs_max_distance_from_start']}, "
            f"attained at {lay['bfs_farthest_cells']} (theory ENV-3 uses D + 1 <= H)",
            "",
            "| goal | V*(s0) VI | VI residual | eta_pi | V*_H(s0) | V^pi*_H(s0) | gap (bound) | q_E(goal) | MC return (SE) | MC success | first action |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, g in lay["goals"].items():
            lines.append(
                f"| {name} | {g['V_star_vi_start']:.6f} | {g['vi_residual']:.1e} | {g['pi_residual']:.1e} | "
                f"{g['V_H_star_start']:.6f} | {g['V_pi_H_start']:.6f} | {g['H_step_gap_start']:.1e} ({g['H_step_gap_bound']:.4f}) | "
                f"{g['q_E_goal_mass']:.4f} | {g['mc_return_mean']:.4f} ({g['mc_return_se']:.4f}) | {g['mc_success_rate']:.3f} | "
                f"{g['expert_action_at_start']} |"
            )
        lines.append("")
    cont = report["continuous"]
    lines += [
        "## Continuous maze (route controller, real noisy rollouts)",
        "",
        f"- Layout {cont['layout']}, free area {cont['free_area']:.6f}, step {cont['step_length']:.6f}, noise std {cont['noise_std']}; "
        f"acceptance lower bound (Phi(L/(2 sigma)) - 1/2)^2 = {cont['acceptance_lower_bound']:.6f}",
        "",
        "| goal | valid | success | hit time mean / max | post-hit in goal cell | post-hit in region | final in region | return mean (sd) | bad legs | blocked | max redraws |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in cont["goals"].items():
        lines.append(
            f"| {name} | {s['valid']} | {s['success_rate']:.3f} | {_f(s['hitting_time_mean'], '.2f')} / {s['hitting_time_max']} | "
            f"{_f(s['post_hit_in_goal_cell_frac'], '.3f')} | {_f(s['post_hit_in_goal_region_frac'], '.3f')} | {s['final_in_goal_rate']:.3f} | "
            f"{s['return_mean']:.3f} ({s['return_std']:.3f}) | {s['bad_legs_dense_check']} | {s['blocked_planned_moves']} | "
            f"{s['max_redraws']} |"
        )
    lines += [
        "",
        "The noise redraw cap (1001 draws) raises an error in `env.step`; these rollouts completed, so it was never reached.",
    ]
    cov = report["coverage"]
    lines += [
        "",
        f"Demonstrations ({report['meta']['n_demos']} per goal, seeded): hitting times "
        + "; ".join(f"{n}: {s['demos']['hitting_times']}" for n, s in cont["goals"].items()),
        "",
        "## Histogram coverage under a uniform random policy (U-13 diagnostic, not a proof)",
        "",
        f"- {cov['n_traj']} trajectories of H = {report['meta']['horizon']}: {cov['cells_visited']} of {cov['n_cells']} cells visited; "
        f"minimum count {cov['min_count']}; {cov['dropped_bins']} bins of zero free area dropped; smallest free area {cov['min_free_area']:.6f}",
        f"- Unvisited cells: {cov['unvisited_cells'] if cov['unvisited_cells'] else 'none'}",
        "",
        "## Density utilities on the demonstrations",
        "",
        "| goal | histogram mass sum | KDE cell-mass sum | min kernel normalizer | min smoothed mass |",
        "|---|---|---|---|---|",
    ]
    for name, d in report["density"].items():
        lines.append(
            f"| {name} | {d['histogram_mass_sum']:.12f} | {d['kde_cell_mass_sum']:.12f} | {d['kde_min_kernel_normalizer']:.4f} | {d['smoothed_hist_min']:.2e} |"
        )
    lines += ["", "Figures: " + ", ".join(report["figures"]) if report["figures"] else "Figures: not drawn", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(ROOT / "configs" / "environment.yaml"))
    parser.add_argument("--out", default=str(ROOT / "audit" / "prompt3A" / "env_check"))
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--n-rollouts", type=int, default=1000)
    parser.add_argument("--n-demos", type=int, default=4)
    parser.add_argument("--coverage-traj", type=int, default=2000)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_environment_config(args.config)
    tasks = make_tasks(cfg)
    code_files, code_hash = file_hashes(list((ROOT / "il_memoir").glob("*.py")))

    finite, ok_finite = check_finite(cfg, tasks, args.seed, args.n_rollouts)
    env, cont, ok_cont, demos = check_continuous(cfg, tasks, args.seed, args.n_rollouts, args.n_demos)
    space = FreeSpace.from_env(env)
    hist = FreeSpaceHistogram(space, cfg.histogram_bins)
    coverage = coverage_diagnostic(env, hist, args.seed, args.coverage_traj, cfg.horizon)
    density = density_summary(cfg, hist, space, demos)
    ok_density = all(abs(d["histogram_mass_sum"] - 1) < 1e-12 and abs(d["kde_cell_mass_sum"] - 1) < 1e-9 for d in density.values())

    figures = []
    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
        figure_finite(cfg, tasks, args.seed, out / "finite_expert_maps.png")
        figure_continuous(cfg, env, tasks, args.seed, demos, out / "continuous_expert_rollouts.png")
        figure_density(cfg, env, space, hist, demos, "goal_5_5", out / "expert_density_goal_5_5.png")
        figures = ["finite_expert_maps.png", "continuous_expert_rollouts.png", "expert_density_goal_5_5.png"]

    all_ok = bool(ok_finite and ok_cont and ok_density)
    report = {
        "meta": {
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": " ".join(["python", "scripts/check_environments.py"] + (argv if argv is not None else sys.argv[1:])),
            "environment": environment_record(),
            "config_path": str(Path(args.config)),
            "config_sha256": cfg.sha256(),
            "code_sha256": code_hash,
            "code_files": code_files,
            "seed": args.seed,
            "n_rollouts": args.n_rollouts,
            "n_demos": args.n_demos,
            "horizon": cfg.horizon,
            "gamma": cfg.gamma,
            "wall_time_s": time.time() - t0,
        },
        "finite": finite,
        "continuous": cont,
        "coverage": coverage,
        "density": density,
        "figures": figures,
        "all_checks_pass": all_ok,
    }
    (out / "env_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out / "env_report.md")

    print(f"config sha256 {cfg.sha256()[:16]}  code sha256 {code_hash[:16]}  seed {args.seed}")
    for layout, lay in finite.items():
        print(f"[finite {layout}] states={lay['n_states']} rank(P~)={lay['rank_P_tilde']} row_err={lay['P_max_row_sum_error']:.1e} "
              f"D={lay['bfs_max_distance_from_start']} at {lay['bfs_farthest_cells']}")
        for name, g in lay["goals"].items():
            print(f"   {name}: V*_H={g['V_H_star_start']:.6f} V^pi*_H={g['V_pi_H_start']:.6f} "
                  f"eta_VI={g['vi_residual']:.1e} eta_pi={g['pi_residual']:.1e} MC={g['mc_return_mean']:.4f}+-{g['mc_return_se']:.4f} "
                  f"checks={'ok' if all(g['checks'].values()) else 'FAIL'}")
    for name, s in cont["goals"].items():
        print(f"[continuous {name}] valid={s['valid']} success={s['success_rate']:.3f} hit_mean={_f(s['hitting_time_mean'], '.2f')} "
              f"in_cell_after_hit={_f(s['post_hit_in_goal_cell_frac'], '.3f')} bad_legs={s['bad_legs_dense_check']} reasons={s['reasons']}")
    print(f"[coverage] {coverage['cells_visited']}/{coverage['n_cells']} histogram cells visited by a uniform random policy")
    print(f"[density] mass sums ok={ok_density}")
    print(f"wrote {out / 'env_report.json'} and {out / 'env_report.md'}; figures: {figures}")
    print("ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
