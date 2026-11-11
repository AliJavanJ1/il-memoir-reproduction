"""Held-out evaluation (METHOD_MAP section 7, Prompt 3C): exact finite metrics, Monte Carlo agreement,
the common continuous estimator, and evaluation that reads checkpoints without changing them.

The exact quantities are checked against independent computations (explicit path
enumeration, the oracle's own identities, many simulator rollouts), not against
the code under test.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from il_memoir import evaluation as ev
from il_memoir import oracle
from il_memoir.checks import output_hashes_intact
from il_memoir.common import make_rng
from il_memoir.env_continuous import make_continuous_maze
from il_memoir.env_tabular import make_layout
from il_memoir.expert import route_controller
from il_memoir.logging import JobContext, JsonlLog, StopController, read_json, sha256_file
from il_memoir.tasks import make_task

from _helpers import random_mdp, random_policy
from _learner_helpers import enumerate_trajectories

SEED = 20261005


def test_hit_probability_equals_explicit_path_enumeration():
    rng = make_rng(SEED, "eval", "hit")
    P = random_mdp(rng, 4, 2)
    pi = random_policy(rng, 4, 2)
    mu0 = np.array([0.5, 0.5, 0.0, 0.0])
    goal = np.array([False, False, True, False])
    H = 5
    states, _, probs = enumerate_trajectories(P, pi, mu0, H)
    hit_t = np.where(goal[states[:, 1:]].any(axis=1), np.argmax(goal[states[:, 1:]], axis=1) + 1, 0)
    exact = float(probs[hit_t > 0].sum())
    got, first = ev.hit_probability(P, pi, mu0, goal, H)
    assert abs(got - exact) < 1e-12
    for t in range(1, H + 1):
        assert abs(first[t] - probs[hit_t == t].sum()) < 1e-12
    # s_0 in the goal does not count as a hit
    got0, _ = ev.hit_probability(P, pi, np.array([0.0, 0.0, 1.0, 0.0]), goal, 1)
    assert abs(got0 - float(oracle.policy_kernel(P, pi)[2, 2])) < 1e-12


def test_finite_oracle_metrics_are_consistent(cfg):
    orc = ev.FiniteOracle(cfg, "four_rooms", "goal_5_5")
    m = orc.exact_metrics(orc.expert.pi)
    assert m["js_nats"] == pytest.approx(0.0, abs=1e-12) and m["js_sa_nats"] == pytest.approx(0.0, abs=1e-12)
    assert m["return"] == pytest.approx(orc.expert.V_pi_H[orc.s0], abs=1e-12)
    assert m["return_from_occupancy"] == pytest.approx(m["return"], abs=1e-9)          # OCC-4
    bound = orc.expert.summary()["H_step_gap_bound"]
    assert -1e-10 <= m["gap"] <= bound + 1e-12                                          # OCC-6
    uniform = np.full((orc.maze.n_states, orc.maze.n_actions), 1.0 / orc.maze.n_actions)
    u = orc.exact_metrics(uniform)
    assert 0.0 < u["js_nats"] <= np.log(2) and u["gap"] > 0 and 0.0 < u["success"] < m["success"]
    assert u["return_from_occupancy"] == pytest.approx(u["return"], abs=1e-9)
    # JS(p, q) <= (KL(p || q) + KL(q || p)) / 4 (convexity of KL in its second argument): an independent bound
    assert u["kl_dq_nats"] is not None and u["kl_qd_nats"] is not None
    assert u["js_nats"] <= (u["kl_dq_nats"] + u["kl_qd_nats"]) / 4 + 1e-12


def test_monte_carlo_metrics_agree_with_exact_finite_values(cfg):
    orc = ev.FiniteOracle(cfg, "four_rooms", "goal_5_1")
    rng = make_rng(SEED, "eval", "pi")
    pi = 0.6 * orc.expert.pi + 0.4 * random_policy(rng, orc.maze.n_states, orc.maze.n_actions)
    exact = orc.exact_metrics(pi)
    n = 4000
    mc = ev.mc_metrics(orc.maze, lambda s: pi[np.asarray(s)], orc.task, cfg.gamma, cfg.horizon, n,
                       make_rng(SEED, "eval", "mc"), q_E=orc.q_E)
    assert abs(mc["mc_return_mean"] - exact["return"]) <= 5 * mc["mc_return_se"]
    se_s = np.sqrt(exact["success"] * (1 - exact["success"]) / n) + 1e-3
    assert abs(mc["mc_success"] - exact["success"]) <= 5 * se_s
    se_f = np.sqrt(exact["final_in_goal"] * (1 - exact["final_in_goal"]) / n) + 1e-3
    assert abs(mc["mc_final_in_goal"] - exact["final_in_goal"]) <= 5 * se_f
    assert 0.0 <= mc["mc_js_counts_nats"] <= np.log(2)


def test_continuous_estimator_is_zero_on_identical_samples_and_orders_policies(cfg):
    env = make_continuous_maze(cfg)
    task = make_task(cfg, "goal_5_5")
    ctrl = route_controller(make_layout(cfg, cfg.continuous_layout), cfg.goal_cell("goal_5_5"), cfg.action_deltas)
    ref = env.rollout(ctrl, 400, cfg.horizon, make_rng(SEED, "eval", "ref")).states
    reference = ev.ContinuousReference(cfg, cfg.continuous_layout, ref)
    assert abs(reference.masses.sum() - 1.0) < 1e-12
    from il_memoir import divergences as dv

    assert dv.js(reference.masses, reference.masses) == 0.0
    expert = ev.mc_metrics(env, ctrl, task, cfg.gamma, cfg.horizon, 400, make_rng(SEED, "eval", "x"),
                           hist=reference.hist, ref_masses=reference.masses)
    uniform = ev.mc_metrics(env, lambda s: np.full((len(s), 5), 0.2), task, cfg.gamma, cfg.horizon, 400,
                            make_rng(SEED, "eval", "x"), hist=reference.hist, ref_masses=reference.masses)
    assert 0.0 <= expert["js_nats"] < 0.05 < uniform["js_nats"] <= np.log(2)
    assert expert["success"] == 1.0 and uniform["success"] < 0.5


def test_evaluation_reads_checkpoints_without_changing_them(smoke_run, tmp_path):
    """Re-running an evaluation job leaves every training output untouched and reproduces its rows exactly."""
    runs, man = smoke_run["runs"], smoke_run["manifest"]
    before = {p: sha256_file(p) for p in (runs / "jobs").rglob("*.pt")}
    checked = 0
    for j in man["jobs"]:
        if j["type"] != "eval":
            continue
        src = runs / "jobs" / j["job_id"]
        copy = tmp_path / j["job_id"]
        copy.mkdir()
        target = j["identity"]["target"]
        tdir = runs / "jobs" / target
        job = type("J", (), {})()
        job.job_id, job.identity, job.data_inputs = j["job_id"], j["identity"], j["data_inputs"]
        from il_memoir import datasets as ds

        entries = {role: ds.manifest_entry(smoke_run["data"], man["data"]["id"], key) for role, key in j["data_inputs"].items()}
        ctx = JobContext(job=job, job_dir=copy, data_root=smoke_run["data"], data_id=man["data"]["id"],
                         env_cfg=smoke_run["suite"].env_cfg, device="cpu", runner=smoke_run["suite"].runner,
                         stop=StopController(), data_entries=entries,
                         parents={target: {"dir": tdir, "result": read_json(tdir / "result.json")}},
                         job_dir_of=lambda jid: runs / "jobs" / jid, log=lambda m: None)
        ev.run_eval(ctx, "test")
        new = JsonlLog(copy / "outputs" / "evaluations.jsonl").records()
        old = JsonlLog(src / "outputs" / "evaluations.jsonl").records()
        strip = lambda rows: [{k: v for k, v in r.items() if k != "seconds"} for r in rows]       # noqa: E731
        assert strip(new) == strip(old)
        checked += 1
    assert checked == 4
    assert before == {p: sha256_file(p) for p in (runs / "jobs").rglob("*.pt")}
    assert all(output_hashes_intact(runs, man).values())


def test_every_checkpoint_is_evaluated_with_the_evaluation_stream(smoke_run):
    runs, man = smoke_run["runs"], smoke_run["manifest"]
    for j in man["jobs"]:
        if j["type"] not in ("eval", "eval_expert"):
            continue
        rows = JsonlLog(runs / "jobs" / j["job_id"] / "outputs" / "evaluations.jsonl").records()
        assert all(r["rng"]["stream"][0] == "evaluation" for r in rows)
        if j["type"] == "eval" and j["identity"]["target_type"] == "fpg":
            res = read_json(runs / "jobs" / j["identity"]["target"] / "result.json")
            assert [r["coords"]["iteration"] for r in rows] == [o["iteration"] for o in res["outputs"]["policies"]]
            assert all(r["source"]["sha256"] == o["sha256"] for r, o in zip(rows, res["outputs"]["policies"]))
        if j["type"] == "eval" and j["identity"]["target_type"] == "adapt":
            methods = {r["method"] for r in rows}
            assert methods == {"er", "dm"}
            final = [r for r in rows if r["coords"]["final_fb_snapshot"]]
            assert sorted(r["coords"]["dm_step"] for r in final if r["method"] == "dm") == [0, 10, 20]
            if j["labels"]["env"] == "finite":
                assert any("fb_model_error" in r["metrics"] for r in final)
