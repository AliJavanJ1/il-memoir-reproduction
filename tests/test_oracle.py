"""Exact finite oracle: normalization, Bellman identities, rollouts, timing, optimality, rank, Jacobian, FB error."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from il_memoir import oracle
from il_memoir.common import make_rng
from il_memoir.env_tabular import initial_distribution, make_grid_maze, transition_matrix

from _helpers import (
    assert_within_se,
    exact_greedy_fb,
    per_trajectory_occupancy,
    random_mdp,
    random_policy,
    softmax_policy_jacobian,
    softmax_rows,
)

GAMMA, H = 0.95, 100


@pytest.fixture(scope="module")
def maze_setup(four_rooms, four_rooms_P):
    rng = make_rng(20261005, "oracle", "policy")
    pi = random_policy(rng, four_rooms.n_states, 5, alpha=2.0)
    return four_rooms, four_rooms_P, pi, initial_distribution(four_rooms)


def test_policy_kernel_and_successor_identities(maze_setup):
    maze, P, pi, mu0 = maze_setup
    P_pi = oracle.policy_kernel(P, pi)
    assert np.allclose(P_pi.sum(axis=1), 1.0, atol=1e-14)
    M = oracle.successor_next_state(P, pi, GAMMA)
    assert np.allclose(M.sum(axis=2), 1.0 / (1.0 - GAMMA), atol=1e-10)          # OCC-1(a)
    M_avg = np.einsum("sa,sat->st", pi, M)
    bellman = P + GAMMA * np.einsum("sau,ut->sat", P, M_avg)                      # OCC-3(b), FB-1
    assert np.allclose(M, bellman, atol=1e-10)
    # independent fixed-point iteration X <- P~ + gamma X P_pi (no linear solve)
    X = np.zeros_like(M)
    for _ in range(1200):
        X = P + GAMMA * np.einsum("sau,ut->sat", X, P_pi)
    assert np.allclose(M, X, atol=1e-9)
    assert np.allclose(oracle.successor_next_state_avg(P, pi, GAMMA), P_pi @ np.linalg.inv(np.eye(40) - GAMMA * P_pi))


def test_occupancy_normalization_methods_and_tail(maze_setup):
    maze, P, pi, mu0 = maze_setup
    d_inf = oracle.occupancy_next_state(P, pi, mu0, GAMMA)
    d_sum = oracle.occupancy_next_state(P, pi, mu0, GAMMA, H, method="sum")
    d_res = oracle.occupancy_next_state(P, pi, mu0, GAMMA, H, method="resolvent")
    for d in (d_inf, d_sum, d_res):
        assert np.all(d >= -1e-15) and np.isclose(d.sum(), 1.0, atol=1e-12)
    assert np.allclose(d_sum, d_res, atol=1e-12)                                   # Prop. 1.8(d)
    tv = 0.5 * np.abs(d_inf - d_sum).sum()
    assert tv <= GAMMA**H + 1e-12                                                 # OCC-5(b)
    law_H = oracle.state_marginals(P, pi, mu0, H)[H]
    tail = oracle.occupancy_next_state(P, pi, law_H, GAMMA)                       # (1-g) sum_u g^u Law_{H+1+u}
    assert np.allclose(d_inf, (1 - GAMMA**H) * d_sum + GAMMA**H * tail, atol=1e-12)  # OCC-5(a)


def test_time_shift_between_pre_action_and_next_state(maze_setup):
    maze, P, pi, mu0 = maze_setup
    d_plus = oracle.occupancy_next_state(P, pi, mu0, GAMMA)
    d_sa = oracle.occupancy_state_action(P, pi, mu0, GAMMA)
    assert np.isclose(d_sa.sum(), 1.0)
    assert np.allclose(d_sa.sum(axis=1), (1 - GAMMA) * mu0 + GAMMA * d_plus, atol=1e-12)   # OCC-2(b)
    z_h = oracle.discount_normalizer(GAMMA, H)
    d_sa_h = oracle.occupancy_state_action(P, pi, mu0, GAMMA, H)
    d_plus_h = oracle.occupancy_next_state(P, pi, mu0, GAMMA, H)
    law_H = oracle.state_marginals(P, pi, mu0, H)[H]
    assert np.allclose(z_h * d_sa_h.sum(axis=1), mu0 + GAMMA * z_h * d_plus_h - GAMMA**H * law_H, atol=1e-12)  # OCC-2(c)
    assert np.allclose(d_sa_h / d_sa_h.sum(axis=1, keepdims=True), pi, atol=1e-12)


def test_occupancies_match_independent_rollouts(maze_setup):
    """Matrix sums against 20 000 simulator trajectories (5 standard errors per state)."""
    maze, P, pi, mu0 = maze_setup
    rng = make_rng(20261005, "oracle", "rollouts")
    n = 20_000
    traj = maze.rollout(pi, n, H, rng)
    Y = per_trajectory_occupancy(traj.next_states, maze.n_states, GAMMA)
    d = oracle.occupancy_next_state(P, pi, mu0, GAMMA, H)
    assert_within_se(Y.mean(axis=0), d, Y.std(axis=0, ddof=1) / np.sqrt(n))
    # state-action occupancy (pre-action timing, weights gamma^t / Z_H on (s_t, a_t))
    g = GAMMA ** np.arange(H) / oracle.discount_normalizer(GAMMA, H)
    Ysa = np.zeros((n, maze.n_states * 5))
    np.add.at(Ysa, (np.repeat(np.arange(n), H), (traj.states[:, :-1] * 5 + traj.actions).ravel()), np.tile(g, n))
    d_sa = oracle.occupancy_state_action(P, pi, mu0, GAMMA, H).ravel()
    assert_within_se(Ysa.mean(axis=0), d_sa, Ysa.std(axis=0, ddof=1) / np.sqrt(n))


def test_h_step_oracle_uses_rollout_reward_timing(maze_setup, tasks):
    """V^pi_H(s0) = Z_H <d+_{pi,H}, r> = Monte Carlo of sum_t gamma^t r(s_{t+1}); reading r(s_t) differs."""
    maze, P, _, mu0 = maze_setup
    task = tasks["goal_5_1"]
    r = task.finite_reward(maze)
    expert = oracle.value_iteration(P, r, GAMMA, tol=1e-12, tie_tol=1e-10).pi
    pi = 0.8 * expert + 0.2 / 5                     # a stochastic policy that reaches the goal
    v_h = oracle.h_step_value(P, pi, r, GAMMA, H)[maze.start_state]
    d_h = oracle.occupancy_next_state(P, pi, mu0, GAMMA, H)
    assert np.isclose(v_h, oracle.discount_normalizer(GAMMA, H) * d_h @ r, atol=1e-10)   # OCC-4(a)
    rng = make_rng(20261005, "oracle", "returns")
    n = 20_000
    traj = maze.rollout(pi, n, H, rng)
    returns = task.discounted_returns(maze.positions(traj.states), GAMMA)
    se = returns.std(ddof=1) / np.sqrt(n)
    assert abs(returns.mean() - v_h) <= 5 * se
    pre_timing = (task.reward(maze.positions(traj.states[:, :-1])) @ (GAMMA ** np.arange(H)))
    assert abs(pre_timing.mean() - v_h) > 10 * se       # the test can tell the two timings apart


def test_policy_evaluation_against_iteration(maze_setup, tasks):
    maze, P, pi, mu0 = maze_setup
    r = tasks["goal_5_5"].finite_reward(maze)
    Q, V = oracle.policy_evaluation(P, pi, r, GAMMA)
    P_pi = oracle.policy_kernel(P, pi)
    V_it = np.zeros(maze.n_states)
    for _ in range(1500):
        V_it = P_pi @ (r + GAMMA * V_it)
    assert np.allclose(V, V_it, atol=1e-10)
    assert np.allclose(V, (pi * Q).sum(axis=1), atol=1e-12)
    M = oracle.successor_next_state(P, pi, GAMMA)
    assert np.allclose(Q, M @ r, atol=1e-10)                       # Q+ = int r dM+


def test_value_iteration_against_brute_force():
    rng = make_rng(20261005, "oracle", "brute")
    S, A, gamma = 4, 3, 0.9
    P = random_mdp(rng, S, A)
    r = rng.uniform(0, 1, S)
    vi = oracle.value_iteration(P, r, gamma, tol=1e-13, tie_tol=1e-11)
    best = np.full(S, -np.inf)
    for acts in itertools.product(range(A), repeat=S):
        _, V = oracle.policy_evaluation(P, oracle.one_hot_policy(np.array(acts), A), r, gamma)
        best = np.maximum(best, V)
    cert = vi.certificates(gamma)
    assert vi.converged and vi.residual <= 1e-13
    assert np.max(np.abs(vi.V - best)) <= cert["value_iterate_error_bound"] + 1e-12
    assert np.all(vi.V_pi >= best - cert["policy_gap_bound"] - 1e-12)
    assert vi.pi_residual <= 1e-11


def test_tie_tolerant_refinement_terminates_with_certificate():
    """Monotone switching (theory ENV-5(a)): even with large tie tolerances the refinement ends tie_tol-greedy."""
    rng = make_rng(20261005, "oracle", "refine")
    for trial in range(300):
        S, A = int(rng.integers(2, 6)), int(rng.integers(2, 4))
        P = random_mdp(rng, S, A)
        r = rng.uniform(0, 1, S)
        gamma = float(rng.uniform(0.5, 0.95))
        kappa = float(rng.uniform(1e-3, 0.3))
        vi = oracle.value_iteration(P, r, gamma, tol=1e-12, tie_tol=kappa)
        assert vi.refinement_converged, trial
        assert vi.pi_residual <= kappa + 1e-12
        best = np.full(S, -np.inf)
        for acts in itertools.product(range(A), repeat=S):
            _, V = oracle.policy_evaluation(P, oracle.one_hot_policy(np.array(acts), A), r, gamma)
            best = np.maximum(best, V)
        assert np.all(best - vi.V_pi <= vi.certificates(gamma)["policy_gap_bound"] + 1e-10)


def test_value_iteration_on_maze_dominates_random_policies(maze_setup, tasks, cfg):
    maze, P, _, _ = maze_setup
    r = tasks["goal_5_5"].finite_reward(maze)
    vi = oracle.value_iteration(P, r, GAMMA, tol=cfg.value_iteration_tol, tie_tol=cfg.tie_tol)
    assert vi.converged and vi.pi_residual <= cfg.tie_tol
    rng = make_rng(20261005, "oracle", "dominance")
    for _ in range(20):
        _, V = oracle.policy_evaluation(P, random_policy(rng, maze.n_states, 5, alpha=0.3), r, GAMMA)
        assert np.all(vi.V_pi >= V - 1e-10)


def test_backward_induction_against_brute_force():
    rng = make_rng(20261005, "oracle", "bi")
    S, A, gamma, Hs = 3, 2, 0.9, 3
    P = random_mdp(rng, S, A)
    r = rng.uniform(0, 1, S)
    bi = oracle.backward_induction(P, r, gamma, Hs)
    best = np.full(S, -np.inf)
    stationary = list(itertools.product(range(A), repeat=S))
    for sigma in itertools.product(stationary, repeat=Hs):          # all non-stationary deterministic Markov policies
        U = np.zeros(S)
        for k in range(Hs - 1, -1, -1):
            pk = oracle.one_hot_policy(np.array(sigma[k]), A)
            U = oracle.policy_kernel(P, pk) @ (r + gamma * U)
        best = np.maximum(best, U)
    assert np.allclose(bi.V_H_star, best, atol=1e-12)
    # the greedy actions attain it
    U = np.zeros(S)
    for k in range(Hs - 1, -1, -1):
        U = oracle.policy_kernel(P, oracle.one_hot_policy(bi.actions[k], A)) @ (r + gamma * U)
    assert np.allclose(U, bi.V_H_star, atol=1e-12)


def test_backward_induction_actions_are_exact_maximizers():
    """Regression (external 3A review): the returned actions attain the reported optimum.

    One step, action 1 best: action 0 is below it by 0.05 or by 1e-11 (inside the
    expert's tie tolerance 1e-10).  The maximizer must be chosen in both cases;
    on an exact tie the first action is chosen.
    """
    r = np.array([1.0, 0.0])
    for gap in (0.05, 1e-11):
        P = np.zeros((2, 2, 2))
        P[:, 0] = [1.0 - gap, gap]
        P[:, 1] = [1.0, 0.0]
        bi = oracle.backward_induction(P, r, 0.9, 1)
        assert np.all(bi.V_H_star == 1.0)
        assert np.all(bi.actions[0] == 1), f"gap {gap}: chose {bi.actions[0]}"
    tie = np.zeros((2, 2, 2))
    tie[:, :, 0] = 1.0
    assert np.all(oracle.backward_induction(tie, r, 0.9, 1).actions[0] == 0)


def test_h_step_gap_of_stationary_optimum(maze_setup, tasks, cfg):
    maze, P, _, _ = maze_setup
    for task in tasks.values():
        r = task.finite_reward(maze)
        vi = oracle.value_iteration(P, r, GAMMA, tol=cfg.value_iteration_tol, tie_tol=cfg.tie_tol)
        bi = oracle.backward_induction(P, r, GAMMA, H)
        U = np.zeros(maze.n_states)                    # the non-stationary policy sigma* attains V*_H
        for k in range(H - 1, -1, -1):
            U = oracle.policy_kernel(P, oracle.one_hot_policy(bi.actions[k], 5)) @ (r + GAMMA * U)
        assert np.allclose(U, bi.V_H_star, rtol=0.0, atol=1e-12)
        v_pi_h = oracle.h_step_value(P, vi.pi, r, GAMMA, H)
        gap = bi.V_H_star - v_pi_h
        assert np.all(gap >= -1e-10)
        assert np.all(gap <= GAMMA**H * r.max() / (1 - GAMMA) + 1e-10)          # Prop. 1.17(c), r >= 0
        rng = make_rng(20261005, "oracle", "hdom", task.name)
        pi = random_policy(rng, maze.n_states, 5, alpha=0.5)
        assert np.all(bi.V_H_star >= oracle.h_step_value(P, pi, r, GAMMA, H) - 1e-10)


@pytest.mark.parametrize("layout, expected", [("four_rooms", 40), ("open", 49)])
@pytest.mark.parametrize("slip", [0.0, 0.1])
def test_transition_rank_is_number_of_states(cfg, layout, expected, slip):
    """FB-3(c), Remark 5.9: a stay action with slip < 1/2 makes rank P~ = #S > 16."""
    rank = oracle.transition_rank(transition_matrix(make_grid_maze(cfg, layout, slip_prob=slip)))
    assert rank.rank == expected and rank.rank > 16
    assert rank.smallest_singular_value > 0.1


def test_transition_rank_detects_low_rank():
    rng = make_rng(1, "lowrank")
    row = rng.dirichlet(np.ones(6))
    P = np.tile(row, (6, 3, 1))
    assert oracle.transition_rank(P).rank == 1


@pytest.mark.parametrize("which", ["tiny", "maze"])
def test_occupancy_jacobian_against_finite_differences(which, maze_setup):
    """Forward-mode Jacobian of d+_{pi,H}, chained through softmax logits, against central differences."""
    rng = make_rng(20261005, "oracle", "jac", which)
    if which == "tiny":
        S, A, Hj, gamma = 4, 3, 7, 0.9
        P = random_mdp(rng, S, A)
        mu0 = rng.dirichlet(np.ones(S))
    else:
        maze, P, _, mu0 = maze_setup
        S, A, Hj, gamma = maze.n_states, 5, 30, GAMMA
    phi = rng.normal(size=(S, A))
    pi = softmax_rows(phi)
    J = oracle.occupancy_next_state_jacobian(P, pi, mu0, gamma, Hj)
    grad = np.einsum("xsb,sba->xsa", J, softmax_policy_jacobian(pi))       # d d(x) / d phi(s, a)
    assert np.allclose(grad.sum(axis=0), 0.0, atol=1e-12)                    # mass is conserved
    h = 1e-6
    idx = list(itertools.product(range(S), range(A)))
    if which == "maze":
        idx = [idx[i] for i in rng.choice(len(idx), 40, replace=False)]
    for s, a in idx:
        e = np.zeros_like(phi)
        e[s, a] = h
        fd = (
            oracle.occupancy_next_state(P, softmax_rows(phi + e), mu0, gamma, Hj)
            - oracle.occupancy_next_state(P, softmax_rows(phi - e), mu0, gamma, Hj)
        ) / (2 * h)
        assert np.allclose(grad[:, s, a], fd, atol=1e-8, rtol=1e-5)


def test_fb_error_diagnostic_exact_representation():
    rng = make_rng(20261005, "oracle", "fb_exact")
    P, nu, z, B, F, gamma = exact_greedy_fb(rng)
    r = rng.uniform(-1, 1, P.shape[0])
    out = oracle.fb_error_diagnostic(F, B, nu, P, z, None, gamma, r=r)
    assert out["sup_tv"] < 1e-10 and out["sup_l2_dual"] < 1e-9
    assert np.max(out["eval_error"]) < 1e-9          # Q+_r of the model's policy equals F^T z_r (FB-2(a))


def test_fb_error_diagnostic_bounds_for_perturbed_models():
    rng = make_rng(20261005, "oracle", "fb_perturbed")
    P, nu, z, B, F, gamma = exact_greedy_fb(rng)
    r = rng.uniform(-1, 1, P.shape[0])
    for tau in (None, 0.5):
        F_pert = F + 0.3 * rng.normal(size=F.shape)
        out = oracle.fb_error_diagnostic(F_pert, B, nu, P, z, tau, gamma, r=r)
        assert out["sup_tv"] > 1e-3
        # Thm 5.17(a): |Q+_r - F^T z_r| <= |r|_A |E(s,a,.)|_B for (sup, TV) and (L2(nu), L2(nu)*)
        assert np.all(out["eval_error"] <= np.max(np.abs(r)) * out["tv"] + 1e-12)
        assert np.all(out["eval_error"] <= np.sqrt(np.sum(nu * r**2)) * out["l2_dual"] + 1e-12)
    nu0 = nu.copy()
    nu0[0] = 0.0
    out = oracle.fb_error_diagnostic(F, B, nu0 / nu0.sum(), P, z, None, gamma)
    assert np.all(np.isinf(out["l2_dual"]))          # M+ charges a nu-null state (Lemma 5.15)
