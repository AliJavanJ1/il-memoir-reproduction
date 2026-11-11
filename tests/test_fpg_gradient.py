"""f-PG: exact gradient checks of the implementation (METHOD_MAP section 3; theory section 3).

The learner module never imports the oracle.  These tests build the exact
objects independently (oracle occupancies, the forward-mode occupancy Jacobian,
exhaustive trajectory enumeration) and compare them with the implemented
estimator, signal-to-go, baselines, cross-fitting and PPO surrogate.

* Exact expectation of the estimator: all trajectories of a tiny MDP are
  enumerated and weighted by their probabilities, so "unbiased" is checked to
  ~1e-12, not statistically.
* Three independent routes to grad J_eps: finite differences of the exact
  objective, the chain rule through ``oracle.occupancy_next_state_jacobian``,
  and the expectation of the implemented estimator.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from il_memoir import divergences as dv
from il_memoir import fpg, oracle
from il_memoir.common import make_rng
from il_memoir.density import uniform_reference
from il_memoir.env_tabular import initial_distribution, transition_matrix
from il_memoir.nets import torch_generator

from _helpers import random_mdp, softmax_rows
from _learner_helpers import (
    central_fd,
    enumerate_trajectories,
    exact_J,
    occupancy_jacobian_logits,
    set_tabular_logits,
)

GAMMA = 0.9
H_TINY = 3
EPS = 0.05


@pytest.fixture(scope="module")
def tiny():
    rng = make_rng(31, "tests", "fpg", "tiny")
    n_s, n_a = 3, 2
    P = random_mdp(rng, n_s, n_a)
    mu0 = rng.dirichlet(np.ones(n_s))
    logits = rng.normal(size=(n_s, n_a))
    q = rng.dirichlet(np.ones(n_s))
    return P, mu0, logits, q


def _policy64(logits):
    pol = fpg.TabularSoftmaxPolicy(logits.shape[0], logits.shape[1], dtype=torch.float64)
    set_tabular_logits(pol, logits)
    return pol


def _exact_expectation(policy, states, actions, probs, signal_next, beta, gamma):
    """E[estimator] by enumeration: scale each trajectory's signal and baseline by N * P(traj) (the estimator is linear in them)."""
    n = states.shape[0]
    scale = (n * probs)[:, None]
    beta = np.broadcast_to(np.asarray(beta, dtype=np.float64), signal_next.shape)
    return fpg.gradient_estimate(policy, states, actions, signal_next * scale, beta * scale, gamma)


# ----------------------------------------------------------------------------- signal-to-go
def test_rewards_to_go_matches_double_sum():
    rng = make_rng(1, "tests", "fpg", "rtg")
    sig = rng.normal(size=(4, 7))
    got = fpg.rewards_to_go(sig, 0.8)
    want = np.array([[sum(0.8 ** (t - k) * sig[i, t] for t in range(k, 7)) for k in range(7)] for i in range(4)])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_score_weights_sum_to_one_and_discount():
    w = fpg.score_weights(5, 0.9)
    np.testing.assert_allclose(w, 0.9 ** np.arange(5) / sum(0.9 ** np.arange(5)), atol=1e-15)
    assert abs(w.sum() - 1) < 1e-15


# ----------------------------------------------------------------------------- exact gradient, three routes
@pytest.mark.parametrize("name", dv.GENERATORS)
def test_estimator_expectation_equals_exact_gradient_and_finite_differences(tiny, name):
    P, mu0, logits, q = tiny
    ref = uniform_reference(3)
    pi = softmax_rows(logits)
    d = oracle.occupancy_next_state(P, pi, mu0, GAMMA, H_TINY)
    jac = occupancy_jacobian_logits(P, logits, mu0, GAMMA, H_TINY)
    exact = fpg.exact_gradient(d, q, jac, name, EPS, ref)
    fd = central_fd(lambda x: exact_J(P, x, mu0, GAMMA, H_TINY, q, name, EPS, ref), logits)
    np.testing.assert_allclose(exact, fd, rtol=0, atol=1e-8)
    states, actions, probs = enumerate_trajectories(P, pi, mu0, H_TINY)
    assert abs(probs.sum() - 1) < 1e-12
    g_cells = dv.smoothed_signal(d, q, name, EPS, ref)          # exact signal (1 - eps) f'(d_eps / q_eps)
    signal_next = g_cells[states[:, 1:]]
    est = _exact_expectation(_policy64(logits), states, actions, probs, signal_next, 0.0, GAMMA)
    np.testing.assert_allclose(est, exact, rtol=0, atol=1e-12)


def test_dropping_score_discount_changes_the_expectation(tiny):
    """Common practice drops gamma^k on the score; that is not grad J (theory Thm 3.7 remark)."""
    P, mu0, logits, q = tiny
    ref = uniform_reference(3)
    pi = softmax_rows(logits)
    d = oracle.occupancy_next_state(P, pi, mu0, GAMMA, H_TINY)
    exact = fpg.exact_gradient(d, q, occupancy_jacobian_logits(P, logits, mu0, GAMMA, H_TINY), "js", EPS, ref)
    states, actions, probs = enumerate_trajectories(P, pi, mu0, H_TINY)
    G = fpg.rewards_to_go(dv.smoothed_signal(d, q, "js", EPS, ref)[states[:, 1:]], GAMMA)
    w_flat = np.full(H_TINY, 1.0 / fpg.discount_normalizer(GAMMA, H_TINY))     # no gamma^k
    pol = _policy64(logits)
    logp = fpg.log_probs(pol, states[:, :-1], actions)
    loss = (logp * torch.as_tensor(w_flat[None, :] * G * probs[:, None])).sum()
    flat = torch.autograd.grad(loss, [pol.phi])[0].numpy().reshape(-1)
    # the implemented estimator matches to 1e-12 (test above); this variant is off by ~10% of |grad J|
    assert np.max(np.abs(flat - exact)) > 1e-5


@pytest.mark.parametrize("kind", ["time", "state_time"])
def test_baselines_leave_the_expectation_unchanged(tiny, kind):
    P, mu0, logits, q = tiny
    ref = uniform_reference(3)
    pi = softmax_rows(logits)
    d = oracle.occupancy_next_state(P, pi, mu0, GAMMA, H_TINY)
    states, actions, probs = enumerate_trajectories(P, pi, mu0, H_TINY)
    sig = dv.smoothed_signal(d, q, "kl_dq", EPS, ref)[states[:, 1:]]
    pol = _policy64(logits)
    base0 = _exact_expectation(pol, states, actions, probs, sig, 0.0, GAMMA)
    rng = make_rng(5, "tests", "baseline", kind)
    if kind == "time":
        beta = np.broadcast_to(rng.normal(size=H_TINY) * 3, sig.shape)
    else:
        table = rng.normal(size=(3, H_TINY)) * 3                # beta(s_k, k): F_k-measurable
        beta = table[states[:, :-1], np.arange(H_TINY)[None, :]]
    with_b = _exact_expectation(pol, states, actions, probs, sig, beta, GAMMA)
    np.testing.assert_allclose(with_b, base0, rtol=0, atol=1e-12)
    # negative control (Prop. 3.8(c)): a "baseline" that depends on a_k changes the expectation
    bad = 2.0 * (actions == 1)
    changed = _exact_expectation(pol, states, actions, probs, sig, bad, GAMMA)
    assert np.max(np.abs(changed - base0)) > 1e-3


def test_fixed_independent_signal_gives_sum_g_grad_d(tiny):
    """Prop. 3.10(a): with a fixed signal function g_hat, E[estimator] = sum_b g_hat(b) grad d(b)."""
    P, mu0, logits, _ = tiny
    pi = softmax_rows(logits)
    g_hat = make_rng(8, "tests", "ghat").normal(size=3)
    jac = occupancy_jacobian_logits(P, logits, mu0, GAMMA, H_TINY)
    states, actions, probs = enumerate_trajectories(P, pi, mu0, H_TINY)
    est = _exact_expectation(_policy64(logits), states, actions, probs, g_hat[states[:, 1:]], 0.0, GAMMA)
    np.testing.assert_allclose(est, g_hat @ jac, rtol=0, atol=1e-12)
    # adding a constant to the signal leaves the expectation unchanged (Prop. 3.8(b))
    est_c = _exact_expectation(_policy64(logits), states, actions, probs, g_hat[states[:, 1:]] + 7.0, 0.0, GAMMA)
    np.testing.assert_allclose(est_c, est, rtol=0, atol=1e-11)


# ----------------------------------------------------------------------------- cross-fitting (U-6)
def _two_branch_mdp():
    """x -> y0 (action 0) or y1 (action 1), deterministic; y0, y1 absorbing (theory Prop. 3.10(b) proof)."""
    P = np.zeros((3, 2, 3))
    P[0, 0, 1] = P[0, 1, 2] = 1.0
    P[1, :, 1] = P[2, :, 2] = 1.0
    mu0 = np.array([1.0, 0.0, 0.0])
    return P, mu0


def test_same_batch_signal_is_biased_counterexample():
    """Prop. 3.10(b): N = 1, H = 1, same batch: the estimator has mean 0 while grad J_eps != 0."""
    P, mu0 = _two_branch_mdp()
    logits = np.array([[0.0, np.log(3.0)], [0.0, 0.0], [0.0, 0.0]])        # pi(a1 | x) = 3/4
    q = np.array([0.0, 0.5, 0.5])
    est = fpg.SignalEstimator("counts", "kl_dq", EPS, GAMMA, n_states=3)
    est.target_masses = q.copy()          # fixed target on the three cells
    pi = softmax_rows(logits)
    states, actions, probs = enumerate_trajectories(P, pi, mu0, 1)
    pol = _policy64(logits)
    mean = np.zeros(6)
    for i in range(states.shape[0]):
        cf = fpg.crossfit_advantages(est, states[i:i + 1, 1:], n_folds=1, baseline_kind="none")
        mean += probs[i] * fpg.gradient_estimate(pol, states[i:i + 1], actions[i:i + 1], cf.signal, 0.0, GAMMA)
    ref = uniform_reference(3)
    d = oracle.occupancy_next_state(P, pi, mu0, GAMMA, 1)
    exact = fpg.exact_gradient(d, q, occupancy_jacobian_logits(P, logits, mu0, GAMMA, 1), "kl_dq", EPS, ref)
    np.testing.assert_allclose(mean, 0.0, atol=1e-14)
    assert np.max(np.abs(exact)) > 0.05


def test_crossfit_signal_is_independent_of_the_score_trajectory():
    """With two folds and N = 2, E[estimator] = sum_b E[g_hat](b) grad d(b) exactly (Prop. 3.10(a)-(b))."""
    rng = make_rng(17, "tests", "crossfit")
    P = random_mdp(rng, 3, 2)
    mu0 = np.array([1.0, 0.0, 0.0])
    logits = rng.normal(size=(3, 2))
    pi = softmax_rows(logits)
    H = 2
    q = rng.dirichlet(np.ones(3))
    est = fpg.SignalEstimator("counts", "js", EPS, GAMMA, n_states=3)
    est.target_masses = q
    states, actions, probs = enumerate_trajectories(P, pi, mu0, H)
    pol = _policy64(logits)
    n = states.shape[0]
    mean = np.zeros(6)
    mean_g = np.zeros(3)
    for i in range(n):
        g_i = dv.smoothed_signal(est.occupancy_estimate(states[i:i + 1, 1:]), q, "js", EPS, est.reference)
        mean_g += probs[i] * g_i
        for j in range(n):
            pair_s = np.stack([states[i], states[j]])
            pair_a = np.stack([actions[i], actions[j]])
            cf = fpg.crossfit_advantages(est, pair_s[:, 1:], n_folds=2, baseline_kind="none")
            # trajectory 0 must use the signal of trajectory 1 only, and vice versa
            g_j = dv.smoothed_signal(est.occupancy_estimate(states[j:j + 1, 1:]), q, "js", EPS, est.reference)
            np.testing.assert_allclose(cf.signal[0], g_j[states[i, 1:]], atol=1e-15)
            mean += probs[i] * probs[j] * fpg.gradient_estimate(pol, pair_s, pair_a, cf.signal, 0.0, GAMMA)
    jac = occupancy_jacobian_logits(P, logits, mu0, GAMMA, H)
    np.testing.assert_allclose(mean, mean_g @ jac, rtol=0, atol=1e-12)


# ----------------------------------------------------------------------------- the real maze
@pytest.fixture(scope="module")
def maze_setup(four_rooms):
    P = transition_matrix(four_rooms)
    mu0 = initial_distribution(four_rooms)
    rng = make_rng(3, "tests", "fpg", "maze")
    logits = 0.7 * rng.normal(size=(four_rooms.n_states, four_rooms.n_actions))
    pi_t = softmax_rows(rng.normal(size=logits.shape))
    q = oracle.occupancy_next_state(P, pi_t, mu0, 0.95, 100)          # a fixed target occupancy
    return P, mu0, logits, q


def test_maze_exact_gradient_matches_finite_differences(maze_setup):
    P, mu0, logits, q = maze_setup
    ref = uniform_reference(P.shape[0])
    d = oracle.occupancy_next_state(P, softmax_rows(logits), mu0, 0.95, 100)
    jac = occupancy_jacobian_logits(P, logits, mu0, 0.95, 100)
    exact = fpg.exact_gradient(d, q, jac, "js", 0.01, ref)
    fd = central_fd(lambda x: exact_J(P, x, mu0, 0.95, 100, q, "js", 0.01, ref), logits, h=1e-6)
    np.testing.assert_allclose(exact, fd, rtol=0, atol=1e-8)
    # FPG-4(b): a small exact descent step lowers J
    j0 = exact_J(P, logits, mu0, 0.95, 100, q, "js", 0.01, ref)
    j1 = exact_J(P, logits - 0.5 * exact.reshape(logits.shape), mu0, 0.95, 100, q, "js", 0.01, ref)
    assert j1 < j0


def test_maze_monte_carlo_estimator_within_standard_errors(maze_setup, four_rooms):
    """Real simulator rollouts (H = 100) with the exact signal: the sample mean is within 5 SE of grad J_eps."""
    P, mu0, logits, q = maze_setup
    ref = uniform_reference(P.shape[0])
    pi = softmax_rows(logits)
    d = oracle.occupancy_next_state(P, pi, mu0, 0.95, 100)
    exact = fpg.exact_gradient(d, q, occupancy_jacobian_logits(P, logits, mu0, 0.95, 100), "js", 0.01, ref)
    g_cells = dv.smoothed_signal(d, q, "js", 0.01, ref)
    rng = make_rng(4, "tests", "fpg", "mc")
    pilot = four_rooms.rollout(pi, 500, 100, rng)                 # independent batch for the time baseline
    beta = fpg.rewards_to_go(g_cells[pilot.states[:, 1:]], 0.95).mean(axis=0)
    traj = four_rooms.rollout(pi, 6000, 100, rng)
    G = fpg.rewards_to_go(g_cells[traj.states[:, 1:]], 0.95)
    A = (G - beta[None, :]) * fpg.score_weights(100, 0.95)[None, :]
    # per-trajectory estimator from the analytic softmax score (Lemma 3.2): row s_k gets A_k (e_{a_k} - pi(.|s_k))
    n, n_s, n_a = traj.actions.shape[0], P.shape[0], P.shape[1]
    s, a = traj.states[:, :-1], traj.actions
    per = np.zeros((n, n_s, n_a))
    rows = np.repeat(np.arange(n), 100)
    np.add.at(per, (rows, s.ravel()), -(A.ravel()[:, None] * pi[s.ravel()]))
    np.add.at(per, (rows, s.ravel(), a.ravel()), A.ravel())
    per = per.reshape(n, -1)
    # the autograd estimator equals the mean of the analytic per-trajectory vectors
    auto = fpg.gradient_estimate(_policy64(logits), traj.states, traj.actions, g_cells[traj.states[:, 1:]], beta, 0.95)
    np.testing.assert_allclose(auto, per.mean(axis=0), rtol=0, atol=1e-10)
    se = per.std(axis=0, ddof=1) / np.sqrt(n)
    assert np.all(np.abs(per.mean(axis=0) - exact) <= 5 * se + 1e-6)


# ----------------------------------------------------------------------------- PPO surrogate (optional)
def test_ppo_surrogate_gradient_at_sampling_parameter(maze_setup, four_rooms):
    P, mu0, logits, q = maze_setup
    pol = _policy64(logits)
    rng = make_rng(6, "tests", "ppo")
    traj = four_rooms.rollout(softmax_rows(logits), 30, 100, rng)
    adv = rng.normal(size=traj.actions.shape)
    with torch.no_grad():
        logp_old = fpg.log_probs(pol, traj.states[:, :-1], traj.actions)
    c = fpg.score_weights(100, 0.95)[None, :] * adv
    clip = fpg.ppo_surrogate(pol, traj.states, traj.actions, logp_old, c, 0.2)
    g_clip = torch.autograd.grad(clip, [pol.phi])[0].numpy()
    plain = fpg.surrogate_loss(pol, traj.states, traj.actions, adv, 0.95)
    g_plain = torch.autograd.grad(plain, [pol.phi])[0].numpy()
    np.testing.assert_allclose(g_clip, g_plain, rtol=0, atol=1e-13)
    # away from phi': L_clip >= L (unclipped ratio surrogate), and they differ for a large move
    set_tabular_logits(pol, logits + 2.0 * rng.normal(size=logits.shape))
    with torch.no_grad():
        ratio = torch.exp(fpg.log_probs(pol, traj.states[:, :-1], traj.actions) - logp_old)
        unclipped = (ratio * torch.as_tensor(c)).sum(1).mean()
        clipped = fpg.ppo_surrogate(pol, traj.states, traj.actions, logp_old, c, 0.2)
    assert float(clipped) >= float(unclipped) - 1e-12
    assert float(clipped) > float(unclipped) + 1e-6


# ----------------------------------------------------------------------------- state-action diagnostic
def test_state_action_diagnostic_keeps_current_step(tiny):
    """FPG-6: E[gradient_estimate_sa] equals finite differences of J^SA_eps; dropping the t = k term does not."""
    P, mu0, logits, _ = tiny
    pi = softmax_rows(logits)
    rng = make_rng(9, "tests", "sa")
    q_sa = rng.dirichlet(np.ones(6)).reshape(3, 2)
    ref = np.full(6, 1 / 6)

    def j_sa(x):
        dsa = oracle.occupancy_state_action(P, softmax_rows(x), mu0, GAMMA, H_TINY)
        return dv.smoothed_divergence(dsa.ravel(), q_sa.ravel(), "js", EPS, ref)

    fd = central_fd(j_sa, logits)
    dsa = oracle.occupancy_state_action(P, pi, mu0, GAMMA, H_TINY)
    g_sa = dv.smoothed_signal(dsa.ravel(), q_sa.ravel(), "js", EPS, ref).reshape(3, 2)
    states, actions, probs = enumerate_trajectories(P, pi, mu0, H_TINY)
    sig = g_sa[states[:, :-1], actions]
    n = states.shape[0]
    pol = _policy64(logits)
    est = fpg.gradient_estimate_sa(pol, states, actions, sig * (n * probs)[:, None], GAMMA)
    np.testing.assert_allclose(est, fd, rtol=0, atol=1e-8)
    # dropping the current-step term g(s_k, a_k) (treating it as a baseline) is wrong
    G = fpg.rewards_to_go(sig, GAMMA) - sig
    logp = fpg.log_probs(pol, states[:, :-1], actions)
    w = fpg.score_weights(H_TINY, GAMMA)[None, :] * G * probs[:, None]
    wrong = torch.autograd.grad((logp * torch.as_tensor(w)).sum(), [pol.phi])[0].numpy().reshape(-1)
    assert np.max(np.abs(wrong - fd)) > 1e-3


# ----------------------------------------------------------------------------- MLP policy
def _positions(states):
    """Embed the tiny MDP's states as fixed positions, so the MLP policy can act on them."""
    table = np.array([[0.2, 0.3], [0.5, 0.7], [0.8, 0.4]])
    return table[states]


def test_mlp_score_matches_finite_differences():
    pol = fpg.MLPPolicy(5, 16, 2, "tanh", torch_generator(11), out_init_scale=1.0, dtype=torch.float64)
    x = np.array([0.31, 0.62])
    params = [p for p in pol.parameters()]
    flat0 = torch.cat([p.detach().reshape(-1) for p in params]).numpy()

    def logp(vec):
        with torch.no_grad():
            off = 0
            for p in params:
                p.copy_(torch.as_tensor(vec[off:off + p.numel()]).reshape(p.shape))
                off += p.numel()
            return float(fpg.log_probs(pol, x[None, :], np.array([3]))[0])

    fd = central_fd(logp, flat0)
    logp(flat0)
    chi = fpg.score(pol, x, 3)
    np.testing.assert_allclose(chi, fd, rtol=0, atol=1e-8)
    probs = fpg.policy_probs(pol, np.random.default_rng(0).random((10, 2)))
    np.testing.assert_allclose(probs.sum(-1).detach().numpy(), 1.0, atol=1e-12)


def test_mlp_estimator_expectation_equals_chain_rule_gradient(tiny):
    """The MLP path: E[estimator] (enumeration) = sum_b g(b) grad d(b), with grad d from the oracle Jacobian
    and grad pi(a|s) from autograd (chain rule through the policy entries)."""
    P, mu0, _, q = tiny
    pol = fpg.MLPPolicy(2, 8, 2, "tanh", torch_generator(12), out_init_scale=1.0, dtype=torch.float64)
    pos = _positions(np.arange(3))
    probs_t = fpg.policy_probs(pol, pos)                         # (3, 2), differentiable
    pi = probs_t.detach().numpy()
    params = list(pol.parameters())
    dpi = np.stack([
        np.concatenate([g.reshape(-1).numpy() for g in torch.autograd.grad(probs_t[s, a], params, retain_graph=True)])
        for s in range(3) for a in range(2)
    ]).reshape(3, 2, -1)                                          # d pi(a|s) / d phi
    jac_pi = oracle.occupancy_next_state_jacobian(P, pi, mu0, GAMMA, H_TINY)   # [x, s, a]
    jac = np.einsum("xsa,sap->xp", jac_pi, dpi)
    ref = uniform_reference(3)
    d = oracle.occupancy_next_state(P, pi, mu0, GAMMA, H_TINY)
    exact = fpg.exact_gradient(d, q, jac, "kl_qd", EPS, ref)
    states, actions, probs = enumerate_trajectories(P, pi, mu0, H_TINY)
    sig = dv.smoothed_signal(d, q, "kl_qd", EPS, ref)[states[:, 1:]]
    n = states.shape[0]
    est = fpg.gradient_estimate(pol, _positions(states), actions, sig * (n * probs)[:, None], 0.0, GAMMA)
    np.testing.assert_allclose(est, exact, rtol=0, atol=1e-11)


# ----------------------------------------------------------------------------- configuration
def test_config_rejects_unknown_and_invalid_settings():
    with pytest.raises(ValueError):
        fpg.FPGConfig.from_dict({"lr": 0.1, "learning_rate": 0.1})
    with pytest.raises(ValueError):
        fpg.FPGConfig(n_traj=1, n_folds=2)
    with pytest.raises(ValueError):
        fpg.FPGConfig(activation="relu")                         # C^1 activations only (U-12)
    with pytest.raises(ValueError):
        fpg.FPGConfig(divergence="kl")
    with pytest.raises(ValueError):
        fpg.FPGConfig(lr=float("nan"))
    cfg = fpg.FPGConfig.from_dict(fpg.FPGConfig().to_dict())
    assert cfg == fpg.FPGConfig()


# ----------------------------------------------------------------------------- estimator wiring
def test_time_baseline_comes_from_the_other_folds():
    """beta for fold k is the mean signal-to-go of the other folds under g_hat_{-k} (Prop. 3.8(a) conditionally)."""
    rng = make_rng(61, "tests", "baseline", "folds")
    est = fpg.SignalEstimator("counts", "js", EPS, GAMMA, n_states=5)
    est.target_masses = rng.dirichlet(np.ones(5))
    nxt = rng.integers(0, 5, size=(6, 4))
    cf = fpg.crossfit_advantages(est, nxt, n_folds=2, baseline_kind="time")
    folds = fpg.fold_ids(6, 2)
    for k in (0, 1):
        own, other = folds == k, folds != k
        g_other = est.signal(est.occupancy_estimate(nxt[other]), nxt[other])
        want = fpg.rewards_to_go(g_other, GAMMA).mean(axis=0)
        np.testing.assert_allclose(cf.baseline[own], np.broadcast_to(want, (own.sum(), 4)), atol=1e-14)
        np.testing.assert_allclose(cf.signal[own], est.signal(est.occupancy_estimate(nxt[other]), nxt[own]), atol=1e-14)
    np.testing.assert_allclose(cf.to_go, fpg.rewards_to_go(cf.signal, GAMMA), atol=1e-14)


def test_expert_target_uses_discounted_next_states(four_rooms, cont_env, cfg):
    """q_hat_E weights s^i_{t+1} by gamma^t / (N_E Z_H) (ENV-4(a)); s_0 is not used."""
    from il_memoir.density import FreeSpaceHistogram, finite_occupancy, discount_weights as dw
    view = cfg.learner_dict()
    rng = make_rng(62, "tests", "target")
    demos = four_rooms.rollout(np.full((40, 5), 0.2), 4, 100, rng).states
    learner = fpg.FPGLearner(four_rooms, view, demos, fpg.FPGConfig(), seed=0)
    np.testing.assert_allclose(learner.estimator.target_masses, finite_occupancy(demos[:, 1:], 40, 0.95), atol=1e-15)
    cdemos = cont_env.rollout(lambda x: np.full((x.shape[0], 5), 0.2), 4, 100, rng).states
    for density in ("histogram", "kde"):
        learner_c = fpg.FPGLearner(cont_env, view, cdemos, fpg.FPGConfig(density=density), seed=0)
        hist = FreeSpaceHistogram(fpg.free_space_from_view(view), 20)
        want = hist.masses(cdemos[:, 1:].reshape(-1, 2), dw(4, 100, 0.95).reshape(-1))
        np.testing.assert_allclose(learner_c.estimator.target_masses, want, atol=1e-15)
        if density == "kde":
            np.testing.assert_array_equal(learner_c.estimator.target_kde.points, cdemos[:, 1:].reshape(-1, 2))


@pytest.mark.parametrize("name", dv.GENERATORS)
def test_kde_signal_mixes_densities_with_the_free_space_density(cont_env, cfg, name):
    """KDE signal: (1 - eps) f'(((1-eps) p + eps/|S|) / ((1-eps) q + eps/|S|)) at the query points (NOTATION section 4)."""
    from il_memoir.density import WeightedKDE, discount_weights as dw
    view = cfg.learner_dict()
    rng = make_rng(63, "tests", "kde-signal")
    demos = cont_env.rollout(lambda x: np.full((x.shape[0], 5), 0.2), 4, 100, rng).states
    learner = fpg.FPGLearner(cont_env, view, demos, fpg.FPGConfig(density="kde", divergence=name), seed=0)
    est = learner.estimator
    nxt = cont_env.rollout(lambda x: np.full((x.shape[0], 5), 0.2), 6, 100, rng).states[:, 1:]
    kde = est.occupancy_estimate(nxt[:3])
    got = est.signal(kde, nxt[3:])
    space = fpg.free_space_from_view(view)
    p = WeightedKDE(space, nxt[:3].reshape(-1, 2), dw(3, 100, 0.95).reshape(-1), 0.05).density(nxt[3:].reshape(-1, 2))
    q = WeightedKDE(space, demos[:, 1:].reshape(-1, 2), dw(4, 100, 0.95).reshape(-1), 0.05).density(nxt[3:].reshape(-1, 2))
    eps, ref = 0.01, 1.0 / space.area
    want = (1 - eps) * dv.fprime(name, ((1 - eps) * p + eps * ref) / ((1 - eps) * q + eps * ref))
    np.testing.assert_allclose(got.reshape(-1), want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", dv.GENERATORS)
def test_histogram_signal_mixes_masses_with_the_free_area_law(cont_env, cfg, name):
    """Histogram signal: (1 - eps) f'(d_eps(b) / q_eps(b)) with upsilon_b = |B_b cap S| / |S| (masses with masses)."""
    from il_memoir.density import discount_weights as dw
    view = cfg.learner_dict()
    rng = make_rng(64, "tests", "hist-signal")
    demos = cont_env.rollout(lambda x: np.full((x.shape[0], 5), 0.2), 4, 100, rng).states
    learner = fpg.FPGLearner(cont_env, view, demos, fpg.FPGConfig(density="histogram", divergence=name), seed=0)
    est = learner.estimator
    nxt = cont_env.rollout(lambda x: np.full((x.shape[0], 5), 0.2), 6, 100, rng).states[:, 1:]
    got = est.signal(est.occupancy_estimate(nxt[:3]), nxt[3:])
    # independent reconstruction: bins of side 1/20 with positive free area, masses by direct counting
    space = fpg.free_space_from_view(view)
    edges = np.linspace(0, 1, 21)
    rects = np.array([[edges[i], edges[j], edges[i + 1], edges[j + 1]] for i in range(20) for j in range(20)])
    area = space.rect_free_area(rects)
    keep = area > 1e-14
    ups = area[keep] / area[keep].sum()
    index = -np.ones(400, dtype=int)
    index[keep] = np.arange(keep.sum())

    def masses(points, weights):
        ij = np.clip(np.floor(points * 20).astype(int), 0, 19)
        ids = index[ij[:, 0] * 20 + ij[:, 1]]
        assert (ids >= 0).all()
        return np.bincount(ids, weights=weights, minlength=keep.sum()), ids

    d, _ = masses(nxt[:3].reshape(-1, 2), dw(3, 100, 0.95).reshape(-1))
    q, _ = masses(demos[:, 1:].reshape(-1, 2), dw(4, 100, 0.95).reshape(-1))
    _, ids = masses(nxt[3:].reshape(-1, 2), np.ones(300))
    eps = 0.01
    want = (1 - eps) * dv.fprime(name, ((1 - eps) * d + eps * ups) / ((1 - eps) * q + eps * ups))
    np.testing.assert_allclose(got.reshape(-1), want[ids], rtol=1e-12, atol=1e-12)
    assert keep.sum() == 356


@pytest.mark.parametrize("n_traj, n_folds, H, name", [(3, 2, 1, "js"), (3, 2, 1, "kl_qd"), (3, 3, 1, "js"), (2, 2, 2, "kl_dq")])
def test_default_estimator_exact_expectation(n_traj, n_folds, H, name):
    """The learner's default estimator (K folds, time baseline from the other folds), uneven folds included:
    E[estimator] = (1/N) sum_k n_k sum_b E[g_hat_{-k}](b) grad d(b), by exhaustive enumeration (Prop. 7.11)."""
    import itertools

    rng = make_rng(65, "tests", "default-estimator", n_traj, n_folds, H)
    P = random_mdp(rng, 3, 2)
    mu0 = np.array([1.0, 0.0, 0.0])
    logits = rng.normal(size=(3, 2))
    pi = softmax_rows(logits)
    est = fpg.SignalEstimator("counts", name, EPS, GAMMA, n_states=3)
    est.target_masses = rng.dirichlet(np.ones(3))
    states, actions, probs = enumerate_trajectories(P, pi, mu0, H)
    pol = _policy64(logits)
    n_paths = states.shape[0]
    mean = np.zeros(6)
    for combo in itertools.product(range(n_paths), repeat=n_traj):
        idx = np.array(combo)
        p = np.prod(probs[idx])
        cf = fpg.crossfit_advantages(est, states[idx, 1:], n_folds, "time")
        mean += p * fpg.gradient_estimate(pol, states[idx], actions[idx], cf.signal, cf.baseline, GAMMA)
    folds = fpg.fold_ids(n_traj, n_folds)
    jac = occupancy_jacobian_logits(P, logits, mu0, GAMMA, H)
    want = np.zeros(6)
    for k in range(n_folds):
        n_k = int((folds == k).sum())
        n_other = n_traj - n_k
        g_mean = np.zeros(3)
        for combo in itertools.product(range(n_paths), repeat=n_other):
            idx = np.array(combo)
            g = dv.smoothed_signal(est.occupancy_estimate(states[idx, 1:]), est.target_masses, name, EPS, est.reference)
            g_mean += np.prod(probs[idx]) * g
        want += n_k / n_traj * (g_mean @ jac)
    np.testing.assert_allclose(mean, want, rtol=0, atol=1e-12)


def test_config_same_batch_flag_and_integer_bins():
    with pytest.raises(ValueError):
        fpg.FPGConfig(n_folds=1)                                    # same batch needs an explicit opt-in
    assert fpg.FPGConfig(n_folds=1, same_batch=True).n_folds == 1
    with pytest.raises(ValueError):
        fpg.FPGConfig(same_batch=True)                              # ... and only with one fold
    for bad in (20.5, "20", True, 0):
        with pytest.raises(ValueError):
            fpg.FPGConfig(histogram_bins=bad)
    with pytest.raises(ValueError):
        fpg.FPGConfig(kde_bandwidth="0.05")


def test_learning_rate_defaults_depend_on_the_policy_class(four_rooms, cont_env, cfg):
    view = cfg.learner_dict()
    rng = make_rng(66, "tests", "lr")
    demos = four_rooms.rollout(np.full((40, 5), 0.2), 4, 100, rng).states
    cdemos = cont_env.rollout(lambda x: np.full((x.shape[0], 5), 0.2), 4, 100, rng).states
    tab = fpg.FPGLearner(four_rooms, view, demos, fpg.FPGConfig(), seed=0)
    mlp = fpg.FPGLearner(cont_env, view, cdemos, fpg.FPGConfig(), seed=0)
    assert tab.optimizer.param_groups[0]["lr"] == fpg.DEFAULT_LR_TABULAR == tab.resolved["lr"]
    assert mlp.optimizer.param_groups[0]["lr"] == fpg.DEFAULT_LR_MLP == mlp.resolved["lr"]
    assert fpg.FPGLearner(four_rooms, view, demos, fpg.FPGConfig(lr=0.3), seed=0).optimizer.param_groups[0]["lr"] == 0.3
