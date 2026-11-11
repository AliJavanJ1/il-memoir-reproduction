"""FB pretraining: algebra of the implemented functions against explicit exact finite factorizations.

METHOD_MAP section 4; theory section 5.  A trained network is never the only
test of the algebra: every identity below is checked with an explicit exact
finite FB representation (one-hot ``B``, ``d = #S``, ``F = M+ / nu`` for the
softmax fixed point; theory Prop. 5.10), or with arbitrary tables, and with
exact enumeration of transitions and target states instead of sampling.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from torch import nn

from il_memoir import fb, oracle
from il_memoir.common import make_rng
from il_memoir.env_tabular import transition_matrix
from il_memoir.nets import ObsSpec, torch_generator

from _helpers import random_mdp, softmax_rows
from _learner_helpers import exact_softmax_fb

GAMMA = 0.9
TAU = 0.5


def t64(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


@pytest.fixture(scope="module")
def mdp():
    rng = make_rng(41, "tests", "fb", "mdp")
    n_s, n_a = 4, 3
    P = random_mdp(rng, n_s, n_a)
    d_sa = rng.dirichlet(np.ones(n_s * n_a)).reshape(n_s, n_a)       # replay law of (s, a)
    nu = np.einsum("sa,sat->t", d_sa, P)                              # law of the replay next state (A-DATA (ii))
    zs = [rng.normal(size=n_s) for _ in range(2)]
    w_z = np.array([0.4, 0.6])
    return P, d_sa, nu, zs, w_z


def _transitions(P, d_sa, w_z):
    """Enumerated replay transitions (s, a, s1, k) and their probabilities D_SA(s,a) P(s1|s,a) w_z[k]."""
    n_s, n_a, _ = P.shape
    rows, probs = [], []
    for s, a, s1, k in itertools.product(range(n_s), range(n_a), range(n_s), range(len(w_z))):
        p = d_sa[s, a] * P[s, a, s1] * w_z[k]
        if p > 0:
            rows.append((s, a, s1, k))
            probs.append(p)
    return np.array(rows), np.array(probs)


def _population_loss(F_tabs, B_tab, Fm_tabs, Bm_tab, zs, P, d_sa, nu, w_z):
    """L of Eq. (5.5) with the implementation's successor_loss, exact weights (transitions x targets)."""
    rows, probs = _transitions(P, d_sa, w_z)
    s, a, s1, k = rows.T
    F_i = torch.stack([F_tabs[kk][ss, aa] for ss, aa, kk in zip(s, a, k)])
    Fbar = torch.stack([fb.expected_features(Fm_tabs[kk][s11], t64(zs[kk]), TAU) for s11, kk in zip(s1, k)])
    B_next = B_tab[t64(s1).long()]
    return fb.successor_loss(F_i, B_tab, Fbar.detach(), Bm_tab.detach(), B_next, GAMMA, w_trans=t64(probs), w_target=t64(nu))


# ----------------------------------------------------------------------------- policy family and FB relations
def test_exact_softmax_representation_relations(mdp):
    """Policy family (5.1), the stated FB measure relation, FB-2(a), FB-7 and FB-10(c) with the implementation functions."""
    P, _, nu, zs, _ = mdp
    n_s = P.shape[0]
    B = np.eye(n_s)
    G = fb.gram_from_features(t64(B), t64(nu)).numpy()
    np.testing.assert_allclose(G, np.diag(nu), atol=1e-15)                # uncentered Gram of one-hot B under nu
    for z in zs:
        F, pi_fp, res = exact_softmax_fb(P, nu, z, TAU, GAMMA)
        assert res < 1e-11
        pi = fb.policy_probs(t64(F), t64(z), TAU).numpy()
        np.testing.assert_allclose(pi, pi_fp, atol=1e-10)                 # softmax(F^T z / tau) is the fixed point
        M = oracle.successor_next_state(P, pi, GAMMA)
        M_hat = np.einsum("sad,xd->sax", F, B) * nu[None, None, :]          # F^T B nu
        np.testing.assert_allclose(M_hat, M, atol=1e-9)                     # exact: M_hat_z = M^{pi_z,+}
        r = make_rng(2, "tests", "fb", "reward").normal(size=n_s)
        z_r = B.T @ (nu * r)                                                # z_r = E_nu[r B]
        Q, _ = oracle.policy_evaluation(P, pi, r, GAMMA)
        np.testing.assert_allclose(np.einsum("sad,d->sa", F, z_r), Q, atol=1e-9)   # FB-2(a)
        np.testing.assert_allclose(np.einsum("de,sae->sad", G, F), np.einsum("sax,xd->sad", M, B), atol=1e-9)  # FB-7
        mean_b = B.T @ nu
        np.testing.assert_allclose(np.einsum("sad,d->sa", F, mean_b), 1.0 / (1.0 - GAMMA), atol=1e-9)        # FB-10(c)
        Fz = fb.expected_features(t64(F), t64(z), TAU).numpy()
        np.testing.assert_allclose(Fz, np.einsum("sa,sad->sd", pi, F), atol=1e-12)


# ----------------------------------------------------------------------------- FB-8: the successor loss
def test_population_loss_is_residual_minus_constant(mdp):
    """Prop. 5.22(a): L = sum D_SA [1/2 ||M_hat - T M_hat^-||^2_{L2(nu)*} - C_z] for arbitrary (non-exact) models."""
    P, d_sa, nu, _, w_z = mdp
    rng = make_rng(43, "tests", "fb", "resid")
    n_s, n_a, _ = P.shape
    d = 3
    zs = [rng.normal(size=d) for _ in w_z]
    F_tabs = [t64(rng.normal(size=(n_s, n_a, d))) for _ in w_z]
    Fm_tabs = [f + 0.2 * t64(rng.normal(size=(n_s, n_a, d))) for f in F_tabs]
    B_tab = t64(rng.normal(size=(n_s, d)))
    Bm_tab = B_tab + 0.2 * t64(rng.normal(size=(n_s, d)))
    L_impl = float(_population_loss(F_tabs, B_tab, Fm_tabs, Bm_tab, zs, P, d_sa, nu, w_z))
    # independent numpy computation of the residual form
    B, Bm = B_tab.numpy(), Bm_tab.numpy()
    total = 0.0
    for k, z in enumerate(zs):
        Fm = Fm_tabs[k].numpy()
        pim = softmax_rows(np.einsum("sad,d->sa", Fm, z) / TAU)
        Fbar = np.einsum("sa,sad->sd", pim, Fm)                           # (S, d) target features at every s1
        F = F_tabs[k].numpy()
        for s in range(n_s):
            for a in range(n_a):
                m_hat = B @ F[s, a]
                p_nu = P[s, a] / nu
                n_bar = GAMMA * P[s, a] @ (Fbar @ Bm.T)
                C1 = 0.5 * GAMMA**2 * sum(P[s, a, s1] * np.sum(nu * (Bm @ Fbar[s1]) ** 2) for s1 in range(n_s))
                Cz = 0.5 * np.sum(nu * (p_nu + n_bar) ** 2) - C1
                resid = 0.5 * np.sum(nu * (m_hat - p_nu - n_bar) ** 2)
                total += w_z[k] * d_sa[s, a] * (resid - Cz)
    assert abs(L_impl - total) < 1e-11


def test_minibatch_loss_unbiased_by_exact_enumeration():
    """Prop. 5.22(b): with b = b' = 2 i.i.d. draws, E[L_hat] = L exactly; target states from the own transition are biased."""
    rng = make_rng(44, "tests", "fb", "minibatch")
    n_s, n_a, d = 3, 2, 2
    P = random_mdp(rng, n_s, n_a)
    d_sa = rng.dirichlet(np.ones(n_s * n_a)).reshape(n_s, n_a)
    nu = np.einsum("sa,sat->t", d_sa, P)
    z = rng.normal(size=d)
    F = t64(rng.normal(size=(n_s, n_a, d)))
    Fm = F + 0.3 * t64(rng.normal(size=F.shape))
    B = t64(rng.normal(size=(n_s, d)))
    Bm = B + 0.3 * t64(rng.normal(size=B.shape))
    rows, probs = _transitions(P, d_sa, np.array([1.0]))
    Fbar_all = fb.expected_features(Fm, t64(z), TAU)                       # (S, d)
    L = float(_population_loss([F], B, [Fm], Bm, [z], P, d_sa, nu, np.array([1.0])))

    def lhat(tr, xs):
        s, a, s1 = tr[:, 0], tr[:, 1], tr[:, 2]
        return float(fb.successor_loss(F[s, a], B[xs], Fbar_all[s1], Bm[xs], B[s1], GAMMA))

    mean = 0.0
    for (i1, i2) in itertools.product(range(len(rows)), repeat=2):
        tr = rows[[i1, i2]]
        p_tr = probs[i1] * probs[i2]
        for (x1, x2) in itertools.product(range(n_s), repeat=2):
            mean += p_tr * nu[x1] * nu[x2] * lhat(tr, np.array([x1, x2]))
    assert abs(mean - L) < 1e-12
    # biased variant: the target state is the transition's own next state (b = b' = 1)
    biased = sum(p * lhat(rows[[i]], rows[[i], 2]) for i, p in enumerate(probs))
    assert abs(biased - L) > 1e-3
    # closed form with Gram matrices equals the double sum
    tr = rows[[0, 3, 5]]
    xs = np.array([0, 2, 2, 1])
    s, a, s1 = tr[:, 0], tr[:, 1], tr[:, 2]
    v1 = fb.successor_loss(F[s, a], B[xs], Fbar_all[s1], Bm[xs], B[s1], GAMMA)
    v2 = fb.successor_loss(F[s, a], B[xs], Fbar_all[s1], Bm[xs], B[s1], GAMMA, closed_form=True)
    assert abs(float(v1) - float(v2)) < 1e-12


def test_exact_softmax_representation_is_stationary(mdp):
    """Prop. 5.22(c): exact softmax FB with target = online is a stationary point of the semi-gradient of L."""
    P, d_sa, nu, zs, w_z = mdp
    n_s = P.shape[0]
    F_ex = [exact_softmax_fb(P, nu, z, TAU, GAMMA)[0] for z in zs]
    F_tabs = [t64(f).requires_grad_(True) for f in F_ex]
    B_tab = t64(np.eye(n_s)).requires_grad_(True)
    L = _population_loss(F_tabs, B_tab, [t64(f) for f in F_ex], t64(np.eye(n_s)), zs, P, d_sa, nu, w_z)
    grads = torch.autograd.grad(L, F_tabs + [B_tab])
    assert max(float(g.abs().max()) for g in grads) < 1e-9
    # a perturbed (non-exact) representation is not stationary
    rng = make_rng(45, "tests", "fb", "perturb")
    F_p = [t64(f + 0.05 * rng.normal(size=f.shape)).requires_grad_(True) for f in F_ex]
    L_p = _population_loss(F_p, B_tab, [f.detach().clone() for f in F_p], t64(np.eye(n_s)), zs, P, d_sa, nu, w_z)
    grads_p = torch.autograd.grad(L_p, F_p)
    assert max(float(g.abs().max()) for g in grads_p) > 1e-3


# ----------------------------------------------------------------------------- FB-9: orthonormality penalty
@pytest.mark.parametrize("param", ["table", "network"])
def test_orth_penalty_gradient_expectation(param):
    """Prop. 5.24: E grad L_reg = grad (1/4)||G_B - I||_F^2 (exact over nu x nu), for a table and for BackwardNet."""
    rng = make_rng(46, "tests", "fb", "orth", param)
    n_s, d = 5, 3
    nu = rng.dirichlet(np.ones(n_s))
    if param == "table":
        B_tab = t64(rng.normal(size=(n_s, d))).requires_grad_(True)
        params = [B_tab]
        feats = lambda: B_tab
    else:
        cfg = fb.FBConfig(d=d, hidden=8, dtype="float64")
        net = fb.BackwardNet(n_s, d, cfg, torch_generator(3), torch.float64)
        params = list(net.parameters())
        onehot = t64(np.eye(n_s))
        feats = lambda: net(onehot)
    Bx = feats()
    reg = fb.orth_penalty(Bx, Bx, t64(nu), t64(nu))                      # = E over independent x ~ nu, x' ~ nu
    g_reg = torch.autograd.grad(reg, params)
    Bx = feats()
    G = fb.gram_from_features(Bx, t64(nu))
    R = 0.25 * ((G - torch.eye(d, dtype=torch.float64)) ** 2).sum()
    g_R = torch.autograd.grad(R, params)
    for a, b in zip(g_reg, g_R):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=0, atol=1e-12)


def test_orth_penalty_uses_stop_gradients():
    """Without the stop-gradients the gradient would differ (the value of L_reg is not R)."""
    rng = make_rng(47, "tests", "fb", "stopgrad")
    B = t64(rng.normal(size=(4, 3))).requires_grad_(True)
    g1 = torch.autograd.grad(fb.orth_penalty(B, B), [B])[0]
    inner = B @ B.T
    naive = (inner**2).mean() - (B * B).sum(-1).mean()
    g2 = torch.autograd.grad(naive, [B])[0]
    assert float((g1 - g2).abs().max()) > 1e-3
    # with stop-gradients: d/dB_i = (1/b^2) sum_j (B_i . B_j) B_j - (1/b) B_i  (Prop. 5.24(b) with x = x')
    b = B.shape[0]
    want = (B @ B.T) @ B / b**2 - B / b
    np.testing.assert_allclose(g1.numpy(), want.detach().numpy(), atol=1e-12)


# ----------------------------------------------------------------------------- FB-10: Gram matrix of nu_hat
def test_gram_is_uncentered_exact_for_buffer_measure():
    rng = make_rng(48, "tests", "fb", "gram")
    spec = ObsSpec("finite", 6)
    cfg = fb.FBConfig(d=4, hidden=8)
    net = fb.BackwardNet(6, 4, cfg, torch_generator(5), torch.float32)
    next_obs = rng.integers(0, 6, size=1000)
    G = fb.gram(net, spec, next_obs, chunk=128)
    with torch.no_grad():
        B = net(torch.eye(6)).double().numpy()
    nu_hat = np.bincount(next_obs, minlength=6) / 1000
    np.testing.assert_allclose(G, B.T @ np.diag(nu_hat) @ B, atol=1e-6)
    np.testing.assert_allclose(G, G.T, atol=1e-12)
    assert np.linalg.eigvalsh(G).min() > -1e-10
    mean = nu_hat @ B
    centered = G - np.outer(mean, mean)
    assert np.abs(G - centered).max() > 1e-3                              # never mean-subtracted


# ----------------------------------------------------------------------------- targets, latents, networks
def test_polyak_update_formula():
    cfg = fb.FBConfig(d=2, hidden=4)
    a = fb.BackwardNet(3, 2, cfg, torch_generator(1), torch.float64)
    b = fb.BackwardNet(3, 2, cfg, torch_generator(2), torch.float64)
    old = [p.detach().clone() for p in a.parameters()]
    fb.polyak_update(a, b, 0.99)
    for p_new, p_old, p_on in zip(a.parameters(), old, b.parameters()):
        np.testing.assert_allclose(p_new.detach().numpy(), (0.99 * p_old + 0.01 * p_on.detach()).numpy(), atol=1e-15)


def test_latents_uniform_on_sphere():
    z = fb.sample_latents(20000, 16, 4.0, make_rng(1, "tests", "latents"))
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 4.0, atol=1e-12)
    assert np.abs(z.mean(axis=0)).max() < 5 * 1.0 / np.sqrt(20000)        # each coordinate has variance R^2/d = 1


def test_networks_shapes_dtype_and_c1_activations():
    spec = ObsSpec("continuous")
    cfg = fb.FBConfig()
    F_net, B_net = fb.networks(spec, 5, cfg, torch_generator(0))
    x = torch.rand(7, 2)
    z = torch.as_tensor(fb.sample_latents(7, 16, cfg.radius, make_rng(0, "z"))).float()
    assert F_net(x, z).shape == (7, 5, 16) and F_net(x, z).dtype == torch.float32
    assert B_net(x).shape == (7, 16)
    kinds = {type(m) for m in list(F_net.modules()) + list(B_net.modules())}
    assert nn.ReLU not in kinds and nn.SiLU in kinds and nn.LayerNorm in kinds and nn.Tanh in kinds
    with pytest.raises(ValueError):
        fb.FBConfig(activation="relu")                                     # U-14: C^1 only


def test_config_validation():
    with pytest.raises(ValueError):
        fb.FBConfig.from_dict({"d": 16, "rank": 16})
    with pytest.raises(ValueError):
        fb.FBConfig(tau=0.0)
    with pytest.raises(ValueError):
        fb.ReplayConfig(n_transitions=1001, segment_length=50)
    assert fb.FBConfig().radius == 4.0
    assert fb.FBConfig.from_dict(fb.FBConfig().to_dict()) == fb.FBConfig()


# ----------------------------------------------------------------------------- task-blind replay
def test_finite_replay_is_task_blind_and_covers_rooms_and_actions(four_rooms):
    buf = fb.collect_replay(four_rooms, fb.ReplayConfig(50_000, 50), make_rng(7, "tests", "replay"))
    assert set(buf.meta["fields"]) == {"obs", "actions", "next_obs", "segment", "step_in_segment"}
    assert not any("reward" in k or "goal" in k for k in buf.meta)          # no reward or goal anywhere
    assert buf.meta["n_resets"] == 1000 and len(buf) == 50_000
    starts = buf.obs[buf.step_in_segment == 0]
    assert starts.size == 1000 and np.unique(starts).size == 40             # resets are spread over every free cell
    cov = fb.replay_coverage(buf, 5, 7)
    assert cov["states_visited"] == four_rooms.n_states == 40
    assert cov["min_state_action_count"] > 0 and cov["min_next_state_count"] > 0
    # within a segment the next state is the following state (continuing environment, no masking)
    same = buf.segment[1:] == buf.segment[:-1]
    np.testing.assert_array_equal(buf.obs[1:][same], buf.next_obs[:-1][same])


def test_uniform_random_kernel_is_symmetric_so_replay_law_is_uniform(four_rooms, open_maze):
    """ENV-6(a): under uniform random actions the finite kernel is symmetric, so uniform resets keep the law uniform."""
    for maze in (four_rooms, open_maze):
        P = transition_matrix(maze)
        P_u = P.mean(axis=1)
        np.testing.assert_allclose(P_u, P_u.T, atol=1e-15)
        u = np.full(maze.n_states, 1.0 / maze.n_states)
        np.testing.assert_allclose(u @ P_u, u, atol=1e-15)


def test_continuous_replay_covers_cells_and_actions(cont_env):
    buf = fb.collect_replay(cont_env, fb.ReplayConfig(20_000, 50), make_rng(8, "tests", "replay", "cont"))
    assert cont_env.is_free(buf.obs).all() and cont_env.is_free(buf.next_obs).all()
    cov = fb.replay_coverage(buf, 5, 7)
    assert cov["cells_visited"] == 40 and cov["min_cell_action_count"] > 0
    path = buf.sha256()
    assert len(path) == 64


def test_replay_save_load_roundtrip(tmp_path, four_rooms):
    buf = fb.collect_replay(four_rooms, fb.ReplayConfig(500, 50), make_rng(9, "tests", "replay", "io"))
    p = tmp_path / "replay.npz"
    buf.save(p)
    back = fb.ReplayBuffer.load(p)
    assert back.sha256() == buf.sha256()
    np.testing.assert_array_equal(back.next_obs, buf.next_obs)


# ----------------------------------------------------------------------------- pretraining target uses the same family
def test_pretraining_target_uses_policy_family_of_the_model(four_rooms):
    """At initialization the target copy equals the online copy, so the pretraining target features equal the
    model's (evaluation) expected features: one policy family pi_{z,tau} for targets, ER, DM and evaluation."""
    buf = fb.collect_replay(four_rooms, fb.ReplayConfig(1000, 50), make_rng(10, "tests", "replay", "family"))
    cfg = fb.FBConfig(n_updates=1)
    tr = fb.FBPretrainer(ObsSpec.from_env(four_rooms), 5, 0.95, cfg, seed=3)
    tr.attach_buffer(buf)
    model = tr.model()
    z = torch.as_tensor(fb.sample_latents(10, 16, cfg.radius, make_rng(1, "z"))).float()
    s1 = buf.next_obs[:10]
    with torch.no_grad():
        target = fb.expected_features(tr.F_target(tr._enc(torch.as_tensor(s1)), z), z, cfg.tau)
        online = torch.stack([model.expected_features(s1[i:i + 1], z[i])[0] for i in range(10)])
        probs = torch.stack([model.policy_probs(s1[i:i + 1], z[i])[0] for i in range(10)])
    np.testing.assert_allclose(target.numpy(), online.numpy(), atol=1e-6)
    pol = np.stack([model.numpy_policy(z[i])(s1[i:i + 1])[0] for i in range(10)])
    np.testing.assert_allclose(pol, probs.double().numpy() / probs.double().numpy().sum(-1, keepdims=True), atol=1e-7)
