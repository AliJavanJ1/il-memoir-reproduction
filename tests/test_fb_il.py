"""ER_FB and DM_FB with a frozen FB model (METHOD_MAP sections 5-6; theory section 6).

Exact checks use explicit finite factorizations (one-hot ``B``, exact softmax
``F``) and the implementation's ``fb_il.dm_loss``; gradient checks use float64
copies of real ``FBModel`` networks.  Freeze checks verify that latent
adaptation leaves every network parameter and ``G_hat_B`` unchanged.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from il_memoir import fb, fb_il, oracle
from il_memoir.common import make_rng
from il_memoir.density import discount_weights
from il_memoir.env_tabular import initial_distribution, transition_matrix
from il_memoir.nets import ObsSpec, params_fingerprint

from _helpers import random_mdp, random_policy
from _learner_helpers import central_fd, exact_softmax_fb


def t64(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


def _random_model(spec: ObsSpec, seed: int, dtype: str = "float32", tau: float = 0.5) -> fb.FBModel:
    """An FBModel from freshly initialized networks and a Gram matrix computed on random states (no training)."""
    cfg = fb.FBConfig(dtype=dtype, tau=tau)
    trainer = fb.FBPretrainer(spec, 5, 0.95, cfg, seed=seed)
    rng = make_rng(seed, "tests", "random-model")
    if spec.finite:
        obs = rng.integers(0, spec.n_states, size=400)
    else:
        obs = 0.05 + 0.9 * rng.random((400, 2))
    buf = fb.ReplayBuffer(obs, rng.integers(0, 5, size=400), obs.copy(), np.zeros(400, np.int64), np.arange(400), {})
    trainer.attach_buffer(buf)
    return trainer.model()


@pytest.fixture(scope="module")
def demos(four_rooms):
    rng = make_rng(51, "tests", "fbil", "demos")
    pi = random_policy(rng, four_rooms.n_states, four_rooms.n_actions)
    return four_rooms.rollout(pi, 4, 100, rng).states


@pytest.fixture(scope="module")
def model_f(four_rooms):
    return _random_model(ObsSpec.from_env(four_rooms), 1)


# ----------------------------------------------------------------------------- ER_FB
def test_er_weighting_uniform_over_s1_to_sH(model_f, demos):
    z = fb_il.er_fb_z(model_f, demos)
    with torch.no_grad():
        B = model_f.backward_features(np.arange(40)).double().numpy()
    want = np.mean([B[demos[i, 1:]].mean(axis=0) for i in range(demos.shape[0])], axis=0)
    np.testing.assert_allclose(z, want, atol=1e-6)
    with_s0 = np.mean([B[demos[i, :]].mean(axis=0) for i in range(demos.shape[0])], axis=0)
    assert np.abs(z - with_s0).max() > 1e-4                                  # s_0 is excluded (FI24 footnote 3)
    w = discount_weights(4, 100, 0.95)                                     # the gamma-weighted f-PG target law
    q_based = (w[..., None] * B[demos[:, 1:]]).sum(axis=(0, 1))
    assert np.abs(z - q_based).max() > 1e-4                                # rho_e != q_hat_E (FBIL-1(e))


def test_projection_norm_and_fallback(model_f, demos):
    er = fb_il.er_fb(model_f, demos)
    assert abs(np.linalg.norm(er.z) - model_f.radius) < 1e-9 and not er.fallback
    assert abs(er.norm_raw - np.linalg.norm(er.z_raw)) < 1e-12
    z, info = fb_il.project_to_sphere(np.zeros(16), 4.0, make_rng(3, "tests", "fallback"))
    assert info["fallback"] and abs(np.linalg.norm(z) - 4.0) < 1e-12
    z2, _ = fb_il.project_to_sphere(np.zeros(16), 4.0, make_rng(3, "tests", "fallback"))
    np.testing.assert_array_equal(z, z2)                                   # the fallback is reproducible from its seed
    with pytest.raises(ValueError):
        fb_il.project_to_sphere(np.zeros(16), 4.0)


# ----------------------------------------------------------------------------- DM targets
def test_dm_start_times_and_target_definition(model_f, demos):
    assert fb_il.dm_start_times(100, 60).tolist() == list(range(41))       # 41 start times, t = 0 included (D4-D5)
    assert fb_il.dm_start_times(100, 60, include_t0=False).tolist() == list(range(1, 41))
    data = fb_il.dm_targets(model_f, demos, 60)
    assert data.psi.shape == (4 * 41, 16)
    with torch.no_grad():
        B = model_f.backward_features(np.arange(40)).double().numpy()
    g = 0.95 ** np.arange(60)
    for row in (0, 17, 40, 41, 163):
        i, t = data.demo_index[row], data.time_index[row]
        want = (g[:, None] * B[demos[i, t + 1:t + 61]]).sum(axis=0)
        np.testing.assert_allclose(data.psi[row].double().numpy(), want, rtol=1e-5, atol=1e-5)
        assert data.start_obs[row] == demos[i, t]


def test_changing_a_future_expert_state_changes_earlier_targets(model_f, demos):
    """Prop. 6.8(e): psi_t depends on s_{t+1..t+K}; changing slot j changes psi_t for j-K <= t <= j-1 by gamma^{j-t-1} dB."""
    K, j = 5, 30
    base = fb_il.dm_targets(model_f, demos, K)
    changed = demos.copy()
    old, new = changed[2, j], (changed[2, j] + 7) % 40
    changed[2, j] = new
    alt = fb_il.dm_targets(model_f, changed, K)
    with torch.no_grad():
        dB = (model_f.backward_features(np.array([new])) - model_f.backward_features(np.array([old])))[0].double().numpy()
    diff = (alt.psi - base.psi).double().numpy()
    for row in range(diff.shape[0]):
        i, t = base.demo_index[row], base.time_index[row]
        if i == 2 and j - K <= t <= j - 1:
            np.testing.assert_allclose(diff[row], 0.95 ** (j - t - 1) * dB, atol=1e-5)
        else:
            assert np.abs(diff[row]).max() < 1e-12
    assert np.abs(diff).max() > 1e-3


def test_truncation_bound(four_rooms, model_f):
    """Prop. 6.8(b): ||psi^(inf) - psi^(K)|| <= gamma^K beta_B / (1 - gamma) (psi^(inf) approximated with 400 steps)."""
    rng = make_rng(52, "tests", "fbil", "long")
    pi = random_policy(rng, 40, 5)
    long_demo = four_rooms.rollout(pi, 1, 500, rng).states
    short = fb_il.dm_targets(model_f, long_demo, 60)
    full = fb_il.dm_targets(model_f, long_demo, 440)
    with torch.no_grad():
        beta = float(model_f.backward_features(np.arange(40)).double().norm(dim=1).max())
    gap = (short.psi[:61] - full.psi[:61]).double().norm(dim=1).max().item()
    assert gap <= 0.95**60 * beta / 0.05 + 1e-5
    assert gap > 0


# ----------------------------------------------------------------------------- DM loss: exact expansion
def test_dm_loss_expansion_up_to_constant_exact_factorization():
    """FBIL-2(a): under exact softmax FB, L_DM(z) - l_DM(z) = C_E for every z (computed with fb_il.dm_loss)."""
    rng = make_rng(53, "tests", "fbil", "expansion")
    n_s, n_a, gamma, tau = 4, 3, 0.9, 0.5
    P = random_mdp(rng, n_s, n_a)
    nu = rng.dirichlet(np.ones(n_s)) * 0.6 + 0.4 / n_s
    pi_E = random_policy(rng, n_s, n_a)                                  # stationary expert (A4)
    M_E = oracle.successor_next_state_avg(P, pi_E, gamma)                # (S, S)
    B = np.eye(n_s)
    psi_bar = M_E @ B                                                     # exact expert successor features
    G = np.diag(nu)                                                       # Gram of one-hot B under nu
    rho = rng.dirichlet(np.ones(n_s))                                     # start law
    C_E = float(np.sum(rho[:, None] * M_E**2 / nu[None, :]))
    gaps = []
    for _ in range(3):
        z = rng.normal(size=n_s)
        F, pi_z, res = exact_softmax_fb(P, nu, z, tau, gamma)
        assert res < 1e-11
        ell = float(fb_il.dm_loss(t64(F), t64(z), t64(psi_bar), t64(G), tau, weights=t64(rho)))
        M_z = oracle.successor_next_state_avg(P, pi_z, gamma)
        L_DM = float(np.sum(rho[:, None] * (M_z - M_E) ** 2 / nu[None, :]))   # L2(nu)* norm of the difference
        gaps.append(L_DM - ell)
    np.testing.assert_allclose(gaps, C_E, rtol=0, atol=1e-9)


# ----------------------------------------------------------------------------- DM gradient into z (frozen nets)
def _detached_policy_objective(model, z, data):
    F_sa = model.F_net(data.start_enc, z)
    probs = fb.policy_probs(F_sa, z, model.tau).detach()                 # policy treated as constant (FBIL-4(e))
    Fz = (probs.unsqueeze(-1) * F_sa).sum(-2)
    return (((Fz @ model.gram) * Fz).sum(-1) - 2 * (Fz * data.psi).sum(-1)).mean()


@pytest.mark.parametrize("kind", ["finite", "continuous"])
def test_dm_gradient_matches_finite_differences_and_formula(kind, four_rooms, cont_env, demos):
    if kind == "finite":
        model = _random_model(ObsSpec.from_env(four_rooms), 2, dtype="float64")
        states = demos
    else:
        model = _random_model(ObsSpec.from_env(cont_env), 3, dtype="float64")
        rng = make_rng(54, "tests", "fbil", "cont-demos")
        states = cont_env.rollout(lambda x: np.full((x.shape[0], 5), 0.2), 2, 100, rng).states
    data = fb_il.dm_targets(model, states, 60)
    z0 = fb.sample_latents(1, 16, model.radius, make_rng(55, "tests", kind))[0]
    g = fb_il.dm_grad(model, z0, data)
    fd = central_fd(lambda z: float(fb_il.dm_objective(model, t64(z), data)), z0, h=1e-6)
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-7)
    # Eq. (6.6) with explicit Jacobians J(s, z) = d F(s, z) / dz
    zt = t64(z0)
    total = np.zeros(16)
    n = data.psi.shape[0]
    for row in range(n):
        enc = data.start_enc[row:row + 1]
        fz = lambda zz: fb.expected_features(model.F_net(enc, zz), zz, model.tau)[0]
        J = torch.autograd.functional.jacobian(fz, zt).numpy()             # (d, d)
        Fv = fz(zt).detach().numpy()
        total += J.T @ (model.gram.numpy() @ Fv - data.psi[row].numpy())
    np.testing.assert_allclose(g, 2.0 * total / n, rtol=1e-9, atol=1e-9)
    # treating the policy as constant gives a different vector (FBIL-4(e))
    zg = t64(z0).requires_grad_(True)
    g_det = torch.autograd.grad(_detached_policy_objective(model, zg, data), [zg])[0].numpy()
    assert np.abs(g_det - g).max() > 1e-6 * max(1.0, np.abs(g).max())
    # Riemannian gradient is tangent to the sphere
    gR = fb_il.riemannian_gradient(z0, g, model.radius)
    assert abs(gR @ z0) < 1e-9 * max(1.0, np.abs(g).max())


def test_dm_fit_freezes_the_model_and_keeps_z_on_sphere(model_f, demos):
    before = params_fingerprint(model_f)
    gram_before = model_f.gram.clone()
    res = fb_il.dm_fit(model_f, demos, fb_il.DMConfig(steps=25))
    assert torch.equal(before, params_fingerprint(model_f)) and torch.equal(gram_before, model_f.gram)
    assert all(p.grad is None and not p.requires_grad for p in model_f.parameters())
    assert len(res.log) == 26 and all(np.isfinite(list(e.values())).all() for e in res.log)
    assert abs(np.linalg.norm(res.z) - model_f.radius) < 1e-4
    assert abs(np.linalg.norm(res.z0) - model_f.radius) < 1e-9
    data = fb_il.dm_targets(model_f, demos, 60)
    with torch.no_grad():
        first = float(fb_il.dm_objective(model_f, model_f.latent(res.z0), data))
        last = float(fb_il.dm_objective(model_f, model_f.latent(res.z), data))
    assert abs(first - res.log[0]["loss"]) < 1e-4 * max(1.0, abs(first))
    assert abs(last - res.log[-1]["loss"]) < 1e-4 * max(1.0, abs(last))
    assert np.abs(res.z - res.z0).max() > 1e-4                              # z moved; the networks did not


def test_dm_refuses_an_unfrozen_model(four_rooms, demos):
    model = _random_model(ObsSpec.from_env(four_rooms), 4)
    for p in model.F_net.parameters():
        p.requires_grad_(True)
    with pytest.raises(RuntimeError):
        fb_il.dm_fit(model, demos, fb_il.DMConfig(steps=2))


def test_dm_config_limits():
    with pytest.raises(ValueError):
        fb_il.DMConfig(steps=301)                                          # at most 300 latent updates (spec section 6)
    with pytest.raises(ValueError):
        fb_il.DMConfig.from_dict({"K": 60, "horizon": 100})
    assert fb_il.DMConfig.from_dict(fb_il.DMConfig().to_dict()) == fb_il.DMConfig()


# ----------------------------------------------------------------------------- one policy family everywhere
def test_policy_consistency_between_dm_er_and_evaluation(model_f, demos):
    res = fb_il.dm_fit(model_f, demos, fb_il.DMConfig(steps=5))
    data = fb_il.dm_targets(model_f, demos, 60)
    z = model_f.latent(res.z)
    with torch.no_grad():
        F_sa = model_f.F_net(data.start_enc, z)
        pi_dm = fb.policy_probs(F_sa, z, model_f.tau).double().numpy()     # the policy inside the DM loss
    pi_eval = model_f.numpy_policy(res.z)(data.start_obs)                  # the policy used by rollouts/evaluation
    np.testing.assert_allclose(pi_eval, pi_dm / pi_dm.sum(-1, keepdims=True), atol=1e-7)
    matrix = model_f.policy_matrix(res.z)
    np.testing.assert_allclose(matrix[data.start_obs], pi_eval, atol=1e-12)
    er = fb_il.er_fb(model_f, demos)
    np.testing.assert_allclose(model_f.policy_matrix(er.z), model_f.numpy_policy(er.z)(np.arange(40)), atol=1e-12)


def test_fb_il_has_no_environment_handle():
    for name in ("er_fb_z", "er_fb", "dm_targets", "dm_loss", "dm_objective", "dm_grad", "dm_fit"):
        params = inspect.signature(getattr(fb_il, name)).parameters
        assert not any(p in params for p in ("env", "maze", "simulator")), name


def test_model_save_load_roundtrip(tmp_path, model_f, demos):
    path = tmp_path / "fb_model.pt"
    model_f.save(path)
    back = fb.FBModel.load(path)
    z = fb_il.er_fb(model_f, demos).z
    np.testing.assert_array_equal(back.policy_matrix(z), model_f.policy_matrix(z))
    np.testing.assert_array_equal(back.gram64, model_f.gram64)
    assert all(not p.requires_grad for p in back.parameters())
    np.testing.assert_array_equal(fb_il.er_fb_z(back, demos), fb_il.er_fb_z(model_f, demos))
