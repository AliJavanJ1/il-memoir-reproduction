"""Integration checks of the three learners on both environments (INTERFACE.md: finite losses, reload, freeze).

Real simulator rollouts, real demonstrations and short real training loops:
every metric must be finite; policies and models must reload to identical
outputs; an interrupted run resumed from ``state_dict`` must continue exactly
as an uninterrupted run (Adam state and random streams restored).  Tests run on
the CPU and, when ``torch.cuda.is_available()``, also on CUDA (native laptop
runs); CPU and CUDA must agree on the same update to float32 precision.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import torch

from il_memoir import fb, fb_il, fpg
from il_memoir.common import make_rng
from il_memoir.env_tabular import make_layout
from il_memoir.expert import finite_expert, generate_demonstrations, route_controller
from il_memoir.nets import ObsSpec

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available (run natively on the laptop)")


def roundtrip(obj):
    """Serialize with torch.save and reload with weights_only=True (the checkpoint path of the runners)."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=True)


def assert_same_params(m1, m2, device):
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        if device == "cpu":
            assert torch.equal(p1, p2), n1
        else:
            torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-6)


@pytest.fixture(scope="module")
def view(cfg):
    return cfg.learner_dict()


@pytest.fixture(scope="module")
def finite_demos(cfg, four_rooms, tasks):
    expert = finite_expert(four_rooms, tasks["goal_5_5"], cfg.gamma, cfg.horizon)
    return generate_demonstrations(four_rooms, expert.pi, 4, cfg.horizon, (7, ("tests", "integration", "demos"))).learner_states()


@pytest.fixture(scope="module")
def cont_demos(cfg, cont_env):
    ctrl = route_controller(make_layout(cfg, cfg.continuous_layout), cfg.goal_cell("goal_5_5"), cfg.action_deltas)
    return generate_demonstrations(cont_env, ctrl, 4, cfg.horizon, (7, ("tests", "integration", "cdemos"))).learner_states()


# ----------------------------------------------------------------------------- f-PG
@pytest.mark.parametrize("device", DEVICES)
def test_fpg_finite_loop_finite_metrics_resume_and_reload(device, four_rooms, view, finite_demos, tmp_path):
    cfg = fpg.FPGConfig(n_traj=10, lr=0.1)
    a = fpg.FPGLearner(four_rooms, view, finite_demos, cfg, seed=0, device=device)
    logs = [a.step(four_rooms) for _ in range(4)]
    assert all(np.isfinite(v) for m in logs for v in m.values())
    assert logs[-1]["env_steps"] == 4 * 10 * 100 and a.n_updates == 4
    b = fpg.FPGLearner(four_rooms, view, finite_demos, cfg, seed=0, device=device)
    for _ in range(2):
        b.step(four_rooms)
    c = fpg.FPGLearner(four_rooms, view, finite_demos, cfg, seed=0, device=device)
    c.load_state_dict(roundtrip(b.state_dict()))
    for _ in range(2):
        c.step(four_rooms)
    assert_same_params(a.policy, c.policy, device)
    assert c.env_steps == a.env_steps and c.iteration == a.iteration
    # the rollout policy, the checkpoint and the evaluation callable are one policy
    path = tmp_path / "fpg_policy.pt"
    fpg.save_policy(a.policy, path)
    back = fpg.load_policy(path, device)
    states = np.arange(four_rooms.n_states)
    np.testing.assert_array_equal(fpg.policy_callable(back)(states), a.policy_callable()(states))
    with torch.no_grad():
        direct = torch.softmax(a.policy.phi, dim=-1).double().cpu().numpy()
    np.testing.assert_allclose(a.policy_callable()(states), direct / direct.sum(-1, keepdims=True), atol=1e-12)


@pytest.mark.parametrize("density", ["kde", "histogram"])
def test_fpg_continuous_loop_finite_metrics_and_reload(density, cont_env, view, cont_demos, tmp_path):
    cfg = fpg.FPGConfig(n_traj=6, n_folds=2, lr=1e-3, density=density)
    learner = fpg.FPGLearner(cont_env, view, cont_demos, cfg, seed=1)
    logs = [learner.step(cont_env) for _ in range(2)]
    assert all(np.isfinite(v) for m in logs for v in m.values())
    assert 0.0 <= logs[-1]["JS_plugin"] <= np.log(2.0)
    path = tmp_path / f"fpg_{density}.pt"
    fpg.save_policy(learner.policy, path)
    back = fpg.load_policy(path)
    pts = cont_env.sample_free_uniform(50, make_rng(2, "tests", "pts"))
    np.testing.assert_array_equal(fpg.policy_callable(back)(pts), learner.policy_callable()(pts))


def test_fpg_ppo_option_and_entropy_logging(four_rooms, view, finite_demos):
    cfg = fpg.FPGConfig(n_traj=6, ppo_epochs=3, entropy_coef=0.0)
    learner = fpg.FPGLearner(four_rooms, view, finite_demos, cfg, seed=2)
    m = learner.step(four_rooms)
    assert learner.n_updates == 3 and np.isfinite(m["policy_entropy"]) and np.isfinite(m["grad_norm"])


def test_fpg_rejects_demonstrations_of_another_horizon(four_rooms, view, finite_demos):
    with pytest.raises(ValueError):
        fpg.FPGLearner(four_rooms, view, finite_demos[:, :50], fpg.FPGConfig(), seed=0)


# ----------------------------------------------------------------------------- FB pretraining
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["finite", "continuous"])
def test_fb_pretraining_finite_losses_resume_and_reload(kind, device, four_rooms, cont_env, view, tmp_path):
    env = four_rooms if kind == "finite" else cont_env
    buf = fb.collect_replay(env, fb.ReplayConfig(2000, 50), make_rng(3, "tests", "integration", kind))
    spec = ObsSpec.from_env(env)
    cfg = fb.FBConfig(n_updates=20, batch_size=32, target_batch_size=32)
    a = fb.FBPretrainer(spec, 5, view["gamma"], cfg, seed=4, device=device)
    a.attach_buffer(buf)
    out = a.train(20, log_every=5)
    assert all(np.isfinite(v) for m in out["logs"] for v in m.values())
    b = fb.FBPretrainer(spec, 5, view["gamma"], cfg, seed=4, device=device)
    b.attach_buffer(buf)
    b.train(10)
    c = fb.FBPretrainer(spec, 5, view["gamma"], cfg, seed=4, device=device)
    c.attach_buffer(buf)
    c.load_state_dict(roundtrip(b.state_dict()))
    c.train(10)
    for m1, m2 in ((a.F, c.F), (a.B, c.B), (a.F_target, c.F_target), (a.B_target, c.B_target)):
        assert_same_params(m1, m2, device)
    model = a.model()
    assert np.all(np.isfinite(model.gram64)) and np.allclose(model.gram64, model.gram64.T)
    diag = fb.diagnostics(model, buf, make_rng(5, "tests", "diag"))
    assert all(np.isfinite(v) for v in diag.values())
    path = tmp_path / f"fb_{kind}.pt"
    model.save(path)
    back = fb.FBModel.load(path, device)
    z = fb.sample_latents(1, 16, model.radius, make_rng(6, "tests", "z"))[0]
    obs = buf.obs[:64]
    np.testing.assert_array_equal(back.numpy_policy(z)(obs), model.numpy_policy(z)(obs))


def test_fb_resume_rejects_another_buffer(four_rooms, view):
    spec = ObsSpec.from_env(four_rooms)
    cfg = fb.FBConfig(batch_size=16, target_batch_size=16)
    buf1 = fb.collect_replay(four_rooms, fb.ReplayConfig(500, 50), make_rng(1, "b1"))
    buf2 = fb.collect_replay(four_rooms, fb.ReplayConfig(500, 50), make_rng(2, "b2"))
    a = fb.FBPretrainer(spec, 5, view["gamma"], cfg, seed=0)
    a.attach_buffer(buf1)
    a.train(2)
    b = fb.FBPretrainer(spec, 5, view["gamma"], cfg, seed=0)
    b.attach_buffer(buf2)
    with pytest.raises(ValueError):
        b.load_state_dict(roundtrip(a.state_dict()))


# ----------------------------------------------------------------------------- ER_FB and DM_FB after pretraining
@pytest.mark.parametrize("kind", ["finite", "continuous"])
def test_er_and_dm_after_pretraining(kind, four_rooms, cont_env, view, finite_demos, cont_demos):
    env, demos = (four_rooms, finite_demos) if kind == "finite" else (cont_env, cont_demos)
    model, buf, _ = fb.pretrain(env, view, fb.ReplayConfig(2000, 50), fb.FBConfig(n_updates=50, batch_size=32, target_batch_size=32), seed=8)
    er = fb_il.er_fb(model, demos)
    assert np.isfinite(er.norm_raw) and abs(np.linalg.norm(er.z) - model.radius) < 1e-9
    res = fb_il.dm_fit(model, demos, fb_il.DMConfig(steps=20))
    assert all(np.isfinite(v) for e in res.log for v in e.values())
    probs = model.numpy_policy(res.z)(demos[:, 0])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    rollout = env.rollout(model.numpy_policy(res.z), 3, 20, make_rng(9, "tests", "rollout", kind))
    assert rollout.states.shape[:2] == (3, 21)


# ----------------------------------------------------------------------------- CPU versus CUDA (native runs)
@needs_cuda
def test_cpu_and_cuda_agree_on_one_fb_update(four_rooms, view):
    buf = fb.collect_replay(four_rooms, fb.ReplayConfig(1000, 50), make_rng(10, "tests", "cuda"))
    spec = ObsSpec.from_env(four_rooms)
    cfg = fb.FBConfig(batch_size=64, target_batch_size=64)
    runs = {}
    for device in ("cpu", "cuda"):
        tr = fb.FBPretrainer(spec, 5, view["gamma"], cfg, seed=11, device=device)
        tr.attach_buffer(buf)
        runs[device] = (tr.update(), tr)
    m_cpu, m_gpu = runs["cpu"][0], runs["cuda"][0]
    assert abs(m_cpu["loss"] - m_gpu["loss"]) <= 1e-4 * max(1.0, abs(m_cpu["loss"]))
    for p1, p2 in zip(runs["cpu"][1].F.parameters(), runs["cuda"][1].F.parameters()):
        torch.testing.assert_close(p1, p2.cpu(), rtol=1e-4, atol=1e-5)


@needs_cuda
def test_cpu_and_cuda_agree_on_the_dm_gradient(four_rooms, view, finite_demos):
    model, _, _ = fb.pretrain(four_rooms, view, fb.ReplayConfig(1000, 50), fb.FBConfig(n_updates=20, batch_size=32, target_batch_size=32), seed=12)
    data_cpu = fb_il.dm_targets(model, finite_demos, 60)
    z = fb_il.er_fb(model, finite_demos).z
    g_cpu = fb_il.dm_grad(model, z, data_cpu)
    model_gpu = model.to("cuda")
    g_gpu = fb_il.dm_grad(model_gpu, z, fb_il.dm_targets(model_gpu, finite_demos, 60))
    np.testing.assert_allclose(g_gpu, g_cpu, rtol=1e-3, atol=1e-3 * max(1.0, np.abs(g_cpu).max()))


# ----------------------------------------------------------------------------- the learners' update wiring
def test_fpg_learner_step_is_a_descent_step_on_the_crossfit_estimator(four_rooms, view, finite_demos):
    """The first Adam step moves phi by -lr * g / (|g| + eps), with g the cross-fitted estimator of its own batch."""
    import copy

    cfg = fpg.FPGConfig(n_traj=10, lr=0.05)
    learner = fpg.FPGLearner(four_rooms, view, finite_demos, cfg, seed=5, device="cpu")
    phi0 = learner.policy.phi.detach().clone().double()
    policy0 = copy.deepcopy(learner.policy)
    rng_copy = copy.deepcopy(learner.rng)
    learner.step(four_rooms)
    traj = fpg.collect(four_rooms, policy0, cfg.n_traj, 100, rng_copy)          # the same batch
    cf = learner.advantages(traj)
    ref = fpg.crossfit_advantages(learner.estimator, traj.states[:, 1:], cfg.n_folds, cfg.baseline)
    np.testing.assert_array_equal(cf.advantages, ref.advantages)
    g = fpg.gradient_estimate(policy0, traj.states, traj.actions, cf.signal, cf.baseline, 0.95).reshape(phi0.shape)
    used = learner.optimizer.state[learner.policy.phi]["exp_avg"].double().numpy() / (1.0 - 0.9)
    np.testing.assert_allclose(used, g, rtol=1e-4, atol=1e-8)         # the gradient Adam received
    step = (learner.policy.phi.detach().double() - phi0).numpy()
    np.testing.assert_allclose(step, -cfg.lr * g / (np.abs(g) + cfg.adam_eps), atol=2e-6)
    assert np.abs(step).max() > 0.9 * cfg.lr


def test_fb_update_computes_eq_5_6_on_an_independent_target_batch(four_rooms, view):
    """Recompute the losses of one update from the pre-update networks and the replayed random draws."""
    import copy

    buf = fb.collect_replay(four_rooms, fb.ReplayConfig(2000, 50), make_rng(21, "tests", "update"))
    cfg = fb.FBConfig(batch_size=32, target_batch_size=24)
    tr = fb.FBPretrainer(ObsSpec.from_env(four_rooms), 5, view["gamma"], cfg, seed=6)
    tr.attach_buffer(buf)
    tr.train(3)                                                              # targets now differ from the online copy
    nets = {k: copy.deepcopy(getattr(tr, k)) for k in ("F", "B", "F_target", "B_target")}
    assert max(float((a - b).detach().abs().max()) for a, b in zip(nets["F"].parameters(), nets["F_target"].parameters())) > 0
    rng_copy = copy.deepcopy(tr.rng)
    m = tr.update()
    i = rng_copy.integers(0, len(buf), size=32)
    j = rng_copy.integers(0, len(buf), size=24)
    z = torch.as_tensor(fb.sample_latents(32, cfg.d, cfg.radius, rng_copy)).float()
    assert not np.array_equal(i[:24], j)
    enc = lambda idx: torch.nn.functional.one_hot(torch.as_tensor(idx), 40).float()
    with torch.no_grad():
        F_i = nets["F"](enc(buf.obs[i]), z)[torch.arange(32), torch.as_tensor(buf.actions[i])]
        B_x = nets["B"](enc(buf.next_obs[j]))
        B_next = nets["B"](enc(buf.next_obs[i]))
        Fbar = fb.expected_features(nets["F_target"](enc(buf.next_obs[i]), z), z, cfg.tau)
        Bm_x = nets["B_target"](enc(buf.next_obs[j]))
        L = fb.successor_loss(F_i, B_x, Fbar, Bm_x, B_next, view["gamma"])
        R = fb.orth_penalty(B_next, B_x)
        # logged score diagnostics (U-2), recomputed by hand: transition k is scored with its own latent z_k
        F_all = nets["F"](enc(buf.obs[i]), z)                                   # (b, A, d)
        scores = torch.einsum("kad,kd->ka", F_all, z)                          # F(s_k, a, z_k)^T z_k
        p_on = torch.softmax(scores / cfg.tau, dim=-1)
        diag = {
            "online_policy_entropy": float(-(p_on * torch.log(p_on.clamp_min(1e-30))).sum(-1).mean()),
            "score_spread_over_tau": float(((scores.max(-1).values - scores.min(-1).values) / cfg.tau).mean()),
            "mean_abs_score": float(torch.einsum("kd,kd->k", F_i, z).abs().mean()),   # taken actions a_k only
        }
    assert abs(float(L) - m["loss"]) <= 1e-5 * max(1.0, abs(float(L)))
    assert abs(float(R) - m["orth_loss"]) <= 1e-5 * max(1.0, abs(float(R)))
    for key, want in diag.items():
        assert abs(m[key] - want) <= 1e-5 * max(1.0, abs(want)), key


def _replay_fpg_batch(learner, env):
    """The batch that learner.step(env) will collect, reproduced from copies of its policy and random stream."""
    import copy

    policy0 = copy.deepcopy(learner.policy)
    rng_copy = copy.deepcopy(learner.rng)
    traj = fpg.collect(env, policy0, learner.cfg.n_traj, learner.horizon, rng_copy)
    return policy0, traj


def test_fpg_ppo_branch_matches_an_independent_reimplementation(four_rooms, view, finite_demos):
    """PPO epochs: c_k = gamma^k (G_k - beta_k) / Z_H and log pi_{phi'} are fixed at the sampling parameter."""
    cfg = fpg.FPGConfig(n_traj=8, lr=0.05, ppo_epochs=3, ppo_clip=0.2)
    learner = fpg.FPGLearner(four_rooms, view, finite_demos, cfg, seed=13)
    policy0, traj = _replay_fpg_batch(learner, four_rooms)
    learner.step(four_rooms)
    adv = learner.advantages(traj).advantages
    with torch.no_grad():
        logp_old = fpg.log_probs(policy0, traj.states[:, :-1], traj.actions)
    c = fpg.score_weights(100, 0.95)[None, :] * adv
    opt = torch.optim.Adam(policy0.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    for epoch in range(3):
        loss = fpg.ppo_surrogate(policy0, traj.states, traj.actions, logp_old, c, cfg.ppo_clip)
        if epoch == 0:                                          # at phi' the PPO gradient is the f-PG estimator
            g0 = torch.autograd.grad(loss, list(policy0.parameters()), retain_graph=True)[0].double().numpy()
            est = fpg.gradient_estimate(policy0, traj.states, traj.actions, learner.advantages(traj).signal,
                                        learner.advantages(traj).baseline, 0.95)
            np.testing.assert_allclose(g0.reshape(-1), est, rtol=1e-5, atol=1e-7)
        opt.zero_grad()
        loss.backward()
        opt.step()
    np.testing.assert_allclose(learner.policy.phi.detach().numpy(), policy0.phi.detach().numpy(), rtol=0, atol=1e-6)


def test_fpg_entropy_regularizer_has_the_minimization_sign(four_rooms, view, finite_demos):
    """With entropy_coef > 0 the minimized loss is surrogate - coef * H(pi): the first Adam step follows its gradient."""
    cfg = fpg.FPGConfig(n_traj=8, lr=0.05, entropy_coef=0.5)
    learner = fpg.FPGLearner(four_rooms, view, finite_demos, cfg, seed=14)
    with torch.no_grad():
        learner.policy.phi.copy_(torch.as_tensor(make_rng(1, "phi").normal(size=(40, 5)) * 0.5))
    policy0, traj = _replay_fpg_batch(learner, four_rooms)
    learner.step(four_rooms)
    cf = learner.advantages(traj)
    loss = fpg.surrogate_loss(policy0, traj.states, traj.actions, cf.advantages, 0.95) \
        - 0.5 * fpg.mean_entropy(policy0, traj.states[:, :-1])
    g = torch.autograd.grad(loss, [policy0.phi])[0].double().numpy()
    used = learner.optimizer.state[learner.policy.phi]["exp_avg"].double().numpy() / (1.0 - 0.9)
    np.testing.assert_allclose(used, g, rtol=1e-4, atol=1e-8)         # the gradient Adam received
    step = (learner.policy.phi.detach().double() - policy0.phi.detach().double()).numpy()
    np.testing.assert_allclose(step, -cfg.lr * g / (np.abs(g) + cfg.adam_eps), atol=2e-6)


def test_fpg_resume_rejects_another_run(four_rooms, view, finite_demos):
    cfg = fpg.FPGConfig(n_traj=4)
    a = fpg.FPGLearner(four_rooms, view, finite_demos, cfg, seed=0)
    a.step(four_rooms)
    sd = roundtrip(a.state_dict())
    with pytest.raises(ValueError):
        fpg.FPGLearner(four_rooms, view, finite_demos, cfg, seed=1).load_state_dict(sd)            # another seed
    with pytest.raises(ValueError):
        fpg.FPGLearner(four_rooms, view, finite_demos, cfg, seed=0, stream=("other",)).load_state_dict(sd)
    other = finite_demos.copy()
    other[0, 5] = (other[0, 5] + 1) % four_rooms.n_states
    with pytest.raises(ValueError):
        fpg.FPGLearner(four_rooms, view, other, cfg, seed=0).load_state_dict(sd)                  # other demonstrations
    with pytest.raises(ValueError):
        fpg.FPGLearner(four_rooms, view, finite_demos, fpg.FPGConfig(n_traj=4, smoothing_eps=0.02), seed=0).load_state_dict(sd)


@needs_cuda
def test_cpu_and_cuda_agree_on_one_fpg_update(four_rooms, view, finite_demos):
    """Tabular policies start uniform, so both devices collect the same first batch; the updates must agree."""
    cfg = fpg.FPGConfig(n_traj=10)
    learners = {dev: fpg.FPGLearner(four_rooms, view, finite_demos, cfg, seed=15, device=dev) for dev in ("cpu", "cuda")}
    for learner in learners.values():
        learner.step(four_rooms)
    torch.testing.assert_close(learners["cpu"].policy.phi, learners["cuda"].policy.phi.cpu(), rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------------- FB update wiring, targets, freezing
def _fb_trainer(env, view, n=2000, seed=21, **kw):
    buf = fb.collect_replay(env, fb.ReplayConfig(n, 50), make_rng(seed, "tests", "fbwiring"))
    cfg = fb.FBConfig(batch_size=32, target_batch_size=24, **kw)
    tr = fb.FBPretrainer(ObsSpec.from_env(env), 5, view["gamma"], cfg, seed=seed)
    tr.attach_buffer(buf)
    return tr, buf, cfg


def _batch_losses(nets, buf, i, j, z, cfg, gamma):
    enc = lambda idx: torch.nn.functional.one_hot(torch.as_tensor(idx), 40).float()
    F_i = nets["F"](enc(buf.obs[i]), z)[torch.arange(len(i)), torch.as_tensor(buf.actions[i])]
    B_x = nets["B"](enc(buf.next_obs[j]))
    B_next = nets["B"](enc(buf.next_obs[i]))
    with torch.no_grad():
        Fbar = fb.expected_features(nets["F_target"](enc(buf.next_obs[i]), z), z, cfg.tau)
        Bm_x = nets["B_target"](enc(buf.next_obs[j]))
    return fb.successor_loss(F_i, B_x, Fbar, Bm_x, B_next, gamma), fb.orth_penalty(B_next, B_x)


def test_fb_first_adam_step_follows_the_derived_gradients(four_rooms, view):
    """xi descends L_hat; omega descends L_hat + lambda_orth L_reg (first Adam step: -lr g / (|g| + eps))."""
    import copy

    tr, buf, cfg = _fb_trainer(four_rooms, view, orth_coef=1.0)
    nets = {k: copy.deepcopy(getattr(tr, k)) for k in ("F", "B", "F_target", "B_target")}
    rng_copy = copy.deepcopy(tr.rng)
    tr.update()
    i = rng_copy.integers(0, len(buf), size=32)
    j = rng_copy.integers(0, len(buf), size=24)
    z = torch.as_tensor(fb.sample_latents(32, cfg.d, cfg.radius, rng_copy)).float()
    L, R = _batch_losses(nets, buf, i, j, z, cfg, view["gamma"])
    gF = torch.autograd.grad(L, list(nets["F"].parameters()), retain_graph=True)
    gB = torch.autograd.grad(L + cfg.orth_coef * R, list(nets["B"].parameters()), retain_graph=True)
    gB_wrong = torch.autograd.grad(L, list(nets["B"].parameters()))
    for name, grads in (("F", gF), ("B", gB)):
        for p_new, p_old, g in zip(getattr(tr, name).parameters(), nets[name].parameters(), grads):
            # Adam's first moment after one step is (1 - beta1) g: the exact gradient that was used
            used = tr.optimizer.state[p_new]["exp_avg"] / (1.0 - 0.9)
            torch.testing.assert_close(used, g, rtol=1e-4, atol=1e-7)
            want = -cfg.lr * g / (g.abs() + cfg.adam_eps)
            torch.testing.assert_close(p_new.detach() - p_old.detach(), want, rtol=0, atol=1.5e-7)
    diff = max(float((a - b).abs().max()) for a, b in zip(gB, gB_wrong))
    assert diff > 1e-4                                            # the penalty does change omega's gradient


def test_polyak_update_follows_the_optimizer_step(four_rooms, view):
    import copy

    tr, _, cfg = _fb_trainer(four_rooms, view, lr=1e-2)          # a large step makes old and new online copies distinct
    tr.train(2)
    old = [p.detach().clone() for p in list(tr.F_target.parameters()) + list(tr.B_target.parameters())]
    online_before = [p.detach().clone() for p in list(tr.F.parameters()) + list(tr.B.parameters())]
    tr.update()
    new_online = list(tr.F.parameters()) + list(tr.B.parameters())
    err_after, err_before = 0.0, 0.0
    for p_t, p_old, p_on, p_on0 in zip(list(tr.F_target.parameters()) + list(tr.B_target.parameters()), old, new_online, online_before):
        err_after = max(err_after, float((p_t.detach() - (0.99 * p_old + 0.01 * p_on.detach())).abs().max()))
        err_before = max(err_before, float((p_t.detach() - (0.99 * p_old + 0.01 * p_on0)).abs().max()))
    assert err_after <= 3e-7            # float32 rounding of the in-place update
    assert err_before > 1e-5            # Polyak with the pre-step online copy would be visibly different


def test_frozen_model_is_the_online_copy_with_its_gram(four_rooms, view):
    """D1 and FB-10: after pretraining the model is the online copy, and G_hat_B is its Gram over the buffer next states."""
    tr, buf, cfg = _fb_trainer(four_rooms, view)
    tr.train(5)
    model = tr.model()
    for a, b in zip(model.F_net.parameters(), tr.F.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(model.B_net.parameters(), tr.B.parameters()):
        assert torch.equal(a, b)
    assert any(not torch.equal(a, b) for a, b in zip(model.F_net.parameters(), tr.F_target.parameters()))
    with torch.no_grad():
        Bx = tr.B(torch.nn.functional.one_hot(torch.as_tensor(buf.next_obs), 40).float()).double().numpy()
    np.testing.assert_allclose(model.gram64, Bx.T @ Bx / len(buf), rtol=1e-6, atol=1e-8)
    with torch.no_grad():
        Bs = tr.B(torch.nn.functional.one_hot(torch.as_tensor(buf.obs), 40).float()).double().numpy()
    assert np.abs(model.gram64 - Bs.T @ Bs / len(buf)).max() > 1e-6           # not the Gram of the current states


def test_behaviour_actions_and_resets_are_uniform(four_rooms, cont_env):
    """ENV-6 hypotheses on real buffers: uniform actions, uniform resets, and (finite) uniform states at a fixed step."""
    from scipy.stats import chisquare

    buf = fb.collect_replay(four_rooms, fb.ReplayConfig(50_000, 50), make_rng(22, "tests", "uniform", "finite"))
    assert chisquare(np.bincount(buf.actions, minlength=5)).pvalue > 1e-4
    assert chisquare(np.bincount(buf.obs[buf.step_in_segment == 0], minlength=40)).pvalue > 1e-4
    assert chisquare(np.bincount(buf.next_obs[buf.step_in_segment == 25], minlength=40)).pvalue > 1e-4
    cbuf = fb.collect_replay(cont_env, fb.ReplayConfig(20_000, 50), make_rng(23, "tests", "uniform", "cont"))
    assert chisquare(np.bincount(cbuf.actions, minlength=5)).pvalue > 1e-4
    starts = cbuf.obs[cbuf.step_in_segment == 0]
    cells = np.floor(starts * 7).astype(int)
    ids = cells[:, 0] * 7 + cells[:, 1]
    counts = np.bincount(ids, minlength=49)
    free = np.array([cont_env.is_free(np.array([[(c // 7 + 0.5) / 7, (c % 7 + 0.5) / 7]]))[0] for c in range(49)])
    assert counts[~free].sum() == 0
    assert chisquare(counts[free]).pvalue > 1e-4                 # every free cell has the same area L^2


def test_fb_resume_guards(four_rooms, view):
    tr, buf, cfg = _fb_trainer(four_rooms, view)
    tr.train(2)
    sd = roundtrip(tr.state_dict())
    spec = ObsSpec.from_env(four_rooms)
    with pytest.raises(ValueError):
        fb.FBPretrainer(spec, 5, view["gamma"], cfg, seed=99).load_state_dict(sd)                 # another seed
    with pytest.raises(ValueError):
        fb.FBPretrainer(spec, 5, 0.9, cfg, seed=21).load_state_dict(sd)                          # another gamma
    with pytest.raises(ValueError):
        fb.FBPretrainer(spec, 5, view["gamma"], cfg, seed=21, stream=("x",)).load_state_dict(sd)  # another stream
    early = fb.FBPretrainer(spec, 5, view["gamma"], cfg, seed=21)
    early.load_state_dict(sd)                                     # loaded before a buffer is attached ...
    other = fb.collect_replay(four_rooms, fb.ReplayConfig(500, 50), make_rng(1, "other"))
    with pytest.raises(ValueError):
        early.attach_buffer(other)                                # ... the buffer hash is still enforced
    early.attach_buffer(buf)


def test_float64_conversion_of_the_model(four_rooms, view):
    tr, buf, cfg = _fb_trainer(four_rooms, view)
    tr.train(3)
    model = tr.model()
    m64 = model.converted(torch.float64)
    assert m64.dtype == torch.float64 and all(not p.requires_grad for p in m64.parameters())
    np.testing.assert_array_equal(m64.gram64, model.gram64)
    z = fb.sample_latents(1, 16, model.radius, make_rng(2, "z"))[0]
    with torch.no_grad():
        a = model.forward_features(buf.obs[:50], z).double().numpy()
        b = m64.forward_features(buf.obs[:50], z).numpy()
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_dm_without_projection_starts_from_the_raw_er_latent(four_rooms, view, finite_demos):
    tr, _, _ = _fb_trainer(four_rooms, view)
    model = tr.model()
    res = fb_il.dm_fit(model, finite_demos, fb_il.DMConfig(steps=0, project=False))
    np.testing.assert_allclose(res.z0, res.er.z_raw, atol=1e-12)
    res_p = fb_il.dm_fit(model, finite_demos, fb_il.DMConfig(steps=0))
    np.testing.assert_allclose(res_p.z0, res_p.er.z, atol=1e-12)
