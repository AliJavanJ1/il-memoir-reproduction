"""Helpers for the 3B learner tests: exact trajectory enumeration, exact f-PG objects and FB constructions.

These helpers may use the oracle: they build the *independent* exact quantities
against which the learner modules (which never import the oracle) are tested.
"""
from __future__ import annotations

import numpy as np
import torch

from il_memoir import divergences as dv
from il_memoir import oracle

from _helpers import softmax_policy_jacobian, softmax_rows


def enumerate_trajectories(P: np.ndarray, pi: np.ndarray, mu0: np.ndarray, H: int):
    """Every trajectory ``(s_0, a_0, ..., s_H)`` with positive probability: states ``(n, H+1)``, actions ``(n, H)``, probabilities."""
    n_s, n_a = pi.shape
    paths = [((s,), (), float(mu0[s])) for s in range(n_s) if mu0[s] > 0]
    for _ in range(H):
        nxt = []
        for states, acts, p in paths:
            s = states[-1]
            for a in range(n_a):
                for s1 in range(n_s):
                    pr = p * pi[s, a] * P[s, a, s1]
                    if pr > 0:
                        nxt.append((states + (s1,), acts + (a,), pr))
        paths = nxt
    states = np.array([p[0] for p in paths], dtype=np.int64)
    actions = np.array([p[1] for p in paths], dtype=np.int64).reshape(len(paths), H)
    probs = np.array([p[2] for p in paths])
    return states, actions, probs


def exact_J(P, logits, mu0, gamma, H, q, name, eps, ref):
    """Exact smoothed objective ``J_eps(phi) = D_f((d_phi)_eps || q_eps)`` (oracle occupancy, finite matrix sum)."""
    d = oracle.occupancy_next_state(P, softmax_rows(logits), mu0, gamma, H)
    return dv.smoothed_divergence(d, q, name, eps, ref)


def occupancy_jacobian_logits(P, logits, mu0, gamma, H) -> np.ndarray:
    """``jac[x, (s, a)] = d d_phi(x) / d phi(s, a)`` for tabular softmax: oracle forward-mode Jacobian times the softmax Jacobian."""
    pi = softmax_rows(logits)
    jac_pi = oracle.occupancy_next_state_jacobian(P, pi, mu0, gamma, H)        # [x, s, b]
    dsoft = softmax_policy_jacobian(pi)                                           # [s, b, a]
    return np.einsum("xsb,sba->xsa", jac_pi, dsoft).reshape(P.shape[0], -1)


def central_fd(fun, x0: np.ndarray, h: float = 1e-6) -> np.ndarray:
    x = np.array(x0, dtype=np.float64, copy=True)
    out = np.zeros(x.size)
    flat = x.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + h
        fp = fun(x)
        flat[i] = old - h
        fm = fun(x)
        flat[i] = old
        out[i] = (fp - fm) / (2 * h)
    return out


def set_tabular_logits(policy, logits: np.ndarray) -> None:
    with torch.no_grad():
        policy.phi.copy_(torch.as_tensor(logits, dtype=policy.phi.dtype))


def exact_softmax_fb(P: np.ndarray, nu: np.ndarray, z: np.ndarray, tau: float, gamma: float,
                     iters: int = 200_000, damp: float = 0.3, tol: float = 1e-13):
    """Exact softmax FB at one latent (theory Prop. 5.10, FB-4(b)): one-hot ``B`` (d = #S), ``r_z = z / nu``.

    Finds a fixed point ``pi = softmax(Q^pi_{r_z} / tau)`` by damped iteration and returns
    ``(F, pi, residual)`` with ``F[s, a, :] = M+^pi(s, a, .) / nu`` (so ``F(s,a,z)^T z = Q^pi_{r_z}``).
    """
    n_s, n_a, _ = P.shape
    r_z = z / nu
    pi = np.full((n_s, n_a), 1.0 / n_a)
    res = np.inf
    for _ in range(iters):
        Q, _ = oracle.policy_evaluation(P, pi, r_z, gamma)
        new = softmax_rows(Q / tau)
        res = float(np.max(np.abs(new - pi)))
        pi = (1 - damp) * pi + damp * new
        if res < tol:
            break
    Q, _ = oracle.policy_evaluation(P, pi, r_z, gamma)
    res = float(np.max(np.abs(softmax_rows(Q / tau) - pi)))
    M = oracle.successor_next_state(P, pi, gamma)
    return M / nu[None, None, :], pi, res
