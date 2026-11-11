"""Test helpers: random finite MDPs, policies, softmax Jacobians and per-trajectory statistics."""
from __future__ import annotations

import numpy as np

from il_memoir import oracle


def random_mdp(rng: np.random.Generator, n_states: int, n_actions: int) -> np.ndarray:
    """``P[s, a, s']`` with Dirichlet(1) rows."""
    return rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))


def random_policy(rng: np.random.Generator, n_states: int, n_actions: int, alpha: float = 1.0) -> np.ndarray:
    """Strictly positive stochastic policy with Dirichlet(alpha) rows."""
    pi = rng.dirichlet(alpha * np.ones(n_actions), size=n_states)
    pi = np.clip(pi, 1e-6, None)
    return pi / pi.sum(axis=1, keepdims=True)


def softmax_rows(phi: np.ndarray) -> np.ndarray:
    z = phi - phi.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def softmax_policy_jacobian(pi: np.ndarray) -> np.ndarray:
    """``D[s, b, a] = d pi(b|s) / d phi(s, a) = pi(b|s) (1[a=b] - pi(a|s))`` for tabular softmax logits."""
    n_s, n_a = pi.shape
    eye = np.eye(n_a)
    return pi[:, :, None] * (eye[None, :, :] - pi[:, None, :])


def per_trajectory_occupancy(next_states: np.ndarray, n_states: int, gamma: float) -> np.ndarray:
    """``Y[i, s] = sum_t gamma^t / Z_H 1[s^i_{t+1} = s]``; the occupancy estimate is ``Y.mean(0)``."""
    n, horizon = next_states.shape
    g = gamma ** np.arange(horizon)
    g = g / g.sum()
    Y = np.zeros((n, n_states))
    np.add.at(Y, (np.repeat(np.arange(n), horizon), next_states.ravel()), np.tile(g, n))
    return Y


def assert_within_se(estimate: np.ndarray, target: np.ndarray, se: np.ndarray, k: float = 5.0, floor: float = 1e-4):
    """``|estimate - target| <= k * se + floor`` elementwise, with an informative message."""
    estimate, target, se = (np.asarray(v, dtype=np.float64) for v in (estimate, target, se))
    err = np.abs(estimate - target)
    bound = k * se + floor
    worst = int(np.argmax(err - bound))
    assert np.all(err <= bound), (
        f"max excess at index {worst}: |{estimate.ravel()[worst]:.6g} - {target.ravel()[worst]:.6g}| = "
        f"{err.ravel()[worst]:.3g} > {bound.ravel()[worst]:.3g}"
    )


def exact_greedy_fb(rng: np.random.Generator, n_states: int = 5, n_actions: int = 3, gamma: float = 0.9):
    """Exact greedy FB at one latent z (theory Prop. 5.8(b)): one-hot B, F = M+/nu for the optimal policy of r_z.

    Returns ``(P, nu, z, B, F, gamma)`` with ``F`` of shape ``(S, A, S)`` evaluated at ``z``.  For use by the
    FB algebra tests of Prompt 3B as well as the oracle diagnostic tests.
    """
    P = random_mdp(rng, n_states, n_actions)
    nu = rng.dirichlet(np.ones(n_states))
    z = rng.normal(size=n_states)
    r_z = z / nu
    vi = oracle.value_iteration(P, r_z, gamma, tol=1e-11)
    M = oracle.successor_next_state(P, vi.pi, gamma)
    F = M / nu[None, None, :]
    return P, nu, z, np.eye(n_states), F, gamma
