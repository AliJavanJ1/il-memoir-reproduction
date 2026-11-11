"""Exact finite oracle (float64).  NOT A LEARNER: evaluation, expert construction and tests only.

Conventions (NOTATION sections 1, 3, 5, 9; theory section 1)
------------------------------------------------------------
* ``P[s, a, s']`` is the transition matrix, ``pi[s, a]`` a stationary policy,
  ``mu0`` the initial distribution and ``r[s]`` a state reward read at the
  **next** state: ``Q(s, a) = sum_{s'} P(s'|s,a) [r(s') + gamma V(s')]``.
* ``Z_H = sum_{t<H} gamma^t``; ``d+_{pi,H} = Z_H^{-1} sum_{t<H} gamma^t Law(s_{t+1})``
  (what rollouts estimate) and ``d+_pi = (1 - gamma) sum_t gamma^t Law(s_{t+1})``.
* ``M+(s, a, .) = P(.|s,a) (I - gamma P_pi)^{-1}`` (mass ``1/(1-gamma)``).

The functions planned in METHOD_MAP section 2 keep their names; the helpers
added in Prompt 3A (``state_marginals``, ``successor_next_state_avg``,
``occupancy_next_state_jacobian``, ``h_step_value``, ``bellman_backup``,
``greedy_actions``, ``one_hot_policy``, ``fb_policy``,
``discount_normalizer``) are listed there too.  Linear systems are solved with
``numpy.linalg.solve``; no matrix is inverted explicitly.  Value iteration
reports its Bellman residual and the certificates of theory Lemma 1.16; nothing
here claims machine-exact optimality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

ROW_ATOL = 1e-12


# --------------------------------------------------------------------------- checks
def _check_P(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 3 or P.shape[0] != P.shape[2]:
        raise ValueError(f"P must have shape (S, A, S), got {P.shape}")
    if np.any(P < 0) or not np.allclose(P.sum(axis=2), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("P must be non-negative with rows summing to one")
    return P


def _check_pi(pi: np.ndarray, n_states: int, n_actions: int) -> np.ndarray:
    pi = np.asarray(pi, dtype=np.float64)
    if pi.shape != (n_states, n_actions):
        raise ValueError(f"pi must have shape {(n_states, n_actions)}, got {pi.shape}")
    if np.any(pi < 0) or not np.allclose(pi.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("pi must be non-negative with rows summing to one")
    return pi


def _check_dist(mu: np.ndarray, n_states: int) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64)
    if mu.shape != (n_states,) or np.any(mu < 0) or abs(mu.sum() - 1.0) > 1e-10:
        raise ValueError("mu0 must be a probability vector over the states")
    return mu


def discount_normalizer(gamma: float, horizon: int) -> float:
    """``Z_H = sum_{t=0}^{H-1} gamma^t = (1 - gamma^H) / (1 - gamma)``."""
    return float((1.0 - gamma**horizon) / (1.0 - gamma))


def one_hot_policy(actions: np.ndarray, n_actions: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.int64)
    pi = np.zeros((actions.shape[0], n_actions))
    pi[np.arange(actions.shape[0]), actions] = 1.0
    return pi


# ---------------------------------------------------------------- kernels and laws
def policy_kernel(P: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """``P_pi[s, s'] = sum_a pi(a|s) P(s'|s,a)`` (row-stochastic)."""
    P = _check_P(P)
    pi = _check_pi(pi, P.shape[0], P.shape[1])
    return np.einsum("sa,sat->st", pi, P)


def state_marginals(P: np.ndarray, pi: np.ndarray, mu0: np.ndarray, horizon: int) -> np.ndarray:
    """``Law(s_t)`` for ``t = 0..horizon`` as rows of a ``(horizon+1, S)`` array (Lemma 1.3)."""
    P_pi = policy_kernel(P, pi)
    mu0 = _check_dist(mu0, P_pi.shape[0])
    laws = np.empty((horizon + 1, P_pi.shape[0]))
    laws[0] = mu0
    for t in range(horizon):
        laws[t + 1] = laws[t] @ P_pi
    return laws


def successor_next_state(P: np.ndarray, pi: np.ndarray, gamma: float) -> np.ndarray:
    """``M+(s, a, s') = [P(.|s,a) (I - gamma P_pi)^{-1}](s')`` with shape ``(S, A, S)`` (OCC-3, FB-1)."""
    P = _check_P(P)
    n_s, n_a, _ = P.shape
    P_pi = policy_kernel(P, pi)
    lhs = (np.eye(n_s) - gamma * P_pi).T
    # X (I - gamma P_pi) = P~  <=>  (I - gamma P_pi)^T X^T = P~^T
    x = np.linalg.solve(lhs, P.reshape(n_s * n_a, n_s).T).T
    return x.reshape(n_s, n_a, n_s)


def successor_next_state_avg(P: np.ndarray, pi: np.ndarray, gamma: float) -> np.ndarray:
    """Action-averaged ``M+(s, .) = sum_a pi(a|s) M+(s, a, .) = P_pi (I - gamma P_pi)^{-1}``."""
    pi = np.asarray(pi, dtype=np.float64)
    return np.einsum("sa,sat->st", pi, successor_next_state(P, pi, gamma))


def occupancy_next_state(
    P: np.ndarray,
    pi: np.ndarray,
    mu0: np.ndarray,
    gamma: float,
    H: Optional[int] = None,
    method: str = "sum",
) -> np.ndarray:
    """Next-state occupancy ``d+_pi`` (``H=None``) or ``d+_{pi,H}`` (OCC-1, OCC-3, OCC-5).

    For finite ``H``, ``method='sum'`` evaluates the finite matrix sum
    ``Z_H^{-1} sum_{t<H} gamma^t mu0^T P_pi^{t+1}`` and ``method='resolvent'``
    the closed form ``Z_H^{-1} mu0^T P_pi (I - gamma^H P_pi^H)(I - gamma P_pi)^{-1}``
    of Prop. 1.8(d).  The two are algebraically equal; tests compare them.
    """
    P_pi = policy_kernel(P, pi)
    n_s = P_pi.shape[0]
    mu0 = _check_dist(mu0, n_s)
    lhs = (np.eye(n_s) - gamma * P_pi).T
    if H is None:
        return np.linalg.solve(lhs, (1.0 - gamma) * (P_pi.T @ mu0))
    H = int(H)
    if H < 1:
        raise ValueError("H must be at least 1")
    z_h = discount_normalizer(gamma, H)
    if method == "sum":
        laws = state_marginals(P, pi, mu0, H)
        weights = gamma ** np.arange(H, dtype=np.float64)
        return (weights @ laws[1:]) / z_h
    if method == "resolvent":
        row = mu0 @ P_pi
        row = row - gamma**H * (row @ np.linalg.matrix_power(P_pi, H))
        return np.linalg.solve(lhs, row) / z_h
    raise ValueError("method must be 'sum' or 'resolvent'")


def occupancy_state_action(
    P: np.ndarray, pi: np.ndarray, mu0: np.ndarray, gamma: float, H: Optional[int] = None
) -> np.ndarray:
    """Pre-action state-action occupancy ``d^SA_pi`` or ``d^SA_{pi,H}`` with shape ``(S, A)`` (OCC-1).

    ``d^SA_{pi,H}(s, a) = Z_H^{-1} sum_{t<H} gamma^t Law(s_t)(s) pi(a|s)``; its
    state marginal is the pre-action occupancy (OCC-2).
    """
    P_pi = policy_kernel(P, pi)
    n_s = P_pi.shape[0]
    mu0 = _check_dist(mu0, n_s)
    pi = np.asarray(pi, dtype=np.float64)
    if H is None:
        pre = np.linalg.solve((np.eye(n_s) - gamma * P_pi).T, (1.0 - gamma) * mu0)
    else:
        laws = state_marginals(P, pi, mu0, int(H) - 1)
        weights = gamma ** np.arange(int(H), dtype=np.float64)
        pre = (weights @ laws) / discount_normalizer(gamma, int(H))
    return pre[:, None] * pi


def occupancy_next_state_jacobian(
    P: np.ndarray, pi: np.ndarray, mu0: np.ndarray, gamma: float, H: int
) -> np.ndarray:
    """Exact Jacobian ``J[x, s, a] = d d+_{pi,H}(x) / d pi(a|s)`` (entries of pi as free variables).

    Forward-mode recursion, independent of the likelihood-ratio formula:
    ``Law_{t+1} = Law_t P_pi`` and ``dLaw_{t+1}[x, (s,a)] = sum_y dLaw_t[y, (s,a)] P_pi[y, x]
    + Law_t(s) P(x | s, a)``.  For a parametrized policy the chain rule gives
    ``grad_phi d(x) = sum_{s,a} J[x, s, a] grad_phi pi(a|s)``; this is the
    independent exact differentiable occupancy used to test f-PG gradients.
    """
    P = _check_P(P)
    P_pi = policy_kernel(P, pi)
    n_s, n_a, _ = P.shape
    law = _check_dist(mu0, n_s).copy()
    dlaw = np.zeros((n_s, n_s, n_a))
    jac = np.zeros((n_s, n_s, n_a))
    for t in range(int(H)):
        dlaw = np.einsum("ysa,yx->xsa", dlaw, P_pi) + np.einsum("s,sax->xsa", law, P)
        law = law @ P_pi
        jac += gamma**t * dlaw
    return jac / discount_normalizer(gamma, int(H))


# ------------------------------------------------------------- values and control
def policy_evaluation(P: np.ndarray, pi: np.ndarray, r: np.ndarray, gamma: float):
    """``(Q, V)`` with ``V = (I - gamma P_pi)^{-1} P_pi r`` and ``Q = P (r + gamma V)``.

    ``Q(s, a) = Q+_r(s, a) = sum_{s'} M+(s, a, s') r(s')`` (Lemma 5.3(a), OCC-4):
    the reward is read at the next state.
    """
    P = _check_P(P)
    r = np.asarray(r, dtype=np.float64)
    P_pi = policy_kernel(P, pi)
    V = np.linalg.solve(np.eye(P.shape[0]) - gamma * P_pi, P_pi @ r)
    Q = P @ (r + gamma * V)
    return Q, V


def bellman_backup(P: np.ndarray, r: np.ndarray, gamma: float, V: np.ndarray) -> np.ndarray:
    """``Q_V(s, a) = sum_{s'} P(s'|s,a) [r(s') + gamma V(s')]``; ``(T V)(s) = max_a Q_V(s, a)``."""
    return np.asarray(P, dtype=np.float64) @ (np.asarray(r, dtype=np.float64) + gamma * np.asarray(V))


def greedy_actions(Q: np.ndarray, tie_tol: float = 0.0) -> np.ndarray:
    """First maximizer in the fixed action order: smallest ``a`` with ``Q(s,a) >= max_b Q(s,b) - tie_tol``."""
    Q = np.asarray(Q, dtype=np.float64)
    best = Q.max(axis=1, keepdims=True)
    return np.argmax(Q >= best - tie_tol, axis=1).astype(np.int64)


@dataclass(frozen=True)
class ValueIterationResult:
    V: np.ndarray             # value-iteration iterate with residual `residual`
    Q: np.ndarray             # Bellman backup of V
    residual: float           # eta = |T V - V|_inf at the returned V
    n_iter: int
    converged: bool           # residual <= tol
    actions: np.ndarray       # stationary deterministic policy after policy-iteration refinement
    pi: np.ndarray            # the same policy as a one-hot matrix
    V_pi: np.ndarray          # exact value of `pi` (linear solve)
    Q_pi: np.ndarray
    pi_residual: float        # eta_pi = |T V_pi - V_pi|_inf
    refinement_steps: int
    refinement_converged: bool  # the returned policy is tie_tol-greedy for its own Q_pi

    def certificates(self, gamma: float) -> Dict[str, float]:
        """Residual certificates of theory Lemma 1.16 (no machine-exactness claim).

        * ``value_iterate_error_bound``: ``|V - V*|_inf <= eta / (1 - gamma)`` (i);
        * ``policy_gap_bound``: ``0 <= V* - V_pi <= eta_pi / (1 - gamma)`` (iii),
          for the returned (refined) policy, up to the floating-point error of
          its linear solve.
        Lemma 1.16(ii) concerns an exactly greedy policy of ``V``; the returned
        policy is refined and tie-tolerant, so (iii) is the certificate to use.
        """
        return {
            "value_iterate_error_bound": self.residual / (1.0 - gamma),
            "policy_gap_bound": self.pi_residual / (1.0 - gamma),
        }


def value_iteration(
    P: np.ndarray,
    r: np.ndarray,
    gamma: float,
    tol: float = 1e-12,
    tie_tol: float = 0.0,
    max_iter: int = 100_000,
    max_refine: int = 100,
) -> ValueIterationResult:
    """Stationary discounted optimum ``V*`` and a deterministic optimal policy (OCC-6).

    Iterate ``V <- T V`` from ``V = 0`` until ``|T V - V|_inf <= tol``.  The
    greedy policy of the final backup (first maximizer within ``tie_tol``) is
    then refined by exact policy iteration with a monotone switching rule: an
    action is changed only where it is more than ``tie_tol`` below
    ``max_a Q_pi(s, a)``, and then to the first action within ``tie_tol`` of the
    maximum.  Every switch strictly improves the policy (policy improvement
    theorem), so no policy repeats and the loop ends with a policy that is
    ``tie_tol``-greedy for its own ``Q_pi`` (``refinement_converged``), unless
    ``max_refine`` is exhausted.  The residual ``eta_pi = |T V_pi - V_pi|_inf``
    gives ``0 <= V* - V_pi <= eta_pi / (1 - gamma)`` (Lemma 1.16(iii)).
    """
    P = _check_P(P)
    r = np.asarray(r, dtype=np.float64)
    n_s, n_a, _ = P.shape
    V = np.zeros(n_s)
    residual = np.inf
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        Q = bellman_backup(P, r, gamma, V)
        TV = Q.max(axis=1)
        residual = float(np.max(np.abs(TV - V)))
        if residual <= tol:
            break
        V = TV
    Q = bellman_backup(P, r, gamma, V)
    residual = float(np.max(np.abs(Q.max(axis=1) - V)))
    actions = greedy_actions(Q, tie_tol)
    steps = 0
    converged = False
    for steps in range(1, max_refine + 1):
        Q_pi, V_pi = policy_evaluation(P, one_hot_policy(actions, n_a), r, gamma)
        current = Q_pi[np.arange(n_s), actions]
        improvable = current < Q_pi.max(axis=1) - tie_tol
        if not improvable.any():
            converged = True
            break
        actions = np.where(improvable, greedy_actions(Q_pi, tie_tol), actions)
    Q_pi, V_pi = policy_evaluation(P, one_hot_policy(actions, n_a), r, gamma)
    pi_residual = float(np.max(Q_pi.max(axis=1) - V_pi))
    return ValueIterationResult(
        V=V,
        Q=Q,
        residual=residual,
        n_iter=n_iter,
        converged=residual <= tol,
        actions=actions,
        pi=one_hot_policy(actions, n_a),
        V_pi=V_pi,
        Q_pi=Q_pi,
        pi_residual=max(pi_residual, 0.0),
        refinement_steps=steps,
        refinement_converged=converged,
    )


@dataclass(frozen=True)
class BackwardInductionResult:
    W: np.ndarray          # (H+1, S): W[k] = optimal value with H-k steps to go; W[H] = 0
    actions: np.ndarray    # (H, S): non-stationary optimal actions sigma*_k (exact maximizers, first on exact ties)

    @property
    def V_H_star(self) -> np.ndarray:
        return self.W[0]


def backward_induction(P: np.ndarray, r: np.ndarray, gamma: float, H: int) -> BackwardInductionResult:
    """H-step optimum ``V*_H`` by backward induction for the next-state-reward return (OCC-6, Prop. 1.17).

    ``W_H = 0`` and ``W_k = T W_{k+1}``; ``V*_H = W_0`` is optimal among all
    history-dependent policies for ``sum_{t<H} gamma^t r(s_{t+1})``.  Each
    returned action is an exact maximizer of its backup (the first in the
    action order on exact ties), so the non-stationary policy ``sigma*`` attains
    ``W`` (Prop. 1.17(b)).  No tolerance is used here: a tolerance belongs only
    to the stationary expert's refinement in :func:`value_iteration`.
    """
    P = _check_P(P)
    r = np.asarray(r, dtype=np.float64)
    H = int(H)
    n_s = P.shape[0]
    W = np.zeros((H + 1, n_s))
    acts = np.zeros((H, n_s), dtype=np.int64)
    for k in range(H - 1, -1, -1):
        Q = bellman_backup(P, r, gamma, W[k + 1])
        acts[k] = np.argmax(Q, axis=1)
        W[k] = Q[np.arange(n_s), acts[k]]
    return BackwardInductionResult(W=W, actions=acts)


def h_step_value(P: np.ndarray, pi: np.ndarray, r: np.ndarray, gamma: float, H: int) -> np.ndarray:
    """``V^pi_H(s) = E^pi_s sum_{t<H} gamma^t r(s_{t+1})`` for a stationary policy (finite recursion).

    ``U_0 = 0``, ``U_k = P_pi (r + gamma U_{k-1})``; returns ``U_H``.  Equals
    ``Z_H sum_x d+_{pi,H}(x) r(x)`` from ``mu0 = delta_s`` (OCC-4).
    """
    P_pi = policy_kernel(P, pi)
    r = np.asarray(r, dtype=np.float64)
    U = np.zeros(P_pi.shape[0])
    for _ in range(int(H)):
        U = P_pi @ (r + gamma * U)
    return U


@dataclass(frozen=True)
class TransitionRank:
    rank: int
    singular_values: np.ndarray
    tol: float

    @property
    def smallest_singular_value(self) -> float:
        return float(self.singular_values.min())


def transition_rank(P: np.ndarray, tol: Optional[float] = None) -> TransitionRank:
    """``rank P~`` of the ``(S*A) x S`` matrix with rows ``P(.|s,a)`` (FB-3(c), Remark 5.9).

    Default tolerance as in ``numpy.linalg.matrix_rank``.  If the rank exceeds
    the FB dimension d = 16, no exact rank-16 FB representation exists.
    """
    P = _check_P(P)
    n_s, n_a, _ = P.shape
    mat = P.reshape(n_s * n_a, n_s)
    sv = np.linalg.svd(mat, compute_uv=False)
    if tol is None:
        tol = float(sv.max() * max(mat.shape) * np.finfo(np.float64).eps)
    return TransitionRank(rank=int(np.sum(sv > tol)), singular_values=sv, tol=float(tol))


# ---------------------------------------------------------------- FB diagnostic
def fb_policy(F: np.ndarray, z: np.ndarray, tau: Optional[float], tie_tol: float = 0.0) -> np.ndarray:
    """Policy of an FB model at latent ``z``: softmax of ``F(s,a,z)^T z / tau``, or greedy if ``tau`` is None."""
    scores = np.einsum("sad,d->sa", np.asarray(F, dtype=np.float64), np.asarray(z, dtype=np.float64))
    if tau is None:
        return one_hot_policy(greedy_actions(scores, tie_tol), scores.shape[1])
    if tau <= 0:
        raise ValueError("tau must be positive (or None for the greedy family)")
    logits = scores / tau
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=1, keepdims=True)


def fb_error_diagnostic(
    F: np.ndarray,
    B: np.ndarray,
    nu_hat: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    tau: Optional[float],
    gamma: float,
    r: Optional[np.ndarray] = None,
    tie_tol: float = 0.0,
    zero_atol: float = 1e-12,
) -> Dict[str, object]:
    """Exact FB model error at one latent (FB-6, Thm 5.17).  EVALUATION ONLY.

    ``F`` has shape ``(S, A, d)`` (forward embeddings evaluated at ``z``), ``B``
    shape ``(S, d)`` and ``nu_hat`` shape ``(S,)`` (the reference measure).  The
    policy is ``pi_{z,tau}`` (or greedy for ``tau=None``).  Returns the error kernel
    ``E_z = M+^{pi_z} - F^T B nu_hat``, its TV norms ``sum_x |E(s,a,x)|`` and
    ``L2(nu_hat)*`` norms (``+inf`` where ``E`` charges a ``nu_hat``-null state,
    Lemma 5.15), and their suprema.  With a reward ``r``: ``z_r = sum_x nu_hat(x)
    r(x) B(x)``, the exact ``Q+_r`` of the model's policy, the model value
    ``F(s,a,z)^T z_r``, and the error levels ``|r|_inf sup TV`` and
    ``|r|_{L2(nu_hat)} sup L2*`` that bound it (Thm 5.17(a)).
    """
    P = _check_P(P)
    F = np.asarray(F, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    nu = np.asarray(nu_hat, dtype=np.float64)
    n_s, n_a, _ = P.shape
    if F.shape[:2] != (n_s, n_a) or B.shape != (n_s, F.shape[2]) or nu.shape != (n_s,):
        raise ValueError("shapes must be F (S, A, d), B (S, d), nu_hat (S,)")
    if np.any(nu < 0):
        raise ValueError("nu_hat must be non-negative")
    pi = fb_policy(F, z, tau, tie_tol)
    M = successor_next_state(P, pi, gamma)
    M_hat = np.einsum("sad,xd->sax", F, B) * nu[None, None, :]
    E = M - M_hat
    tv = np.abs(E).sum(axis=2)
    support = nu > 0
    l2 = np.sqrt(np.sum(E[:, :, support] ** 2 / nu[None, None, support], axis=2))
    singular = np.any(np.abs(E[:, :, ~support]) > zero_atol, axis=2) if (~support).any() else np.zeros_like(tv, bool)
    l2 = np.where(singular, np.inf, l2)
    out: Dict[str, object] = {
        "pi": pi,
        "M": M,
        "M_hat": M_hat,
        "E": E,
        "tv": tv,
        "l2_dual": l2,
        "sup_tv": float(tv.max()),
        "sup_l2_dual": float(l2.max()),
    }
    if r is not None:
        r = np.asarray(r, dtype=np.float64)
        z_r = B.T @ (nu * r)
        Q_true, _ = policy_evaluation(P, pi, r, gamma)
        Q_model = np.einsum("sad,d->sa", F, z_r)
        out.update(
            {
                "z_r": z_r,
                "Q_true": Q_true,
                "Q_model": Q_model,
                "eval_error": np.abs(Q_true - Q_model),
                "delta_tv": float(np.max(np.abs(r)) * tv.max()),
                "delta_l2_dual": float(np.sqrt(np.sum(nu * r**2)) * l2.max()),
            }
        )
    return out
