"""f-PG adapted to a fixed expert target (theory section 3; METHOD_MAP section 3).

Objective and direction
-----------------------
The learner minimizes ``J_eps(phi) = D_f((d_phi)_eps || (q_hat_E)_eps)`` on a finite
cell partition (states of the finite maze; histogram cells in the continuous
maze), with both arguments smoothed by the strictly positive reference law
(Prop. 3.10(c)).  The update is **descent**: ``phi <- phi - eta * grad_hat``
(Adam in practice; Heuristic, approximation O1).

Exact gradient and estimator (Thm 3.6, Thm 3.7, Prop. 3.8, Prop. 3.9)
---------------------------------------------------------------------
``grad J = Z_H^{-1} E[ sum_k gamma^k chi_k (G_k - beta_k) ]`` with the score
``chi_k = grad log pi(a_k | s_k)``, the signal-to-go
``G_k = sum_{t=k}^{H-1} gamma^{t-k} g(s_{t+1})`` and the signal
``g(b) = (1 - eps) f'((d_phi)_eps(b) / q_eps(b))``.  :func:`surrogate_loss` is the
scalar whose autograd gradient is exactly this estimator; :func:`gradient_estimate`
returns it as a vector.

Signal batch (U-6, decided in Prompt 3B): cross-fitting with ``n_folds`` folds
-------------------------------------------------------------------------------
All ``n_traj`` trajectories of an iteration are collected with the current
policy (all counted).  Trajectory ``i`` has fold id ``i mod n_folds``; theory
section 7.7 writes the folds ``I_j`` with ``j = 1 + fold id`` and
``n_f = n_folds``.  For the score terms of fold ``j`` the plug-in signal
``g_hat_{-j}`` and the time baseline ``beta_{-j}(m)`` are computed from the other
folds only.  Conditionally on the other folds they are a fixed bounded function
of the next state and a constant per time step, so each fold's term has
conditional mean ``grad_phi Z_H^{-1} E_phi sum_t gamma^t g_hat_{-j}(s_{t+1})``
(theory Prop. 7.11, FPG-9; the baseline drops out by Prop. 3.8(a)).  For counts
and histogram cells this is ``sum_b g_hat_{-j}(b) grad d_phi(b)``, with the error
bound of Prop. 3.10(a).  For the KDE signal no exact cell signal exists, and no bound
against a gradient of ``J`` is claimed (Remark 3.11).  The plug-in signal itself
is biased (Prop. 3.10(b)); nothing here is an unbiased estimate of ``grad J``.
``n_folds = 1`` (with ``same_batch=True``) is the same-batch estimator (codes S3,
S4), biased by Prop. 3.10(b); it is kept for diagnostics only.

Estimators (the same convention for learner and expert data)
------------------------------------------------------------
Weights ``gamma^t / (n Z_H)`` on ``s_{t+1}`` (``density.discount_weights``).
Finite maze: discount-weighted counts, uniform reference on the free cells.
Continuous maze: the free-space histogram (masses with masses, free-area
reference) or the cut-and-normalize weighted KDE (densities with densities,
reference ``1/|S|``); the KDE update is a plug-in approximation (Remark 3.11,
code S2), not a theorem.  The wall geometry of the learner view is used only in
:func:`free_space_from_view` (METHOD_MAP section 1 exception).

Policies (U-12, decided): tabular softmax logits (finite) and a categorical MLP
with two tanh hidden layers (continuous); both are C^1, as Remark 3.3 requires.

Data access: this module receives a simulator (reset/step/rollout), the learner
view ``EnvironmentConfig.learner_dict()`` and the demonstration **states**
(``learner_states()``).  It never sees P, a reward, a goal or expert actions.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from . import divergences as dv
from .common import Trajectories, check_probability_rows, make_rng
from .density import FreeSpace, FreeSpaceHistogram, WeightedKDE, discount_weights, finite_occupancy, uniform_reference
from .nets import (
    ACTIVATIONS,
    DTYPES,
    MLP,
    ObsSpec,
    check_finite_tensor,
    dtype_name,
    dtype_of,
    flat_grad,
    init_seed,
    optimizer_to,
    resolve_device,
    rng_state,
    set_rng_state,
    torch_generator,
)

BASELINES = ("none", "time")
DEFAULT_LR_TABULAR = 0.1      # starting values when FPGConfig.lr is None (pilot candidates)
DEFAULT_LR_MLP = 1e-3
CONTINUOUS_DENSITIES = ("kde", "histogram")


# =============================================================================
# configuration
# =============================================================================
@dataclass(frozen=True)
class FPGConfig:
    """f-PG settings.  Defaults are the 3B starting values; the pilot (Prompt 4) may change them.

    ``kde_bandwidth``, ``smoothing_eps`` and ``histogram_bins`` default to ``None``,
    meaning the frozen value of the learner view (``configs/environment.yaml``).
    ``lr = None`` means the starting value for the policy class: 0.1 for tabular
    logits and 1e-3 for the MLP (both are pilot candidates, not tuned values).
    """

    divergence: str = "js"
    n_traj: int = 20
    n_folds: int = 2
    baseline: str = "time"
    lr: Optional[float] = None
    adam_eps: float = 1e-8
    density: str = "kde"
    kde_bandwidth: Optional[float] = None
    smoothing_eps: Optional[float] = None
    histogram_bins: Optional[int] = None
    hidden: int = 64
    n_hidden_layers: int = 2
    activation: str = "tanh"
    out_init_scale: float = 0.01
    ppo_epochs: int = 0
    ppo_clip: float = 0.2
    entropy_coef: float = 0.0
    dtype: str = "float32"
    same_batch: bool = False

    def __post_init__(self) -> None:
        if self.divergence not in dv.GENERATORS:
            raise ValueError(f"divergence must be one of {dv.GENERATORS}")
        for name in ("n_traj", "n_folds", "hidden", "n_hidden_layers", "ppo_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.n_folds < 1 or self.n_traj < self.n_folds:
            raise ValueError("n_folds >= 1 and n_traj >= n_folds are required (every fold needs a trajectory)")
        if not isinstance(self.same_batch, bool):
            raise ValueError("same_batch must be a boolean")
        if (self.n_folds == 1) != self.same_batch:
            raise ValueError("n_folds = 1 is the biased same-batch estimator (Prop. 3.10(b)): it requires same_batch=True, "
                             "and same_batch=True requires n_folds = 1")
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}")
        if self.density not in CONTINUOUS_DENSITIES:
            raise ValueError(f"density must be one of {CONTINUOUS_DENSITIES}")
        for name in ("lr", "adam_eps", "out_init_scale", "ppo_clip", "entropy_coef"):
            value = getattr(self, name)
            if name == "lr" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if (self.lr is not None and self.lr <= 0) or self.adam_eps <= 0 or self.out_init_scale <= 0:
            raise ValueError("lr, adam_eps and out_init_scale must be positive")
        if not 0.0 < self.ppo_clip < 1.0 or self.entropy_coef < 0 or self.ppo_epochs < 0:
            raise ValueError("0 < ppo_clip < 1, entropy_coef >= 0 and ppo_epochs >= 0 are required")
        for name in ("kde_bandwidth", "smoothing_eps"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)):
                raise ValueError(f"{name} must be a finite number or None")
        if self.kde_bandwidth is not None and not self.kde_bandwidth > 0:
            raise ValueError("kde_bandwidth must be positive")
        if self.smoothing_eps is not None and not 0.0 < self.smoothing_eps <= 1.0:
            raise ValueError("smoothing_eps must lie in (0, 1]")
        if self.histogram_bins is not None and (
            isinstance(self.histogram_bins, bool) or not isinstance(self.histogram_bins, (int, np.integer)) or self.histogram_bins < 1
        ):
            raise ValueError("histogram_bins must be a positive integer or None")
        if self.hidden < 1 or self.n_hidden_layers < 1:
            raise ValueError("hidden and n_hidden_layers must be positive")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)} (C^1 only)")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FPGConfig":
        allowed = {f.name for f in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown f-PG setting(s): {sorted(unknown)}")
        return cls(**dict(raw))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# policies (METHOD_MAP: fpg.policy_probs)
# =============================================================================
class TabularSoftmaxPolicy(nn.Module):
    """``pi_phi(a | s) = softmax_a phi[s, a]`` with ``phi`` in R^{S x A} (theory Lemma 3.2); zeros give the uniform policy."""

    kind = "tabular"

    def __init__(self, n_states: int, n_actions: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.phi = nn.Parameter(torch.zeros(self.n_states, self.n_actions, dtype=dtype))

    def logits(self, obs: Any) -> torch.Tensor:
        idx = torch.as_tensor(np.asarray(obs, dtype=np.int64), device=self.phi.device)
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.n_states):
            raise ValueError("state index out of range")
        return self.phi[idx]

    def record(self) -> Dict[str, Any]:
        return {"kind": self.kind, "n_states": self.n_states, "n_actions": self.n_actions}


class MLPPolicy(nn.Module):
    """Categorical policy ``softmax_a l_phi(x, a)`` with an MLP on raw positions (theory Remark 3.3; C^1 activation)."""

    kind = "mlp"

    def __init__(
        self,
        n_actions: int,
        hidden: int,
        n_hidden_layers: int,
        activation: str,
        generator: torch.Generator,
        out_init_scale: float = 0.01,
        dtype: torch.dtype = torch.float32,
        obs_dim: int = 2,
    ):
        super().__init__()
        self.n_actions = int(n_actions)
        self.obs_dim = int(obs_dim)
        self.hidden = int(hidden)
        self.n_hidden_layers = int(n_hidden_layers)
        self.activation = str(activation)
        self.mlp = MLP(
            self.obs_dim, [self.hidden] * self.n_hidden_layers, self.n_actions, self.activation, generator,
            first_layer_norm=False, out_scale=out_init_scale, dtype=dtype,
        )

    def logits(self, obs: Any) -> torch.Tensor:
        p = next(self.parameters())
        if isinstance(obs, torch.Tensor):
            x = obs.to(device=p.device, dtype=p.dtype)
        else:
            arr = np.asarray(obs, dtype=np.float64)
            if arr.shape[-1:] != (self.obs_dim,) or not np.all(np.isfinite(arr)):
                raise ValueError(f"positions must be finite with last dimension {self.obs_dim}")
            x = torch.as_tensor(arr, device=p.device).to(p.dtype)
        return self.mlp(x)

    def record(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "n_actions": self.n_actions,
            "obs_dim": self.obs_dim,
            "hidden": self.hidden,
            "n_hidden_layers": self.n_hidden_layers,
            "activation": self.activation,
        }


Policy = Union[TabularSoftmaxPolicy, MLPPolicy]


def make_policy(obs_spec: ObsSpec, n_actions: int, cfg: FPGConfig, generator: torch.Generator) -> Policy:
    dtype = dtype_of(cfg.dtype)
    if obs_spec.finite:
        return TabularSoftmaxPolicy(obs_spec.n_states, n_actions, dtype=dtype)
    return MLPPolicy(n_actions, cfg.hidden, cfg.n_hidden_layers, cfg.activation, generator, cfg.out_init_scale, dtype)


def policy_probs(policy: Policy, obs: Any) -> torch.Tensor:
    """``pi_phi(. | s)`` for a batch of observations (rows sum to one)."""
    return torch.softmax(policy.logits(obs), dim=-1)


def log_probs(policy: Policy, obs: Any, actions: Any) -> torch.Tensor:
    """``log pi_phi(a | s)`` elementwise; ``obs`` has shape ``actions.shape`` (+ ``(2,)`` for positions)."""
    logits = policy.logits(obs)
    act = torch.as_tensor(np.asarray(actions, dtype=np.int64), device=logits.device)
    if act.shape != logits.shape[:-1]:
        raise ValueError(f"actions shape {tuple(act.shape)} does not match observations {tuple(logits.shape[:-1])}")
    return torch.log_softmax(logits, dim=-1).gather(-1, act.unsqueeze(-1)).squeeze(-1)


def numpy_probs(policy: Policy, obs: Any) -> np.ndarray:
    """Action probabilities as float64 rows, renormalized (``common.check_probability_rows``)."""
    with torch.no_grad():
        probs = policy_probs(policy, obs)
    arr = probs.detach().to("cpu").numpy()
    return check_probability_rows(arr.reshape(-1, arr.shape[-1]))


def tabular_matrix(policy: TabularSoftmaxPolicy) -> np.ndarray:
    """The ``(S, A)`` float64 matrix of a tabular policy (rollouts, evaluation and oracle calls)."""
    return numpy_probs(policy, np.arange(policy.n_states))


def policy_callable(policy: Policy) -> Callable[[np.ndarray], np.ndarray]:
    """``states -> (n, A)`` float64 probabilities: the same policy for rollouts, checkpoints and evaluation."""
    if isinstance(policy, TabularSoftmaxPolicy):
        matrix = tabular_matrix(policy)
        return lambda states: matrix[np.asarray(states, dtype=np.int64)]
    return lambda states: numpy_probs(policy, states)


def mean_entropy(policy: Policy, obs: Any) -> torch.Tensor:
    """Mean action entropy (nats) of ``pi_phi(. | s)`` over the given observations."""
    logp = torch.log_softmax(policy.logits(obs), dim=-1)
    return -(logp.exp() * logp).sum(-1).mean()


def policy_record(policy: Policy) -> Dict[str, Any]:
    """A checkpoint record: architecture, dtype and parameters (loadable with ``torch.load(weights_only=True)``)."""
    rec = policy.record()
    p = next(policy.parameters())
    rec["dtype"] = dtype_name(p.dtype)
    rec["state_dict"] = {k: v.detach().to("cpu").clone() for k, v in policy.state_dict().items()}
    return rec


def policy_from_record(rec: Mapping[str, Any], device: Union[str, torch.device] = "cpu") -> Policy:
    dtype = dtype_of(rec["dtype"])
    if rec["kind"] == "tabular":
        policy: Policy = TabularSoftmaxPolicy(rec["n_states"], rec["n_actions"], dtype=dtype)
    elif rec["kind"] == "mlp":
        policy = MLPPolicy(
            rec["n_actions"], rec["hidden"], rec["n_hidden_layers"], rec["activation"], torch_generator(0),
            dtype=dtype, obs_dim=rec["obs_dim"],
        )
    else:
        raise ValueError(f"unknown policy kind {rec['kind']!r}")
    policy.load_state_dict(rec["state_dict"])
    return policy.to(resolve_device(device))


def save_policy(policy: Policy, path) -> None:
    torch.save(policy_record(policy), path)


def load_policy(path, device: Union[str, torch.device] = "cpu") -> Policy:
    return policy_from_record(torch.load(path, map_location="cpu", weights_only=True), device)


# =============================================================================
# rollouts (METHOD_MAP: fpg.collect)
# =============================================================================
def collect(env: Any, policy: Policy, n: int, horizon: int, rng: np.random.Generator) -> Trajectories:
    """``n`` i.i.d. trajectories of ``horizon`` transitions from the start law (A-FPG; Prop. 3.9(a)).

    Uses only the simulator's ``rollout``; the probabilities are the policy's
    own (float64, renormalized), exactly those of :func:`policy_callable`.
    """
    return env.rollout(policy_callable(policy), int(n), int(horizon), rng)


# =============================================================================
# occupancy estimators and the plug-in signal (METHOD_MAP: fpg.occupancy_estimate, fpg.signal)
# =============================================================================
def free_space_from_view(view: Mapping[str, Any]) -> FreeSpace:
    """The continuous free space from the learner view's wall cells.

    This is the only use of the wall geometry by the learners (METHOD_MAP
    section 1 exception): it feeds the KDE normalization and the free-area
    reference law of the histogram, and nothing else.
    """
    size = int(view["grid"]["size"])
    layout = view["continuous"]["layout"]
    blocked = view["grid"]["layouts"][layout]
    cell = 1.0 / size
    walls = [[x * cell, y * cell, (x + 1) * cell, (y + 1) * cell] for (x, y) in sorted(tuple(c) for c in blocked)]
    return FreeSpace(np.asarray(walls, dtype=np.float64).reshape(-1, 4))


def _array_sha256(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _next_state_array(states: np.ndarray, finite: bool) -> np.ndarray:
    states = np.asarray(states)
    if finite:
        if states.ndim != 2:
            raise ValueError("finite trajectories must have shape (n, H+1)")
        return states[:, 1:].astype(np.int64)
    if states.ndim != 3 or states.shape[-1] != 2:
        raise ValueError("continuous trajectories must have shape (n, H+1, 2)")
    return states[:, 1:].astype(np.float64)


class SignalEstimator:
    """Plug-in occupancy estimates, the fixed smoothed expert target, and the f-PG signal.

    ``kind`` is ``counts`` (finite), ``histogram`` or ``kde`` (continuous).  The
    learner and the expert data use the same weights (``gamma^t / (n Z_H)`` on
    ``s_{t+1}``) and the same estimator.  For logging, the continuous objective
    is always evaluated on the histogram cells.
    """

    def __init__(
        self,
        kind: str,
        name: str,
        eps: float,
        gamma: float,
        n_states: Optional[int] = None,
        space: Optional[FreeSpace] = None,
        bins: Optional[int] = None,
        bandwidth: Optional[float] = None,
    ):
        if kind not in ("counts",) + CONTINUOUS_DENSITIES:
            raise ValueError(f"unknown estimator kind {kind!r}")
        if name not in dv.GENERATORS:
            raise ValueError(f"unknown generator {name!r}")
        if not (np.isfinite(eps) and 0.0 < eps <= 1.0):
            raise ValueError("the smoothing weight must lie in (0, 1] (both arguments are smoothed, Prop. 3.10(c))")
        self.kind = kind
        self.name = name
        self.eps = float(eps)
        self.gamma = float(gamma)
        self.finite = kind == "counts"
        if self.finite:
            if n_states is None or n_states < 1:
                raise ValueError("counts need n_states")
            self.n_states = int(n_states)
            self.reference = uniform_reference(self.n_states)
            self.hist = None
            self.space = None
        else:
            if space is None or bins is None:
                raise ValueError("continuous estimators need the free space and the histogram bins")
            self.space = space
            self.hist = FreeSpaceHistogram(space, int(bins))
            self.reference = self.hist.reference
            if kind == "kde":
                if bandwidth is None or not (np.isfinite(bandwidth) and bandwidth > 0):
                    raise ValueError("the KDE needs a positive bandwidth")
            self.bandwidth = None if bandwidth is None else float(bandwidth)
        self.target_masses: Optional[np.ndarray] = None
        self.target_kde: Optional[WeightedKDE] = None
        self.target_horizon: Optional[int] = None

    # ------------------------------------------------------------ weights
    def _points_weights(self, next_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n, horizon = next_states.shape[0], next_states.shape[1]
        w = discount_weights(n, horizon, self.gamma).reshape(-1)
        points = next_states.reshape((-1,) if self.finite else (-1, 2))
        return points, w

    def cell_masses(self, next_states: np.ndarray) -> np.ndarray:
        """Discount-weighted masses on the cells (states, or histogram cells): unbiased for ``d+_{pi,H}`` cells (ENV-4(a))."""
        if self.finite:
            return finite_occupancy(next_states, self.n_states, self.gamma)
        points, w = self._points_weights(next_states)
        return self.hist.masses(points, w)

    # ------------------------------------------------------------- target
    def set_target(self, demo_states: np.ndarray) -> None:
        """The fixed target ``q_hat_E`` from the demonstration states ``s_0..s_H`` (``s_1..s_H`` are weighted)."""
        nxt = _next_state_array(demo_states, self.finite)
        self.target_horizon = int(nxt.shape[1])
        self.target_masses = self.cell_masses(nxt)
        if self.kind == "kde":
            points, w = self._points_weights(nxt)
            self.target_kde = WeightedKDE(self.space, points, w, self.bandwidth)

    def _require_target(self) -> None:
        if self.target_masses is None:
            raise RuntimeError("set_target must be called with the demonstration states first")

    # -------------------------------------------------------------- learner
    def occupancy_estimate(self, next_states: np.ndarray) -> Union[np.ndarray, WeightedKDE]:
        """Learner occupancy from ``next_states`` ``(n, H[, 2])``: cell masses, or a KDE object."""
        if self.kind == "kde":
            points, w = self._points_weights(next_states)
            return WeightedKDE(self.space, points, w, self.bandwidth)
        return self.cell_masses(next_states)

    def signal(self, estimate: Union[np.ndarray, WeightedKDE], query_next_states: np.ndarray) -> np.ndarray:
        """``g_hat(s_{t+1}) = (1 - eps) f'(d_hat_eps / q_hat_eps)`` at the query next states, shape ``(n, H)``."""
        self._require_target()
        shape = query_next_states.shape[:2]
        if self.kind == "kde":
            points = query_next_states.reshape(-1, 2)
            p_vals = estimate.density(points)
            q_vals = self.target_kde.density(points)
            g = dv.smoothed_signal(p_vals, q_vals, self.name, self.eps, 1.0 / self.space.area)
            return g.reshape(shape)
        g_cells = dv.smoothed_signal(estimate, self.target_masses, self.name, self.eps, self.reference)
        if self.finite:
            return g_cells[np.asarray(query_next_states, dtype=np.int64)]
        ids = self.hist.cell_ids(query_next_states.reshape(-1, 2))
        return g_cells[ids].reshape(shape)

    def plugin_objective(self, next_states: np.ndarray, name: Optional[str] = None) -> float:
        """``D_f(d_hat_eps || q_hat_eps)`` on the cells (a plug-in estimate of ``J_eps``; logging only)."""
        self._require_target()
        return dv.smoothed_divergence(
            self.cell_masses(next_states), self.target_masses, name or self.name, self.eps, self.reference
        )


# =============================================================================
# signal-to-go, baseline, gradient estimate (METHOD_MAP: rewards_to_go, baseline, gradient_estimate)
# =============================================================================
def discount_normalizer(gamma: float, horizon: int) -> float:
    """``Z_H = (1 - gamma^H) / (1 - gamma)``."""
    return float((1.0 - gamma**horizon) / (1.0 - gamma))


def rewards_to_go(signal_next: np.ndarray, gamma: float) -> np.ndarray:
    """``G[i, k] = sum_{t=k}^{H-1} gamma^{t-k} signal_next[i, t]`` (Eq. (3.4); ``signal_next[i, t] = g(s^i_{t+1})``)."""
    sig = np.asarray(signal_next, dtype=np.float64)
    if sig.ndim != 2:
        raise ValueError("signal_next must have shape (n, H)")
    out = np.empty_like(sig)
    acc = np.zeros(sig.shape[0])
    for k in range(sig.shape[1] - 1, -1, -1):
        acc = sig[:, k] + gamma * acc
        out[:, k] = acc
    return out


def score_weights(horizon: int, gamma: float) -> np.ndarray:
    """Weights ``gamma^k / Z_H`` of the score terms ``k = 0..H-1`` (Thm 3.7)."""
    g = gamma ** np.arange(int(horizon), dtype=np.float64)
    return g / g.sum()


def fold_ids(n_traj: int, n_folds: int) -> np.ndarray:
    """Fold id ``i mod n_folds`` of trajectory ``i`` (theory section 7.7: fold ``I_j``, ``j = 1 + id``).

    The trajectories are i.i.d., so no randomness is needed.
    """
    return np.arange(int(n_traj)) % int(n_folds)


def baseline(kind: str, other_rewards_to_go: np.ndarray, horizon: int) -> np.ndarray:
    """Baseline ``beta(t)`` from **other** trajectories: ``none`` (zeros) or ``time`` (mean signal-to-go per step).

    Given the other folds it is a constant per time step, hence bounded and
    ``F_k``-measurable, and Prop. 3.8(a) leaves the conditional mean unchanged.
    """
    if kind == "none":
        return np.zeros(int(horizon))
    if kind == "time":
        return np.asarray(other_rewards_to_go, dtype=np.float64).mean(axis=0)
    raise ValueError(f"unknown baseline {kind!r}")


@dataclass
class CrossFit:
    signal: np.ndarray      # (n, H): g_hat_{-fold(i)}(s^i_{t+1})
    to_go: np.ndarray       # (n, H): G^i_k with that signal
    baseline: np.ndarray    # (n, H): beta_{-fold(i)}(k)
    folds: np.ndarray       # (n,)

    @property
    def advantages(self) -> np.ndarray:
        return self.to_go - self.baseline


def crossfit_advantages(
    estimator: SignalEstimator, next_states: np.ndarray, n_folds: int, baseline_kind: str
) -> CrossFit:
    """Cross-fitted signal, signal-to-go and baseline (U-6).  ``n_folds = 1`` is the biased same-batch variant."""
    n, horizon = next_states.shape[0], next_states.shape[1]
    folds = fold_ids(n, n_folds)
    signal = np.empty((n, horizon))
    to_go = np.empty((n, horizon))
    base = np.empty((n, horizon))
    for k in range(int(n_folds)):
        own = folds == k
        other = ~own if n_folds > 1 else own
        est = estimator.occupancy_estimate(next_states[other])
        sig_own = estimator.signal(est, next_states[own])
        signal[own] = sig_own
        to_go[own] = rewards_to_go(sig_own, estimator.gamma)
        if baseline_kind == "none":
            base[own] = 0.0
        else:
            sig_other = sig_own if n_folds == 1 else estimator.signal(est, next_states[other])
            base[own] = baseline(baseline_kind, rewards_to_go(sig_other, estimator.gamma), horizon)[None, :]
    return CrossFit(signal, to_go, base, folds)


def surrogate_loss(policy: Policy, states: np.ndarray, actions: np.ndarray, advantages: np.ndarray, gamma: float) -> torch.Tensor:
    """``(1/N) sum_i sum_k (gamma^k / Z_H) A^i_k log pi(a^i_k | s^i_k)``.

    Its gradient is the f-PG estimator ``(1/N) sum_i Z_H^{-1} sum_k gamma^k chi^i_k A^i_k``
    (Prop. 3.9(a)); minimizing it is gradient **descent** on ``J``.
    """
    actions = np.asarray(actions, dtype=np.int64)
    adv = np.asarray(advantages, dtype=np.float64)
    if adv.shape != actions.shape:
        raise ValueError("advantages must have the shape (n, H) of the actions")
    logp = log_probs(policy, np.asarray(states)[:, :-1], actions)
    w = torch.as_tensor(score_weights(actions.shape[1], gamma)[None, :] * adv, device=logp.device, dtype=logp.dtype)
    return (logp * w).sum(dim=1).mean()


def gradient_estimate(
    policy: Policy,
    states: np.ndarray,
    actions: np.ndarray,
    signal_next: np.ndarray,
    beta: Union[np.ndarray, float],
    gamma: float,
) -> np.ndarray:
    """``(1/N) sum_i Z_H^{-1} sum_k gamma^k chi^i_k (G^i_k - beta^i_k)`` as a float64 vector (Eq. (3.4), Prop. 3.8)."""
    to_go = rewards_to_go(signal_next, gamma)
    adv = to_go - np.broadcast_to(np.asarray(beta, dtype=np.float64), to_go.shape)
    loss = surrogate_loss(policy, states, actions, adv, gamma)
    return flat_grad(loss, list(policy.parameters()))


def score(policy: Policy, state: Any, action: int) -> np.ndarray:
    """``chi = grad_phi log pi_phi(a | s)`` for one pair (autograd; tests compare it with finite differences)."""
    obs = np.asarray([state]) if np.ndim(state) == 0 else np.asarray(state)[None, ...]
    logp = log_probs(policy, obs, np.asarray([int(action)]))[0]
    return flat_grad(logp, list(policy.parameters()))


def exact_gradient(d: np.ndarray, q: np.ndarray, d_jacobian: np.ndarray, name: str, eps: float, ref: np.ndarray) -> np.ndarray:
    """``grad J_eps = sum_b (1 - eps) f'(d_eps(b) / q_eps(b)) grad d(b)`` (Eq. (3.2) with Prop. 3.10(c)).

    ``d`` is the exact occupancy on the cells and ``d_jacobian[b, :] = grad_phi d_phi(b)``,
    both supplied by the caller (tests build them with the oracle, which this
    learner module never imports).
    """
    g = dv.smoothed_signal(np.asarray(d, dtype=np.float64), q, name, eps, ref)
    return g @ np.asarray(d_jacobian, dtype=np.float64)


def gradient_estimate_sa(policy: Policy, states: np.ndarray, actions: np.ndarray, signal_sa: np.ndarray, gamma: float) -> np.ndarray:
    """State-action diagnostic (FPG-6, Prop. 3.13): ``signal_sa[i, t] = g^SA(s_t, a_t)``; the ``t = k`` term is kept.

    ``G^SA_k = sum_{t=k}^{H-1} gamma^{t-k} g^SA(s_t, a_t)``.  Tests only: learners never receive expert actions.
    """
    to_go = rewards_to_go(signal_sa, gamma)
    loss = surrogate_loss(policy, states, actions, to_go, gamma)
    return flat_grad(loss, list(policy.parameters()))


def ppo_surrogate(
    policy: Policy, states: np.ndarray, actions: np.ndarray, logp_old: torch.Tensor, c: np.ndarray, clip: float
) -> torch.Tensor:
    """``L^clip = (1/N) sum_i sum_k max(rho c, clip(rho, 1 - e, 1 + e) c)`` with ``c_k = gamma^k (G_k - beta_k) / Z_H``.

    Prop. 3.10(d): ``c`` is computed once under the sampling parameter ``phi'``; the
    gradient equals the f-PG estimator at ``phi'`` only (minimized; the max is the
    minimization form of PPO's min).  Optional (``ppo_epochs > 0``); off by default.
    """
    actions = np.asarray(actions, dtype=np.int64)
    logp = log_probs(policy, np.asarray(states)[:, :-1], actions)
    ratio = torch.exp(logp - logp_old)
    ct = torch.as_tensor(np.asarray(c, dtype=np.float64), device=logp.device, dtype=logp.dtype)
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    return torch.maximum(ratio * ct, clipped * ct).sum(dim=1).mean()


# =============================================================================
# the learner (METHOD_MAP: FPGLearner.step)
# =============================================================================
class FPGLearner:
    """On-policy f-PG learner: collect, cross-fit the plug-in signal, one descent step (or PPO epochs).

    ``state_dict`` / ``load_state_dict`` save and restore the policy, the Adam
    state, the rollout stream and the counters, so an interrupted run resumes
    exactly (3C runners checkpoint through them).
    """

    def __init__(
        self,
        env: Any,
        view: Mapping[str, Any],
        demo_states: np.ndarray,
        cfg: FPGConfig,
        seed: int,
        stream: Tuple[Union[str, int], ...] = ("fpg",),
        device: Union[str, torch.device] = "cpu",
    ):
        self.cfg = cfg
        self.obs_spec = ObsSpec.from_env(env)
        self.n_actions = int(env.n_actions)
        self.gamma = float(view["gamma"])
        self.horizon = int(view["horizon"])
        dens = view["density"]
        eps = cfg.smoothing_eps if cfg.smoothing_eps is not None else float(dens["smoothing_eps"])
        if self.obs_spec.finite:
            self.estimator = SignalEstimator("counts", cfg.divergence, eps, self.gamma, n_states=self.obs_spec.n_states)
        else:
            bins = cfg.histogram_bins if cfg.histogram_bins is not None else int(dens["histogram_bins"])
            bandwidth = cfg.kde_bandwidth if cfg.kde_bandwidth is not None else float(dens["kde_bandwidth"])
            self.estimator = SignalEstimator(
                cfg.density, cfg.divergence, eps, self.gamma, space=free_space_from_view(view), bins=bins, bandwidth=bandwidth
            )
        demo_states = np.asarray(demo_states)
        if demo_states.shape[1] != self.horizon + 1:
            raise ValueError(f"demonstrations must have H + 1 = {self.horizon + 1} states each")
        self.estimator.set_target(demo_states)
        self.demo_sha256 = _array_sha256(demo_states)
        self.resolved = {
            "gamma": self.gamma,
            "horizon": self.horizon,
            "estimator": self.estimator.kind,
            "smoothing_eps": float(self.estimator.eps),
            "kde_bandwidth": None if self.obs_spec.finite else self.estimator.bandwidth,
            "histogram_bins": None if self.obs_spec.finite else int(self.estimator.hist.bins),
            "obs_spec": self.obs_spec.to_dict(),
            "n_actions": self.n_actions,
        }
        self.n_demos = int(demo_states.shape[0])
        self.device = resolve_device(device)
        self.seed = int(seed)
        self.stream = tuple(stream)
        gen = torch_generator(init_seed(self.seed, *self.stream, "policy_init"))
        self.policy = make_policy(self.obs_spec, self.n_actions, cfg, gen).to(self.device)
        self.lr = float(cfg.lr) if cfg.lr is not None else (DEFAULT_LR_TABULAR if self.obs_spec.finite else DEFAULT_LR_MLP)
        self.resolved["lr"] = self.lr
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, eps=cfg.adam_eps)
        self.rng = make_rng(self.seed, *self.stream, "rollouts")
        self.iteration = 0
        self.env_steps = 0
        self.n_updates = 0

    # ----------------------------------------------------------- one iteration
    def advantages(self, traj: Trajectories) -> CrossFit:
        """Cross-fitted signal, signal-to-go and baseline of a batch, from its next states ``s_1..s_H``."""
        nxt = _next_state_array(traj.states, self.obs_spec.finite)
        return crossfit_advantages(self.estimator, nxt, self.cfg.n_folds, self.cfg.baseline)

    def _descent_step(self, total: torch.Tensor, what: str) -> float:
        """Check the loss and every gradient for finiteness, then take one optimizer step; returns the gradient norm."""
        check_finite_tensor(total, what)
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        sq = torch.zeros((), dtype=torch.float64, device=self.device)
        for p in self.policy.parameters():
            if p.grad is not None:
                check_finite_tensor(p.grad, f"{what} gradient")
                sq = sq + (p.grad.detach().double() ** 2).sum()
        self.optimizer.step()
        self.n_updates += 1
        return float(torch.sqrt(sq))

    def step(self, env: Any) -> Dict[str, float]:
        """Collect ``n_traj`` trajectories with the current policy and apply the update; returns measured metrics.

        The batch statistics are computed and checked before any parameter
        changes; the counters change only after a successful update.
        """
        cfg = self.cfg
        t0 = time.perf_counter()
        traj = collect(env, self.policy, cfg.n_traj, self.horizon, self.rng)
        t_collect = time.perf_counter() - t0
        nxt = _next_state_array(traj.states, self.obs_spec.finite)
        t1 = time.perf_counter()
        cf = self.advantages(traj)
        adv = cf.advantages
        t_signal = time.perf_counter() - t1
        visited = np.asarray(traj.states)[:, :-1]
        with torch.no_grad():
            entropy_before = float(mean_entropy(self.policy, visited))
        metrics = {
            "J_plugin": float(self.estimator.plugin_objective(nxt)),
            "JS_plugin": float(self.estimator.plugin_objective(nxt, "js")),
            "policy_entropy": entropy_before,
            "signal_mean": float(cf.signal.mean()),
            "signal_std": float(cf.signal.std()),
            "advantage_std": float(adv.std()),
        }
        for key, value in metrics.items():
            if not np.isfinite(value):
                raise FloatingPointError(f"f-PG batch statistic {key} is not finite")
        t2 = time.perf_counter()
        if cfg.ppo_epochs == 0:
            loss = surrogate_loss(self.policy, traj.states, traj.actions, adv, self.gamma)
            total = loss - cfg.entropy_coef * mean_entropy(self.policy, visited) if cfg.entropy_coef > 0 else loss
            grad_norm = self._descent_step(total, "f-PG surrogate loss")
            surrogate = float(loss.detach())
        else:
            with torch.no_grad():
                logp_old = log_probs(self.policy, visited, traj.actions)       # fixed at phi' for every epoch
            c = score_weights(self.horizon, self.gamma)[None, :] * adv
            grad_norm, surrogate = float("nan"), float("nan")
            for epoch in range(cfg.ppo_epochs):
                loss = ppo_surrogate(self.policy, traj.states, traj.actions, logp_old, c, cfg.ppo_clip)
                total = loss - cfg.entropy_coef * mean_entropy(self.policy, visited) if cfg.entropy_coef > 0 else loss
                norm = self._descent_step(total, "PPO surrogate loss")
                if epoch == 0:
                    grad_norm, surrogate = norm, float(loss.detach())
        t_update = time.perf_counter() - t2
        self.env_steps += cfg.n_traj * self.horizon
        self.iteration += 1
        metrics.update({
            "iteration": float(self.iteration),
            "env_steps": float(self.env_steps),
            "n_updates": float(self.n_updates),
            "grad_norm": grad_norm,
            "surrogate": surrogate,
            "time_collect_s": t_collect,
            "time_signal_s": t_signal,
            "time_update_s": t_update,
        })
        return metrics

    # ------------------------------------------------------------ evaluation
    def policy_callable(self) -> Callable[[np.ndarray], np.ndarray]:
        return policy_callable(self.policy)

    # ------------------------------------------------------------ checkpoints
    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": policy_record(self.policy),
            "optimizer": self.optimizer.state_dict(),
            "rng": rng_state(self.rng),
            "iteration": int(self.iteration),
            "env_steps": int(self.env_steps),
            "n_updates": int(self.n_updates),
            "config": self.cfg.to_dict(),
            "seed": self.seed,
            "stream": list(self.stream),
            "demo_sha256": self.demo_sha256,
            "resolved": dict(self.resolved),
        }

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        """Restore a checkpoint of the **same** run: configuration, seed, stream, demonstrations and resolved settings."""
        if dict(sd["config"]) != self.cfg.to_dict():
            raise ValueError("checkpoint configuration differs from this learner's configuration")
        if int(sd["seed"]) != self.seed or list(sd["stream"]) != list(self.stream):
            raise ValueError("checkpoint seed or stream differs from this learner's")
        if sd.get("demo_sha256") != self.demo_sha256:
            raise ValueError("checkpoint was trained on other demonstrations")
        if dict(sd.get("resolved", {})) != self.resolved:
            raise ValueError("checkpoint was trained with other resolved settings (gamma, H, estimator, eps, h, bins)")
        self.policy.load_state_dict(sd["policy"]["state_dict"])
        self.optimizer.load_state_dict(sd["optimizer"])
        optimizer_to(self.optimizer, self.device)
        set_rng_state(self.rng, sd["rng"])
        self.iteration = int(sd["iteration"])
        self.env_steps = int(sd["env_steps"])
        self.n_updates = int(sd["n_updates"])
