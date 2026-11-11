"""Task-blind forward-backward (FB) pretraining with discrete actions (theory section 5; METHOD_MAP section 4).

Model.  ``F_xi(s, a, z)`` in R^d for every action (one MLP head of size ``A d``
on ``[s, z]``) and ``B_omega(x)`` in R^d; rank ``d = 16``, two hidden layers of
width 64, float32.  The first hidden layer is ``Linear -> LayerNorm -> tanh``
(FI24 D.3, p. 28); the second uses SiLU instead of FI24's ReLU, because the
gradient results FB-8(b)-(c) and FB-9 need C^1 activations (U-14, decided).

Policy family (Eq. (5.1)).  ``pi_{z,tau}(a | s) = softmax_a(F(s,a,z)^T z / tau)``;
the same function (:func:`policy_probs`) defines the pretraining target policy
(with the target copy, Eq. (5.3)), ER_FB, DM_FB and evaluation (with the
online copy, decision D1).

Data (Assumption 5.21).  :func:`collect_replay` runs the simulator with uniform
random actions from uniform free-space resets (segments of ``segment_length``
transitions; every reset and transition counted) and stores ``(s, a, s')`` only:
no reward, no expert data, no terminal masking.  Minibatches are drawn uniformly
from the buffer, independently of ``s'`` beyond ``(s, a)``; ``nu_hat`` is the
empirical law of the stored next states, used for the target batch, the
orthonormality batch and the Gram matrix (one reference measure, A3).

Losses.  :func:`successor_loss` is ``L_hat`` of Eq. (5.6) with an **independent**
target batch; :func:`orth_penalty` is ``L_reg`` of Eq. (5.8) with ``x_i = s_{i+1}``
and the independent target batch as ``x'_j``.  ``xi`` descends ``L_hat`` and
``omega`` descends ``L_hat + lambda_orth L_reg`` (Adam; Heuristic O1).  Targets
are Polyak averages (D3).  After pretraining the online copy is frozen into an
:class:`FBModel` with the Gram matrix ``G_hat_B`` of the buffer next states (FB-10).

Nothing here claims convergence or exactness (Remark 5.27); at d = 16 exactness
is impossible in the finite mazes (Remark 5.9).
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from .common import check_probability_rows, make_rng
from .nets import (
    ACTIVATIONS,
    DTYPES,
    MLP,
    ObsSpec,
    check_finite_tensor,
    dtype_name,
    dtype_of,
    init_seed,
    optimizer_to,
    resolve_device,
    rng_state,
    set_rng_state,
    torch_generator,
)

MODEL_FORMAT = "il_memoir/fb-model-v1"
BUFFER_FORMAT = "il_memoir/fb-replay-v1"


# =============================================================================
# configuration
# =============================================================================
def _check_int(value: Any, what: str, low: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
        raise ValueError(f"{what} must be an integer >= {low}, got {value!r}")


def _check_real(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class ReplayConfig:
    """Task-blind replay: ``n_transitions`` in segments of ``segment_length`` from uniform free-space resets."""

    n_transitions: int = 50_000
    segment_length: int = 50

    def __post_init__(self) -> None:
        _check_int(self.n_transitions, "n_transitions")
        _check_int(self.segment_length, "segment_length")
        if self.n_transitions % self.segment_length:
            raise ValueError("n_transitions must be a multiple of segment_length (complete segments only)")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReplayConfig":
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown replay setting(s): {sorted(unknown)}")
        return cls(**dict(raw))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FBConfig:
    """FB pretraining settings (3B starting values; ``tau`` and ``latent_radius`` are pilot choices, U-2).

    ``latent_radius = None`` means ``R = sqrt(d)`` (decision D2).  ``orth_coef = 1``
    follows FB21 D.3 (p. 37) and FI24 D.4 (p. 29); ``target_momentum = 0.99``
    after every update follows FI24 D.4 (p. 29); batch 128 and Adam 3e-4 are the
    spec's initial choices.
    """

    d: int = 16
    hidden: int = 64
    n_hidden_layers: int = 2
    activation: str = "silu"
    first_layer_norm: bool = True
    tau: float = 0.1
    latent_radius: Optional[float] = None
    batch_size: int = 128
    target_batch_size: int = 128
    lr: float = 3e-4
    adam_eps: float = 1e-8
    orth_coef: float = 1.0
    target_momentum: float = 0.99
    n_updates: int = 10_000
    dtype: str = "float32"

    def __post_init__(self) -> None:
        for name in ("d", "hidden", "n_hidden_layers", "batch_size", "target_batch_size", "n_updates"):
            _check_int(getattr(self, name), name)
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)} (C^1 only, U-14)")
        if not isinstance(self.first_layer_norm, bool):
            raise ValueError("first_layer_norm must be a boolean")
        tau = _check_real(self.tau, "tau")
        lr = _check_real(self.lr, "lr")
        eps = _check_real(self.adam_eps, "adam_eps")
        orth = _check_real(self.orth_coef, "orth_coef")
        mom = _check_real(self.target_momentum, "target_momentum")
        if tau <= 0 or lr <= 0 or eps <= 0 or orth < 0 or not 0.0 <= mom < 1.0:
            raise ValueError("tau > 0, lr > 0, adam_eps > 0, orth_coef >= 0 and 0 <= target_momentum < 1 are required")
        if self.latent_radius is not None and _check_real(self.latent_radius, "latent_radius") <= 0:
            raise ValueError("latent_radius must be positive")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}")

    @property
    def radius(self) -> float:
        return float(np.sqrt(self.d)) if self.latent_radius is None else float(self.latent_radius)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FBConfig":
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown FB setting(s): {sorted(unknown)}")
        return cls(**dict(raw))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# task-blind replay (METHOD_MAP: fb.collect_replay)
# =============================================================================
@dataclass
class ReplayBuffer:
    """Stored transitions ``(s, a, s')`` only: no reward, no expert data, no terminal flag."""

    obs: np.ndarray
    actions: np.ndarray
    next_obs: np.ndarray
    segment: np.ndarray
    step_in_segment: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.actions.shape[0]
        for name in ("obs", "next_obs", "segment", "step_in_segment"):
            if getattr(self, name).shape[0] != n:
                raise ValueError(f"replay field {name} has the wrong length")

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def sha256(self) -> str:
        h = hashlib.sha256()
        for arr in (self.obs, self.actions, self.next_obs, self.segment, self.step_in_segment):
            a = np.ascontiguousarray(arr)
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def save(self, path) -> None:
        meta = dict(self.meta)
        meta["format"] = BUFFER_FORMAT
        meta["sha256"] = self.sha256()
        np.savez_compressed(
            path, obs=self.obs, actions=self.actions, next_obs=self.next_obs, segment=self.segment,
            step_in_segment=self.step_in_segment, meta=np.array(json.dumps(meta, sort_keys=True)),
        )

    @classmethod
    def load(cls, path) -> "ReplayBuffer":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            buf = cls(data["obs"], data["actions"], data["next_obs"], data["segment"], data["step_in_segment"], meta)
        if meta.get("format") != BUFFER_FORMAT or meta.get("sha256") != buf.sha256():
            raise ValueError("replay file format or content hash mismatch")
        return buf


def collect_replay(env: Any, cfg: ReplayConfig, rng: np.random.Generator) -> ReplayBuffer:
    """Task-blind behaviour (Assumption 5.21): uniform random actions from uniform free-space resets.

    Finite maze: resets uniform over the free cells (the simulator's states).
    Continuous maze: resets from ``env.sample_free_uniform`` (uniform on the free
    space).  Each segment is one simulator rollout of ``segment_length``
    transitions; the environment is continuing and no transition is masked.
    """
    obs_spec = ObsSpec.from_env(env)
    n_seg = cfg.n_transitions // cfg.segment_length
    n_actions = int(env.n_actions)
    if obs_spec.finite:
        starts = rng.integers(0, obs_spec.n_states, size=n_seg)
        uniform = np.full((obs_spec.n_states, n_actions), 1.0 / n_actions)
        policy: Any = uniform
    else:
        starts = env.sample_free_uniform(n_seg, rng)
        policy = lambda states: np.full((np.asarray(states).shape[0], n_actions), 1.0 / n_actions)
    traj = env.rollout(policy, n_seg, cfg.segment_length, rng, starts)
    states = np.asarray(traj.states)
    tail = () if obs_spec.finite else (2,)
    obs = states[:, :-1].reshape((-1,) + tail).copy()
    nxt = states[:, 1:].reshape((-1,) + tail).copy()
    meta = {
        "behaviour": "uniform random actions",
        "reset_law": "uniform on the free space" if not obs_spec.finite else "uniform on the free cells",
        "n_segments": int(n_seg),
        "n_resets": int(n_seg),
        "n_transitions": int(cfg.n_transitions),
        "segment_length": int(cfg.segment_length),
        "obs_spec": obs_spec.to_dict(),
        "n_actions": n_actions,
        "fields": ["obs", "actions", "next_obs", "segment", "step_in_segment"],
    }
    return ReplayBuffer(
        obs=obs,
        actions=np.asarray(traj.actions).reshape(-1).astype(np.int64),
        next_obs=nxt,
        segment=np.repeat(np.arange(n_seg, dtype=np.int64), cfg.segment_length),
        step_in_segment=np.tile(np.arange(cfg.segment_length, dtype=np.int64), n_seg),
        meta=meta,
    )


def replay_coverage(buffer: ReplayBuffer, n_actions: int, grid_size: int) -> Dict[str, Any]:
    """Coverage of states (finite) or lattice cells (continuous) and actions in the replay (a check, not a proof).

    For the continuous maze the cell of a position is ``floor(grid_size * x)``;
    no wall geometry is used.
    """
    if buffer.obs.ndim == 1:
        n_states = int(buffer.meta.get("obs_spec", {}).get("n_states", int(buffer.obs.max()) + 1))
        counts = np.zeros((n_states, n_actions), dtype=np.int64)
        np.add.at(counts, (buffer.obs, buffer.actions), 1)
        nxt = np.bincount(buffer.next_obs, minlength=n_states)
        return {
            "state_action_counts": counts,
            "min_state_action_count": int(counts.min()),
            "states_visited": int(np.sum(counts.sum(axis=1) > 0)),
            "next_states_visited": int(np.sum(nxt > 0)),
            "min_next_state_count": int(nxt.min()),
            "action_counts": counts.sum(axis=0),
        }
    cells = np.clip(np.floor(buffer.obs * grid_size).astype(np.int64), 0, grid_size - 1)
    counts = np.zeros((grid_size, grid_size, n_actions), dtype=np.int64)
    np.add.at(counts, (cells[:, 0], cells[:, 1], buffer.actions), 1)
    visited = counts.sum(axis=2) > 0
    per_cell = counts[visited]
    return {
        "cell_action_counts": counts,
        "cells_visited": int(visited.sum()),
        "min_cell_action_count": int(per_cell.min()) if per_cell.size else 0,
        "action_counts": counts.sum(axis=(0, 1)),
    }


# =============================================================================
# latents (METHOD_MAP: fb.sample_latents)
# =============================================================================
def sample_latents(n: int, d: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """``n`` latents uniform on the sphere ``{||z||_2 = radius}`` (D2): normalized standard Gaussians."""
    x = rng.standard_normal((int(n), int(d)))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    while np.any(norms == 0.0):                       # probability zero; kept for completeness
        bad = norms[:, 0] == 0.0
        x[bad] = rng.standard_normal((int(bad.sum()), int(d)))
        norms = np.linalg.norm(x, axis=1, keepdims=True)
    return float(radius) * x / norms


# =============================================================================
# networks (METHOD_MAP: fb.networks)
# =============================================================================
class ForwardNet(nn.Module):
    """``F_xi(s, a, z)`` for all actions: ``[enc(s), z] -> MLP -> (A, d)``."""

    def __init__(self, obs_dim: int, n_actions: int, d: int, cfg: FBConfig, generator: torch.Generator, dtype: torch.dtype):
        super().__init__()
        self.n_actions = int(n_actions)
        self.d = int(d)
        self.mlp = MLP(
            int(obs_dim) + self.d, [cfg.hidden] * cfg.n_hidden_layers, self.n_actions * self.d, cfg.activation,
            generator, first_layer_norm=cfg.first_layer_norm, out_scale=1.0, dtype=dtype,
        )

    def forward(self, obs_enc: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.expand(obs_enc.shape[0], -1)
        out = self.mlp(torch.cat([obs_enc, z], dim=-1))
        return out.reshape(out.shape[:-1] + (self.n_actions, self.d))


class BackwardNet(nn.Module):
    """``B_omega(x)``: ``enc(x) -> MLP -> d``."""

    def __init__(self, obs_dim: int, d: int, cfg: FBConfig, generator: torch.Generator, dtype: torch.dtype):
        super().__init__()
        self.d = int(d)
        self.mlp = MLP(
            int(obs_dim), [cfg.hidden] * cfg.n_hidden_layers, self.d, cfg.activation, generator,
            first_layer_norm=cfg.first_layer_norm, out_scale=1.0, dtype=dtype,
        )

    def forward(self, obs_enc: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs_enc)


def networks(obs_spec: ObsSpec, n_actions: int, cfg: FBConfig, generator: torch.Generator) -> Tuple[ForwardNet, BackwardNet]:
    """Online ``(F, B)`` initialized from ``generator`` (F first, then B)."""
    dtype = dtype_of(cfg.dtype)
    F_net = ForwardNet(obs_spec.dim, n_actions, cfg.d, cfg, generator, dtype)
    B_net = BackwardNet(obs_spec.dim, cfg.d, cfg, generator, dtype)
    return F_net, B_net


# =============================================================================
# policy family, target features, losses (METHOD_MAP: policy_probs, expected_features,
# successor_loss, orth_penalty, polyak_update)
# =============================================================================
def policy_scores(F_sa: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """``F(s, a, z)^T z`` for ``F_sa`` of shape ``(..., A, d)`` and ``z`` of shape ``(..., d)`` or ``(d,)``."""
    return (F_sa * z.unsqueeze(-2)).sum(dim=-1)


def policy_probs(F_sa: torch.Tensor, z: torch.Tensor, tau: float) -> torch.Tensor:
    """``pi_{z,tau}(a | s) = softmax_a(F(s,a,z)^T z / tau)`` (Eq. (5.1)); differentiable in ``F`` and ``z``."""
    if not tau > 0:
        raise ValueError("tau must be positive")
    return torch.softmax(policy_scores(F_sa, z) / tau, dim=-1)


def expected_features(F_sa: torch.Tensor, z: torch.Tensor, tau: float) -> torch.Tensor:
    """``F(s, z) = sum_a pi_{z,tau}(a | s) F(s, a, z)`` with the policy computed from the same ``F_sa``.

    With the target copy ``F^-`` this is ``F_bar^-`` of Eq. (5.3) (callers pass
    detached tensors); with the online copy it is DM_FB's ``F(x, z)``.
    """
    probs = policy_probs(F_sa, z, tau)
    return (probs.unsqueeze(-1) * F_sa).sum(dim=-2)


def _weights(n: int, w: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    if w is None:
        return torch.full((n,), 1.0 / n, dtype=like.dtype, device=like.device)
    w = torch.as_tensor(w, dtype=like.dtype, device=like.device)
    if w.shape != (n,):
        raise ValueError("weights must have one entry per sample")
    return w


def successor_loss(
    F_i: torch.Tensor,
    B_x: torch.Tensor,
    Fbar_target: torch.Tensor,
    B_target_x: torch.Tensor,
    B_next: torch.Tensor,
    gamma: float,
    w_trans: Optional[torch.Tensor] = None,
    w_target: Optional[torch.Tensor] = None,
    closed_form: bool = False,
) -> torch.Tensor:
    """``L_hat`` of Eq. (5.6) [eq:fb-loss-batch].

    ``1/2 sum_{i,j} w_i w'_j (F_i^T B(x_j) - gamma Fbar^-_i^T B^-(x_j))^2 - sum_i w_i F_i^T B(s_{i+1})``
    with uniform weights ``1/b``, ``1/b'`` by default.  ``F_i = F_xi(s_i, a_i, z_i)``,
    ``B_x = B_omega(x_j)`` for the **independent** target batch, ``Fbar_target =
    Fbar^-(s_{i+1}, z_i)`` and ``B_target_x = B^-(x_j)`` (no gradient), ``B_next =
    B_omega(s_{i+1})`` (the next-state diagonal term, Remark 5.23).  With weights equal
    to the probabilities of an enumerated law, the value is the population loss
    ``L`` (5.5) exactly.  ``closed_form`` uses the Gram matrices of Prop. 5.22(b).
    """
    b, bp = F_i.shape[0], B_x.shape[0]
    w = _weights(b, w_trans, F_i)
    wp = _weights(bp, w_target, F_i)
    if closed_form:
        G = B_x.T @ (wp[:, None] * B_x)
        Gx = B_x.T @ (wp[:, None] * B_target_x)
        Gm = B_target_x.T @ (wp[:, None] * B_target_x)
        quad = (F_i @ G * F_i).sum(-1) - 2 * gamma * (F_i @ Gx * Fbar_target).sum(-1) \
            + gamma**2 * (Fbar_target @ Gm * Fbar_target).sum(-1)
        quad = 0.5 * (w * quad).sum()
    else:
        resid = F_i @ B_x.T - gamma * (Fbar_target @ B_target_x.T)          # (b, b')
        quad = 0.5 * (w[:, None] * wp[None, :] * resid**2).sum()
    diag = (w * (F_i * B_next).sum(-1)).sum()
    return quad - diag


def orth_penalty(
    B_x: torch.Tensor,
    B_xp: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    wp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``L_reg`` of Eq. (5.8): ``sum_{i,j} w_i w'_j B(x_i)^T [B(x'_j)] [B(x_i)^T B(x'_j)] - sum_i w_i B(x_i)^T [B(x_i)]``.

    ``[.]`` is a stop-gradient.  Only its gradient is meaningful:
    ``E grad L_reg = grad (1/4)||G_B - I||_F^2`` for independent ``nu``-batches (Prop. 5.24(b)).
    """
    b, bp = B_x.shape[0], B_xp.shape[0]
    w = _weights(b, w, B_x)
    wp = _weights(bp, wp, B_x)
    Bp = B_xp.detach()
    inner = B_x.detach() @ Bp.T                                             # (b, b'), stop-gradient
    term1 = (w[:, None] * wp[None, :] * (B_x @ Bp.T) * inner).sum()
    term2 = (w * (B_x * B_x.detach()).sum(-1)).sum()
    return term1 - term2


@torch.no_grad()
def polyak_update(target: nn.Module, online: nn.Module, momentum: float) -> None:
    """``theta^- <- m theta^- + (1 - m) theta`` (D3; FB21 Algorithm 1, FI24 D.4)."""
    for p_t, p in zip(target.parameters(), online.parameters()):
        p_t.mul_(momentum).add_(p.detach(), alpha=1.0 - momentum)


def gram_from_features(B_x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``sum_j w_j B(x_j) B(x_j)^T`` (uniform weights by default): uncentered, never mean-subtracted (FB-10)."""
    w = _weights(B_x.shape[0], weights, B_x)
    return B_x.T @ (w[:, None] * B_x)


@torch.no_grad()
def gram(B_net: nn.Module, obs_spec: ObsSpec, next_obs: np.ndarray, device: Union[str, torch.device] = "cpu", chunk: int = 8192) -> np.ndarray:
    """``G_hat_B = (1/N) sum_j B(x_j) B(x_j)^T`` over all buffer next states (float64): the exact Gram matrix of ``nu_hat``."""
    dev = torch.device(device)
    p = next(B_net.parameters())
    n = int(np.asarray(next_obs).shape[0])
    acc = None
    for lo in range(0, n, chunk):
        x = obs_spec.encode(np.asarray(next_obs)[lo:lo + chunk], dev, p.dtype)
        bx = B_net(x).to(torch.float64)
        part = bx.T @ bx
        acc = part if acc is None else acc + part
    return (acc / n).to("cpu").numpy()


# =============================================================================
# the frozen model used after pretraining (decision D1: online copy)
# =============================================================================
class FBModel(nn.Module):
    """Frozen online ``(F, B)`` with ``G_hat_B``, ``tau`` and ``R``: the single model for ER_FB, DM_FB and evaluation."""

    def __init__(
        self,
        obs_spec: ObsSpec,
        n_actions: int,
        gamma: float,
        cfg: FBConfig,
        F_net: ForwardNet,
        B_net: BackwardNet,
        gram_matrix: np.ndarray,
        meta: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.obs_spec = obs_spec
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.cfg = cfg
        self.F_net = F_net
        self.B_net = B_net
        g = np.asarray(gram_matrix, dtype=np.float64)
        if g.shape != (cfg.d, cfg.d) or not np.all(np.isfinite(g)) or not np.allclose(g, g.T, atol=1e-10):
            raise ValueError("the Gram matrix must be a finite symmetric d x d matrix")
        self.gram64 = g
        p = next(F_net.parameters())
        self.register_buffer("gram", torch.as_tensor(g, dtype=p.dtype, device=p.device))
        self.meta = dict(meta or {})
        self.freeze()

    # ------------------------------------------------------------- freezing
    def freeze(self) -> "FBModel":
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    @property
    def tau(self) -> float:
        return float(self.cfg.tau)

    @property
    def radius(self) -> float:
        return self.cfg.radius

    @property
    def d(self) -> int:
        return int(self.cfg.d)

    @property
    def device(self) -> torch.device:
        return self.gram.device

    @property
    def dtype(self) -> torch.dtype:
        return self.gram.dtype

    # ----------------------------------------------------------- evaluation
    def encode(self, obs: Any) -> torch.Tensor:
        return self.obs_spec.encode(obs, self.device, self.dtype)

    def latent(self, z: Any) -> torch.Tensor:
        if isinstance(z, torch.Tensor):
            return z.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(np.asarray(z, dtype=np.float64), device=self.device).to(self.dtype)

    def forward_features(self, obs: Any, z: Any) -> torch.Tensor:
        """``F(s, a, z)`` for all actions, shape ``(n, A, d)``; differentiable in ``z``."""
        return self.F_net(self.encode(obs), self.latent(z))

    def backward_features(self, obs: Any) -> torch.Tensor:
        """``B(x)``, shape ``(n, d)``."""
        return self.B_net(self.encode(obs))

    def policy_probs(self, obs: Any, z: Any) -> torch.Tensor:
        z_t = self.latent(z)
        return policy_probs(self.F_net(self.encode(obs), z_t), z_t, self.tau)

    def expected_features(self, obs: Any, z: Any) -> torch.Tensor:
        z_t = self.latent(z)
        return expected_features(self.F_net(self.encode(obs), z_t), z_t, self.tau)

    def numpy_policy(self, z: Any) -> Callable[[np.ndarray], np.ndarray]:
        """``states -> (n, A)`` float64 rows of ``pi_{z,tau}``: the policy used for rollouts and evaluation."""
        z_fixed = self.latent(z).detach().clone()

        def probs(states: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                out = self.policy_probs(states, z_fixed)
            return check_probability_rows(out.to("cpu").numpy())

        return probs

    def policy_matrix(self, z: Any) -> np.ndarray:
        """``(S, A)`` float64 matrix of ``pi_{z,tau}`` (finite mazes: oracle evaluation in 3C)."""
        if not self.obs_spec.finite:
            raise ValueError("policy_matrix is defined for finite mazes only")
        return self.numpy_policy(z)(np.arange(self.obs_spec.n_states))

    # ---------------------------------------------------------- persistence
    def record(self) -> Dict[str, Any]:
        return {
            "format": MODEL_FORMAT,
            "obs_spec": self.obs_spec.to_dict(),
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "config": self.cfg.to_dict(),
            "F": {k: v.detach().to("cpu").clone() for k, v in self.F_net.state_dict().items()},
            "B": {k: v.detach().to("cpu").clone() for k, v in self.B_net.state_dict().items()},
            "gram": torch.as_tensor(self.gram64),
            "meta_json": json.dumps(self.meta, sort_keys=True),
        }

    def save(self, path) -> None:
        torch.save(self.record(), path)

    @classmethod
    def from_record(cls, rec: Mapping[str, Any], device: Union[str, torch.device] = "cpu") -> "FBModel":
        if rec.get("format") != MODEL_FORMAT:
            raise ValueError("not an FB model record")
        cfg = FBConfig.from_dict(rec["config"])
        obs_spec = ObsSpec.from_dict(rec["obs_spec"])
        F_net, B_net = networks(obs_spec, int(rec["n_actions"]), cfg, torch_generator(0))
        F_net.load_state_dict(rec["F"])
        B_net.load_state_dict(rec["B"])
        model = cls(obs_spec, int(rec["n_actions"]), float(rec["gamma"]), cfg, F_net, B_net,
                    rec["gram"].numpy(), json.loads(rec["meta_json"]))
        return model.to(resolve_device(device))

    @classmethod
    def load(cls, path, device: Union[str, torch.device] = "cpu") -> "FBModel":
        return cls.from_record(torch.load(path, map_location="cpu", weights_only=True), device)

    def converted(self, dtype: torch.dtype) -> "FBModel":
        """A copy in another dtype (float64 copies are used for finite-difference tests)."""
        rec = self.record()
        cfg = FBConfig.from_dict(dict(rec["config"], dtype=dtype_name(dtype)))
        F_net, B_net = networks(self.obs_spec, self.n_actions, cfg, torch_generator(0))
        F_net.load_state_dict({k: v.to(dtype) for k, v in rec["F"].items()})
        B_net.load_state_dict({k: v.to(dtype) for k, v in rec["B"].items()})
        return FBModel(self.obs_spec, self.n_actions, self.gamma, cfg, F_net, B_net, self.gram64, self.meta).to(self.device)


# =============================================================================
# pretraining (METHOD_MAP: fb.pretrain, fb.gram, fb.diagnostics)
# =============================================================================
class FBPretrainer:
    """Adam on ``L_hat`` (for ``xi``) and ``L_hat + lambda_orth L_reg`` (for ``omega``), Polyak targets.

    ``state_dict`` / ``load_state_dict`` restore networks, targets, Adam state,
    the minibatch/latent stream and the update counter (exact resume).
    """

    def __init__(
        self,
        obs_spec: ObsSpec,
        n_actions: int,
        gamma: float,
        cfg: FBConfig,
        seed: int,
        stream: Tuple[Union[str, int], ...] = ("fb",),
        device: Union[str, torch.device] = "cpu",
    ):
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must lie in (0, 1)")
        self.obs_spec = obs_spec
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.cfg = cfg
        self.seed = int(seed)
        self.stream = tuple(stream)
        self.device = resolve_device(device)
        self.dtype = dtype_of(cfg.dtype)
        gen = torch_generator(init_seed(self.seed, *self.stream, "init"))
        F_net, B_net = networks(obs_spec, self.n_actions, cfg, gen)
        self.F = F_net.to(self.device)
        self.B = B_net.to(self.device)
        self.F_target = copy.deepcopy(self.F)
        self.B_target = copy.deepcopy(self.B)
        for p in list(self.F_target.parameters()) + list(self.B_target.parameters()):
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(list(self.F.parameters()) + list(self.B.parameters()), lr=cfg.lr, eps=cfg.adam_eps)
        self.rng = make_rng(self.seed, *self.stream, "minibatches")
        self.n_updates = 0
        self._buf: Optional[Dict[str, Any]] = None
        self.buffer_sha256: Optional[str] = None
        self._expected_buffer_sha256: Optional[str] = None

    def attach_buffer(self, buffer: ReplayBuffer) -> None:
        """Use ``buffer`` (and its next states as ``nu_hat``) for every later update."""
        if len(buffer) < 2:
            raise ValueError("the replay buffer needs at least two transitions")
        if self._expected_buffer_sha256 is not None and buffer.sha256() != self._expected_buffer_sha256:
            raise ValueError("this replay buffer differs from the one the loaded checkpoint was trained on")
        obs = self.obs_spec.as_array(buffer.obs)
        nxt = self.obs_spec.as_array(buffer.next_obs)
        acts = np.asarray(buffer.actions, dtype=np.int64)
        if acts.min() < 0 or acts.max() >= self.n_actions:
            raise ValueError("replay action out of range")
        kind = torch.int64 if self.obs_spec.finite else self.dtype
        self._buf = {
            "n": len(buffer),
            "obs": torch.as_tensor(obs, device=self.device).to(kind),
            "next": torch.as_tensor(nxt, device=self.device).to(kind),
            "actions": torch.as_tensor(acts, device=self.device),
            "next_np": nxt,
        }
        self.buffer_sha256 = buffer.sha256()

    def _enc(self, x: torch.Tensor) -> torch.Tensor:
        return self.obs_spec.encode(x, self.device, self.dtype)

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transition indices ``i`` (size b), **independent** target indices ``j`` (size b', next states ~ nu_hat)
        and latents ``z_i ~ D_z``, drawn in this order from the pretrainer's stream."""
        if self._buf is None:
            raise RuntimeError("attach_buffer must be called before sampling")
        cfg, n = self.cfg, self._buf["n"]
        i = self.rng.integers(0, n, size=cfg.batch_size)
        j = self.rng.integers(0, n, size=cfg.target_batch_size)
        z = sample_latents(cfg.batch_size, cfg.d, cfg.radius, self.rng)
        return i, j, z

    def update(self) -> Dict[str, float]:
        """One minibatch update.

        Returns measured scalars of the batch used by this update, computed from
        the networks **before** the step (losses, the batch Gram deviation, the
        target and online policy entropies, the spread of the action scores
        ``F(s,a,z)^T z`` divided by ``tau``, which is what the softmax sees, and
        the mean of ``|F(s_i, a_i, z_i)^T z_i|`` over the batch, each transition
        with its own latent), keyed with the incremented update count.  Nothing is stepped unless the loss
        and every gradient are finite.
        """
        if self._buf is None:
            raise RuntimeError("attach_buffer must be called before update")
        cfg, buf = self.cfg, self._buf
        b = cfg.batch_size
        i_np, j_np, z_np = self.sample_batch()
        i = torch.as_tensor(i_np, device=self.device)
        j = torch.as_tensor(j_np, device=self.device)                         # independent nu_hat batch
        z = torch.as_tensor(z_np, device=self.device).to(self.dtype)
        s = self._enc(buf["obs"][i])
        s1 = self._enc(buf["next"][i])
        x = self._enc(buf["next"][j])
        a = buf["actions"][i]
        F_all = self.F(s, z)                                       # (b, A, d)
        F_i = F_all[torch.arange(b, device=self.device), a]        # (b, d)
        B_x = self.B(x)                                            # (b', d)
        B_next = self.B(s1)                                        # (b, d)
        with torch.no_grad():
            Fm_next = self.F_target(s1, z)                         # target copy at s_{i+1}
            Fbar = expected_features(Fm_next, z, cfg.tau)          # Eq. (5.3): pi^-_z from F^-, no gradient
            probs_m = policy_probs(Fm_next, z, cfg.tau)            # (logged entropy only)
            Bm_x = self.B_target(x)
        loss = successor_loss(F_i, B_x, Fbar, Bm_x, B_next, self.gamma)
        reg = orth_penalty(B_next, B_x)                            # x_i = s_{i+1}, x'_j = target batch
        total = loss + cfg.orth_coef * reg if cfg.orth_coef > 0 else loss
        check_finite_tensor(total, "FB loss")
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        for p in list(self.F.parameters()) + list(self.B.parameters()):
            if p.grad is not None:
                check_finite_tensor(p.grad, "FB gradient")
        with torch.no_grad():                                      # pre-step batch diagnostics
            g_batch = gram_from_features(B_x.detach().to(torch.float64))
            eye = torch.eye(cfg.d, dtype=torch.float64, device=g_batch.device)
            dev = float(torch.linalg.norm(g_batch - eye))
            ent_t = float(-(probs_m * torch.log(probs_m.clamp_min(1e-30))).sum(-1).mean())
            scores_on = policy_scores(F_all.detach(), z)            # (b, A) online scores at s_i
            probs_on = torch.softmax(scores_on / cfg.tau, dim=-1)
            ent_on = float(-(probs_on * torch.log(probs_on.clamp_min(1e-30))).sum(-1).mean())
            spread = float(((scores_on.max(dim=-1).values - scores_on.min(dim=-1).values) / cfg.tau).mean())
            mean_abs = float((F_i.detach() * z).sum(dim=-1).abs().mean())   # |F(s_i,a_i,z_i)^T z_i|, per sample
        self.optimizer.step()
        polyak_update(self.F_target, self.F, cfg.target_momentum)
        polyak_update(self.B_target, self.B, cfg.target_momentum)
        self.n_updates += 1
        return {
            "update": float(self.n_updates),
            "loss": float(loss.detach()),
            "orth_loss": float(reg.detach()),
            "total": float(total.detach()),
            "batch_gram_dev_fro": dev,
            "target_policy_entropy": ent_t,
            "online_policy_entropy": ent_on,
            "score_spread_over_tau": spread,
            "mean_abs_score": mean_abs,
        }

    def train(self, n_updates: int, log_every: int = 0) -> Dict[str, Any]:
        """Run ``n_updates`` updates; returns the logged metrics (every ``log_every`` updates and the last)."""
        logs = []
        t0 = time.perf_counter()
        for k in range(int(n_updates)):
            m = self.update()
            if (log_every and (k + 1) % log_every == 0) or k == n_updates - 1:
                m["wall_s"] = time.perf_counter() - t0
                logs.append(m)
        return {"logs": logs, "wall_s": time.perf_counter() - t0}

    def model(self) -> FBModel:
        """Freeze the **online** copy with ``G_hat_B`` over all buffer next states (D1, FB-10)."""
        if self._buf is None:
            raise RuntimeError("attach_buffer must be called first")
        F_net = copy.deepcopy(self.F)
        B_net = copy.deepcopy(self.B)
        g = gram(B_net, self.obs_spec, self._buf["next_np"], self.device)
        meta = {
            "seed": self.seed,
            "stream": list(self.stream),
            "n_updates": int(self.n_updates),
            "buffer_sha256": self.buffer_sha256,
            "n_buffer": int(self._buf["n"]),
        }
        return FBModel(self.obs_spec, self.n_actions, self.gamma, self.cfg, F_net, B_net, g, meta)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "F": self.F.state_dict(),
            "B": self.B.state_dict(),
            "F_target": self.F_target.state_dict(),
            "B_target": self.B_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": rng_state(self.rng),
            "n_updates": int(self.n_updates),
            "config": self.cfg.to_dict(),
            "obs_spec": self.obs_spec.to_dict(),
            "gamma": self.gamma,
            "seed": self.seed,
            "stream": list(self.stream),
            "buffer_sha256": self.buffer_sha256,
        }

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        """Restore a checkpoint of the **same** run: configuration, observation spec, gamma, seed, stream and buffer."""
        if dict(sd["config"]) != self.cfg.to_dict() or dict(sd["obs_spec"]) != self.obs_spec.to_dict():
            raise ValueError("checkpoint configuration differs from this pretrainer's configuration")
        if float(sd["gamma"]) != self.gamma or int(sd["seed"]) != self.seed or list(sd["stream"]) != list(self.stream):
            raise ValueError("checkpoint gamma, seed or stream differs from this pretrainer's")
        saved = sd.get("buffer_sha256")
        if self.buffer_sha256 is not None and saved != self.buffer_sha256:
            raise ValueError("checkpoint was trained on a different replay buffer")
        if self.buffer_sha256 is None:
            self._expected_buffer_sha256 = saved
        self.F.load_state_dict(sd["F"])
        self.B.load_state_dict(sd["B"])
        self.F_target.load_state_dict(sd["F_target"])
        self.B_target.load_state_dict(sd["B_target"])
        self.optimizer.load_state_dict(sd["optimizer"])
        optimizer_to(self.optimizer, self.device)
        set_rng_state(self.rng, sd["rng"])
        self.n_updates = int(sd["n_updates"])


@torch.no_grad()
def diagnostics(model: FBModel, buffer: ReplayBuffer, rng: np.random.Generator, n_samples: int = 1024) -> Dict[str, float]:
    """Logged, never used for training: ``lambda_min(G_hat_B)``, ``||G_hat_B - I||`` and ``F(s,a,z)^T int B dnu_hat``.

    Under exactness ``F(s,a,z)^T int B dnu = 1/(1 - gamma)`` (Prop. 5.20(c)); the trained
    model is not exact, so this is a diagnostic only.
    """
    G = model.gram64
    eig = np.linalg.eigvalsh(G)
    n = len(buffer)
    mean_b = np.zeros(model.d)
    for lo in range(0, n, 8192):
        mean_b += model.backward_features(buffer.next_obs[lo:lo + 8192]).to(torch.float64).sum(0).cpu().numpy()
    mean_b /= n
    idx = rng.integers(0, n, size=int(n_samples))
    z = sample_latents(int(n_samples), model.d, model.radius, rng)
    F_all = model.forward_features(buffer.obs[idx], z).to(torch.float64).cpu().numpy()
    vals = F_all[np.arange(int(n_samples)), buffer.actions[idx]] @ mean_b
    return {
        "gram_lambda_min": float(eig.min()),
        "gram_lambda_max": float(eig.max()),
        "gram_dev_fro": float(np.linalg.norm(G - np.eye(model.d))),
        "gram_dev_op": float(np.max(np.abs(eig - 1.0))),
        "mean_B_norm": float(np.linalg.norm(mean_b)),
        "F_dot_meanB_mean": float(vals.mean()),
        "F_dot_meanB_std": float(vals.std()),
        "one_over_one_minus_gamma": float(1.0 / (1.0 - model.gamma)),
    }


def pretrain(
    env: Any,
    view: Mapping[str, Any],
    replay_cfg: ReplayConfig,
    cfg: FBConfig,
    seed: int,
    stream: Tuple[Union[str, int], ...] = ("fb",),
    device: Union[str, torch.device] = "cpu",
    log_every: int = 0,
) -> Tuple[FBModel, ReplayBuffer, Dict[str, Any]]:
    """Collect the task-blind replay, pretrain, and freeze the online copy (a convenience for checks; runners
    call :class:`FBPretrainer` directly to checkpoint)."""
    obs_spec = ObsSpec.from_env(env)
    buffer = collect_replay(env, replay_cfg, make_rng(seed, *stream, "replay"))
    trainer = FBPretrainer(obs_spec, env.n_actions, float(view["gamma"]), cfg, seed, stream, device)
    trainer.attach_buffer(buffer)
    out = trainer.train(cfg.n_updates, log_every=log_every)
    return trainer.model(), buffer, out
