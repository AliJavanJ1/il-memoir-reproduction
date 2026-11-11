"""Imitation with a frozen FB model: ER_FB and DM_FB (theory section 6; METHOD_MAP sections 5-6).

The pretrained online copy (``fb.FBModel``: ``F``, ``B``, ``G_hat_B``, ``tau``,
``R``) is frozen.  Only the latent ``z`` changes, and no simulator data is
collected: this module has no environment handle.  It receives the
demonstration **states** ``s^i_0..s^i_H`` only (``learner_states()``).

ER_FB (Eq. (6.1), FI24 Eq. 8 and footnote 3)
    ``z_ER = (1/N_E) sum_i (1/H) sum_{t=1}^{H} B(s^i_t)``: uniform over ``s_1..s_H``
    within a demonstration, equal trajectory weight (``s_0`` excluded).  It is
    then projected onto the latent sphere ``Z_R`` (D2; Prop. 6.11(e)); the
    pre-projection norm is recorded (U-2).

DM_FB (Eqs. (6.2), (6.5), (6.6); decisions D4-D5)
    Targets ``psi^{(K),i}_t = sum_{k<K} gamma^k B(s^i_{t+k+1})`` for
    ``t in T_K = {0, ..., H-K}`` (``K = 60``, ``t = 0`` included), computed once
    with the frozen ``B``.  The loss is
    ``l_hat(z) = (1/n) sum_{(i,t)} [F(s^i_t,z)^T G_hat_B F(s^i_t,z) - 2 F(s^i_t,z)^T psi^{(K),i}_t]``
    with ``F(s,z) = sum_a pi_{z,tau}(a|s) F(s,a,z)``.  Its ``z``-gradient flows through
    ``F(s,a,z)`` **and** ``pi_{z,tau}`` (nothing is detached, Prop. 6.10).  The
    optimizer is Adam with a projection onto ``Z_R`` after each step, at most
    ``steps = 300`` updates from the projected ``z_ER`` (Heuristic; Prop. 6.11(b)
    covers exact projected gradient steps only).  The data reuse (warm start and
    loss from the same 4 demonstrations) is approximation S6.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .fb import FBModel, expected_features, policy_probs, sample_latents
from .nets import params_fingerprint

TINY_NORM = 1e-12


@dataclass(frozen=True)
class DMConfig:
    """DM_FB settings (D4: ``K = 60``; D5: ``t = 0`` included; at most 300 projected Adam steps)."""

    K: int = 60
    include_t0: bool = True
    steps: int = 300
    lr: float = 0.05
    adam_eps: float = 1e-8
    project: bool = True

    def __post_init__(self) -> None:
        for name in ("K", "steps"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < (1 if name == "K" else 0):
                raise ValueError(f"{name} must be a non-negative integer (K >= 1)")
        if self.steps > 300:
            raise ValueError("at most 300 latent updates (spec section 6)")
        for name in ("lr", "adam_eps"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if not isinstance(self.include_t0, bool) or not isinstance(self.project, bool):
            raise ValueError("include_t0 and project must be booleans")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DMConfig":
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown DM setting(s): {sorted(unknown)}")
        return cls(**dict(raw))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _demo_array(model: FBModel, demo_states: np.ndarray) -> np.ndarray:
    arr = model.obs_spec.as_array(demo_states)
    if (model.obs_spec.finite and arr.ndim != 2) or (not model.obs_spec.finite and arr.ndim != 3):
        raise ValueError("demonstration states must have shape (N_E, H+1) or (N_E, H+1, 2)")
    if arr.shape[1] < 2:
        raise ValueError("demonstrations need at least one transition")
    return arr


def _require_frozen(model: FBModel) -> None:
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("the FB model must be frozen (requires_grad False) for latent adaptation")


# =============================================================================
# ER_FB (METHOD_MAP: fb_il.er_fb_z, fb_il.project_to_sphere)
# =============================================================================
@torch.no_grad()
def er_fb_z(model: FBModel, demo_states: np.ndarray) -> np.ndarray:
    """``z_ER = (1/N_E) sum_i (1/H) sum_{t=1}^H B(s^i_t)`` (Eq. (6.1)), float64."""
    arr = _demo_array(model, demo_states)
    n_e, horizon = arr.shape[0], arr.shape[1] - 1
    later = arr[:, 1:].reshape((-1,) + arr.shape[2:])
    b = model.backward_features(later).to(torch.float64).reshape(n_e, horizon, -1)
    return b.mean(dim=1).mean(dim=0).cpu().numpy()


def project_to_sphere(z: np.ndarray, radius: float, fallback_rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """``z -> R z / ||z||`` (Prop. 6.11(e)); if ``||z|| <= 1e-12`` a stated fallback draws ``z`` from ``D_z``."""
    z = np.asarray(z, dtype=np.float64)
    norm = float(np.linalg.norm(z))
    if norm > TINY_NORM:
        return radius * z / norm, {"norm_before": norm, "fallback": False}
    if fallback_rng is None:
        raise ValueError("z has (numerically) zero norm and no fallback generator was given")
    return sample_latents(1, z.shape[0], radius, fallback_rng)[0], {"norm_before": norm, "fallback": True}


@dataclass
class ERResult:
    z: np.ndarray          # projected latent used by the policy
    z_raw: np.ndarray      # z_ER before projection
    norm_raw: float        # ||z_ER||_2 (reported with tau and R, U-2)
    fallback: bool


def er_fb(model: FBModel, demo_states: np.ndarray, fallback_rng: Optional[np.random.Generator] = None) -> ERResult:
    """ER_FB: ``z_ER`` projected onto ``Z_R``; the policy is ``model.numpy_policy(result.z)``."""
    z_raw = er_fb_z(model, demo_states)
    z, info = project_to_sphere(z_raw, model.radius, fallback_rng)
    return ERResult(z=z, z_raw=z_raw, norm_raw=info["norm_before"], fallback=info["fallback"])


# =============================================================================
# DM_FB targets and loss (METHOD_MAP: fb_il.dm_targets, fb_il.dm_loss, fb_il.dm_grad, fb_il.dm_fit)
# =============================================================================
def dm_start_times(horizon: int, K: int, include_t0: bool = True) -> np.ndarray:
    """``T_K = {0, ..., H-K}`` (D4-D5), or ``{1, ..., H-K}`` without ``t = 0``."""
    if K < 1 or K > horizon:
        raise ValueError("1 <= K <= H is required so that psi^(K) is observed")
    times = np.arange(0 if include_t0 else 1, horizon - K + 1)
    if times.size == 0:
        raise ValueError("no start time left")
    return times


@dataclass
class DMData:
    start_obs: np.ndarray        # (n,) or (n, 2): s^i_t for (i, t) in demonstrations x T_K
    start_enc: torch.Tensor      # encoded start states on the model's device
    psi: torch.Tensor            # (n, d) truncated targets, frozen
    demo_index: np.ndarray       # (n,) i
    time_index: np.ndarray       # (n,) t
    K: int


@torch.no_grad()
def dm_targets(model: FBModel, demo_states: np.ndarray, K: int, include_t0: bool = True) -> DMData:
    """``psi^{(K),i}_t = sum_{k=0}^{K-1} gamma^k B(s^i_{t+k+1})`` for ``t in T_K`` (Eq. (6.2)); frozen ``B``."""
    arr = _demo_array(model, demo_states)
    n_e, horizon = arr.shape[0], arr.shape[1] - 1
    times = dm_start_times(horizon, K, include_t0)
    flat = arr.reshape((-1,) + arr.shape[2:])
    b_all = model.backward_features(flat).reshape(n_e, horizon + 1, -1)          # B(s^i_j), j = 0..H
    disc = torch.as_tensor(model.gamma ** np.arange(K, dtype=np.float64), device=b_all.device).to(b_all.dtype)
    psi = torch.stack([torch.einsum("k,ikd->id", disc, b_all[:, t + 1:t + K + 1]) for t in times], dim=1)
    demo_index = np.repeat(np.arange(n_e), times.size)
    time_index = np.tile(times, n_e)
    start_obs = arr[demo_index, time_index]
    return DMData(
        start_obs=start_obs,
        start_enc=model.encode(start_obs),
        psi=psi.reshape(n_e * times.size, -1),
        demo_index=demo_index,
        time_index=time_index,
        K=int(K),
    )


def dm_loss(
    F_sa: torch.Tensor, z: torch.Tensor, psi: torch.Tensor, G: torch.Tensor, tau: float, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``l_hat(z; G) = sum_n w_n [F(s_n,z)^T G F(s_n,z) - 2 F(s_n,z)^T psi_n]`` (Eq. (6.5); uniform weights by default).

    ``F_sa`` is ``F(s_n, a, z)`` of shape ``(n, A, d)``; ``F(s_n, z)`` averages it with
    ``pi_{z,tau}`` computed from the same ``F_sa`` and ``z`` (``fb.expected_features``),
    so the gradient flows through both the features and the policy.
    """
    Fz = expected_features(F_sa, z, tau)
    n = Fz.shape[0]
    w = torch.full((n,), 1.0 / n, dtype=Fz.dtype, device=Fz.device) if weights is None else torch.as_tensor(weights, dtype=Fz.dtype, device=Fz.device)
    quad = ((Fz @ G) * Fz).sum(dim=-1)
    cross = (Fz * psi).sum(dim=-1)
    return (w * (quad - 2.0 * cross)).sum()


def dm_objective(model: FBModel, z: torch.Tensor, data: DMData) -> torch.Tensor:
    """``l_hat^{(K)}(z; G_hat_B)`` with the frozen model; differentiable in ``z`` only."""
    F_sa = model.F_net(data.start_enc, z)
    return dm_loss(F_sa, z, data.psi, model.gram, model.tau)


def dm_grad(model: FBModel, z: np.ndarray, data: DMData) -> np.ndarray:
    """``grad_z l_hat`` by autograd through ``F(s,a,z)`` and ``pi_{z,tau}`` (Eq. (6.6)); networks, ``G_hat_B``, targets frozen."""
    _require_frozen(model)
    z_t = model.latent(z).detach().clone().requires_grad_(True)
    loss = dm_objective(model, z_t, data)
    (g,) = torch.autograd.grad(loss, [z_t])
    return g.detach().to("cpu", torch.float64).numpy()


def riemannian_gradient(z: np.ndarray, grad: np.ndarray, radius: float) -> np.ndarray:
    """``g_R = (I - z z^T / R^2) grad`` on the sphere ``Z_R`` (FBIL-5)."""
    z = np.asarray(z, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    return grad - z * (z @ grad) / radius**2


@dataclass
class DMResult:
    z: np.ndarray                      # final latent (projected), float64
    z0: np.ndarray                     # warm start (projected z_ER, or a given z0)
    er: Optional[ERResult]
    log: List[Dict[str, float]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    snapshots: Dict[int, np.ndarray] = field(default_factory=dict)   # z at the requested steps (3C: evaluation curves)

    @property
    def final_loss(self) -> float:
        return self.log[-1]["loss"]


def dm_fit(
    model: FBModel,
    demo_states: np.ndarray,
    cfg: DMConfig = DMConfig(),
    z0: Optional[np.ndarray] = None,
    fallback_rng: Optional[np.random.Generator] = None,
    snapshot_steps: Optional[Sequence[int]] = None,
) -> DMResult:
    """DM_FB: projected Adam on ``l_hat^{(K)}(z; G_hat_B)`` from the projected ``z_ER`` (Heuristic, O1/X1).

    Logs ``l_hat``, ``||grad||``, ``||g_R||``, ``||z||`` and the mean action
    entropy of ``pi_{z,tau}`` at the start states, at every iterate (``steps + 1``
    entries).  Raises if the model is not frozen or if any network parameter or
    ``G_hat_B`` changed (freeze check).  ``fallback_rng`` is used only if
    ``z_ER = 0``; the caller records its seed and stream labels.  With
    ``project = False`` nothing is projected, and the warm start is the raw ``z_ER``.
    ``snapshot_steps`` (Prompt 3C) records a float64 copy of the iterate ``z`` after
    each listed number of steps (0 is the warm start) in ``DMResult.snapshots``; the
    iterates themselves are unchanged, so a snapshot at step ``k`` equals the final
    ``z`` of a run with ``steps = k``.
    """
    _require_frozen(model)
    before = params_fingerprint(model)
    gram_before = model.gram.detach().clone()
    data = dm_targets(model, demo_states, cfg.K, cfg.include_t0)
    er = None
    if z0 is None:
        er = er_fb(model, demo_states, fallback_rng)
        z_init = er.z if cfg.project else er.z_raw          # without projection, start from the raw z_ER
    else:
        z_init = np.asarray(z0, dtype=np.float64)
        if cfg.project:
            z_init, _ = project_to_sphere(z_init, model.radius, fallback_rng)
    R = model.radius
    z = model.latent(z_init).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=cfg.lr, eps=cfg.adam_eps)
    log: List[Dict[str, float]] = []
    wanted = set(int(k) for k in (snapshot_steps or ()))
    if any(k < 0 or k > cfg.steps for k in wanted):
        raise ValueError("snapshot_steps must lie in 0..steps")
    snapshots: Dict[int, np.ndarray] = {}
    for step in range(cfg.steps + 1):
        if step in wanted:
            snapshots[step] = z.detach().to("cpu", torch.float64).numpy().copy()
        loss = dm_objective(model, z, data)
        (g,) = torch.autograd.grad(loss, [z])
        if not (torch.isfinite(loss) and bool(torch.isfinite(g).all())):
            raise FloatingPointError(f"DM loss or gradient not finite at step {step}")
        z_np = z.detach().to("cpu", torch.float64).numpy()
        g_np = g.detach().to("cpu", torch.float64).numpy()
        with torch.no_grad():
            probs = policy_probs(model.F_net(data.start_enc, z), z, model.tau)
            entropy = float(-(probs * torch.log(probs.clamp_min(1e-30))).sum(-1).mean())
        log.append({
            "step": float(step),
            "loss": float(loss.detach()),
            "grad_norm": float(np.linalg.norm(g_np)),
            "riemannian_grad_norm": float(np.linalg.norm(riemannian_gradient(z_np, g_np, R))),
            "z_norm": float(np.linalg.norm(z_np)),
            "policy_entropy": entropy,
        })
        if step == cfg.steps:
            break
        opt.zero_grad(set_to_none=True)
        z.grad = g.detach()
        opt.step()
        if cfg.project:
            with torch.no_grad():
                z.mul_(R / torch.linalg.norm(z))
    after = params_fingerprint(model)
    if not torch.equal(before, after) or not torch.equal(gram_before, model.gram):
        raise RuntimeError("frozen FB parameters or G_hat_B changed during latent adaptation")
    return DMResult(
        z=z.detach().to("cpu", torch.float64).numpy(),
        z0=np.asarray(z_init, dtype=np.float64),
        er=er,
        log=log,
        config=cfg.to_dict(),
        snapshots=snapshots,
    )
