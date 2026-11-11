"""f-divergences: explicit generators, orientation, smoothing and the f-PG signal factor.

Convention (NOTATION section 6, theory Def. 2.2): ``D_f(p || q) = sum_x q(x) f(p(x)/q(x))
+ f'(inf) p({q = 0})``, with the **learner first**: ``J(phi) = D_f(d_phi || q)``.
Logarithms are natural; values are in nats.

Generators (theory Prop. 2.4), always named by their arguments
---------------------------------------------------------------
=========  ==============================================  ==================  ========  =========
name       f(u)                                            f'(u)               f(0)      f'(inf)
=========  ==============================================  ==================  ========  =========
``js``     1/2 [u log u - (1+u) log((1+u)/2)]  (JS(d, q))  1/2 log(2u/(1+u))   log(2)/2  log(2)/2
``kl_dq``  u log u                             KL(d || q)  1 + log u           0         +inf
``kl_qd``  -log u                              KL(q || d)  -1/u                +inf      0
=========  ==============================================  ==================  ========  =========

``kl_dq`` is FPG's "fkl" and ``kl_qd`` FPG's "rkl"; FIRL uses the opposite
names (SOURCE_AUDIT D2).  The JS generator printed in FPG Table 1 and FIRL
Table 2 is twice ``f_js`` (SOURCE_AUDIT D3); ``0 <= JS <= log 2`` here.

Smoothing (DIV-3, NOTATION section 4)
-------------------------------------
``smooth(p, eps, ref) = (1 - eps) p + eps ref`` with a strictly positive
reference: probability **masses** are mixed with masses (uniform free cells;
free-area-weighted histogram cells) and **densities** with densities
(``1 / A_free`` on the continuous free space).  With both arguments smoothed,
the derivative of ``D_f(p_eps || q_eps)`` in ``p`` carries the factor
``(1 - eps)``; :func:`smoothed_signal` returns ``(1 - eps) f'(p_eps / q_eps)``.
"""
from __future__ import annotations

from typing import Union

import numpy as np

GENERATORS = ("js", "kl_dq", "kl_qd")
LOG2 = float(np.log(2.0))
DESCRIPTIONS = {
    "js": "JS(d, q) with half weights; f(u) = 1/2 [u log u - (1+u) log((1+u)/2)]",
    "kl_dq": "KL(d || q), learner first; f(u) = u log u (FPG 'fkl')",
    "kl_qd": "KL(q || d), target first; f(u) = -log u (FPG 'rkl')",
}
PERSPECTIVE = {"js": "js", "kl_dq": "kl_qd", "kl_qd": "kl_dq"}  # f -> f°, D_f(p||q) = D_f°(q||p) (DIV-1)

ArrayLike = Union[float, np.ndarray]


def _name(name: str) -> str:
    if name not in GENERATORS:
        raise ValueError(f"unknown generator {name!r}; use one of {GENERATORS}")
    return name


def _xlogx(u: np.ndarray) -> np.ndarray:
    """``u log u`` with the convention ``0 log 0 = 0``."""
    out = np.zeros_like(u)
    pos = u > 0
    out[pos] = u[pos] * np.log(u[pos])
    return out


def _check_u(u: ArrayLike, what: str) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    if np.any(np.isnan(u)):
        raise ValueError(f"{what}: u is NaN (a 0/0 ratio: cells with p = q = 0 must be removed or smoothed first)")
    if np.any(u < 0):
        raise ValueError(f"{what} is defined for u >= 0")
    return u


def f(name: str, u: ArrayLike) -> np.ndarray:
    """Generator ``f(u)`` for finite ``u >= 0`` (``u = 0`` gives the limit ``f(0)``)."""
    name = _name(name)
    u = _check_u(u, "f")
    if np.any(np.isinf(u)):
        raise ValueError("f: u = inf is not a value of f; use fprime_at_infinity for the q = 0 boundary term")
    if name == "kl_dq":
        return _xlogx(u)
    if name == "kl_qd":
        with np.errstate(divide="ignore"):
            return -np.log(u)
    return 0.5 * (_xlogx(u) - (1.0 + u) * np.log((1.0 + u) / 2.0))


def fprime(name: str, u: ArrayLike) -> np.ndarray:
    """Derivative ``f'(u)`` for ``u > 0``; ``u = 0`` gives ``-inf`` and ``u = inf`` gives ``f'(inf)``."""
    name = _name(name)
    u = _check_u(u, "f'")
    with np.errstate(divide="ignore", invalid="ignore"):
        if name == "kl_dq":
            out = 1.0 + np.log(u)
        elif name == "kl_qd":
            out = -1.0 / u
        else:
            out = 0.5 * np.log(2.0 * u / (1.0 + u))
    return np.where(np.isinf(u), fprime_at_infinity(name), out)


def f_at_zero(name: str) -> float:
    """``f(0) = lim_{u -> 0} f(u)``."""
    return {"js": 0.5 * LOG2, "kl_dq": 0.0, "kl_qd": np.inf}[_name(name)]


def fprime_at_infinity(name: str) -> float:
    """``f'(inf) = lim_{u -> inf} f(u) / u``."""
    return {"js": 0.5 * LOG2, "kl_dq": np.inf, "kl_qd": 0.0}[_name(name)]


def check_probability(p: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """Validate a vector (or array) of probability masses that sums to one."""
    p = np.asarray(p, dtype=np.float64)
    if not np.all(np.isfinite(p)) or np.any(p < -atol):
        raise ValueError("probability masses must be finite and non-negative")
    if abs(float(p.sum()) - 1.0) > atol:
        raise ValueError(f"probability masses must sum to one (sum = {float(p.sum())!r})")
    return np.clip(p, 0.0, None)


def _terms(p: np.ndarray, q: np.ndarray, name: str) -> np.ndarray:
    """Pointwise ``q f(p/q)`` with the zero conventions of theory Def. 2.2 (``f'(inf) p`` where ``q = 0``).

    Computed without the ratio ``p/q``: JS by ``1/2 [p log(2p/(p+q)) + q log(2q/(p+q))]``
    (Prop. 2.4(a)), the KL generators by log differences.  The term is positively
    homogeneous of degree one in ``(p, q)``, so it applies to masses and to densities.
    """
    out = np.zeros_like(p)
    if name == "js":
        m = p + q
        pos = m > 0
        pp, qq, mm = p[pos], q[pos], m[pos]
        out[pos] = 0.5 * (_xlogx(pp) - pp * np.log(mm / 2.0) + _xlogx(qq) - qq * np.log(mm / 2.0))
        return out
    a, b = (p, q) if name == "kl_dq" else (q, p)       # KL(a || b) term a log(a / b)
    both = (a > 0) & (b > 0)
    out[both] = a[both] * (np.log(a[both]) - np.log(b[both]))
    out[(a > 0) & (b == 0)] = np.inf
    return out


def divergence(p: np.ndarray, q: np.ndarray, name: str, atol: float = 1e-9) -> float:
    """``D_f(p || q)`` for probability masses on a finite set (theory Def. 2.2); may be ``+inf`` for KL."""
    name = _name(name)
    p = check_probability(p, atol).ravel()
    q = check_probability(q, atol).ravel()
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    return float(np.sum(_terms(p, q, name)))


def js(p: np.ndarray, q: np.ndarray) -> float:
    """``JS(p, q)`` in nats, ``0 <= JS <= log 2``."""
    return divergence(p, q, "js")


def smooth(p: ArrayLike, eps: float, ref: ArrayLike) -> np.ndarray:
    """Mixture ``(1 - eps) p + eps ref`` (masses with masses, densities with densities)."""
    if not 0.0 <= eps <= 1.0:
        raise ValueError("eps must lie in [0, 1]")
    ref = np.asarray(ref, dtype=np.float64)
    if np.any(ref <= 0):
        raise ValueError("the smoothing reference must be strictly positive")
    return (1.0 - eps) * np.asarray(p, dtype=np.float64) + eps * ref


def smoothed_divergence(p: np.ndarray, q: np.ndarray, name: str, eps: float, ref: np.ndarray) -> float:
    """``D_f(p_eps || q_eps)``: finite for ``eps > 0`` (DIV-3(a))."""
    return divergence(smooth(p, eps, ref), smooth(q, eps, ref), name)


def smoothed_signal(p: ArrayLike, q: ArrayLike, name: str, eps: float, ref: ArrayLike) -> np.ndarray:
    """``(1 - eps) f'(p_eps / q_eps)``: the derivative factor of ``D_f(p_eps || q_eps)`` in ``p`` (DIV-3(b)).

    With masses, ``grad_phi D_f((d_phi)_eps || q_eps) = sum_x smoothed_signal(x) grad_phi d_phi(x)``.
    With KDE densities this is the plug-in f-PG signal of theory Remark 3.11
    (an implementation approximation, not an unbiased gradient).
    """
    if not eps > 0.0:
        raise ValueError("smoothed_signal needs eps > 0 (with eps = 0 a cell with q = 0 has no finite signal)")
    ps = smooth(p, eps, ref)
    qs = smooth(q, eps, ref)
    return (1.0 - eps) * fprime(name, ps / qs)


def divergence_from_densities(
    p_vals: np.ndarray, q_vals: np.ndarray, cell_areas: np.ndarray, name: str
) -> float:
    """Quadrature of ``int q f(p/q) dx + f'(inf) int_{q=0} p dx`` from density values on grid cells.

    ``p_vals``, ``q_vals`` are density values at the cell midpoints and
    ``cell_areas`` the cell areas (a midpoint rule; an approximation of the
    integral, used for diagnostics and the common continuous estimator).
    """
    name = _name(name)
    p = np.asarray(p_vals, dtype=np.float64).ravel()
    q = np.asarray(q_vals, dtype=np.float64).ravel()
    w = np.broadcast_to(np.asarray(cell_areas, dtype=np.float64), p.shape).ravel()
    if p.shape != q.shape:
        raise ValueError("p_vals and q_vals must have the same shape")
    if np.any(p < 0) or np.any(q < 0) or np.any(w < 0) or not np.all(np.isfinite(p) & np.isfinite(q)):
        raise ValueError("densities must be finite and non-negative, areas non-negative")
    terms = _terms(p, q, name)
    pos = w > 0
    return float(np.sum(w[pos] * terms[pos]))
