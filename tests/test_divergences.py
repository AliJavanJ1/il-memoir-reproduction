"""f-divergences: generator table, direction, zero masses, smoothing and derivative factors."""
from __future__ import annotations

import numpy as np
import pytest

from il_memoir import divergences as dv
from il_memoir.common import make_rng

from _helpers import softmax_rows

NAMES = ("js", "kl_dq", "kl_qd")
LOG2 = np.log(2.0)


def kl(p, q):
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def entropy(p):
    m = p > 0
    return float(-np.sum(p[m] * np.log(p[m])))


def test_generator_table_values_and_limits():
    for name in NAMES:
        assert abs(dv.f(name, 1.0)) < 1e-15
        assert np.isclose(dv.f(name, 1e-14), dv.f_at_zero(name), atol=1e-12) or np.isinf(dv.f_at_zero(name))
        assert np.isclose(float(dv.f(name, 0.0)), dv.f_at_zero(name)) or np.isinf(dv.f_at_zero(name))
    assert np.isclose(dv.fprime("js", 1.0), 0.0) and np.isclose(dv.fprime("kl_dq", 1.0), 1.0)
    assert np.isclose(dv.fprime("kl_qd", 1.0), -1.0)
    assert dv.f_at_zero("js") == 0.5 * LOG2 and dv.f_at_zero("kl_dq") == 0.0 and np.isinf(dv.f_at_zero("kl_qd"))
    big = 1e12
    assert np.isclose(dv.f("js", big) / big, 0.5 * LOG2, atol=1e-9)        # f'(inf) for JS
    assert abs(dv.f("kl_qd", big) / big) < 1e-9                             # f'(inf) = 0 for KL(q||d)
    assert dv.fprime_at_infinity("js") == 0.5 * LOG2 and dv.fprime_at_infinity("kl_qd") == 0.0
    assert np.isinf(dv.fprime_at_infinity("kl_dq"))
    # the JS generator printed in FPG Table 1 / FIRL Table 2 is twice ours (SOURCE_AUDIT D3)
    u = np.linspace(0.1, 5, 7)
    printed = u * np.log(u) - (1 + u) * np.log((1 + u) / 2)
    assert np.allclose(printed, 2 * dv.f("js", u))


def test_fprime_matches_finite_differences_and_convexity():
    u = np.geomspace(1e-2, 50, 60)
    h = 1e-6 * u
    for name in NAMES:
        fd = (dv.f(name, u + h) - dv.f(name, u - h)) / (2 * h)
        assert np.allclose(fd, dv.fprime(name, u), rtol=1e-6, atol=1e-9)
        second = dv.f(name, u + h) - 2 * dv.f(name, u) + dv.f(name, u - h)
        assert np.all(second > 0)


def test_divergence_formulas_direction_and_perspective():
    rng = make_rng(20261005, "div", "formulas")
    for _ in range(20):
        p, q = rng.dirichlet(np.ones(8)), rng.dirichlet(np.ones(8))
        m = 0.5 * (p + q)
        assert np.isclose(dv.divergence(p, q, "kl_dq"), kl(p, q))
        assert np.isclose(dv.divergence(p, q, "kl_qd"), kl(q, p))
        jsv = dv.divergence(p, q, "js")
        assert np.isclose(jsv, 0.5 * kl(p, m) + 0.5 * kl(q, m))
        assert np.isclose(jsv, entropy(m) - 0.5 * entropy(p) - 0.5 * entropy(q))
        assert np.isclose(jsv, float(np.sum(q * dv.f("js", p / q))))     # generic sum q f(p/q)
        assert np.isclose(jsv, dv.divergence(q, p, "js")) and 0 <= jsv <= LOG2
        for name in NAMES:                                              # DIV-1: D_f(p||q) = D_f°(q||p)
            assert np.isclose(dv.divergence(p, q, name), dv.divergence(q, p, dv.PERSPECTIVE[name]))
            assert abs(dv.divergence(p, p, name)) < 1e-14


def test_zero_masses_and_orientation():
    p = np.array([1.0, 0.0])
    q = np.array([0.5, 0.5])
    assert np.isclose(dv.divergence(p, q, "kl_dq"), LOG2)               # KL(d||q) with d = p
    assert np.isinf(dv.divergence(p, q, "kl_qd"))                       # KL(q||d): q charges a d-null cell
    m = 0.5 * (p + q)
    assert np.isclose(dv.divergence(p, q, "js"), 0.5 * kl(p, m) + 0.5 * kl(q, m))
    assert np.isclose(dv.js(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.5, 0.5])), LOG2)  # disjoint supports
    assert np.isinf(dv.divergence(q, p, "kl_dq")) and np.isclose(dv.divergence(q, p, "kl_qd"), LOG2)
    with pytest.raises(ValueError):
        dv.divergence(np.array([0.6, 0.6]), q, "js")
    with pytest.raises(ValueError):
        dv.divergence(p, q, "chi2")


def test_smoothing_properties():
    rng = make_rng(20261005, "div", "smooth")
    eps = 0.01
    ref = rng.dirichlet(np.ones(10)) * 0.5 + 0.05
    ref /= ref.sum()
    p = np.zeros(10)
    p[:3] = rng.dirichlet(np.ones(3))
    q = np.zeros(10)
    q[5:] = rng.dirichlet(np.ones(5))
    ps = dv.smooth(p, eps, ref)
    assert np.isclose(ps.sum(), 1.0) and ps.min() >= eps * ref.min() - 1e-16
    for name in NAMES:
        assert np.isfinite(dv.smoothed_divergence(p, q, name, eps, ref))     # DIV-3(a)
    for _ in range(10):                                                    # DIV-3(d)
        pp, qq = rng.dirichlet(np.ones(10)), rng.dirichlet(np.ones(10))
        for name in NAMES:
            assert dv.smoothed_divergence(pp, qq, name, eps, ref) <= (1 - eps) * dv.divergence(pp, qq, name) + 1e-14
            assert abs(dv.smoothed_divergence(pp, qq, name, 1e-9, ref) - dv.divergence(pp, qq, name)) < 1e-6
    assert dv.smoothed_divergence(p, q, "js", eps, ref) <= (1 - eps) * dv.js(p, q) + 1e-14
    with pytest.raises(ValueError):
        dv.smooth(p, eps, np.zeros(10))


@pytest.mark.parametrize("name", NAMES)
def test_smoothed_objective_gradient_matches_finite_differences(name):
    """grad_phi D_f((d_phi)_eps || q_eps) = sum_x (1-eps) f'(.)(x) grad d_phi(x), d_phi = softmax(phi) (DIV-3(b))."""
    rng = make_rng(20261005, "div", "grad", name)
    n, eps = 7, 0.05
    ref = np.full(n, 1.0 / n)
    q = np.zeros(n)
    q[:4] = rng.dirichlet(np.ones(4))                    # target with zeros
    phi = rng.normal(size=n)

    def J(ph):
        return dv.smoothed_divergence(softmax_rows(ph), q, name, eps, ref)

    d = softmax_rows(phi)
    jac = np.diag(d) - np.outer(d, d)                    # d d(x) / d phi(y)
    grad = dv.smoothed_signal(d, q, name, eps, ref) @ jac
    h = 1e-6
    fd = np.array([(J(phi + h * e) - J(phi - h * e)) / (2 * h) for e in np.eye(n)])
    assert np.allclose(grad, fd, rtol=1e-6, atol=1e-9)
    wrong = (dv.smoothed_signal(d, q, name, eps, ref) / (1 - eps)) @ jac     # without the (1 - eps) factor
    assert np.max(np.abs(wrong - fd)) > 1e-3 * np.max(np.abs(fd))
    step = J(phi - 1e-3 * grad)                          # descent lowers the objective
    assert step < J(phi)


def test_edge_cases_infinite_ratio_zero_eps_and_tiny_masses():
    # f'(inf) is the boundary value used where q = 0 (theory Lemma 2.1)
    for name in NAMES:
        assert dv.fprime(name, np.inf) == dv.fprime_at_infinity(name)
        assert dv.fprime(name, 0.0) == -np.inf
    with pytest.raises(ValueError, match="0/0"):
        dv.fprime("js", np.array([np.nan]))              # what 0/0 produces for a cell with p = q = 0
    with pytest.raises(ValueError):
        dv.smoothed_signal(np.array([0.5, 0.5]), np.array([1.0, 0.0]), "js", 0.0, np.array([0.5, 0.5]))
    # masses far below the float64 normal range must not overflow the ratio p/q
    p = np.array([0.5, 0.5])
    q = np.array([1.0 - 1e-310, 1e-310])
    exact_kl_dq = 0.5 * np.log(0.5 / (1.0 - 1e-310)) + 0.5 * (np.log(0.5) - np.log(1e-310))
    assert np.isclose(dv.divergence(p, q, "kl_dq"), exact_kl_dq)
    assert np.isclose(dv.divergence(p, q, "kl_qd"), (1.0 - 1e-310) * np.log((1.0 - 1e-310) / 0.5), atol=1e-12)
    assert 0.0 <= dv.divergence(p, q, "js") <= LOG2


def test_divergence_from_densities_matches_masses():
    rng = make_rng(20261005, "div", "dens")
    areas = rng.uniform(0.5, 2.0, 12)
    areas /= areas.sum()
    pm, qm = rng.dirichlet(np.ones(12)), rng.dirichlet(np.ones(12))
    for name in NAMES:
        got = dv.divergence_from_densities(pm / areas, qm / areas, areas, name)
        assert np.isclose(got, dv.divergence(pm, qm, name))
