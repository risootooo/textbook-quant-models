"""Tests for textbook_quant.garch."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textbook_quant.garch import (  # noqa: E402
    diebold_mariano,
    egarch_filter,
    egarch_fit,
    forecast,
    gjr_filter,
    gjr_fit,
    garch_filter,
    garch_fit,
    qlike,
    realized_vol,
)


def test_garch_filter_hand_computed():
    """5-point recursion checked against hand arithmetic.

    omega=0.05, alpha=0.10, beta=0.80, h0=1.0, r = [.5, -1, .25, 1.5, -.75]
    h1 = .05 + .10*.25   + .80*1.0    = 0.875
    h2 = .05 + .10*1.0   + .80*.875   = 0.85
    h3 = .05 + .10*.0625 + .80*.85    = 0.73625
    h4 = .05 + .10*2.25  + .80*.73625 = 0.864
    """
    r = np.array([0.5, -1.0, 0.25, 1.5, -0.75])
    h = garch_filter(r, omega=0.05, alpha=0.10, beta=0.80, h0=1.0)
    expected = np.array([1.0, 0.875, 0.85, 0.73625, 0.864])
    np.testing.assert_allclose(h, expected, rtol=0, atol=1e-12)


def test_gjr_filter_hand_computed_leverage():
    """Negative shocks must load alpha+gamma, positive shocks alpha."""
    r = np.array([-2.0, 2.0, 0.0])
    h = gjr_filter(r, omega=0.1, alpha=0.05, gamma=0.10, beta=0.80, h0=1.0)
    # h1 uses r0=-2 (negative): .1 + .15*4 + .8*1 = 1.5
    # h2 uses r1=+2 (positive): .1 + .05*4 + .8*1.5 = 1.5
    np.testing.assert_allclose(h, [1.0, 1.5, 1.5], atol=1e-12)


def test_egarch_filter_positive_and_leverage_sign():
    """A negative shock must raise EGARCH variance more than a +shock
    of the same size when gamma < 0."""
    params = dict(omega=0.0, alpha=0.15, gamma=-0.08, beta=0.9)
    h_neg = egarch_filter(np.array([-2.0, 0.0]), h0=1.0, **params)
    h_pos = egarch_filter(np.array([2.0, 0.0]), h0=1.0, **params)
    assert np.all(h_neg > 0) and np.all(h_pos > 0)
    assert h_neg[1] > h_pos[1]


def _simulate_garch(n=4000, omega=0.02, alpha=0.08, beta=0.90, seed=7):
    rng = np.random.default_rng(seed)
    h = np.empty(n)
    r = np.empty(n)
    h[0] = omega / (1 - alpha - beta)
    r[0] = rng.normal(0, np.sqrt(h[0]))
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = rng.normal(0, np.sqrt(h[t]))
    return r


def test_garch_fit_recovers_parameters():
    """MLE on seeded synthetic GARCH data recovers alpha/beta."""
    r = _simulate_garch()
    fit = garch_fit(r)
    assert abs(fit.params["alpha"] - 0.08) < 0.05
    assert abs(fit.params["beta"] - 0.90) < 0.08
    assert 0.8 < fit.persistence < 1.05
    assert fit.unconditional_var is not None
    assert abs(fit.unconditional_var - 1.0) < 0.6  # true value 0.02/0.02


def test_gjr_and_egarch_fit_run_and_are_positive():
    r = _simulate_garch(n=2500, seed=11)
    for fit_fn, model in ((gjr_fit, "gjr"), (egarch_fit, "egarch")):
        fit = fit_fn(r)
        assert fit.model == model
        assert np.isfinite(fit.loglik)
        fc = forecast(fit.params, model, fit.last_var, fit.last_ret, horizon=10)
        assert np.all(fc > 0)


def test_garch_forecast_mean_reverts_to_unconditional():
    """With last_var far above the unconditional variance, the forecast
    path must decay monotonically toward it (and vice versa)."""
    params = {"omega": 0.02, "alpha": 0.08, "beta": 0.90}
    hbar = 0.02 / (1 - 0.98)  # = 1.0
    fc = forecast(params, "garch", last_var=4.0, last_ret=0.0, horizon=200)
    assert np.all(fc > 0)
    assert np.all(np.diff(fc) < 0)  # declining toward hbar from above
    assert abs(fc[-1] - hbar) < abs(fc[0] - hbar)
    # 0.98^199 ~ 0.018, so after 200 steps the gap is nearly closed.
    assert abs(fc[-1] - hbar) < 0.1


def test_forecast_one_step_matches_filter_recursion():
    r = np.array([0.3, -0.7, 1.2])
    params = {"omega": 0.05, "alpha": 0.1, "beta": 0.8}
    h = garch_filter(np.append(r, 0.0), h0=0.9, **params)
    one = forecast(params, "garch", last_var=h[2], last_ret=r[2], horizon=1)
    assert one[0] == pytest.approx(h[3], rel=1e-12)


def test_realized_vol_known_series():
    """Constant-variance returns: RV should sit near the true vol."""
    rng = np.random.default_rng(3)
    rets = rng.normal(0, 0.01, 1000)
    rv = realized_vol(rets, window=21)
    # A 21-observation rolling window first becomes valid at index 20.
    assert np.isnan(rv[:20]).all() and not np.isnan(rv[20])
    assert abs(np.nanmean(rv) - 0.01 * np.sqrt(252)) < 0.004


def test_qlike_prefers_true_variance():
    rng = np.random.default_rng(5)
    r2 = rng.normal(0, 1.0, 500) ** 2  # true variance = 1
    good = qlike(r2, np.full(500, 1.0))
    bad = qlike(r2, np.full(500, 2.5))
    assert good.mean() < bad.mean()


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(9)
    loss_good = rng.gamma(2.0, 0.4, 400)
    loss_bad = loss_good + 0.3 + rng.normal(0, 0.05, 400)
    dm, p = diebold_mariano(loss_good, loss_bad)
    assert dm < 0 and p < 0.01
    dm2, p2 = diebold_mariano(loss_good, loss_good + rng.normal(0, 0.01, 400))
    assert p2 > 0.05  # no spurious significance for a near-tie
