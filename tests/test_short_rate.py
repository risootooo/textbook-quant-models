"""Tests for Model 5 — Vasicek & CIR short-rate calibration and pricing.

Run with:  python -m pytest tests/test_short_rate.py -q
"""
import numpy as np
import pytest

from textbook_quant.short_rate import (
    cir_bond_price,
    cir_bond_yield,
    cir_mle,
    cir_simulate,
    vasicek_bond_price,
    vasicek_bond_yield,
    vasicek_mle,
    vasicek_simulate,
)

DT = 1.0 / 252.0


# ---------------------------------------------------------------------------
# Parameter recovery (the core honesty test: simulate truth, recover it)
# ---------------------------------------------------------------------------

def test_vasicek_parameter_recovery():
    """Simulate Vasicek with known parameters; exact MLE must recover
    theta closely and kappa / sigma inside sane bands (kappa is the
    weakly identified parameter for mean-reverting diffusions, hence
    the wider band)."""
    kappa, theta, sigma = 0.85, 0.045, 0.011
    path = vasicek_simulate(0.03, kappa, theta, sigma,
                            n_steps=20_000, dt=DT, seed=7)[0]
    fit = vasicek_mle(path, DT)
    assert fit.model == "vasicek"
    assert abs(fit.theta - theta) < 0.005
    assert 0.5 * kappa < fit.kappa < 1.5 * kappa
    assert 0.85 * sigma < fit.sigma < 1.15 * sigma
    assert np.isfinite(fit.loglik)
    assert fit.half_life_years == pytest.approx(np.log(2) / fit.kappa)


def test_vasicek_parameter_recovery_slow_reversion():
    """A second regime: slow mean reversion (kappa = 0.35) and higher
    vol — recovery must still hold."""
    kappa, theta, sigma = 0.35, 0.030, 0.020
    path = vasicek_simulate(0.05, kappa, theta, sigma,
                            n_steps=20_000, dt=DT, seed=11)[0]
    fit = vasicek_mle(path, DT)
    assert abs(fit.theta - theta) < 0.006
    assert 0.5 * kappa < fit.kappa < 1.8 * kappa
    assert 0.85 * sigma < fit.sigma < 1.15 * sigma


def test_cir_parameter_recovery():
    """CIR recovery from an *exact* noncentral-chi-square simulation,
    so the check is not contaminated by the Euler scheme used in
    cir_simulate."""
    kappa, theta, sigma = 0.60, 0.050, 0.120
    rng = np.random.default_rng(3)
    n = 4_000
    c = sigma ** 2 * (1.0 - np.exp(-kappa * DT)) / (4.0 * kappa)
    df = 4.0 * kappa * theta / sigma ** 2
    path = np.empty(n + 1)
    path[0] = 0.04
    for i in range(n):
        nc = path[i] * np.exp(-kappa * DT) / c
        path[i + 1] = c * rng.noncentral_chisquare(df, nc)
    fit = cir_mle(path, DT)
    assert fit.model == "cir"
    assert abs(fit.theta - theta) < 0.010
    assert 0.4 * kappa < fit.kappa < 1.8 * kappa
    assert 0.75 * sigma < fit.sigma < 1.30 * sigma
    # True parameters satisfy Feller (2*.6*.05 = .06 >= .0144); the fit
    # should agree.
    assert fit.feller_holds


def test_cir_mle_rejects_nonpositive_rates():
    r = np.full(100, 0.03)
    r[50] = 0.0
    with pytest.raises(ValueError):
        cir_mle(r, DT)


# ---------------------------------------------------------------------------
# Bond prices
# ---------------------------------------------------------------------------

def test_bond_price_at_zero_maturity_is_one():
    for price_fn in (vasicek_bond_price, cir_bond_price):
        assert price_fn(0.04, 0.5, 0.045, 0.10, 0.0) == 1.0
        assert price_fn(0.0, 1.2, 0.02, 0.30, 0.0) == 1.0


def test_bond_prices_decrease_with_maturity_and_rate():
    taus = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
    for price_fn in (vasicek_bond_price, cir_bond_price):
        prices = [price_fn(0.04, 0.5, 0.045, 0.08, t) for t in taus]
        assert all(0.0 < p <= 1.0 for p in prices)
        assert all(b < a for a, b in zip(prices, prices[1:]))
        # A higher short rate must mean a lower bond price.
        assert price_fn(0.09, 0.5, 0.045, 0.08, 5.0) < \
            price_fn(0.02, 0.5, 0.045, 0.08, 5.0)


def test_vasicek_yields_increase_with_theta():
    """Holding today's rate fixed, a higher long-run mean theta must
    lift long-maturity yields."""
    y_low = vasicek_bond_yield(0.03, 0.5, 0.025, 0.010, 10.0)
    y_high = vasicek_bond_yield(0.03, 0.5, 0.060, 0.010, 10.0)
    assert y_high > y_low


def test_vasicek_yield_curve_slopes_toward_theta():
    """If r < theta the curve rises with maturity; if r > theta it
    falls — yields are pulled toward the long-run level."""
    kappa, theta, sigma = 0.5, 0.05, 0.010
    up = [vasicek_bond_yield(0.01, kappa, theta, sigma, t)
          for t in (1.0, 5.0, 10.0)]
    assert up[0] < up[1] < up[2]
    down = [vasicek_bond_yield(0.10, kappa, theta, sigma, t)
            for t in (1.0, 5.0, 10.0)]
    assert down[0] > down[1] > down[2]


def test_long_yield_converges_to_model_long_run_level():
    """At very long maturity the Vasicek yield tends to
    theta - sigma^2/(2 kappa^2) and the CIR yield to its affine
    long-run level; check the 100y yield is close, not equal."""
    kappa, theta, sigma = 0.6, 0.045, 0.012
    long_run_v = theta - sigma ** 2 / (2.0 * kappa ** 2)
    y100 = vasicek_bond_yield(0.03, kappa, theta, sigma, 100.0)
    assert abs(y100 - long_run_v) < 0.002

    k2, t2, s2 = 0.5, 0.05, 0.10
    h = np.sqrt(k2 ** 2 + 2.0 * s2 ** 2)
    long_run_c = 2.0 * k2 * t2 / (k2 + h)
    y100c = cir_bond_yield(0.03, k2, t2, s2, 100.0)
    assert abs(y100c - long_run_c) < 0.002


def test_bond_yield_rejects_nonpositive_tau():
    with pytest.raises(ValueError):
        vasicek_bond_yield(0.03, 0.5, 0.04, 0.01, 0.0)
    with pytest.raises(ValueError):
        cir_bond_yield(0.03, 0.5, 0.04, 0.10, -1.0)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def test_cir_simulated_paths_stay_nonnegative():
    """Even with Feller badly violated (sigma large), the
    full-truncation scheme must never produce a negative rate."""
    paths = cir_simulate(0.02, 0.30, 0.04, 0.45,
                         n_steps=5_000, dt=DT, n_paths=50, seed=5)
    assert paths.shape == (50, 5_001)
    assert paths.min() >= 0.0
    assert np.all(paths[:, 0] == 0.02)


def test_cir_simulation_mean_reverts_toward_theta():
    paths = cir_simulate(0.001, 0.8, 0.05, 0.10,
                         n_steps=2_520, dt=DT, n_paths=200, seed=9)
    terminal_mean = paths[:, -1].mean()
    assert 0.03 < terminal_mean < 0.07  # theta = 0.05


def test_vasicek_simulation_shape_seed_and_level():
    a = vasicek_simulate(0.03, 0.85, 0.045, 0.011,
                         n_steps=2_520, dt=DT, n_paths=100, seed=21)
    b = vasicek_simulate(0.03, 0.85, 0.045, 0.011,
                         n_steps=2_520, dt=DT, n_paths=100, seed=21)
    assert a.shape == (100, 2_521)
    assert np.array_equal(a, b)  # seeded reproducibility
    assert abs(a[:, -1].mean() - 0.045) < 0.01  # mean-reverts to theta
