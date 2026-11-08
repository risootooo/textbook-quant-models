"""Tests for the Merton jump-diffusion pricer."""
import numpy as np
import pytest

from textbook_quant.jump_diffusion import (
    bs_call_price,
    bs_put_price,
    calibrate_smile,
    implied_vol,
    merton_call_price,
    merton_put_price,
    simulate_paths,
)

S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.20


def test_lambda_zero_reduces_to_black_scholes():
    for strike in (80.0, 100.0, 120.0):
        jd = merton_call_price(S, strike, T, R, SIG, lam=0.0, mu_j=-0.1, sigma_j=0.15)
        bs = bs_call_price(S, strike, T, R, SIG)
        assert jd == pytest.approx(bs, abs=1e-8)


def test_put_call_parity():
    call = merton_call_price(S, K, T, R, 0.15, lam=2.0, mu_j=-0.08, sigma_j=0.12)
    put = merton_put_price(S, K, T, R, 0.15, lam=2.0, mu_j=-0.08, sigma_j=0.12)
    assert call - put == pytest.approx(S - K * np.exp(-R * T), abs=1e-8)


def test_negative_jumps_raise_put_and_otm_skew():
    # Downward jumps must make downside protection dearer than under BS.
    put_jd = merton_put_price(S, 85.0, T, R, SIG, lam=3.0, mu_j=-0.10, sigma_j=0.10)
    put_bs = bs_put_price(S, 85.0, T, R, SIG)
    assert put_jd > put_bs


def test_monte_carlo_within_3_standard_errors():
    params = dict(sigma=0.15, lam=2.0, mu_j=-0.08, sigma_j=0.12)
    closed = merton_call_price(S, K, T, R, **params)
    paths = simulate_paths(S, T, R, n_paths=200_000, seed=123, **params)
    terminal = paths[:, -1]
    payoff = np.maximum(terminal - K, 0.0) * np.exp(-R * T)
    mc = payoff.mean()
    se = payoff.std(ddof=1) / np.sqrt(len(payoff))
    assert abs(mc - closed) < 3.0 * se


def test_implied_vol_roundtrip():
    px = merton_call_price(S, K, T, R, 0.15, lam=1.5, mu_j=-0.05, sigma_j=0.10)
    iv = implied_vol(px, S, K, T, R, "call")
    assert bs_call_price(S, K, T, R, iv) == pytest.approx(px, rel=1e-8)


def test_calibration_recovers_known_parameters():
    true = np.array([0.14, 1.8, -0.09, 0.11])  # sigma, lam, mu_j, sigma_j
    strikes = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])
    types = np.array(["put", "put", "put", "call", "call", "call", "call"])
    from textbook_quant.jump_diffusion import model_prices

    prices = model_prices(true, strikes, types, S, T, R)
    fitted = calibrate_smile(strikes, prices, types, S, T, R, x0=(0.12, 1.0, -0.05, 0.10))

    fitted_prices = model_prices(fitted, strikes, types, S, T, R)
    mkt_iv = np.array(
        [implied_vol(p, S, float(k), T, R, t) for p, k, t in zip(prices, strikes, types)]
    )
    fit_iv = np.array(
        [
            implied_vol(p, S, float(k), T, R, t)
            for p, k, t in zip(fitted_prices, strikes, types)
        ]
    )
    rmse_vol_pts = float(np.sqrt(np.mean((fit_iv - mkt_iv) ** 2)) * 100.0)
    assert rmse_vol_pts < 0.5
    # Diffusion vol and jump direction should be close to the truth.
    assert fitted[0] == pytest.approx(true[0], abs=0.03)
    assert fitted[2] < 0  # negative mean jump recovered
