"""Tests for the Hamilton regime-switching HMM."""
import numpy as np
import pytest

from textbook_quant.regimes import (
    expected_durations,
    filtered_probs,
    fit,
    simulate,
    smoothed_probs,
    viterbi_path,
)

TRUE_MU = np.array([0.0005, -0.001])
TRUE_SIGMA = np.array([0.006, 0.020])
TRUE_P = np.array([[0.97, 0.03], [0.05, 0.95]])


@pytest.fixture(scope="module")
def fitted():
    returns, states = simulate(3000, TRUE_MU, TRUE_SIGMA, TRUE_P, seed=11)
    return returns, states, fit(returns)


def test_recovers_regime_parameters(fitted):
    _, _, params = fitted
    assert params["sigma"][1] > params["sigma"][0]
    assert params["sigma"][0] == pytest.approx(TRUE_SIGMA[0], abs=0.002)
    assert params["sigma"][1] == pytest.approx(TRUE_SIGMA[1], abs=0.005)
    assert params["mu"][0] == pytest.approx(TRUE_MU[0], abs=0.0015)
    assert params["mu"][1] == pytest.approx(TRUE_MU[1], abs=0.003)


def test_transition_diagonal_persistent(fitted):
    _, _, params = fitted
    assert params["P"][0, 0] > 0.8
    assert params["P"][1, 1] > 0.8
    assert params["P"].sum(axis=1) == pytest.approx(np.ones(2))


def test_loglik_non_decreasing(fitted):
    _, _, params = fitted
    hist = np.array(params["loglik_history"])
    assert len(hist) >= 2
    assert np.all(np.diff(hist) >= -1e-6)


def test_filtered_probs_sum_to_one(fitted):
    returns, _, params = fitted
    filt = filtered_probs(returns, params)
    assert filt.shape == (len(returns), 2)
    assert filt.sum(axis=1) == pytest.approx(np.ones(len(returns)), abs=1e-9)
    assert np.all((filt >= 0) & (filt <= 1))


def test_smoothed_and_viterbi_agree_broadly(fitted):
    returns, states, params = fitted
    smooth = smoothed_probs(returns, params)
    assert smooth.sum(axis=1) == pytest.approx(np.ones(len(returns)), abs=1e-6)
    path = viterbi_path(returns, params)
    # The most likely path should match the true states far above chance.
    agreement = (path == states).mean()
    assert agreement > 0.75


def test_expected_durations(fitted):
    _, _, params = fitted
    durs = expected_durations(params)
    assert durs[0] == pytest.approx(1.0 / (1.0 - params["P"][0, 0]))
    assert np.all(durs > 1.0)
