"""Tests for textbook_quant.credit_copula."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textbook_quant.credit_copula import (  # noqa: E402
    expected_loss,
    gaussian_copula_mc,
    portfolio_var_es,
    t_copula_mc,
    vasicek_expected_shortfall,
    vasicek_loss_quantile,
)

PD, LGD = 0.02, 0.60


def test_mc_mean_matches_analytic_el_at_zero_correlation():
    """rho = 0: MC mean loss must equal pd * lgd within MC error."""
    losses = gaussian_copula_mc(
        n_obligors=100, pd=PD, rho=0.0, lgd=LGD, n_sims=200_000, seed=123
    )
    el = expected_loss(PD, LGD)
    se = losses.std(ddof=1) / np.sqrt(losses.size)
    assert abs(losses.mean() - el) < 4 * se


def test_vasicek_quantile_ordering():
    """VaR99.9 > VaR99 > EL whenever rho > 0."""
    el = expected_loss(PD, LGD)
    var99 = vasicek_loss_quantile(PD, 0.20, LGD, 0.99)
    var999 = vasicek_loss_quantile(PD, 0.20, LGD, 0.999)
    assert var999 > var99 > el


def test_vasicek_degenerate_at_zero_correlation():
    """rho = 0: every quantile collapses to the expected loss."""
    for q in (0.5, 0.99, 0.999):
        assert vasicek_loss_quantile(PD, 0.0, LGD, q) == pytest.approx(PD * LGD)
    assert vasicek_expected_shortfall(PD, 0.0, LGD, 0.99) == pytest.approx(PD * LGD)


def test_vasicek_es_above_var():
    es = vasicek_expected_shortfall(PD, 0.20, LGD, 0.99)
    var = vasicek_loss_quantile(PD, 0.20, LGD, 0.99)
    assert es > var


def test_mc_var_matches_vasicek_for_large_pool():
    """Granular pool (2,000 obligors): MC VaR99 near the closed form."""
    losses = gaussian_copula_mc(
        n_obligors=2000, pd=PD, rho=0.10, lgd=LGD, n_sims=100_000, seed=2024
    )
    var_mc, _ = portfolio_var_es(losses, 0.99)
    var_cf = vasicek_loss_quantile(PD, 0.10, LGD, 0.99)
    # Tolerance reflects MC quantile noise + finite-pool idiosyncratic risk.
    assert abs(var_mc - var_cf) < 0.008


def test_t_copula_has_fatter_tail_than_gaussian():
    """nu = 4 at the same rho: 99.9% tail loss must exceed Gaussian."""
    kw = dict(n_obligors=500, pd=PD, rho=0.20, lgd=LGD, n_sims=200_000, seed=99)
    g = gaussian_copula_mc(**kw)
    t = t_copula_mc(nu=4.0, **kw)
    var_g, es_g = portfolio_var_es(g, 0.999)
    var_t, es_t = portfolio_var_es(t, 0.999)
    assert var_t > var_g
    assert es_t > es_g
    # ... while the means (EL) stay essentially equal.
    assert abs(g.mean() - t.mean()) < 0.0015


def test_heterogeneous_pd_supported():
    pds = np.linspace(0.005, 0.05, 200)
    losses = gaussian_copula_mc(
        n_obligors=200, pd=pds, rho=0.1, lgd=LGD, n_sims=50_000, seed=5
    )
    assert abs(losses.mean() - expected_loss(pds, LGD)) < 0.002


def test_correlation_inflates_tail_not_mean():
    kw = dict(n_obligors=200, pd=PD, lgd=LGD, n_sims=100_000, seed=77)
    lo = gaussian_copula_mc(rho=0.0, **kw)
    hi = gaussian_copula_mc(rho=0.30, **kw)
    var_lo, _ = portfolio_var_es(lo, 0.999)
    var_hi, _ = portfolio_var_es(hi, 0.999)
    assert var_hi > 2.0 * var_lo
    assert abs(lo.mean() - hi.mean()) < 0.0015
