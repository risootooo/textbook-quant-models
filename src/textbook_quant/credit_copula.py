"""Single-factor credit portfolio models: Vasicek + copula Monte Carlo.

The large homogeneous pool (Vasicek 1987/2002) loss distribution is
closed form under a Gaussian single-factor model: obligor *i* defaults
when its asset return ``A_i = sqrt(rho) * Z + sqrt(1-rho) * e_i`` falls
below ``Phi^{-1}(pd)``.  Conditional on the systematic factor ``Z`` the
default probability is

    p(Z) = Phi( (Phi^{-1}(pd) - sqrt(rho) * Z) / sqrt(1 - rho) ),

and in the infinitely granular limit the portfolio loss rate is
``LGD * p(Z)`` with ``Z ~ N(0, 1)``, which gives analytic quantiles and
(together with one numerical integral) analytic expected shortfall.

The Monte Carlo functions simulate the *finite* portfolio under a
Gaussian copula or a Student-t copula (shared chi-square mixing
variable, which creates tail dependence the Gaussian copula lacks).

Losses are expressed as a fraction of total portfolio notional with
equal notionals: ``loss = LGD * (number of defaults) / n_obligors``.

Limitations: single systematic factor, constant recovery (LGD fixed,
no recovery risk), no correlation between PD and LGD, no sector
concentration, static one-period horizon.  Parameters in the case
study are illustrative, not calibrated to any real portfolio.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy import integrate, stats

__all__ = [
    "expected_loss",
    "vasicek_loss_quantile",
    "vasicek_var",
    "vasicek_expected_shortfall",
    "vasicek_loss_distribution",
    "gaussian_copula_mc",
    "t_copula_mc",
    "portfolio_var_es",
    "unexpected_loss",
]


# ----------------------------------------------------------------------
# Closed-form (Vasicek large-pool) results
# ----------------------------------------------------------------------
def expected_loss(pd: Union[float, np.ndarray], lgd: float) -> float:
    """Expected portfolio loss rate: mean PD (x LGD).  Correlation-free."""
    return float(np.mean(np.asarray(pd, dtype=float)) * lgd)


def _conditional_pd(z: np.ndarray, pd: float, rho: float) -> np.ndarray:
    """Default probability conditional on the systematic factor z."""
    return stats.norm.cdf(
        (stats.norm.ppf(pd) - np.sqrt(rho) * z) / np.sqrt(1.0 - rho)
    )


def vasicek_loss_quantile(pd: float, rho: float, lgd: float, q: float) -> float:
    """q-quantile of the large-pool loss distribution.

    ``L_q = LGD * Phi( (Phi^{-1}(pd) + sqrt(rho) * Phi^{-1}(q))
                        / sqrt(1 - rho) )``.
    With rho = 0 the distribution is degenerate at the expected loss.
    """
    if rho == 0.0:
        return float(lgd * pd)
    return float(
        lgd
        * stats.norm.cdf(
            (stats.norm.ppf(pd) + np.sqrt(rho) * stats.norm.ppf(q))
            / np.sqrt(1.0 - rho)
        )
    )


def vasicek_var(pd: float, rho: float, lgd: float, level: float = 0.99) -> float:
    """Value-at-Risk at ``level`` (a loss quantile, not net of EL)."""
    return vasicek_loss_quantile(pd, rho, lgd, level)


def vasicek_expected_shortfall(
    pd: float, rho: float, lgd: float, level: float = 0.99
) -> float:
    """Expected loss in the tail beyond ``level`` (large-pool, analytic).

    Losses exceed the quantile exactly when ``Z <= Phi^{-1}(1-level)``,
    so ``ES = LGD / (1-level) * \\int_{-inf}^{z*} p(z) phi(z) dz``,
    evaluated by quadrature.
    """
    if rho == 0.0:
        return float(lgd * pd)
    z_star = stats.norm.ppf(1.0 - level)

    def integrand(z: float) -> float:
        return float(_conditional_pd(z, pd, rho)) * float(stats.norm.pdf(z))

    val, _ = integrate.quad(integrand, -np.inf, z_star, limit=200)
    return float(lgd * val / (1.0 - level))


def vasicek_loss_distribution(
    pd: float,
    rho: float,
    lgd: float,
    levels: tuple[float, ...] = (0.99, 0.999),
) -> dict[str, float]:
    """Bundle of large-pool risk numbers: EL, VaR and ES at ``levels``."""
    out: dict[str, float] = {"el": expected_loss(pd, lgd)}
    for lev in levels:
        tag = f"{lev * 100:g}".replace(".", "p")
        out[f"var{tag}"] = vasicek_var(pd, rho, lgd, lev)
        out[f"es{tag}"] = vasicek_expected_shortfall(pd, rho, lgd, lev)
    return out


def unexpected_loss(var: float, el: float) -> float:
    """Unexpected loss = VaR - EL (capital-style decomposition)."""
    return float(var - el)


# ----------------------------------------------------------------------
# Monte Carlo (finite portfolio)
# ----------------------------------------------------------------------
def _simulate_losses(
    n_obligors: int,
    pd_arr: np.ndarray,
    rho: float,
    lgd: float,
    n_sims: int,
    seed: int,
    nu: Optional[float],
) -> np.ndarray:
    """Shared simulation engine.  nu=None -> Gaussian copula.

    Common random numbers: for a given seed the factor and idiosyncratic
    draws are identical between the Gaussian and t variants (the t
    variant draws its mixing variable afterwards), which makes
    Gaussian-vs-t comparisons less noisy.
    """
    rng = np.random.default_rng(seed)
    threshold_norm = stats.norm.ppf(pd_arr)
    threshold_t = stats.t.ppf(pd_arr, nu) if nu is not None else None
    losses = np.empty(n_sims)
    chunk = max(1, int(2_000_000 // max(n_obligors, 1)))
    sqrt_rho, sqrt_1mrho = np.sqrt(rho), np.sqrt(1.0 - rho)
    for start in range(0, n_sims, chunk):
        m = min(chunk, n_sims - start)
        z = rng.standard_normal((m, 1))
        e = rng.standard_normal((m, n_obligors))
        a = sqrt_rho * z + sqrt_1mrho * e
        if nu is not None:
            w = rng.chisquare(nu, size=(m, 1))
            a = a / np.sqrt(w / nu)
            defaults = a < threshold_t
        else:
            defaults = a < threshold_norm
        losses[start : start + m] = lgd * defaults.mean(axis=1)
    return losses


def gaussian_copula_mc(
    n_obligors: int,
    pd: Union[float, np.ndarray],
    rho: float,
    lgd: float,
    n_sims: int = 100_000,
    seed: int = 42,
) -> np.ndarray:
    """Monte Carlo loss distribution under a Gaussian copula.

    ``pd`` may be a scalar (homogeneous pool) or an array of length
    ``n_obligors`` (heterogeneous PDs).  Returns the array of simulated
    loss rates (fraction of portfolio notional).
    """
    pd_arr = np.broadcast_to(np.asarray(pd, dtype=float), (n_obligors,)).copy()
    return _simulate_losses(n_obligors, pd_arr, rho, lgd, n_sims, seed, nu=None)


def t_copula_mc(
    n_obligors: int,
    pd: Union[float, np.ndarray],
    rho: float,
    lgd: float,
    nu: float,
    n_sims: int = 100_000,
    seed: int = 42,
) -> np.ndarray:
    """Monte Carlo loss distribution under a Student-t copula.

    Same factor structure as the Gaussian copula, but all asset returns
    share a chi-square mixing variable (df = ``nu``), producing tail
    dependence: joint extreme defaults are more likely than under the
    Gaussian copula at the same ``rho``.
    """
    if nu <= 2:
        raise ValueError("nu must exceed 2 for finite asset-return variance")
    pd_arr = np.broadcast_to(np.asarray(pd, dtype=float), (n_obligors,)).copy()
    return _simulate_losses(n_obligors, pd_arr, rho, lgd, n_sims, seed, nu=nu)


def portfolio_var_es(
    losses: np.ndarray, level: float = 0.99
) -> tuple[float, float]:
    """(VaR, ES) of a simulated loss sample at ``level``.

    VaR is the empirical quantile; ES is the mean of losses at or above
    it.  Both are Monte Carlo estimates -- their standard errors shrink
    like 1/sqrt(n_sims) and are material at the 99.9% level.
    """
    losses = np.asarray(losses, dtype=float)
    var = float(np.quantile(losses, level))
    tail = losses[losses >= var]
    es = float(tail.mean()) if tail.size else var
    return var, es
