"""Merton (1976) jump-diffusion option pricing, from scratch.

Model
-----
Under the risk-neutral measure the log-price follows a Brownian motion with
drift plus a compound Poisson jump component::

    d ln S = (r - sigma^2/2 - lam * k) dt + sigma dW + dJ

where jumps arrive at rate ``lam`` per year, jump log-sizes are
``N(mu_j, sigma_j^2)`` and ``k = E[e^J - 1] = exp(mu_j + sigma_j^2/2) - 1``
is the compensator that keeps the discounted price a martingale.

Conditioned on exactly ``n`` jumps over the option's life, the terminal price
is lognormal with adjusted volatility and rate, so the call price is a
Poisson-weighted series of Black-Scholes prices (Merton's series solution).

Why it matters: a single-sigma Black-Scholes model cannot produce the steep
short-dated downside skew seen in index options. Negative mean jumps
(``mu_j < 0``) generate exactly that skew, which the case study in
``cases/jump_diffusion_case.py`` quantifies on a real NIFTY smile.

Limitations: jumps are i.i.d. and independent of the diffusion (no leverage
effect / vol clustering); the jump intensity is constant; calibration to one
expiry says nothing about the term structure of the smile.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

__all__ = [
    "bs_call_price",
    "bs_put_price",
    "merton_call_price",
    "merton_put_price",
    "simulate_paths",
    "implied_vol",
    "calibrate_smile",
]


# --------------------------------------------------------------------------
# Black-Scholes (the lam -> 0 limit, and the IV inversion target)
# --------------------------------------------------------------------------
def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


# --------------------------------------------------------------------------
# Merton jump-diffusion
# --------------------------------------------------------------------------
def _jump_compensator(mu_j: float, sigma_j: float) -> float:
    """k = E[e^J - 1] for J ~ N(mu_j, sigma_j^2)."""
    return float(np.exp(mu_j + 0.5 * sigma_j**2) - 1.0)


def merton_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    max_jumps: int = 100,
    tol: float = 1e-12,
) -> float:
    """Merton jump-diffusion European call price (Poisson series).

    Sums Black-Scholes prices over the number of jumps ``n`` with
    ``sigma_n = sqrt(sigma^2 + n sigma_j^2 / T)`` and
    ``r_n = r - lam k + n ln(1 + k) / T``, weighted by a Poisson
    distribution with mean ``lam (1 + k) T``. With ``lam = 0`` the series
    collapses exactly onto Black-Scholes.
    """
    if T <= 0:
        return max(S - K, 0.0)
    k = _jump_compensator(mu_j, sigma_j)
    lam_adj = lam * (1.0 + k)
    mean_n = lam_adj * T

    price = 0.0
    # Poisson weights computed recursively for numerical stability.
    weight = np.exp(-mean_n)
    for n in range(max_jumps + 1):
        if n > 0:
            weight = weight * mean_n / n
        sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)
        r_n = r - lam * k + n * np.log1p(k) / T
        price += weight * bs_call_price(S, K, T, r_n, float(sigma_n))
        if n > mean_n and weight < tol:
            break
    return float(price)


def merton_put_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
) -> float:
    """Merton jump-diffusion put via put-call parity (holds exactly here)."""
    call = merton_call_price(S, K, T, r, sigma, lam, mu_j, sigma_j)
    return float(call - S + K * np.exp(-r * T))


def simulate_paths(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    n_steps: int = 250,
    n_paths: int = 100_000,
    seed: int = 42,
) -> np.ndarray:
    """Risk-neutral price paths, shape ``(n_paths, n_steps + 1)``.

    Each step applies the exact increment: Gaussian diffusion plus a
    Poisson number of Gaussian jumps, with the compensator ``-lam * k``
    in the drift so discounted paths are martingales. Because increments
    are independent, the terminal distribution is exact (no time-stepping
    bias in the terminal law).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    k = _jump_compensator(mu_j, sigma_j)
    drift = (r - 0.5 * sigma**2 - lam * k) * dt

    z = rng.standard_normal((n_paths, n_steps))
    n_jumps = rng.poisson(lam * dt, size=(n_paths, n_steps))
    # Sum of n_jumps iid N(mu_j, sigma_j^2) variables, per step.
    jump_sum = n_jumps * mu_j + np.sqrt(n_jumps) * sigma_j * rng.standard_normal(
        (n_paths, n_steps)
    )
    log_increments = drift + sigma * np.sqrt(dt) * z + jump_sum

    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


# --------------------------------------------------------------------------
# Implied volatility and calibration
# --------------------------------------------------------------------------
def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes implied volatility of a price, via Brent's method."""
    intrinsic = (
        max(S - K * np.exp(-r * T), 0.0)
        if option_type == "call"
        else max(K * np.exp(-r * T) - S, 0.0)
    )
    if price <= intrinsic + 1e-12:
        return 0.0
    price_fn = bs_call_price if option_type == "call" else bs_put_price

    def gap(sig: float) -> float:
        return price_fn(S, K, T, r, sig) - price

    lo, hi = 1e-6, 5.0
    if gap(lo) > 0 or gap(hi) < 0:
        raise ValueError("price outside the range implied by vol bounds")
    return float(brentq(gap, lo, hi, xtol=1e-12))


def model_prices(
    params: np.ndarray,
    strikes: np.ndarray,
    types: np.ndarray,
    S: float,
    T: float,
    r: float,
) -> np.ndarray:
    """Merton prices for a vector of strikes/types given (sigma, lam, mu_j, sigma_j)."""
    sigma, lam, mu_j, sigma_j = params
    out = np.empty(len(strikes))
    for i, (K, typ) in enumerate(zip(strikes, types)):
        if typ == "call":
            out[i] = merton_call_price(S, float(K), T, r, sigma, lam, mu_j, sigma_j)
        else:
            out[i] = merton_put_price(S, float(K), T, r, sigma, lam, mu_j, sigma_j)
    return out


def calibrate_smile(
    strikes: np.ndarray,
    prices: np.ndarray,
    types: np.ndarray,
    S: float,
    T: float,
    r: float,
    x0: tuple[float, float, float, float] = (0.12, 1.0, -0.05, 0.10),
) -> np.ndarray:
    """Least-squares calibration of (sigma, lam, mu_j, sigma_j) to market prices.

    Minimises squared price errors. Bounds keep parameters in the region
    where the model is meaningful (non-negative intensity and jump vol,
    log-jump mean in [-2, 1]).
    """
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(prices, dtype=float)
    types = np.asarray(types)

    def resid(params: np.ndarray) -> np.ndarray:
        return model_prices(params, strikes, types, S, T, r) - prices

    sol = least_squares(
        resid,
        x0=np.array(x0, dtype=float),
        bounds=([1e-4, 0.0, -2.0, 1e-4], [2.0, 50.0, 1.0, 2.0]),
        method="trf",
        xtol=1e-12,
        ftol=1e-12,
    )
    return sol.x


if __name__ == "__main__":
    # Sanity: lam=0 must reproduce Black-Scholes.
    bs = bs_call_price(100.0, 100.0, 1.0, 0.05, 0.20)
    jd = merton_call_price(100.0, 100.0, 1.0, 0.05, 0.20, 0.0, -0.1, 0.15)
    assert abs(bs - jd) < 1e-10, (bs, jd)
    print(f"jump_diffusion: OK (BS={bs:.6f}, Merton lam=0={jd:.6f})")
