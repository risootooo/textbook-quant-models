"""Model 5 — One-factor short-rate models: Vasicek and CIR.

Textbook references: Vasicek (1977); Cox, Ingersoll & Ross (1985);
Brigo & Mercurio, *Interest Rate Models*, ch. 3.

Both models are mean-reverting diffusions for the instantaneous short
rate :math:`r_t`:

* Vasicek:  ``dr = kappa (theta - r) dt + sigma dW``
  Gaussian, analytically tractable, and honest about its flaw: rates
  can go negative.
* CIR:      ``dr = kappa (theta - r) dt + sigma sqrt(r) dW``
  Volatility scales with the level, so the rate stays non-negative;
  if the Feller condition ``2 kappa theta >= sigma**2`` holds it
  never even reaches zero.

Estimation is by *exact* maximum likelihood — the true transition
density of each process, not an Euler approximation:

* Vasicek's transition is Gaussian with closed-form conditional mean
  and variance.
* CIR's transition is a scaled noncentral chi-square
  (``scipy.stats.ncx2``).

Bond prices use the affine closed forms: in both models
``P(tau) = exp(A(tau) - B(tau) * r)``.

Simulation: Vasicek uses the exact Gaussian transition; CIR uses the
full-truncation Euler scheme of Lord, Koekkoek & van Dijk (2010),
which evaluates drift and diffusion at ``max(r, 0)`` and floors the
update at zero, so simulated paths are non-negative by construction.

Conventions: rates are decimals (0.05 = 5%), times are in years, and
``dt`` is the observation spacing in years. Everything is seeded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

__all__ = [
    "ShortRateFit",
    "vasicek_mle",
    "cir_mle",
    "vasicek_bond_price",
    "cir_bond_price",
    "vasicek_bond_yield",
    "cir_bond_yield",
    "vasicek_simulate",
    "cir_simulate",
]


# ---------------------------------------------------------------------------
# Fit container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShortRateFit:
    """Result of an exact-MLE calibration of a one-factor short-rate model.

    Attributes:
        model: ``"vasicek"`` or ``"cir"``.
        kappa: Mean-reversion speed (per year).
        theta: Long-run mean level of the short rate (decimal).
        sigma: Volatility parameter (decimal; for CIR it multiplies
            ``sqrt(r)``).
        loglik: Maximised log-likelihood of the observed transitions.
        n_obs: Number of rate observations used (transitions = n_obs - 1).
        dt: Observation spacing in years used for the fit.
    """

    model: str
    kappa: float
    theta: float
    sigma: float
    loglik: float
    n_obs: int
    dt: float

    @property
    def half_life_years(self) -> float:
        """Mean-reversion half-life: time for a shock's expected effect
        to halve, ``ln(2) / kappa``, in years."""
        return float(np.log(2.0) / self.kappa)

    @property
    def feller_margin(self) -> float:
        """``2*kappa*theta - sigma**2``. For CIR this is the Feller
        quantity: positive means the zero boundary is unattainable.
        (Reported for Vasicek too, where it carries no boundary meaning.)
        """
        return float(2.0 * self.kappa * self.theta - self.sigma ** 2)

    @property
    def feller_holds(self) -> bool:
        """True if ``2*kappa*theta >= sigma**2``."""
        return self.feller_margin >= 0.0


# ---------------------------------------------------------------------------
# Vasicek: exact-transition MLE
# ---------------------------------------------------------------------------

def _vasicek_neg_loglik(params: np.ndarray, r: np.ndarray, dt: float) -> float:
    """Negative exact log-likelihood of the Vasicek transitions.

    Params are ``(log kappa, theta, log sigma)``.
    ``r_{t+dt} | r_t ~ N(theta + (r_t - theta) e^{-k dt},
    sigma^2 (1 - e^{-2 k dt}) / (2 k))``.
    """
    kappa = float(np.exp(params[0]))
    theta = float(params[1])
    sigma = float(np.exp(params[2]))
    decay = np.exp(-kappa * dt)
    mean = theta + (r[:-1] - theta) * decay
    var = sigma ** 2 * (1.0 - decay ** 2) / (2.0 * kappa)
    ll = stats.norm.logpdf(r[1:], loc=mean, scale=np.sqrt(var))
    return -float(np.sum(ll))


def _ar1_start(r: np.ndarray, dt: float) -> tuple[float, float, float]:
    """OLS AR(1) starting values (the Gaussian conditional MLE itself,
    used purely as a starting point for the optimiser)."""
    x, y = r[:-1], r[1:]
    slope, intercept = np.polyfit(x, y, 1)
    slope = float(np.clip(slope, 1e-6, 1.0 - 1e-6))
    kappa0 = -np.log(slope) / dt
    theta0 = intercept / (1.0 - slope)
    resid_sd = float(np.std(y - (intercept + slope * x), ddof=2))
    var0 = resid_sd ** 2
    sigma0 = np.sqrt(max(var0 * 2.0 * kappa0 / (1.0 - slope ** 2), 1e-12))
    return float(kappa0), float(theta0), float(sigma0)


def vasicek_mle(rates: np.ndarray, dt: float) -> ShortRateFit:
    """Exact-transition maximum likelihood for the Vasicek model.

    Optimisation is L-BFGS-B over ``(log kappa, theta, log sigma)``
    inside a wide, economically meaningful box
    (``kappa in [1e-3, 50]``, ``theta in [-0.5, 2]``,
    ``sigma in [1e-4, 1]``). The box matters: on samples whose level
    shifts regime (e.g. U.S. bill rates 2019-2026) the likelihood has a
    ridge running to ``kappa -> 0`` with ``theta`` unidentified, and
    the constrained optimum then sits on the kappa boundary — which
    the caller can detect from the returned value instead of receiving
    a meaningless ``theta`` of several thousand percent.

    Args:
        rates: 1-D array of observed short rates (decimals), equally
            spaced.
        dt: Spacing between observations, in years.

    Returns:
        A :class:`ShortRateFit` with ``model="vasicek"``.
    """
    r = np.asarray(rates, dtype=float).ravel()
    if r.size < 10:
        raise ValueError("need at least 10 observations to fit Vasicek")
    if dt <= 0:
        raise ValueError("dt must be positive")
    k0, t0, s0 = _ar1_start(r, dt)
    x0 = np.array([
        np.log(np.clip(k0, 1.1e-3, 49.0)),
        np.clip(t0, -0.49, 1.99),
        np.log(np.clip(s0, 1.1e-4, 0.99)),
    ])
    bounds = [(np.log(1e-3), np.log(50.0)), (-0.5, 2.0),
              (np.log(1e-4), np.log(1.0))]
    res = optimize.minimize(
        _vasicek_neg_loglik, x0, args=(r, dt),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500},
    )
    if _vasicek_neg_loglik(x0, r, dt) < res.fun:  # keep the better point
        res_x, res_fun = x0, _vasicek_neg_loglik(x0, r, dt)
    else:
        res_x, res_fun = res.x, res.fun
    kappa = float(np.exp(res_x[0]))
    theta = float(res_x[1])
    sigma = float(np.exp(res_x[2]))
    return ShortRateFit(
        model="vasicek", kappa=kappa, theta=theta, sigma=sigma,
        loglik=-float(res_fun), n_obs=int(r.size), dt=float(dt),
    )


# ---------------------------------------------------------------------------
# CIR: noncentral chi-square MLE
# ---------------------------------------------------------------------------

def _cir_neg_loglik(params: np.ndarray, r: np.ndarray, dt: float) -> float:
    """Negative exact log-likelihood of the CIR transitions.

    Params are ``(log kappa, log theta, log sigma)``.
    With ``c = sigma^2 (1 - e^{-k dt}) / (4 k)``, the scaled rate
    ``r_{t+dt} / c`` is noncentral chi-square with
    ``df = 4 k theta / sigma^2`` degrees of freedom and noncentrality
    ``r_t e^{-k dt} / c``; the density of ``r_{t+dt}`` picks up the
    Jacobian factor ``1 / c``.
    """
    kappa = float(np.exp(params[0]))
    theta = float(np.exp(params[1]))
    sigma = float(np.exp(params[2]))
    decay = np.exp(-kappa * dt)
    c = sigma ** 2 * (1.0 - decay) / (4.0 * kappa)
    df = 4.0 * kappa * theta / sigma ** 2
    nc = r[:-1] * decay / c
    ll = stats.ncx2.logpdf(r[1:] / c, df, nc) - np.log(c)
    if not np.all(np.isfinite(ll)):
        return 1e12
    return -float(np.sum(ll))


def cir_mle(rates: np.ndarray, dt: float) -> ShortRateFit:
    """Exact-transition maximum likelihood for the CIR model.

    Uses the noncentral chi-square transition density
    (``scipy.stats.ncx2``). The returned fit reports the Feller
    quantity via :attr:`ShortRateFit.feller_margin` /
    :attr:`ShortRateFit.feller_holds`. Optimisation is bounded
    L-BFGS-B from multiple starts, for the same ridge reason
    documented in :func:`vasicek_mle`.

    Args:
        rates: 1-D array of observed short rates (decimals), equally
            spaced. All values must be strictly positive — the CIR
            transition density is defined on the positive half-line.
        dt: Spacing between observations, in years.

    Returns:
        A :class:`ShortRateFit` with ``model="cir"``.
    """
    r = np.asarray(rates, dtype=float).ravel()
    if r.size < 10:
        raise ValueError("need at least 10 observations to fit CIR")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if np.any(r <= 0.0):
        raise ValueError(
            "CIR MLE requires strictly positive rates; got a non-positive "
            "observation (the transition density lives on r > 0)"
        )
    # Bounded L-BFGS-B over (log kappa, log theta, log sigma), from
    # several starts; the box (kappa in [1e-3, 50], theta in
    # [1e-4, 2], sigma in [1e-3, 2]) keeps the optimiser off the
    # kappa -> 0 ridge where theta is unidentified (see vasicek_mle).
    k0, t0, s0 = _ar1_start(r, dt)
    mean_r = float(np.mean(r))
    starts = [
        np.array([np.log(np.clip(k0, 0.01, 10.0)),
                  np.log(np.clip(t0, 1e-3, 0.5)),
                  np.log(np.clip(s0 * 5.0, 0.01, 0.5))]),
        np.array([np.log(0.5), np.log(max(mean_r, 1e-3)), np.log(0.10)]),
        np.array([np.log(0.1), np.log(max(mean_r, 1e-3)), np.log(0.04)]),
    ]
    bounds = [(np.log(1e-3), np.log(50.0)), (np.log(1e-4), np.log(2.0)),
              (np.log(1e-3), np.log(2.0))]
    best_x: np.ndarray | None = None
    best_fun = np.inf
    for x0 in starts:
        res = optimize.minimize(
            _cir_neg_loglik, x0, args=(r, dt),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500},
        )
        for cand_x in (x0, res.x):
            cand_fun = _cir_neg_loglik(cand_x, r, dt)
            if cand_fun < best_fun:
                best_fun, best_x = cand_fun, cand_x
    assert best_x is not None
    kappa = float(np.exp(best_x[0]))
    theta = float(np.exp(best_x[1]))
    sigma = float(np.exp(best_x[2]))
    return ShortRateFit(
        model="cir", kappa=kappa, theta=theta, sigma=sigma,
        loglik=-float(best_fun), n_obs=int(r.size), dt=float(dt),
    )


# ---------------------------------------------------------------------------
# Affine bond prices
# ---------------------------------------------------------------------------

def _check_tau(tau: float) -> None:
    if tau < 0:
        raise ValueError("tau (time to maturity, years) must be non-negative")


def vasicek_bond_price(r: float, kappa: float, theta: float,
                       sigma: float, tau: float) -> float:
    """Zero-coupon bond price under Vasicek: ``P = exp(A - B r)`` with

    ``B = (1 - e^{-k tau}) / k`` and
    ``A = (theta - sigma^2 / (2 k^2)) (B - tau) - sigma^2 B^2 / (4 k)``.
    At ``tau = 0`` the price is exactly 1.
    """
    _check_tau(tau)
    if tau == 0.0:
        return 1.0
    b = (1.0 - np.exp(-kappa * tau)) / kappa
    a = ((theta - sigma ** 2 / (2.0 * kappa ** 2)) * (b - tau)
         - sigma ** 2 * b ** 2 / (4.0 * kappa))
    return float(np.exp(a - b * r))


def cir_bond_price(r: float, kappa: float, theta: float,
                   sigma: float, tau: float) -> float:
    """Zero-coupon bond price under CIR: ``P = exp(A - B r)`` with
    ``h = sqrt(kappa^2 + 2 sigma^2)``,

    ``B = 2 (e^{h tau} - 1) / ((kappa + h)(e^{h tau} - 1) + 2 h)`` and
    ``A = (2 kappa theta / sigma^2) * ln( 2 h e^{(kappa + h) tau / 2}
          / ((kappa + h)(e^{h tau} - 1) + 2 h) )``.

    At ``tau = 0`` the price is exactly 1.
    """
    _check_tau(tau)
    if tau == 0.0:
        return 1.0
    h = np.sqrt(kappa ** 2 + 2.0 * sigma ** 2)
    e_ht = np.exp(h * tau)
    denom = (kappa + h) * (e_ht - 1.0) + 2.0 * h
    b = 2.0 * (e_ht - 1.0) / denom
    a = (2.0 * kappa * theta / sigma ** 2) * np.log(
        2.0 * h * np.exp((kappa + h) * tau / 2.0) / denom)
    return float(np.exp(a - b * r))


def vasicek_bond_yield(r: float, kappa: float, theta: float,
                       sigma: float, tau: float) -> float:
    """Continuously compounded zero yield implied by the Vasicek price."""
    if tau <= 0:
        raise ValueError("tau must be positive for a yield")
    return float(-np.log(vasicek_bond_price(r, kappa, theta, sigma, tau)) / tau)


def cir_bond_yield(r: float, kappa: float, theta: float,
                   sigma: float, tau: float) -> float:
    """Continuously compounded zero yield implied by the CIR price."""
    if tau <= 0:
        raise ValueError("tau must be positive for a yield")
    return float(-np.log(cir_bond_price(r, kappa, theta, sigma, tau)) / tau)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def vasicek_simulate(r0: float, kappa: float, theta: float, sigma: float,
                     n_steps: int, dt: float, n_paths: int = 1,
                     seed: int = 42) -> np.ndarray:
    """Simulate Vasicek paths with the *exact* Gaussian transition.

    Returns an array of shape ``(n_paths, n_steps + 1)`` whose first
    column is ``r0``. Paths may go negative — that is a property of
    the model, not a bug.
    """
    rng = np.random.default_rng(seed)
    decay = np.exp(-kappa * dt)
    mean_step = theta * (1.0 - decay)
    sd_step = sigma * np.sqrt((1.0 - decay ** 2) / (2.0 * kappa))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = r0
    shocks = rng.standard_normal((n_paths, n_steps))
    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] * decay + mean_step + sd_step * shocks[:, t]
    return paths


def cir_simulate(r0: float, kappa: float, theta: float, sigma: float,
                 n_steps: int, dt: float, n_paths: int = 1,
                 seed: int = 42) -> np.ndarray:
    """Simulate CIR paths with the full-truncation Euler scheme.

    Drift and diffusion are evaluated at ``r_plus = max(r, 0)`` and the
    update is floored at zero::

        r <- max(r + kappa (theta - r_plus) dt
                 + sigma sqrt(r_plus) sqrt(dt) Z, 0)

    so every simulated value is non-negative by construction. Returns
    an array of shape ``(n_paths, n_steps + 1)`` whose first column is
    ``r0``.
    """
    if r0 < 0:
        raise ValueError("CIR starting rate r0 must be non-negative")
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = r0
    shocks = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    for t in range(n_steps):
        r_plus = np.maximum(paths[:, t], 0.0)
        nxt = (paths[:, t] + kappa * (theta - r_plus) * dt
               + sigma * np.sqrt(r_plus) * shocks[:, t])
        paths[:, t + 1] = np.maximum(nxt, 0.0)
    return paths
