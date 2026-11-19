"""One-command end-to-end run of all five textbook models.

Previously the five models could only be exercised one case script at a
time, and three of the five cases need live network data. This entry
point runs every model in one command, fully offline, on data already
in the repo, and writes practitioner-usable exports:

    python run.py
    -> outputs/model_summary.csv   one row per model result
    -> outputs/bond_curve.csv      Vasicek vs CIR zero curve (documented
                                   post-2022 calibration, see below)

Data and parameters (all labelled, nothing silently switched):
- Returns: the cached NIFTY daily history in cases/nsei_daily.csv
  (2015-2026, the same cache cases/regimes_case.py uses).
- Merton: the documented 3 Nov 2026 NSE smile calibration quoted in
  results/jump_diffusion.md (diffusion sigma 2.83%, lambda 10.12/yr,
  mu_j -2.2%, sigma_j 4.2%; Black-Scholes single vol 13.82%), applied
  to an at-the-money 3-month call on the latest cached close at
  r = 6.5% (the rate convention of cases/jump_diffusion_case.py).
- Credit copula: the case study's labelled illustrative pool
  (100 names, PD 2%, LGD 60%) — analytic Vasicek sweep only.
- Short rates: the documented post-2022 calibrations quoted in
  results/short_rate.md (Vasicek kappa 0.9807, theta 5.0031%,
  sigma 0.5883%; CIR kappa 0.4745, theta 5.9330%, sigma 3.9365%),
  priced from r = 5.00%. These are physical-measure fits fed into
  risk-neutral pricing formulas — the P/Q caveat in that results
  file applies unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from textbook_quant.credit_copula import (  # noqa: E402
    expected_loss,
    vasicek_expected_shortfall,
    vasicek_loss_quantile,
)
from textbook_quant.garch import egarch_fit, garch_fit, gjr_fit  # noqa: E402
from textbook_quant.jump_diffusion import bs_call_price, merton_call_price  # noqa: E402
from textbook_quant.regimes import expected_durations, fit, smoothed_probs  # noqa: E402
from textbook_quant.short_rate import (  # noqa: E402
    cir_bond_price,
    cir_bond_yield,
    vasicek_bond_price,
    vasicek_bond_yield,
)

CACHE = ROOT / "cases" / "nsei_daily.csv"
OUT_DIR = ROOT / "outputs"
TRADING_DAYS = 252

# Documented calibrations — see module docstring for provenance.
MERTON = dict(sigma=0.0283, lam=10.12, mu_j=-0.022, sigma_j=0.042)
BS_SIGMA = 0.1382
RATE = 0.065
VASICEK = dict(kappa=0.9807, theta=0.050031, sigma=0.005883)
CIR = dict(kappa=0.4745, theta=0.059330, sigma=0.039365)
CURVE_R = 0.05
TENORS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)


def load_returns() -> tuple[pd.Series, float]:
    """Cached NIFTY log returns (decimal) and the latest close."""
    df = pd.read_csv(CACHE, parse_dates=["Date"])
    close = df.set_index("Date")["Close"].sort_index()
    rets = np.log(close / close.shift(1)).dropna()
    return rets, float(close.iloc[-1])


def merton_row(spot: float) -> dict:
    """ATM 3-month call: Black-Scholes vs Merton at documented params."""
    K, T = spot, 0.25
    bs = bs_call_price(spot, K, T, RATE, BS_SIGMA)
    jd = merton_call_price(spot, K, T, RATE, **MERTON)
    return {
        "model": "merton_jump_diffusion",
        "quantity": "atm_call_price_3m",
        "value": jd,
        "detail": (
            f"S=K={spot:.2f}, r=6.5%, BS(sigma=13.82%)={bs:.2f}, "
            f"Merton premium over BS={jd - bs:+.2f}"
        ),
    }


def regimes_rows(rets: np.ndarray) -> list[dict]:
    """Two-state Hamilton fit on cached returns (decimal)."""
    params = fit(rets)
    mu, sigma, P = params["mu"], params["sigma"], params["P"]
    durs = expected_durations(params)
    high_share = float((smoothed_probs(rets, params)[:, 1] > 0.5).mean())
    rows = []
    for state, name in ((0, "calm"), (1, "turbulent")):
        rows.append(
            {
                "model": "hamilton_regimes",
                "quantity": f"{name}_ann_vol_pct",
                "value": float(sigma[state] * np.sqrt(TRADING_DAYS) * 100),
                "detail": (
                    f"mu_ann={mu[state] * TRADING_DAYS * 100:.2f}%, "
                    f"expected_duration_days={durs[state]:.1f}, "
                    f"stay_prob={P[state, state]:.4f}"
                ),
            }
        )
    rows.append(
        {
            "model": "hamilton_regimes",
            "quantity": "turbulent_share_pct",
            "value": high_share * 100,
            "detail": "share of days with smoothed P(turbulent) > 0.5",
        }
    )
    return rows


def garch_rows(pct_rets: np.ndarray) -> list[dict]:
    """GARCH/GJR/EGARCH MLE fits on returns in percent (case convention)."""
    rows = []
    for name, fn in (("garch", garch_fit), ("gjr", gjr_fit), ("egarch", egarch_fit)):
        f = fn(pct_rets)
        ann_vol = float(np.sqrt(f.last_var) / 100 * np.sqrt(TRADING_DAYS) * 100)
        rows.append(
            {
                "model": f"garch_family_{name}",
                "quantity": "last_day_ann_vol_pct",
                "value": ann_vol,
                "detail": (
                    f"loglik={f.loglik:.1f}, persistence={f.persistence:.4f}, "
                    f"params={ {k: round(v, 6) for k, v in f.params.items()} }"
                ),
            }
        )
    return rows


def credit_rows() -> list[dict]:
    """Analytic Vasicek sweep for the labelled illustrative pool."""
    pd_, lgd = 0.02, 0.60
    rows = [
        {
            "model": "credit_copula_vasicek",
            "quantity": "expected_loss_pct",
            "value": expected_loss(pd_, lgd) * 100,
            "detail": "illustrative pool: 100 names, PD=2%, LGD=60%; EL is rho-invariant",
        }
    ]
    for rho in (0.0, 0.10, 0.20, 0.30):
        rows.append(
            {
                "model": "credit_copula_vasicek",
                "quantity": f"var999_pct_rho{rho:.2f}",
                "value": vasicek_loss_quantile(pd_, rho, lgd, 0.999) * 100,
                "detail": f"ES99={vasicek_expected_shortfall(pd_, rho, lgd, 0.99) * 100:.2f}%",
            }
        )
    return rows


def bond_curve() -> pd.DataFrame:
    """Vasicek vs CIR zero prices/yields at the documented calibrations."""
    rows = []
    for tau in TENORS:
        rows.append(
            {
                "tenor_years": tau,
                "vasicek_price": vasicek_bond_price(CURVE_R, tau=tau, **VASICEK),
                "vasicek_yield_pct": vasicek_bond_yield(CURVE_R, tau=tau, **VASICEK) * 100,
                "cir_price": cir_bond_price(CURVE_R, tau=tau, **CIR),
                "cir_yield_pct": cir_bond_yield(CURVE_R, tau=tau, **CIR) * 100,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    rets, spot = load_returns()
    r = rets.to_numpy()
    summary = pd.DataFrame(
        [merton_row(spot)]
        + regimes_rows(r)
        + garch_rows(r * 100)
        + credit_rows()
    )
    curve = bond_curve()
    OUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(OUT_DIR / "model_summary.csv", index=False)
    curve.to_csv(OUT_DIR / "bond_curve.csv", index=False)
    print(f"Returns: {len(r)} days, {rets.index[0].date()} to {rets.index[-1].date()}, spot={spot:.2f}")
    for row in summary.itertuples():
        print(f"{row.model:28s} {row.quantity:26s} {row.value:12.4f}  {row.detail}")
    print("Bond curve (10y): "
          f"Vasicek yield {curve.iloc[7]['vasicek_yield_pct']:.2f}%, "
          f"CIR yield {curve.iloc[7]['cir_yield_pct']:.2f}%")
    print(f"Wrote {OUT_DIR / 'model_summary.csv'} ({len(summary)} rows) and "
          f"{OUT_DIR / 'bond_curve.csv'} ({len(curve)} rows)")


if __name__ == "__main__":
    main()
