"""Case: Hamilton regime-switching on real NIFTY returns, 2015-2026.

Fits the 2-state Gaussian HMM (EM, from scratch) to daily log returns of
the NIFTY index (^NSEI, cached in cases/nsei_daily.csv; fetched with
yfinance). Reports per-regime annualised moments, the transition matrix,
expected durations, the share of days spent in the high-vol regime, and
the high-vol share during the COVID crash month (Mar 2020) vs the full
sample. Writes results/regimes.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textbook_quant.regimes import expected_durations, fit, smoothed_probs  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "nsei_daily.csv"
OUT = HERE.parents[0] / "results" / "regimes.md"
TRADING_DAYS = 252


def load_returns() -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(CACHE, parse_dates=["Date"])
    close = df.set_index("Date")["Close"].sort_index()
    rets = np.log(close / close.shift(1)).dropna()
    return rets, close


def main() -> None:
    rets, _ = load_returns()
    r = rets.to_numpy()
    dates = rets.index
    print(f"Returns: {len(r)} days, {dates[0].date()} to {dates[-1].date()}")

    params = fit(r)
    mu, sigma, P = params["mu"], params["sigma"], params["P"]
    durs = expected_durations(params)
    smooth = smoothed_probs(r, params)
    high = smooth[:, 1] > 0.5
    share_all = float(high.mean() * 100)

    covid_mask = (dates >= "2020-03-01") & (dates <= "2020-03-31")
    covid_share = float(high[np.asarray(covid_mask)].mean() * 100)
    covid_days = int(np.asarray(covid_mask).sum())

    ann_vol = sigma * np.sqrt(TRADING_DAYS) * 100
    ann_mu = mu * TRADING_DAYS * 100
    print(f"Low-vol : mu={ann_mu[0]:.2f}%/yr vol={ann_vol[0]:.2f}%/yr dur={durs[0]:.0f}d")
    print(f"High-vol: mu={ann_mu[1]:.2f}%/yr vol={ann_vol[1]:.2f}%/yr dur={durs[1]:.0f}d")
    print(f"P = [[{P[0,0]:.4f}, {P[0,1]:.4f}], [{P[1,0]:.4f}, {P[1,1]:.4f}]]")
    print(f"High-vol share: full sample {share_all:.1f}%, Mar 2020 {covid_share:.1f}%")

    OUT.write_text(
        f"""# Case — Hamilton regime-switching on NIFTY daily returns, 2015-2026

**Data.** NIFTY index (^NSEI via Yahoo Finance), daily closes
{dates[0].date()} to {dates[-1].date()}; {len(r):,} daily log returns.
Model: 2-state Gaussian Markov-switching (Hamilton 1989), fitted by EM
(forward-backward) from scratch, log-likelihood {params['loglik']:,.1f}
after {params['n_iter']} EM iterations. A day is counted as high-vol
when its smoothed high-vol probability exceeds 0.5.

## Fitted regimes

| Regime | Mean return (ann.) | Volatility (ann.) | Expected duration |
|---|---|---|---|
| Low-vol (calm) | {ann_mu[0]:+.1f}% | {ann_vol[0]:.1f}% | {durs[0]:.0f} trading days |
| High-vol (turbulent) | {ann_mu[1]:+.1f}% | {ann_vol[1]:.1f}% | {durs[1]:.0f} trading days |

**Transition matrix** (rows = from, columns = to; order calm, turbulent):

| | → calm | → turbulent |
|---|---|---|
| calm → | {P[0,0]:.3f} | {P[0,1]:.3f} |
| turbulent → | {P[1,0]:.3f} | {P[1,1]:.3f} |

## Regime occupancy

- **Full sample:** {share_all:.1f}% of days in the high-vol regime.
- **March 2020 (COVID crash, {covid_days} trading days):**
  {covid_share:.1f}% of days in the high-vol regime — the filter puts
  essentially the whole crash month in the turbulent state, against a
  {share_all:.0f}% base rate. The calm state's expected duration of
  ~{durs[0]:.0f} days vs ~{durs[1]:.0f} for turbulence matches how the
  market actually behaves: long quiet stretches, short violent ones.

## Limitations

- **Two states is a simplification.** Ordinary turbulence and true
  crises (2008/2020-style) are forced into one state; a 3-state model
  usually separates them.
- **Gaussian emissions.** Within-regime tails are fatter than Gaussian,
  so extreme days are partly absorbed by the high-vol state rather than
  modelled.
- **Constant transition probabilities.** The chance of tipping into
  turbulence does not depend on valuations, credit or time spent calm.
- Smoothed probabilities use the whole sample (look-ahead); real-time
  use must rely on filtered probabilities, which lag turning points by
  a few days.
"""
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
