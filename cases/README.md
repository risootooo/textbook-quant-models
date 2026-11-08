# Case studies

Each case takes one textbook model from `src/textbook_quant/` to real
data and writes its findings — numbers, method, and limitations — into
`results/`. Run any case from the repo root:

```bash
python cases/<case_file>.py
```

Shared conventions: rates and returns are decimals inside the models
(0.05 = 5%), random generators are seeded, and every case states its
data source in the output file. If a primary source is unreachable, a
case falls back only where documented below — and says so in its
results file rather than silently switching data.

| Case file | Model | Data it needs |
|---|---|---|
| `jump_diffusion_case.py` | Merton jump-diffusion option pricing | Equity index daily closes (S&P 500 via yfinance `^GSPC`) for the return/jump calibration, plus a listed option chain snapshot for the pricing comparison. Network access for yfinance. |
| `regimes_case.py` | Two-state Markov regime-switching returns | A long daily price history (S&P 500 via yfinance `^GSPC`, longest available) — no other inputs; regimes are estimated, not imported. |
| `garch_case.py` | GARCH(1,1) volatility | Daily equity index closes (S&P 500 via yfinance `^GSPC`); log returns are computed in the case. |
| `credit_copula_case.py` | One-factor Gaussian copula credit portfolio | A loan/credit portfolio dataset (e.g. the UCI German Credit data used by the sibling `credit-scorecard-pro` repo) for default rates and the loss-distribution calibration. |
| `short_rate_case.py` | Vasicek & CIR short-rate calibration | Official U.S. Treasury Daily Par Yield Curve Rates via the TextView route documented in the sibling `rates-lab` repo (3 Mo tenor, daily, 2019–latest; also reads that day's 2/5/10 Yr par yields). Fallback if Treasury is unreachable: yfinance `^IRX` (+ `^FVX`/`^TNX`), flagged in the output. |

Notes per case:

- **jump_diffusion_case.py** — calibrates jump intensity/size from
  returns, then prices options under Merton vs Black–Scholes.
- **regimes_case.py** — Hamilton-style two-regime estimation by EM;
  reports smoothed regime probabilities around known stress episodes.
- **garch_case.py** — MLE fit, volatility forecast, and a VaR
  backtest against realised exceedances.
- **credit_copula_case.py** — asset-correlation calibration and the
  portfolio loss distribution; compares copula tail losses against
  the independence benchmark.
- **short_rate_case.py** — exact-transition MLE for both models on
  the full sample *and* the post-2022 regime; the full-sample fit is
  a worked example of weak identification, so the case reports the
  profile likelihood and both samples instead of one fragile point
  estimate.
