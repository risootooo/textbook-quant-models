# Case — Hamilton regime-switching on NIFTY daily returns, 2015-2026

**Data.** NIFTY index (^NSEI via Yahoo Finance), daily closes
2015-01-05 to 2026-11-03; 2,913 daily log returns.
Model: 2-state Gaussian Markov-switching (Hamilton 1989), fitted by EM
(forward-backward) from scratch, log-likelihood 9,664.2
after 36 EM iterations. A day is counted as high-vol
when its smoothed high-vol probability exceeds 0.5.

## Fitted regimes

| Regime | Mean return (ann.) | Volatility (ann.) | Expected duration |
|---|---|---|---|
| Low-vol (calm) | +15.1% | 11.7% | 71 trading days |
| High-vol (turbulent) | -39.0% | 35.4% | 9 trading days |

**Transition matrix** (rows = from, columns = to; order calm, turbulent):

| | → calm | → turbulent |
|---|---|---|
| calm → | 0.986 | 0.014 |
| turbulent → | 0.114 | 0.886 |

## Regime occupancy

- **Full sample:** 10.2% of days in the high-vol regime.
- **March 2020 (COVID crash, 21 trading days):**
  100.0% of days in the high-vol regime — the filter puts
  essentially the whole crash month in the turbulent state, against a
  10% base rate. The calm state's expected duration of
  ~71 days vs ~9 for turbulence matches how the
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
