# Model 3 — GARCH Volatility Forecasting Tournament (NIFTY)

Data: ^NSEI daily closes, 2010-01-04 to 2026-11-08 (4138 log returns), Yahoo Finance. Models are estimated on percentage returns and forecasts rescaled to decimal variance for scoring.

## Method

- Rolling out-of-sample exercise over the last 750 trading days (2023-10-25 to 2026-11-08).
- Estimation window: rolling 1000 days; parameters refit every 21 trading days; between refits the variance recursion is updated daily with the observed return (true 1-step-ahead forecasts).
- Scoring proxy: next day's squared return. Losses: QLIKE (log h + r^2/h, Patton 2011) and MSE on variance; also MAE of the vol forecast against |r|.
- Diebold-Mariano test with Newey-West HAC variance on the QLIKE loss differential of the best two models.

## Out-of-sample scores (lower is better)

| Model | QLIKE | MSE on variance | MAE vs abs(r) |
|---|---|---|---|
| GARCH | -8.6049 | 3.659e-08 | 0.0049 |
| GJR | -8.5927 | 3.692e-08 | 0.0049 |
| EGARCH | -8.5860 | 3.639e-08 | 0.0048 |

Diebold-Mariano (GARCH vs GJR, QLIKE): DM = -0.636, p = 0.5247. The gap favours GARCH but is not significant at the 5% level.

## Full-sample fits (2010–2026, percentage returns)

| Model | Parameters | Persistence | Unconditional vol (ann.) |
|---|---|---|---|
| GARCH | omega=0.0208, alpha=0.0875, beta=0.8936 | 0.9812 | 16.7% |
| GJR | omega=0.0291, alpha=0.0215, gamma=0.1354, beta=0.8847 | 0.9739 | 16.8% |
| EGARCH | omega=0.0026, alpha=0.1473, gamma=-0.0956, beta=0.9702 | 0.9702 | 16.6% |

Persistence is alpha+beta (GARCH), alpha+beta+gamma/2 (GJR) and beta (EGARCH). All three sit close to, but below, one: NIFTY volatility is highly persistent yet stationary over this sample.

## COVID crash week exhibit

Each model was fit on the 1,500 days ending Fri 20 Mar 2020 and then produced genuine 1-step-ahead forecasts for Mon 23 – Fri 27 Mar 2020 (23 Mar itself was a -13.9% day for NIFTY). The table shows the average annualised vol forecast across that week against what actually realised.

| Model | Avg vol forecast, crash week | Realised vol that week |
|---|---|---|
| GARCH | 82.7% | 126.8% |
| GJR | 104.2% | 126.8% |
| EGARCH | 66.6% | 126.8% |

For scale, the 21-day realised vol ending 27 Mar 2020 was 78.6%. Every model's level going into the week was far below this; the exhibit measures how fast each recursion catches up once the crash returns arrive, not foresight of the crash itself.

## Limitations

- Daily squared returns are an extremely noisy variance proxy; QLIKE rankings are robust to that noise in expectation, but the DM test still often cannot separate close models — exactly what we find.
- Gaussian innovations: fat tails load entirely onto the variance dynamics. A Student-t GARCH would likely improve QLIKE.
- Parameters are refit every 21 days, so the exercise understates what daily refitting would deliver and overstates what monthly refitting would.
- EGARCH variance forecasts exponentiate the log-variance path (median-style forecast, no Jensen correction).
- Yahoo's ^NSEI history is taken as published; the final row (8 Nov 2026, a Sunday-dated print) is included as delivered by the data vendor.


*Runtime of this script: 51s.*
