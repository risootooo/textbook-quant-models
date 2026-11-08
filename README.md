# Textbook Quant Models

![tests](https://img.shields.io/badge/tests-43%20passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![release](https://img.shields.io/badge/release-v1.0.0-orange)

**Five canonical quantitative-finance models, implemented from scratch — no quant libraries — and taken to real data.** Each model is built the way the textbooks define it (Merton 1976, Hamilton 1989, Bollerslev / Glosten-Jagannathan-Runkle / Nelson, Vasicek's large-pool limit, Vasicek 1977 / Cox-Ingersoll-Ross 1985), tested against closed-form properties, and then run on a real dataset with the results — including the failures — written down.

This repo fills the theory layer of a derivatives-first stack: [Greeks Lab](https://github.com/risootooo/greeks-lab) prices and hedges, [Vol Surface Atlas](https://github.com/risootooo/vol-surface-atlas) maps the smile, [EdgeGate](https://github.com/risootooo/edgegate) validates strategies — this one rebuilds the underlying models from their papers.

## The five models and what the data said

| # | Model | Case study | Headline result |
|---|---|---|---|
| 1 | **Merton jump-diffusion** | NIFTY smile, real NSE bhavcopy 3 Nov 2026 | IV RMSE **2.54** vol pts vs **8.99** for Black-Scholes — λ≈10 jumps/yr of −2.2% carries the skew |
| 2 | **Hamilton regime-switching** | NIFTY daily returns 2015–2026 | Calm regime 11.7% vol / 71-day duration; turbulent 35.4% vol / 9 days; **March 2020: 100% high-vol** vs 10.2% base rate |
| 3 | **GARCH / GJR / EGARCH** | NIFTY rolling out-of-sample tournament, 750 days | GARCH best on QLIKE (−8.605) but Diebold-Mariano p = 0.52 — *the three are statistically hard to separate*; GJR catches up fastest in the COVID week |
| 4 | **Credit copula (Vasicek)** | Illustrative 100-name pool, labelled as such | EL is 1.20% at **every** ρ; VaR99.9 goes 1.20% → **19.98%** as ρ rises 0 → 0.30; a t-copula (ν=4) **doubles** the Gaussian tail |
| 5 | **Vasicek & CIR short rates** | US Treasury 3M bill, 2019–2026 | Full-sample MLE is *degenerate* (κ→0: three rate regimes look like a unit root) — documented, with the post-2022 calibration as the headline (Vasicek κ=0.98, θ=5.00%; Feller holds for CIR) |

Full write-ups: [results/](results/) — one file per model, each with method, tables and limitations.

## What's inside

```
src/textbook_quant/
    jump_diffusion.py   Merton series pricer, MC, IV inversion, smile calibration
    regimes.py          2-state Gaussian HMM: EM (forward-backward) from scratch,
                        Hamilton filter, smoothed probabilities, Viterbi
    garch.py            GARCH/GJR/EGARCH filters + Gaussian MLE, forecasts,
                        QLIKE/MSE losses, Diebold-Mariano with Newey-West HAC
    credit_copula.py    Vasicek closed-form loss quantiles + Gaussian/t copula MC
    short_rate.py       Vasicek exact-transition MLE, CIR noncentral-χ² MLE,
                        affine zero-coupon bond prices, simulators
cases/                  one runnable case study per model (see cases/README.md)
results/                the verified outputs the table above quotes
tests/                  43 tests — closed-form identities, parameter recovery,
                        seeded Monte Carlo cross-checks
```

## Design rules

- **From scratch.** Estimation and filtering are hand-written (EM, MLE recursions, the Merton series). `numpy`/`pandas`/`scipy` provide numerics, not models — no `arch`, no `statsmodels`, no `QuantLib`.
- **Tests check mathematics, not just code paths** — λ=0 collapses Merton to Black-Scholes to 1e-8, put-call parity is exact, the Vasicek quantile is checked against 200k-simulation copula output, estimators must recover known parameters from synthetic data.
- **Negative results stay in.** The Diebold-Mariano test that won't separate GARCH from GJR, and the short-rate likelihood that runs to the boundary, are reported as findings — with the profile-likelihood evidence — not quietly re-run until they behave.
- **Illustrative means labelled.** The credit case uses textbook parameters (PD 2%, LGD 60%) and says so in its title; it demonstrates a mechanism, not a portfolio.

## Run it

```bash
pip install -e .            # or: pip install numpy pandas scipy
python -m pytest tests -q   # 43 tests
python cases/jump_diffusion_case.py   # needs the NSE bhavcopy CSV (see cases/README.md)
python cases/regimes_case.py          # uses the cached NIFTY history in cases/
python cases/garch_case.py            # downloads ^NSEI via yfinance
python cases/credit_copula_case.py    # pure simulation, seeded
python cases/short_rate_case.py       # fetches the official Treasury curve pages
```

## Limitations (the short version)

One factor everywhere: one jump process, two regimes, one systematic credit factor, one short rate. Gaussian innovations in the GARCH and HMM cores understate within-state tails. The short-rate calibrations are physical-measure fits fed into risk-neutral pricing formulas — the gap to actual par yields is shown, not hidden. Each results file carries its own full limitations section.

## Author

**Samrat Banerjee** — derivatives & quantitative research.
[LinkedIn](https://www.linkedin.com/in/samrat-banerjee-919a45447) · More of the stack: [profile](https://github.com/risootooo)

MIT licensed. Research code — not investment advice.
