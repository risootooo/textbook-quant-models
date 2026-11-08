# Model 4 — Credit Portfolio Copula (Illustrative Simulation Study)

> **Scope label.** This is a simulation study with illustrative parameters (100 obligors, PD 2%, LGD 60%, equal notionals). It is not calibrated to any real portfolio and makes no claim about real-world losses. Its purpose is to isolate one mechanism: what asset correlation does to a credit portfolio's loss distribution.

## The one-line result

Expected loss is 1.20% at *every* correlation — correlation cannot move the mean of the loss distribution, only its shape. But the 99.9% VaR rises from 1.20% at rho = 0 to 19.98% at rho = 0.30 (x16.6), and ES99 from 1.20% to 14.57%. Correlation is a tail phenomenon, not a mean phenomenon.

## Vasicek large-pool (analytic) vs Gaussian-copula Monte Carlo

Monte Carlo: finite pool of 100 obligors, 200,000 simulations, seed 20261108. The analytic column is the infinitely granular limit, so the MC tail sits slightly *above* it: a 100-name pool keeps idiosyncratic risk that the limit assumes away.

| rho | EL (both) | VaR99 Vasicek | VaR99 MC | VaR99.9 Vasicek | VaR99.9 MC | ES99 Vasicek | ES99 MC |
|---|---|---|---|---|---|---|---|
| 0.00 | 1.20% | 1.20% | 3.60% | 1.20% | 4.20% | 1.20% | 3.80% |
| 0.05 | 1.20% | 3.47% | 4.80% | 4.86% | 6.60% | 4.07% | 5.36% |
| 0.10 | 1.20% | 4.94% | 6.00% | 7.69% | 9.00% | 6.13% | 7.06% |
| 0.20 | 1.20% | 7.72% | 8.40% | 13.58% | 15.00% | 10.23% | 10.75% |
| 0.30 | 1.20% | 10.54% | 11.40% | 19.98% | 21.00% | 14.57% | 15.20% |

At rho = 0 the loss distribution is purely idiosyncratic diversifiable risk and the analytic quantiles collapse to EL (the large-pool limit); the finite-pool MC still shows a small tail, which is exactly the granularity the limit ignores.

## Does the dependence *shape* matter? t-copula at rho = 0.20

Same marginals, same rho, same seed (common random numbers). The Student-t copula (nu = 4) adds tail dependence — a shared chi-square shock that can push many obligors toward default together — which the Gaussian copula rules out by construction.

| Copula | Mean loss | VaR99 | ES99 | VaR99.9 | ES99.9 |
|---|---|---|---|---|---|
| Gaussian | 1.20% | 8.40% | 10.75% | 15.00% | 17.65% |
| Student-t (nu=4) | 1.18% | 17.40% | 23.58% | 31.80% | 36.42% |

The means agree to within Monte Carlo noise; the 99.9% tail does not (VaR99.9 x2.12). Choosing a copula is choosing how bad the joint bad day can be.

## Limitations

- Single systematic factor and a single correlation for all pairs of obligors; real portfolios have sectors, concentrations and name-specific correlations.
- LGD is a constant 60%: no recovery risk, no PD-LGD correlation (in downturns the two tend to worsen together, fattening tails further).
- Static one-period model: no migration risk, no spreads, no funding or liquidity effects.
- MC tail estimates at the 99.9% level use 200,000 simulations (~200 tail observations), so the third digit of VaR99.9 is noise; the Vasicek column is the precise benchmark where it applies.
- Homogeneous PDs: heterogeneity changes the shape (the engine supports PD arrays; the case keeps the textbook pool).
