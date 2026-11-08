# Case — Merton jump-diffusion vs Black-Scholes on a real NIFTY smile

**Data.** NSE F&O bhavcopy, trade date 3 Nov 2026 (settlement prices).
NIFTY options, expiry 2026-11-23 (20 calendar days out), spot
23,810.85, risk-free rate 6.5%. Out-of-the-money options only (calls
above spot, puts below), restricted to contracts with trading volume
that day: **121 options** used.

**Method.** Both models fitted by least squares on prices. Fit quality
is measured in implied-volatility space — the Black-Scholes IV of each
model price vs the market IV — reported as RMSE in volatility points.

## Results

| Model | IV RMSE (vol pts) | Fitted parameters |
|---|---|---|
| Black-Scholes (single vol) | 8.99 | sigma = 13.82% |
| Merton jump-diffusion | 2.54 | sigma = 2.83%, lambda = 10.12 jumps/yr, mu_j = -2.2% per jump, sigma_j = 4.2% |

The jump-diffusion fit cuts IV RMSE by **6.45 vol
points (72% lower)**. The economics are in
the parameters: a modest diffusion vol plus roughly 10 downward
jumps a year averaging -2% reproduces the downside skew
that a single Black-Scholes vol can only average over — the flat-vol
model overprices the wings' centre and underprices crash protection.

**What the settlement smile actually looks like.** In this file the
put side carries the skew: near-the-money OTM puts imply roughly
16-17% vol, rising into the downside wing, while near-the-money OTM
calls imply only about 11%. (Put-call parity on the same strikes is
coherent — the implied forward sits within ~40 points of spot — so
this is a skew in the marks, not a data error.) Much of the
Black-Scholes RMSE is this put/call asymmetry, which no single vol
can fit; even the jump model cannot fully reproduce a step in the
marks at the money, which is why its RMSE is 2.54 and not lower.

## Limitations

- **Settlement prices, not trades.** Bhavcopy ClsPric is the exchange
  settlement price; for illiquid strikes it is a theoretical mark, not a
  transaction. Volume filtering mitigates but does not remove this.
- **Single expiry.** A 20-day smile identifies the jump parameters of
  one slice only; the term structure (how skew decays with maturity) is
  where jump vs stochastic-vol models are really separated.
- **Constant intensity, i.i.d. jumps.** Real crash risk clusters; the
  fitted lambda is an average, and the model has no vol-of-vol channel.
- Fit is on prices, judged on IVs; deep-out-of-the-money points carry
  wide bid-ask spreads in vol terms.
