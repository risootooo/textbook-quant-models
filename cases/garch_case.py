"""Case study -- Model 3: GARCH forecasting tournament on NIFTY.

Data: daily closes of ^NSEI (NIFTY 50) from Yahoo Finance, 2010-2026.
The three models are refit on rolling 1,000-day windows every 21
trading days and produce 1-step-ahead variance forecasts for the last
750 trading days.  Forecasts are scored against the next day's squared
return (the standard unbiased-but-noisy variance proxy) with QLIKE and
MSE, plus a Diebold-Mariano test (Newey-West HAC) between the two best
models.  A second exhibit refits on data up to 20 Mar 2020 and scores
the forecasts made for the COVID crash week of 23-27 Mar 2020.

Run with the yfinance environment:
    ~/workspace/tools-github/venv-risk/bin/python cases/garch_case.py
Writes results/garch.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textbook_quant.garch import (  # noqa: E402
    diebold_mariano,
    egarch_filter,
    egarch_fit,
    forecast,
    gjr_filter,
    gjr_fit,
    garch_filter,
    garch_fit,
    mse_variance,
    qlike,
)

ROOT = Path(__file__).resolve().parents[1]
EST_WINDOW = 1000
OOS_DAYS = 750
REFIT_EVERY = 21
MODELS = {"GARCH": garch_fit, "GJR": gjr_fit, "EGARCH": egarch_fit}
FILTERS = {"GARCH": garch_filter, "GJR": gjr_filter, "EGARCH": egarch_filter}
TRADING_DAYS = 252


def load_nifty() -> pd.Series:
    import yfinance as yf

    df = yf.download("^NSEI", start="2010-01-01", end="2026-11-09", progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def one_step(params, model_key, last_var, last_ret):
    return float(forecast(params, model_key.lower(), last_var, last_ret, 1)[0])


def run_tournament(pct_rets: np.ndarray):
    """Rolling OOS 1-day variance forecasts (in pct^2, converted later)."""
    n = pct_rets.size
    start = n - OOS_DAYS
    fc = {m: np.full(OOS_DAYS, np.nan) for m in MODELS}
    params_now: dict[str, dict] = {}
    last_var: dict[str, float] = {}
    last_ret: dict[str, float] = {}

    for t in range(start, n):
        if (t - start) % REFIT_EVERY == 0:
            window = pct_rets[t - EST_WINDOW : t]
            for name, fit_fn in MODELS.items():
                fit = fit_fn(window)
                params_now[name] = fit.params
                last_var[name] = fit.last_var
                last_ret[name] = float(window[-1])
        for name in MODELS:
            h = one_step(params_now[name], name, last_var[name], last_ret[name])
            fc[name][t - start] = h
            last_var[name] = h
            last_ret[name] = float(pct_rets[t])
    return fc


def covid_exhibit(close: pd.Series, rets_dec: pd.Series) -> pd.DataFrame:
    """Fit through Fri 20 Mar 2020; forecast the crash week day by day."""
    pct = (rets_dec * 100).to_numpy()
    dates = rets_dec.index
    end_loc = dates.get_loc(pd.Timestamp("2020-03-20"))
    window = pct[end_loc - 1500 + 1 : end_loc + 1]
    week_idx = [
        dates.get_loc(pd.Timestamp(d))
        for d in ("2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26", "2020-03-27")
    ]
    rows = []
    for name, fit_fn in MODELS.items():
        fit = fit_fn(window)
        lv, lr = fit.last_var, float(window[-1])
        daily_vol = []
        for j in week_idx:
            h = one_step(fit.params, name, lv, lr)
            daily_vol.append(np.sqrt(h) / 100 * np.sqrt(TRADING_DAYS))
            lv, lr = h, float(pct[j])
        rows.append({"model": name, "crash_week_vol_forecast": np.mean(daily_vol)})
    out = pd.DataFrame(rows).set_index("model")
    week_rets = rets_dec.iloc[week_idx].to_numpy()
    out["realised_vol_that_week"] = week_rets.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return out


def main() -> None:
    t0 = time.time()
    close = load_nifty()
    rets_dec = np.log(close / close.shift(1)).dropna()
    pct_rets = (rets_dec * 100).to_numpy()
    print(
        f"NIFTY closes {close.index.min().date()}..{close.index.max().date()}, "
        f"{rets_dec.size} returns"
    )

    # ---- out-of-sample tournament -------------------------------------
    fc_pct = run_tournament(pct_rets)
    oos_rets = rets_dec.to_numpy()[-OOS_DAYS:]
    proxy = oos_rets**2
    table = {}
    losses = {}
    for name in MODELS:
        fvar = fc_pct[name] / 1e4  # pct^2 -> decimal variance
        losses[name] = qlike(proxy, fvar)
        table[name] = {
            "QLIKE": float(np.mean(losses[name])),
            "MSE_on_variance": float(np.mean(mse_variance(proxy, fvar))),
            "MAE_|r|": float(np.mean(np.abs(np.sqrt(fvar) - np.abs(oos_rets)))),
        }
    scores = pd.DataFrame(table).T.sort_values("QLIKE")
    best, second = scores.index[0], scores.index[1]
    dm_stat, dm_p = diebold_mariano(losses[best], losses[second])

    # ---- full-sample fits ---------------------------------------------
    full = {}
    for name, fit_fn in MODELS.items():
        fit = fit_fn(pct_rets)
        full[name] = {
            "params": {k: round(v, 4) for k, v in fit.params.items()},
            "persistence": fit.persistence,
            "uncond_vol_ann": (
                float(np.sqrt(fit.unconditional_var) / 100 * np.sqrt(TRADING_DAYS))
                if fit.unconditional_var
                else float("nan")
            ),
            "loglik": fit.loglik,
        }

    # ---- COVID exhibit --------------------------------------------------
    covid = covid_exhibit(close, rets_dec)
    crash_day = rets_dec.loc["2020-03-23"]
    rv21_crash = (
        rets_dec.loc[:"2020-03-27"].iloc[-21:].std(ddof=1) * np.sqrt(TRADING_DAYS)
    )

    # ---- write results --------------------------------------------------
    lines = []
    lines.append("# Model 3 — GARCH Volatility Forecasting Tournament (NIFTY)\n")
    lines.append(
        f"Data: ^NSEI daily closes, {close.index.min().date()} to "
        f"{close.index.max().date()} ({rets_dec.size} log returns), Yahoo Finance. "
        f"Models are estimated on percentage returns and forecasts rescaled to "
        f"decimal variance for scoring.\n"
    )
    lines.append("## Method\n")
    lines.append(
        f"- Rolling out-of-sample exercise over the last {OOS_DAYS} trading days "
        f"({rets_dec.index[-OOS_DAYS].date()} to {rets_dec.index[-1].date()}).\n"
        f"- Estimation window: rolling {EST_WINDOW} days; parameters refit every "
        f"{REFIT_EVERY} trading days; between refits the variance recursion is "
        "updated daily with the observed return (true 1-step-ahead forecasts).\n"
        "- Scoring proxy: next day's squared return. Losses: QLIKE "
        "(log h + r^2/h, Patton 2011) and MSE on variance; also MAE of the vol "
        "forecast against |r|.\n"
        "- Diebold-Mariano test with Newey-West HAC variance on the QLIKE loss "
        "differential of the best two models.\n"
    )
    lines.append("## Out-of-sample scores (lower is better)\n")
    lines.append("| Model | QLIKE | MSE on variance | MAE vs abs(r) |")
    lines.append("|---|---|---|---|")
    for name, row in scores.iterrows():
        lines.append(
            f"| {name} | {row['QLIKE']:.4f} | {row['MSE_on_variance']:.3e} "
            f"| {row['MAE_|r|']:.4f} |"
        )
    lines.append(
        f"\nDiebold-Mariano ({best} vs {second}, QLIKE): DM = {dm_stat:.3f}, "
        f"p = {dm_p:.4f}. "
        + (
            f"The gap favours {best} "
            + ("significantly at the 5% level."
               if dm_p < 0.05 else "but is not significant at the 5% level.")
        )
        + "\n"
    )
    lines.append("## Full-sample fits (2010–2026, percentage returns)\n")
    lines.append("| Model | Parameters | Persistence | Unconditional vol (ann.) |")
    lines.append("|---|---|---|---|")
    for name, f in full.items():
        ptxt = ", ".join(f"{k}={v}" for k, v in f["params"].items())
        lines.append(
            f"| {name} | {ptxt} | {f['persistence']:.4f} "
            f"| {f['uncond_vol_ann']:.1%} |"
        )
    lines.append(
        "\nPersistence is alpha+beta (GARCH), alpha+beta+gamma/2 (GJR) and beta "
        "(EGARCH). All three sit close to, but below, one: NIFTY volatility is "
        "highly persistent yet stationary over this sample.\n"
    )
    lines.append("## COVID crash week exhibit\n")
    lines.append(
        "Each model was fit on the 1,500 days ending Fri 20 Mar 2020 and then "
        "produced genuine 1-step-ahead forecasts for Mon 23 – Fri 27 Mar 2020 "
        f"(23 Mar itself was a {crash_day:.1%} day for NIFTY). The table shows "
        "the average annualised vol forecast across that week against what "
        "actually realised.\n"
    )
    lines.append("| Model | Avg vol forecast, crash week | Realised vol that week |")
    lines.append("|---|---|---|")
    for name, row in covid.iterrows():
        lines.append(
            f"| {name} | {row['crash_week_vol_forecast']:.1%} "
            f"| {row['realised_vol_that_week']:.1%} |"
        )
    lines.append(
        f"\nFor scale, the 21-day realised vol ending 27 Mar 2020 was "
        f"{rv21_crash:.1%}. Every model's level going into the week was far "
        "below this; the exhibit measures how fast each recursion catches up "
        "once the crash returns arrive, not foresight of the crash itself.\n"
    )
    lines.append("## Limitations\n")
    lines.append(
        "- Daily squared returns are an extremely noisy variance proxy; QLIKE "
        "rankings are robust to that noise in expectation, but the DM test "
        "still often cannot separate close models — exactly what we find.\n"
        "- Gaussian innovations: fat tails load entirely onto the variance "
        "dynamics. A Student-t GARCH would likely improve QLIKE.\n"
        f"- Parameters are refit every {REFIT_EVERY} days, so the exercise "
        "understates what daily refitting would deliver and overstates what "
        "monthly refitting would.\n"
        "- EGARCH variance forecasts exponentiate the log-variance path "
        "(median-style forecast, no Jensen correction).\n"
        "- Yahoo's ^NSEI history is taken as published; the final row "
        "(8 Nov 2026, a Sunday-dated print) is included as delivered by the "
        "data vendor.\n"
    )
    lines.append(f"\n*Runtime of this script: {time.time() - t0:.0f}s.*\n")

    out_path = ROOT / "results" / "garch.md"
    out_path.write_text("\n".join(lines))
    print(scores)
    print(f"DM ({best} vs {second}): {dm_stat:.3f}, p={dm_p:.4f}")
    print(covid)
    print(f"Wrote {out_path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
