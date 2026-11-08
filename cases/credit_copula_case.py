"""Case study -- Model 4: credit portfolio copula (ILLUSTRATIVE).

A deliberately textbook simulation study: a homogeneous portfolio of
100 obligors, each with PD = 2% and LGD = 60%.  Asset correlation rho
is swept from 0 to 0.30.  Expected loss does not depend on rho at all;
the tail of the loss distribution multiplies.  Analytic (Vasicek
large-pool) numbers are shown next to Gaussian-copula Monte Carlo for
the finite 100-name pool, and a Student-t copula (nu = 4) is compared
at rho = 0.20 to show what tail dependence adds.

These parameters are illustrative teaching values, NOT calibrated to
any real portfolio, bank, or market.  Nothing here is a claim about
real-world credit losses.

Run:
    /tmp/venv-test/bin/python cases/credit_copula_case.py
Writes results/credit_copula.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textbook_quant.credit_copula import (  # noqa: E402
    expected_loss,
    gaussian_copula_mc,
    portfolio_var_es,
    t_copula_mc,
    vasicek_expected_shortfall,
    vasicek_loss_quantile,
)

ROOT = Path(__file__).resolve().parents[1]
N_OBLIGORS = 100
PD = 0.02
LGD = 0.60
N_SIMS = 200_000
SEED = 20261108
RHOS = (0.0, 0.05, 0.10, 0.20, 0.30)


def pct(x: float) -> str:
    return f"{x:.2%}"


def main() -> None:
    el = expected_loss(PD, LGD)
    rows = []
    for rho in RHOS:
        v = {
            "EL": el,
            "VaR99": vasicek_loss_quantile(PD, rho, LGD, 0.99),
            "VaR999": vasicek_loss_quantile(PD, rho, LGD, 0.999),
            "ES99": vasicek_expected_shortfall(PD, rho, LGD, 0.99),
        }
        losses = gaussian_copula_mc(N_OBLIGORS, PD, rho, LGD, N_SIMS, SEED)
        m99 = portfolio_var_es(losses, 0.99)
        m999 = portfolio_var_es(losses, 0.999)
        m = {
            "EL": float(losses.mean()),
            "VaR99": m99[0],
            "ES99": m99[1],
            "VaR999": m999[0],
            "ES999": m999[1],
        }
        rows.append((rho, v, m))
        print(f"rho={rho:>4}: Vasicek VaR999={v['VaR999']:.4f}  MC={m['VaR999']:.4f}")

    # Student-t comparison at rho = 0.20
    kw = dict(n_obligors=N_OBLIGORS, pd=PD, rho=0.20, lgd=LGD,
              n_sims=N_SIMS, seed=SEED)
    g = gaussian_copula_mc(**kw)
    t = t_copula_mc(nu=4.0, **kw)
    g99, g999 = portfolio_var_es(g, 0.99), portfolio_var_es(g, 0.999)
    t99, t999 = portfolio_var_es(t, 0.99), portfolio_var_es(t, 0.999)

    lines = []
    lines.append("# Model 4 — Credit Portfolio Copula (Illustrative Simulation Study)\n")
    lines.append(
        "> **Scope label.** This is a simulation study with illustrative "
        "parameters (100 obligors, PD 2%, LGD 60%, equal notionals). It is "
        "not calibrated to any real portfolio and makes no claim about "
        "real-world losses. Its purpose is to isolate one mechanism: what "
        "asset correlation does to a credit portfolio's loss distribution.\n"
    )
    lines.append("## The one-line result\n")
    v0 = rows[0][1]
    v30 = rows[-1][1]
    lines.append(
        f"Expected loss is {pct(el)} at *every* correlation — correlation "
        f"cannot move the mean of the loss distribution, only its shape. "
        f"But the 99.9% VaR rises from {pct(v0['VaR999'])} at rho = 0 to "
        f"{pct(v30['VaR999'])} at rho = 0.30 "
        f"(x{v30['VaR999'] / max(v0['VaR999'], 1e-12):.1f}), and ES99 from "
        f"{pct(v0['ES99'])} to {pct(v30['ES99'])}. Correlation is a tail "
        "phenomenon, not a mean phenomenon.\n"
    )
    lines.append("## Vasicek large-pool (analytic) vs Gaussian-copula Monte Carlo\n")
    lines.append(
        f"Monte Carlo: finite pool of {N_OBLIGORS} obligors, {N_SIMS:,} "
        f"simulations, seed {SEED}. The analytic column is the infinitely "
        "granular limit, so the MC tail sits slightly *above* it: a "
        "100-name pool keeps idiosyncratic risk that the limit assumes away.\n"
    )
    lines.append(
        "| rho | EL (both) | VaR99 Vasicek | VaR99 MC | VaR99.9 Vasicek "
        "| VaR99.9 MC | ES99 Vasicek | ES99 MC |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for rho, v, m in rows:
        lines.append(
            f"| {rho:.2f} | {pct(v['EL'])} | {pct(v['VaR99'])} | {pct(m['VaR99'])} "
            f"| {pct(v['VaR999'])} | {pct(m['VaR999'])} | {pct(v['ES99'])} "
            f"| {pct(m['ES99'])} |"
        )
    lines.append(
        "\nAt rho = 0 the loss distribution is purely idiosyncratic "
        "diversifiable risk and the analytic quantiles collapse to EL "
        "(the large-pool limit); the finite-pool MC still shows a small "
        "tail, which is exactly the granularity the limit ignores.\n"
    )
    lines.append("## Does the dependence *shape* matter? t-copula at rho = 0.20\n")
    lines.append(
        "Same marginals, same rho, same seed (common random numbers). The "
        "Student-t copula (nu = 4) adds tail dependence — a shared "
        "chi-square shock that can push many obligors toward default "
        "together — which the Gaussian copula rules out by construction.\n"
    )
    lines.append("| Copula | Mean loss | VaR99 | ES99 | VaR99.9 | ES99.9 |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| Gaussian | {pct(g.mean())} | {pct(g99[0])} | {pct(g99[1])} "
        f"| {pct(g999[0])} | {pct(g999[1])} |"
    )
    lines.append(
        f"| Student-t (nu=4) | {pct(t.mean())} | {pct(t99[0])} | {pct(t99[1])} "
        f"| {pct(t999[0])} | {pct(t999[1])} |"
    )
    lines.append(
        f"\nThe means agree to within Monte Carlo noise; the 99.9% tail "
        f"does not (VaR99.9 x{t999[0] / g999[0]:.2f}). Choosing a copula is "
        "choosing how bad the joint bad day can be.\n"
    )
    lines.append("## Limitations\n")
    lines.append(
        "- Single systematic factor and a single correlation for all pairs "
        "of obligors; real portfolios have sectors, concentrations and "
        "name-specific correlations.\n"
        "- LGD is a constant 60%: no recovery risk, no PD-LGD correlation "
        "(in downturns the two tend to worsen together, fattening tails "
        "further).\n"
        "- Static one-period model: no migration risk, no spreads, no "
        "funding or liquidity effects.\n"
        f"- MC tail estimates at the 99.9% level use {N_SIMS:,} simulations "
        "(~200 tail observations), so the third digit of VaR99.9 is noise; "
        "the Vasicek column is the precise benchmark where it applies.\n"
        "- Homogeneous PDs: heterogeneity changes the shape (the engine "
        "supports PD arrays; the case keeps the textbook pool).\n"
    )
    out_path = ROOT / "results" / "credit_copula.md"
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
