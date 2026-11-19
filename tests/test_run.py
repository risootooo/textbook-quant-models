"""Tests for the one-command entry point (run.py).

Kept light: the sections that need an MLE/EM fit are exercised on a
small synthetic sample, and the analytic sections (Merton, credit,
bond curve) are checked against the closed-form properties the
library suites already establish.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_run():
    spec = importlib.util.spec_from_file_location("run", ROOT / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run"] = module
    spec.loader.exec_module(module)
    return module


def test_bond_curve_starts_at_par_and_decays():
    run = _load_run()
    curve = run.bond_curve()
    assert list(curve["tenor_years"]) == list(run.TENORS)
    assert curve.iloc[0]["vasicek_price"] < 1.0
    assert curve.iloc[0]["cir_price"] < 1.0
    # Longer tenor -> lower zero price at a positive short rate
    assert curve["vasicek_price"].is_monotonic_decreasing
    assert curve["cir_price"].is_monotonic_decreasing
    assert (curve["vasicek_yield_pct"] > 0).all()
    assert (curve["cir_yield_pct"] > 0).all()


def test_credit_rows_el_rho_invariant_and_var_rises_with_rho():
    run = _load_run()
    rows = run.credit_rows()
    el = [r for r in rows if r["quantity"] == "expected_loss_pct"]
    assert len(el) == 1
    assert el[0]["value"] == np.float64(1.2) or abs(el[0]["value"] - 1.2) < 1e-12
    vars_ = [r["value"] for r in rows if r["quantity"].startswith("var999")]
    assert len(vars_) == 4
    assert vars_ == sorted(vars_)
    # At rho = 0 the 99.9% quantile collapses onto expected loss
    assert abs(vars_[0] - 1.2) < 1e-9


def test_merton_row_premium_is_finite_and_detail_mentions_bs():
    run = _load_run()
    row = run.merton_row(25_000.0)
    assert np.isfinite(row["value"]) and row["value"] > 0
    assert "BS(sigma=13.82%)" in row["detail"]


def test_regimes_and_garch_rows_on_synthetic_sample():
    run = _load_run()
    rng = np.random.default_rng(7)
    calm = rng.normal(0.0004, 0.007, 300)
    turbulent = rng.normal(-0.001, 0.025, 60)
    rets = np.concatenate([calm, turbulent, calm])
    rows = run.regimes_rows(rets)
    assert {r["quantity"] for r in rows} == {
        "calm_ann_vol_pct",
        "turbulent_ann_vol_pct",
        "turbulent_share_pct",
    }
    by_q = {r["quantity"]: r["value"] for r in rows}
    assert by_q["turbulent_ann_vol_pct"] > by_q["calm_ann_vol_pct"]
    grows = run.garch_rows(rets * 100)
    assert len(grows) == 3
    assert all(np.isfinite(r["value"]) and r["value"] > 0 for r in grows)
