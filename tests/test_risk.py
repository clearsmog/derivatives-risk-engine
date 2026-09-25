import numpy as np
import pandas as pd
import pytest

from commodity_risk_engine import pnl_explain, var
from commodity_risk_engine.market import sample_market
from commodity_risk_engine.portfolio import build_book
from commodity_risk_engine.scenarios import synthetic_history
from commodity_risk_engine.sensitivities import leg_sensitivities


@pytest.fixture(scope="module")
def setup():
    mkt = sample_market()
    book = build_book(mkt)
    return mkt, book, leg_sensitivities(book, mkt)


def test_var_and_es_on_known_sample():
    # VaR is the ceil(n * 1%)-th worst loss: the worst of 100, the 5th worst of 500.
    pnl = -np.arange(1, 101, dtype=float)  # losses 1..100
    assert var.var(pnl, 0.99) == 100.0
    assert var.var(-np.arange(1, 501, dtype=float), 0.99) == 496.0
    assert var.expected_shortfall(pnl, 0.975) == pytest.approx((100 + 99) / 2)


def test_linear_legs_have_zero_cash_gamma_and_vega(setup):
    _, _, legs = setup
    linear = legs[~legs.is_option]
    assert np.allclose(linear.gamma_cash, 0.0, atol=1e-6 * linear.delta_cash.abs().max())
    assert (linear.vega == 0).all()


def test_taylor_matches_full_reval_for_small_moves(setup):
    mkt, book, legs = setup
    h = synthetic_history(n_days=100, seed=1, stress_start=None)
    full = var.scenario_pnl(book, mkt, h).sum()
    approx = var.taylor_pnl(legs, h).sum()
    assert np.corrcoef(full, approx)[0, 1] > 0.999


def test_pla_zones():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(250)
    assert pnl_explain.pla_test(x, x)["zone"] == "green"
    assert pnl_explain.pla_test(x, rng.standard_normal(250))["zone"] == "red"


def test_explain_components_add_up(setup):
    mkt, book, legs = setup
    path = synthetic_history(n_days=20, seed=3, stress_start=None)
    ex = pnl_explain.explain(book, mkt, legs, path)
    parts = ex[["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl"]].sum(axis=1)
    assert np.allclose(parts, ex.rtpl)
    assert np.allclose(ex.rtpl + ex.unexplained, ex.hpl)
    assert isinstance(ex, pd.DataFrame) and len(ex) == 20 * len(book)
