import numpy as np
import pandas as pd
import pytest

from derivatives_risk_engine import pnl_explain, var
from derivatives_risk_engine.market import sample_market
from derivatives_risk_engine.portfolio import build_book
from derivatives_risk_engine.scenarios import synthetic_history
from derivatives_risk_engine.sensitivities import sensitivities


@pytest.fixture(scope="module")
def setup():
    mkt = sample_market()
    book = build_book(mkt)
    legs, sens = sensitivities(book, mkt)
    return mkt, book, legs, sens


def test_var_and_es_on_known_sample():
    # VaR is the loss at rank floor((n - 1) * 1%): the worst of 100, the 5th worst of 500.
    assert var.var(-np.arange(1, 101, dtype=float), 0.99) == 100.0
    assert var.var(-np.arange(1, 501, dtype=float), 0.99) == 496.0
    assert var.expected_shortfall(
        -np.arange(1, 101, dtype=float), 0.975
    ) == pytest.approx(99.5)


def test_linear_commodity_legs_have_zero_cash_gamma(setup):
    _, book, legs, sens = setup
    linear = legs[legs.leg_type == "LinearLeg"][["trade_id", "leg"]]
    rows = sens.merge(linear)
    rows = rows[rows.kind == "price"]
    assert len(rows) and np.allclose(rows.d2, 0.0, atol=1e-6 * rows.d1.abs().max())


def test_every_desk_has_sensitivities(setup):
    _, book, _, sens = setup
    classes = set(sens.asset_class)
    assert {"CMDTY", "EQ", "FX", "IR", "CS"} <= classes


def test_taylor_matches_full_reval_for_small_moves(setup):
    mkt, book, _, sens = setup
    h = synthetic_history(mkt, n_days=100, seed=1, stress_start=None)
    full = var.scenario_pnl(book, mkt, h).sum()
    approx = var.taylor_pnl(sens, h).sum()
    assert np.corrcoef(full, approx)[0, 1] > 0.999


def test_pla_zones():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(250)
    assert pnl_explain.pla_test(x, x)["zone"] == "green"
    assert pnl_explain.pla_test(x, rng.standard_normal(250))["zone"] == "red"


def test_explain_components_add_up(setup):
    mkt, book, legs, sens = setup
    path = synthetic_history(mkt, n_days=20, seed=3, stress_start=None)
    ex = pnl_explain.explain(book, mkt, legs, sens, path)
    parts = ex[["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl"]].sum(axis=1)
    assert np.allclose(parts, ex.rtpl)
    assert np.allclose(ex.rtpl + ex.unexplained, ex.hpl)
    assert isinstance(ex, pd.DataFrame) and len(ex) == 20 * len(book)


def test_naive_proxy_copies_anchor_onto_curve(setup):
    mkt, *_ = setup
    h = synthetic_history(mkt, n_days=50, seed=5, stress_start=None)
    p = pnl_explain.proxied(mkt, h, pnl_explain.proxy_model(mkt, h, naive=True))
    usd = p.moves["IR:USD"]
    assert np.allclose(usd, usd[:, [mkt["IR:USD"].anchor]])
    assert np.allclose(
        p.moves["EQ:SPX"], h.moves["EQ:SPX"]
    )  # single-pillar factors untouched
