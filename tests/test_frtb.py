"""Hand-computed checks of the MAR21 aggregation, one risk class at a time."""

from functools import partial

import numpy as np
import pandas as pd
import pytest

from derivatives_risk_engine import frtb
from derivatives_risk_engine.instruments import SwapLeg, SwaptionLeg, Trade, asian_strip
from derivatives_risk_engine.market import sample_market
from derivatives_risk_engine.sensitivities import sensitivities

RC = frtb.RISK_CLASSES


def ws(cls, rows):
    """rows: (name, tenor, sensitivity) -> weighted sensitivities for class cls."""
    rc = RC[cls]
    out = []
    for name, tenor, s in rows:
        b = rc.bucket(name)
        out.append(("T", cls, b, name, tenor, 0.0, s, rc.delta_rw(b, tenor) * s))
    cols = [
        "trade_id",
        "risk_class",
        "bucket",
        "name",
        "tenor",
        "tenor2",
        "sensitivity",
        "weighted_sensitivity",
    ]
    return pd.DataFrame(out, columns=cols)


def delta_charge(cls, rows, scenario="medium", distinct=True):
    rc = RC[cls]
    return frtb._linear_charge(
        ws(cls, rows), rc, partial(rc.rho_delta, distinct=distinct), scenario
    )[0]


@pytest.mark.parametrize(
    "cls,name,tenor,rw",
    [
        ("CMDTY", "BRENT", 1.0, 0.35),
        ("CMDTY", "TTF", 1.0, 0.45),
        ("GIRR", "USD", 0.25, 0.017),
        ("GIRR", "USD", 10.0, 0.011),
        ("CSR", "ACME", 5.0, 0.03),
        ("CSR", "ZETA", 5.0, 0.055),
        ("CSR", "CDXIG", 5.0, 0.015),
        ("EQ", "SPX", 0.0, 0.15),
        ("EQ", "NWT", 0.0, 0.50),
        ("FX", "EUR", 0.0, 0.15),
    ],
)
def test_single_sensitivity_is_risk_weight_times_size(cls, name, tenor, rw):
    assert delta_charge(cls, [(name, tenor, -100.0)]) == pytest.approx(100 * rw)


def test_girr_table_matches_basel_formula():
    t = frtb.GIRR_TENORS
    formula = np.maximum(
        np.exp(-0.03 * np.abs(t[:, None] - t[None, :]) / np.minimum.outer(t, t)), 0.40
    )
    assert np.allclose(frtb.GIRR_RHO, formula, atol=5e-4)


def test_girr_two_tenors_and_two_currencies():
    w1, w2 = 0.011 * 100, 0.012 * -80
    within = np.sqrt(w1**2 + w2**2 + 2 * 0.980 * w1 * w2)  # 5y vs 3y
    assert delta_charge(
        "GIRR", [("USD", 5.0, 100.0), ("USD", 3.0, -80.0)]
    ) == pytest.approx(within)
    across = np.sqrt(1.1**2 + 1.1**2 + 2 * 0.5 * 1.1 * 1.1)
    assert delta_charge(
        "GIRR", [("USD", 5.0, 100.0), ("EUR", 5.0, 100.0)]
    ) == pytest.approx(across)


def test_commodity_tenor_and_bucket_correlations():
    w1, w2 = 35.0, -28.0
    expected = np.sqrt(w1**2 + w2**2 + 2 * 0.99 * w1 * w2)
    assert delta_charge(
        "CMDTY", [("BRENT", 1.0, 100.0), ("BRENT", 2.0, -80.0)]
    ) == pytest.approx(expected)
    cross = np.sqrt(35.0**2 + 45.0**2 + 2 * 0.20 * 35.0 * 45.0)
    assert delta_charge(
        "CMDTY", [("BRENT", 1.0, 100.0), ("TTF", 1.0, 100.0)]
    ) == pytest.approx(cross)


def test_ttf_jkm_distinct_vs_one_commodity():
    rows = [("TTF", 1.0, 100.0), ("JKM", 1.0, -100.0)]
    for sc, rho in [("medium", 0.65), ("high", 0.8125), ("low", 0.4875)]:
        assert delta_charge("CMDTY", rows, sc, distinct=True) == pytest.approx(
            np.sqrt(2 * 45.0**2 * (1 - rho))
        )
    assert delta_charge("CMDTY", rows, distinct=False) == pytest.approx(
        np.sqrt(2 * 45.0**2 * (1 - 0.999))
    )


def test_csr_name_tenor_and_cross_bucket():
    # Same name, two tenors (rho_tenor 65%), then the cross-bucket gammas.
    w1, w2 = 3.0, -3.0
    assert delta_charge(
        "CSR", [("ACME", 5.0, 100.0), ("ACME", 3.0, -100.0)]
    ) == pytest.approx(np.sqrt(w1**2 + w2**2 + 2 * 0.65 * w1 * w2))
    assert RC["CSR"].gamma(4, 14) == pytest.approx(
        0.5 * 0.25
    )  # IG vs HY, industrials vs telecom
    assert RC["CSR"].gamma(4, 17) == pytest.approx(0.45)
    assert (
        RC["CSR"].name_rho(4, "A", "B", True) == 0.35
        and RC["CSR"].name_rho(17, "A", "B", True) == 0.80
    )


def test_equity_correlations():
    rc = RC["EQ"]
    assert rc.name_rho(12, "SPX", "SX5E", True) == 0.80
    assert (
        rc.gamma(8, 12) == 0.45 and rc.gamma(12, 13) == 0.75 and rc.gamma(3, 7) == 0.15
    )
    w = 0.15 * 100
    assert delta_charge(
        "EQ", [("SPX", 0.0, 100.0), ("SX5E", 0.0, 100.0)]
    ) == pytest.approx(np.sqrt(2 * w * w * (1 + 0.80)))


def test_fx_cross_currency_gamma():
    assert RC["FX"].gamma("EUR", "JPY") == 0.60


def test_vega_risk_weights():
    assert RC["EQ"].vega_rw(12) == pytest.approx(0.55 * np.sqrt(2))  # 77.78%
    assert RC["EQ"].vega_rw(9) == 1.0
    assert RC["GIRR"].vega_rw("USD") == 1.0 and RC["FX"].vega_rw("EUR") == 1.0


def test_vega_correlation_decay_including_girr_underlying():
    rc = RC["GIRR"]
    rho = rc.rho_vega("USD", ("USD", 1.0, 5.0), ("USD", 3.0, 10.0), True)
    assert rho == pytest.approx(np.exp(-0.01 * 2 / 1) * np.exp(-0.01 * 5 / 5))


def test_curvature_bucket_aggregation_by_hand():
    cvr = pd.DataFrame(
        [("T1", "CMDTY", 6, "TTF", 10.0, -5.0), ("T2", "CMDTY", 6, "JKM", 4.0, -2.0)],
        columns=["trade_id", "risk_class", "bucket", "name", "cvr_up", "cvr_down"],
    )
    k, kb = frtb._curvature_charge(cvr, RC["CMDTY"], "medium", distinct=True)
    expected = np.sqrt(10**2 + 4**2 + 2 * 0.65**2 * 10 * 4)
    assert kb[6] == pytest.approx(expected) and k == pytest.approx(expected)


def test_curvature_psi_drops_two_negative_cvrs():
    cvr = pd.DataFrame(
        [("T1", "CMDTY", 6, "TTF", -10.0, -6.0), ("T2", "CMDTY", 6, "JKM", -4.0, -3.0)],
        columns=["trade_id", "risk_class", "bucket", "name", "cvr_up", "cvr_down"],
    )
    k, kb = frtb._curvature_charge(cvr, RC["CMDTY"], "medium", distinct=True)
    assert kb[6] == 0.0 and k == 0.0


def test_tenor_allocation_is_linear():
    assert frtb.allocate(0.75, frtb.CMDTY_TENORS) == [(0.5, 0.5), (1.0, 0.5)]
    assert frtb.allocate(40.0, frtb.CMDTY_TENORS) == [(30.0, 1.0)]
    assert frtb.allocate(0.1, frtb.VEGA_TENORS) == [(0.5, 1.0)]


def _run(trades, mkt):
    _, sens = sensitivities(trades, mkt)
    return frtb.run_sbm(trades, mkt, sens)


def test_linear_trades_have_no_curvature_or_vega():
    mkt = sample_market()
    t = Trade(
        "S", "swap", "D", "B", "C", "bank", "IR", (SwapLeg("USD", 0, 10, 0.04, 1e8),)
    )
    res = _run([t], mkt)
    assert res.by_scenario.vega.max() == 0 and res.by_scenario.curvature.max() == 0
    assert res.by_class.index.tolist() == ["GIRR"]


def test_short_options_attract_curvature_long_options_do_not():
    mkt = sample_market()
    for side, charged in [(-1, True), (1, False)]:
        legs = tuple(asian_strip("TTF", range(1, 13), 1e6, 12.0, True, side))
        res = _run([Trade("O", "opt", "D", "B", "C", "bank", "CMDTY", legs)], mkt)
        cmdty = res.detail[
            (res.detail.risk_class == "CMDTY") & (res.detail.measure == "curvature")
        ]
        assert (cmdty.charge.max() > 1e-6) == charged


def test_swaption_carries_girr_vega_and_capital_is_max_over_scenarios():
    mkt = sample_market()
    t = Trade(
        "W",
        "swaption",
        "D",
        "B",
        "C",
        "bank",
        "IR",
        (SwaptionLeg("USD", 1.0, 10.0, 0.04, False, -1e8),),
    )
    res = _run([t], mkt)
    assert res.by_scenario.vega.max() > 0
    assert res.capital == pytest.approx(res.by_scenario.total.max())
