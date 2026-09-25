"""Hand-computed checks of the MAR21 commodity aggregation."""

from functools import partial

import numpy as np
import pandas as pd
import pytest

from commodity_risk_engine import frtb
from commodity_risk_engine.instruments import Trade, asian_strip, swap
from commodity_risk_engine.market import sample_market
from commodity_risk_engine.sensitivities import leg_sensitivities


def ws(rows):
    """rows: (commodity, tenor, sensitivity) -> weighted-sensitivity frame."""
    out = []
    for c, tenor, s in rows:
        b = frtb.BUCKET[c]
        out.append(("T", c, b, tenor, s, frtb.DELTA_RW[b] * s))
    return pd.DataFrame(
        out,
        columns=[
            "trade_id",
            "commodity",
            "bucket",
            "tenor",
            "sensitivity",
            "weighted_sensitivity",
        ],
    )


NO_CVR = pd.DataFrame(columns=["trade_id", "commodity", "bucket", "cvr_up", "cvr_down"])
NO_VEGA = ws([]).iloc[0:0]


def delta_charge(rows, scenario="medium", distinct=True):
    rho = partial(frtb._delta_rho, distinct=distinct)
    return frtb._linear_charge(ws(rows), rho, scenario)[0]


def test_single_sensitivity_is_risk_weight_times_size():
    assert delta_charge([("BRENT", 1.0, 100.0)]) == pytest.approx(35.0)
    assert delta_charge([("TTF", 1.0, -100.0)]) == pytest.approx(45.0)


def test_same_commodity_different_tenor_uses_99pct():
    w1, w2 = 0.35 * 100, 0.35 * -80
    expected = np.sqrt(w1**2 + w2**2 + 2 * 0.99 * w1 * w2)
    assert delta_charge(
        [("BRENT", 1.0, 100.0), ("BRENT", 2.0, -80.0)]
    ) == pytest.approx(expected)


def test_ttf_jkm_distinct_commodities_use_bucket_rho():
    w1, w2 = 45.0, -45.0
    for sc, rho in [("medium", 0.65), ("high", 0.8125), ("low", 0.4875)]:
        expected = np.sqrt(w1**2 + w2**2 + 2 * rho * w1 * w2)
        rows = [("TTF", 1.0, 100.0), ("JKM", 1.0, -100.0)]
        assert delta_charge(rows, sc, distinct=True) == pytest.approx(expected)


def test_ttf_jkm_as_one_commodity_use_basis_rho():
    w1, w2 = 45.0, -45.0
    expected = np.sqrt(w1**2 + w2**2 + 2 * 0.999 * w1 * w2)
    rows = [("TTF", 1.0, 100.0), ("JKM", 1.0, -100.0)]
    assert delta_charge(rows, distinct=False) == pytest.approx(expected)


def test_curvature_bucket_aggregation_by_hand():
    # Two positive CVRs in bucket 6 (distinct): Kb = sqrt(a^2 + b^2 + 2 rho^2 a b).
    cvr = pd.DataFrame(
        [("T1", "TTF", 6, 10.0, -5.0), ("T2", "JKM", 6, 4.0, -2.0)],
        columns=["trade_id", "commodity", "bucket", "cvr_up", "cvr_down"],
    )
    k, kb = frtb._curvature_charge(cvr, "medium", distinct=True)
    expected = np.sqrt(10**2 + 4**2 + 2 * 0.65**2 * 10 * 4)
    assert kb[6] == pytest.approx(expected) and k == pytest.approx(expected)


def test_curvature_psi_drops_two_negative_cvrs():
    cvr = pd.DataFrame(
        [("T1", "TTF", 6, -10.0, -6.0), ("T2", "JKM", 6, -4.0, -3.0)],
        columns=["trade_id", "commodity", "bucket", "cvr_up", "cvr_down"],
    )
    k, kb = frtb._curvature_charge(cvr, "medium", distinct=True)
    assert kb[6] == 0.0 and k == 0.0


def test_vega_maturity_decay():
    rho = frtb._vega_rho(6, ("TTF", 0.5), ("TTF", 1.0), distinct=True)
    assert rho == pytest.approx(np.exp(-0.01 * 0.5 / 0.5))


def test_cross_bucket_gamma_is_20pct():
    kb2, kb6 = 35.0, 45.0
    expected = np.sqrt(kb2**2 + kb6**2 + 2 * 0.20 * kb2 * kb6)
    assert delta_charge([("BRENT", 1.0, 100.0), ("TTF", 1.0, 100.0)]) == pytest.approx(
        expected
    )


def test_alternative_sb_when_cross_term_negative():
    # Hedged buckets: Kb small, S_b large and opposite -> use capped S_b.
    rows = [("BRENT", 1.0, 100.0), ("BRENT", 1.5, -100.0), ("TTF", 1.0, -1.0)]
    assert delta_charge(rows) >= 0.0


def test_tenor_allocation_is_linear():
    assert frtb.allocate(0.75, frtb.DELTA_TENORS) == [(0.5, 0.5), (1.0, 0.5)]
    assert frtb.allocate(40.0, frtb.DELTA_TENORS) == [(30.0, 1.0)]
    assert frtb.allocate(0.1, frtb.VEGA_TENORS) == [(0.5, 1.0)]


def test_vega_risk_weight_is_capped_at_100pct():
    assert frtb.VEGA_RW == 1.0


def test_linear_book_has_no_curvature_or_vega():
    mkt = sample_market()
    trades = [
        Trade(
            "S",
            "swap",
            "B",
            "C",
            "bank",
            tuple(swap("BRENT", range(1, 13), 1e5, 75.0, 1)),
        )
    ]
    legs = leg_sensitivities(trades, mkt)
    cvr = frtb.curvature_cvr(trades, mkt, legs)
    assert np.allclose(cvr[["cvr_up", "cvr_down"]].to_numpy(), 0.0, atol=1e-3)
    assert frtb.vega_weighted_sensitivities(legs).empty


def test_short_options_attract_curvature_long_options_do_not():
    mkt = sample_market()
    for side, charged in [(-1, True), (1, False)]:
        trades = [
            Trade(
                "O",
                "opt",
                "B",
                "C",
                "bank",
                tuple(asian_strip("TTF", range(1, 13), 1e6, 12.0, True, side)),
            )
        ]
        legs = leg_sensitivities(trades, mkt)
        res = frtb.sbm_commodity(
            frtb.delta_weighted_sensitivities(legs),
            frtb.vega_weighted_sensitivities(legs),
            frtb.curvature_cvr(trades, mkt, legs),
        )
        assert (res.by_scenario.curvature.max() > 0) == charged


def test_capital_is_max_over_scenarios():
    mkt = sample_market()
    trades = [
        Trade(
            "O",
            "opt",
            "B",
            "C",
            "bank",
            tuple(asian_strip("JKM", range(1, 7), 1e6, 12.0, False, -1)),
        )
    ]
    legs = leg_sensitivities(trades, mkt)
    res = frtb.sbm_commodity(
        frtb.delta_weighted_sensitivities(legs),
        frtb.vega_weighted_sensitivities(legs),
        frtb.curvature_cvr(trades, mkt, legs),
    )
    assert res.capital == pytest.approx(res.by_scenario.total.max())
