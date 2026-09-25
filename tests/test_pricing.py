import numpy as np
import pytest

from commodity_risk_engine.instruments import (
    EuropeanLeg,
    collar,
    solve_par_fixed,
    solve_zero_cost_put,
    swap,
)
from commodity_risk_engine.market import fixing_times, sample_market
from commodity_risk_engine.pricing import asian_levy, asian_mc, black76, black76_greeks


def test_black76_put_call_parity():
    f, k, v, t, df = 80.0, 75.0, 0.3, 0.5, 0.98
    c = black76(f, k, v, t, df, True)
    p = black76(f, k, v, t, df, False)
    assert c - p == pytest.approx(df * (f - k), rel=1e-12)


def test_black76_zero_vol_is_discounted_intrinsic():
    assert black76(80, 75, 0.0, 1.0, 0.9, True) == pytest.approx(0.9 * 5)
    assert black76(80, 75, 0.3, 0.0, 0.9, False) == 0.0


@pytest.mark.parametrize("strike", [9.0, 11.0, 13.0])
@pytest.mark.parametrize("is_call", [True, False])
def test_asian_levy_close_to_monte_carlo(strike, is_call):
    t = fixing_times(3)
    fwds, vols = np.full(len(t), 11.0), np.full(len(t), 0.6)
    approx = asian_levy(fwds, vols, t, strike, 0.97, is_call)
    mc, se = asian_mc(fwds, vols, t, strike, 0.97, is_call)
    # Moment matching carries a small, known bias; allow it on top of MC noise.
    assert abs(approx - mc) < 4 * se + 0.01 * mc + 1e-3


def test_asian_with_single_fixing_equals_black76():
    t = np.array([0.5])
    a = asian_levy(np.array([80.0]), np.array([0.3]), t, 82.0, 0.98, True)
    assert a == pytest.approx(black76(80.0, 82.0, 0.3, 0.5, 0.98, True), rel=1e-10)


def test_averaging_lowers_option_value():
    t = fixing_times(6)
    avg = asian_levy(np.full(len(t), 80.0), np.full(len(t), 0.3), t, 80.0, 1.0, True)
    euro = black76(80.0, 80.0, 0.3, t[-1], 1.0, True)
    assert avg < euro


def test_par_swap_and_zero_cost_collar_are_worth_zero():
    mkt = sample_market()
    months = range(1, 13)
    k = solve_par_fixed(mkt, "TTF", months)
    assert sum(l.pv(mkt) for l in swap("TTF", months, 1e6, k, 1)) == pytest.approx(
        0, abs=1e-3
    )
    put_k = solve_zero_cost_put(mkt, "TTF", months, 14.0)
    assert put_k < 14.0
    assert sum(
        l.pv(mkt) for l in collar("TTF", months, 1e6, put_k, 14.0, 1)
    ) == pytest.approx(0, abs=1e-3)


def test_european_leg_matches_analytic_greeks():
    from commodity_risk_engine.sensitivities import leg_sensitivities
    from commodity_risk_engine.instruments import Trade

    mkt = sample_market()
    leg = EuropeanLeg("BRENT", 6, 1.0, 80.0, True)
    s = leg_sensitivities([Trade("T", "opt", "B", "C", "bank", (leg,))], mkt).iloc[0]
    c = mkt.curve("BRENT")
    f, v = c.fwd[6], c.vol[6]
    g = black76_greeks(f, 80.0, v, leg.expiry, mkt.df(leg.expiry), True)
    assert s.delta_ln == pytest.approx(f * g["delta"], rel=1e-3)
    assert s.gamma_cash == pytest.approx(f * f * g["gamma"], rel=1e-3)
    assert s.vega == pytest.approx(g["vega"], rel=1e-4)
