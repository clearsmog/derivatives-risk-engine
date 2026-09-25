import numpy as np
import pytest

from derivatives_risk_engine.instruments import (
    CDSLeg,
    DownAndOutCallLeg,
    EquityForwardLeg,
    EquityOptionLeg,
    EuropeanLeg,
    FXForwardLeg,
    FXOptionLeg,
    SwapLeg,
    SwaptionLeg,
    Trade,
    annuity,
    collar,
    forward_swap_rate,
    par_rate,
    par_spread,
    solve_par_fixed,
    solve_zero_cost_put,
    swap,
)
from derivatives_risk_engine.market import fixing_times, sample_market
from derivatives_risk_engine.pricing import (
    asian_levy,
    asian_mc,
    bachelier,
    barrier_mc,
    black76,
    black76_greeks,
    down_and_out_call,
)
from derivatives_risk_engine.sensitivities import sensitivities


@pytest.fixture(scope="module")
def mkt():
    return sample_market()


# ---- Black-76 and Asians ---------------------------------------------------------


def test_black76_put_call_parity():
    f, k, v, t, df = 80.0, 75.0, 0.3, 0.5, 0.98
    c, p = black76(f, k, v, t, df, True), black76(f, k, v, t, df, False)
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
    a = asian_levy(np.array([80.0]), np.array([0.3]), np.array([0.5]), 82.0, 0.98, True)
    assert a == pytest.approx(black76(80.0, 82.0, 0.3, 0.5, 0.98, True), rel=1e-10)


def test_averaging_lowers_option_value():
    t = fixing_times(6)
    avg = asian_levy(np.full(len(t), 80.0), np.full(len(t), 0.3), t, 80.0, 1.0, True)
    assert avg < black76(80.0, 80.0, 0.3, t[-1], 1.0, True)


def test_commodity_par_swap_and_zero_cost_collar_are_worth_zero(mkt):
    months = range(1, 13)
    k = solve_par_fixed(mkt, "TTF", months)
    assert sum(leg.pv(mkt) for leg in swap("TTF", months, 1e6, k, 1)) == pytest.approx(
        0, abs=1e-3
    )
    put_k = solve_zero_cost_put(mkt, "TTF", months, 14.0)
    assert put_k < 14.0
    assert sum(
        leg.pv(mkt) for leg in collar("TTF", months, 1e6, put_k, 14.0, 1)
    ) == pytest.approx(0, abs=1e-3)


def test_european_leg_sensitivities_match_analytic_greeks(mkt):
    leg = EuropeanLeg("BRENT", 6, 1.0, 80.0, True)
    _, sens = sensitivities(
        [Trade("T", "opt", "D", "B", "C", "bank", "CMDTY", (leg,))], mkt
    )
    px = sens[sens.factor == "CMDTY:BRENT"].iloc[0]
    vol = sens[sens.factor == "VOL:CMDTY:BRENT"].iloc[0]
    f, v = mkt["CMDTY:BRENT"].values[6], mkt["VOL:CMDTY:BRENT"].values[6]
    g = black76_greeks(f, 80.0, v, leg.expiry, mkt.df("USD", leg.expiry), True)
    assert px.d1 == pytest.approx(f * g["delta"], rel=1e-3)
    assert px.d2 == pytest.approx(f * f * g["gamma"], rel=1e-3)
    assert vol.d1 == pytest.approx(g["vega"], rel=1e-4)


# ---- Equity ---------------------------------------------------------------------


@pytest.mark.parametrize("name,expiry", [("EQ:SPX", 1.0), ("EQ:SX5E", 0.5)])
def test_equity_put_call_forward_parity(mkt, name, expiry):
    k = mkt.spot(name)
    c = EquityOptionLeg(name, expiry, k, True, 1).pv(mkt)
    p = EquityOptionLeg(name, expiry, k, False, 1).pv(mkt)
    assert c - p == pytest.approx(
        EquityForwardLeg(name, expiry, k, 1).pv(mkt), abs=1e-9
    )


@pytest.mark.parametrize("barrier", [150.0, 170.0])
def test_down_and_out_call_matches_monte_carlo(barrier):
    exact = down_and_out_call(180, 180, barrier, 0.32, 1.0, 0.04, 0.005)
    mc, se = barrier_mc(180, 180, barrier, 0.32, 1.0, 0.04, 0.005)
    assert abs(exact - mc) < 3.5 * se


def test_down_and_out_call_bounds(mkt):
    vanilla = EquityOptionLeg("EQ:NWT", 1.0, 180.0, True, 1).pv(mkt)
    knock = DownAndOutCallLeg("EQ:NWT", 1.0, 180.0, 144.0, 1).pv(mkt)
    assert 0 < knock < vanilla
    assert down_and_out_call(150.0, 180, 150.0, 0.3, 1.0, 0.04, 0.0) == 0.0
    # A barrier far away leaves the vanilla price.
    assert down_and_out_call(180, 180, 1.0, 0.32, 1.0, 0.04, 0.005) == pytest.approx(
        black76(180 * np.exp(0.035), 180, 0.32, 1.0, np.exp(-0.04), True), rel=1e-9
    )


# ---- FX --------------------------------------------------------------------------


def test_fx_put_call_forward_parity(mkt):
    c = FXOptionLeg("EUR", 1.0, 1.12, True, 1e6).pv(mkt)
    p = FXOptionLeg("EUR", 1.0, 1.12, False, 1e6).pv(mkt)
    assert c - p == pytest.approx(FXForwardLeg("EUR", 1.0, 1.12, 1e6).pv(mkt), abs=1e-6)


def test_fx_forward_at_covered_interest_parity_is_worth_zero(mkt):
    fwd = mkt.spot("FX:EUR") * mkt.df("EUR", 0.5) / mkt.df("USD", 0.5)
    assert FXForwardLeg("EUR", 0.5, fwd, 1e9).pv(mkt) == pytest.approx(0, abs=1e-4)


# ---- Rates ------------------------------------------------------------------------


def test_par_swap_is_worth_zero(mkt):
    k = par_rate(mkt, "USD", 0, 10)
    assert SwapLeg("USD", 0, 10, k, 1e9).pv(mkt) == pytest.approx(0, abs=1e-3)
    assert k == pytest.approx(forward_swap_rate(mkt, "USD", 0, 10), rel=1e-9)


def test_swaption_payer_receiver_parity(mkt):
    k, n = 0.04, 1e8
    pay = SwaptionLeg("USD", 1, 10, k, True, n).pv(mkt)
    rec = SwaptionLeg("USD", 1, 10, k, False, n).pv(mkt)
    fwd, a = forward_swap_rate(mkt, "USD", 1, 11), annuity(mkt, "USD", 1, 11)
    assert pay - rec == pytest.approx(n * a * (fwd - k), rel=1e-9)


def test_bachelier_atm_closed_form():
    assert bachelier(0.04, 0.04, 0.01, 1.0, 5.0, True) == pytest.approx(
        5.0 * 0.01 / np.sqrt(2 * np.pi)
    )


# ---- Credit ------------------------------------------------------------------------


def test_cds_at_par_spread_is_worth_zero_and_protection_gains_on_widening(mkt):
    s = par_spread(mkt, "CS:ACME", 5.0)
    leg = CDSLeg("CS:ACME", 5.0, s, 1e7)
    assert leg.pv(mkt) == pytest.approx(0, abs=1e-6)
    assert leg.pv(mkt.bump("CS:ACME", 0.001)) > 0
    # Credit triangle: flat spread s gives a par spread close to s.
    assert s == pytest.approx(mkt["CS:ACME"].values[3], rel=0.15)
