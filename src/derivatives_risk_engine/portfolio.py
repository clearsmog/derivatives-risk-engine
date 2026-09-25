"""A synthetic multi-asset book, organised as a fund's trading desks.

Every name, size and price is invented.

COMMODITIES  an LNG importer-style book: Brent-indexed supply contracts resold
             at JKM / TTF, structures sold to downstream buyers (zero-cost
             collar, capped swap, fixed swap) and partial bank hedges
EQUITY       SPX 3m puts paid for by selling 1y calls (zero premium, net short
             vega), a Euro Stoxx put (EUR, so it carries FX), a single-stock
             down-and-out call, and forwards sized to hedge SPX and NWT delta
RATES        a PV01-neutral USD 2s10s steepener, a payer swaption, a EUR receiver
FX           a long EUR forward, a short EUR call, a long EUR put
CREDIT       ACME protection bought against CDX IG protection sold, CS01-neutral
             (single name vs index), and HY protection bought on ZETA
"""

from __future__ import annotations

from .instruments import (
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
    asian_strip,
    capped_swap,
    collar,
    forward_swap_rate,
    oil_indexed_supply,
    par_rate,
    solve_par_fixed,
    solve_zero_cost_put,
    swap,
)
from .market import Market

CARGO_MMBTU = 3_400_000  # one standard cargo per month


def _commodities(mkt: Market) -> list[Trade]:
    y1, y2, y12 = range(1, 13), range(13, 25), range(1, 25)
    ttf_call, brent_cap = 14.0, 95.0
    rows = [
        (
            "SUP-001",
            "Oil-indexed LNG supply",
            "LNG_SUPPLY",
            "Supplier A",
            "supplier",
            oil_indexed_supply(
                y12, CARGO_MMBTU, slope=0.125, constant=0.50, sell_index="JKM"
            ),
        ),
        (
            "SUP-002",
            "Oil-indexed LNG supply",
            "LNG_SUPPLY",
            "Supplier B",
            "supplier",
            oil_indexed_supply(
                y1, CARGO_MMBTU, slope=0.118, constant=0.30, sell_index="TTF"
            ),
        ),
        (
            "CLI-001",
            "Zero-cost collar",
            "CLIENT_STRUCTURES",
            "Gas Power Plant",
            "client",
            collar(
                "TTF",
                y1,
                500_000,
                solve_zero_cost_put(mkt, "TTF", y1, ttf_call),
                ttf_call,
                side=-1,
            ),
        ),
        (
            "CLI-002",
            "Capped swap",
            "CLIENT_STRUCTURES",
            "Petrochemical Plant",
            "client",
            capped_swap(
                "BRENT",
                y1,
                100_000,
                solve_par_fixed(mkt, "BRENT", y1, cap=brent_cap),
                brent_cap,
                side=-1,
            ),
        ),
        (
            "CLI-003",
            "Fixed-price swap",
            "CLIENT_STRUCTURES",
            "City Gas Distributor",
            "client",
            swap("JKM", y1, 500_000, solve_par_fixed(mkt, "JKM", y1), side=-1),
        ),
        (
            "HDG-001",
            "Brent swap",
            "BANK_HEDGES",
            "Bank A",
            "bank",
            swap("BRENT", y1, 400_000, solve_par_fixed(mkt, "BRENT", y1), side=+1),
        ),
        (
            "HDG-002",
            "Brent swap",
            "BANK_HEDGES",
            "Bank B",
            "bank",
            swap("BRENT", y2, 300_000, solve_par_fixed(mkt, "BRENT", y2), side=+1),
        ),
        (
            "HDG-003",
            "JKM swap",
            "BANK_HEDGES",
            "Bank C",
            "bank",
            swap(
                "JKM",
                range(1, 7),
                2_000_000,
                solve_par_fixed(mkt, "JKM", range(1, 7)),
                side=-1,
            ),
        ),
        (
            "HDG-004",
            "TTF swap",
            "BANK_HEDGES",
            "Bank D",
            "bank",
            swap("TTF", y1, 2_000_000, solve_par_fixed(mkt, "TTF", y1), side=-1),
        ),
        (
            "HDG-005",
            "TTF Asian call strip",
            "BANK_HEDGES",
            "Bank A",
            "bank",
            asian_strip("TTF", y1, 300_000, ttf_call, is_call=True, side=+1),
        ),
        (
            "HDG-006",
            "Brent futures options",
            "BANK_HEDGES",
            "Bank E",
            "bank",
            [EuropeanLeg("BRENT", m, 200_000, 90.0, True) for m in range(3, 9)],
        ),
    ]
    return [
        Trade(tid, prod, "COMMODITIES", book, cp, ct, "CMDTY", tuple(legs))
        for tid, prod, book, cp, ct, legs in rows
    ]


def _parallel_delta(legs: list, mkt: Market, factor: str, bump: float) -> float:
    """P&L of `legs` for a parallel bump of one factor (whole curve or spot)."""
    shocked = mkt.bump(factor, bump)
    return sum(leg.pv(shocked) - leg.pv(mkt) for leg in legs)


def _equity(mkt: Market) -> list[Trade]:
    spx, sx5e, nwt = mkt.spot("EQ:SPX"), mkt.spot("EQ:SX5E"), mkt.spot("EQ:NWT")
    put = EquityOptionLeg("EQ:SPX", 0.25, 0.95 * spx, False, 80_000)
    # Sell enough 1y 105% calls to pay for the puts (a zero-premium risk reversal).
    call_unit = EquityOptionLeg("EQ:SPX", 1.0, 1.05 * spx, True, 1.0)
    call = EquityOptionLeg("EQ:SPX", 1.0, 1.05 * spx, True, -put.pv(mkt) / call_unit.pv(mkt))
    fwd_unit = EquityForwardLeg("EQ:SPX", 0.5, spx, 1.0)
    spx_hedge = EquityForwardLeg(
        "EQ:SPX", 0.5, spx,
        -_parallel_delta([put, call], mkt, "EQ:SPX", 0.01) / _parallel_delta([fwd_unit], mkt, "EQ:SPX", 0.01),
    )
    doc = DownAndOutCallLeg("EQ:NWT", 1.0, nwt, 0.80 * nwt, 2_000_000)
    nwt_unit = EquityForwardLeg("EQ:NWT", 1.0, nwt, 1.0)
    nwt_hedge = EquityForwardLeg(
        "EQ:NWT", 1.0, nwt,
        -_parallel_delta([doc], mkt, "EQ:NWT", 0.01) / _parallel_delta([nwt_unit], mkt, "EQ:NWT", 0.01),
    )
    rows = [
        ("EQ-001", "SPX 3m put 95%", "Bank A", [put]),
        ("EQ-002", "SPX 1y call 105% (sold, finances the put)", "Bank B", [call]),
        ("EQ-003", "SPX 6m forward (delta hedge)", "Clearing house", [spx_hedge]),
        ("EQ-004", "SX5E 6m put 90% (EUR)", "Bank C", [EquityOptionLeg("EQ:SX5E", 0.5, 0.90 * sx5e, False, 60_000)]),
        ("EQ-005", "NWT 1y down-and-out call", "Bank D", [doc]),
        ("EQ-006", "NWT 1y forward (delta hedge)", "Bank D", [nwt_hedge]),
    ]
    return [
        Trade(tid, prod, "EQUITY", "EQ_DERIVATIVES", cp, "bank" if cp.startswith("Bank") else "exchange", "EQ", tuple(legs))
        for tid, prod, cp, legs in rows
    ]


def _rates(mkt: Market) -> list[Trade]:
    k10, k2, k5e = par_rate(mkt, "USD", 0, 10), par_rate(mkt, "USD", 0, 2), par_rate(mkt, "EUR", 0, 5)
    ten = SwapLeg.spot_starting(mkt, "USD", 10, k10, 1.5e9)
    two_unit = SwapLeg.spot_starting(mkt, "USD", 2, k2, 1.0)
    # Receive 2y in PV01-neutral size: a pure 2s10s steepener.
    two = SwapLeg.spot_starting(
        mkt, "USD", 2, k2,
        -_parallel_delta([ten], mkt, "IR:USD", 1e-4) / _parallel_delta([two_unit], mkt, "IR:USD", 1e-4),
    )
    atm = forward_swap_rate(mkt, "USD", 1, 11)
    rows = [
        ("IR-001", "USD 10y pay fixed (steepener, long leg)", "Bank A", [ten]),
        ("IR-002", "USD 2y receive fixed (steepener, PV01-neutral)", "Bank B", [two]),
        ("IR-003", "USD 1y x 10y payer swaption", "Bank C", [SwaptionLeg("USD", 1.0, 10.0, atm, True, 1e9)]),
        ("IR-004", "EUR 5y receive fixed", "Bank E", [SwapLeg.spot_starting(mkt, "EUR", 5, k5e, -1.2e9)]),
    ]
    return [Trade(tid, prod, "RATES", "RATES", cp, "bank", "IR", tuple(legs)) for tid, prod, cp, legs in rows]


def _fx(mkt: Market) -> list[Trade]:
    s = mkt.spot("FX:EUR")
    fwd6m = s * mkt.df("EUR", 0.5) / mkt.df("USD", 0.5)
    fwd3m = s * mkt.df("EUR", 0.25) / mkt.df("USD", 0.25)
    rows = [
        (
            "FX-001",
            "EURUSD 6m forward (long EUR)",
            "Bank A",
            [FXForwardLeg("EUR", 0.5, fwd6m, 1.2e9)],
        ),
        (
            "FX-002",
            "EUR call 3m ATM (sold)",
            "Bank B",
            [FXOptionLeg("EUR", 0.25, fwd3m, True, -800e6)],
        ),
        (
            "FX-003",
            "EUR put 1y 1.05 (bought)",
            "Bank C",
            [FXOptionLeg("EUR", 1.0, 1.05, False, 1e9)],
        ),
    ]
    return [
        Trade(tid, prod, "FX", "FX", cp, "bank", "FX", tuple(legs))
        for tid, prod, cp, legs in rows
    ]


def _credit(mkt: Market) -> list[Trade]:
    cdx = CDSLeg("CS:CDXIG", 5.0, 0.01, -2.5e9)
    acme_unit = CDSLeg("CS:ACME", 5.0, 0.01, 1.0)
    # Buy single-name protection in CS01-neutral size against the index sold.
    acme = CDSLeg(
        "CS:ACME", 5.0, 0.01,
        -_parallel_delta([cdx], mkt, "CS:CDXIG", 1e-4) / _parallel_delta([acme_unit], mkt, "CS:ACME", 1e-4),
    )
    rows = [
        ("CS-001", "ACME 5y CDS (protection bought, CS01-neutral)", "Bank A", [acme]),
        ("CS-002", "CDX IG 5y (protection sold)", "Bank B", [cdx]),
        ("CS-003", "ZETA 3y CDS (protection bought)", "Bank C", [CDSLeg("CS:ZETA", 3.0, 0.05, 400e6)]),
    ]
    return [Trade(tid, prod, "CREDIT", "CREDIT", cp, "bank", "CS", tuple(legs)) for tid, prod, cp, legs in rows]


def build_book(mkt: Market) -> list[Trade]:
    return _commodities(mkt) + _equity(mkt) + _rates(mkt) + _fx(mkt) + _credit(mkt)
