"""A synthetic LNG trading book.

The shape follows what an LNG importer's trading desk actually carries, but
every name, volume and price is invented:

* LNG_SUPPLY    long-term purchases indexed to Brent, resold at JKM / TTF
                -> a structural short-oil / long-gas spread position
* CLIENT        structures sold to downstream buyers (the book is the dealer):
                zero-cost collars, a capped swap, a fixed-price swap
* BANK_HEDGES   swaps and options with bank counterparties that offset part
                of the above, deliberately leaving residual risk to find
"""

from __future__ import annotations

from .instruments import (
    Trade,
    asian_strip,
    capped_swap,
    collar,
    oil_indexed_supply,
    solve_par_fixed,
    solve_zero_cost_put,
    swap,
)
from .instruments import EuropeanLeg
from .market import Market

CARGO_MMBTU = 3_400_000  # one standard cargo per month


def build_book(mkt: Market) -> list[Trade]:
    y1, y2, y12 = range(1, 13), range(13, 25), range(1, 25)
    trades: list[Trade] = []

    def add(tid, product, book, cpty, ctype, legs):
        trades.append(Trade(tid, product, book, cpty, ctype, tuple(legs)))

    # --- Oil-indexed supply contracts -------------------------------------
    add(
        "SUP-001",
        "Oil-indexed LNG supply",
        "LNG_SUPPLY",
        "Supplier A",
        "supplier",
        oil_indexed_supply(
            y12, CARGO_MMBTU, slope=0.125, constant=0.50, sell_index="JKM"
        ),
    )
    add(
        "SUP-002",
        "Oil-indexed LNG supply",
        "LNG_SUPPLY",
        "Supplier B",
        "supplier",
        oil_indexed_supply(
            y1, CARGO_MMBTU, slope=0.118, constant=0.30, sell_index="TTF"
        ),
    )

    # --- Client structures (book is the dealer, side = -1) ----------------
    ttf_call = 14.0
    ttf_put = solve_zero_cost_put(mkt, "TTF", y1, ttf_call)
    add(
        "CLI-001",
        "Zero-cost collar",
        "CLIENT",
        "Gas Power Plant",
        "client",
        collar("TTF", y1, 500_000, ttf_put, ttf_call, side=-1),
    )

    brent_cap = 95.0
    brent_fixed = solve_par_fixed(mkt, "BRENT", y1, cap=brent_cap)
    add(
        "CLI-002",
        "Capped swap",
        "CLIENT",
        "Petrochemical Plant",
        "client",
        capped_swap("BRENT", y1, 100_000, brent_fixed, brent_cap, side=-1),
    )

    jkm_fixed = solve_par_fixed(mkt, "JKM", y1)
    add(
        "CLI-003",
        "Fixed-price swap",
        "CLIENT",
        "City Gas Distributor",
        "client",
        swap("JKM", y1, 500_000, jkm_fixed, side=-1),
    )

    # --- Bank hedges ------------------------------------------------------
    brent_y1 = solve_par_fixed(mkt, "BRENT", y1)
    brent_y2 = solve_par_fixed(mkt, "BRENT", y2)
    add(
        "HDG-001",
        "Brent swap",
        "BANK_HEDGES",
        "Bank A",
        "bank",
        swap("BRENT", y1, 400_000, brent_y1, side=+1),
    )
    add(
        "HDG-002",
        "Brent swap",
        "BANK_HEDGES",
        "Bank B",
        "bank",
        swap("BRENT", y2, 300_000, brent_y2, side=+1),
    )
    add(
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
    )
    add(
        "HDG-004",
        "TTF swap",
        "BANK_HEDGES",
        "Bank D",
        "bank",
        swap("TTF", y1, 2_000_000, solve_par_fixed(mkt, "TTF", y1), side=-1),
    )
    add(
        "HDG-005",
        "TTF Asian call strip",
        "BANK_HEDGES",
        "Bank A",
        "bank",
        asian_strip("TTF", y1, 300_000, ttf_call, is_call=True, side=+1),
    )
    add(
        "HDG-006",
        "Brent futures options",
        "BANK_HEDGES",
        "Bank E",
        "bank",
        [EuropeanLeg("BRENT", m, 200_000, 90.0, True) for m in range(3, 9)],
    )
    return trades
