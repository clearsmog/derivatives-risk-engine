"""Trades as lists of legs.

Every structure in the book decomposes into three leg types, so pricing,
Greeks, VaR and FRTB all work on legs and never need to know what a
"capped swap" is:

    LinearLeg      quantity * (coeff * monthly average forward) + fixed cash
    AsianLeg       option on a contract month's arithmetic average
    EuropeanLeg    option on a single contract month's forward (Black-76)

Quantities are signed from the book's point of view: positive = long.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .market import Market, fixing_times, payment_time, window
from .pricing import asian_levy, black76


@dataclass(frozen=True)
class LinearLeg:
    commodity: str
    month: int
    quantity: float
    coeff: float = 1.0  # e.g. an oil-indexation slope
    fixed: float = 0.0  # price paid per unit, enters with a minus sign

    is_option = False

    def pv(self, mkt: Market) -> float:
        f = mkt.curve(self.commodity).fwd[self.month]
        return (
            self.quantity
            * mkt.df(payment_time(self.month))
            * (self.coeff * f - self.fixed)
        )


@dataclass(frozen=True)
class AsianLeg:
    commodity: str
    month: int
    quantity: float
    strike: float
    is_call: bool

    is_option = True

    @property
    def expiry(self) -> float:
        return window(self.month)[1]

    def pv(self, mkt: Market) -> float:
        c = mkt.curve(self.commodity)
        t = fixing_times(self.month) - mkt.time_shift
        n = len(t)
        fwds = np.full(n, c.fwd[self.month])
        vols = np.full(n, c.vol[self.month])
        df = mkt.df(payment_time(self.month))
        return self.quantity * asian_levy(fwds, vols, t, self.strike, df, self.is_call)


@dataclass(frozen=True)
class EuropeanLeg:
    commodity: str
    month: int
    quantity: float
    strike: float
    is_call: bool

    is_option = True

    @property
    def expiry(self) -> float:
        # Listed monthly options expire a few business days before delivery.
        return window(self.month)[0] - 3 / 252.0

    def pv(self, mkt: Market) -> float:
        c = mkt.curve(self.commodity)
        t = self.expiry - mkt.time_shift
        return self.quantity * black76(
            c.fwd[self.month],
            self.strike,
            c.vol[self.month],
            t,
            mkt.df(self.expiry),
            self.is_call,
        )


Leg = LinearLeg | AsianLeg | EuropeanLeg


@dataclass(frozen=True)
class Trade:
    trade_id: str
    product: str
    book: str
    counterparty: str
    counterparty_type: str  # "client", "bank", "supplier"
    legs: tuple[Leg, ...] = field(default_factory=tuple)

    def pv(self, mkt: Market) -> float:
        return sum(leg.pv(mkt) for leg in self.legs)

    @property
    def commodities(self) -> set[str]:
        return {leg.commodity for leg in self.legs}

    @property
    def has_optionality(self) -> bool:
        return any(leg.is_option for leg in self.legs)



# ---- Structure factories. `side=+1` means the book is long the structure. ----


def swap(
    commodity: str, months: range, qty: float, fixed: float, side: int
) -> list[Leg]:
    """Fixed-for-floating monthly average price swap. Long = receive floating."""
    return [LinearLeg(commodity, m, side * qty, 1.0, fixed) for m in months]


def asian_strip(
    commodity: str, months: range, qty: float, strike: float, is_call: bool, side: int
) -> list[Leg]:
    return [AsianLeg(commodity, m, side * qty, strike, is_call) for m in months]


def collar(
    commodity: str, months: range, qty: float, put_k: float, call_k: float, side: int
) -> list[Leg]:
    """Consumer collar: long call (cap), short put (floor). side=-1 is the dealer's view."""
    return asian_strip(commodity, months, qty, call_k, True, side) + asian_strip(
        commodity, months, qty, put_k, False, -side
    )


def capped_swap(
    commodity: str, months: range, qty: float, fixed: float, cap: float, side: int
) -> list[Leg]:
    """Consumer pays fixed, receives floating, but the upside is capped at `cap`:
    payoff min(A, cap) - fixed = long swap + short Asian call at cap."""
    return swap(commodity, months, qty, fixed, side) + asian_strip(
        commodity, months, qty, cap, True, -side
    )


def oil_indexed_supply(
    months: range,
    qty: float,
    slope: float,
    constant: float,
    sell_index: str,
    side: int = 1,
) -> list[Leg]:
    """Long-term LNG purchase priced at slope * Brent + constant, resold at a gas
    hub index. The book is long the hub and short the oil formula: a spread
    position between two commodity buckets."""
    legs: list[Leg] = []
    for m in months:
        legs.append(LinearLeg(sell_index, m, side * qty, 1.0, 0.0))
        legs.append(LinearLeg("BRENT", m, -side * qty, slope, -constant))
    return legs


def solve_zero_cost_put(
    mkt: Market, commodity: str, months: range, call_k: float
) -> float:
    """Put strike that makes a consumer collar cost nothing upfront."""

    def cost(put_k: float) -> float:
        return sum(
            leg.pv(mkt) for leg in collar(commodity, months, 1.0, put_k, call_k, 1)
        )

    fwd = float(np.mean(mkt.curve(commodity).fwd[list(months)]))
    return brentq(cost, 0.05 * fwd, call_k)


def solve_par_fixed(
    mkt: Market, commodity: str, months: range, cap: float | None = None
) -> float:
    """Fixed price that makes a (capped) swap worth zero at inception."""

    def value(k: float) -> float:
        legs = (
            capped_swap(commodity, months, 1.0, k, cap, 1)
            if cap is not None
            else swap(commodity, months, 1.0, k, 1)
        )
        return sum(leg.pv(mkt) for leg in legs)

    fwd = float(np.mean(mkt.curve(commodity).fwd[list(months)]))
    return brentq(value, 0.01 * fwd, 3.0 * fwd)
