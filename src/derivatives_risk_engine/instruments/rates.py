"""Interest-rate legs: vanilla swaps and European swaptions.

Single-curve OIS pricing: the zero curve both discounts and projects. A
floating leg compounds the overnight rate, so once it has started it is worth
the compounding already realised minus DF(end); before it starts it is worth
DF(start) - DF(end). Fixed legs pay annually, on plain year fractions.
Swaptions use the Bachelier (normal) model on the forward swap rate, which
MAR21.28 allows for GIRR vega.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..market import Market
from ..pricing import bachelier
from .base import all_pillars


def annuity(mkt: Market, ccy: str, start: float, end: float) -> float:
    """Annual fixed-leg annuity over coupons not yet paid."""
    pay = np.arange(start + 1.0, end + 1e-9, 1.0)
    return float(sum(mkt.df(ccy, t) for t in pay if t > mkt.time_shift))


def forward_swap_rate(mkt: Market, ccy: str, start: float, end: float) -> float:
    return (mkt.df(ccy, start) - mkt.df(ccy, end)) / annuity(mkt, ccy, start, end)


def _pillars(mkt: Market, ccy: str) -> list[tuple[str, int]]:
    out = all_pillars(mkt, f"IR:{ccy}")
    if ccy != "USD":
        out.append((f"FX:{ccy}", 0))
    return out


@dataclass(frozen=True)
class SwapLeg:
    ccy: str
    start: float
    end: float
    fixed_rate: float
    notional: float  # positive = pay fixed, receive floating
    overnight: float | None = None  # overnight rate realised since the start (flat)

    is_option = False
    expiry = None
    underlying_maturity = None

    @classmethod
    def spot_starting(cls, mkt: Market, ccy: str, end: float, fixed_rate: float, notional: float):
        """A swap starting today; accrual runs at today's shortest zero rate."""
        return cls(ccy, 0.0, end, fixed_rate, notional, float(mkt[f"IR:{ccy}"].values[0]))

    def pv(self, mkt: Market) -> float:
        if self.overnight is not None and self.start <= mkt.time_shift:
            realised = np.exp(self.overnight * (mkt.time_shift - self.start))
            floating = realised - mkt.df(self.ccy, self.end)
        else:
            floating = mkt.df(self.ccy, self.start) - mkt.df(self.ccy, self.end)
        fixed = self.fixed_rate * annuity(mkt, self.ccy, self.start, self.end)
        return self.notional * (floating - fixed) * mkt.fx(self.ccy)

    def pillars(self, mkt: Market) -> list[tuple[str, int]]:
        return _pillars(mkt, self.ccy)


@dataclass(frozen=True)
class SwaptionLeg:
    ccy: str
    expiry: float
    tenor: float
    strike: float
    is_payer: bool
    notional: float

    is_option = True

    @property
    def underlying_maturity(self) -> float:
        return self.tenor

    def pv(self, mkt: Market) -> float:
        end = self.expiry + self.tenor
        a = annuity(mkt, self.ccy, self.expiry, end)
        fwd = forward_swap_rate(mkt, self.ccy, self.expiry, end)
        t = self.expiry - mkt.time_shift
        vol = mkt.vol_at(f"VOL:IR:{self.ccy}", t)
        return (
            self.notional
            * bachelier(fwd, self.strike, vol, t, a, self.is_payer)
            * mkt.fx(self.ccy)
        )

    def pillars(self, mkt: Market) -> list[tuple[str, int]]:
        return [*_pillars(mkt, self.ccy), *all_pillars(mkt, f"VOL:IR:{self.ccy}")]


def par_rate(mkt: Market, ccy: str, start: float, end: float) -> float:
    return brentq(lambda k: SwapLeg(ccy, start, end, k, 1.0).pv(mkt), -0.05, 0.25)
