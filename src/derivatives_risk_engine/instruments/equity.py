"""Equity legs: index and single-stock options, forwards, and a barrier option.

Options are Black-Scholes with a continuous dividend yield, written on the
forward F = S e^{-qT} / DF. Non-USD underlyings (SX5E) are converted to USD at
the FX spot, so they carry FX delta as well as equity delta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..market import Market
from ..pricing import black76, down_and_out_call
from .base import all_pillars


def _ccy(mkt: Market, name: str) -> str:
    return mkt.static["currency"][name]


def _common_pillars(mkt: Market, name: str, with_vol: bool) -> list[tuple[str, int]]:
    ccy = _ccy(mkt, name)
    out = [(name, 0), *all_pillars(mkt, f"IR:{ccy}")]
    if with_vol:
        out += all_pillars(mkt, f"VOL:{name}")
    if ccy != "USD":
        out.append((f"FX:{ccy}", 0))
    return out


def _forward(mkt: Market, name: str, expiry: float) -> tuple[float, float, float]:
    """(forward, discount factor, time to expiry) in the underlying's currency."""
    t = expiry - mkt.time_shift
    df = mkt.df(_ccy(mkt, name), expiry)
    q = mkt.static["dividend_yield"][name]
    return mkt.spot(name) * np.exp(-q * t) / df, df, t


@dataclass(frozen=True)
class EquityOptionLeg:
    underlying: str  # e.g. "EQ:SPX"
    expiry: float
    strike: float
    is_call: bool
    quantity: float  # units of the index / shares

    is_option = True
    underlying_maturity = None

    def pv(self, mkt: Market) -> float:
        fwd, df, t = _forward(mkt, self.underlying, self.expiry)
        vol = mkt.vol_at(f"VOL:{self.underlying}", t)
        local = black76(fwd, self.strike, vol, t, df, self.is_call)
        return self.quantity * local * mkt.fx(_ccy(mkt, self.underlying))

    def pillars(self, mkt: Market) -> list[tuple[str, int]]:
        return _common_pillars(mkt, self.underlying, with_vol=True)


@dataclass(frozen=True)
class EquityForwardLeg:
    underlying: str
    maturity: float
    strike: float
    quantity: float

    is_option = False
    expiry = None
    underlying_maturity = None

    def pv(self, mkt: Market) -> float:
        fwd, df, _ = _forward(mkt, self.underlying, self.maturity)
        return (
            self.quantity
            * df
            * (fwd - self.strike)
            * mkt.fx(_ccy(mkt, self.underlying))
        )

    def pillars(self, mkt: Market) -> list[tuple[str, int]]:
        return _common_pillars(mkt, self.underlying, with_vol=False)


@dataclass(frozen=True)
class DownAndOutCallLeg:
    """Continuously monitored down-and-out call (strike >= barrier), USD underlyings."""

    underlying: str
    expiry: float
    strike: float
    barrier: float
    quantity: float

    is_option = True
    underlying_maturity = None

    def pv(self, mkt: Market) -> float:
        _, df, t = _forward(mkt, self.underlying, self.expiry)
        if t <= 0.0:
            s = mkt.spot(self.underlying)
            return self.quantity * max(s - self.strike, 0.0) * mkt.fx(_ccy(mkt, self.underlying))
        r = -np.log(df) / t
        q = mkt.static["dividend_yield"][self.underlying]
        vol = mkt.vol_at(f"VOL:{self.underlying}", t)
        price = down_and_out_call(
            mkt.spot(self.underlying), self.strike, self.barrier, vol, t, r, q
        )
        return self.quantity * price * mkt.fx(_ccy(mkt, self.underlying))

    def pillars(self, mkt: Market) -> list[tuple[str, int]]:
        return _common_pillars(mkt, self.underlying, with_vol=True)
