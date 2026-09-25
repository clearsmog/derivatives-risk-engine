"""FX legs against USD: outright forwards and Garman-Kohlhagen options.

FX:EUR is EURUSD (USD per EUR). Notionals are in the foreign currency.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..market import Market
from ..pricing import black76
from .base import all_pillars


def _pillars(mkt: Market, ccy: str, with_vol: bool) -> list[tuple[str, int]]:
    out = [
        (f"FX:{ccy}", 0),
        *all_pillars(mkt, f"IR:{ccy}"),
        *all_pillars(mkt, "IR:USD"),
    ]
    if with_vol:
        out += all_pillars(mkt, f"VOL:FX:{ccy}")
    return out


@dataclass(frozen=True)
class FXForwardLeg:
    ccy: str
    maturity: float
    strike: float  # USD per unit of ccy
    notional: float  # units of ccy bought (negative = sold)

    is_option = False
    expiry = None
    underlying_maturity = None

    def pv(self, mkt: Market) -> float:
        t = self.maturity
        return self.notional * (
            mkt.fx(self.ccy) * mkt.df(self.ccy, t) - self.strike * mkt.df("USD", t)
        )

    def pillars(self, mkt: Market) -> list[tuple[str, int]]:
        return _pillars(mkt, self.ccy, with_vol=False)


@dataclass(frozen=True)
class FXOptionLeg:
    ccy: str
    expiry: float
    strike: float
    is_call: bool  # call on the foreign currency
    notional: float

    is_option = True
    underlying_maturity = None

    def pv(self, mkt: Market) -> float:
        t = self.expiry - mkt.time_shift
        df_usd = mkt.df("USD", self.expiry)
        fwd = mkt.fx(self.ccy) * mkt.df(self.ccy, self.expiry) / df_usd
        vol = mkt.vol_at(f"VOL:FX:{self.ccy}", t)
        return self.notional * black76(fwd, self.strike, vol, t, df_usd, self.is_call)

    def pillars(self, mkt: Market) -> list[tuple[str, int]]:
        return _pillars(mkt, self.ccy, with_vol=True)
