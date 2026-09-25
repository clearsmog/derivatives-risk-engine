"""Credit default swaps on single names and an index (no look-through)."""

from __future__ import annotations

from dataclasses import dataclass

from ..market import Market
from ..pricing import cds_legs
from .base import all_pillars


def _legs(mkt: Market, name: str, maturity: float) -> tuple[float, float]:
    f = mkt[name]
    return cds_legs(
        f.tenors,
        f.values,
        mkt.static["recovery"][name],
        maturity,
        lambda t: mkt.df("USD", t),
        mkt.time_shift,
    )


@dataclass(frozen=True)
class CDSLeg:
    name: str  # e.g. "CS:ACME"
    maturity: float
    coupon: float  # running spread paid by the protection buyer
    notional: float  # positive = bought protection

    is_option = False
    expiry = None
    underlying_maturity = None

    def pv(self, mkt: Market) -> float:
        annuity, protection = _legs(mkt, self.name, self.maturity)
        return self.notional * (protection - self.coupon * annuity)

    def pillars(self, mkt: Market) -> list[tuple[str, int]]:
        return [*all_pillars(mkt, self.name), *all_pillars(mkt, "IR:USD")]


def par_spread(mkt: Market, name: str, maturity: float) -> float:
    annuity, protection = _legs(mkt, name, maturity)
    return protection / annuity
