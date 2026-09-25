"""Trades are lists of legs; every leg exposes the same small interface.

    pv(mkt)                present value in USD
    pillars(mkt)           the (factor, pillar) pairs its value depends on, so
                           the sensitivity engine bumps only what matters
    is_option              whether it carries vega and curvature
    expiry                 option expiry in years, for vega tenor mapping
    underlying_maturity    residual maturity of the underlying at expiry (GIRR vega)

Quantities are signed from the book's point of view: positive = long.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..market import Market


class Leg(Protocol):
    is_option: bool

    def pv(self, mkt: Market) -> float: ...

    def pillars(self, mkt: Market) -> list[tuple[str, int]]: ...


def all_pillars(mkt: Market, name: str) -> list[tuple[str, int]]:
    return [(name, i) for i in range(mkt[name].n)]


@dataclass(frozen=True)
class Trade:
    trade_id: str
    product: str
    desk: str
    book: str
    counterparty: str
    counterparty_type: str  # "client", "bank", "supplier", "exchange"
    asset_class: str  # "CMDTY", "EQ", "FX", "IR", "CS"
    legs: tuple = field(default_factory=tuple)

    def pv(self, mkt: Market) -> float:
        return sum(leg.pv(mkt) for leg in self.legs)

    @property
    def has_optionality(self) -> bool:
        return any(leg.is_option for leg in self.legs)
