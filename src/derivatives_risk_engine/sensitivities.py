"""Bump-and-reprice sensitivities, one row per (leg, risk-factor pillar).

Each leg declares the pillars it depends on, so only those are bumped. For
every pillar the engine stores first and second derivatives in the units the
scenario engine moves that factor in, which makes the Taylor P&L one formula
for every asset class:

    kind     move m                        d1                 d2
    price    simple return (x' / x - 1)    x dV/dx            x^2 d2V/dx2      (cash delta, cash gamma)
    rate     absolute change               dV/dr              d2V/dr2
    spread   absolute change               dV/ds              d2V/ds2
    vol      absolute change               dV/dsigma (vega)   d2V/dsigma2 (volga)

    Taylor P&L = sum over pillars of d1 * m + 0.5 * d2 * m^2  (+ theta)

Cross terms (spot-vol, FX-equity, rate-spot) are not captured, which is what
the unexplained P&L in pnl_explain.py picks up.

FRTB sensitivities follow MAR21.19-25: one-sided 1% relative bumps for prices,
1bp for rates and spreads (divided by the bump), vega times implied vol.
"""

from __future__ import annotations

import pandas as pd

from .instruments import Trade
from .market import Market

BUMP = {"price": 0.01, "rate": 1e-4, "spread": 1e-4, "vol": 0.01}
NORMAL_VOL_BUMP = 1e-4  # VOL:IR:* are normal vols quoted in rate units


def _bump_size(mkt: Market, name: str) -> float:
    f = mkt[name]
    if f.kind == "vol" and f.unit == "normal":
        return NORMAL_VOL_BUMP
    return BUMP[f.kind]


def sensitivities(
    trades: list[Trade], mkt: Market
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (legs, sens).

    legs: one row per leg with PV, theta and static attributes.
    sens: one row per leg and pillar with d1, d2 and the FRTB sensitivity.
    """
    leg_rows, sens_rows = [], []
    rolled = mkt.roll(1)
    for t in trades:
        for i, leg in enumerate(t.legs):
            v0 = leg.pv(mkt)
            leg_rows.append(
                {
                    "trade_id": t.trade_id,
                    "leg": i,
                    "leg_type": type(leg).__name__,
                    "is_option": leg.is_option,
                    "pv": v0,
                    "theta": leg.pv(rolled) - v0,
                    "option_expiry": leg.expiry
                    if leg.expiry is not None
                    else float("nan"),
                    "underlying_maturity": leg.underlying_maturity
                    if leg.underlying_maturity is not None
                    else float("nan"),
                }
            )
            for name, p in leg.pillars(mkt):
                f = mkt[name]
                h = _bump_size(mkt, name)
                up = leg.pv(mkt.bump(name, h, p))
                dn = leg.pv(mkt.bump(name, -h, p))
                d1 = (up - dn) / (2 * h)
                d2 = (up - 2 * v0 + dn) / h**2
                if abs(d1) < 1e-9 and abs(d2) < 1e-9:
                    continue
                x = float(f.values[p])
                sens_rows.append(
                    {
                        "trade_id": t.trade_id,
                        "leg": i,
                        "factor": name,
                        "kind": f.kind,
                        "asset_class": f.asset_class,
                        "pillar": p,
                        "tenor": float(f.tenors[p]),
                        "level": x,
                        "d1": d1,
                        "d2": d2,
                        # MAR21: one-sided bump divided by its size; vega * vol
                        "frtb": (up - v0) / h if f.kind != "vol" else d1 * x,
                        "option_expiry": leg.expiry
                        if leg.expiry is not None
                        else float("nan"),
                        "underlying_maturity": leg.underlying_maturity
                        if leg.underlying_maturity is not None
                        else float("nan"),
                    }
                )
    return pd.DataFrame(leg_rows), pd.DataFrame(sens_rows)


def trade_frame(trades: list[Trade]) -> pd.DataFrame:
    """Static trade attributes, one row per trade, for joins and the cube."""
    return pd.DataFrame(
        [
            {
                "trade_id": t.trade_id,
                "product": t.product,
                "desk": t.desk,
                "book": t.book,
                "asset_class": t.asset_class,
                "counterparty": t.counterparty,
                "counterparty_type": t.counterparty_type,
                "has_optionality": t.has_optionality,
            }
            for t in trades
        ]
    )
