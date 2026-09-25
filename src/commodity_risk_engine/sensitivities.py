"""Leg-level sensitivities by bump and reprice.

Each leg depends on exactly one forward pillar and one vol pillar (its
commodity and contract month), so bumping only that pillar gives the full
sensitivity set with no cross-gamma to worry about. Trade and book numbers are
sums over legs.

Conventions
    delta_ln   dV / d ln F                    (value change per 100% log move)
    gamma_ln   d2V / d(ln F)^2                (used by the Taylor P&L)
    gamma_cash F^2 d2V / dF^2 = gamma_ln - delta_ln, zero for linear legs
    delta_frtb (V(1.01 F) - V(F)) / 0.01       MAR21.23 commodity delta
    vega       dV / d sigma                   (per 1.00 of vol)
    vega_frtb  vega * sigma                   MAR21.25
    theta      V(t + 1 business day) - V(t)
"""

from __future__ import annotations

import pandas as pd

from .instruments import Trade
from .market import Market, payment_time

H_LN = 0.01
H_VOL = 0.01


def leg_sensitivities(trades: list[Trade], mkt: Market) -> pd.DataFrame:
    rows = []
    rolled = mkt.roll(1)
    for trade in trades:
        for i, leg in enumerate(trade.legs):
            c, m = leg.commodity, leg.month
            v0 = leg.pv(mkt)
            up = leg.pv(mkt.bump_fwd(c, H_LN, m, log=True))
            dn = leg.pv(mkt.bump_fwd(c, -H_LN, m, log=True))
            row = {
                "trade_id": trade.trade_id,
                "leg": i,
                "commodity": c,
                "month": m,
                "maturity": payment_time(m),
                "is_option": leg.is_option,
                "pv": v0,
                "delta_ln": (up - dn) / (2 * H_LN),
                "gamma_ln": (up - 2 * v0 + dn) / H_LN**2,
                "delta_frtb": (leg.pv(mkt.bump_fwd(c, 0.01, m)) - v0) / 0.01,
                "theta": leg.pv(rolled) - v0,
                "vega": 0.0,
                "vega_frtb": 0.0,
                "option_expiry": float("nan"),
            }
            row["gamma_cash"] = row["gamma_ln"] - row["delta_ln"]
            if leg.is_option:
                vu = leg.pv(mkt.bump_vol(c, H_VOL, m))
                vd = leg.pv(mkt.bump_vol(c, -H_VOL, m))
                row["vega"] = (vu - vd) / (2 * H_VOL)
                row["vega_frtb"] = row["vega"] * mkt.curve(c).vol[m]
                row["option_expiry"] = leg.expiry
            rows.append(row)
    return pd.DataFrame(rows)


def trade_frame(trades: list[Trade]) -> pd.DataFrame:
    """Static trade attributes, one row per trade, for joins and the cube."""
    return pd.DataFrame(
        [
            {
                "trade_id": t.trade_id,
                "product": t.product,
                "book": t.book,
                "counterparty": t.counterparty,
                "counterparty_type": t.counterparty_type,
                "has_optionality": t.has_optionality,
            }
            for t in trades
        ]
    )
