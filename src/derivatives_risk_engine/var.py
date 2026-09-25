"""Historical-simulation VaR / ES by full revaluation, and the Taylor proxies
that approximate it.

Full revaluation reprices every leg under every scenario. The Taylor proxy
rebuilds P&L from the pillar sensitivities (see sensitivities.py):

    dV ~ sum over pillars of d1 * m + 0.5 * d2 * m^2

with m the simple return for price factors and the absolute change otherwise.
Comparing the two shows where a sensitivity-based view stops being trustworthy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .instruments import Trade
from .market import Market
from .scenarios import ScenarioSet


def leg_pnl(trades: list[Trade], mkt: Market, scen: ScenarioSet) -> np.ndarray:
    """Full-revaluation P&L vectors, shape (n_legs, n_scenarios), legs in book order."""
    markets = [mkt.apply_moves(scen.row(i)) for i in range(scen.n)]
    rows = []
    for t in trades:
        for leg in t.legs:
            v0 = leg.pv(mkt)
            rows.append([leg.pv(m) - v0 for m in markets])
    return np.array(rows)


def scenario_pnl(trades: list[Trade], mkt: Market, scen: ScenarioSet) -> pd.DataFrame:
    """Full-revaluation P&L, one row per trade, one column per scenario."""
    ids = [t.trade_id for t in trades for _ in t.legs]
    return pd.DataFrame(leg_pnl(trades, mkt, scen), columns=scen.labels).groupby(ids, sort=False).sum()


def moves_for(sens: pd.DataFrame, scen: ScenarioSet) -> np.ndarray:
    """Scenario move per sensitivity row, shape (n_rows, n_scenarios)."""
    m = np.stack([scen.moves[f][:, p] for f, p in zip(sens.factor, sens.pillar)])
    price = (sens.kind == "price").to_numpy()
    m[price] = np.expm1(m[price])
    return m


def taylor_pnl(sens: pd.DataFrame, scen: ScenarioSet, order: str = "delta_gamma_vega") -> pd.DataFrame:
    """Sensitivity-based P&L per trade.

    order: 'delta' (first order, no vol), 'delta_gamma' (adds second order in
    the underlying factors) or 'delta_gamma_vega' (adds vega and volga).
    """
    m = moves_for(sens, scen)
    is_vol = (sens.kind == "vol").to_numpy()[:, None]
    d1 = sens.d1.to_numpy()[:, None]
    d2 = sens.d2.to_numpy()[:, None]
    pnl = d1 * m
    if order != "delta":
        pnl = pnl + 0.5 * d2 * m**2
    if order != "delta_gamma_vega":
        pnl = np.where(is_vol, 0.0, pnl)
    return pd.DataFrame(pnl, columns=scen.labels).groupby(sens.trade_id.to_numpy(), sort=False).sum()


def _var_index(n: int, level: float) -> int:
    """Rank of the VaR scenario in ascending P&L order (numpy 'lower' quantile)."""
    return int(np.floor((n - 1) * (1.0 - level)))


def var(pnl: np.ndarray, level: float = 0.99) -> float:
    """VaR as a positive loss number, empirical quantile."""
    return float(-np.sort(pnl)[_var_index(len(pnl), level)])


def expected_shortfall(pnl: np.ndarray, level: float = 0.975) -> float:
    """Average loss beyond the (1 - level) tail, as a positive number."""
    k = max(int(np.floor(len(pnl) * (1.0 - level))), 1)
    return float(-np.sort(pnl)[:k].mean())
