"""Historical-simulation VaR / ES by full revaluation, and the Taylor proxies
that approximate it.

Full revaluation reprices every leg under every scenario. The Taylor proxy
rebuilds P&L from the leg sensitivities:

    dV ~ cash_delta * R + 0.5 * cash_gamma * R^2 + vega * d_sigma   (R simple return)

Comparing the two by trade is how the engine shows where a sensitivity-based
view of an option book stops being trustworthy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .instruments import Trade
from .market import Market
from .scenarios import ScenarioSet


def scenario_pnl(trades: list[Trade], mkt: Market, scen: ScenarioSet) -> pd.DataFrame:
    """Full-revaluation P&L, one row per trade, one column per scenario."""
    markets = [mkt.apply_returns(*scen.row(i)) for i in range(scen.n)]
    out = {}
    for t in trades:
        v0 = t.pv(mkt)
        out[t.trade_id] = np.array([t.pv(ms) - v0 for ms in markets])
    return pd.DataFrame.from_dict(out, orient="index", columns=scen.labels)


def taylor_pnl(
    legs: pd.DataFrame, scen: ScenarioSet, order: str = "delta_gamma_vega"
) -> pd.DataFrame:
    """Sensitivity-based P&L per trade in cash terms, on simple returns R:
    delta_cash * R + 0.5 * gamma_cash * R^2 + vega * d_sigma.
    order: 'delta', 'delta_gamma' or 'delta_gamma_vega'."""
    out: dict[str, np.ndarray] = {}
    for row in legs.itertuples():
        ret = np.expm1(scen.fwd[row.commodity][:, row.month])
        pnl = row.delta_cash * ret
        if order != "delta":
            pnl = pnl + 0.5 * row.gamma_cash * ret**2
        if order == "delta_gamma_vega":
            pnl = pnl + row.vega * scen.vol[row.commodity][:, row.month]
        out[row.trade_id] = out.get(row.trade_id, 0.0) + pnl
    return pd.DataFrame.from_dict(out, orient="index", columns=scen.labels)


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
