"""Historical-simulation VaR / ES by full revaluation, and the Taylor proxies
that approximate it.

Full revaluation reprices every leg under every scenario. The Taylor proxy
rebuilds P&L from the leg sensitivities:

    dV ~ delta_ln * r + 0.5 * gamma_ln * r^2 + vega * d_sigma

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
    """Sensitivity-based P&L per trade. order: 'delta', 'delta_gamma' or 'delta_gamma_vega'."""
    out: dict[str, np.ndarray] = {}
    for row in legs.itertuples():
        r = scen.fwd[row.commodity][:, row.month]
        pnl = row.delta_ln * r
        if order != "delta":
            pnl = pnl + 0.5 * row.gamma_ln * r**2
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


def component_var(pnl: pd.DataFrame, level: float = 0.99) -> pd.Series:
    """Each trade's P&L in the scenario that sets portfolio VaR.

    Contributions sum exactly to the portfolio VaR number (Euler allocation
    for a historical-simulation quantile). Noisy for a single scenario; an
    ES-based allocation averages the tail instead.
    """
    total = pnl.sum(axis=0).to_numpy()
    idx = int(np.argsort(total, kind="stable")[_var_index(len(total), level)])
    return -pnl.iloc[:, idx]


def risk_summary(pnl: pd.DataFrame) -> dict[str, float]:
    total = pnl.sum(axis=0).to_numpy()
    return {
        "VaR 99% 1d": var(total, 0.99),
        "ES 97.5% 1d": expected_shortfall(total, 0.975),
        "Worst day": float(-total.min()),
        "Undiversified VaR 99%": float(sum(var(r, 0.99) for r in pnl.to_numpy())),
    }
