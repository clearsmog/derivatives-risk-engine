"""Daily P&L explain and the FRTB P&L attribution (PLA) test.

Today's book is exposed to each day of an out-of-sample history of market
moves (it is not aged along the path, so no option fixes or expires mid-test):

    HPL   (hypothetical P&L)       full revaluation, one business day later
    RTPL  (risk-theoretical P&L)   rebuilt from today's sensitivities:
          sum of d1 * m + 0.5 * d2 * m^2 over pillars, plus theta
    unexplained = HPL - RTPL       cross terms (spot-vol, FX-equity) and
                                   higher orders

MAR32 compares HPL and RTPL per trading desk over 250 days with a Spearman
correlation and a Kolmogorov-Smirnov distance. A desk is green if
Spearman > 0.80 and KS < 0.09, red if Spearman < 0.70 or KS > 0.12, and amber
otherwise (MAR32.42).

A risk model's RTPL can also come from a coarser factor set than HPL. The
proxy models here drive every pillar of a curve (commodity, rate, spread or
vol term structure) from one anchor pillar, either through fitted betas or
one-for-one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from .instruments import Trade
from .market import Market
from .scenarios import ScenarioSet
from .var import moves_for

GREEN = {"spearman": 0.80, "ks": 0.09}
RED = {"spearman": 0.70, "ks": 0.12}


def proxy_model(
    mkt: Market, history: ScenarioSet, naive: bool = False
) -> dict[str, np.ndarray]:
    """Per multi-pillar factor, the beta of each pillar's move on the anchor
    pillar's move, fitted by least squares on `history`. naive=True sets every
    beta to 1, copying the anchor's move onto the whole curve."""
    betas = {}
    for name, x in history.moves.items():
        f = mkt[name]
        if f.n == 1:
            continue
        if naive:
            betas[name] = np.ones(f.n)
        else:
            a = x[:, f.anchor]
            betas[name] = (x * a[:, None]).sum(axis=0) / (a @ a)
    return betas


def proxied(
    mkt: Market, path: ScenarioSet, betas: dict[str, np.ndarray]
) -> ScenarioSet:
    moves = {}
    for name, x in path.moves.items():
        if name in betas:
            moves[name] = np.outer(x[:, mkt[name].anchor], betas[name])
        else:
            moves[name] = x
    return ScenarioSet(moves, path.labels)


def explain(
    trades: list[Trade],
    mkt: Market,
    legs: pd.DataFrame,
    sens: pd.DataFrame,
    path: ScenarioSet,
    risk_model_path: ScenarioSet | None = None,
) -> pd.DataFrame:
    """Long table: one row per trade per day with HPL, the Greek components of
    RTPL, and the unexplained residual. `risk_model_path` (from `proxied`)
    replaces the moves the risk model sees; HPL always uses `path`."""
    rolled = [mkt.apply_moves(path.row(d)).roll(1) for d in range(path.n)]
    hpl = {t.trade_id: np.array([t.pv(m) - t.pv(mkt) for m in rolled]) for t in trades}

    m = moves_for(sens, risk_model_path or path)
    is_vol = (sens.kind == "vol").to_numpy()[:, None]
    first = sens.d1.to_numpy()[:, None] * m
    second = 0.5 * sens.d2.to_numpy()[:, None] * m**2
    ids = sens.trade_id.to_numpy()

    def by_trade(x: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(x, columns=path.labels).groupby(ids).sum()

    delta = by_trade(np.where(is_vol, 0.0, first))
    gamma = by_trade(np.where(is_vol, 0.0, second))
    vega = by_trade(np.where(is_vol, first + second, 0.0))
    theta = legs.groupby("trade_id").theta.sum()

    rows = []
    for t in trades:
        tid = t.trade_id
        comp = {
            k: (df.loc[tid].to_numpy() if tid in df.index else np.zeros(path.n))
            for k, df in (
                ("delta_pnl", delta),
                ("gamma_pnl", gamma),
                ("vega_pnl", vega),
            )
        }
        comp["theta_pnl"] = np.full(path.n, theta.get(tid, 0.0))
        rtpl = sum(comp.values())
        rows.append(
            pd.DataFrame(
                {
                    "day": path.labels,
                    "trade_id": tid,
                    "hpl": hpl[tid],
                    **comp,
                    "rtpl": rtpl,
                    "unexplained": hpl[tid] - rtpl,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def pla_test(hpl: np.ndarray, rtpl: np.ndarray) -> dict[str, float | str]:
    rho = float(spearmanr(hpl, rtpl).statistic)
    ks = float(ks_2samp(hpl, rtpl).statistic)
    if rho > GREEN["spearman"] and ks < GREEN["ks"]:
        zone = "green"
    elif rho < RED["spearman"] or ks > RED["ks"]:
        zone = "red"
    else:
        zone = "amber"
    return {"spearman": rho, "ks": ks, "zone": zone}


def pla_by_desk(explained: pd.DataFrame, trade_desks: pd.Series) -> pd.DataFrame:
    """PLA per trading desk."""
    df = explained.assign(desk=explained.trade_id.map(trade_desks))
    daily = df.groupby(["desk", "day"])[["hpl", "rtpl"]].sum().reset_index()
    out = {
        d: pla_test(g.hpl.to_numpy(), g.rtpl.to_numpy())
        for d, g in daily.groupby("desk")
    }
    return pd.DataFrame(out).T
