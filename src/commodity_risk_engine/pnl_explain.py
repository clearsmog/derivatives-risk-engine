"""Daily P&L explain and the FRTB P&L attribution (PLA) test.

Today's book is exposed to each day of an out-of-sample history of market
moves (it is not aged along the path, so no option fixes or expires mid-test):

    HPL   (hypothetical P&L)       full revaluation, one business day later
    RTPL  (risk-theoretical P&L)   rebuilt from today's sensitivities:
          delta_cash * R + 0.5 * gamma_cash * R^2 + vega * d_sigma + theta  (R simple return)
    unexplained = HPL - RTPL

MAR32 compares the two series per trading desk over 250 days with a Spearman
correlation and a Kolmogorov-Smirnov distance. A desk is green if
Spearman > 0.80 and KS < 0.09, red if Spearman < 0.70 or KS > 0.12, and amber
otherwise (MAR32.42). Here RTPL is a Taylor proxy standing in for a risk
model's own P&L, which is the comparison PLA is designed to police.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from .instruments import Trade
from .market import Market
from .scenarios import ScenarioSet

GREEN = {"spearman": 0.80, "ks": 0.09}
RED = {"spearman": 0.70, "ks": 0.12}


def proxy_model(history: ScenarioSet, anchor: int = 1, naive: bool = False) -> ScenarioSet:
    """A coarse one-factor risk model per commodity, fitted on `history`.

    Every pillar's move is predicted as beta_m times the anchor month's move,
    with beta_m estimated by least squares (separately for forwards and vols).
    It captures the level of each curve and the Samuelson decay of vol along
    it, but not independent curve-shape moves such as a winter-summer spread.
    Returned as betas packed in a one-row ScenarioSet-shaped container.

    naive=True sets every beta to 1: the anchor month's move is copied onto the
    whole curve, overstating back-month moves (they are less volatile).
    """
    def betas(x: np.ndarray) -> np.ndarray:
        if naive:
            return np.ones(x.shape[1])
        a = x[:, anchor]
        return (x * a[:, None]).sum(axis=0) / (a @ a)

    return ScenarioSet(
        {c: betas(v)[None, :] for c, v in history.fwd.items()},
        {c: betas(v)[None, :] for c, v in history.vol.items()},
        ["beta"],
    )


def explain(
    trades: list[Trade],
    mkt: Market,
    legs: pd.DataFrame,
    path: ScenarioSet,
    proxy: ScenarioSet | None = None,
    anchor: int = 1,
) -> pd.DataFrame:
    """Long table: one row per trade per day with HPL, the Greek components of
    RTPL, and the unexplained residual.

    With `proxy` (from proxy_model), RTPL sees only the anchor month's move
    mapped through the fitted betas, as a coarse risk model would.
    """
    rolled = [mkt.apply_returns(*path.row(d)).roll(1) for d in range(path.n)]
    base = {t.trade_id: t.pv(mkt) for t in trades}
    hpl = {
        t.trade_id: np.array([t.pv(m) - base[t.trade_id] for m in rolled])
        for t in trades
    }

    comp = {}
    for r in legs.itertuples():
        c, m = r.commodity, r.month
        if proxy is None:
            log_ret, dvol = path.fwd[c][:, m], path.vol[c][:, m]
        else:
            log_ret = proxy.fwd[c][0, m] * path.fwd[c][:, anchor]
            dvol = proxy.vol[c][0, m] * path.vol[c][:, anchor]
        ret = np.expm1(log_ret)
        parts = np.stack(
            [
                r.delta_cash * ret,
                0.5 * r.gamma_cash * ret**2,
                r.vega * dvol,
                np.full(path.n, r.theta),
            ]
        )
        comp[r.trade_id] = comp.get(r.trade_id, 0.0) + parts

    rows = []
    for t in trades:
        d, g, v, th = comp[t.trade_id]
        rtpl = d + g + v + th
        rows.append(
            pd.DataFrame(
                {
                    "day": path.labels,
                    "trade_id": t.trade_id,
                    "hpl": hpl[t.trade_id],
                    "delta_pnl": d,
                    "gamma_pnl": g,
                    "vega_pnl": v,
                    "theta_pnl": th,
                    "rtpl": rtpl,
                    "unexplained": hpl[t.trade_id] - rtpl,
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


def pla_by_desk(explained: pd.DataFrame, trade_books: pd.Series) -> pd.DataFrame:
    """PLA per book, treating each book as a trading desk."""
    df = explained.assign(book=explained.trade_id.map(trade_books))
    daily = df.groupby(["book", "day"])[["hpl", "rtpl"]].sum().reset_index()
    out = {
        b: pla_test(g.hpl.to_numpy(), g.rtpl.to_numpy())
        for b, g in daily.groupby("book")
    }
    return pd.DataFrame(out).T
