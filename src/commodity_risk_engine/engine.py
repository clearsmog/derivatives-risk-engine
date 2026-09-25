"""One call that runs the whole daily risk batch and returns every table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import frtb, pnl_explain, var
from .instruments import Trade
from .market import Market, sample_market
from .portfolio import build_book
from .scenarios import ScenarioSet, stress_scenarios, synthetic_history
from .sensitivities import leg_sensitivities, trade_frame


@dataclass
class RiskRun:
    market: Market
    trades: list[Trade]
    trade_info: pd.DataFrame
    legs: pd.DataFrame  # sensitivities, one row per leg
    history: ScenarioSet
    stress: ScenarioSet
    leg_pnl: np.ndarray  # (n_legs, n_history) full-reval P&L vectors
    leg_stress: np.ndarray  # (n_legs, n_stress)
    trade_pnl: pd.DataFrame  # trade x history scenario
    explain: pd.DataFrame  # P&L explain on the out-of-sample path
    explain_proxy: pd.DataFrame  # same, risk model on a front-month proxy
    frtb_book: frtb.SbmResult
    frtb_desks: dict[str, frtb.SbmResult]


def _leg_pnl(trades: list[Trade], mkt: Market, scen: ScenarioSet) -> np.ndarray:
    markets = [mkt.apply_returns(*scen.row(i)) for i in range(scen.n)]
    rows = []
    for t in trades:
        for leg in t.legs:
            v0 = leg.pv(mkt)
            rows.append([leg.pv(m) - v0 for m in markets])
    return np.array(rows)


def _sbm(trades: list[Trade], mkt: Market, legs: pd.DataFrame) -> frtb.SbmResult:
    return frtb.sbm_commodity(
        frtb.delta_weighted_sensitivities(legs),
        frtb.vega_weighted_sensitivities(legs),
        frtb.curvature_cvr(trades, mkt, legs),
    )


def run(
    mkt: Market | None = None, n_history: int = 500, n_explain: int = 250
) -> RiskRun:
    mkt = mkt or sample_market()
    trades = build_book(mkt)
    info = trade_frame(trades)
    legs = leg_sensitivities(trades, mkt)
    hist = synthetic_history(n_days=n_history)
    stress = stress_scenarios()
    leg_pnl = _leg_pnl(trades, mkt, hist)
    trade_pnl = (
        pd.DataFrame(leg_pnl, columns=hist.labels)
        .groupby(legs.trade_id.to_numpy())
        .sum()
    )

    path = synthetic_history(n_days=n_explain, seed=7, stress_start=None)
    books = info.set_index("trade_id").book
    desks = {}
    for book in books.unique():
        ids = set(books[books == book].index)
        desks[book] = _sbm(
            [t for t in trades if t.trade_id in ids], mkt, legs[legs.trade_id.isin(ids)]
        )

    return RiskRun(
        market=mkt,
        trades=trades,
        trade_info=info,
        legs=legs,
        history=hist,
        stress=stress,
        leg_pnl=leg_pnl,
        leg_stress=_leg_pnl(trades, mkt, stress),
        trade_pnl=trade_pnl,
        explain=pnl_explain.explain(trades, mkt, legs, path),
        explain_proxy=pnl_explain.explain(trades, mkt, legs, path, proxy_month=1),
        frtb_book=_sbm(trades, mkt, legs),
        frtb_desks=desks,
    )


def summary_tables(r: RiskRun) -> dict[str, pd.DataFrame]:
    """The tables a morning risk pack would show."""
    books = r.trade_info.set_index("trade_id").book
    by_book = r.trade_pnl.groupby(r.trade_pnl.index.map(books)).sum()
    total = r.trade_pnl.sum().to_numpy()

    var_rows = {
        b: {
            "VaR 99%": var.var(p.to_numpy()),
            "ES 97.5%": var.expected_shortfall(p.to_numpy()),
        }
        for b, p in by_book.iterrows()
    }
    var_rows["BOOK"] = {
        "VaR 99%": var.var(total),
        "ES 97.5%": var.expected_shortfall(total),
    }
    var_table = pd.DataFrame(var_rows).T
    var_table.loc["Sum of desks"] = var_table.drop("BOOK").sum()

    greeks = r.legs.groupby(["commodity"])[
        ["pv", "delta_ln", "gamma_cash", "vega", "theta"]
    ].sum()
    greeks = greeks.rename(
        columns={"delta_ln": "delta (1% move x100)", "gamma_cash": "cash gamma"}
    )

    taylor = {}
    for order in ("delta", "delta_gamma", "delta_gamma_vega"):
        tp = var.taylor_pnl(r.legs, r.history, order).sum().to_numpy()
        taylor[order] = {"VaR 99%": var.var(tp), "ES 97.5%": var.expected_shortfall(tp)}
    taylor["full revaluation"] = {
        "VaR 99%": var.var(total),
        "ES 97.5%": var.expected_shortfall(total),
    }

    stress = (
        pd.DataFrame(r.leg_stress, columns=r.stress.labels)
        .groupby(r.legs.trade_id.to_numpy())
        .sum()
    )
    stress_by_book = stress.groupby(stress.index.map(books)).sum().T
    stress_by_book["BOOK"] = stress_by_book.sum(axis=1)

    pla = pnl_explain.pla_by_desk(r.explain, books).add_prefix("full_")
    pla_proxy = pnl_explain.pla_by_desk(r.explain_proxy, books).add_prefix("proxy_")

    frtb_desk = pd.DataFrame(
        {
            d: res.by_scenario.loc[res.binding_scenario]
            for d, res in r.frtb_desks.items()
        }
    ).T
    frtb_desk["binding"] = [res.binding_scenario for res in r.frtb_desks.values()]
    frtb_desk.loc["BOOK"] = [
        *r.frtb_book.by_scenario.loc[r.frtb_book.binding_scenario],
        r.frtb_book.binding_scenario,
    ]

    return {
        "var": var_table,
        "greeks": greeks,
        "taylor_vs_full": pd.DataFrame(taylor).T,
        "stress": stress_by_book,
        "pla": pd.concat([pla, pla_proxy], axis=1),
        "frtb_by_scenario": r.frtb_book.by_scenario,
        "frtb_by_desk": frtb_desk,
    }
