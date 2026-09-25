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
from .sensitivities import sensitivities, trade_frame


@dataclass
class RiskRun:
    market: Market
    trades: list[Trade]
    trade_info: pd.DataFrame
    legs: pd.DataFrame  # one row per leg: pv, theta
    sens: pd.DataFrame  # one row per leg and risk-factor pillar
    history: ScenarioSet
    stress: ScenarioSet
    leg_pnl: np.ndarray  # (n_legs, n_history) full-revaluation P&L vectors
    leg_stress: np.ndarray  # (n_legs, n_stress)
    trade_pnl: pd.DataFrame  # trade x history scenario
    explain: dict[str, pd.DataFrame]  # risk model -> P&L explain on the test path
    frtb_book: frtb.SbmResult
    frtb_desks: dict[str, frtb.SbmResult]
    frtb_book_hubs_distinct: frtb.SbmResult  # TTF and JKM as distinct commodities


def run(
    mkt: Market | None = None, n_history: int = 500, n_explain: int = 250
) -> RiskRun:
    mkt = mkt or sample_market()
    trades = build_book(mkt)
    info = trade_frame(trades)
    legs, sens = sensitivities(trades, mkt)
    hist = synthetic_history(mkt, n_days=n_history)
    stress = stress_scenarios(mkt)
    leg_pnl = var.leg_pnl(trades, mkt, hist)
    trade_pnl = (
        pd.DataFrame(leg_pnl, columns=hist.labels)
        .groupby(legs.trade_id.to_numpy(), sort=False)
        .sum()
    )

    path = synthetic_history(mkt, n_days=n_explain, seed=7, stress_start=None)
    models = {
        "full": path,
        "beta": pnl_explain.proxied(mkt, path, pnl_explain.proxy_model(mkt, hist)),
        "naive": pnl_explain.proxied(
            mkt, path, pnl_explain.proxy_model(mkt, hist, naive=True)
        ),
    }
    explain = {
        k: pnl_explain.explain(trades, mkt, legs, sens, path, p)
        for k, p in models.items()
    }

    desks = info.set_index("trade_id").desk
    frtb_desks = {
        d: frtb.run_sbm([t for t in trades if t.desk == d], mkt, sens)
        for d in desks.unique()
    }
    return RiskRun(
        market=mkt,
        trades=trades,
        trade_info=info,
        legs=legs,
        sens=sens,
        history=hist,
        stress=stress,
        leg_pnl=leg_pnl,
        leg_stress=var.leg_pnl(trades, mkt, stress),
        trade_pnl=trade_pnl,
        explain=explain,
        frtb_book=frtb.run_sbm(trades, mkt, sens),
        frtb_desks=frtb_desks,
        frtb_book_hubs_distinct=frtb.run_sbm(trades, mkt, sens, gas_hubs_distinct=True),
    )


def _var_es(p: np.ndarray) -> dict[str, float]:
    return {"VaR 99%": var.var(p), "ES 97.5%": var.expected_shortfall(p)}


def summary_tables(r: RiskRun) -> dict[str, pd.DataFrame]:
    """The tables a morning risk pack would show."""
    desks = r.trade_info.set_index("trade_id").desk
    by_desk = r.trade_pnl.groupby(r.trade_pnl.index.map(desks)).sum()
    total = r.trade_pnl.sum().to_numpy()

    var_table = pd.DataFrame(
        {d: _var_es(p.to_numpy()) for d, p in by_desk.iterrows()}
    ).T
    var_table.loc["Sum of desks"] = var_table.sum()
    var_table.loc["FIRM"] = _var_es(total)

    s = r.sens.assign(desk=r.sens.trade_id.map(desks))
    scale = {"price": 0.01, "rate": 1e-4, "spread": 1e-4, "vol": 0.01}
    s = s.assign(per_unit=s.d1 * s.kind.map(scale))
    s.loc[s.factor.str.startswith("VOL:IR"), "per_unit"] = s.d1 * 1e-4
    greeks = s.pivot_table(
        index="desk", columns="kind", values="per_unit", aggfunc="sum"
    ).fillna(0.0)
    greeks = greeks.rename(
        columns={
            "price": "delta: P&L per +1% (all prices)",
            "rate": "PV01: P&L per +1bp (all curves)",
            "spread": "CS01: P&L per +1bp",
            "vol": "vega: P&L per +1 vol pt (1bp normal)",
        }
    )

    taylor = {
        o: _var_es(var.taylor_pnl(r.sens, r.history, o).sum().to_numpy())
        for o in ("delta", "delta_gamma", "delta_gamma_vega")
    }
    taylor["full revaluation"] = _var_es(total)

    stress = (
        pd.DataFrame(r.leg_stress, columns=r.stress.labels)
        .groupby(r.legs.trade_id.to_numpy())
        .sum()
    )
    stress_by_desk = stress.groupby(stress.index.map(desks)).sum().T
    stress_by_desk["FIRM"] = stress_by_desk.sum(axis=1)

    pla = pd.concat(
        [
            pnl_explain.pla_by_desk(ex, desks).add_prefix(f"{k}_")
            for k, ex in r.explain.items()
        ],
        axis=1,
    )
    cols = ["hpl", "delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl", "unexplained"]
    ex = r.explain["full"].assign(desk=lambda d: d.trade_id.map(desks))
    explain_std = ex.groupby(["desk", "day"])[cols].sum().groupby("desk").std()

    frtb_desk = pd.DataFrame(
        {
            d: res.by_scenario.loc[res.binding_scenario]
            for d, res in r.frtb_desks.items()
        }
    ).T
    frtb_desk["binding"] = [res.binding_scenario for res in r.frtb_desks.values()]
    for label, res in (
        ("FIRM", r.frtb_book),
        ("FIRM, TTF/JKM distinct", r.frtb_book_hubs_distinct),
    ):
        frtb_desk.loc[label] = [
            *res.by_scenario.loc[res.binding_scenario],
            res.binding_scenario,
        ]

    return {
        "var": var_table,
        "greeks_by_desk": greeks,
        "taylor_vs_full": pd.DataFrame(taylor).T,
        "stress": stress_by_desk,
        "pnl_explain_std": explain_std,
        "pla": pla,
        "frtb_by_scenario": r.frtb_book.by_scenario,
        "frtb_by_class": r.frtb_book.by_class,
        "frtb_by_desk": frtb_desk,
    }
