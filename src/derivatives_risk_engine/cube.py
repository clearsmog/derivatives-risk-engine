"""Load a risk run into Atoti.

Two cubes share one Trades table (desk, book, asset class, counterparty):

* "Risk" at leg grain. Each leg carries its full-revaluation P&L vector over
  the historical scenarios and over the stress scenarios, so VaR and ES
  aggregate correctly at any level: vectors are summed first, and the
  quantile is taken on the sum. Summing VaRs would not.
* "Sensitivities" at (leg, risk-factor pillar) grain: delta per 1%, PV01,
  CS01, vega and the FRTB sensitivity, sliceable by factor, tenor and desk.

    uv run --extra cube python -m derivatives_risk_engine.cube
"""

from __future__ import annotations

import atoti as tt

from .engine import RiskRun, run


def _trades(r: RiskRun, session: tt.Session) -> tt.Table:
    return session.read_pandas(r.trade_info, keys={"trade_id"}, table_name="Trades")


def build_risk_cube(r: RiskRun, session: tt.Session, trades: tt.Table) -> tt.Cube:
    legs = r.legs.assign(pnl_vector=list(r.leg_pnl), stress_vector=list(r.leg_stress))
    legs = legs.drop(columns=["option_expiry", "underlying_maturity"])
    fact = session.read_pandas(legs, keys={"trade_id", "leg"}, table_name="Legs")
    fact.join(trades, fact["trade_id"] == trades["trade_id"])

    cube = session.create_cube(fact, "Risk")
    h, lvl, m = cube.hierarchies, cube.levels, cube.measures
    for col in (
        "desk",
        "book",
        "asset_class",
        "counterparty",
        "counterparty_type",
        "product",
    ):
        h[col] = [trades[col]]

    pnl = m["pnl_vector.SUM"]
    m["VaR 99%"] = -tt.array.quantile(pnl, 0.01, mode="inc", interpolation="lower")
    n_tail = max(int(r.history.n * 0.025), 1)
    m["ES 97.5%"] = -tt.array.mean(tt.array.n_lowest(pnl, n_tail))
    m["Worst day"] = -tt.array.min(pnl)
    # Sum of standalone trade VaRs, to read diversification off any cell.
    m["Sum of trade VaR"] = tt.agg.sum(
        m["VaR 99%"], scope=tt.OriginScope({lvl["trade_id"]})
    )
    m["Diversification"] = m["Sum of trade VaR"] - m["VaR 99%"]

    stress = m["stress_vector.SUM"]
    for i, label in enumerate(r.stress.labels):
        m[f"Stress: {label}"] = stress[i]

    for label, col in {"PV": "pv", "Theta 1d": "theta"}.items():
        m[label] = tt.agg.sum(fact[col])
        m[label].formatter = "DOUBLE[#,###]"
    return cube


def build_sensitivity_cube(
    r: RiskRun, session: tt.Session, trades: tt.Table
) -> tt.Cube:
    s = r.sens.drop(columns=["option_expiry", "underlying_maturity"]).rename(
        columns={"asset_class": "factor_class"}
    )
    scale = s.kind.map({"price": 0.01, "rate": 1e-4, "spread": 1e-4, "vol": 0.01})
    scale[s.factor.str.startswith("VOL:IR")] = 1e-4
    s = s.assign(per_unit=s.d1 * scale)
    fact = session.read_pandas(
        s,
        keys={"trade_id", "leg", "factor", "pillar"},
        table_name="Sensitivities",
        default_values={"pillar": 0, "tenor": 0.0},
    )
    fact.join(trades, fact["trade_id"] == trades["trade_id"])

    cube = session.create_cube(fact, "Sensitivities")
    h, m = cube.hierarchies, cube.measures
    for col in ("desk", "book", "asset_class", "counterparty", "product"):
        h[col] = [trades[col]]
    h["tenor"] = [fact["tenor"]]

    m["Sensitivity per unit move"] = tt.agg.sum(fact["per_unit"])
    m["FRTB sensitivity"] = tt.agg.sum(fact["frtb"])
    for label, kind in {
        "Delta per 1%": "price",
        "PV01": "rate",
        "CS01": "spread",
        "Vega per vol pt": "vol",
    }.items():
        m[label] = tt.agg.sum(tt.where(fact["kind"] == kind, fact["per_unit"], 0.0))
    for name in (
        "Sensitivity per unit move",
        "FRTB sensitivity",
        "Delta per 1%",
        "PV01",
        "CS01",
        "Vega per vol pt",
    ):
        m[name].formatter = "DOUBLE[#,###]"
    return cube


def build(r: RiskRun, session: tt.Session) -> tuple[tt.Cube, tt.Cube]:
    trades = _trades(r, session)
    return build_risk_cube(r, session, trades), build_sensitivity_cube(
        r, session, trades
    )


def main() -> None:
    r = run()
    session = tt.Session.start()
    risk, sens = build(r, session)
    lvl, m = risk.levels, risk.measures
    print(
        risk.query(
            m["VaR 99%"],
            m["ES 97.5%"],
            m["Sum of trade VaR"],
            m["Diversification"],
            levels=[lvl["desk"]],
        )
    )
    print(
        sens.query(
            sens.measures["PV01"], levels=[sens.levels["desk"], sens.levels["tenor"]]
        )
    )
    print(f"Atoti UI: {session.url}  (Ctrl+C to stop)")
    session.wait()


if __name__ == "__main__":
    main()
