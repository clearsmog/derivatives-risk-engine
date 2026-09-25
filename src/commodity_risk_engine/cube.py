"""Load a risk run into an Atoti cube.

Grain of the fact table is the leg (trade x contract month). Each leg carries
its full-revaluation P&L vector over the historical scenarios and over the
stress scenarios, so VaR and ES aggregate correctly at any level: vectors are
summed first, and the quantile is taken on the sum. Summing VaRs would not.

    uv run --extra cube python -m commodity_risk_engine.cube
"""

from __future__ import annotations

import atoti as tt

from .engine import RiskRun, run
from .frtb import BUCKET


def build_cube(r: RiskRun, session: tt.Session) -> tt.Cube:
    legs = r.legs.assign(
        bucket=r.legs.commodity.map(BUCKET).astype(int),
        pnl_vector=list(r.leg_pnl),
        stress_vector=list(r.leg_stress),
    ).drop(columns=["option_expiry"])
    fact = session.read_pandas(
        legs,
        keys={"trade_id", "leg"},
        table_name="Legs",
        default_values={"month": 0, "bucket": 0},
    )
    trades = session.read_pandas(r.trade_info, keys={"trade_id"}, table_name="Trades")
    fact.join(trades, fact["trade_id"] == trades["trade_id"])

    cube = session.create_cube(fact, "Commodity Risk")
    h, lvl, m = cube.hierarchies, cube.levels, cube.measures

    for col in ("book", "counterparty", "counterparty_type", "product"):
        h[col] = [trades[col]]
    h["month"] = [fact["month"]]
    h["bucket"] = [fact["bucket"]]

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

    for label, col in {
        "PV": "pv",
        "Delta (x1% move = /100)": "delta_ln",
        "Cash gamma": "gamma_cash",
        "Vega (per vol point = /100)": "vega",
        "Theta 1d": "theta",
        "FRTB delta sensitivity": "delta_frtb",
        "FRTB vega sensitivity": "vega_frtb",
    }.items():
        m[label] = tt.agg.sum(fact[col])
        m[label].formatter = "DOUBLE[#,###]"
    return cube


def main() -> None:
    r = run()
    session = tt.Session.start()
    cube = build_cube(r, session)
    lvl, m = cube.levels, cube.measures
    print(
        cube.query(
            m["VaR 99%"],
            m["ES 97.5%"],
            m["Sum of trade VaR"],
            m["Diversification"],
            levels=[lvl["book"]],
        )
    )
    print(cube.query(m["VaR 99%"], m["PV"], m["Cash gamma"], levels=[lvl["commodity"]]))
    print(f"Atoti UI: {session.url}  (Ctrl+C to stop)")
    session.wait()


if __name__ == "__main__":
    main()
