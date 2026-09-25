"""The cube must reproduce the Python VaR/ES exactly (skipped without Atoti)."""

import pytest

tt = pytest.importorskip("atoti")

from commodity_risk_engine import var  # noqa: E402
from commodity_risk_engine.cube import build_cube  # noqa: E402
from commodity_risk_engine.engine import run  # noqa: E402


def test_cube_var_and_es_reconcile_with_python():
    r = run(n_history=250, n_explain=5)
    total = r.trade_pnl.sum().to_numpy()
    with tt.Session.start() as session:
        cube = build_cube(r, session)
        m = cube.measures
        res = cube.query(m["VaR 99%"], m["ES 97.5%"])
    assert float(res["VaR 99%"].iloc[0]) == pytest.approx(var.var(total), rel=1e-9)
    assert float(res["ES 97.5%"].iloc[0]) == pytest.approx(
        var.expected_shortfall(total), rel=1e-9
    )
