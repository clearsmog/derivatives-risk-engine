"""The cube must reproduce the Python VaR/ES exactly (skipped without Atoti)."""

import pytest

tt = pytest.importorskip("atoti")

from derivatives_risk_engine import var  # noqa: E402
from derivatives_risk_engine.cube import build  # noqa: E402
from derivatives_risk_engine.engine import run  # noqa: E402


def test_cube_var_and_es_reconcile_with_python():
    r = run(n_history=250, n_explain=5)
    total = r.trade_pnl.sum().to_numpy()
    with tt.Session.start() as session:
        cube, sens_cube = build(r, session)
        m = cube.measures
        res = cube.query(m["VaR 99%"], m["ES 97.5%"])
        pv01 = sens_cube.query(sens_cube.measures["PV01"])
    assert float(res["VaR 99%"].iloc[0]) == pytest.approx(var.var(total), rel=1e-9)
    assert float(res["ES 97.5%"].iloc[0]) == pytest.approx(
        var.expected_shortfall(total), rel=1e-9
    )
    assert float(pv01["PV01"].iloc[0]) == pytest.approx(
        (r.sens.d1[r.sens.kind == "rate"] * 1e-4).sum(), rel=1e-9
    )
