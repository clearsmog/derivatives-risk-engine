"""FRTB standardised approach, sensitivities-based method: commodity risk class.

Implements delta, vega and curvature for the commodity risk class under
BCBS MAR21 (version in force from 1 Jan 2023). Paragraph references are to
that text. The UK (PRA) and EU (CRR3) rulebooks carry the same commodity
parameters; check the local text before relying on this for a live filing.

TTF and JKM. MAR21.83 treats two commodities as distinct only when two
contracts differ solely in the commodity delivered. TTF (pipeline gas, NW
Europe) and JKM (LNG, NE Asia) are both natural gas, so the default treats them
as one commodity at two delivery locations: rho = rho_tenor * rho_basis (99.9%).
`gas_hubs_distinct=True` applies the 65% bucket-6 rho_cty instead. The default
is the more conservative reading for this book.

Out of scope: the default risk charge (MAR22, no issuer risk in a commodity
book) and the residual risk add-on (MAR23), which Asian options and collars
would likely attract.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from .instruments import Trade
from .market import Market

# MAR21.82 Table 11 and MAR21.83 Table 12
BUCKET = {"BRENT": 2, "TTF": 6, "JKM": 6}
DELTA_RW = {
    1: 0.30,
    2: 0.35,
    3: 0.60,
    4: 0.80,
    5: 0.40,
    6: 0.45,
    7: 0.20,
    8: 0.35,
    9: 0.25,
    10: 0.35,
    11: 0.50,
}
RHO_CTY = {
    1: 0.55,
    2: 0.95,
    3: 0.40,
    4: 0.80,
    5: 0.60,
    6: 0.65,
    7: 0.55,
    8: 0.45,
    9: 0.15,
    10: 0.40,
    11: 0.15,
}
RHO_TENOR = 0.99  # MAR21.83(1)(b)
RHO_BASIS = 0.999  # MAR21.83(1)(c)
UNDERLYING = {"BRENT": "crude oil", "TTF": "natural gas", "JKM": "natural gas"}

DELTA_TENORS = np.array([0.0, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30])  # MAR21.13(1)
VEGA_TENORS = np.array([0.5, 1, 3, 5, 10])  # MAR21.13(2)

# MAR21.92 Table 13: RW = min(RW_sigma * sqrt(LH / 10), 100%), RW_sigma = 55%, LH = 120
VEGA_RW = min(0.55 * np.sqrt(120 / 10), 1.0)
ALPHA = 0.01  # MAR21.93

SCENARIOS = ("low", "medium", "high")


def gamma_bc(b: int, c: int) -> float:
    """MAR21.85 cross-bucket correlation."""
    return 0.0 if 11 in (b, c) else 0.20


def adjust(rho: float, scenario: str) -> float:
    """MAR21.6 correlation scenarios."""
    if scenario == "high":
        return min(1.25 * rho, 1.0)
    if scenario == "low":
        return max(2.0 * rho - 1.0, 0.75 * rho)
    return rho


def allocate(t: float, vertices: np.ndarray) -> list[tuple[float, float]]:
    """Split a sensitivity at maturity t across the two adjacent vertices, linearly."""
    t = float(np.clip(t, vertices[0], vertices[-1]))
    j = int(np.searchsorted(vertices, t))
    if vertices[j] == t:
        return [(float(vertices[j]), 1.0)]
    lo, hi = vertices[j - 1], vertices[j]
    w_hi = (t - lo) / (hi - lo)
    return [(float(lo), 1.0 - w_hi), (float(hi), w_hi)]


# ---------------------------------------------------------------- inputs


def delta_weighted_sensitivities(
    legs: pd.DataFrame, time_shift: float = 0.0
) -> pd.DataFrame:
    """Per trade, commodity and regulatory tenor: s_k and WS_k = RW * s_k."""
    rows = []
    for r in legs.itertuples():
        for tenor, w in allocate(r.maturity - time_shift, DELTA_TENORS):
            b = BUCKET[r.commodity]
            s = w * r.delta_frtb
            rows.append((r.trade_id, r.commodity, b, tenor, s, DELTA_RW[b] * s))
    cols = [
        "trade_id",
        "commodity",
        "bucket",
        "tenor",
        "sensitivity",
        "weighted_sensitivity",
    ]
    return pd.DataFrame(rows, columns=cols).groupby(cols[:4], as_index=False).sum()


def vega_weighted_sensitivities(
    legs: pd.DataFrame, time_shift: float = 0.0
) -> pd.DataFrame:
    rows = []
    for r in legs[legs.is_option].itertuples():
        for tenor, w in allocate(r.option_expiry - time_shift, VEGA_TENORS):
            s = w * r.vega_frtb
            rows.append(
                (r.trade_id, r.commodity, BUCKET[r.commodity], tenor, s, VEGA_RW * s)
            )
    cols = [
        "trade_id",
        "commodity",
        "bucket",
        "tenor",
        "sensitivity",
        "weighted_sensitivity",
    ]
    return pd.DataFrame(rows, columns=cols).groupby(cols[:4], as_index=False).sum()


def curvature_cvr(trades: list[Trade], mkt: Market, legs: pd.DataFrame) -> pd.DataFrame:
    """MAR21.5(2): per trade and commodity, the curvature loss beyond delta
    under a parallel shift of the whole curve by the bucket's delta risk weight
    (MAR21.99). Summing over trades gives the net CVR_k per commodity."""
    delta_sum = legs.groupby(["trade_id", "commodity"]).delta_frtb.sum()
    rows = []
    for t in trades:
        for c in sorted(t.commodities):
            rw = DELTA_RW[BUCKET[c]]
            v0 = t.pv(mkt)
            s = delta_sum.get((t.trade_id, c), 0.0)
            up = -(t.pv(mkt.bump_fwd(c, rw)) - v0 - rw * s)
            dn = -(t.pv(mkt.bump_fwd(c, -rw)) - v0 + rw * s)
            rows.append((t.trade_id, c, BUCKET[c], up, dn))
    return pd.DataFrame(
        rows, columns=["trade_id", "commodity", "bucket", "cvr_up", "cvr_down"]
    )


# ----------------------------------------------------------- aggregation


def _cross_bucket(
    kb: dict[int, float], sb: dict[int, float], gamma_fn, psi: bool = False
) -> float:
    total = sum(k * k for k in kb.values())
    for b in kb:
        for c in kb:
            if b != c and not (psi and sb[b] < 0 and sb[c] < 0):
                total += gamma_fn(b, c) * sb[b] * sb[c]
    return total


def _linear_charge(
    ws: pd.DataFrame, rho_fn, scenario: str
) -> tuple[float, dict[int, float]]:
    """MAR21.4(4)-(5) for delta or vega, with the alternative S_b if needed."""
    net = ws.groupby(
        ["commodity", "bucket", "tenor"], as_index=False
    ).weighted_sensitivity.sum()
    kb, sb = {}, {}
    for b, g in net.groupby("bucket"):
        w = g.weighted_sensitivity.to_numpy()
        keys = list(zip(g.commodity, g.tenor))
        rho = np.array(
            [
                [
                    1.0 if i == j else adjust(rho_fn(b, keys[i], keys[j]), scenario)
                    for j in range(len(keys))
                ]
                for i in range(len(keys))
            ]
        )
        kb[b] = float(np.sqrt(max(w @ rho @ w, 0.0)))
        sb[b] = float(w.sum())
    gamma = lambda b, c: adjust(gamma_bc(b, c), scenario)  # noqa: E731
    total = _cross_bucket(kb, sb, gamma)
    if total < 0:
        sb = {b: max(min(sb[b], kb[b]), -kb[b]) for b in kb}
        total = _cross_bucket(kb, sb, gamma)
    return float(np.sqrt(max(total, 0.0))), kb


def _name_rho(b: int, a: str, c: str, distinct: bool) -> float:
    """rho_cty * rho_basis between two curves in bucket b."""
    if a == c:
        return 1.0
    if not distinct and UNDERLYING.get(a, a) == UNDERLYING.get(c, c):
        return RHO_BASIS
    return RHO_CTY[b]


def _delta_rho(b: int, k: tuple, l: tuple, distinct: bool) -> float:
    rho_tenor = 1.0 if k[1] == l[1] else RHO_TENOR
    return _name_rho(b, k[0], l[0], distinct) * rho_tenor


def _vega_rho(b: int, k: tuple, l: tuple, distinct: bool) -> float:
    """MAR21.94: delta correlation between the underlyings times option-maturity decay."""
    rho_delta = _name_rho(b, k[0], l[0], distinct)
    rho_opt = np.exp(-ALPHA * abs(k[1] - l[1]) / min(k[1], l[1]))
    return min(rho_delta * rho_opt, 1.0)


def _curvature_charge(
    cvr: pd.DataFrame, scenario: str, distinct: bool
) -> tuple[float, dict[int, float]]:
    """MAR21.5(3)-(4). Curvature correlations are the squared delta ones
    (MAR21.100-101); the scenario adjustment is applied to the squared value."""
    net = cvr.groupby(["commodity", "bucket"], as_index=False)[
        ["cvr_up", "cvr_down"]
    ].sum()
    kb, sb = {}, {}
    for b, g in net.groupby("bucket"):
        names = list(g.commodity)

        def k_side(x: np.ndarray) -> float:
            tot = float(np.sum(np.maximum(x, 0.0) ** 2))
            for i in range(len(x)):
                for j in range(len(x)):
                    if i != j and not (x[i] < 0 and x[j] < 0):
                        r = adjust(_name_rho(b, names[i], names[j], distinct) ** 2, scenario)
                        tot += r * x[i] * x[j]
            return float(np.sqrt(max(tot, 0.0)))

        up, dn = g.cvr_up.to_numpy(), g.cvr_down.to_numpy()
        k_up, k_dn = k_side(up), k_side(dn)
        use_up = k_up > k_dn or (k_up == k_dn and up.sum() > dn.sum())
        kb[b] = k_up if use_up else k_dn
        sb[b] = float(up.sum() if use_up else dn.sum())
    gamma = lambda b, c: adjust(gamma_bc(b, c) ** 2, scenario)  # noqa: E731
    return float(np.sqrt(max(_cross_bucket(kb, sb, gamma, psi=True), 0.0))), kb


@dataclass(frozen=True)
class SbmResult:
    by_scenario: (
        pd.DataFrame
    )  # rows: low/medium/high; cols: delta, vega, curvature, total
    binding_scenario: str
    capital: float
    bucket_detail: pd.DataFrame


def sbm_commodity(
    delta_ws: pd.DataFrame,
    vega_ws: pd.DataFrame,
    cvr: pd.DataFrame,
    gas_hubs_distinct: bool = False,
) -> SbmResult:
    if not gas_hubs_distinct and len(vega_ws):
        # MAR21.13(2): vega factors are not split by delivery location, so
        # same-underlying vegas become one risk factor and net.
        vega_ws = vega_ws.assign(commodity=vega_ws.commodity.map(UNDERLYING))
    rho_d = partial(_delta_rho, distinct=gas_hubs_distinct)
    rho_v = partial(_vega_rho, distinct=gas_hubs_distinct)
    rows, detail = {}, []
    for sc in SCENARIOS:
        d, d_kb = _linear_charge(delta_ws, rho_d, sc)
        v, v_kb = _linear_charge(vega_ws, rho_v, sc) if len(vega_ws) else (0.0, {})
        c, c_kb = _curvature_charge(cvr, sc, gas_hubs_distinct)
        rows[sc] = {"delta": d, "vega": v, "curvature": c, "total": d + v + c}
        for measure, kbs in (("delta", d_kb), ("vega", v_kb), ("curvature", c_kb)):
            detail += [(sc, measure, b, k) for b, k in kbs.items()]
    table = pd.DataFrame(rows).T.loc[list(SCENARIOS)]
    binding = str(table.total.idxmax())
    return SbmResult(
        table,
        binding,
        float(table.total.max()),
        pd.DataFrame(detail, columns=["scenario", "measure", "bucket", "Kb"]),
    )
