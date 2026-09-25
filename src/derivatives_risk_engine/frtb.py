"""FRTB standardised approach, sensitivities-based method (BCBS MAR21).

Risk classes: GIRR, CSR non-securitisation, equity, commodity and FX; delta,
vega and curvature for each; three correlation scenarios. Parameters are
transcribed in docs/mar21-parameters.md with paragraph references.

Choices the text leaves open, made explicitly here:
* TTF and JKM default to one commodity at two delivery locations
  (rho_basis 99.9%); `gas_hubs_distinct=True` uses the bucket-6 rho_cty (65%).
* Curvature is computed for instruments with optionality (MAR21.2(1)-(3)).
* The optional sqrt(2) GIRR / FX risk-weight reductions (MAR21.44, 21.88) and
  the 1.5 FX curvature scalar (MAR21.98) are not applied.
* Curvature scenario correlations adjust the squared delta correlation.

Out of scope: CSR securitisation, equity repo, inflation and cross-currency
basis, the default risk charge (MAR22) and the residual risk add-on (MAR23).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from .instruments import Trade
from .market import Market

SCENARIOS = ("low", "medium", "high")
ALPHA = 0.01  # MAR21.93 option-maturity decay
VEGA_TENORS = np.array([0.5, 1, 3, 5, 10])

# ---- GIRR (MAR21.41-50) ---------------------------------------------------------
GIRR_TENORS = np.array([0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30])
GIRR_RW = np.array(
    [0.017, 0.017, 0.016, 0.013, 0.012, 0.011, 0.011, 0.011, 0.011, 0.011]
)
GIRR_RHO = np.array(  # Table 2, same curve, different tenors
    [
        [1.000, 0.970, 0.914, 0.811, 0.719, 0.566, 0.400, 0.400, 0.400, 0.400],
        [0.970, 1.000, 0.970, 0.914, 0.861, 0.763, 0.566, 0.419, 0.400, 0.400],
        [0.914, 0.970, 1.000, 0.970, 0.942, 0.887, 0.763, 0.657, 0.566, 0.419],
        [0.811, 0.914, 0.970, 1.000, 0.985, 0.956, 0.887, 0.823, 0.763, 0.657],
        [0.719, 0.861, 0.942, 0.985, 1.000, 0.980, 0.932, 0.887, 0.844, 0.763],
        [0.566, 0.763, 0.887, 0.956, 0.980, 1.000, 0.970, 0.942, 0.914, 0.861],
        [0.400, 0.566, 0.763, 0.887, 0.932, 0.970, 1.000, 0.985, 0.970, 0.942],
        [0.400, 0.419, 0.657, 0.823, 0.887, 0.942, 0.985, 1.000, 0.990, 0.970],
        [0.400, 0.400, 0.566, 0.763, 0.844, 0.914, 0.970, 0.990, 1.000, 0.985],
        [0.400, 0.400, 0.419, 0.657, 0.763, 0.861, 0.942, 0.970, 0.985, 1.000],
    ]
)

# ---- CSR non-securitisation (MAR21.51-57) ---------------------------------------
CSR_TENORS = np.array([0.5, 1, 3, 5, 10])
CSR_RW = {
    1: 0.005,
    2: 0.010,
    3: 0.050,
    4: 0.030,
    5: 0.030,
    6: 0.020,
    7: 0.015,
    8: 0.025,
    9: 0.020,
    10: 0.040,
    11: 0.120,
    12: 0.070,
    13: 0.085,
    14: 0.055,
    15: 0.050,
    16: 0.120,
    17: 0.015,
    18: 0.050,
}
_SECTOR = {
    **{b: b for b in range(1, 9)},
    **{b: b - 8 for b in range(9, 16)},
    16: 16,
    17: 17,
    18: 18,
}
_CSR_SECTOR_ROWS = {  # Table 5, upper triangle by sector group
    1: [0.75, 0.10, 0.20, 0.25, 0.20, 0.15, 0.10, 0.00, 0.45, 0.45],
    2: [0.05, 0.15, 0.20, 0.15, 0.10, 0.10, 0.00, 0.45, 0.45],
    3: [0.05, 0.15, 0.20, 0.05, 0.20, 0.00, 0.45, 0.45],
    4: [0.20, 0.25, 0.05, 0.05, 0.00, 0.45, 0.45],
    5: [0.25, 0.05, 0.15, 0.00, 0.45, 0.45],
    6: [0.05, 0.20, 0.00, 0.45, 0.45],
    7: [0.05, 0.00, 0.45, 0.45],
    8: [0.00, 0.45, 0.45],
    16: [0.00, 0.00],
    17: [0.75],
}
_SECTOR_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18]
CSR_SECTOR_GAMMA: dict[tuple[int, int], float] = {}
for _i, _a in enumerate(_SECTOR_ORDER[:-1]):
    for _b, _g in zip(_SECTOR_ORDER[_i + 1 :], _CSR_SECTOR_ROWS[_a]):
        CSR_SECTOR_GAMMA[(_a, _b)] = CSR_SECTOR_GAMMA[(_b, _a)] = _g

# ---- Equity (MAR21.72-80) --------------------------------------------------------
EQ_RW = {
    1: 0.55,
    2: 0.60,
    3: 0.45,
    4: 0.55,
    5: 0.30,
    6: 0.35,
    7: 0.40,
    8: 0.50,
    9: 0.70,
    10: 0.50,
    11: 0.70,
    12: 0.15,
    13: 0.25,
}
EQ_RHO = {
    **dict.fromkeys(range(1, 5), 0.15),
    **dict.fromkeys(range(5, 9), 0.25),
    9: 0.075,
    10: 0.125,
    12: 0.80,
    13: 0.80,
}

# ---- Commodity (MAR21.81-85) -----------------------------------------------------
CMDTY_TENORS = np.array([0.0, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30])
CMDTY_RW = {
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
CMDTY_RHO = {
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
UNDERLYING = {"BRENT": "crude oil", "TTF": "natural gas", "JKM": "natural gas"}

# ---- Instrument classification for this book -------------------------------------
BUCKET = {
    "CMDTY": {"BRENT": 2, "TTF": 6, "JKM": 6},
    "EQ": {"SPX": 12, "SX5E": 12, "NWT": 8},  # NWT: large cap, advanced, technology
    "CSR": {
        "ACME": 4,
        "ZETA": 14,
        "CDXIG": 17,
    },  # IG industrials, HY telecom/tech, IG index
}


# MAR21.92 Table 13: min(55% * sqrt(LH / 10), 100%)
def _vega_rw(lh: int) -> float:
    return min(0.55 * np.sqrt(lh / 10), 1.0)


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


def _opt_decay(a: float, b: float) -> float:
    return float(np.exp(-ALPHA * abs(a - b) / min(a, b)))


# ---- Risk class definitions --------------------------------------------------------


@dataclass(frozen=True)
class RiskClass:
    """Everything MAR21 needs per risk class. Keys: delta (name, tenor),
    vega (name, option vertex, underlying vertex)."""

    code: str
    delta_tenors: np.ndarray | None

    def bucket(self, name: str):
        if self.code in ("GIRR", "FX"):
            return name  # one bucket per currency
        return BUCKET[self.code][name]

    def delta_rw(self, b, tenor: float) -> float:
        if self.code == "GIRR":
            return float(GIRR_RW[int(np.searchsorted(GIRR_TENORS, tenor))])
        if self.code == "CSR":
            return CSR_RW[b]
        if self.code == "EQ":
            return EQ_RW[b]
        if self.code == "CMDTY":
            return CMDTY_RW[b]
        return 0.15  # FX, MAR21.87

    def name_rho(self, b, a: str, c: str, distinct: bool) -> float:
        """Correlation between two different names (curves, issuers, stocks) in bucket b."""
        if a == c:
            return 1.0
        if self.code == "CSR":
            return 0.80 if b in (17, 18) else 0.35
        if self.code == "EQ":
            return EQ_RHO[b]
        if self.code == "CMDTY":
            if not distinct and UNDERLYING[a] == UNDERLYING[c]:
                return 0.999  # rho_basis
            return CMDTY_RHO[b]
        return 0.999  # GIRR different curves, same currency

    def rho_delta(self, b, k: tuple, l: tuple, distinct: bool) -> float:
        rho_name = self.name_rho(b, k[0], l[0], distinct)
        if self.code == "GIRR":
            i, j = (int(np.searchsorted(GIRR_TENORS, x)) for x in (k[1], l[1]))
            return rho_name * float(GIRR_RHO[i, j])
        if self.code == "CSR":
            return rho_name * (1.0 if k[1] == l[1] else 0.65)
        if self.code == "CMDTY":
            return rho_name * (1.0 if k[1] == l[1] else 0.99)
        return rho_name

    def gamma(self, b, c) -> float:
        if self.code == "GIRR":
            return 0.50
        if self.code == "FX":
            return 0.60
        if self.code == "CMDTY":
            return 0.0 if 11 in (b, c) else 0.20
        if self.code == "EQ":
            if 11 in (b, c):
                return 0.0
            if b <= 10 and c <= 10:
                return 0.15
            if {b, c} == {12, 13}:
                return 0.75
            return 0.45
        # CSR: gamma_rating * gamma_sector
        rating = 0.5 if b <= 15 and c <= 15 and (b <= 8) != (c <= 8) else 1.0
        sb, sc = _SECTOR[b], _SECTOR[c]
        return rating * (1.0 if sb == sc else CSR_SECTOR_GAMMA[(sb, sc)])

    def vega_rw(self, b) -> float:
        if self.code == "EQ":
            return _vega_rw(20) if b in (*range(1, 9), 12, 13) else _vega_rw(60)
        lh = {"GIRR": 60, "CSR": 120, "CMDTY": 120, "FX": 40}[self.code]
        return _vega_rw(lh)

    def rho_vega(self, b, k: tuple, l: tuple, distinct: bool) -> float:
        """MAR21.93-94."""
        rho_opt = _opt_decay(k[1], l[1])
        if self.code == "GIRR":
            return min(rho_opt * _opt_decay(k[2], l[2]), 1.0)
        return min(self.name_rho(b, k[0], l[0], distinct) * rho_opt, 1.0)

    def curvature_rw(self, b) -> float:
        """MAR21.98-99: delta RW; for GIRR the highest tenor RW, shifted in parallel."""
        if self.code == "GIRR":
            return float(GIRR_RW.max())
        return self.delta_rw(b, 0.0)


RISK_CLASSES = {
    "GIRR": RiskClass("GIRR", GIRR_TENORS),
    "CSR": RiskClass("CSR", CSR_TENORS),
    "EQ": RiskClass("EQ", None),
    "CMDTY": RiskClass("CMDTY", CMDTY_TENORS),
    "FX": RiskClass("FX", None),
}
_FACTOR_CLASS = {"IR": "GIRR", "CS": "CSR", "EQ": "EQ", "CMDTY": "CMDTY", "FX": "FX"}


def classify(factor: str) -> tuple[str, str]:
    """Factor name -> (risk class, underlying name). VOL:<cls>:<name> maps to its underlying."""
    parts = factor.split(":")
    if parts[0] == "VOL":
        parts = parts[1:]
    return _FACTOR_CLASS[parts[0]], parts[1]


# ---- Weighted sensitivities --------------------------------------------------------


def delta_weighted_sensitivities(sens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in sens[sens.kind != "vol"].itertuples():
        cls, name = classify(r.factor)
        rc = RISK_CLASSES[cls]
        b = rc.bucket(name)
        vertices = (
            [(0.0, 1.0)]
            if rc.delta_tenors is None
            else allocate(r.tenor, rc.delta_tenors)
        )
        for tenor, w in vertices:
            s = w * r.frtb
            rows.append(
                (r.trade_id, cls, b, name, tenor, 0.0, s, rc.delta_rw(b, tenor) * s)
            )
    return _frame(rows)


def vega_weighted_sensitivities(
    sens: pd.DataFrame, gas_hubs_distinct: bool = False
) -> pd.DataFrame:
    rows = []
    for r in sens[sens.kind == "vol"].itertuples():
        cls, name = classify(r.factor)
        rc = RISK_CLASSES[cls]
        b = rc.bucket(name)
        if cls == "CMDTY" and not gas_hubs_distinct:
            # MAR21.13(2): commodity vega factors are not split by delivery location
            name = UNDERLYING[name]
        und = (
            allocate(r.underlying_maturity, VEGA_TENORS)
            if cls == "GIRR"
            else [(0.0, 1.0)]
        )
        for t_opt, w1 in allocate(r.option_expiry, VEGA_TENORS):
            for t_und, w2 in und:
                s = w1 * w2 * r.frtb
                rows.append(
                    (r.trade_id, cls, b, name, t_opt, t_und, s, rc.vega_rw(b) * s)
                )
    return _frame(rows)


def _frame(rows: list) -> pd.DataFrame:
    cols = [
        "trade_id",
        "risk_class",
        "bucket",
        "name",
        "tenor",
        "tenor2",
        "sensitivity",
        "weighted_sensitivity",
    ]
    df = pd.DataFrame(rows, columns=cols)
    return df.groupby(cols[:6], as_index=False, dropna=False)[cols[6:]].sum()


def curvature_cvr(trades: list[Trade], mkt: Market, sens: pd.DataFrame) -> pd.DataFrame:
    """MAR21.5(2) per trade and curvature risk factor (a whole curve or a spot),
    for trades with optionality. Summing over trades gives the net CVR_k."""
    delta_sum = sens[sens.kind != "vol"].groupby(["trade_id", "factor"]).frtb.sum()
    rows = []
    for t in trades:
        if not t.has_optionality:
            continue
        v0 = t.pv(mkt)
        factors = {
            f for leg in t.legs for f, _ in leg.pillars(mkt) if not f.startswith("VOL:")
        }
        for f in sorted(factors):
            cls, name = classify(f)
            rc = RISK_CLASSES[cls]
            b = rc.bucket(name)
            rw = rc.curvature_rw(b)
            s = delta_sum.get((t.trade_id, f), 0.0)
            up = -(t.pv(mkt.bump(f, rw)) - v0 - rw * s)
            dn = -(t.pv(mkt.bump(f, -rw)) - v0 + rw * s)
            rows.append((t.trade_id, cls, b, name, up, dn))
    return pd.DataFrame(
        rows, columns=["trade_id", "risk_class", "bucket", "name", "cvr_up", "cvr_down"]
    )


# ---- Aggregation --------------------------------------------------------------------


def _cross_bucket(kb: dict, sb: dict, gamma_fn, psi: bool = False) -> float:
    total = sum(k * k for k in kb.values())
    for b in kb:
        for c in kb:
            if b != c and not (psi and sb[b] < 0 and sb[c] < 0):
                total += gamma_fn(b, c) * sb[b] * sb[c]
    return total


def _linear_charge(
    ws: pd.DataFrame, rc: RiskClass, rho_fn, scenario: str
) -> tuple[float, dict]:
    """MAR21.4(4)-(5), with the alternative S_b when the cross term goes negative."""
    net = ws.groupby(
        ["bucket", "name", "tenor", "tenor2"], as_index=False
    ).weighted_sensitivity.sum()
    kb, sb = {}, {}
    for b, g in net.groupby("bucket"):
        w = g.weighted_sensitivity.to_numpy()
        keys = list(zip(g.name, g.tenor, g.tenor2))
        n = len(keys)
        rho = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                rho[i, j] = rho[j, i] = adjust(rho_fn(b, keys[i], keys[j]), scenario)
        kb[b] = float(np.sqrt(max(w @ rho @ w, 0.0)))
        sb[b] = float(w.sum())
    gamma = lambda b, c: adjust(rc.gamma(b, c), scenario)  # noqa: E731
    total = _cross_bucket(kb, sb, gamma)
    if total < 0:
        sb = {b: max(min(sb[b], kb[b]), -kb[b]) for b in kb}
        total = _cross_bucket(kb, sb, gamma)
    return float(np.sqrt(max(total, 0.0))), kb


def _curvature_charge(
    cvr: pd.DataFrame, rc: RiskClass, scenario: str, distinct: bool
) -> tuple[float, dict]:
    """MAR21.5(3)-(4); correlations are the squared delta ones (MAR21.100-101)."""
    net = cvr.groupby(["bucket", "name"], as_index=False)[["cvr_up", "cvr_down"]].sum()
    kb, sb = {}, {}
    for b, g in net.groupby("bucket"):
        names = list(g.name)

        def k_side(x: np.ndarray) -> float:
            tot = float(np.sum(np.maximum(x, 0.0) ** 2))
            for i in range(len(x)):
                for j in range(len(x)):
                    if i != j and not (x[i] < 0 and x[j] < 0):
                        tot += (
                            adjust(
                                rc.name_rho(b, names[i], names[j], distinct) ** 2,
                                scenario,
                            )
                            * x[i]
                            * x[j]
                        )
            return float(np.sqrt(max(tot, 0.0)))

        up, dn = g.cvr_up.to_numpy(), g.cvr_down.to_numpy()
        k_up, k_dn = k_side(up), k_side(dn)
        use_up = k_up > k_dn or (k_up == k_dn and up.sum() > dn.sum())
        kb[b] = k_up if use_up else k_dn
        sb[b] = float(up.sum() if use_up else dn.sum())
    gamma = lambda b, c: adjust(rc.gamma(b, c) ** 2, scenario)  # noqa: E731
    return float(np.sqrt(max(_cross_bucket(kb, sb, gamma, psi=True), 0.0))), kb


@dataclass(frozen=True)
class SbmResult:
    detail: pd.DataFrame  # scenario, risk_class, measure, charge
    by_scenario: pd.DataFrame  # scenario x measure (delta, vega, curvature, total)
    binding_scenario: str
    capital: float

    @property
    def by_class(self) -> pd.DataFrame:
        """Risk class x measure in the binding scenario."""
        d = self.detail[self.detail.scenario == self.binding_scenario]
        t = d.pivot_table(
            index="risk_class", columns="measure", values="charge", aggfunc="sum"
        ).fillna(0.0)
        t["total"] = t.sum(axis=1)
        return t


def sbm(
    delta_ws: pd.DataFrame,
    vega_ws: pd.DataFrame,
    cvr: pd.DataFrame,
    gas_hubs_distinct: bool = False,
) -> SbmResult:
    rows = []
    for sc in SCENARIOS:
        for code, rc in RISK_CLASSES.items():
            d = delta_ws[delta_ws.risk_class == code]
            v = vega_ws[vega_ws.risk_class == code]
            c = cvr[cvr.risk_class == code]
            if len(d):
                rows.append(
                    (
                        sc,
                        code,
                        "delta",
                        _linear_charge(
                            d, rc, partial(rc.rho_delta, distinct=gas_hubs_distinct), sc
                        )[0],
                    )
                )
            if len(v):
                rows.append(
                    (
                        sc,
                        code,
                        "vega",
                        _linear_charge(
                            v, rc, partial(rc.rho_vega, distinct=gas_hubs_distinct), sc
                        )[0],
                    )
                )
            if len(c):
                rows.append(
                    (
                        sc,
                        code,
                        "curvature",
                        _curvature_charge(c, rc, sc, gas_hubs_distinct)[0],
                    )
                )
    detail = pd.DataFrame(rows, columns=["scenario", "risk_class", "measure", "charge"])
    table = detail.pivot_table(
        index="scenario", columns="measure", values="charge", aggfunc="sum"
    ).fillna(0.0)
    table = table.reindex(index=list(SCENARIOS), columns=["delta", "vega", "curvature"], fill_value=0.0)
    table["total"] = table.sum(axis=1)
    binding = str(table.total.idxmax())
    return SbmResult(detail, table, binding, float(table.total.max()))


def run_sbm(
    trades: list[Trade],
    mkt: Market,
    sens: pd.DataFrame,
    gas_hubs_distinct: bool = False,
) -> SbmResult:
    ids = {t.trade_id for t in trades}
    s = sens[sens.trade_id.isin(ids)]
    return sbm(
        delta_weighted_sensitivities(s),
        vega_weighted_sensitivities(s, gas_hubs_distinct),
        curvature_cvr(trades, mkt, s),
        gas_hubs_distinct,
    )
