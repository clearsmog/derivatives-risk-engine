"""Risk-factor scenarios: a synthetic multi-asset daily history and named stresses.

The history is generated, not downloaded, so the repo runs offline and the
numbers are reproducible. Eight correlated drivers (global equity, Brent,
TTF, JKM, USD rates, EUR rates, credit, EURUSD) share one Student-t day scale
and one GARCH(1,1) volatility path, and each factor loads on them:

* fat tails and vol clustering   t(4) day scale, GARCH multiplier, a 60-day
                                 stress window at 2.5x inside the sample
* commodity curves               level (Samuelson decay), slope, curvature, and
                                 a winter-summer seasonal factor on gas
* rates                          level, slope and curvature per curve, in bp
* credit                         spread moves proportional to spread level
* vol-price links                equity and oil vol rise when prices fall; gas
                                 vol rises when gas rallies

Moves are log returns for price factors and absolute changes for rates,
spreads and vols (see Market.apply_moves).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market import Market

DRIVERS = ["EQ", "BRENT", "TTF", "JKM", "USD", "EUR", "CREDIT", "FX"]
CORR = np.array(
    [
        #  EQ   BRENT  TTF   JKM   USD   EUR  CREDIT  FX
        [1.00, 0.30, 0.10, 0.10, 0.20, 0.20, -0.60, 0.20],  # EQ
        [0.30, 1.00, 0.35, 0.40, 0.15, 0.10, -0.20, 0.10],  # BRENT
        [0.10, 0.35, 1.00, 0.80, 0.05, 0.10, -0.05, -0.05],  # TTF
        [0.10, 0.40, 0.80, 1.00, 0.05, 0.05, -0.05, 0.00],  # JKM
        [0.20, 0.15, 0.05, 0.05, 1.00, 0.60, -0.20, -0.30],  # USD rates
        [0.20, 0.10, 0.10, 0.05, 0.60, 1.00, -0.15, 0.30],  # EUR rates
        [-0.60, -0.20, -0.05, -0.05, -0.20, -0.15, 1.00, -0.10],  # credit
        [0.20, 0.10, -0.05, 0.00, -0.30, 0.30, -0.10, 1.00],  # EURUSD
    ]
)

# Commodity settings: front-month daily log vol, vol reaction per unit return
CMDTY = {"BRENT": (0.020, -0.3), "TTF": (0.040, 1.2), "JKM": (0.035, 1.0)}
# Equity: daily vol, loading on the global driver, currency-local vol reaction
EQUITY = {"SPX": (0.011, 1.00), "SX5E": (0.012, 0.85), "NWT": (0.022, 0.70)}
EQ_VOL_BETA = -0.9  # d(implied vol) per unit return (leverage effect)
RATES = {"USD": 0.00035, "EUR": 0.0003}  # daily level move, 3.5bp / 3bp
FX_VOL = 0.0055
CREDIT_REL_VOL = 0.03  # daily relative spread move on the common driver


@dataclass(frozen=True)
class ScenarioSet:
    """n scenarios of moves per factor: dict name -> (n, n_pillars)."""

    moves: dict[str, np.ndarray]
    labels: list[str]

    @property
    def n(self) -> int:
        return len(self.labels)

    def row(self, i: int) -> dict[str, np.ndarray]:
        return {k: v[i] for k, v in self.moves.items()}


def _garch_multiplier(n: int, rng: np.random.Generator) -> np.ndarray:
    omega, alpha, beta = 0.05, 0.08, 0.87  # long-run variance 1.0
    var = np.empty(n)
    var[0] = 1.0
    eps = rng.standard_normal(n)
    for t in range(1, n):
        var[t] = omega + alpha * var[t - 1] * eps[t - 1] ** 2 + beta * var[t - 1]
    return np.sqrt(var)


def synthetic_history(
    mkt: Market, n_days: int = 500, seed: int = 2024, stress_start: int | None = 400
) -> ScenarioSet:
    rng = np.random.default_rng(seed)
    df_t = 4
    t_scale = np.sqrt((df_t - 2) / df_t)
    # Floor the chi-square draw so one day cannot scale moves by more than 2x t(4)'s unit scale.
    chi = np.maximum(rng.chisquare(df_t, size=(n_days, 1)) / df_t, 0.25)
    z = (
        (rng.standard_normal((n_days, len(DRIVERS))) @ np.linalg.cholesky(CORR).T)
        / np.sqrt(chi)
        * t_scale
    )
    drv = dict(zip(DRIVERS, z.T))
    scale = _garch_multiplier(n_days, rng)
    if stress_start is not None:
        scale = scale.copy()
        scale[stress_start : stress_start + 60] *= 2.5
    s = scale[:, None]
    moves: dict[str, np.ndarray] = {}

    # Commodities
    season = (
        rng.standard_normal((n_days, 2))
        @ np.linalg.cholesky([[1.0, 0.7], [0.7, 1.0]]).T
    )
    for j, (c, (front, vol_beta)) in enumerate(CMDTY.items()):
        f = mkt[f"CMDTY:{c}"]
        m = np.arange(f.n)
        level = 0.45 + 0.55 * np.exp(-m / 8.0)
        slope = 0.30 * (m / (f.n - 1) - 0.5)
        curv = 0.15 * np.cos(np.pi * m / (f.n - 1))
        ret = (
            front
            * s
            * (
                np.outer(drv[c], level)
                + np.outer(rng.standard_normal(n_days), slope)
                + np.outer(rng.standard_normal(n_days), curv)
            )
        )
        if c != "BRENT":
            ret += (
                front
                * s
                * 0.35
                * np.outer(season[:, j - 1], np.cos(2 * np.pi * (m - 1) / 12.0))
            )
        moves[f.name] = ret
        moves[f"VOL:{f.name}"] = vol_beta * ret + 0.004 * rng.standard_normal(ret.shape)

    # Equities and their vols
    for name, (vol, beta) in EQUITY.items():
        idio = np.sqrt(1 - beta**2) * rng.standard_normal(n_days)
        ret = vol * scale * (beta * drv["EQ"] + idio)
        moves[f"EQ:{name}"] = ret[:, None]
        vf = mkt[f"VOL:EQ:{name}"]
        decay = np.exp(-vf.tenors / 2.0)
        moves[vf.name] = EQ_VOL_BETA * np.outer(
            ret, decay
        ) + 0.003 * rng.standard_normal((n_days, vf.n))

    # FX
    fx = FX_VOL * scale * drv["FX"]
    moves["FX:EUR"] = fx[:, None]
    moves["VOL:FX:EUR"] = 0.002 * s * rng.standard_normal((n_days, mkt["VOL:FX:EUR"].n))

    # Rates: level, slope, curvature in absolute rate units
    for ccy, bp in RATES.items():
        f = mkt[f"IR:{ccy}"]
        lt = np.log(f.tenors)
        slope = np.clip((lt - np.log(5.0)) / np.log(6.0), -1.0, 1.0)
        curv = np.exp(-((lt - np.log(3.0)) ** 2))
        moves[f.name] = (
            bp
            * s
            * (
                np.outer(drv[ccy], np.ones(f.n))
                + 0.5 * np.outer(rng.standard_normal(n_days), slope)
                + 0.25 * np.outer(rng.standard_normal(n_days), curv)
            )
        )
    level_usd = np.abs(moves["IR:USD"][:, 5])
    moves["VOL:IR:USD"] = 0.00005 * s * rng.standard_normal(
        (n_days, mkt["VOL:IR:USD"].n)
    ) + 0.1 * (level_usd[:, None] - level_usd.mean())

    # Credit: relative moves on a common driver plus name-specific noise
    for name in ("CS:ACME", "CS:ZETA", "CS:CDXIG"):
        f = mkt[name]
        idio = 0.0 if name == "CS:CDXIG" else 0.6
        rel = (
            CREDIT_REL_VOL
            * scale
            * (drv["CREDIT"] + idio * rng.standard_normal(n_days))
        )
        moves[name] = np.outer(rel, f.values)

    return ScenarioSet(moves, [f"D{d:03d}" for d in range(n_days)])


STRESSES = {
    # label: {factor or prefix: shock}; prices relative, vols/rates absolute,
    # spreads as a multiplier on the current curve.
    "Oil demand collapse (2020-style)": {
        "CMDTY:BRENT": -0.50,
        "VOL:CMDTY:BRENT": 0.40,
        "CMDTY:TTF": -0.35,
        "CMDTY:JKM": -0.35,
        "VOL:CMDTY:TTF": 0.20,
        "VOL:CMDTY:JKM": 0.20,
        "EQ:SPX": -0.30,
        "EQ:SX5E": -0.35,
        "EQ:NWT": -0.35,
        "VOL:EQ:SPX": 0.30,
        "VOL:EQ:SX5E": 0.30,
        "VOL:EQ:NWT": 0.40,
        "FX:EUR": -0.03,
        "IR:USD": -0.0100,
        "IR:EUR": -0.0040,
        "VOL:IR:USD": 0.004,
        "CS:ACME": 2.5,
        "CS:ZETA": 3.0,
        "CS:CDXIG": 2.5,
    },
    "European gas squeeze (2022-style)": {
        "CMDTY:BRENT": 0.20,
        "VOL:CMDTY:BRENT": 0.10,
        "CMDTY:TTF": 1.50,
        "CMDTY:JKM": 1.20,
        "VOL:CMDTY:TTF": 0.60,
        "VOL:CMDTY:JKM": 0.50,
        "EQ:SPX": -0.10,
        "EQ:SX5E": -0.20,
        "EQ:NWT": -0.15,
        "VOL:EQ:SPX": 0.08,
        "VOL:EQ:SX5E": 0.15,
        "VOL:EQ:NWT": 0.10,
        "FX:EUR": -0.12,
        "IR:USD": 0.0150,
        "IR:EUR": 0.0150,
        "VOL:IR:USD": 0.003,
        "CS:ACME": 1.5,
        "CS:ZETA": 1.8,
        "CS:CDXIG": 1.5,
    },
    "Oil spike with gas flat": {"CMDTY:BRENT": 0.40, "VOL:CMDTY:BRENT": 0.15},
    "Rates shock (+200bp)": {
        "IR:USD": 0.0200,
        "IR:EUR": 0.0150,
        "VOL:IR:USD": 0.003,
        "EQ:SPX": -0.10,
        "EQ:SX5E": -0.08,
        "EQ:NWT": -0.15,
        "FX:EUR": -0.04,
        "CS:ACME": 1.3,
        "CS:ZETA": 1.5,
        "CS:CDXIG": 1.3,
    },
    "Credit event (HY default wave)": {
        "CS:ZETA": 4.0,
        "CS:ACME": 2.0,
        "CS:CDXIG": 2.0,
        "EQ:SPX": -0.15,
        "EQ:SX5E": -0.15,
        "EQ:NWT": -0.20,
        "VOL:EQ:SPX": 0.12,
        "VOL:EQ:SX5E": 0.12,
        "VOL:EQ:NWT": 0.15,
        "IR:USD": -0.0050,
    },
    "Vol crush": {
        "VOL:CMDTY:BRENT": -0.10,
        "VOL:CMDTY:TTF": -0.20,
        "VOL:CMDTY:JKM": -0.20,
        "VOL:EQ:SPX": -0.05,
        "VOL:EQ:SX5E": -0.05,
        "VOL:EQ:NWT": -0.08,
        "VOL:FX:EUR": -0.02,
        "VOL:IR:USD": -0.002,
    },
}


def stress_scenarios(mkt: Market) -> ScenarioSet:
    labels = list(STRESSES)
    moves = {name: np.zeros((len(labels), f.n)) for name, f in mkt.factors.items()}
    for i, spec in enumerate(STRESSES.values()):
        for name, shock in spec.items():
            f = mkt[name]
            if f.kind == "price":
                moves[name][i] = np.log1p(shock)
            elif f.kind == "spread":
                moves[name][i] = (shock - 1.0) * f.values
            else:
                moves[name][i] = shock
    return ScenarioSet(moves, labels)
