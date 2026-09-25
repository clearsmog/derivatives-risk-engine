"""Risk-factor scenarios: a synthetic daily history and named stress tests.

The history is generated, not downloaded, so the repo runs offline and the
numbers are reproducible. It is built to have the features that break naive
risk models:

* fat tails        Student-t shocks, 4 degrees of freedom
* vol clustering   a GARCH(1,1) multiplier shared across the market
* Samuelson effect front months move more than back months
* curve shape      level, slope and curvature factors per commodity
* spread risk      Brent and the gas hubs are only loosely correlated
* a stress window  a 60-day gas squeeze late in the sample, for stressed ES
* vol-price link   implied vol rises when gas rallies (positive skew)

Swap in a real history by supplying the same dict-of-arrays shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COMMODITIES = ("BRENT", "TTF", "JKM")
FRONT_VOL = {"BRENT": 0.020, "TTF": 0.040, "JKM": 0.035}  # daily log-return vol
LEVEL_CORR = np.array(
    [
        [1.00, 0.35, 0.40],
        [0.35, 1.00, 0.80],
        [0.40, 0.80, 1.00],
    ]
)
VOL_BETA = {"BRENT": 0.3, "TTF": 1.2, "JKM": 1.0}  # d(implied vol) per unit log-return


@dataclass(frozen=True)
class ScenarioSet:
    """n scenarios of forward log-returns and absolute implied-vol changes."""

    fwd: dict[str, np.ndarray]  # commodity -> (n, n_months)
    vol: dict[str, np.ndarray]  # commodity -> (n, n_months)
    labels: list[str]

    @property
    def n(self) -> int:
        return len(self.labels)

    def row(self, i: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        return (
            {k: v[i] for k, v in self.fwd.items()},
            {k: v[i] for k, v in self.vol.items()},
        )


def _garch_multiplier(n: int, rng: np.random.Generator) -> np.ndarray:
    omega, alpha, beta = 0.05, 0.08, 0.87  # long-run variance 1.0
    var = np.empty(n)
    var[0] = 1.0
    eps = rng.standard_normal(n)
    for t in range(1, n):
        var[t] = omega + alpha * var[t - 1] * eps[t - 1] ** 2 + beta * var[t - 1]
    return np.sqrt(var)


def synthetic_history(
    n_days: int = 500,
    n_months: int = 36,
    seed: int = 2024,
    stress_start: int | None = 400,
) -> ScenarioSet:
    rng = np.random.default_rng(seed)
    m = np.arange(n_months)
    samuelson = 0.45 + 0.55 * np.exp(-m / 8.0)
    loadings = np.stack(
        [
            samuelson,  # level
            0.30 * (m / (n_months - 1) - 0.5),  # slope
            0.15 * np.cos(np.pi * m / (n_months - 1)),  # curvature
        ]
    )

    df_t = 4
    t_scale = np.sqrt((df_t - 2) / df_t)  # unit variance
    chol = np.linalg.cholesky(LEVEL_CORR)
    chi = rng.chisquare(df_t, size=(n_days, 1)) / df_t
    level = (rng.standard_normal((n_days, 3)) @ chol.T) / np.sqrt(chi) * t_scale
    other = rng.standard_normal((n_days, 3, 2))
    garch = _garch_multiplier(n_days, rng)

    stress = np.ones(n_days)
    if stress_start is not None:
        stress[stress_start : stress_start + 60] = 2.5

    fwd, vol = {}, {}
    for j, c in enumerate(COMMODITIES):
        gas = c != "BRENT"
        mult = garch * (stress if gas else 1.0 + 0.3 * (stress - 1.0))
        factors = np.column_stack([level[:, j], other[:, j, 0], other[:, j, 1]])
        fwd[c] = FRONT_VOL[c] * mult[:, None] * (factors @ loadings)
        vol[c] = VOL_BETA[c] * fwd[c] + 0.004 * rng.standard_normal((n_days, n_months))
    return ScenarioSet(fwd, vol, [f"D{d:03d}" for d in range(n_days)])


def stress_scenarios(n_months: int = 36) -> ScenarioSet:
    """Parallel shocks sized on real episodes (magnitudes rounded)."""
    specs = {
        # label: {commodity: (relative price move, absolute vol move)}
        "Oil demand collapse (2020-style)": {
            "BRENT": (-0.50, 0.40),
            "TTF": (-0.35, 0.20),
            "JKM": (-0.35, 0.20),
        },
        "European gas squeeze (2022-style)": {
            "BRENT": (0.20, 0.10),
            "TTF": (1.50, 0.60),
            "JKM": (1.20, 0.50),
        },
        "Oil spike, gas flat": {
            "BRENT": (0.40, 0.15),
            "TTF": (0.0, 0.0),
            "JKM": (0.0, 0.0),
        },
        "JKM-TTF spread blowout": {
            "BRENT": (0.0, 0.0),
            "TTF": (-0.20, 0.10),
            "JKM": (0.40, 0.25),
        },
        "Vol crush": {"BRENT": (0.0, -0.10), "TTF": (0.0, -0.20), "JKM": (0.0, -0.20)},
    }
    fwd = {c: np.zeros((len(specs), n_months)) for c in COMMODITIES}
    vol = {c: np.zeros((len(specs), n_months)) for c in COMMODITIES}
    for i, shocks in enumerate(specs.values()):
        for c, (px, dv) in shocks.items():
            fwd[c][i] = np.log1p(px)
            vol[c][i] = dv
    return ScenarioSet(fwd, vol, list(specs))
