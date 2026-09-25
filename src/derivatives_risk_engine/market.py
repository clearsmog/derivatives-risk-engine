"""Market state as a registry of named risk factors.

Every market input is a Factor: a vector of pillar values with a kind that
fixes how it is shocked.

    kind     examples                         bump / scenario move
    price    CMDTY:TTF, EQ:SPX, FX:EUR        relative (log return)
    rate     IR:USD, IR:EUR (zero rates)      absolute
    spread   CS:ACME (CDS par spreads)        absolute
    vol      VOL:EQ:SPX, VOL:IR:USD           absolute

Naming is <class>:<name>; vols are VOL:<class>:<name>. FX spots are quoted as
USD per unit of foreign currency, so FX:EUR is EURUSD. The reporting currency
is USD.

Commodity curves have one pillar per contract month: month m averages over
[m/12, (m+1)/12) and pays at (m+1)/12. Vols are ATM term structures with no
smile, so the sticky-strike vs sticky-delta choice (MAR21.27) does not arise.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

FIXINGS_PER_MONTH = 21
IR_TENORS = np.array([0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30])  # = GIRR vertices
CS_TENORS = np.array([0.5, 1, 3, 5, 10])  # = CSR vertices


def window(month: int) -> tuple[float, float]:
    return month / 12.0, (month + 1) / 12.0


def fixing_times(month: int) -> np.ndarray:
    """Business-day fixing times inside a contract month's averaging window."""
    start, end = window(month)
    return np.linspace(start, end, FIXINGS_PER_MONTH + 1)[1:]


def payment_time(month: int) -> float:
    return window(month)[1]


@dataclass(frozen=True)
class Factor:
    name: str
    kind: str  # "price" | "rate" | "spread" | "vol"
    tenors: np.ndarray  # pillar times in years (a single 0.0 for spots)
    values: np.ndarray
    unit: str = ""
    anchor: int = 0  # pillar that drives the whole factor in the coarse proxy model

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def asset_class(self) -> str:
        parts = self.name.split(":")
        return parts[1] if parts[0] == "VOL" else parts[0]


@dataclass(frozen=True)
class Market:
    factors: dict[str, Factor]
    time_shift: float = 0.0  # years rolled forward, used for theta
    static: dict = field(
        default_factory=dict
    )  # non-risk inputs: dividend yields, recoveries

    # ---- lookups -----------------------------------------------------------

    def __getitem__(self, name: str) -> Factor:
        return self.factors[name]

    def interp(self, name: str, t: float) -> float:
        f = self.factors[name]
        if f.n == 1:
            return float(f.values[0])
        return float(np.interp(t, f.tenors, f.values))

    def df(self, ccy: str, t: float) -> float:
        """Discount factor to time t (valuation-date time) in currency ccy."""
        tau = max(t - self.time_shift, 0.0)
        return float(np.exp(-self.interp(f"IR:{ccy}", tau) * tau))

    def fx(self, ccy: str) -> float:
        """USD value of one unit of ccy."""
        return 1.0 if ccy == "USD" else float(self.factors[f"FX:{ccy}"].values[0])

    def spot(self, name: str) -> float:
        return float(self.factors[name].values[0])

    def vol_at(self, name: str, t: float) -> float:
        """Vol for expiry t, interpolated linearly in total variance."""
        f = self.factors[name]
        if f.n == 1 or t <= f.tenors[0]:
            return float(f.values[0])
        if t >= f.tenors[-1]:
            return float(f.values[-1])  # flat vol beyond the last pillar
        var = np.interp(t, f.tenors, f.values**2 * f.tenors)
        return float(np.sqrt(var / t))

    # ---- shocks ------------------------------------------------------------

    def _with(self, name: str, values: np.ndarray) -> Market:
        factors = dict(self.factors)
        factors[name] = replace(factors[name], values=values)
        return replace(self, factors=factors)

    def bump(
        self, name: str, amount: float, pillar: int | None = None, log: bool = False
    ) -> Market:
        """Shock one pillar (or the whole factor when pillar is None).

        price factors move relatively: x * (1 + amount), or x * exp(amount) with
        log=True; every other kind moves by `amount` in absolute terms.
        """
        f = self.factors[name]
        idx = slice(None) if pillar is None else pillar
        v = f.values.copy()
        if f.kind == "price":
            v[idx] = v[idx] * (np.exp(amount) if log else 1.0 + amount)
        else:
            v[idx] = v[idx] + amount
        return self._with(name, v)

    def apply_moves(self, moves: dict[str, np.ndarray]) -> Market:
        """Scenario market: log returns for prices, absolute changes otherwise."""
        factors = dict(self.factors)
        for name, m in moves.items():
            f = factors[name]
            if f.kind == "price":
                v = f.values * np.exp(m)
            elif f.kind == "vol":
                v = np.maximum(f.values + m, 0.2 * f.values)
            else:
                v = f.values + m
            factors[name] = replace(f, values=v)
        return replace(self, factors=factors)

    def roll(self, days: int) -> Market:
        return replace(self, time_shift=self.time_shift + days / 252.0)


def sample_market(n_months: int = 36) -> Market:
    """Illustrative levels, loosely shaped like late-2020s markets."""
    m = np.arange(n_months)
    months_t = np.array([payment_time(i) for i in m])
    seasonal = np.cos(2 * np.pi * (m - 1) / 12.0)
    vol_decay = np.exp(-m / 10.0)
    eq_t = np.array([0.25, 0.5, 1.0, 2.0])
    swn_t = np.array([0.5, 1.0, 2.0, 5.0])

    def f(name, kind, tenors, values, unit="", anchor=0):
        return Factor(
            name,
            kind,
            np.asarray(tenors, float),
            np.asarray(values, float),
            unit,
            anchor,
        )

    factors = [
        # Commodities, USD (TTF in USD/MMBtu so gas legs share a unit)
        f("CMDTY:BRENT", "price", months_t, 78.0 - 0.35 * m, "USD/bbl", 1),
        f(
            "CMDTY:TTF",
            "price",
            months_t,
            11.0 + 1.6 * seasonal - 0.04 * m,
            "USD/MMBtu",
            1,
        ),
        f(
            "CMDTY:JKM",
            "price",
            months_t,
            12.2 + 1.9 * seasonal - 0.04 * m,
            "USD/MMBtu",
            1,
        ),
        f("VOL:CMDTY:BRENT", "vol", months_t, 0.26 + 0.10 * vol_decay, anchor=1),
        f("VOL:CMDTY:TTF", "vol", months_t, 0.45 + 0.25 * vol_decay, anchor=1),
        f("VOL:CMDTY:JKM", "vol", months_t, 0.42 + 0.25 * vol_decay, anchor=1),
        # Rates: zero curves on the GIRR vertices
        f(
            "IR:USD",
            "rate",
            IR_TENORS,
            [
                0.0430,
                0.0420,
                0.0400,
                0.0380,
                0.0375,
                0.0378,
                0.0395,
                0.0410,
                0.0420,
                0.0415,
            ],
            anchor=5,
        ),
        f(
            "IR:EUR",
            "rate",
            IR_TENORS,
            [
                0.0200,
                0.0198,
                0.0195,
                0.0200,
                0.0210,
                0.0228,
                0.0260,
                0.0280,
                0.0285,
                0.0275,
            ],
            anchor=5,
        ),
        f("VOL:IR:USD", "vol", swn_t, [0.0105, 0.0102, 0.0098, 0.0090], "normal", 1),
        # Equities: spots and ATM vol term structures
        f("EQ:SPX", "price", [0.0], [5800.0], "index pts"),
        f("EQ:SX5E", "price", [0.0], [5000.0], "index pts (EUR)"),
        f("EQ:NWT", "price", [0.0], [180.0], "USD/share"),
        f("VOL:EQ:SPX", "vol", eq_t, [0.17, 0.175, 0.18, 0.19]),
        f("VOL:EQ:SX5E", "vol", eq_t, [0.19, 0.195, 0.20, 0.205]),
        f("VOL:EQ:NWT", "vol", eq_t, [0.34, 0.33, 0.32, 0.31]),
        # FX
        f("FX:EUR", "price", [0.0], [1.10], "USD per EUR"),
        f("VOL:FX:EUR", "vol", eq_t, [0.075, 0.078, 0.080, 0.083]),
        # Credit: CDS par spreads on the CSR vertices
        f(
            "CS:ACME",
            "spread",
            CS_TENORS,
            [0.0060, 0.0065, 0.0080, 0.0095, 0.0110],
            anchor=3,
        ),
        f(
            "CS:ZETA",
            "spread",
            CS_TENORS,
            [0.0280, 0.0300, 0.0340, 0.0380, 0.0400],
            anchor=3,
        ),
        f(
            "CS:CDXIG",
            "spread",
            CS_TENORS,
            [0.0040, 0.0045, 0.0055, 0.0065, 0.0080],
            anchor=3,
        ),
    ]
    static = {
        "dividend_yield": {"EQ:SPX": 0.013, "EQ:SX5E": 0.030, "EQ:NWT": 0.005},
        "currency": {"EQ:SPX": "USD", "EQ:SX5E": "EUR", "EQ:NWT": "USD"},
        "recovery": {"CS:ACME": 0.40, "CS:ZETA": 0.40, "CS:CDXIG": 0.40},
    }
    return Market({x.name: x for x in factors}, static=static)
