"""Market state: monthly forward curves and ATM vol term structures.

Time is measured in years from the valuation date. Contract month m (m >= 1)
averages over [m/12, (m+1)/12); month 0 is the prompt month already in delivery
and is not traded in the synthetic book. Implied vol is one ATM number per
contract month and no smile, so the sticky-strike vs sticky-delta choice
(MAR21.27) does not arise: vol stays put when forwards move.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

FIXINGS_PER_MONTH = 21


def window(month: int) -> tuple[float, float]:
    return month / 12.0, (month + 1) / 12.0


def fixing_times(month: int) -> np.ndarray:
    """Business-day fixing times inside a contract month's averaging window."""
    start, end = window(month)
    return np.linspace(start, end, FIXINGS_PER_MONTH + 1)[1:]


def payment_time(month: int) -> float:
    return window(month)[1]


@dataclass(frozen=True)
class Curve:
    commodity: str
    unit: str
    fwd: np.ndarray  # forward price per contract month, index = month
    vol: np.ndarray  # ATM implied vol per contract month

    @property
    def n_months(self) -> int:
        return len(self.fwd)


@dataclass(frozen=True)
class Market:
    curves: dict[str, Curve]
    rate: float = 0.04
    time_shift: float = 0.0  # years rolled forward, used for theta

    def df(self, t: float) -> float:
        return float(np.exp(-self.rate * max(t - self.time_shift, 0.0)))

    def curve(self, commodity: str) -> Curve:
        return self.curves[commodity]

    def _with(self, commodity: str, **changes) -> Market:
        curves = dict(self.curves)
        curves[commodity] = replace(curves[commodity], **changes)
        return replace(self, curves=curves)

    def bump_fwd(
        self,
        commodity: str,
        rel: float,
        months: slice | int | None = None,
        log: bool = False,
    ) -> Market:
        """Relative forward shift: F * (1 + rel), or F * exp(rel) when log=True.
        months=None shifts the whole curve in parallel."""
        c = self.curves[commodity]
        mult = np.ones(c.n_months)
        mult[slice(None) if months is None else months] = (
            np.exp(rel) if log else 1.0 + rel
        )
        return self._with(commodity, fwd=c.fwd * mult)

    def bump_vol(
        self, commodity: str, abs_shift: float, months: slice | int | None = None
    ) -> Market:
        c = self.curves[commodity]
        vol = c.vol.copy()
        vol[slice(None) if months is None else months] += abs_shift
        return self._with(commodity, vol=vol)

    def apply_returns(
        self,
        fwd_log_ret: dict[str, np.ndarray],
        vol_abs: dict[str, np.ndarray] | None = None,
    ) -> Market:
        """Scenario market: log-returns on each forward pillar, absolute vol moves."""
        curves = {}
        for name, c in self.curves.items():
            fwd = c.fwd * np.exp(fwd_log_ret.get(name, 0.0))
            vol = c.vol + (vol_abs or {}).get(name, 0.0)
            curves[name] = replace(c, fwd=fwd, vol=np.maximum(vol, 0.01))
        return replace(self, curves=curves)

    def roll(self, days: int) -> Market:
        return replace(self, time_shift=self.time_shift + days / 252.0)


def sample_market(n_months: int = 36) -> Market:
    """Illustrative curves in USD. TTF is quoted in USD/MMBtu so all gas legs
    share a unit; the book therefore carries no FX risk by construction."""
    m = np.arange(n_months)
    seasonal = np.cos(
        2 * np.pi * (m - 1) / 12.0
    )  # winter premium, peak in month 1 ~ January
    brent = 78.0 - 0.35 * m  # mild backwardation, no seasonality
    ttf = 11.0 + 1.6 * seasonal - 0.04 * m
    jkm = 12.2 + 1.9 * seasonal - 0.04 * m
    vol_decay = np.exp(-m / 10.0)
    return Market(
        curves={
            "BRENT": Curve("BRENT", "USD/bbl", brent, 0.26 + 0.10 * vol_decay),
            "TTF": Curve("TTF", "USD/MMBtu", ttf, 0.45 + 0.25 * vol_decay),
            "JKM": Curve("JKM", "USD/MMBtu", jkm, 0.42 + 0.25 * vol_decay),
        }
    )
