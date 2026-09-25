"""Closed-form and Monte Carlo pricers for commodity futures options.

Everything prices off forwards (Black-76), which is how listed Brent, TTF and
JKM options are quoted. Rates are a flat continuously compounded discount rate.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black76(
    fwd: float, strike: float, vol: float, t: float, df: float, is_call: bool
) -> float:
    """Black-76 price of a European option on a forward, per unit of notional."""
    if t <= 0.0 or vol <= 0.0:
        intrinsic = max(fwd - strike, 0.0) if is_call else max(strike - fwd, 0.0)
        return df * intrinsic
    sd = vol * np.sqrt(t)
    d1 = (np.log(fwd / strike) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if is_call:
        return df * (fwd * norm.cdf(d1) - strike * norm.cdf(d2))
    return df * (strike * norm.cdf(-d2) - fwd * norm.cdf(-d1))


def black76_greeks(
    fwd: float, strike: float, vol: float, t: float, df: float, is_call: bool
) -> dict[str, float]:
    """Analytic forward delta, gamma and vega (per 1.00 of vol) for Black-76.

    Used by the tests to check the bump-and-reprice engine, not by the engine.
    """
    sd = vol * np.sqrt(t)
    d1 = (np.log(fwd / strike) + 0.5 * sd * sd) / sd
    delta = df * (norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0)
    gamma = df * norm.pdf(d1) / (fwd * sd)
    vega = df * fwd * norm.pdf(d1) * np.sqrt(t)
    return {"delta": delta, "gamma": gamma, "vega": vega}


def asian_moments(
    fwds: np.ndarray, vols: np.ndarray, fix_times: np.ndarray
) -> tuple[float, float]:
    """First two moments of an arithmetic average of lognormal forward fixings.

    Fixing i observes forward F_i at time t_i with Black vol sigma_i. Fixings on
    the same contract are perfectly correlated, so
        E[A]   = mean(F_i)
        E[A^2] = (1/n^2) sum_ij F_i F_j exp(sigma_i sigma_j min(t_i, t_j)).
    Already-fixed observations (t_i <= 0) enter with zero variance.
    """
    t = np.maximum(fix_times, 0.0)
    m1 = float(np.mean(fwds))
    cov_t = np.minimum.outer(t, t)
    m2 = float(np.mean(np.outer(fwds, fwds) * np.exp(np.outer(vols, vols) * cov_t)))
    return m1, m2


def asian_levy(
    fwds: np.ndarray,
    vols: np.ndarray,
    fix_times: np.ndarray,
    strike: float,
    df: float,
    is_call: bool,
) -> float:
    """Arithmetic-average Asian option by lognormal moment matching (Levy 1992).

    The average is replaced by a lognormal variable with the same first two
    moments, then priced with Black-76. This is the Turnbull-Wakeman family of
    approximations and is the market convention for monthly-average oil and
    LNG options. Accuracy is checked against Monte Carlo in the tests.
    """
    m1, m2 = asian_moments(fwds, vols, fix_times)
    t_last = float(np.max(fix_times))
    if t_last <= 0.0 or m2 <= m1 * m1:
        return black76(m1, strike, 0.0, 0.0, df, is_call)
    sigma_a = np.sqrt(np.log(m2 / (m1 * m1)) / t_last)
    return black76(m1, strike, sigma_a, t_last, df, is_call)


def asian_mc(
    fwds: np.ndarray,
    vols: np.ndarray,
    fix_times: np.ndarray,
    strike: float,
    df: float,
    is_call: bool,
    n_paths: int = 200_000,
    seed: int = 7,
) -> tuple[float, float]:
    """Monte Carlo benchmark for asian_levy. Returns (price, standard error).

    Simulates one Brownian path per scenario on the sorted fixing grid, which is
    exact for fixings on a single forward contract.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(fix_times)
    t = np.maximum(fix_times[order], 0.0)
    f, v = fwds[order], vols[order]
    dt = np.diff(np.concatenate([[0.0], t]))
    w = np.cumsum(rng.standard_normal((n_paths, len(t))) * np.sqrt(dt), axis=1)
    paths = f * np.exp(v * w - 0.5 * v * v * t)
    avg = paths.mean(axis=1)
    payoff = np.maximum(avg - strike, 0.0) if is_call else np.maximum(strike - avg, 0.0)
    disc = df * payoff
    return float(disc.mean()), float(disc.std(ddof=1) / np.sqrt(n_paths))
