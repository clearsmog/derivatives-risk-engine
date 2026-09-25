"""Closed-form and Monte Carlo pricers for commodity futures options.

Everything prices off forwards (Black-76), which is how listed Brent, TTF and
JKM options are quoted. Rates are a flat continuously compounded discount rate.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
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
    moments, then priced with Black-76. Same family as Turnbull-Wakeman;
    a common quick pricer for monthly-average oil and gas options. It carries a
    small upward bias against Monte Carlo that the tests bound.
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


def bachelier(
    fwd: float, strike: float, normal_vol: float, t: float, annuity: float, is_payer: bool
) -> float:
    """Normal-model swaption: annuity * [(F - K) N(d) + sigma sqrt(t) n(d)] for a payer."""
    if t <= 0.0 or normal_vol <= 0.0:
        return annuity * max((fwd - strike) if is_payer else (strike - fwd), 0.0)
    sd = normal_vol * np.sqrt(t)
    d = (fwd - strike) / sd
    if is_payer:
        return annuity * ((fwd - strike) * norm.cdf(d) + sd * norm.pdf(d))
    return annuity * ((strike - fwd) * norm.cdf(-d) + sd * norm.pdf(d))


def down_and_out_call(
    spot: float, strike: float, barrier: float, vol: float, t: float, r: float, q: float
) -> float:
    """Continuously monitored down-and-out call, strike at or above the barrier.

    Reflection principle (Merton 1973; Reiner-Rubinstein 1991):
        C_do = C(S, K) - (H / S)^(2 lambda - 2) * C(H^2 / S, K),
        lambda = (r - q + sigma^2 / 2) / sigma^2,
    where C is the Black-Scholes call with rate r and dividend yield q.
    """
    if spot <= barrier:
        return 0.0
    if strike < barrier:
        raise ValueError("down_and_out_call implemented for strike >= barrier only")

    def bs_call(s: float) -> float:
        fwd = s * np.exp((r - q) * t)
        return black76(fwd, strike, vol, t, np.exp(-r * t), True)

    lam = (r - q + 0.5 * vol * vol) / (vol * vol)
    return bs_call(spot) - (barrier / spot) ** (2 * lam - 2) * bs_call(barrier * barrier / spot)


def barrier_mc(
    spot: float, strike: float, barrier: float, vol: float, t: float, r: float, q: float,
    n_paths: int = 100_000, n_steps: int = 500, seed: int = 11,
) -> tuple[float, float]:
    """Monte Carlo check for down_and_out_call. To mimic continuous monitoring on
    a discrete grid, the barrier is moved towards spot by the Broadie-Glasserman-
    Kou shift, H * exp(+0.5826 sigma sqrt(dt))."""
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    h_eff = barrier * np.exp(0.5826 * vol * np.sqrt(dt))
    log_s = np.full(n_paths, np.log(spot))
    alive = np.ones(n_paths, dtype=bool)
    drift = (r - q - 0.5 * vol * vol) * dt
    for _ in range(n_steps):
        log_s += drift + vol * np.sqrt(dt) * rng.standard_normal(n_paths)
        alive &= log_s > np.log(h_eff)
    payoff = np.where(alive, np.maximum(np.exp(log_s) - strike, 0.0), 0.0) * np.exp(-r * t)
    return float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n_paths))


def _legs_from_hazard(
    edges: np.ndarray,
    lam: np.ndarray,
    recovery: float,
    maturity: float,
    discount,
    time_shift: float,
    freq: int,
) -> tuple[float, float]:
    """Risky annuity and protection leg for a piecewise-constant hazard curve.
    Premium accrued to the default date is included (half a period on average)."""
    width = np.append(np.diff(edges), np.inf)

    def survival(t: float) -> float:
        return float(np.exp(-np.sum(lam * np.minimum(np.clip(t - edges, 0.0, None), width))))

    pay = np.arange(1, int(round(maturity * freq)) + 1) / freq
    annuity = protection = 0.0
    prev, q_prev = time_shift, 1.0
    for t in pay[pay > time_shift]:
        q = survival(t - time_shift)
        df = discount(t)
        annuity += (t - prev) * df * 0.5 * (q + q_prev)
        protection += (1.0 - recovery) * df * (q_prev - q)
        prev, q_prev = t, q
    return annuity, protection


def bootstrap_hazard(
    spread_tenors: np.ndarray,
    par_spreads: np.ndarray,
    recovery: float,
    discount,
    time_shift: float = 0.0,
    freq: int = 4,
) -> np.ndarray:
    """Piecewise-constant hazard rates such that a CDS maturing at each pillar
    prices at its quoted par spread, solved pillar by pillar."""
    edges = np.concatenate([[0.0], spread_tenors[:-1]])
    lam = par_spreads / (1.0 - recovery)  # credit-triangle starting point
    for i, (tenor, s) in enumerate(zip(spread_tenors, par_spreads)):

        def value(x: float) -> float:
            trial = lam.copy()
            trial[i] = x
            a, p = _legs_from_hazard(edges[: i + 1], trial[: i + 1], recovery, tenor, discount, time_shift, freq)
            return p - s * a

        lam[i] = brentq(value, 1e-8, 5.0)
    return lam


def cds_legs(
    spread_tenors: np.ndarray,
    par_spreads: np.ndarray,
    recovery: float,
    maturity: float,
    discount,
    time_shift: float = 0.0,
    freq: int = 4,
) -> tuple[float, float]:
    """Risky annuity (premium leg per unit spread) and protection leg per unit
    notional, on a hazard curve bootstrapped from the quoted par spreads.

    Simplified against the ISDA standard model: quarterly premiums on a plain
    year fraction, default at period end for the protection leg, no step-in or
    accrued-premium bookkeeping. `discount(t)` returns the discount factor to
    valuation-date time t.
    """
    lam = bootstrap_hazard(spread_tenors, par_spreads, recovery, discount, time_shift, freq)
    edges = np.concatenate([[0.0], spread_tenors[:-1]])
    return _legs_from_hazard(edges, lam, recovery, maturity, discount, time_shift, freq)
