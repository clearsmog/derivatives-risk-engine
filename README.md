# derivatives-risk-engine

A daily risk batch for an LNG trading book: pricing, Greeks, historical VaR and
ES by full revaluation, stress tests, P&L explain, the FRTB P&L attribution
test, and FRTB standardised-approach capital for the commodity risk class. The
results load into an [Atoti](https://www.atoti.io) cube, where VaR can be
broken down by desk, counterparty, commodity or contract month.

The book is synthetic. Its shape is what an LNG importer's desk actually
carries: long-term purchases indexed to Brent and resold at JKM or TTF,
structures sold to downstream buyers (zero-cost collars, a capped swap, a
fixed-price swap), and swaps and options with banks that hedge part of it.
Every name, volume and price is invented.

```
uv sync
uv run pytest                              # the cube test is skipped without Atoti
uv run python scripts/run_demo.py          # risk pack -> outputs/
uv sync --extra cube                       # adds Atoti
uv run --extra cube python -m derivatives_risk_engine.cube   # cube + web UI
```

## What it computes

| Module | Does |
|---|---|
| `pricing.py` | Black-76; arithmetic Asian options by lognormal moment matching (Levy / Turnbull-Wakeman), checked against Monte Carlo |
| `instruments.py` | Every structure is a list of three leg types (linear, Asian, European), so the risk code never needs to know what a "capped swap" is |
| `sensitivities.py` | Bump-and-reprice cash delta, cash gamma, vega and theta per leg, plus the FRTB delta and vega |
| `scenarios.py` | A 500-day synthetic history with fat tails, GARCH volatility clustering, the Samuelson effect, curve-shape and gas seasonal factors, and a 60-day gas squeeze; five named stress scenarios |
| `var.py` | Full-revaluation VaR 99% and ES 97.5%, and delta / delta-gamma / delta-gamma-vega cash-Greeks proxies |
| `pnl_explain.py` | Hypothetical P&L against risk-theoretical P&L split into delta, gamma, vega, theta and unexplained; the MAR32 PLA test per desk against three risk models |
| `frtb.py` | MAR21 sensitivities-based method, commodity risk class: delta, vega and curvature, three correlation scenarios, per desk and for the book, with a switch for how TTF and JKM correlate |
| `cube.py` | Atoti cube at leg grain. Each leg carries its P&L vector, so VaR is taken on summed vectors at any level of aggregation |

## Results on the sample book (USD)

**VaR and diversification.** The supply contracts are a short-oil, long-gas
spread. The bank hedges carry about 57% of the supply contracts' Brent delta
(+617m against -1,074m per 100% move), so a net short-oil position remains.

| Desk | VaR 99% 1d | ES 97.5% 1d |
|---|--:|--:|
| LNG_SUPPLY | 97.6m | 115.4m |
| BANK_HEDGES | 34.6m | 32.3m |
| CLIENT | 10.4m | 11.6m |
| **Book** | **59.9m** | **67.0m** |
| Sum of desks | 142.6m | 159.3m |

**Greeks against full revaluation.** A cash-Greeks proxy
(`delta * R + 0.5 * gamma * R^2 + vega * d_sigma`, R the simple return) lands
within 0.4% of full revaluation. Option gamma barely moves the number; vega
does. The VaR day is a gas sell-off (TTF and JKM front months -12% to -14%)
with implied vol down about 15 points, and the book is long gas vega.

| Method | VaR 99% |
|---|--:|
| Delta only | 59.5m |
| Delta + gamma | 59.4m |
| Delta + gamma + vega | 59.7m |
| Full revaluation | 59.9m |

**Stress.** The book makes money in a 2022-style gas squeeze (+972m) and loses
in an oil spike with flat gas (-182m), because the supply contracts pay an
oil-linked price and earn a gas price.

**P&L attribution.** Three risk models are run against full-revaluation HPL on
250 out-of-sample days.

| Desk | Full curve | One factor per curve, fitted betas | Front month copied onto the curve |
|---|---|---|---|
| BANK_HEDGES | 1.000 / 0.012 green | 0.942 / 0.068 green | 0.938 / 0.112 amber |
| CLIENT | 1.000 / 0.020 green | 0.964 / 0.044 green | 0.962 / 0.092 amber |
| LNG_SUPPLY | 1.000 / 0.008 green | 0.930 / 0.080 green | 0.929 / 0.128 red |

Cells are Spearman / KS. The full-curve model uses the same pricer and risk
factors as HPL, so it only measures Taylor truncation and passes trivially.
The naive proxy ranks days almost as well as the fitted one (Spearman 0.93 to
0.96) but copies the front month's volatility onto back months that move less,
so its P&L distribution is too wide. Only the KS test catches that. A
one-factor model with fitted betas passes on this book; a desk holding
calendar spreads would be the one to expose it.

**FRTB SA capital (commodity).** 408.6m for the book, binding in the low
correlation scenario. Delta dominates (396.3m); curvature is zero for the book,
which is net long convexity.

- **TTF and JKM.** Treated as one commodity at two delivery locations
  (rho_basis 99.9%) by default; see `frtb.py`. Treating them as distinct
  commodities (bucket-6 rho_cty 65%) gives 382.9m, binding in the high
  scenario. The default is the more conservative reading here.
- **Desks.** Run on its own, the client desk carries 2.4m of curvature. All of
  it comes from the dealer side of the TTF collar (short the call); the capped
  swap leaves it long Brent convexity, whose negative S_b offsets part of the
  TTF charge across buckets. Desk-level capital (655.6m + 262.4m + 66.6m) far
  exceeds the book figure because the hedges sit on a different desk from the
  exposure they hedge.
- **Scale.** The risk weights are calibrated to stressed ES over a liquidity
  horizon (20 days for energy), not to a daily move. Scaled by sqrt(20), the
  book's 1-day ES is about 300m, the same order as the SA charge.

## Conventions and choices worth knowing

- **Cash Greeks.** `delta_cash = F dV/dF` (P&L per 100% move, first order) and
  `gamma_cash = F^2 d^2V/dF^2` by central differences on a 1% relative bump;
  cash gamma is exactly zero for linear legs. The risk pack reports both per 1%
  move. The FRTB delta is the one-sided 1% bump of MAR21.23; FRTB vega is
  `vega * sigma` (MAR21.25).
- **Vol** is one ATM number per contract month with no smile, so the
  sticky-strike vs sticky-delta choice of MAR21.27 does not arise.
- **Tenor mapping.** Leg sensitivities are split linearly between the two
  adjacent MAR21.13 vertices: delta by payment date (end of the delivery
  month), vega by option expiry. With 99% tenor correlation the choice of
  delta date barely matters.
- **Curvature correlation scenarios.** MAR21.100-101 square the delta
  correlations; the high/low adjustment is applied to the squared value.
- **VaR quantile.** The loss at rank `floor((n - 1) * 1%)` in ascending order
  (numpy `lower`, Atoti `mode="inc", interpolation="lower"`), so the cube and
  Python agree to the cent. ES averages the worst `floor(n * 2.5%)` days. With
  500 days that is 12 days against the 5th-worst for VaR, so ES 97.5% can come
  out below VaR 99% for a desk with a short, sharp tail (BANK_HEDGES).
- **P&L explain does not age the book** along the 250-day path. Each day's
  market move is applied to today's positions, so no option fixes or expires
  mid-test and theta stays a one-day number.

## Limitations

- Synthetic market data. `synthetic_history` returns the same shape as a real
  history would; swap in a real one to use it. The 60-day gas squeeze sits
  inside the 500-day window; there is no separate stressed-ES calibration.
- Oil indexation uses same-month Brent. Real contracts average with a lag
  (3-0-1, 6-1-1), which adds a tenor mismatch between the oil and gas legs.
- Asian options average one contract month's forward over its own delivery
  month and use that contract's vol throughout; real monthly averages roll
  across front-month futures. Seasoned Asians (some fixings already set) are
  not supported. Moment matching carries a small bias against Monte Carlo that
  the tests bound.
- All prices in USD (TTF in USD/MMBtu), so no FX risk; real TTF trades in EUR.
- Out of scope: GIRR delta from discounting (material on long-dated supply
  contracts), the FRTB default risk charge, the residual risk add-on (which
  the Asians and collars would likely attract), the internal models approach
  beyond the PLA test, and credit and funding valuation adjustments.
- The Atoti community edition is free for non-commercial use, and each release
  stops working 100 days after it comes out. Upgrade the package if the cube
  refuses to start.

## References

- BCBS, *Minimum capital requirements for market risk*: MAR21 (sensitivities-based
  method, version in force from 1 Jan 2023) and MAR32 (backtesting and P&L
  attribution), Basel Framework.
- Levy, E. (1992). Pricing European average rate currency options. *Journal of
  International Money and Finance* 11(5).
- Turnbull, S. and Wakeman, L. (1991). A quick algorithm for pricing European
  average options. *Journal of Financial and Quantitative Analysis* 26(3).
- Black, F. (1976). The pricing of commodity contracts. *Journal of Financial
  Economics* 3(1-2).
